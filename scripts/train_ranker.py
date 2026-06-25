"""Train the stage-2 ranker and compare the full two-stage pipeline against
retrieval-only and the popularity baseline.

    python scripts/train_ranker.py --config config/default.yaml

Reuses the two-tower artifacts under artifacts/<dataset>/ (training them first if
they are missing).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from reelrank.baselines.popularity import PopularityRecommender
from reelrank.config import load_config
from reelrank.eval.harness import evaluate
from reelrank.ranking.reranker import RankedRecommender
from reelrank.report import metrics_table
from reelrank.retrieval.recommender import TwoTowerRecommender
from reelrank.training.pipeline import ensure_two_tower_artifacts
from reelrank.training.ranker_trainer import train_ranker
from reelrank.utils import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    art = ensure_two_tower_artifacts(cfg)
    data, user_emb, item_emb, content_emb, index = (
        art.data, art.user_emb, art.item_emb, art.content_emb, art.index
    )
    print(f"dataset {cfg.data.dataset}: users {data.n_users:,}, items {data.n_items:,}")

    t0 = time.perf_counter()
    ranker = train_ranker(cfg, data, user_emb, item_emb, content_emb)
    print(f"trained ranker in {time.perf_counter() - t0:.1f}s")

    device = resolve_device(cfg.train.device)
    retrieval = TwoTowerRecommender(user_emb, index, data.n_items)
    ranked = RankedRecommender(
        user_emb, item_emb, content_emb, ranker, index, data, cfg.retrieval.candidates, device=device
    )
    popularity = PopularityRecommender().fit(data)

    results = {
        "popularity": evaluate(popularity, data, cfg.eval.k_values),
        "retrieval only (two-tower)": evaluate(retrieval, data, cfg.eval.k_values),
        "retrieval + ranker": evaluate(ranked, data, cfg.eval.k_values),
    }
    table = metrics_table(results, cfg.eval.k_values)
    print()
    print(table)

    outdir = Path(cfg.paths.artifacts) / cfg.data.dataset
    torch.save(ranker.state_dict(), outdir / "ranker.pt")
    (outdir / "ranker_metrics.json").write_text(json.dumps(results, indent=2))
    (outdir / "ranker_metrics.md").write_text(table + "\n")
    print(f"\nartifacts written to {outdir}")


if __name__ == "__main__":
    main()
