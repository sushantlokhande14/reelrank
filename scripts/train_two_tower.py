"""Train the two-tower model, build the Proxima index, and evaluate retrieval.

    python scripts/train_two_tower.py --config config/default.yaml

Writes artifacts (embeddings, Proxima index, metrics) under
artifacts/<dataset>/ and prints a comparison against the popularity baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from reelrank.baselines.content import ContentRecommender
from reelrank.baselines.popularity import PopularityRecommender
from reelrank.config import load_config
from reelrank.data.movielens import SPLIT_TRAIN, build_dataset
from reelrank.eval.harness import evaluate
from reelrank.features.content import build_content_embeddings
from reelrank.models.factory import model_label
from reelrank.report import metrics_table
from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.retrieval.recommender import TwoTowerRecommender
from reelrank.training.two_tower_trainer import export_embeddings, train_two_tower
from reelrank.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    data = build_dataset(cfg)
    print(
        f"dataset {cfg.data.dataset}: users {data.n_users:,}, items {data.n_items:,}, "
        f"train interactions {len(data.split_frame(SPLIT_TRAIN)):,}"
    )

    print("building content embeddings (first run downloads the sentence model)...")
    content_emb = build_content_embeddings(cfg, data)

    t0 = time.perf_counter()
    model = train_two_tower(cfg, data, content_features=content_emb)
    print(f"trained {model_label(cfg, content_emb)} in {time.perf_counter() - t0:.1f}s")

    user_emb, item_emb = export_embeddings(model, data)
    dim = item_emb.shape[1]

    artifacts = Path(cfg.paths.artifacts) / cfg.data.dataset
    artifacts.mkdir(parents=True, exist_ok=True)
    np.save(artifacts / "user_embeddings.npy", user_emb)
    np.save(artifacts / "item_embeddings.npy", item_emb)
    torch.save(model.state_dict(), artifacts / "two_tower.pt")

    t0 = time.perf_counter()
    index = ProximaIndex.from_config(dim, cfg.retrieval)
    index.build(item_emb, labels=np.arange(data.n_items, dtype=np.int64))
    index.save(artifacts / "items.idx")
    print(f"built Proxima index ({len(index):,} items, {cfg.retrieval.mode}) in "
          f"{time.perf_counter() - t0:.2f}s")

    two_tower = TwoTowerRecommender(user_emb, index, data.n_items)
    popularity = PopularityRecommender().fit(data)
    content = ContentRecommender.fit(content_emb, cfg.retrieval)
    results = {
        "popularity": evaluate(popularity, data, cfg.eval.k_values),
        "content similarity": evaluate(content, data, cfg.eval.k_values),
        model_label(cfg, content_emb): evaluate(two_tower, data, cfg.eval.k_values),
    }

    table = metrics_table(results, cfg.eval.k_values)
    print()
    print(table)
    (artifacts / "metrics.json").write_text(json.dumps(results, indent=2))
    (artifacts / "metrics.md").write_text(table + "\n")
    print(f"\nartifacts written to {artifacts}")


if __name__ == "__main__":
    main()
