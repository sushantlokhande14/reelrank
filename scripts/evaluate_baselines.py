"""Build the dataset and evaluate baseline recommenders.

This is the MVP's "first real numbers" entry point. Run:

    python scripts/evaluate_baselines.py --config config/default.yaml
"""

from __future__ import annotations

import argparse

from reelrank.baselines.content import ContentRecommender
from reelrank.baselines.popularity import PopularityRecommender
from reelrank.config import load_config
from reelrank.data.movielens import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, build_dataset
from reelrank.eval.harness import build_ground_truth, evaluate
from reelrank.features.content import build_content_embeddings
from reelrank.report import metrics_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = build_dataset(cfg)

    print(f"dataset:        {cfg.data.dataset}")
    print(f"users / items:  {data.n_users:,} / {data.n_items:,}")
    print(f"interactions:   {len(data.ratings):,}")
    for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        print(f"  {split:<5}       {len(data.split_frame(split)):,}")

    ground_truth, _ = build_ground_truth(data, SPLIT_TEST)
    print(f"eval users:     {len(ground_truth):,} (seen in train, with held-out test positives)")
    print()

    pop = PopularityRecommender().fit(data)
    results = {"popularity": evaluate(pop, data, cfg.eval.k_values)}

    print("building content embeddings (first run downloads the sentence model)...")
    content_emb = build_content_embeddings(cfg, data)
    content = ContentRecommender.fit(content_emb, cfg.retrieval)
    results["content similarity"] = evaluate(content, data, cfg.eval.k_values)

    print(metrics_table(results, cfg.eval.k_values))


if __name__ == "__main__":
    main()
