"""Offline evaluation: ranking metrics and a leakage-free eval harness."""

from reelrank.eval.harness import Recommender, build_ground_truth, evaluate
from reelrank.eval.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "Recommender",
    "build_ground_truth",
    "evaluate",
    "average_precision_at_k",
    "evaluate_rankings",
    "ndcg_at_k",
    "recall_at_k",
]
