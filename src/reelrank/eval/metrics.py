"""Ranking metrics for top-K recommendation.

Each metric takes a ranked list of item ids and a set of relevant (ground-truth)
item ids. Relevance is binary. ``evaluate_rankings`` averages over users.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """Fraction of the relevant items that appear in the top-K."""
    if not truth:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in truth)
    return hits / len(truth)


def ndcg_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if not truth:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked[:k]):
        if item in truth:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: Sequence[int], truth: set[int], k: int) -> float:
    """Average precision at K (the per-user term of MAP)."""
    if not truth:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked[:k]):
        if item in truth:
            hits += 1
            score += hits / (rank + 1)
    return score / min(len(truth), k)


def evaluate_rankings(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Average Recall/NDCG/MAP @K over all users with non-empty ground truth.

    Returns a flat dict like {"recall@10": ..., "ndcg@10": ..., "map@10": ...,
    "n_users": ...}.
    """
    sums: dict[str, float] = {}
    for k in k_values:
        sums[f"recall@{k}"] = 0.0
        sums[f"ndcg@{k}"] = 0.0
        sums[f"map@{k}"] = 0.0

    n_users = 0
    for user, truth in ground_truth.items():
        if not truth:
            continue
        ranked = recommendations.get(user, [])
        for k in k_values:
            sums[f"recall@{k}"] += recall_at_k(ranked, truth, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(ranked, truth, k)
            sums[f"map@{k}"] += average_precision_at_k(ranked, truth, k)
        n_users += 1

    out = {metric: (total / n_users if n_users else 0.0) for metric, total in sums.items()}
    out["n_users"] = float(n_users)
    return out
