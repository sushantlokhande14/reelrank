"""Evaluation harness: builds leakage-free ground truth and scores a recommender.

A recommender only needs to implement ``recommend_all`` (batch interface) so that
neural/ANN recommenders can be evaluated efficiently. The harness excludes each
user's training history from both the recommendations and the ground truth, which
is the standard implicit-feedback top-K protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from reelrank.data.movielens import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    MovieLensData,
    user_item_lists,
)
from reelrank.eval.metrics import evaluate_rankings


class Recommender(Protocol):
    def recommend_all(
        self,
        users: Sequence[int],
        k: int,
        exclude: Mapping[int, set[int]],
    ) -> dict[int, np.ndarray]:
        """Return top-k item ids for each user, skipping their excluded items."""
        ...


def build_ground_truth(
    data: MovieLensData,
    eval_split: str = SPLIT_TEST,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Construct (ground_truth, train_history) for users seen in training.

    ground_truth[u] = positives in `eval_split` that the user had NOT already
    interacted with in training. Users without training history (cold start) are
    excluded from this collaborative evaluation.
    """
    train_hist = user_item_lists(data, SPLIT_TRAIN)
    eval_hist = user_item_lists(data, eval_split)

    ground_truth: dict[int, set[int]] = {}
    train_sets: dict[int, set[int]] = {}
    for user, eval_items in eval_hist.items():
        if user not in train_hist:
            continue
        seen = set(train_hist[user].tolist())
        held_out = {int(i) for i in eval_items.tolist() if int(i) not in seen}
        if held_out:
            ground_truth[user] = held_out
            train_sets[user] = seen
    return ground_truth, train_sets


def evaluate(
    recommender: Recommender,
    data: MovieLensData,
    k_values: Sequence[int],
    eval_split: str = SPLIT_TEST,
) -> dict[str, float]:
    """Score a recommender on the held-out split."""
    ground_truth, train_sets = build_ground_truth(data, eval_split)
    users = list(ground_truth.keys())
    max_k = max(k_values)
    recommendations = recommender.recommend_all(users, max_k, train_sets)
    return evaluate_rankings(recommendations, ground_truth, k_values)
