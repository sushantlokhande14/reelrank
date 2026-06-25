"""Stage-1 recommender: embed the user, pull candidates from Proxima.

Implements the ``Recommender`` protocol so it drops straight into the eval
harness. Also exposes ``retrieve`` for serving arbitrary query vectors (used by
cold-start onboarding and natural-language search).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from reelrank.retrieval.proxima_index import ProximaIndex


class TwoTowerRecommender:
    def __init__(
        self,
        user_embeddings: np.ndarray,
        index: ProximaIndex,
        n_items: int,
    ) -> None:
        self.user_embeddings = np.ascontiguousarray(user_embeddings, dtype=np.float32)
        self.index = index
        self.n_items = n_items

    def retrieve(
        self, query: np.ndarray, k: int, exclude: set[int] | None = None
    ) -> np.ndarray:
        """Top-k item ids for one or more query vectors (single query -> 1-D)."""
        pool = int(min(self.n_items, k + (len(exclude) if exclude else 0)))
        labels, _ = self.index.search(query, k=pool)
        rows = [self._filter(labels[i], exclude, k) for i in range(labels.shape[0])]
        return rows[0] if len(rows) == 1 else np.array(rows, dtype=object)

    def recommend_all(
        self,
        users: Sequence[int],
        k: int,
        exclude: Mapping[int, set[int]],
    ) -> dict[int, np.ndarray]:
        user_ids = np.fromiter(users, dtype=np.int64, count=len(users))
        queries = self.user_embeddings[user_ids]
        max_excl = max((len(exclude.get(u, ())) for u in users), default=0)
        pool = int(min(self.n_items, k + max_excl))
        labels, _ = self.index.search(queries, k=pool)
        return {
            int(u): self._filter(labels[row], exclude.get(u), k)
            for row, u in enumerate(users)
        }

    @staticmethod
    def _filter(candidates: np.ndarray, seen: set[int] | None, k: int) -> np.ndarray:
        candidates = candidates[candidates >= 0]  # drop padding
        if seen:
            candidates = candidates[~np.isin(candidates, list(seen))]
        return candidates[:k].astype(np.int64)
