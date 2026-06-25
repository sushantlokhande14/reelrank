"""Popularity baseline: recommend the globally most-interacted items.

It ignores personalization entirely, which is exactly why it is a useful floor:
a real recommender has to beat "just show everyone the popular stuff."
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from reelrank.data.movielens import SPLIT_TRAIN, MovieLensData


class PopularityRecommender:
    def __init__(self) -> None:
        # item ids sorted by descending train interaction count
        self.ranked_items: np.ndarray = np.empty(0, dtype=np.int64)

    def fit(self, data: MovieLensData) -> "PopularityRecommender":
        counts = data.split_frame(SPLIT_TRAIN)["item_idx"].value_counts()
        self.ranked_items = counts.index.to_numpy(dtype=np.int64)
        return self

    def recommend_all(
        self,
        users: Sequence[int],
        k: int,
        exclude: Mapping[int, set[int]],
    ) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for user in users:
            seen = exclude.get(user, set())
            if not seen:
                out[user] = self.ranked_items[:k]
                continue
            # Among the first k + |seen| popular items at most |seen| are excluded,
            # so this slice always leaves at least k candidates.
            window = self.ranked_items[: k + len(seen)]
            kept = window[~np.isin(window, list(seen))]
            out[user] = kept[:k]
        return out
