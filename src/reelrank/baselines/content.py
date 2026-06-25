"""Content-similarity baseline.

Build a user profile as the mean of the content embeddings of the items they
liked in training, then retrieve the nearest items by cosine similarity (through
Proxima, the same ANN path the real system uses). This is the second baseline the
two-tower has to beat, and unlike popularity it is fully personalized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from reelrank.config import RetrievalCfg
from reelrank.retrieval.proxima_index import ProximaIndex


class ContentRecommender:
    def __init__(self, content_embeddings: np.ndarray, index: ProximaIndex) -> None:
        self.content = np.ascontiguousarray(content_embeddings, dtype=np.float32)
        self.index = index
        self.n_items = content_embeddings.shape[0]

    @classmethod
    def fit(
        cls, content_embeddings: np.ndarray, retrieval_cfg: RetrievalCfg
    ) -> "ContentRecommender":
        index = ProximaIndex.from_config(content_embeddings.shape[1], retrieval_cfg)
        index.build(
            content_embeddings, labels=np.arange(content_embeddings.shape[0], dtype=np.int64)
        )
        return cls(content_embeddings, index)

    def recommend_all(
        self,
        users: Sequence[int],
        k: int,
        exclude: Mapping[int, set[int]],
    ) -> dict[int, np.ndarray]:
        # In this implicit-feedback setup every interaction is a positive, so a
        # user's training history (the `exclude` set) is exactly their liked items.
        profiles = np.zeros((len(users), self.content.shape[1]), dtype=np.float32)
        for row, user in enumerate(users):
            liked = exclude.get(user)
            if liked:
                idx = np.fromiter(liked, dtype=np.int64, count=len(liked))
                profiles[row] = self.content[idx].mean(axis=0)
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        profiles /= norms

        max_excl = max((len(exclude.get(u, ())) for u in users), default=0)
        pool = int(min(self.n_items, k + max_excl))
        labels, _ = self.index.search(profiles, k=pool)
        out: dict[int, np.ndarray] = {}
        for row, user in enumerate(users):
            cand = labels[row]
            cand = cand[cand >= 0]
            liked = exclude.get(user)
            if liked:
                cand = cand[~np.isin(cand, list(liked))]
            out[int(user)] = cand[:k].astype(np.int64)
        return out
