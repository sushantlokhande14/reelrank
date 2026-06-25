"""Two-stage recommender: Proxima retrieves candidates, the ranker re-scores them.

Implements the ``Recommender`` protocol, so it slots into the same eval harness as
the retrieval-only recommender for an apples-to-apples comparison. Scoring is done
in user-chunks to keep GPU memory bounded on large datasets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from reelrank.data.movielens import MovieLensData, item_log_popularity
from reelrank.features.content import user_content_profiles
from reelrank.models.ranker import Ranker
from reelrank.retrieval.proxima_index import ProximaIndex


class RankedRecommender:
    def __init__(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        content_emb: np.ndarray,
        ranker: Ranker,
        index: ProximaIndex,
        data: MovieLensData,
        n_candidates: int,
        device: str = "cpu",
        chunk_size: int = 4096,
    ) -> None:
        self.device = device
        self.user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        self.user_t = torch.from_numpy(self.user_emb).to(device)
        self.item_t = torch.from_numpy(np.ascontiguousarray(item_emb, dtype=np.float32)).to(device)
        self.pop_t = torch.from_numpy(item_log_popularity(data)).to(device)
        self.item_content_t = torch.from_numpy(
            np.ascontiguousarray(content_emb, dtype=np.float32)
        ).to(device)
        self.profile_t = torch.from_numpy(user_content_profiles(data, content_emb)).to(device)
        self.ranker = ranker.to(device).eval()
        self.index = index
        self.n_items = data.n_items
        self.emb_dim = item_emb.shape[1]
        self.n_candidates = n_candidates
        self.chunk_size = chunk_size

    @torch.no_grad()
    def recommend_all(
        self,
        users: Sequence[int],
        k: int,
        exclude: Mapping[int, set[int]],
    ) -> dict[int, np.ndarray]:
        users = list(users)
        max_excl = max((len(exclude.get(u, ())) for u in users), default=0)
        pool = int(min(self.n_items, self.n_candidates + max_excl))

        out: dict[int, np.ndarray] = {}
        for start in range(0, len(users), self.chunk_size):
            chunk = users[start : start + self.chunk_size]
            user_ids = np.fromiter(chunk, dtype=np.int64, count=len(chunk))

            labels, _ = self.index.search(self.user_emb[user_ids], k=pool)  # (c, pool)
            cand = torch.from_numpy(np.clip(labels, 0, None).astype(np.int64)).to(self.device)

            user_idx_t = torch.from_numpy(user_ids).to(self.device)
            user_vecs = self.user_t[user_idx_t].unsqueeze(1).expand(len(chunk), pool, self.emb_dim)
            content_match = (self.profile_t[user_idx_t].unsqueeze(1) * self.item_content_t[cand]).sum(-1)
            extra = torch.stack([content_match, self.pop_t[cand]], dim=-1)
            scores = self.ranker(user_vecs, self.item_t[cand], extra)
            scores = scores.masked_fill(torch.from_numpy(labels < 0).to(self.device), float("-inf"))

            order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            ranked = np.take_along_axis(labels, order, axis=1)  # reorder ids by ranker score

            for row, user in enumerate(chunk):
                row_ids = ranked[row]
                row_ids = row_ids[row_ids >= 0]
                seen = exclude.get(user)
                if seen:
                    row_ids = row_ids[~np.isin(row_ids, list(seen))]
                out[int(user)] = row_ids[:k].astype(np.int64)
        return out
