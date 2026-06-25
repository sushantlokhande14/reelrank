"""Ranker forward shapes and the two-stage recommender's contract."""

import numpy as np
import pandas as pd
import torch

from reelrank.data.movielens import MovieLensData
from reelrank.models.ranker import Ranker
from reelrank.ranking.reranker import RankedRecommender
from reelrank.retrieval.proxima_index import ProximaIndex


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _toy_data(n_items: int) -> MovieLensData:
    ratings = pd.DataFrame(
        {
            "user_idx": [0] * n_items,
            "item_idx": list(range(n_items)),
            "rating": [5.0] * n_items,
            "timestamp": list(range(n_items)),
            "split": ["train"] * n_items,
        }
    )
    movies = pd.DataFrame(
        {
            "item_idx": list(range(n_items)),
            "movieId": list(range(n_items)),
            "title": ["x"] * n_items,
            "genres": ["g"] * n_items,
        }
    )
    return MovieLensData(ratings, movies, n_users=3, n_items=n_items, user_id_map={}, item_id_map={})


def test_ranker_forward_shapes():
    ranker = Ranker(emb_dim=8, n_extra=2, hidden=[16], dropout=0.0)
    u = torch.randn(5, 8)
    v = torch.randn(5, 8)
    extra = torch.randn(5, 2)
    assert ranker(u, v, extra).shape == (5,)
    # grouped (batch, n_candidates, dim)
    assert ranker(torch.randn(3, 4, 8), torch.randn(3, 4, 8), torch.randn(3, 4, 2)).shape == (3, 4)


def test_ranked_recommender_contract():
    rng = np.random.default_rng(0)
    item_emb = _unit(rng, 40, 8)
    user_emb = _unit(rng, 3, 8)
    content_emb = _unit(rng, 40, 6)
    data = _toy_data(40)
    index = ProximaIndex(dim=8, space="ip", mode="float").build(
        item_emb, labels=np.arange(40, dtype=np.int64)
    )
    ranker = Ranker(emb_dim=8, n_extra=2, hidden=[16], dropout=0.0)
    rec = RankedRecommender(
        user_emb, item_emb, content_emb, ranker, index, data, n_candidates=20, device="cpu"
    )

    out = rec.recommend_all([0, 1, 2], k=5, exclude={})
    for user in (0, 1, 2):
        assert len(out[user]) == 5
        assert len(set(out[user].tolist())) == 5

    banned = set(out[0][:3].tolist())
    out2 = rec.recommend_all([0], k=5, exclude={0: banned})
    assert not banned & set(out2[0].tolist())
