"""Proxima index round-trip and the two-tower recommender's filtering."""

import numpy as np

from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.retrieval.recommender import TwoTowerRecommender


def _unit(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_proxima_index_retrieves_self_as_nearest():
    rng = np.random.default_rng(0)
    vecs = _unit(rng, 200, 16)
    index = ProximaIndex(dim=16, space="ip", mode="float").build(
        vecs, labels=np.arange(200, dtype=np.int64)
    )
    labels, _ = index.search(vecs, k=1)
    # each vector's nearest neighbour by cosine is itself
    assert (labels[:, 0] == np.arange(200)).mean() > 0.98


def test_recommender_respects_k_and_exclusions():
    rng = np.random.default_rng(1)
    item_emb = _unit(rng, 40, 8)
    user_emb = _unit(rng, 3, 8)
    index = ProximaIndex(dim=8, space="ip", mode="float").build(
        item_emb, labels=np.arange(40, dtype=np.int64)
    )
    rec = TwoTowerRecommender(user_emb, index, n_items=40)

    out = rec.recommend_all([0, 1, 2], k=5, exclude={})
    for user in (0, 1, 2):
        assert len(out[user]) == 5
        assert len(set(out[user].tolist())) == 5  # no duplicates

    banned = set(out[0][:3].tolist())
    out2 = rec.recommend_all([0], k=5, exclude={0: banned})
    assert not banned & set(out2[0].tolist())  # excluded items never returned
