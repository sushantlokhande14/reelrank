"""Serving engine retrieval/onboarding logic and explanation templating."""

import numpy as np

from reelrank.config import Config
from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.serving.engine import RecommendEngine
from reelrank.serving.explain import template_reason


def _engine() -> RecommendEngine:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = ProximaIndex(dim=8, space="ip", mode="float").build(
        emb, labels=np.arange(20, dtype=np.int64)
    )
    catalog = [
        {"id": i, "title": f"M{i}", "genres": ["Drama"], "source": "movielens"}
        for i in range(20)
    ]
    return RecommendEngine(Config(), emb, index, catalog)


def test_recommend_from_seeds_excludes_the_seeds():
    engine = _engine()
    out = engine.recommend_from_seeds([0, 1, 2], k=5)
    ids = {o["id"] for o in out}
    assert len(out) == 5
    assert not ids & {0, 1, 2}


def test_search_text_uses_the_query_encoder():
    engine = _engine()
    # an encoder that returns item 7's own vector -> item 7 should rank first
    engine._encoder = lambda text: engine.embeddings[7:8].copy()
    out = engine.search_text("anything", k=3)
    assert out[0]["id"] == 7


def test_template_reason_is_nonempty():
    reason = template_reason("funny sci-fi", {"title": "X", "genres": ["Comedy", "Sci-Fi"]})
    assert isinstance(reason, str) and reason
