"""Serving engine: natural-language search, seed-based onboarding, explanations.

This is the layer the FastAPI backend wraps. It loads the serving artifacts the
TMDB refresh produces (the combined MovieLens + live index, embeddings, catalog)
and answers three kinds of request: a free-text vibe query, a cold-start onboard
from a few liked titles, and a known-user recommendation.
"""

from reelrank.serving.engine import RecommendEngine, build_query_encoder
from reelrank.serving.explain import reason_for, template_reason

__all__ = ["RecommendEngine", "build_query_encoder", "reason_for", "template_reason"]
