"""Baselines to beat: popularity and content-based similarity."""

from reelrank.baselines.content import ContentRecommender
from reelrank.baselines.popularity import PopularityRecommender

__all__ = ["PopularityRecommender", "ContentRecommender"]
