"""Stage-2 re-ranking: retrieve from Proxima, then re-score with the ranker."""

from reelrank.ranking.reranker import RankedRecommender

__all__ = ["RankedRecommender"]
