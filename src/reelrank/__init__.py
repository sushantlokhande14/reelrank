"""reelrank: a two-stage hybrid movie recommender.

Stage 1 (retrieval): a two-tower neural model embeds users and items into one
space; candidates are pulled with approximate nearest-neighbor search over the
Proxima HNSW index.

Stage 2 (ranking): a neural ranker re-scores the candidates.

The package is dataset- and config-driven; see config/default.yaml.
"""

from reelrank.config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "__version__"]
