"""reelrank: a two-stage hybrid movie recommender.

Stage 1 (retrieval): a two-tower neural model embeds users and items into one
space; candidates are pulled with approximate nearest-neighbor search over the
Proxima HNSW index.

Stage 2 (ranking): a neural ranker re-scores the candidates.

The package is dataset- and config-driven; see config/default.yaml.
"""

import os as _os

# torch, sentence-transformers, and the Proxima C++ extension can each bundle an
# OpenMP runtime. On Windows, loading more than one aborts the process with an
# access violation (0xC0000005). Permit the duplicate load before any of them
# import. Set as a real env var in production to be safe.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from reelrank.config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "__version__"]
