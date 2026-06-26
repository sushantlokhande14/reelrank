"""reelrank: a two-stage hybrid movie recommender.

Stage 1 (retrieval): a two-tower neural model embeds users and items into one
space; candidates are pulled with approximate nearest-neighbor search over the
Proxima HNSW index.

Stage 2 (ranking): a neural ranker re-scores the candidates.

The package is dataset- and config-driven; see config/default.yaml.
"""

import os as _os

# torch, sentence-transformers, and the Proxima C++ extension can each bundle an
# OpenMP/MKL runtime. On Windows, loading more than one (or contending over their
# thread pools) aborts the process with an access violation (0xC0000005). Permit
# the duplicate load and pin the math libraries to a single thread before any of
# them import. Single-threaded inference is also the right default for the small
# CPU serving host. Set these as real env vars in production to be safe.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

from reelrank.config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "__version__"]
