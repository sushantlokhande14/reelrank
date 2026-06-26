"""Thin wrapper around the Proxima HNSW index.

Proxima (github.com/sushantlokhande14/proxima) is the ANN engine for stage-1
retrieval. We store L2-normalized embeddings in an inner-product space, so search
ranks by cosine similarity, and use the SQ8 mode (int8 traversal + float32
re-rank) to keep the serving memory footprint small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reelrank.config import RetrievalCfg

# proxima (a native C++ extension) is imported lazily inside the methods below.
# Importing it at module load can pull in an OpenMP/MKL runtime before torch and
# sentence-transformers have loaded theirs, which on Windows aborts the process
# with an access violation. Deferring the import lets callers load the ML
# libraries first.


class ProximaIndex:
    def __init__(
        self,
        dim: int,
        space: str = "ip",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 128,
        mode: str = "sq8",
    ) -> None:
        import proxima

        self.index = proxima.Index(dim=dim, space=space, M=M, ef_construction=ef_construction)
        self.index.ef = ef_search
        self.mode = mode

    @classmethod
    def from_config(cls, dim: int, cfg: RetrievalCfg) -> "ProximaIndex":
        return cls(dim, cfg.proxima_space, cfg.M, cfg.ef_construction, cfg.ef_search, cfg.mode)

    def build(self, vectors: np.ndarray, labels: np.ndarray | None = None) -> "ProximaIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        lab = None if labels is None else np.ascontiguousarray(labels, dtype=np.int64)
        self.index.add(vectors, lab)
        self.index.reorder()  # BFS relabel for cache locality; results unchanged
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Batch k-NN. Returns (labels (m, k) int64, distances (m, k) float32);
        labels are ordered best-first, missing slots padded with -1."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        return self.index.search(queries, k=k, mode=self.mode, num_threads=0)

    def save(self, path: str | Path) -> None:
        self.index.save(str(path))

    @classmethod
    def load(cls, path: str | Path, ef_search: int = 128, mode: str = "sq8") -> "ProximaIndex":
        import proxima

        obj = cls.__new__(cls)
        obj.index = proxima.load(str(path))
        obj.index.ef = ef_search
        obj.mode = mode
        return obj

    def __len__(self) -> int:
        return len(self.index)
