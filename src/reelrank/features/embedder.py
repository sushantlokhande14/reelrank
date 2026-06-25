"""Sentence-transformer wrapper. Imported lazily so the heavy model load only
happens when content embeddings are actually built."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class ContentEmbedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
