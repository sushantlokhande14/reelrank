"""The recommendation engine that backs the API.

Natural-language search works by treating the query like a brand-new item: the
text is embedded with the same sentence-transformer used for movie content, then
projected into the collaborative item space through the hybrid tower's cold-start
path, and used as the retrieval query against the serving index. So "a slow-burn
sci-fi like Arrival but funnier" is answered in the same space as everything else.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch  # noqa: F401 - import torch before Proxima: on Windows, loading the

# Proxima C++ extension before torch's native runtime triggers an OpenMP/MKL
# load-order access violation. Keeping this import above proxima_index fixes it.
from reelrank.config import Config
from reelrank.retrieval.proxima_index import ProximaIndex

QueryEncoder = Callable[[str], np.ndarray]


def build_query_encoder(cfg: Config) -> QueryEncoder:
    """A closure that turns text into one item-space vector. Heavy to build (loads
    the sentence model and the two-tower), so the engine creates it lazily."""
    from reelrank.data.movielens import build_dataset
    from reelrank.features.content import build_content_embeddings
    from reelrank.features.embedder import ContentEmbedder
    from reelrank.training.pipeline import load_two_tower_model

    data = build_dataset(cfg)
    content_emb = build_content_embeddings(cfg, data)
    model = load_two_tower_model(cfg, data, content_emb)
    # Serve on CPU: the deploy host has no GPU, and it sidesteps concurrent-CUDA
    # crashes when several requests arrive at once.
    embedder = ContentEmbedder(cfg.content.model, device="cpu")

    def encode(text: str) -> np.ndarray:
        content = embedder.encode([text], batch_size=cfg.content.batch_size)
        with torch.no_grad():
            vec = model.item_tower.embed_cold(torch.from_numpy(content)).cpu().numpy()
        return vec.astype(np.float32)

    return encode


class RecommendEngine:
    def __init__(
        self,
        cfg: Config,
        embeddings: np.ndarray,
        index: ProximaIndex,
        catalog: list[dict],
        query_encoder: QueryEncoder | None = None,
    ) -> None:
        self.cfg = cfg
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = index
        self.catalog = catalog
        self.catalog_by_id = {int(c["id"]): c for c in catalog}
        self._encoder = query_encoder
        # Serialize inference: torch forward passes and the lazy encoder build are
        # not safe to run from several request threads at once.
        self._lock = threading.Lock()

    @classmethod
    def from_artifacts(cls, cfg: Config) -> "RecommendEngine":
        serve = Path(cfg.paths.artifacts) / cfg.data.dataset / "serving"
        if not (serve / "index.idx").exists():
            raise FileNotFoundError(
                f"{serve} has no serving index; run scripts/refresh_tmdb.py first"
            )
        # Build the query encoder first: this loads torch and sentence-transformers
        # before Proxima's native extension, which avoids the Windows OpenMP/MKL
        # load-order crash. It also warms the model so the first query is fast.
        encoder = build_query_encoder(cfg)
        embeddings = np.load(serve / "embeddings.npy")
        index = ProximaIndex.load(
            serve / "index.idx", cfg.retrieval.ef_search, cfg.retrieval.mode
        )
        catalog = json.loads((serve / "catalog.json").read_text())
        return cls(cfg, embeddings, index, catalog, query_encoder=encoder)

    # -- query types ------------------------------------------------------- #
    def search_text(self, query: str, k: int = 20) -> list[dict]:
        with self._lock:
            if self._encoder is None:
                self._encoder = build_query_encoder(self.cfg)
            vector = self._encoder(query)
            return self._retrieve(vector, k, exclude=set())

    def recommend_from_seeds(self, seed_ids: Sequence[int], k: int = 20) -> list[dict]:
        """Cold-start onboarding: average the picked titles' embeddings into a
        taste vector and retrieve neighbours, excluding the picks themselves."""
        seeds = [int(s) for s in seed_ids if 0 <= int(s) < len(self.embeddings)]
        if not seeds:
            return []
        vector = self.embeddings[seeds].mean(axis=0, keepdims=True)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        with self._lock:
            return self._retrieve(vector.astype(np.float32), k, exclude=set(seeds))

    def item(self, item_id: int) -> dict | None:
        return self.catalog_by_id.get(int(item_id))

    # -- internals --------------------------------------------------------- #
    def _retrieve(self, vector: np.ndarray, k: int, exclude: set[int]) -> list[dict]:
        labels, _ = self.index.search(vector, k=k + len(exclude))
        results: list[dict] = []
        for label in labels[0]:
            label = int(label)
            if label < 0 or label in exclude:
                continue
            item = self.catalog_by_id.get(label)
            if item is not None:
                results.append(item)
            if len(results) >= k:
                break
        return results
