"""Load the two-tower retrieval artifacts, training them if they are missing.

The ranker reuses the two-tower's user/item embeddings and Proxima index, so this
avoids retraining the retrieval stage every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch import nn

from reelrank.config import Config
from reelrank.data.movielens import MovieLensData, build_dataset
from reelrank.features.content import build_content_embeddings
from reelrank.models.factory import build_model
from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.training.two_tower_trainer import export_embeddings, train_two_tower
from reelrank.utils import set_seed


@dataclass
class RetrievalArtifacts:
    data: MovieLensData
    content_emb: np.ndarray
    user_emb: np.ndarray
    item_emb: np.ndarray
    index: ProximaIndex


def ensure_two_tower_artifacts(cfg: Config, verbose: bool = True) -> RetrievalArtifacts:
    data = build_dataset(cfg)
    content_emb = build_content_embeddings(cfg, data)

    artifacts = Path(cfg.paths.artifacts) / cfg.data.dataset
    user_path = artifacts / "user_embeddings.npy"
    item_path = artifacts / "item_embeddings.npy"
    index_path = artifacts / "items.idx"

    if user_path.exists() and item_path.exists() and index_path.exists():
        if verbose:
            print(f"loading two-tower artifacts from {artifacts}")
        user_emb = np.load(user_path)
        item_emb = np.load(item_path)
        index = ProximaIndex.load(index_path, cfg.retrieval.ef_search, cfg.retrieval.mode)
    else:
        if verbose:
            print("no two-tower artifacts found; training the retrieval stage first")
        set_seed(cfg.seed)
        model = train_two_tower(cfg, data, content_features=content_emb, verbose=verbose)
        user_emb, item_emb = export_embeddings(model, data)
        artifacts.mkdir(parents=True, exist_ok=True)
        np.save(user_path, user_emb)
        np.save(item_path, item_emb)
        index = ProximaIndex.from_config(item_emb.shape[1], cfg.retrieval).build(
            item_emb, labels=np.arange(data.n_items, dtype=np.int64)
        )
        index.save(index_path)

    return RetrievalArtifacts(data, content_emb, user_emb, item_emb, index)


def load_two_tower_model(
    cfg: Config, data: MovieLensData, content_emb: np.ndarray
) -> nn.Module:
    """Reload the trained two-tower from its saved weights (for embed_cold etc.)."""
    state_path = Path(cfg.paths.artifacts) / cfg.data.dataset / "two_tower.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} not found; run scripts/train_two_tower.py first"
        )
    model = build_model(cfg, data, content_emb)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
