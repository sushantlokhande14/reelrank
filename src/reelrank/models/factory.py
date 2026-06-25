"""Pick the collaborative or hybrid two-tower based on config + available content."""

from __future__ import annotations

import numpy as np
from torch import nn

from reelrank.config import Config
from reelrank.data.movielens import MovieLensData
from reelrank.models.hybrid import HybridTwoTower
from reelrank.models.two_tower import TwoTower


def build_model(
    cfg: Config, data: MovieLensData, content_features: np.ndarray | None = None
) -> nn.Module:
    use_content = content_features is not None and cfg.content.content_weight > 0.0
    if use_content:
        return HybridTwoTower(data.n_users, data.n_items, content_features, cfg.model, cfg.content)
    return TwoTower(data.n_users, data.n_items, cfg.model)


def model_label(cfg: Config, content_features: np.ndarray | None) -> str:
    if content_features is not None and cfg.content.content_weight > 0.0:
        return "two-tower + content (hybrid)"
    return "two-tower (collaborative)"
