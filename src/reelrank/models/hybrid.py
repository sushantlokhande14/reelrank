"""Content-aware (hybrid) two-tower.

The item tower fuses a frozen content embedding with a learned id embedding:

    item_vec = MLP( id_weight * id_embed[item] + content_weight * W·content[item] )

Items seen in training get a real id embedding; items with no collaborative
history (cold start: a new or unrated movie, or a fresh TMDB title) use a reserved
"unknown" id row, so the content feature alone carries them into the same space.
The user tower stays purely collaborative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reelrank.config import ContentCfg, ModelCfg
from reelrank.models.two_tower import Tower


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class HybridItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        content_features: np.ndarray,
        hidden: Sequence[int],
        out_dim: int,
        dropout: float,
        id_weight: float,
        content_weight: float,
    ) -> None:
        super().__init__()
        content_features = np.ascontiguousarray(content_features, dtype=np.float32)
        self.register_buffer("content", torch.from_numpy(content_features))
        self.unknown_id = n_items  # reserved id row for cold items

        self.id_embed = nn.Embedding(n_items + 1, hidden[0])
        nn.init.normal_(self.id_embed.weight, std=0.05)
        self.content_proj = nn.Linear(content_features.shape[1], hidden[0])
        self.id_weight = float(id_weight)
        self.content_weight = float(content_weight)
        self.mlp = _mlp(list(hidden) + [out_dim], dropout)

    def _combine(self, id_emb: torch.Tensor, content_vecs: torch.Tensor) -> torch.Tensor:
        fused = self.id_weight * id_emb + self.content_weight * self.content_proj(content_vecs)
        return F.normalize(self.mlp(fused), dim=-1)

    def forward(self, item_ids: torch.Tensor, content_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if content_vecs is None:
            content_vecs = self.content[item_ids]
        return self._combine(self.id_embed(item_ids), content_vecs)

    @torch.no_grad()
    def embed_cold(self, content_vecs: torch.Tensor) -> torch.Tensor:
        """Embed items with no collaborative history from content alone."""
        ids = torch.full(
            (content_vecs.shape[0],), self.unknown_id, dtype=torch.long, device=content_vecs.device
        )
        return self._combine(self.id_embed(ids), content_vecs)


class HybridTwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        content_features: np.ndarray,
        cfg: ModelCfg,
        content_cfg: ContentCfg,
    ) -> None:
        super().__init__()
        self.user_tower = Tower(n_users, cfg.user_hidden, cfg.embedding_dim, cfg.dropout)
        self.item_tower = HybridItemTower(
            n_items,
            content_features,
            cfg.item_hidden,
            cfg.embedding_dim,
            cfg.dropout,
            content_cfg.id_weight,
            content_cfg.content_weight,
        )

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)
