"""Two-tower retrieval model.

Separate user and item encoders map ids into one shared embedding space. Outputs
are L2-normalized, so an inner-product index ranks by cosine similarity and the
int8 (SQ8) codes in Proxima stay well-scaled.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from reelrank.config import ModelCfg


class Tower(nn.Module):
    """Id embedding followed by an MLP, producing an L2-normalized vector."""

    def __init__(self, n_ids: int, hidden: Sequence[int], out_dim: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_ids, hidden[0])
        nn.init.normal_(self.embed.weight, std=0.05)

        dims = list(hidden) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on the final projection
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.embed(ids))
        return F.normalize(x, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, cfg: ModelCfg) -> None:
        super().__init__()
        self.user_tower = Tower(n_users, cfg.user_hidden, cfg.embedding_dim, cfg.dropout)
        self.item_tower = Tower(n_items, cfg.item_hidden, cfg.embedding_dim, cfg.dropout)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_tower(user_ids), self.item_tower(item_ids)
