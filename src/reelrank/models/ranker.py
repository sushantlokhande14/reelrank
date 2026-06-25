"""Stage-2 neural ranker.

An MLP that re-scores a (user, candidate item) pair. To earn its place it needs
signal the retriever did not already use, so alongside the collaborative two-tower
embeddings it takes extra scalar features (a content-match score and item
popularity). Features per pair:

    [ user_emb | item_emb | user_emb * item_emb | dot(user, item) | extra... ]

Input dimension is 3 * emb_dim + 1 + n_extra.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Ranker(nn.Module):
    def __init__(self, emb_dim: int, n_extra: int, hidden: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.n_extra = n_extra
        in_dim = 3 * emb_dim + 1 + n_extra
        dims = [in_dim] + list(hidden) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user: torch.Tensor, item: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """Score pairs. user/item: (..., emb_dim); extra: (..., n_extra). Returns (...)."""
        cross = user * item
        dot = cross.sum(dim=-1, keepdim=True)
        feats = torch.cat([user, item, cross, dot, extra], dim=-1)
        return self.mlp(feats).squeeze(-1)
