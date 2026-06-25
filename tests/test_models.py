"""Two-tower and hybrid model shapes, normalization, and the cold-start path."""

import numpy as np
import torch

from reelrank.config import Config
from reelrank.models.hybrid import HybridTwoTower
from reelrank.models.two_tower import TwoTower


def test_two_tower_outputs_unit_vectors():
    cfg = Config()
    model = TwoTower(n_users=10, n_items=20, cfg=cfg.model)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    user_emb, item_emb = model(users, items)
    assert user_emb.shape == (3, cfg.model.embedding_dim)
    assert item_emb.shape == (3, cfg.model.embedding_dim)
    assert torch.allclose(user_emb.norm(dim=1), torch.ones(3), atol=1e-5)


def test_hybrid_item_tower_cold_path():
    cfg = Config()
    content = np.random.default_rng(0).standard_normal((20, 16)).astype(np.float32)
    model = HybridTwoTower(
        n_users=10, n_items=20, content_features=content, cfg=cfg.model, content_cfg=cfg.content
    )
    warm = model.item_tower(torch.tensor([0, 5, 9]))
    assert warm.shape == (3, cfg.model.embedding_dim)

    # an item with no id (a brand-new movie) is embeddable from content alone
    cold = model.item_tower.embed_cold(torch.from_numpy(content[:3]))
    assert cold.shape == (3, cfg.model.embedding_dim)
    assert torch.allclose(cold.norm(dim=1), torch.ones(3), atol=1e-5)
