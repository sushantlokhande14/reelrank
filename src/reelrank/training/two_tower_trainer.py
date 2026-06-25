"""Train the two-tower model with in-batch-negative sampled softmax.

For each positive (user, item) pair in a batch, the other items in the batch act
as negatives. We apply the logQ / sampling-bias correction of Yi et al. (2019):
subtracting log p(item) counteracts popular items being over-represented as
in-batch negatives, which materially improves retrieval quality.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reelrank.config import Config
from reelrank.data.movielens import SPLIT_TRAIN, MovieLensData
from reelrank.models.factory import build_model
from reelrank.utils import resolve_device


def _item_log_prob(data: MovieLensData) -> np.ndarray:
    """log p(item) over training interactions, with add-one smoothing."""
    counts = np.ones(data.n_items, dtype=np.float64)
    vc = data.split_frame(SPLIT_TRAIN)["item_idx"].value_counts()
    counts[vc.index.to_numpy()] += vc.to_numpy()
    return np.log(counts / counts.sum()).astype(np.float32)


def train_two_tower(
    cfg: Config,
    data: MovieLensData,
    content_features: np.ndarray | None = None,
    train_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Train the two-tower. When content_features is given and content_weight > 0,
    the content-aware hybrid item tower is used (otherwise pure collaborative).

    train_pairs overrides the (user_idx, item_idx) training interactions, which the
    cold-start experiment uses to withhold a set of items from training entirely.
    """
    device = resolve_device(cfg.train.device)
    model = build_model(cfg, data, content_features).to(device)

    if train_pairs is None:
        train = data.split_frame(SPLIT_TRAIN)
        user_arr = train["user_idx"].to_numpy()
        item_arr = train["item_idx"].to_numpy()
    else:
        user_arr, item_arr = train_pairs
    users = torch.as_tensor(user_arr, dtype=torch.long)
    items = torch.as_tensor(item_arr, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(users, items),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
    )

    log_p = torch.from_numpy(_item_log_prob(data)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    temperature = cfg.train.temperature

    for epoch in range(cfg.train.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)

            user_emb, item_emb = model(batch_users, batch_items)
            logits = user_emb @ item_emb.t() / temperature  # (B, B)
            logits = logits - log_p[batch_items].unsqueeze(0)  # logQ correction per column
            labels = torch.arange(batch_users.size(0), device=device)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        if verbose:
            print(f"epoch {epoch + 1:>2}/{cfg.train.epochs}  loss {running / max(n_batches, 1):.4f}")

    return model


@torch.no_grad()
def export_embeddings(
    model: nn.Module, data: MovieLensData, batch: int = 16384
) -> tuple[np.ndarray, np.ndarray]:
    """Run both towers over every id and return (user_emb, item_emb) as float32."""
    model.eval()
    device = next(model.parameters()).device

    def encode(tower: nn.Module, n: int) -> np.ndarray:
        chunks = []
        for start in range(0, n, batch):
            ids = torch.arange(start, min(start + batch, n), device=device)
            chunks.append(tower(ids).cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    user_emb = encode(model.user_tower, data.n_users)
    item_emb = encode(model.item_tower, data.n_items)
    return user_emb, item_emb
