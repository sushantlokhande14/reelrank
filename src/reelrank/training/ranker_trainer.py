"""Train the stage-2 ranker with listwise sampled-softmax.

Positives are training interactions; negatives are sampled from a popularity^alpha
distribution. A note on a tempting idea that does not work here: drawing negatives
from the items Proxima retrieves ("hard negatives") sounds right, but on sparse
implicit feedback those retrieved-yet-unobserved items include the user's held-out
future positives, so training against them teaches the ranker to bury exactly what
the eval rewards. Popularity-sampled negatives avoid that contamination.

The ranker reuses the frozen two-tower embeddings and adds features the retriever
did not directly use, an explicit content-match term and item popularity, so it
can refine the head of the ranking rather than merely reproduce the retrieval
score. Loss is listwise softmax over [positive, N negatives].
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reelrank.config import Config
from reelrank.data.movielens import (
    SPLIT_TRAIN,
    MovieLensData,
    item_log_popularity,
    item_train_counts,
)
from reelrank.features.content import user_content_profiles
from reelrank.models.ranker import Ranker
from reelrank.utils import resolve_device

N_EXTRA = 2  # content-match score + item popularity


def _negative_sampling_probs(data: MovieLensData, alpha: float) -> np.ndarray:
    weights = np.power(item_train_counts(data).astype(np.float64), alpha)
    total = weights.sum()
    if total <= 0:
        weights = np.ones_like(weights)
        total = weights.sum()
    return weights / total


def train_ranker(
    cfg: Config,
    data: MovieLensData,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    content_emb: np.ndarray,
    verbose: bool = True,
) -> Ranker:
    device = resolve_device(cfg.train.device)
    rc = cfg.ranker
    emb_dim = item_emb.shape[1]
    ranker = Ranker(emb_dim, N_EXTRA, rc.hidden, rc.dropout).to(device)

    user_t = torch.from_numpy(np.ascontiguousarray(user_emb, dtype=np.float32)).to(device)
    item_t = torch.from_numpy(np.ascontiguousarray(item_emb, dtype=np.float32)).to(device)
    pop_t = torch.from_numpy(item_log_popularity(data)).to(device)
    item_content_t = torch.from_numpy(np.ascontiguousarray(content_emb, dtype=np.float32)).to(device)
    profile_t = torch.from_numpy(user_content_profiles(data, content_emb)).to(device)
    neg_probs = torch.from_numpy(_negative_sampling_probs(data, rc.neg_alpha)).to(device)

    train = data.split_frame(SPLIT_TRAIN)
    if verbose:
        print(f"ranker trains on {len(train):,} positives")
    users = torch.as_tensor(train["user_idx"].to_numpy(), dtype=torch.long)
    items = torch.as_tensor(train["item_idx"].to_numpy(), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(users, items), batch_size=rc.batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.Adam(ranker.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n_neg = rc.n_negatives

    for epoch in range(rc.epochs):
        ranker.train()
        running, n_batches = 0.0, 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch = batch_users.size(0)

            negatives = torch.multinomial(neg_probs, batch * n_neg, replacement=True).view(
                batch, n_neg
            )
            candidates = torch.cat([batch_items.unsqueeze(1), negatives], dim=1)

            user_vecs = user_t[batch_users].unsqueeze(1).expand(batch, n_neg + 1, emb_dim)
            item_vecs = item_t[candidates]
            content_match = (profile_t[batch_users].unsqueeze(1) * item_content_t[candidates]).sum(-1)
            extra = torch.stack([content_match, pop_t[candidates]], dim=-1)
            scores = ranker(user_vecs, item_vecs, extra)

            target = torch.zeros(batch, dtype=torch.long, device=device)
            loss = loss_fn(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        if verbose:
            print(f"ranker epoch {epoch + 1}/{rc.epochs}  loss {running / max(n_batches, 1):.4f}")

    return ranker
