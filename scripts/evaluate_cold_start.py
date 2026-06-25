"""Cold-start evaluation.

Withhold a random set of items from training entirely (their id embedding never
learns anything), then represent them at serving time from content alone via the
hybrid item tower's ``embed_cold`` path, the same path a brand-new TMDB title with
no ratings would take. We then measure how well those never-trained items are
retrieved for users who liked them, against a no-content control that gives cold
items a random vector (what a purely collaborative system effectively has).

    python scripts/evaluate_cold_start.py --config config/default.yaml --cold-frac 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from reelrank.config import load_config
from reelrank.data.movielens import SPLIT_TRAIN, build_dataset
from reelrank.eval.harness import build_ground_truth
from reelrank.eval.metrics import evaluate_rankings
from reelrank.features.content import build_content_embeddings
from reelrank.report import metrics_table
from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.retrieval.recommender import TwoTowerRecommender
from reelrank.training.two_tower_trainer import train_two_tower
from reelrank.utils import resolve_device, set_seed


@torch.no_grad()
def _user_embeddings(model: torch.nn.Module, n_users: int, device: str) -> np.ndarray:
    ids = torch.arange(n_users, device=device)
    return model.user_tower(ids).cpu().numpy().astype(np.float32)


@torch.no_grad()
def _catalog(
    model: torch.nn.Module,
    content_emb: np.ndarray,
    cold_mask: np.ndarray,
    device: str,
    cold_mode: str,
) -> np.ndarray:
    n_items, dim = content_emb.shape[0], model.item_tower.id_embed.embedding_dim
    out_dim = model.user_tower(torch.zeros(1, dtype=torch.long, device=device)).shape[1]
    catalog = np.zeros((n_items, out_dim), dtype=np.float32)

    warm_ids = np.where(~cold_mask)[0]
    cold_ids = np.where(cold_mask)[0]
    catalog[warm_ids] = (
        model.item_tower(torch.as_tensor(warm_ids, device=device)).cpu().numpy()
    )

    if cold_mode == "content":
        cvec = torch.as_tensor(content_emb[cold_ids], device=device)
        catalog[cold_ids] = model.item_tower.embed_cold(cvec).cpu().numpy()
    else:  # no-content control: a random unit vector per cold item
        rng = np.random.default_rng(0)
        r = rng.standard_normal((len(cold_ids), out_dim)).astype(np.float32)
        catalog[cold_ids] = r / np.linalg.norm(r, axis=1, keepdims=True)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cold-frac", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.content.content_weight <= 0:
        raise SystemExit("cold-start needs the hybrid model (content_weight > 0)")
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)

    data = build_dataset(cfg)
    content_emb = build_content_embeddings(cfg, data)

    rng = np.random.default_rng(cfg.seed)
    cold_mask = rng.random(data.n_items) < args.cold_frac
    n_cold = int(cold_mask.sum())

    # Train on interactions whose item is NOT cold.
    train = data.split_frame(SPLIT_TRAIN)
    keep = ~cold_mask[train["item_idx"].to_numpy()]
    pairs = (train["user_idx"].to_numpy()[keep], train["item_idx"].to_numpy()[keep])
    print(f"items: {data.n_items:,}  cold (withheld from training): {n_cold:,}")
    print(f"training on {keep.sum():,} of {len(train):,} interactions")
    model = train_two_tower(cfg, data, content_features=content_emb, train_pairs=pairs)

    # Ground truth restricted to cold items the user genuinely liked in test.
    full_gt, train_sets = build_ground_truth(data)
    cold_gt = {u: {i for i in items if cold_mask[i]} for u, items in full_gt.items()}
    cold_gt = {u: s for u, s in cold_gt.items() if s}
    users = list(cold_gt)
    excl = {u: train_sets[u] for u in users}
    print(f"cold-start eval users: {len(users):,} (have a held-out cold-item positive)\n")

    user_emb = _user_embeddings(model, data.n_users, device)
    max_k = max(cfg.eval.k_values)
    results = {}
    for label, mode in [("hybrid (content cold-start)", "content"), ("no content (control)", "random")]:
        catalog = _catalog(model, content_emb, cold_mask, device, mode)
        index = ProximaIndex.from_config(catalog.shape[1], cfg.retrieval).build(
            catalog, labels=np.arange(data.n_items, dtype=np.int64)
        )
        rec = TwoTowerRecommender(user_emb, index, data.n_items)
        recs = rec.recommend_all(users, max_k, excl)
        results[label] = evaluate_rankings(recs, cold_gt, cfg.eval.k_values)

    print("Recall/NDCG/MAP on items never seen during training:\n")
    print(metrics_table(results, cfg.eval.k_values))


if __name__ == "__main__":
    main()
