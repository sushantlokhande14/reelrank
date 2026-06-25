"""Load MovieLens, filter to implicit-feedback positives, and build a
leakage-free temporal train/val/test split.

Protocol (kept deliberately honest for the metrics we report):

1. Keep ratings >= ``positive_threshold`` as positive interactions.
2. Iteratively drop users/items below the min-interaction thresholds (k-core).
3. Split globally by timestamp: the earliest interactions are train, the latest
   are test. Nothing from the future leaks into training.
4. Build the user/item vocabulary from TRAIN ONLY. A purely collaborative model
   cannot embed a user or item it never saw, so val/test rows referencing unseen
   users/items are dropped here and handled later as the cold-start case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from reelrank.config import Config
from reelrank.data.download import download_movielens

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


@dataclass
class MovieLensData:
    """A filtered, index-encoded, temporally split MovieLens dataset."""

    ratings: pd.DataFrame  # columns: user_idx, item_idx, rating, timestamp, split
    movies: pd.DataFrame   # columns: item_idx, movieId, title, genres
    n_users: int
    n_items: int
    user_id_map: dict[int, int]  # original userId -> contiguous user_idx
    item_id_map: dict[int, int]  # original movieId -> contiguous item_idx

    def split_frame(self, split: str) -> pd.DataFrame:
        return self.ratings[self.ratings["split"] == split]


def build_dataset(cfg: Config, use_cache: bool = True) -> MovieLensData:
    """Build (or load from cache) the processed MovieLens dataset."""
    cache_dir = Path(cfg.paths.data_processed) / cfg.data.dataset
    if use_cache and _cache_valid(cache_dir, cfg):
        return _load_cache(cache_dir)

    folder = download_movielens(cfg.data.dataset, cfg.paths.data_raw)
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv")

    pos = ratings[ratings["rating"] >= cfg.data.positive_threshold].copy()
    pos = _kcore_filter(pos, cfg.data.min_user_interactions, cfg.data.min_item_interactions)
    pos = _assign_split(pos, cfg)

    train = pos[pos["split"] == SPLIT_TRAIN]
    user_ids = np.sort(train["userId"].unique())
    item_ids = np.sort(train["movieId"].unique())
    user_id_map = {int(u): i for i, u in enumerate(user_ids)}
    item_id_map = {int(m): i for i, m in enumerate(item_ids)}

    pos = pos[pos["userId"].isin(user_id_map) & pos["movieId"].isin(item_id_map)].copy()
    pos["user_idx"] = pos["userId"].map(user_id_map).astype(np.int64)
    pos["item_idx"] = pos["movieId"].map(item_id_map).astype(np.int64)

    movies = movies[movies["movieId"].isin(item_id_map)].copy()
    movies["item_idx"] = movies["movieId"].map(item_id_map).astype(np.int64)
    movies = movies.sort_values("item_idx").reset_index(drop=True)

    ratings_out = pos[["user_idx", "item_idx", "rating", "timestamp", "split"]].reset_index(
        drop=True
    )
    movies_out = movies[["item_idx", "movieId", "title", "genres"]].reset_index(drop=True)

    data = MovieLensData(
        ratings=ratings_out,
        movies=movies_out,
        n_users=len(user_id_map),
        n_items=len(item_id_map),
        user_id_map=user_id_map,
        item_id_map=item_id_map,
    )
    if use_cache:
        _save_cache(cache_dir, cfg, data)
    return data


def user_item_lists(data: MovieLensData, split: str) -> dict[int, np.ndarray]:
    """Map each user_idx to the array of item_idx they interacted with in `split`."""
    grouped = data.split_frame(split).groupby("user_idx")["item_idx"]
    return {int(u): v.to_numpy(dtype=np.int64) for u, v in grouped}


def item_train_counts(data: MovieLensData) -> np.ndarray:
    """Per-item training interaction count, indexed by item_idx."""
    counts = np.zeros(data.n_items, dtype=np.int64)
    vc = data.split_frame(SPLIT_TRAIN)["item_idx"].value_counts()
    counts[vc.index.to_numpy()] = vc.to_numpy()
    return counts


def item_log_popularity(data: MovieLensData) -> np.ndarray:
    """Standardized log popularity feature, indexed by item_idx (a ranking signal)."""
    feature = np.log1p(item_train_counts(data).astype(np.float64))
    feature = (feature - feature.mean()) / (feature.std() + 1e-8)
    return feature.astype(np.float32)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _kcore_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Repeatedly drop users/items below the thresholds until stable."""
    while True:
        before = len(df)
        uc = df["userId"].value_counts()
        df = df[df["userId"].isin(uc[uc >= min_user].index)]
        ic = df["movieId"].value_counts()
        df = df[df["movieId"].isin(ic[ic >= min_item].index)]
        if len(df) == before:
            return df.reset_index(drop=True)


def _assign_split(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.split.strategy == "global":
        return _global_temporal_split(df, cfg.split.val_fraction, cfg.split.test_fraction)
    if cfg.split.strategy == "user":
        return _user_temporal_split(df, cfg.split.val_fraction, cfg.split.test_fraction)
    raise ValueError(f"unknown split strategy {cfg.split.strategy!r}")


def _global_temporal_split(df: pd.DataFrame, val_frac: float, test_frac: float) -> pd.DataFrame:
    """One global timestamp cut. The latest interactions become test; nothing from
    the future leaks into training. mergesort keeps it stable/deterministic."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1.0 - val_frac - test_frac))
    val_end = int(n * (1.0 - test_frac))
    split = np.empty(n, dtype=object)
    split[:train_end] = SPLIT_TRAIN
    split[train_end:val_end] = SPLIT_VAL
    split[val_end:] = SPLIT_TEST
    df["split"] = split
    return df


def _user_temporal_split(df: pd.DataFrame, val_frac: float, test_frac: float) -> pd.DataFrame:
    """Per-user temporal holdout: each user's most recent interactions become test,
    the slice before them validation. A user's own future never trains their past,
    and every active user contributes to evaluation (stable on sparse data)."""
    df = df.sort_values(["userId", "timestamp"], kind="mergesort").reset_index(drop=True)
    labels: list[str] = []
    for _, group in df.groupby("userId", sort=False):
        n = len(group)
        n_test = min(int(round(n * test_frac)), max(0, n - 1))
        n_val = min(int(round(n * val_frac)), max(0, n - 1 - n_test))
        n_train = n - n_val - n_test
        labels.extend([SPLIT_TRAIN] * n_train + [SPLIT_VAL] * n_val + [SPLIT_TEST] * n_test)
    df["split"] = labels
    return df


def _cache_signature(cfg: Config) -> dict:
    return {
        "dataset": cfg.data.dataset,
        "positive_threshold": cfg.data.positive_threshold,
        "min_user_interactions": cfg.data.min_user_interactions,
        "min_item_interactions": cfg.data.min_item_interactions,
        "split_strategy": cfg.split.strategy,
        "val_fraction": cfg.split.val_fraction,
        "test_fraction": cfg.split.test_fraction,
    }


def _cache_valid(cache_dir: Path, cfg: Config) -> bool:
    meta = cache_dir / "meta.json"
    if not (meta.exists() and (cache_dir / "ratings.parquet").exists()):
        return False
    return json.loads(meta.read_text()) == _cache_signature(cfg)


def _save_cache(cache_dir: Path, cfg: Config, data: MovieLensData) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.ratings.to_parquet(cache_dir / "ratings.parquet")
    data.movies.to_parquet(cache_dir / "movies.parquet")
    (cache_dir / "maps.json").write_text(
        json.dumps({"user_id_map": data.user_id_map, "item_id_map": data.item_id_map})
    )
    (cache_dir / "meta.json").write_text(json.dumps(_cache_signature(cfg)))


def _load_cache(cache_dir: Path) -> MovieLensData:
    ratings = pd.read_parquet(cache_dir / "ratings.parquet")
    movies = pd.read_parquet(cache_dir / "movies.parquet")
    maps = json.loads((cache_dir / "maps.json").read_text())
    user_id_map = {int(k): int(v) for k, v in maps["user_id_map"].items()}
    item_id_map = {int(k): int(v) for k, v in maps["item_id_map"].items()}
    return MovieLensData(
        ratings=ratings,
        movies=movies,
        n_users=len(user_id_map),
        n_items=len(item_id_map),
        user_id_map=user_id_map,
        item_id_map=item_id_map,
    )
