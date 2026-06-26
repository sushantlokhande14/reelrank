"""Typed configuration, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class Paths(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    artifacts: Path = Path("artifacts")


class DataCfg(BaseModel):
    dataset: Literal["ml-latest-small", "ml-25m"] = "ml-latest-small"
    positive_threshold: float = 4.0
    min_user_interactions: int = 5
    min_item_interactions: int = 5


class SplitCfg(BaseModel):
    # global: one timestamp cut across all interactions (strict no-future-leakage).
    # user:   per-user temporal holdout of each user's most recent interactions.
    strategy: Literal["global", "user"] = "global"
    val_fraction: float = 0.1
    test_fraction: float = 0.1


class ModelCfg(BaseModel):
    embedding_dim: int = 64
    user_hidden: list[int] = [128, 64]
    item_hidden: list[int] = [128, 64]
    dropout: float = 0.1


class TrainCfg(BaseModel):
    epochs: int = 10
    batch_size: int = 4096
    lr: float = 3e-3
    weight_decay: float = 1e-4
    temperature: float = 0.05
    device: Literal["auto", "cpu", "cuda"] = "auto"


class RetrievalCfg(BaseModel):
    proxima_space: Literal["ip", "cosine", "l2"] = "ip"
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 128
    mode: Literal["float", "sq8"] = "sq8"
    candidates: int = 200


class ContentCfg(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    max_tags: int = 10
    # Weight on the content feature when fused into the item tower (0 disables the
    # hybrid path and the model is purely collaborative).
    id_weight: float = 1.0
    content_weight: float = 1.0


class TmdbCfg(BaseModel):
    base_url: str = "https://api.themoviedb.org/3"
    image_base: str = "https://image.tmdb.org/t/p"
    poster_size: str = "w500"
    language: str = "en-US"
    region: str = "US"
    max_cast: int = 5            # top-billed cast folded into the content text
    request_timeout: float = 15.0
    live_pages: int = 2          # pages of trending + now-playing to pull (20/page)
    onboarding_size: int = 60    # popular MovieLens titles enriched with posters for onboarding


class RankerCfg(BaseModel):
    hidden: list[int] = [128, 64]
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_negatives: int = 20        # sampled negatives per positive (listwise softmax)
    neg_alpha: float = 0.75      # popularity exponent for negative sampling


class EvalCfg(BaseModel):
    k_values: list[int] = [10, 20, 50, 100]


class Config(BaseModel):
    seed: int = 42
    paths: Paths = Paths()
    data: DataCfg = DataCfg()
    split: SplitCfg = SplitCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    content: ContentCfg = ContentCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    ranker: RankerCfg = RankerCfg()
    tmdb: TmdbCfg = TmdbCfg()
    eval: EvalCfg = EvalCfg()


def load_config(path: str | Path | None = None) -> Config:
    """Load config from a YAML file, or return defaults when path is None."""
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
