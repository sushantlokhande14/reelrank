"""Build a short text document per movie and embed it with a sentence-transformer.

The document is assembled from MovieLens metadata available today (title, genres,
top user tags). It is designed to be enriched with TMDB fields (plot, cast,
director) later: ``build_item_documents`` simply appends whatever extra text is
provided per movieId.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from reelrank.config import Config
from reelrank.data.download import download_movielens
from reelrank.data.movielens import SPLIT_TRAIN, MovieLensData, user_item_lists

NO_GENRES = "(no genres listed)"


def user_content_profiles(data: MovieLensData, content_emb: np.ndarray) -> np.ndarray:
    """Per-user content taste vector: the L2-normalized mean of the content
    embeddings of the items they liked in training. Shape (n_users, content_dim)."""
    profiles = np.zeros((data.n_users, content_emb.shape[1]), dtype=np.float32)
    for user, items in user_item_lists(data, SPLIT_TRAIN).items():
        if len(items):
            profiles[user] = content_emb[items].mean(axis=0)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return profiles / norms


def build_item_documents(
    cfg: Config,
    data: MovieLensData,
    extra_text: Mapping[int, str] | None = None,
) -> list[str]:
    """One content string per item_idx. ``extra_text`` maps movieId -> extra text
    (e.g. a TMDB plot/cast/director blob) appended to the document."""
    folder = download_movielens(cfg.data.dataset, cfg.paths.data_raw)
    tag_map = _aggregate_tags(folder, data.item_id_map, cfg.content.max_tags)
    extra_text = extra_text or {}

    docs = [""] * data.n_items
    for row in data.movies.itertuples(index=False):
        movie_id = int(row.movieId)
        parts = [str(row.title)]

        genres = [g for g in str(row.genres).split("|") if g and g != NO_GENRES]
        if genres:
            parts.append("Genres: " + ", ".join(genres))

        tags = tag_map.get(movie_id)
        if tags:
            parts.append("Tags: " + ", ".join(tags))

        if movie_id in extra_text and extra_text[movie_id]:
            parts.append(str(extra_text[movie_id]))

        docs[int(row.item_idx)] = ". ".join(parts)
    return docs


def build_content_embeddings(
    cfg: Config,
    data: MovieLensData,
    use_cache: bool = True,
) -> np.ndarray:
    """L2-normalized sentence embeddings, shape (n_items, content_dim). Cached."""
    artifacts = Path(cfg.paths.artifacts) / cfg.data.dataset
    emb_path = artifacts / "content_embeddings.npy"
    meta_path = artifacts / "content_meta.json"
    signature = {
        "model": cfg.content.model,
        "max_tags": cfg.content.max_tags,
        "n_items": data.n_items,
    }
    if (
        use_cache
        and emb_path.exists()
        and meta_path.exists()
        and json.loads(meta_path.read_text()) == signature
    ):
        return np.load(emb_path)

    from reelrank.features.embedder import ContentEmbedder

    docs = build_item_documents(cfg, data)
    embedder = ContentEmbedder(cfg.content.model)
    embeddings = embedder.encode(docs, batch_size=cfg.content.batch_size)

    artifacts.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    meta_path.write_text(json.dumps(signature))
    return embeddings


def _aggregate_tags(
    folder: Path, item_id_map: Mapping[int, int], max_tags: int
) -> dict[int, list[str]]:
    """Top-`max_tags` user tags per movie, ordered by how often they were applied."""
    tags_path = folder / "tags.csv"
    if not tags_path.exists():
        return {}
    tags = pd.read_csv(tags_path)
    tags = tags[tags["movieId"].isin(item_id_map)]
    counts = (
        tags.groupby(["movieId", "tag"]).size().reset_index(name="n")
        .sort_values(["movieId", "n"], ascending=[True, False])
    )
    out: dict[int, list[str]] = {}
    for movie_id, group in counts.groupby("movieId"):
        out[int(movie_id)] = [str(t) for t in group["tag"].head(max_tags)]
    return out
