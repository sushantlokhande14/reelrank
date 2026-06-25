"""Bridge TMDB and MovieLens, and assemble the live catalog.

MovieLens ships a links.csv mapping its movieId to a TMDB id, which lets us attach
posters and metadata to known movies and, in the other direction, recognise when a
trending TMDB title is already in the dataset. The live catalog is the set of
current titles (trending + now-playing) that the daily refresh folds into the
serving index so recommendations include movies released after the dataset froze.
"""

from __future__ import annotations

import pandas as pd

from reelrank.config import Config
from reelrank.data.download import download_movielens
from reelrank.data.movielens import MovieLensData
from reelrank.tmdb.client import MovieMeta, TMDBClient


def load_links(cfg: Config) -> pd.DataFrame:
    folder = download_movielens(cfg.data.dataset, cfg.paths.data_raw)
    return pd.read_csv(folder / "links.csv")  # columns: movieId, imdbId, tmdbId


def movielens_to_tmdb(cfg: Config, data: MovieLensData) -> dict[int, int]:
    """item_idx -> tmdbId for the MovieLens items that carry a TMDB id."""
    links = load_links(cfg).dropna(subset=["tmdbId"])
    movieid_to_tmdb = {int(r.movieId): int(r.tmdbId) for r in links.itertuples(index=False)}
    return {
        item_idx: movieid_to_tmdb[movie_id]
        for movie_id, item_idx in data.item_id_map.items()
        if movie_id in movieid_to_tmdb
    }


def fetch_live_catalog(client: TMDBClient, pages: int = 2) -> list[MovieMeta]:
    """Trending + now-playing across `pages`, de-duplicated, with full metadata."""
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for page in range(1, pages + 1):
        for entry in client.trending(page=page) + client.now_playing(page=page):
            tmdb_id = int(entry["id"])
            if tmdb_id not in seen:
                seen.add(tmdb_id)
                ordered_ids.append(tmdb_id)

    metas: list[MovieMeta] = []
    for tmdb_id in ordered_ids:
        try:
            metas.append(client.fetch_meta(tmdb_id))
        except Exception:  # noqa: BLE001 - skip a title that fails, keep the rest
            continue
    return metas
