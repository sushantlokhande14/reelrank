"""A small TMDB API client.

Covers what the recommender needs: trending and now-playing lists, full movie
details with credits (so we get plot, cast, and director), and poster URLs. The
key is read from the TMDB_API_KEY environment variable; nothing secret is hard
coded. Requests retry briefly on transient errors and 429 rate limits.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

from reelrank.config import TmdbCfg


@dataclass
class MovieMeta:
    """Normalized TMDB movie metadata used across the project."""

    tmdb_id: int
    title: str
    year: str
    genres: list[str]
    overview: str
    director: str
    cast: list[str] = field(default_factory=list)
    poster_url: str | None = None
    popularity: float = 0.0
    vote_average: float = 0.0


class TMDBClient:
    def __init__(self, cfg: TmdbCfg | None = None, api_key: str | None = None) -> None:
        self.cfg = cfg or TmdbCfg()
        self.api_key = api_key or os.environ.get("TMDB_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TMDB_API_KEY is not set. Put it in reelrank/.env or the host env."
            )
        self.session = requests.Session()

    def _get(self, path: str, **params) -> dict:
        params["api_key"] = self.api_key
        params.setdefault("language", self.cfg.language)
        url = f"{self.cfg.base_url}/{path}"
        for attempt in range(4):
            resp = self.session.get(url, params=params, timeout=self.cfg.request_timeout)
            if resp.status_code == 429:  # rate limited: honor Retry-After
                time.sleep(float(resp.headers.get("Retry-After", 1)) + 0.5)
                continue
            if resp.status_code >= 500 and attempt < 3:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def trending(self, window: str = "day", page: int = 1) -> list[dict]:
        return self._get(f"trending/movie/{window}", page=page).get("results", [])

    def now_playing(self, page: int = 1) -> list[dict]:
        return self._get("movie/now_playing", page=page, region=self.cfg.region).get(
            "results", []
        )

    def movie_details(self, tmdb_id: int) -> dict:
        return self._get(f"movie/{tmdb_id}", append_to_response="credits")

    def poster_url(self, poster_path: str | None) -> str | None:
        if not poster_path:
            return None
        return f"{self.cfg.image_base}/{self.cfg.poster_size}{poster_path}"

    def fetch_meta(self, tmdb_id: int) -> MovieMeta:
        """Full normalized metadata for one movie."""
        details = self.movie_details(tmdb_id)
        credits = details.get("credits", {})
        crew = credits.get("crew", [])
        cast = credits.get("cast", [])
        director = next((c["name"] for c in crew if c.get("job") == "Director"), "")
        return MovieMeta(
            tmdb_id=int(details["id"]),
            title=details.get("title", ""),
            year=(details.get("release_date") or "")[:4],
            genres=[g["name"] for g in details.get("genres", [])],
            overview=details.get("overview", "") or "",
            director=director,
            cast=[c["name"] for c in cast[: self.cfg.max_cast]],
            poster_url=self.poster_url(details.get("poster_path")),
            popularity=float(details.get("popularity", 0.0)),
            vote_average=float(details.get("vote_average", 0.0)),
        )


def tmdb_content_text(meta: MovieMeta) -> str:
    """Build the content document for a movie from its TMDB metadata, matching the
    style used for MovieLens items (title, genres, overview, cast, director)."""
    parts = [f"{meta.title} ({meta.year})" if meta.year else meta.title]
    if meta.genres:
        parts.append("Genres: " + ", ".join(meta.genres))
    if meta.overview:
        parts.append(meta.overview)
    if meta.cast:
        parts.append("Cast: " + ", ".join(meta.cast))
    if meta.director:
        parts.append("Director: " + meta.director)
    return ". ".join(parts)
