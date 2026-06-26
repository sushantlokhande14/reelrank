"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 18


class RecommendRequest(BaseModel):
    seed_ids: list[int]
    k: int = 18


class Movie(BaseModel):
    id: int
    title: str
    year: str | None = None
    genres: list[str] = []
    poster_url: str | None = None
    source: str = "movielens"
    overview: str | None = None
    reason: str | None = None


class Results(BaseModel):
    results: list[Movie]
