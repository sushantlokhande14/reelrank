"""reelrank serving API.

Thin wrapper over the RecommendEngine. The engine (sentence model, two-tower,
Proxima index) loads lazily on first use, so /health stays cheap for warm pings
and free-tier cold starts. Endpoints:

    GET  /health            liveness, no model load
    POST /warmup            force-load the engine (call once after a cold start)
    GET  /onboarding        recognizable titles with posters to pick from
    POST /search            natural-language vibe search
    POST /recommend         recommendations from a few liked titles (cold start)
    GET  /item/{id}         a single catalog entry
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import Movie, RecommendRequest, Results, SearchRequest
from reelrank.config import load_config
from reelrank.utils import load_env

load_env()
cfg = load_config(os.environ.get("REELRANK_CONFIG", "config/default.yaml"))

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from reelrank.serving.engine import RecommendEngine

        _engine = RecommendEngine.from_artifacts(cfg)
    return _engine


app = FastAPI(title="reelrank", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOW_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_movie(item: dict, reason: str | None = None) -> Movie:
    return Movie(
        id=item["id"],
        title=item.get("title", ""),
        year=item.get("year"),
        genres=item.get("genres", []),
        poster_url=item.get("poster_url"),
        source=item.get("source", "movielens"),
        overview=item.get("overview"),
        reason=reason,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/warmup")
def warmup() -> dict:
    engine = get_engine()
    return {"ready": True, "catalog": len(engine.catalog)}


@app.get("/onboarding", response_model=Results)
def onboarding(n: int = 24) -> Results:
    engine = get_engine()
    picks = [c for c in engine.catalog if c.get("onboarding") and c.get("poster_url")]
    return Results(results=[_to_movie(c) for c in picks[:n]])


@app.post("/search", response_model=Results)
def search(req: SearchRequest) -> Results:
    from reelrank.serving.explain import reason_for

    engine = get_engine()
    items = engine.search_text(req.query, k=req.k)
    return Results(results=[_to_movie(it, reason_for(req.query, it)) for it in items])


@app.post("/recommend", response_model=Results)
def recommend(req: RecommendRequest) -> Results:
    from reelrank.serving.explain import reason_for

    engine = get_engine()
    items = engine.recommend_from_seeds(req.seed_ids, k=req.k)
    return Results(results=[_to_movie(it, reason_for(None, it)) for it in items])


@app.get("/item/{item_id}", response_model=Movie)
def item(item_id: int) -> Movie:
    engine = get_engine()
    found = engine.item(item_id)
    if found is None:
        raise HTTPException(status_code=404, detail="item not found")
    return _to_movie(found)
