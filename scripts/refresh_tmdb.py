"""Daily refresh: pull the live TMDB catalog and rebuild the serving index.

Current titles (trending + now-playing) are embedded from their content via the
hybrid item tower's cold-start path and concatenated with the trained MovieLens
item embeddings, so recommendations include movies released after the dataset
froze. Writes a serving index, embeddings, and a catalog (titles + posters) under
artifacts/<dataset>/serving/.

    python scripts/refresh_tmdb.py --config config/default.yaml

Requires TMDB_API_KEY (in reelrank/.env or the host env) and a trained two-tower
(run scripts/train_two_tower.py first).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from reelrank.config import load_config
from reelrank.data.movielens import build_dataset
from reelrank.features.content import build_content_embeddings
from reelrank.features.embedder import ContentEmbedder
from reelrank.retrieval.proxima_index import ProximaIndex
from reelrank.tmdb.catalog import fetch_live_catalog, movielens_to_tmdb
from reelrank.tmdb.client import TMDBClient, tmdb_content_text
from reelrank.training.pipeline import load_two_tower_model
from reelrank.utils import load_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config)
    data = build_dataset(cfg)
    content_emb = build_content_embeddings(cfg, data)
    model = load_two_tower_model(cfg, data, content_emb)

    client = TMDBClient(cfg.tmdb)
    metas = fetch_live_catalog(client, cfg.tmdb.live_pages)
    print(f"fetched {len(metas)} live titles from TMDB")

    embedder = ContentEmbedder(cfg.content.model)
    live_content = embedder.encode(
        [tmdb_content_text(m) for m in metas], batch_size=cfg.content.batch_size
    )
    with torch.no_grad():
        live_emb = (
            model.item_tower.embed_cold(torch.from_numpy(live_content)).cpu().numpy().astype(np.float32)
        )

    ml_emb = np.load(Path(cfg.paths.artifacts) / cfg.data.dataset / "item_embeddings.npy")
    serving_emb = np.concatenate([ml_emb, live_emb], axis=0)
    index = ProximaIndex.from_config(serving_emb.shape[1], cfg.retrieval).build(
        serving_emb, labels=np.arange(serving_emb.shape[0], dtype=np.int64)
    )

    ml_to_tmdb = movielens_to_tmdb(cfg, data)
    catalog: list[dict] = []
    for row in data.movies.itertuples(index=False):
        catalog.append(
            {
                "id": int(row.item_idx),
                "source": "movielens",
                "title": str(row.title),
                "genres": [g for g in str(row.genres).split("|") if g],
                "tmdb_id": ml_to_tmdb.get(int(row.item_idx)),
                "poster_url": None,
            }
        )
    for offset, meta in enumerate(metas):
        catalog.append(
            {
                "id": data.n_items + offset,
                "source": "tmdb_live",
                "title": meta.title,
                "year": meta.year,
                "genres": meta.genres,
                "overview": meta.overview,
                "director": meta.director,
                "cast": meta.cast,
                "poster_url": meta.poster_url,
                "tmdb_id": meta.tmdb_id,
            }
        )

    serve_dir = Path(cfg.paths.artifacts) / cfg.data.dataset / "serving"
    serve_dir.mkdir(parents=True, exist_ok=True)
    np.save(serve_dir / "embeddings.npy", serving_emb)
    index.save(str(serve_dir / "index.idx"))
    (serve_dir / "catalog.json").write_text(json.dumps(catalog))
    (serve_dir / "refreshed_at.txt").write_text(datetime.now(timezone.utc).isoformat())

    print(
        f"serving index rebuilt: {len(index):,} items "
        f"({data.n_items:,} MovieLens + {len(metas)} live) -> {serve_dir}"
    )


if __name__ == "__main__":
    main()
