"""CLI demo for natural-language vibe search over the live serving index.

    python scripts/demo_search.py --query "a slow-burn sci-fi like Arrival but funnier"

Requires a serving index (run scripts/refresh_tmdb.py, or build one without the
live titles by adapting it). Loads the sentence model and two-tower on first query.
"""

from __future__ import annotations

import argparse

from reelrank.config import load_config
from reelrank.serving.engine import RecommendEngine
from reelrank.serving.explain import reason_for
from reelrank.utils import load_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--query", default="a slow-burn sci-fi like Arrival but funnier")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config)
    engine = RecommendEngine.from_artifacts(cfg)

    print(f"query: {args.query!r}\n")
    for item in engine.search_text(args.query, k=args.k):
        tag = " [now in theaters]" if item.get("source") == "tmdb_live" else ""
        print(f"  - {item['title']}{tag}")
        print(f"      {reason_for(args.query, item)}")


if __name__ == "__main__":
    main()
