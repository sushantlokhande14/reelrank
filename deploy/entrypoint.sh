#!/bin/sh
set -e

# If a TMDB key is present, refresh the live catalog so the index includes current
# titles. Falls back to the baked MovieLens-only index if the refresh fails.
if [ -n "$TMDB_API_KEY" ]; then
  echo "refreshing live catalog from TMDB..."
  python scripts/refresh_tmdb.py --config config/default.yaml || \
    echo "refresh failed; serving the baked index"
fi

exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}"
