# Deploying reelrank

Two pieces: the FastAPI backend (Docker, on Render) and the React frontend (on
Vercel). Both have free/cheap tiers.

## 1. Backend on Render

The image builds the Proxima extension, trains the two-tower, and bakes a
serving index, all on Linux (which also avoids the Windows native-library issue).

1. Push this repo to GitHub (already done).
2. In Render, **New > Blueprint**, point it at the repo. It reads `render.yaml`
   and creates the `reelrank-api` Docker web service.
3. Set environment variables on the service:
   - `TMDB_API_KEY` — your TMDB key (enables live titles + onboarding posters).
   - `ALLOW_ORIGINS` — your Vercel URL, e.g. `https://reelrank.vercel.app`.
4. Deploy. The build takes a few minutes (it compiles Proxima and trains the
   model). When it's live, `GET /health` returns `{"status":"ok"}`.

Notes:
- The model loads lazily on the first real request, so `/health` stays cheap for
  warm pings. The frontend pings `/warmup` on load.
- torch + the sentence model need memory; if the `starter` plan OOMs, bump the
  plan. The Proxima SQ8 index itself is tiny.

## 2. Frontend on Vercel

1. In Vercel, **Add New > Project**, import the repo.
2. Set **Root Directory** to `frontend`. Vercel detects Vite from `vercel.json`.
3. Add an environment variable:
   - `VITE_API_BASE` — your Render backend URL, e.g.
     `https://reelrank-api.onrender.com`.
4. Deploy. Then set `ALLOW_ORIGINS` on Render to the Vercel URL and redeploy the
   backend so CORS allows the frontend.

## 3. Daily live-catalog refresh

`.github/workflows/refresh.yml` runs daily and rebuilds the serving index with
current TMDB titles (it needs the `TMDB_API_KEY` repo secret, already set). The
running backend also refreshes on startup when `TMDB_API_KEY` is present, so a
redeploy or restart picks up fresh titles.

## 4. Portfolio link

The portfolio's reelrank entry carries a `live` field pointing at the Vercel URL,
which renders a "Live demo" button. The resume project name deep-links to the
portfolio case study, and from there to the live demo and the code.
