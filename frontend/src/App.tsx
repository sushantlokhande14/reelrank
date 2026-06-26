import { useEffect, useState } from "react";

import { api } from "./api";
import Onboarding from "./components/Onboarding";
import ResultsGrid from "./components/ResultsGrid";
import SearchBar from "./components/SearchBar";
import type { Movie } from "./types";

type Backend = "waking" | "ready" | "down";

export default function App() {
  const [backend, setBackend] = useState<Backend>("waking");
  const [onboard, setOnboard] = useState<Movie[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [results, setResults] = useState<Movie[]>([]);
  const [heading, setHeading] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let i = 0; i < 30 && !cancelled; i++) {
        try {
          await api.warmup(); // loads the model on a cold backend (can take ~15s)
          if (cancelled) return;
          setBackend("ready");
          const o = await api.onboarding(18);
          if (!cancelled) setOnboard(o.results);
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
      if (!cancelled) setBackend("down");
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function toggleSeed(id: number) {
    setSelected((s) =>
      s.includes(id) ? s.filter((x) => x !== id) : s.length < 5 ? [...s, id] : s,
    );
  }

  async function runSearch(query: string) {
    setBusy(true);
    setError(null);
    try {
      const r = await api.search(query);
      setResults(r.results);
      setHeading(`Results for “${query}”`);
    } catch {
      setError("Search failed. The backend may still be waking up.");
    } finally {
      setBusy(false);
    }
  }

  async function runRecommend() {
    setBusy(true);
    setError(null);
    try {
      const r = await api.recommend(selected);
      setResults(r.results);
      setHeading("Because you liked those");
    } catch {
      setError("Could not fetch recommendations. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  }

  function startOver() {
    setResults([]);
    setSelected([]);
    setHeading("");
  }

  return (
    <div className="wrap">
      <header className="site-header">
        <div className="brand">
          <span className="dot" />
          reel<span>rank</span>
        </div>
        <div className="header-tag">two-stage recommender · vector search over Proxima</div>
      </header>

      <section className="hero">
        <h1>Describe a vibe. Get movies.</h1>
        <p>
          A hybrid recommender that answers the feeling you're after, blending your taste with
          what's in theaters right now.
        </p>
        <SearchBar onSearch={runSearch} disabled={backend !== "ready" || busy} />
      </section>

      {results.length > 0 ? (
        <section>
          <div className="section-head">
            <h2>{heading}</h2>
            <button className="btn ghost" onClick={startOver}>
              Start over
            </button>
          </div>
          <ResultsGrid movies={results} showReason />
        </section>
      ) : backend === "waking" ? (
        <div className="center">
          <div className="spinner" />
          Waking the recommender… free-tier backends sleep when idle, so the first load takes a
          few seconds.
        </div>
      ) : backend === "down" ? (
        <div className="center">
          Couldn't reach the backend. If you're running locally, start it with
          <br />
          <code>uvicorn backend.app:app</code>.
        </div>
      ) : (
        <Onboarding
          movies={onboard}
          selected={selected}
          onToggle={toggleSeed}
          onSubmit={runRecommend}
          busy={busy}
        />
      )}

      <footer className="footer">
        Built on MovieLens + the TMDB API. This product uses the TMDB API but is not endorsed or
        certified by TMDB.
        <br />
        Retrieval runs on Proxima, a C++ HNSW engine.
      </footer>

      {error && <div className="toast">{error}</div>}
    </div>
  );
}
