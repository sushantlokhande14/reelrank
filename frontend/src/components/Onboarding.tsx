import type { Movie } from "../types";
import MovieCard from "./MovieCard";

interface Props {
  movies: Movie[];
  selected: number[];
  onToggle: (id: number) => void;
  onSubmit: () => void;
  busy?: boolean;
}

export default function Onboarding({ movies, selected, onToggle, onSubmit, busy }: Props) {
  const enough = selected.length >= 3;

  return (
    <section>
      <div className="section-head">
        <h2>New here? Pick a few you like</h2>
        <span className="muted">Choose 3 to 5 — we'll take it from there</span>
      </div>
      <div className="grid">
        {movies.map((m) => (
          <MovieCard
            key={m.id}
            movie={m}
            selectable
            selected={selected.includes(m.id)}
            onToggle={onToggle}
          />
        ))}
      </div>
      {selected.length > 0 && (
        <div className="cta-bar">
          <span className="label">
            {selected.length} selected{enough ? "" : ` — pick ${3 - selected.length} more`}
          </span>
          <button className="btn" onClick={onSubmit} disabled={!enough || busy}>
            {busy ? "Finding picks…" : "Show me what to watch"}
          </button>
        </div>
      )}
    </section>
  );
}
