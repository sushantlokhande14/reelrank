import { useState } from "react";

import type { Movie } from "../types";

interface Props {
  movie: Movie;
  selectable?: boolean;
  selected?: boolean;
  onToggle?: (id: number) => void;
  showReason?: boolean;
}

export default function MovieCard({ movie, selectable, selected, onToggle, showReason }: Props) {
  const [broken, setBroken] = useState(false);
  const hasPoster = movie.poster_url && !broken;
  const live = movie.source === "tmdb_live";

  return (
    <div
      className={`card${selectable ? " selectable" : ""}${selected ? " selected" : ""}`}
      onClick={selectable && onToggle ? () => onToggle(movie.id) : undefined}
    >
      {live && <span className="badge">In theaters</span>}
      {selected && <span className="check">✓</span>}
      {hasPoster ? (
        <img
          className="poster"
          src={movie.poster_url ?? ""}
          alt={movie.title}
          loading="lazy"
          onError={() => setBroken(true)}
        />
      ) : (
        <div className="poster-fallback">{movie.title}</div>
      )}
      <div className="card-body">
        <div className="card-title">{movie.title}</div>
        <div className="card-sub">
          {[movie.year, movie.genres.slice(0, 2).join(" · ")].filter(Boolean).join("  ·  ")}
        </div>
        {showReason && movie.reason && <div className="reason">{movie.reason}</div>}
      </div>
    </div>
  );
}
