import type { Movie } from "../types";
import MovieCard from "./MovieCard";

interface Props {
  movies: Movie[];
  showReason?: boolean;
}

export default function ResultsGrid({ movies, showReason }: Props) {
  return (
    <div className="grid">
      {movies.map((m) => (
        <MovieCard key={`${m.source}-${m.id}`} movie={m} showReason={showReason} />
      ))}
    </div>
  );
}
