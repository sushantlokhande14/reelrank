export interface Movie {
  id: number;
  title: string;
  year?: string | null;
  genres: string[];
  poster_url?: string | null;
  source: string;
  overview?: string | null;
  reason?: string | null;
}

export interface Results {
  results: Movie[];
}
