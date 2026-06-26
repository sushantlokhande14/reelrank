import type { Results } from "./types";

const BASE = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: (): Promise<boolean> =>
    fetch(`${BASE}/health`).then((r) => r.ok).catch(() => false),
  warmup: (): Promise<{ ready: boolean; catalog: number }> => postJson("/warmup", {}),
  onboarding: (n = 18): Promise<Results> =>
    fetch(`${BASE}/onboarding?n=${n}`).then((r) => r.json() as Promise<Results>),
  search: (query: string, k = 18): Promise<Results> => postJson("/search", { query, k }),
  recommend: (seedIds: number[], k = 18): Promise<Results> =>
    postJson("/recommend", { seed_ids: seedIds, k }),
};
