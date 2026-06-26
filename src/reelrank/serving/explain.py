"""Short "why we picked this" explanations.

The default path is a deterministic template built from the movie's genres and
overview, so the API always returns a reason with no external dependency. When the
Relay LLM gateway is configured (RELAY_BASE_URL and RELAY_API_KEY), explanations
can be routed through it to reuse its caching and failover. Relay is optional and
any failure falls back to the template.
"""

from __future__ import annotations

import os

import requests


def template_reason(query: str | None, item: dict) -> str:
    """A genre/plot-based reason that needs no LLM."""
    genres = [g for g in item.get("genres", []) if g][:2]
    genre_phrase = " and ".join(genres).lower() if genres else "its overall feel"
    if query:
        return f"Close to your request, with {genre_phrase} at its core."
    overview = (item.get("overview") or "").strip()
    if overview:
        snippet = overview.split(". ")[0]
        return f"{snippet}." if not snippet.endswith(".") else snippet
    return f"A strong {genre_phrase} match based on what you like."


def relay_available() -> bool:
    return bool(os.environ.get("RELAY_BASE_URL") and os.environ.get("RELAY_API_KEY"))


def relay_reason(query: str | None, item: dict, timeout: float = 8.0) -> str | None:
    """Ask Relay (OpenAI-compatible gateway) for a one-line reason. None on failure."""
    if not relay_available():
        return None
    base = os.environ["RELAY_BASE_URL"].rstrip("/")
    title = item.get("title", "this movie")
    genres = ", ".join(item.get("genres", []))
    ask = (
        f"In one sentence, tell a user why '{title}' ({genres}) fits this request: "
        f"{query!r}." if query else
        f"In one sentence, tell a user why they might enjoy '{title}' ({genres})."
    )
    try:
        resp = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['RELAY_API_KEY']}"},
            json={
                "model": os.environ.get("RELAY_MODEL", "claude-haiku-4-5-20251001"),
                "messages": [{"role": "user", "content": ask}],
                "max_tokens": 60,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001 - explanations must never break a recommendation
        return None


def reason_for(query: str | None, item: dict, use_relay: bool = True) -> str:
    if use_relay:
        via_relay = relay_reason(query, item)
        if via_relay:
            return via_relay
    return template_reason(query, item)
