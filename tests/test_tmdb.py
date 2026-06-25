"""TMDB content-text formatting and live-catalog de-duplication (no network)."""

from reelrank.tmdb.catalog import fetch_live_catalog
from reelrank.tmdb.client import MovieMeta, tmdb_content_text


def test_tmdb_content_text_includes_all_fields():
    meta = MovieMeta(
        tmdb_id=1,
        title="Arrival",
        year="2016",
        genres=["Science Fiction", "Drama"],
        overview="A linguist works to communicate with aliens.",
        director="Denis Villeneuve",
        cast=["Amy Adams", "Jeremy Renner"],
    )
    text = tmdb_content_text(meta)
    assert "Arrival (2016)" in text
    assert "Science Fiction" in text
    assert "linguist" in text
    assert "Amy Adams" in text
    assert "Denis Villeneuve" in text


class _FakeClient:
    def trending(self, page=1):
        return [{"id": 1}, {"id": 2}] if page == 1 else [{"id": 3}]

    def now_playing(self, page=1):
        return [{"id": 2}, {"id": 4}] if page == 1 else []

    def fetch_meta(self, tmdb_id):
        return MovieMeta(
            tmdb_id=tmdb_id, title=f"M{tmdb_id}", year="2026", genres=["Drama"],
            overview="o", director="D", cast=["A"],
        )


def test_fetch_live_catalog_dedupes_across_lists_and_pages():
    metas = fetch_live_catalog(_FakeClient(), pages=2)
    assert sorted(m.tmdb_id for m in metas) == [1, 2, 3, 4]
