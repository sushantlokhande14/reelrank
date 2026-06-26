"""The API app constructs and registers its routes (no engine load)."""

from backend.app import _to_movie, app


def test_routes_are_registered():
    paths = {route.path for route in app.routes}
    for expected in ("/health", "/warmup", "/onboarding", "/search", "/recommend"):
        assert expected in paths


def test_to_movie_maps_catalog_entry():
    movie = _to_movie(
        {"id": 3, "title": "Arrival", "genres": ["Sci-Fi"], "poster_url": "u", "source": "movielens"},
        reason="because",
    )
    assert movie.id == 3
    assert movie.title == "Arrival"
    assert movie.reason == "because"
