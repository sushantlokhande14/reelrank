"""Temporal split, k-core filtering, and the eval ground-truth construction."""

import pandas as pd

from reelrank.baselines.popularity import PopularityRecommender
from reelrank.data.movielens import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    MovieLensData,
    _global_temporal_split,
    _kcore_filter,
    _user_temporal_split,
)
from reelrank.eval.harness import build_ground_truth


def test_temporal_split_has_no_future_leak():
    df = pd.DataFrame(
        {
            "userId": [1] * 10,
            "movieId": list(range(10)),
            "rating": [5.0] * 10,
            "timestamp": list(range(100, 110)),
        }
    )
    out = _global_temporal_split(df, val_frac=0.2, test_frac=0.2)
    assert (out["split"] == "train").sum() == 6
    assert (out["split"] == "val").sum() == 2
    assert (out["split"] == "test").sum() == 2
    # every training interaction is strictly older than every test interaction
    assert out.loc[out["split"] == "train", "timestamp"].max() < out.loc[
        out["split"] == "test", "timestamp"
    ].min()


def test_user_temporal_split_holds_out_each_users_recent_items():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "movieId": list(range(10)),
            "rating": [5.0] * 10,
            "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        }
    )
    out = _user_temporal_split(df, val_frac=0.2, test_frac=0.2)
    for user in (1, 2):
        g = out[out["userId"] == user]
        # the single most recent interaction per user is held out for test
        assert (g["split"] == SPLIT_TEST).sum() == 1
        assert (g["split"] == SPLIT_VAL).sum() == 1
        test_ts = g.loc[g["split"] == SPLIT_TEST, "timestamp"].min()
        train_ts = g.loc[g["split"] == SPLIT_TRAIN, "timestamp"].max()
        assert train_ts < test_ts


def test_kcore_filter_drops_sparse_users():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],  # user 2 has a single interaction
            "movieId": [10, 11, 12, 10],
            "rating": [5.0] * 4,
            "timestamp": [1, 2, 3, 4],
        }
    )
    out = _kcore_filter(df, min_user=2, min_item=1)
    assert set(out["userId"].unique()) == {1}
    assert len(out) == 3


def _toy_data() -> MovieLensData:
    rows = [
        (0, 0, 5.0, 1, SPLIT_TRAIN),
        (0, 1, 5.0, 2, SPLIT_TRAIN),
        (0, 2, 5.0, 3, SPLIT_TEST),  # genuinely new positive
        (0, 0, 5.0, 4, SPLIT_TEST),  # already seen in train -> excluded from GT
        (1, 0, 5.0, 1, SPLIT_TRAIN),
        (1, 3, 5.0, 3, SPLIT_TEST),
    ]
    ratings = pd.DataFrame(
        rows, columns=["user_idx", "item_idx", "rating", "timestamp", "split"]
    )
    movies = pd.DataFrame(
        {
            "item_idx": [0, 1, 2, 3],
            "movieId": [100, 101, 102, 103],
            "title": list("abcd"),
            "genres": ["x"] * 4,
        }
    )
    return MovieLensData(ratings, movies, n_users=2, n_items=4, user_id_map={}, item_id_map={})


def test_build_ground_truth_excludes_seen_items():
    gt, train_sets = build_ground_truth(_toy_data())
    assert gt[0] == {2}  # item 0 was already seen in training, item 2 is new
    assert gt[1] == {3}
    assert train_sets[0] == {0, 1}


def test_popularity_ranks_by_count_and_skips_seen():
    pop = PopularityRecommender().fit(_toy_data())
    assert pop.ranked_items[0] == 0  # item 0 has the most training interactions
    recs = pop.recommend_all([1], k=2, exclude={1: {0}})
    assert 0 not in recs[1].tolist()
