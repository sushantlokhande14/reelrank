"""Metric correctness against hand-computed values."""

import math

import pytest

from reelrank.eval.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4, 5], {2, 4, 6}, 3) == pytest.approx(1 / 3)
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert recall_at_k([9, 8, 7], {1, 2}, 3) == 0.0
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # hits at rank 0 and rank 2
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # two relevant items, best case
    assert ndcg_at_k([2, 1, 3], {2, 3}, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([9], {1}, 3) == 0.0


def test_average_precision_at_k():
    # hits at rank 0 (prec 1/1) and rank 2 (prec 2/3), normalized by min(|truth|, k)=2
    assert average_precision_at_k([2, 1, 3], {2, 3}, 3) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert average_precision_at_k([9, 8], {1}, 2) == 0.0


def test_evaluate_rankings_aggregates_over_users():
    recs = {1: [10, 20, 30], 2: [40, 50, 60]}
    gt = {1: {20}, 2: {40}}
    out = evaluate_rankings(recs, gt, [3])
    assert out["n_users"] == 2.0
    assert out["recall@3"] == pytest.approx(1.0)
    ndcg_user1 = (1 / math.log2(3)) / 1.0  # single relevant item, idcg = 1
    ndcg_user2 = 1.0
    assert out["ndcg@3"] == pytest.approx((ndcg_user1 + ndcg_user2) / 2)
    assert out["map@3"] == pytest.approx((0.5 + 1.0) / 2)  # rank2 -> 1/2, rank1 -> 1/1
