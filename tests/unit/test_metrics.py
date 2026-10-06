import math

import numpy as np

from predictor.evaluation.metrics import expected_random, mean_and_se, ranking_metrics, reliability


def test_ranking_metrics_by_hand():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    rel = np.array([1, 0, 1, 0, 0])
    m = ranking_metrics(scores, rel, k=2)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0
    # DCG@2 = 1/log2(2) = 1; IDCG@2 = 1 + 1/log2(3)
    assert math.isclose(m["ndcg"], 1 / (1 + 1 / math.log2(3)), rel_tol=1e-9)
    # AP = (1/1 + 2/3) / 2
    assert math.isclose(m["map"], (1 + 2 / 3) / 2)


def test_ties_do_not_get_lucky():
    # All scores equal: ordering falls back to item order, so the relevant last item ranks last.
    m = ranking_metrics(np.zeros(4), np.array([0, 0, 0, 1]), k=1)
    assert m["hit_rate"] == 0.0


def test_expected_random_matches_simulation():
    rng = np.random.default_rng(0)
    N, R, k = 12, 4, 3
    rel = np.zeros(N)
    rel[:R] = 1
    sims = [ranking_metrics(rng.random(N), rel, k) for _ in range(20000)]
    exp = expected_random(N, R, k)
    for key in ("precision", "recall", "hit_rate", "ndcg", "mrr"):
        assert abs(np.mean([s[key] for s in sims]) - exp[key]) < 0.01, key


def test_reliability_and_se():
    table, ece = reliability(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0, 0, 1, 1]), bins=2)
    assert len(table) == 2 and math.isclose(ece, 0.1)
    mean, se = mean_and_se([1.0, 2.0, 3.0])
    assert mean == 2.0 and math.isclose(se, 1 / math.sqrt(3))
