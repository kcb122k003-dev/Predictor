"""Regression test on a fixed synthetic dataset.

The demo generator is seeded, so the whole pipeline is deterministic. A code change that shifts
parsing, mapping, model selection or ranking shows up here. If the change is intended, refresh
the snapshot with ``python tests/regression/snapshot_tools.py`` and review the diff.
"""

import json

import pytest

from tests.regression.snapshot_tools import SNAPSHOT, build_snapshot


@pytest.fixture(scope="module")
def current():
    return build_snapshot()


@pytest.fixture(scope="module")
def expected():
    if not SNAPSHOT.exists():
        pytest.skip("No snapshot yet: run python tests/regression/snapshot_tools.py")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_data_layer_unchanged(current, expected):
    for key in ("exams", "questions", "topics", "k", "status_counts"):
        assert current[key] == expected[key], key


def test_model_selection_and_metrics_stable(current, expected):
    assert current["selected_model"] == expected["selected_model"]
    assert current["calibrated"] == expected["calibrated"]
    for name, value in expected["model_ndcg"].items():
        assert name in current["model_ndcg"], name
        if value is not None:
            assert abs(current["model_ndcg"][name] - value) < 1e-6, name


def test_ranking_stable(current, expected):
    assert current["ranking"][:10] == expected["ranking"][:10]
    assert current["categories"] == expected["categories"]
