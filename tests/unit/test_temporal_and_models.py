import numpy as np
import pytest

from predictor.evaluation.backtest import BacktestEngine, leakage_audit, select_model
from predictor.features.builder import FEATURE_NAMES, FeatureStore, compute_features
from predictor.models.registry import build_models
from predictor.temporal import dynamics as dyn
from predictor.temporal.panel import ExamInfo, QuestionRecord, build_panel


def make_panel(Y: np.ndarray):
    T, K = Y.shape
    exams = [ExamInfo(t, 2000 + t, str(2000 + t), 80.0, 2000 + t) for t in range(T)]
    qs = []
    qid = 0
    for t in range(T):
        for k in range(K):
            if Y[t, k]:
                qid += 1
                qs.append(QuestionRecord(qid, t, f"q{qid}", 8.0, "theory", ["conceptual_explanation"],
                                         [(k, 1.0)], soft={k: 0.5}))
    return build_panel(exams, list(range(K)), [f"T{k}" for k in range(K)], qs)


def test_dynamics_by_hand():
    Y = np.array([[1, 0], [0, 0], [1, 0], [1, 1]], dtype=float)
    assert dyn.since_last(Y).tolist() == [1, 1]
    present, absent = dyn.streaks(Y)
    assert present.tolist() == [2, 1] and absent.tolist() == [0, 0]
    mean, std, n = dyn.gap_stats(Y)
    assert mean[0] == 1.5 and n[0] == 2 and np.isnan(mean[1])
    w = dyn.ewma(Y[:, 0], 1.0)  # weights 1/8, 1/4, 1/2, 1 normalised
    assert np.isclose(w, (0.125 + 0.5 + 1) / 1.875)
    n00, n01, n10, n11 = dyn.transition_counts(Y)
    assert (n00[0], n01[0], n10[0], n11[0]) == (0, 1, 1, 1)


def test_pooled_hazard_counts():
    # Item 0 alternates: after appearing, it never appears next exam; at gap 2 it always does.
    Y = np.array([[1], [0], [1], [0], [1], [0], [1]], dtype=float)
    haz = dyn.pooled_hazard(Y, prior_strength=0.0)
    assert haz.counts[1] == (0.0, 3.0)
    assert haz.counts[2] == (3.0, 3.0)


def test_rotation_detection():
    Y = np.zeros((15, 2))
    Y[::3, 0] = 1  # every third exam
    Y[[0, 1, 2, 7, 13], 1] = 1  # irregular
    res = {r.item: r for r in dyn.rotation_tests(Y, permutations=500)}
    assert res[0].significant and res[0].period == 3.0
    assert not res[1].significant


def test_features_use_only_history(settings):
    rng = np.random.default_rng(1)
    Y = (rng.random((10, 6)) < 0.4).astype(float)
    panel = make_panel(Y)
    fm = compute_features(panel.until(6), settings)
    assert fm.X.shape == (6, len(FEATURE_NAMES))
    Y2 = Y.copy()
    Y2[6:] = 1 - Y2[6:]
    fm2 = compute_features(make_panel(Y2).until(6), settings)
    assert np.allclose(fm.X, fm2.X)


@pytest.fixture
def planted_panel():
    rng = np.random.default_rng(7)
    T, K = 14, 16
    Y = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            p = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1][k % 8]
            if k in (8, 9):
                p = 0.9 if t % 2 == 0 else 0.05  # alternating
            Y[t, k] = rng.random() < p
    return make_panel(Y)


def test_backtest_beats_random_and_selects(settings, planted_panel):
    rep = BacktestEngine(planted_panel, settings).run()
    assert rep.targets == list(range(1, 14))  # rolling origin over every fold
    rnd = rep.models["random"].mean["ndcg"]
    assert rep.models[rep.selected].mean["ndcg"] > rnd + 0.15
    assert rep.leakage_audit["passed"], rep.leakage_audit
    assert rep.final_scores.shape == (16,)
    # Hidden tuned variants are not offered for selection.
    assert not rep.models[rep.selected].hidden


def test_small_data_keeps_every_component_running(settings):
    rng = np.random.default_rng(3)
    panel = make_panel((rng.random((5, 8)) < 0.5).astype(float))
    rep = BacktestEngine(panel, settings).run()
    for name in ("logistic", "random_forest", "gradient_boosting", "hmm", "hazard", "semantic", "beta_binomial"):
        assert rep.models[name].enabled and not rep.models[name].gate_reason
        assert rep.models[name].fold_metrics, name
    weights = rep.final_outputs["ensemble"].info["weights"]
    rel = rep.final_outputs["ensemble"].info["reliability"]
    # Course-learned models run but are downweighted by their reliability prior.
    assert rel["logistic"] < rel["beta_binomial"] and rel["random_forest"] < rel["logistic"]
    assert weights["random_forest"] < weights["beta_binomial"]
    tiny = make_panel((rng.random((3, 8)) < 0.5).astype(float))
    rep2 = BacktestEngine(tiny, settings).run()
    assert rep2.targets == [1, 2]
    assert not any("needs at least" in n.lower() for n in rep2.notes)


def test_one_se_rule_prefers_simpler(settings, planted_panel):
    rep = BacktestEngine(planted_panel, settings).run()
    name, best_name, _ = select_model(rep.models, "ndcg", "one_se", rep.targets)
    assert rep.models[name].complexity <= rep.models[best_name].complexity
    name, best_name, reason = select_model(rep.models, "ndcg", "best", rep.targets)
    assert name == best_name and "highest mean" in reason
    # The final model is the ensemble unless a single method is better by more than one standard error.
    if rep.selected != "ensemble":
        assert "more than one standard error" in rep.selection_reason


def test_leakage_audit_catches_a_leaky_model(settings, planted_panel):
    from predictor.models.base import BaseModel, ModelOutput

    class Leaky(BaseModel):
        name = "leaky"

        def predict(self, t, ctx):
            return ModelOutput(ctx.panel.Y[min(t, ctx.panel.T - 1)].copy())  # reads the target exam

    store = FeatureStore(planted_panel, settings)
    from predictor.models.base import ModelContext

    ctx = ModelContext(planted_panel, store, settings)
    ref = {"leaky": {5: Leaky().predict(5, ctx).scores}}
    audit = leakage_audit(planted_panel, settings, [Leaky()], 5, ref)
    assert not audit["passed"] and audit["failures"] == ["leaky"]


def test_calibration_reports_validity(settings, planted_panel):
    rep = BacktestEngine(planted_panel, settings).run()
    cal = rep.calibration
    assert cal is not None and cal.n_rows == 16 * len(rep.targets)
    if cal.valid:
        p, lo, hi = cal.probabilities(rep.final_scores)
        assert np.all((lo <= p + 1e-9) & (p <= hi + 1e-9))


def test_registry_respects_enabled(settings):
    over = settings.merged({"models": {"enabled": ["frequency", "ewma"]}})
    names = [m.name for m in build_models(over) if not m.hidden]
    assert names == ["frequency", "ewma"]
