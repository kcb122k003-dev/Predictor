"""Low-data behaviour of the inference engine at 2, 3, 5, 8-15 and more than 15 papers.

Properties checked (on synthetic panels with planted structure, so the tests are fast):

* no blanket cutoff: every component runs on every fold at every size;
* pretrained semantic and Bayesian components stay active at every size;
* course-specific learned models are downweighted when evidence is thin;
* richer models gain weight as papers accumulate;
* Bayesian posteriors reflect how much evidence exists (1/2 vs 8/16);
* question counts never inflate exam-level (temporal) evidence;
* no leakage from later papers, including through the ensemble;
* the user-facing message describes low-data inference, never "disabled";
* runs are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from predictor.evaluation.backtest import BacktestEngine
from predictor.inference.bayes import fit_recurrence
from predictor.inference.evidence import build_profile, component_status, low_data_message
from predictor.temporal.panel import ExamInfo, QuestionRecord, build_panel

COMPONENTS = ("recency", "beta_binomial", "semantic", "coverage", "question_type", "cooccurrence", "hazard", "markov",
              "hmm", "general", "logistic", "gradient_boosting", "random_forest")
LEARNED = ("logistic", "gradient_boosting", "random_forest")


def planted(T: int, K: int = 12, seed: int = 0, questions_per_topic: int = 1):
    """A course with stable, alternating, emerging and rare topics; units of three topics."""
    rng = np.random.default_rng(seed)
    kinds = ["core", "core", "regular", "alt", "emerging", "rare"] * (K // 6 + 1)
    kinds = [kinds[i] for i in rng.permutation(K)]
    Y = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            p = {"core": 0.85, "regular": 0.45, "alt": 0.9 if t % 2 == 0 else 0.08,
                 "emerging": 0.08 if t < T * 0.6 else 0.85, "rare": 0.1}[kinds[k]]
            Y[t, k] = rng.random() < p
        if not Y[t].any():
            Y[t, 0] = 1
    exams = [ExamInfo(t, 2000 + t, str(2000 + t), 80.0, 2000 + t) for t in range(T)]
    qs, qid = [], 0
    for t in range(T):
        for k in range(K):
            if Y[t, k]:
                for _ in range(questions_per_topic):
                    qid += 1
                    qs.append(QuestionRecord(qid, t, f"q{qid}", 8.0, "theory" if k % 2 else "numerical",
                                             ["conceptual_explanation"], [(k, 1.0)], soft={k: 0.6},
                                             semantic={k: 1.0}))
    panel = build_panel(exams, list(range(K)), [f"T{k}" for k in range(K)], qs,
                        item_unit=np.arange(K) // 3, unit_ids=list(range((K + 2) // 3)))
    panel.meta["semantic_source"] = "test"
    return panel


@pytest.fixture(scope="module")
def fast_settings():
    from predictor.config.settings import defaults

    return defaults().merged({"models": {"forest_trees": 15, "gbm_max_iter": 30, "hmm_iterations": 20}})


def _run(T, settings, **kw):
    return BacktestEngine(planted(T, **kw), settings).run()


@pytest.mark.parametrize("T", [2, 3, 5, 9, 15, 18])
def test_no_blanket_cutoff_every_component_runs(T, fast_settings):
    rep = _run(T, fast_settings)
    assert rep.targets == list(range(1, T))
    info = rep.final_outputs["ensemble"].info
    for name in COMPONENTS:
        r = rep.models[name]
        assert r.enabled and not r.gate_reason, name
        # Every component produced a forecast; structural unavailability is limited to co-occurrence
        # before two consecutive papers exist.
        assert name in info["weights"] or (name == "cooccurrence" and T < 3), (name, info["excluded"])
    assert abs(sum(info["weights"].values()) - 1.0) < 1e-3  # weights are reported to 4 decimals
    text = " ".join(rep.notes + [rep.selection_reason]).lower()
    assert "disabled" not in text and "needs at least" not in text


@pytest.mark.parametrize("T", [2, 3, 5])
def test_semantic_and_bayesian_active_at_small_sizes(T, fast_settings):
    rep = _run(T, fast_settings)
    w = rep.final_outputs["ensemble"].info["weights"]
    assert w["semantic"] > 0.02 and w["beta_binomial"] > 0.02 and w["general"] > 0.02
    profile = build_profile(rep.ctx.panel, questions=10, questions_total=10, questions_with_marks=10,
                            syllabus_weights="uniform", pretrained=True, pretrained_note="", repository_courses=0,
                            report=rep)
    rows = {r["model"]: r for r in component_status(rep, profile)}
    for name in ("semantic", "beta_binomial", "general"):
        assert rows[name]["status"] in ("ACTIVE", "LIMITED"), rows[name]


def test_course_models_downweighted_when_uncertain_and_grow_with_data(fast_settings):
    small = _run(3, fast_settings).final_outputs["ensemble"].info
    large = _run(25, fast_settings).final_outputs["ensemble"].info
    for name in LEARNED:
        assert small["reliability"][name] < 0.2 < large["reliability"][name]
    assert sum(small["weights"][n] for n in LEARNED) < small["weights"]["beta_binomial"] + small["weights"]["general"]
    assert sum(large["weights"][n] for n in LEARNED) > sum(small["weights"][n] for n in LEARNED)


def test_bayesian_posterior_reflects_amount_of_evidence():
    few = np.zeros((2, 4))
    few[0, 0] = 1
    few[:, 1] = 1
    many = np.zeros((16, 4))
    many[::2, 0] = 1
    many[:, 1] = 1
    p_few, p_many = fit_recurrence(few), fit_recurrence(many)
    width_few = p_few.high[0] - p_few.low[0]
    width_many = p_many.high[0] - p_many.low[0]
    assert width_few > 1.5 * width_many
    assert p_few.prior_contribution[0] > p_many.prior_contribution[0]
    assert p_few.summary(0)["observed"] == {"appearances": 1, "exams": 2}
    assert p_many.summary(0)["observed"] == {"appearances": 8, "exams": 16}
    assert p_few.ess[0] < p_many.ess[0]


def test_questions_do_not_inflate_temporal_evidence(fast_settings):
    one = BacktestEngine(planted(6, seed=3, questions_per_topic=1), fast_settings).run(audit_leakage=False)
    ten = BacktestEngine(planted(6, seed=3, questions_per_topic=10), fast_settings).run(audit_leakage=False)
    assert ten.ctx.panel.n_questions.sum() == 10 * one.ctx.panel.n_questions.sum()
    for name in ("beta_binomial", "hazard", "recency", "markov"):
        assert np.allclose(one.final_outputs[name].scores, ten.final_outputs[name].scores), name
    assert one.final_outputs["ensemble"].info["reliability"] == ten.final_outputs["ensemble"].info["reliability"]
    p1 = one.final_outputs["beta_binomial"].info["posterior"]
    p10 = ten.final_outputs["beta_binomial"].info["posterior"]
    assert p1.exams == p10.exams == 6 and np.allclose(p1.ess, p10.ess)


@pytest.mark.parametrize("T", [3, 5, 12])
def test_no_leakage_including_meta_models(T, fast_settings):
    rep = _run(T, fast_settings)
    audit = rep.leakage_audit
    assert audit["passed"], audit
    assert audit["models_checked"] >= len(COMPONENTS)  # base components plus tuned and ensemble models


def test_low_data_message_never_says_disabled(fast_settings):
    for T, expect in ((1, "Only 1 historical examination is available"),
                      (5, "Only 5 historical examinations are available")):
        rep = _run(T, fast_settings)
        profile = build_profile(rep.ctx.panel, questions=10, questions_total=10, questions_with_marks=10,
                                syllabus_weights="uniform", pretrained=True, pretrained_note="", repository_courses=0,
                                report=rep)
        component_status(rep, profile)
        mode, message = low_data_message(profile, rep)
        assert mode == "Low-data advanced inference"
        assert message.startswith(expect)
        assert "Advanced semantic and Bayesian inference remains active" in message
        assert "disabled" not in message.lower() and "off" not in message.lower().split()


def test_reproducible(fast_settings):
    a = _run(7, fast_settings, seed=11)
    b = _run(7, fast_settings, seed=11)
    assert a.selected == b.selected
    assert np.allclose(a.final_scores, b.final_scores)
    assert a.final_outputs["ensemble"].info["weights"] == b.final_outputs["ensemble"].info["weights"]


def test_zero_papers_still_ranks_from_prior_knowledge(fast_settings):
    panel = planted(1)
    empty = panel.until(0)
    rep = BacktestEngine(empty, fast_settings).run()
    assert rep.targets == [] and rep.selected == "ensemble"
    info = rep.final_outputs["ensemble"].info
    assert info["weights"].get("general", 0) > 0 or info["weights"].get("coverage", 0) > 0


def test_later_papers_never_change_earlier_predictions(fast_settings):
    """Changing papers t.. (including how many topics they contain, which moves the automatic K) must not
    change any model's prediction for paper t, meta models included."""
    base = planted(10, seed=4)
    other = planted(10, seed=4)
    other.Y[6:] = 1.0  # every topic in every later paper: the median topics per paper changes
    other.n_questions[6:] = 1.0
    a = BacktestEngine(base, fast_settings).run(audit_leakage=False)
    b = BacktestEngine(other, fast_settings).run(audit_leakage=False)
    assert a.k != b.k  # the reporting K differs, the guarantee must still hold
    for name, preds in a.predictions.items():
        for t in range(1, 7):
            if t in preds:
                assert np.allclose(preds[t], b.predictions[name][t]), (name, t)


def test_semantic_evidence_is_capped_per_paper():
    """Six questions on a topic in one paper must not outweigh one question in each of four papers."""
    exams = [ExamInfo(t, 2000 + t, str(2000 + t), 80.0, 2000 + t) for t in range(4)]
    qs, qid = [], 0
    for t in range(4):
        qid += 1
        qs.append(QuestionRecord(qid, t, "q", 5.0, "theory", [], [(1, 1.0)], semantic={1: 1.0}))
    for _ in range(6):
        qid += 1
        qs.append(QuestionRecord(qid, 0, "q", 5.0, "theory", [], [(0, 1.0)], semantic={0: 0.9, 2: 0.1}))
    panel = build_panel(exams, [0, 1, 2], ["a", "b", "c"], qs)
    sem = panel.semantic_matrix()
    assert sem.max() <= 1.0 + 1e-12
    assert sem[:, 1].sum() > sem[:, 0].sum()  # four papers beat one long paper


def test_input_quality_changes_component_reliability(fast_settings):
    clean = planted(8, seed=2)
    poor = planted(8, seed=2)
    poor.quality = poor.quality.copy()
    poor.quality[:, :, 0] *= 0.2  # weak syllabus mapping
    poor.quality[:, :, 2] = 0.0  # no marks anywhere
    for p in (clean, poor):
        p.meta["pretrained"] = True
    a = BacktestEngine(clean, fast_settings).run(audit_leakage=False).final_outputs["ensemble"].info
    b = BacktestEngine(poor, fast_settings).run(audit_leakage=False).final_outputs["ensemble"].info
    assert b["reliability"]["semantic"] < a["reliability"]["semantic"]
    assert b["reliability"]["logistic"] < a["reliability"]["logistic"]
    assert b["reliability"]["general"] == a["reliability"]["general"]  # outside knowledge is unaffected
    assert "mapping confidence 20%" in b["input_note"]["semantic"]


def _report(name: str, role: str, folds: dict[int, float]):
    from predictor.evaluation.backtest import ModelReport

    r = ModelReport(name=name, display=name, family="test", complexity=1, enabled=True, gate_reason="", hidden=False,
                    meta=True, role=role)
    r.fold_metrics = {t: {"ndcg": v} for t, v in folds.items()}
    r.mean = {"ndcg": float(np.mean(list(folds.values())))}
    return r


def test_selection_does_not_replace_the_ensemble_on_noise(fast_settings):
    """When the ensemble and the best-single selector are equally good, the ensemble is replaced at about the
    one-sided 5% rate of the t-bound, not whenever best_single happens to score higher."""
    from predictor.evaluation.backtest import select_final

    rng = np.random.default_rng(0)
    replaced = 0
    trials = 400
    for _ in range(trials):
        targets = list(range(1, 6))
        ens = dict(zip(targets, rng.normal(0.7, 0.05, 5)))
        best = dict(zip(targets, rng.normal(0.7, 0.05, 5)))
        reports = {"ensemble": _report("ensemble", "ensemble", ens), "best_single": _report("best_single", "meta", best)}
        replaced += select_final(reports, "ndcg", targets, fast_settings)[0] == "best_single"
    assert replaced / trials < 0.08
    # A consistent, large advantage does replace it.
    targets = list(range(1, 9))
    ens = {t: 0.60 + 0.01 * (t % 3) for t in targets}
    best = {t: 0.75 + 0.01 * (t % 2) for t in targets}
    reports = {"ensemble": _report("ensemble", "ensemble", ens), "best_single": _report("best_single", "meta", best)}
    choice, _, reason = select_final(reports, "ndcg", targets, fast_settings)
    assert choice == "best_single" and "beyond the one-sided 95% bound" in reason


def test_calibration_rejects_scores_unrelated_to_outcomes(fast_settings):
    """Scores that carry no information must almost never be presented as validated probabilities."""
    from predictor.models.calibration import calibrate

    settings = fast_settings.merged({"calibration": {"bootstrap_samples": 0}})
    rng = np.random.default_rng(1)
    valid = 0
    trials = 60
    for _ in range(trials):
        Y = (rng.random((10, 20)) < 0.4).astype(float)
        preds = {t: rng.random(20) for t in range(1, 10)}
        valid += calibrate(preds, Y, list(range(1, 10)), settings).valid
    assert valid / trials <= 0.05
    # Informative scores on enough papers do validate.
    Y = (rng.random((12, 20)) < 0.4).astype(float)
    preds = {t: Y[t] + rng.normal(0, 0.35, 20) for t in range(1, 12)}
    assert calibrate(preds, Y, list(range(1, 12)), settings).valid


def test_without_simulated_prior_the_general_model_is_reported_unavailable(fast_settings):
    """models.use_simulated_prior = false and no other real course: no outside knowledge exists, so the general
    component says so and the ensemble reweights the rest. Nothing else is switched off."""
    from predictor.inference.generic import flat_prior

    rep = BacktestEngine(planted(6, seed=8), fast_settings, general=flat_prior()).run(audit_leakage=False)
    info = rep.final_outputs["ensemble"].info
    assert "general" not in info["weights"] and "general" in info["excluded"]
    assert abs(sum(info["weights"].values()) - 1.0) < 1e-3
    assert info["weights"]["semantic"] > 0 and info["weights"]["beta_binomial"] > 0
    profile = build_profile(rep.ctx.panel, questions=10, questions_total=10, questions_with_marks=10,
                            syllabus_weights="uniform", pretrained=True, pretrained_note="", repository_courses=0,
                            report=rep)
    rows = {r["model"]: r for r in component_status(rep, profile)}
    assert rows["general"]["status"] == "UNAVAILABLE"
    assert rows["semantic"]["status"] in ("ACTIVE", "LIMITED")
