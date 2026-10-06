import numpy as np
import pytest

from predictor.analysis.coverage import benjamini_hochberg, cooccurrence
from predictor.analysis.structure import ExamSummary, LeafInfo, MainQuestion, discover_structure
from predictor.config.settings import SettingsError, defaults, deep_merge
from predictor.generation.questions import (HistoricalQuestion, check_grounding, display_phrase,
                                            generate_formulations)
from predictor.prediction.type_forecast import forecast, run_type_forecast
from predictor.syllabus.tree import TopicNode, TopicTree
from predictor.temporal.panel import FORMATS, ExamInfo, QuestionRecord, build_panel


def _tree():
    return TopicTree([
        TopicNode(1, None, 1, "Fluid Dynamics", "4"),
        TopicNode(2, 1, 2, "Euler's equation of motion and Bernoulli's equation", "4.1",
                  concepts=["Bernoulli's equation", "Euler's equation of motion", "applications"], kinds=["derivation"]),
        TopicNode(3, 1, 2, "Flow measurement", "4.2", concepts=["venturimeter", "orifice meter", "pitot tube"]),
    ])


def _hist():
    return [
        HistoricalQuestion(1, "Derive Bernoulli's equation from Euler's equation of motion. State the assumptions made.",
                           "derivation", ["derivation"], 8, "2019", 5),
        HistoricalQuestion(2, "Water flows through a pipe of 20 cm diameter at 300 kPa. Calculate the pressure at "
                              "section 2 using Bernoulli's equation.", "numerical", ["numerical"], 10, "2021", 7),
    ]


def test_grounding_rejects_unsupported_terms(settings):
    allowed = {"bernoulli", "equat", "euler"}
    ok = check_grounding("Derive Bernoulli's equation.", allowed, settings.alignment.instruction_words)
    bad = check_grounding("Derive the Navier-Stokes equation.", allowed, settings.alignment.instruction_words)
    assert ok["grounded"] and not bad["grounded"] and "navier" in bad["unsupported_terms"]


def test_formulations_are_grounded_and_reuse_numbers(settings):
    forms = generate_formulations(_tree(), 2, _hist(), _hist(), {"derivation": 0.5, "numerical": 0.3, "theory": 0.2},
                                  settings)
    by_format = {f.format: f for f in forms}
    assert by_format["derivation"].text.startswith("Derive")
    assert "Bernoulli's equation" in by_format["derivation"].text
    num = by_format["numerical"]
    assert num.basis == "historical_variant" and num.text == _hist()[1].text and "no new values" in num.note
    assert all(f.grounding["grounded"] for f in forms)
    # "applications" is an aspect, not something to ask about.
    assert not any(f.text.lower().endswith("explain applications.") for f in forms)


def test_no_numerical_without_history(settings):
    forms = generate_formulations(_tree(), 3, [], _hist(), {"numerical": 0.9, "theory": 0.1}, settings)
    assert forms and all(f.format != "numerical" for f in forms)


def test_display_phrase_keeps_names():
    assert display_phrase("Continuity equation", set()) == "continuity equation"
    assert display_phrase("Bernoulli's equation", set()) == "Bernoulli's equation"
    assert display_phrase("Reynolds number", {"reynolds"}) == "Reynolds number"


def test_structure_patterns():
    exams = []
    for year in range(2015, 2021):
        mains = [MainQuestion(str(i), 16, 2, None, False, None) for i in range(1, 6)]
        leaves = [LeafInfo("theory", 8, 0, 0)] * 10
        exams.append(ExamSummary(str(year), year, 80, mains, leaves))
    report = discover_structure(exams)
    pattern = {p["property"]: p for p in report.patterns}
    assert pattern["main_questions"]["typical"] == 5
    assert pattern["main_questions"]["level"] == "high-confidence historical pattern"
    assert all(h.startswith("Speculative") for h in report.hypotheses)


def test_benjamini_hochberg_monotone():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q[0] <= q[2] <= q[1] <= q[3] <= 1.0


def _panel(Y, formats=None):
    T, K = Y.shape
    qs, qid = [], 0
    for t in range(T):
        for k in range(K):
            if Y[t, k]:
                qid += 1
                fmt = formats[t][k] if formats else "theory"
                qs.append(QuestionRecord(qid, t, "q", 5.0, fmt, [fmt], [(k, 1.0)]))
    return build_panel([ExamInfo(t, t, str(t), 50.0) for t in range(T)], list(range(K)), [str(k) for k in range(K)], qs)


def test_cooccurrence_finds_strong_pair():
    rng = np.random.default_rng(0)
    T = 30
    a = rng.random(T) < 0.5
    Y = np.column_stack([a, a, rng.random(T) < 0.5, rng.random(T) < 0.5]).astype(float)
    res = cooccurrence(_panel(Y))
    assert res["available"]
    top = res["significant"][0]
    assert {top["a"], top["b"]} == {0, 1} and top["lift"] > 1.5


def test_type_forecast_learns_topic_format(settings):
    T, K = 10, 3
    Y = np.ones((T, K))
    formats = [["numerical", "derivation", "definition"] for _ in range(T)]
    panel = _panel(Y, formats)
    pred = forecast(panel.formats, "topic", 2.0)
    assert [FORMATS[int(np.argmax(pred[k]))] for k in range(K)] == ["numerical", "derivation", "definition"]
    report = run_type_forecast(panel, settings, list(range(3, T)))
    assert report.method in ("topic", "transition")
    assert report.accuracy["topic"]["mean"] == 1.0


def test_settings_validation():
    base = defaults().as_dict()
    with pytest.raises(SettingsError):
        deep_merge(base, {"ocr": {"dpi": "high"}})
    with pytest.raises(SettingsError):
        deep_merge(base, {"nonexistent": 1})
    merged = deep_merge(base, {"models": {"top_k": 8}})
    assert merged["models"]["top_k"] == 8
