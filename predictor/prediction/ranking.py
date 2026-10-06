"""Final topic ranking with categories, confidence, evidence, contributions and why-not notes.

Spec sections 26, 29, 30, 48, 65 and 71. Percentages are shown as probabilities only when
calibration was validated on held-out exams; otherwise they are labelled relative scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.settings import Settings
from ..evaluation.backtest import BacktestReport
from ..features.builder import GROUP_LABELS, GROUP_OF, FeatureMatrix
from ..models.base import percentile_rank
from ..temporal import dynamics as dyn
from ..temporal.panel import FORMATS, Panel

CATEGORY_ORDER = ["Extremely High Priority", "High Priority", "Moderate Priority", "Low Priority", "Excluded"]
DISCLAIMER = ("Exam prediction is probabilistic. It ranks topics by historical evidence and cannot guarantee "
              "which questions will appear.")


@dataclass
class TopicPrediction:
    item_index: int
    item_id: int
    label: str
    rank: int
    score: float
    relative_score: float
    probability: float | None
    prob_low: float | None
    prob_high: float | None
    calibrated: bool
    category: str
    confidence: str
    signals_for: list[str]
    facts: dict[str, Any]
    evidence: list[str]
    contributions: dict[str, float]
    contribution_source: str
    why_not: list[str] = field(default_factory=list)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def supporting_signals(fm: FeatureMatrix, i: int, base_rate: float) -> list[str]:
    col = fm.column
    out = []
    if col("freq_all")[i] >= max(base_rate, 0.0) and col("freq_all")[i] > 0:
        out.append("frequency")
    if col("ewma_short")[i] >= 0.5:
        out.append("recency")
    hazard = col("hazard")[i]
    if (hazard > 0 and hazard >= base_rate) or col("due_z")[i] > 0.5:
        out.append("recurrence timing")
    soft = col("soft_ewma")
    if soft[i] > 0 and soft[i] >= np.median(soft):
        out.append("semantic recurrence")
    ms = col("marks_share")
    positive = ms[ms > 0]
    if ms[i] > 0 and positive.size and ms[i] >= np.median(positive):
        out.append("marks weight")
    return out


def build_topic_predictions(panel: Panel, report: BacktestReport, fm: FeatureMatrix, settings: Settings, *,
                            explain_output=None, mapping_confidence: np.ndarray | None = None,
                            rotation: dict[int, Any] | None = None, type_forecast: dict[int, Any] | None = None,
                            syllabus_location: dict[int, str] | None = None,
                            lab_items: set[int] | None = None) -> list[TopicPrediction]:
    cfg = settings.prediction
    scores = np.asarray(report.final_scores, dtype=float)
    K = len(scores)
    rel = percentile_rank(scores)
    order = np.lexsort((np.arange(K), -scores))
    ranks = np.empty(K, dtype=int)
    ranks[order] = np.arange(1, K + 1)
    calib = report.calibration
    probs = calib.probabilities(scores) if (calib is not None and calib.valid) else None
    T = panel.T
    base_rate = float(panel.Y.mean()) if panel.Y.size else 0.0
    m = report.k
    haz = dyn.pooled_hazard(panel.Y, float(settings.temporal.hazard_prior_strength)) if T else None
    contributions, contrib_source = _contributions(explain_output, report, K)

    preds: list[TopicPrediction] = []
    for i in range(K):
        signals = supporting_signals(fm, i, base_rate)
        if probs is not None:
            p, lo, hi = float(probs[0][i]), float(probs[1][i]), float(probs[2][i])
            if p >= float(cfg.very_high) and len(signals) >= int(cfg.min_signals_very_high):
                cat = CATEGORY_ORDER[0]
            elif p >= float(cfg.high):
                cat = CATEGORY_ORDER[1]
            elif p >= float(cfg.moderate):
                cat = CATEGORY_ORDER[2]
            else:
                cat = CATEGORY_ORDER[3]
        else:
            p = lo = hi = None
            r = ranks[i]
            top_share = max(1, math.ceil(m * float(cfg.very_high_rank_share)))
            if r <= top_share and len(signals) >= int(cfg.min_signals_very_high):
                cat = CATEGORY_ORDER[0]
            elif r <= m:
                cat = CATEGORY_ORDER[1]
            elif r <= 2 * m:
                cat = CATEGORY_ORDER[2]
            else:
                cat = CATEGORY_ORDER[3]
        facts = _facts(panel, fm, i, haz, mapping_confidence, rotation, type_forecast)
        facts["is_lab"] = bool(lab_items and panel.item_ids[i] in lab_items)
        conf = _confidence(T, settings, facts, p, lo, hi, probs is not None)
        evidence = _evidence_lines(panel, facts, syllabus_location.get(panel.item_ids[i]) if syllabus_location else None)
        preds.append(TopicPrediction(
            item_index=i, item_id=panel.item_ids[i], label=panel.item_labels[i], rank=int(ranks[i]),
            score=round(float(scores[i]), 5), relative_score=round(float(rel[i]), 4),
            probability=None if p is None else round(p, 4), prob_low=None if lo is None else round(lo, 4),
            prob_high=None if hi is None else round(hi, 4), calibrated=probs is not None, category=cat,
            confidence=conf, signals_for=signals, facts=facts, evidence=evidence,
            contributions=contributions[i] if contributions else {}, contribution_source=contrib_source))
    preds.sort(key=lambda p: p.rank)
    n_why = int(cfg.why_not_count)
    for pred in preds:
        if pred.category in (CATEGORY_ORDER[2], CATEGORY_ORDER[3]) or pred.rank > m:
            pred.why_not = _why_not(pred, panel, haz, base_rate, mapping_confidence)
    # Keep why-not notes for the topics just below the cut and the lowest ones (most informative).
    flagged = [p for p in preds if p.why_not]
    keep = set(id(p) for p in flagged[:n_why]) | set(id(p) for p in flagged[-n_why // 2:])
    for p in flagged:
        if id(p) not in keep:
            p.why_not = p.why_not[:1]
    return preds


def _contributions(explain_output, report: BacktestReport, K: int) -> tuple[list[dict[str, float]] | None, str]:
    if explain_output is None or explain_output.contributions is None:
        return None, ""
    names = explain_output.contribution_names or []
    out = []
    for i in range(K):
        groups: dict[str, float] = {}
        for j, name in enumerate(names):
            g = GROUP_LABELS[GROUP_OF[name]]
            groups[g] = groups.get(g, 0.0) + float(explain_output.contributions[i, j])
        out.append({g: round(v, 4) for g, v in groups.items()})
    source = ("logistic regression (the selected model)" if report.selected == "logistic"
              else f"the logistic explanation model (the selected model is {report.models[report.selected].display})")
    return out, source


def _facts(panel: Panel, fm: FeatureMatrix, i: int, haz, mapping_confidence, rotation, type_forecast) -> dict[str, Any]:
    T = panel.T
    y = panel.Y[:, i] if T else np.zeros(0)
    idx = np.flatnonzero(y)
    window = min(6, T)
    facts: dict[str, Any] = {
        "exams": T, "appearances": int(y.sum()), "recent_window": window,
        "recent_appearances": int(y[-window:].sum()) if T else 0,
        "last_label": panel.exams[idx[-1]].label if idx.size else None,
        "exams_since_last": int(T - idx[-1]) if idx.size else None,
        "streak_present": int(fm.column("streak_present")[i] * 6 + 0.5),
        "streak_absent": int(fm.column("streak_absent")[i] * 10 + 0.5),
        "gaps": np.diff(idx).astype(int).tolist() if idx.size >= 2 else [],
    }
    if idx.size:
        marks = panel.marks[idx, i]
        known = marks[marks > 0]
        if known.size:
            facts["marks_mean"] = round(float(known.mean()), 1)
            facts["marks_min"] = round(float(known.min()), 1)
            facts["marks_max"] = round(float(known.max()), 1)
        share = panel.marks[idx, i] / np.array([max(panel.exams[j].total_marks, 1e-9) for j in idx])
        facts["major_question_rate"] = round(float((share >= 0.1).mean()), 3)
        fmt = panel.formats[:, i, :].sum(axis=0)
        if fmt.sum():
            facts["format_counts"] = {FORMATS[j]: int(c) for j, c in enumerate(fmt) if c}
            last_t = idx[-1]
            last_fmt = panel.formats[last_t, i, :]
            if last_fmt.sum():
                facts["last_format"] = FORMATS[int(np.argmax(last_fmt))]
        facts["exact_repeats"] = int(panel.exact_repeat[:, i].sum())
        facts["paraphrase_repeats"] = int(panel.para_repeat[:, i].sum())
    if haz is not None and idx.size:
        gap = T - idx[-1]  # gap if it appears in the next exam (1 = it was in the latest exam)
        g = int(min(gap, haz.max_gap))
        if g in haz.counts:
            hits, opps = haz.counts[g]
            facts["hazard_gap"] = gap
            facts["hazard_rate"] = round(float(hits / opps), 3) if opps else None
            facts["hazard_opportunities"] = int(opps)
        facts["base_rate"] = round(haz.base_rate, 3)
    if mapping_confidence is not None:
        facts["mapping_confidence"] = round(float(mapping_confidence[i]), 3) if mapping_confidence[i] > 0 else None
    if rotation and i in rotation:
        facts["rotation"] = rotation[i]
    if type_forecast and panel.item_ids[i] in type_forecast:
        facts["type_forecast"] = type_forecast[panel.item_ids[i]]
    return facts


def _confidence(T: int, settings: Settings, facts: dict[str, Any], p, lo, hi, calibrated: bool) -> str:
    low_exams = int(settings.models.sufficiency.low_confidence_exams)
    if T < low_exams or facts.get("appearances", 0) == 0:
        return "Low"
    mc = facts.get("mapping_confidence")
    if mc is not None and mc < 0.55:
        return "Low"
    if calibrated and hi is not None and lo is not None:
        width = hi - lo
        if width <= 0.2 and T >= 10 and (mc is None or mc >= 0.7):
            return "High"
        if width > 0.35:
            return "Low"
        return "Medium"
    return "Medium" if T >= 10 else "Low"


def _evidence_lines(panel: Panel, f: dict[str, Any], location: str | None) -> list[str]:
    lines = []
    T = f["exams"]
    if T == 0:
        return ["No exams have been analysed yet."]
    if f["appearances"] == 0:
        lines.append(f"Never appeared in the {T} supplied exams.")
    else:
        lines.append(f"Appeared in {f['recent_appearances']} of the last {f['recent_window']} exams and "
                     f"{f['appearances']} of {T} overall.")
        since = f.get("exams_since_last")
        if since == 1:
            lines.append(f"Tested in the most recent exam ({f['last_label']}).")
        elif since:
            lines.append(f"Last appeared in {f['last_label']}, {since} exams ago.")
    if f.get("streak_present", 0) >= 2:
        lines.append(f"Tested in each of the last {f['streak_present']} exams.")
    if f.get("streak_absent", 0) >= 2 and f["appearances"]:
        lines.append(f"Not tested in the last {f['streak_absent']} exams.")
    gaps = f.get("gaps") or []
    if len(gaps) >= 2:
        lines.append(f"Gaps between appearances: {', '.join(map(str, gaps))} exams (mean {np.mean(gaps):.1f}).")
    rot = f.get("rotation")
    if rot and rot.get("significant"):
        lines.append(f"Regular rotation detected: about every {rot['period']:.1f} exams (permutation test "
                     f"p = {rot['p_value']:.3f}).")
    if f.get("hazard_rate") is not None and f.get("hazard_opportunities", 0) >= 5:
        g = f["hazard_gap"]
        when = "after appearing in the previous exam" if g == 1 else f"{g} exams after their last appearance"
        lines.append(f"In this course, topics reappeared {when} {_fmt_pct(f['hazard_rate'])} of the time "
                     f"({f['hazard_opportunities']} cases; average appearance rate {_fmt_pct(f['base_rate'])}).")
    if f.get("marks_mean") is not None:
        rng = f"{f['marks_min']:g}-{f['marks_max']:g}" if f['marks_min'] != f['marks_max'] else f"{f['marks_mean']:g}"
        lines.append(f"Typical marks when tested: {rng} (mean {f['marks_mean']:g}).")
    if f.get("major_question_rate") is not None and f["appearances"]:
        lines.append(f"Tested as a major question (10% or more of the paper) in {_fmt_pct(f['major_question_rate'])} "
                     f"of appearances.")
    counts = f.get("format_counts")
    if counts:
        top = max(counts, key=counts.get)
        line = f"Most common question format: {top} ({counts[top]} of {sum(counts.values())})"
        if f.get("last_format") and f["last_format"] != top:
            line += f"; last time it was {f['last_format']}"
        lines.append(line + ".")
    tf = f.get("type_forecast")
    if tf and tf.get("format"):
        source = {"global": "the course-wide format mix", "topic": "this topic's own recent formats",
                  "transition": "this topic's recent formats and format transitions"}.get(tf.get("method"), "")
        lines.append(f"Predicted format if it appears: {tf['format']} ({_fmt_pct(tf['probability'])}"
                     + (f", from {source}, which forecast formats best in the backtest" if source else "") + ").")
    if f.get("exact_repeats"):
        lines.append(f"{f['exact_repeats']} question(s) on this topic repeated an earlier question almost word for word.")
    if f.get("mapping_confidence") is not None:
        lines.append(f"Average syllabus match of its questions: {_fmt_pct(f['mapping_confidence'])}.")
    if location:
        lines.append(f"Syllabus location: {location}.")
    return lines


def _why_not(pred: TopicPrediction, panel: Panel, haz, base_rate: float, mapping_confidence) -> list[str]:
    f = pred.facts
    out = []
    T = f["exams"]
    if T == 0:
        return out
    if f["appearances"] == 0 and f.get("is_lab"):
        out.append(f"Laboratory or practical item: none of the {T} written papers asked about it.")
    elif f["appearances"] == 0:
        out.append(f"No history: never appeared in the {T} supplied exams.")
    elif f["appearances"] <= max(1, round(0.2 * T)):
        out.append(f"Rare historically: appeared in only {f['appearances']} of {T} exams.")
    if f.get("exams_since_last") == 1 and haz is not None and 1 in haz.counts:
        hits, opps = haz.counts[1]
        rate = hits / opps if opps else 0.0
        if opps >= 5 and rate < base_rate:
            out.append(f"Recently repeated: it was in the last exam, and in this course topics reappear in the very "
                       f"next exam only {_fmt_pct(rate)} of the time (average {_fmt_pct(base_rate)}).")
    if f.get("streak_absent", 0) >= 3 and f["appearances"]:
        out.append(f"Absent from the last {f['streak_absent']} exams.")
    mc = f.get("mapping_confidence")
    if mc is not None and mc < 0.55:
        out.append(f"Weak semantic evidence: its historical questions match the syllabus at only {_fmt_pct(mc)}.")
    if 0 < f["appearances"] < 2:
        out.append("Insufficient evidence: one appearance cannot establish a pattern.")
    if not out:
        out.append("Other topics have stronger combined evidence; the gap to the cut-off is small.")
    return out


def excluded_entries(oos_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Out-of-syllabus question groups, shown in the Excluded category with their reason."""
    return [{"label": c["label"], "category": CATEGORY_ORDER[4], "reason": c["reason"],
             "questions": c.get("questions", []), "years": c.get("years", [])} for c in oos_clusters]
