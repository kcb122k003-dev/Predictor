"""Evidence profile of a course and the status of every inference component.

Reliability is not a single exam count. The profile records the things that decide how much
each kind of evidence can be trusted:

* exam-level evidence (temporal statistics): number of papers, papers without a year,
  calendar span and gaps, topics per paper, how many topics were ever observed;
* question-level evidence (language and semantics only): in-syllabus questions, questions
  with marks, question formats. Question counts never enter temporal reliability: a paper
  with forty questions is still one paper;
* validation evidence: held-out folds, fold-to-fold spread, calibration result;
* feature availability: pretrained model, syllabus units and weights, marks, formats,
  other real courses in the repository.

Component statuses (shown in the app):

  ACTIVE        running, inputs available, and its reliability prior is at least one half
  LIMITED       running and contributing, but with high uncertainty (little course evidence)
  DOWNWEIGHTED  running, but earlier papers showed it ranks worse than the other components
  UNAVAILABLE   an input it needs does not exist for this course (the reason is given)
  REFERENCE     a baseline kept for comparison, not part of the final ranking

The app never switches inference off because a course is small.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

ACTIVE, LIMITED, DOWNWEIGHTED, UNAVAILABLE, REFERENCE = "ACTIVE", "LIMITED", "DOWNWEIGHTED", "UNAVAILABLE", "REFERENCE"
# Components whose ranking parameters are learned from this course's own rows (the low-data message
# refers to these).
COURSE_LEARNED = ("logistic", "gradient_boosting", "random_forest")


@dataclass
class EvidenceProfile:
    exams: int
    effective_exams: float
    exams_without_year: int
    year_span: list[int] | None
    calendar_gaps: int
    questions: int            # in-syllabus leaf questions that count (question level)
    questions_total: int
    questions_with_marks: int
    topics: int
    topics_observed: int
    units: int
    mean_topics_per_exam: float
    coverage_entropy: float   # 0 = always the same topic, 1 = appearances spread evenly
    formats_observed: int
    syllabus_weights: str
    pretrained: bool
    pretrained_note: str = ""
    repository_courses: int = 0
    folds: int = 0
    fold_sd: float | None = None
    ci_width: float | None = None
    calibration_valid: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
            elif isinstance(v, float):
                d[k] = round(v, 4)
        return d


def build_profile(panel, *, questions: int, questions_total: int, questions_with_marks: int,
                  syllabus_weights: str, pretrained: bool, pretrained_note: str, repository_courses: int,
                  report=None) -> EvidenceProfile:
    T, K = panel.Y.shape
    years = [e.year for e in panel.exams if e.year is not None]
    span = [min(years), max(years)] if years else None
    gaps = 0
    if len(years) >= 2:
        ys = sorted(set(years))
        gaps = int(sum(max(b - a - 1, 0) for a, b in zip(ys, ys[1:])))
    counts = panel.Y.sum(axis=0) if T else np.zeros(K)
    total = counts.sum()
    if total > 0 and K > 1:
        p = counts[counts > 0] / total
        entropy = float(-(p * np.log(p)).sum() / math.log(K))
    else:
        entropy = 0.0
    formats = int((panel.formats.sum(axis=(0, 1)) > 0).sum()) if T else 0
    units = len(set(np.asarray(panel.item_unit).tolist())) if panel.item_unit is not None and K else 0
    prof = EvidenceProfile(
        exams=T, effective_exams=float(T - 0.5 * sum(1 for e in panel.exams if e.year is None)),
        exams_without_year=sum(1 for e in panel.exams if e.year is None), year_span=span, calendar_gaps=gaps,
        questions=int(questions), questions_total=int(questions_total), questions_with_marks=int(questions_with_marks),
        topics=K, topics_observed=int((counts > 0).sum()), units=units,
        mean_topics_per_exam=float(panel.Y.sum(axis=1).mean()) if T else 0.0, coverage_entropy=entropy,
        formats_observed=formats, syllabus_weights=syllabus_weights, pretrained=pretrained,
        pretrained_note=pretrained_note, repository_courses=repository_courses)
    if report is not None:
        v = report.validation or {}
        prof.folds = len(report.targets)
        prof.fold_sd = v.get("sd")
        prof.ci_width = v.get("ci_width")
        prof.calibration_valid = bool(report.calibration and report.calibration.valid)
    if prof.exams_without_year:
        prof.notes.append(f"{prof.exams_without_year} paper(s) have no year, so their order is uncertain; each counts "
                          f"as half a paper of temporal evidence.")
    if gaps:
        prof.notes.append(f"The papers span {span[0]}-{span[1]} with {gaps} calendar year(s) missing; gaps are counted "
                          f"in papers, not years.")
    if questions_total and questions_with_marks / max(questions_total, 1) < 0.5:
        prof.notes.append("Fewer than half of the questions have marks, so marks-based signals carry little weight.")
    return prof


def component_status(report, profile: EvidenceProfile) -> list[dict[str, Any]]:
    """Fill status, weight and reliability on every visible model report; return the status table."""
    ens = report.final_outputs.get("ensemble")
    info = ens.info if ens is not None else {}
    weights, rel, skill = info.get("weights", {}), info.get("reliability", {}), info.get("skill", {})
    excluded = info.get("excluded", {})
    members = set(weights) | set(excluded)
    n_members = max(len(weights), 1)
    rows = []
    for r in report.models.values():
        if r.hidden:
            continue
        if r.name in members:
            r.weight = float(weights.get(r.name, 0.0))
            r.reliability = rel.get(r.name)
            r.skill = skill.get(r.name)
            if r.name in excluded:
                r.status, r.status_reason = UNAVAILABLE, excluded[r.name]
                r.weight = 0.0
            elif r.skill is not None and r.skill < -0.25 and r.weight < 0.5 / n_members:
                r.status = DOWNWEIGHTED
                r.status_reason = (f"Ranked earlier papers worse than the average component (skill {r.skill:+.2f}); "
                                   f"weight {r.weight:.1%}.")
            elif (r.reliability or 0.0) < 0.5:
                r.status = LIMITED
                note = info.get("input_note", {}).get(r.name, "")
                r.status_reason = (f"Running with reliability {r.reliability:.2f}: it estimates about {r.df:g} "
                                   f"parameter(s) from this course and only {profile.effective_exams:g} paper(s) exist"
                                   + (f" ({note})" if note else "") + f"; weight {r.weight:.1%}.")
            else:
                r.status = ACTIVE
                note = info.get("input_note", {}).get(r.name, "")
                r.status_reason = f"Weight {r.weight:.1%}, reliability {r.reliability:.2f}" + (f" ({note})." if note else ".")
            if r.name == "semantic" and not profile.pretrained and r.status != UNAVAILABLE:
                r.status_reason = ("The pretrained model is unavailable, so semantic evidence comes from TF-IDF "
                                   "alignment only. " + r.status_reason)
        elif r.role == "ensemble":
            r.status, r.status_reason = ACTIVE, "Combines the components below."
        else:
            r.status, r.status_reason = REFERENCE, "Baseline for comparison; not part of the final ranking."
        rows.append({"model": r.name, "display": r.display, "role": r.role, "scope": r.scope, "status": r.status,
                     "reason": r.status_reason, "weight": None if r.weight is None else round(r.weight, 4),
                     "reliability": None if r.reliability is None else round(float(r.reliability), 4),
                     "skill": None if r.skill is None else round(float(r.skill), 4), "df": r.df,
                     "folds": len(r.fold_metrics), "selected": r.name == report.selected})
    return rows


def low_data_message(profile: EvidenceProfile, report) -> tuple[str, str]:
    """(mode label, message). Never says that inference is switched off."""
    T = profile.exams
    learned = [report.models[n].reliability for n in COURSE_LEARNED
               if n in report.models and report.models[n].reliability is not None]
    low = (max(learned) if learned else 0.0) < 0.5
    sem = "Advanced semantic and Bayesian inference remains active" if profile.pretrained else \
        "Bayesian inference and the cross-course model remain active (the pretrained semantic model is unavailable)"
    if T == 0:
        return ("Low-data advanced inference",
                "No historical examinations are available yet. The ranking uses the syllabus structure and the general "
                "cross-course model; every topic has high uncertainty.")
    if low:
        noun = "examination is" if T == 1 else "examinations are"
        return ("Low-data advanced inference",
                f"Only {T} historical {noun} available. {sem}, but course-specific learned ranking parameters have "
                f"high uncertainty.")
    return ("Advanced inference",
            f"{T} historical examinations are available. {sem}; course-specific learned components carry weight in "
            f"proportion to their measured skill on earlier papers.")


TEMPORAL_COMPONENTS = ("recency", "beta_binomial", "hazard", "markov", "hmm", "cooccurrence")
WEIGHT_SOURCE_QUALITY = {"teaching hours": 1.0, "marks weights": 1.0, "breadth (sub-topics and concepts)": 0.7,
                         "uniform": 0.5}


def input_quality(history, panel_meta: dict) -> dict[str, tuple[float, str]]:
    """Quality (0-1) of each component's inputs, from papers before the cutoff only.

    It multiplies the component's reliability prior, so the evidence profile (question mapping
    confidence, metadata completeness, calendar coverage, format variety, syllabus weights, pretrained
    availability) changes the weights, not only the report.
    """
    from ..models.components import syllabus_share

    T = history.T
    out: dict[str, tuple[float, str]] = {}
    nq = float(history.n_questions.sum()) if T else 0.0
    if history.quality is not None and nq > 0:
        q = history.quality.sum(axis=(0, 1)) / nq
        map_conf, parse_conf, marks_known = float(q[0]), float(q[1]), float(q[2])
    else:
        map_conf = parse_conf = marks_known = 1.0
    pretrained = bool(panel_meta.get("pretrained", False))
    sem = (0.5 + 0.5 * map_conf) * (1.0 if pretrained else 0.8)
    out["semantic"] = (sem, f"mapping confidence {map_conf:.0%}" + ("" if pretrained else ", TF-IDF only"))
    formats = int((history.formats.sum(axis=(0, 1)) > 0).sum()) if T else 0
    out["question_type"] = (min(1.0, max(formats, 1) / 3.0), f"{formats} question format(s) seen")
    source = syllabus_share(history)[1]
    out["coverage"] = (WEIGHT_SOURCE_QUALITY.get(source, 0.7), f"syllabus weights from {source}")
    years = sorted({e.year for e in history.exams if e.year is not None})
    if len(years) >= 2:
        span = years[-1] - years[0] + 1
        missing = span - len(years)
        temporal = 1.0 - 0.5 * missing / span
        why = f"{missing} missing calendar year(s) in {span}"
    else:
        temporal, why = 1.0, "calendar coverage not measurable yet"
    for m in TEMPORAL_COMPONENTS:
        out[m] = (temporal, why)
    learned = 0.5 + 0.25 * marks_known + 0.25 * parse_conf
    for m in COURSE_LEARNED:
        out[m] = (learned, f"marks known {marks_known:.0%}, parse confidence {parse_conf:.0%}")
    out["general"] = (1.0, "cross-course knowledge")
    return out
