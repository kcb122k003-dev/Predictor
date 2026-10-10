"""The "Analyze & Predict" pipeline: from stored papers and syllabus to persisted predictions.

Order of work:
 1. load the current syllabus tree and the included exams (time order)
 2. align every leaf question to the syllabus (manual mappings are kept)
 3. classify question types (manual types are kept)
 4. detect question recurrence (exact, paraphrase, concept)
 5. build the topic, concept and unfiltered panels
 6. backtest every candidate model, select, calibrate, audit leakage
 7. ablation and syllabus-filter check
 8. format forecast, recurring question families, per-topic history (the counts both the
    Syllabus Explorer and the Predictions tab show), format guides, grounded formulations
 9. structure discovery, coverage, co-occurrence, charts, paper simulation
10. persist everything with the run, with a fingerprint of the data it read
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from sqlalchemy import delete, select

from ..analysis.coverage import cooccurrence, topic_coverage, unit_coverage
from ..analysis.topic_history import ExcludedPaper, PaperInfo, QuestionInfo, build_history, relation_of
from ..analysis.structure import ExamSummary, LeafInfo, MainQuestion, discover_structure
from ..database.models import (AnalysisArtifact, AnalysisRun, BacktestFold, Course, CourseTopic, DocumentPage, Exam,
                               ExamQuestion, ModelResult, PredictedQuestion, Prediction, QuestionTopicMapping,
                               SourceFile, SyllabusVersion)
from ..embeddings.backends import get_backend, get_pretrained
from ..embeddings.index import DbEmbeddingCache
from ..evaluation.ablation import run_ablation, syllabus_filter_check
from ..evaluation.backtest import BacktestEngine, BacktestReport
from ..features.builder import GROUP_LABELS
from ..generation.paper import simulate_papers
from ..generation.questions import HistoricalQuestion, generate_formulations, proper_nouns
from ..generation.verify import FormulationVerifier
from ..inference.evidence import build_profile, component_status, low_data_message
from ..inference.generic import panel_rows
from ..inference.repository import general_model_for, register_course_model, save_course_features
from ..inference.uncertainty import topic_uncertainty
from ..models.components import syllabus_share
from ..parsing.format_labels import FAMILY_GENERATOR_FORMAT, families_of, format_labels
from ..parsing.question_types import QuestionTypeClassifier
from ..prediction.families import backtest_families, predict_families
from ..prediction.format_guide import build_format_guide, clean_question, format_reliability
from ..prediction.priority_reason import priority_reason
from ..prediction.ranking import CATEGORY_ORDER, DISCLAIMER, build_topic_predictions
from ..prediction.type_forecast import run_type_forecast
from ..syllabus.alignment import (AlignmentResult, QuestionItem, SyllabusAligner, counts_for_prediction,
                                  describe_status, match_confidence, thresholds_for)
from ..syllabus.tree import TopicTree
from ..temporal.dynamics import rotation_tests
from ..temporal.panel import FORMATS, ExamInfo, Panel, QuestionRecord, build_panel
from ..topic_modeling.families import FamilyQuestion, find_recurrence
from ..utils.logging import get_logger, log_event
from ..visualization.charts import build_charts
from .fingerprint import combined as combine_fingerprint
from .fingerprint import data_fingerprint
from .context import AppContext
from .syllabus_store import current_version, to_tree, topics_of

log = get_logger("analysis")
RANK_WEIGHTS = {1: 1.0, 2: 0.5, 3: 0.33}
ENGINE_VERSION = "2.0"
# Exams with these sources never enter an analysis (nothing creates them; this is a guard).
NON_HISTORICAL_SOURCES = ("generated", "simulated")
KIND_FORMATS = {"numerical": "numerical", "derivation": "derivation", "theory": "theory", "definition": "definition",
                "diagram": "diagram", "objective": "objective"}


class AnalysisError(RuntimeError):
    """A user-facing problem (missing data) rather than a bug."""


@dataclass
class Leaf:
    id: int
    exam_index: int
    exam_id: int
    text: str
    context: str
    marks: float | None
    types: list[str]
    type_user_edited: bool
    options: list[Any]
    path_label: str
    manual: list[tuple[int, str]] = field(default_factory=list)  # (topic id, status)
    parse_confidence: float = 1.0
    optional: bool = False
    format: str = "theory"
    alignment: AlignmentResult | None = None
    counted: list[tuple[int, float]] = field(default_factory=list)  # (mappable node id, weight)
    status: str = "C"
    ranks: dict[int, int] = field(default_factory=dict)  # counted node -> mapping rank (1 = primary)
    uncounted: list[tuple[int, str, float]] = field(default_factory=list)  # (node, status, score) shown, not counted
    page_no: int | None = None
    line_no: int | None = None
    needs_review: bool = False
    labels: list[str] = field(default_factory=list)  # student-facing format labels
    families: list[str] = field(default_factory=list)


class AnalysisService:
    def __init__(self, app: AppContext):
        self.app = app

    # ---------------------------------------------------------------- run API
    def create_run(self, course_id: int) -> int:
        with self.app.db.session() as s:
            if s.get(Course, course_id) is None:
                raise AnalysisError("Course not found.")
            run = AnalysisRun(course_id=course_id, status="queued", progress=0.0, message="Queued")
            s.add(run)
            s.flush()
            return run.id

    def _progress(self, run_id: int, value: float, message: str) -> None:
        with self.app.db.session() as s:
            run = s.get(AnalysisRun, run_id)
            run.progress = round(value, 3)
            run.message = message
            run.status = "running"
        log_event(log, "analysis_progress", run_id=run_id, progress=round(value, 2), step=message)

    def run(self, run_id: int) -> dict[str, Any]:
        started = time.time()
        try:
            # The matrices are small: one BLAS/OpenMP thread avoids oversubscription (about 15% faster).
            try:
                from threadpoolctl import threadpool_limits

                limits = threadpool_limits(limits=1)
            except Exception:  # pragma: no cover - threadpoolctl ships with scikit-learn
                limits = None
            try:
                summary = self._run(run_id, started)
            finally:
                if limits is not None:
                    limits.restore_original_limits()
            with self.app.db.session() as s:
                run = s.get(AnalysisRun, run_id)
                run.status = "done"
                run.progress = 1.0
                run.message = "Analysis complete"
                run.finished_at = datetime.now(timezone.utc)
                run.summary = summary
            return summary
        except AnalysisError as exc:
            self._fail(run_id, str(exc), user_error=True)
            raise
        except Exception as exc:
            log.exception("analysis failed", extra={"event": "analysis_failed", "run_id": run_id})
            self._fail(run_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}", user_error=False)
            raise

    def _fail(self, run_id: int, message: str, user_error: bool) -> None:
        with self.app.db.session() as s:
            run = s.get(AnalysisRun, run_id)
            if run is not None:
                run.status = "error"
                run.message = message[:4000]
                run.finished_at = datetime.now(timezone.utc)
                run.summary = {"error": message[:4000], "user_error": user_error}

    # ----------------------------------------------------------------- loading
    def _load(self, course_id: int):
        with self.app.db.session() as s:
            course = s.get(Course, course_id)
            settings = self.app.course_settings(course)
            version = current_version(s, course_id, create=False)
            topics = topics_of(s, course_id, version.id) if version else []
            if not topics:
                raise AnalysisError("No syllabus topics yet. Upload the course contents (syllabus) first, or add "
                                    "topics in the Syllabus tab.")
            tree = to_tree(topics)
            historical = []
            for v in s.execute(select(SyllabusVersion).where(SyllabusVersion.course_id == course_id,
                                                             SyllabusVersion.is_current.is_(False))).scalars():
                vt = topics_of(s, course_id, v.id)
                if vt:
                    historical.append((v.label, to_tree(vt)))
            synthetic = bool(course.is_synthetic)
            exams = s.execute(select(Exam).where(Exam.course_id == course_id, Exam.include_in_analysis.is_(True),
                                                 Exam.source.notin_(NON_HISTORICAL_SOURCES))
                              .order_by(Exam.order_index, Exam.id)).scalars().all()
            # With no past paper the ranking comes from the syllabus structure and the cross-course model
            # (and every topic is marked highly uncertain); there is no minimum number of papers.
            infos, leaves, exam_rows, papers = [], [], [], []
            for idx, e in enumerate(exams):
                rows = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == e.id)
                                 .order_by(ExamQuestion.order_no)).scalars().all()
                main_marks = _main_marks(rows)
                total = e.full_marks or main_marks or 100.0
                label = _exam_label(e)
                infos.append(ExamInfo(e.id, e.order_index, label, float(total), e.year))
                exam_rows.append((e, rows))
                src = s.get(SourceFile, e.source_file_id) if e.source_file_id else None
                papers.append(PaperInfo(idx, e.id, label, e.year, e.session or "", src.filename if src else None,
                                        src.id if src else None, _paper_quality(s, e, src, rows)))
                for q in rows:
                    if not q.is_leaf:
                        continue
                    # Manual mappings, including "outside the syllabus" (status D with no topic), override alignment.
                    manual = [(m.topic_id, m.status) for m in sorted(q.mappings, key=lambda m: m.rank or 1)
                              if m.method == "manual"]
                    leaves.append(Leaf(q.id, idx, e.id, q.text or "", q.context_text or q.text or "", q.marks,
                                       list(q.question_types or []), q.type_user_edited, list(q.options or []),
                                       q.path_label, manual,
                                       parse_confidence=float(q.parse_confidence if q.parse_confidence is not None
                                                              else 1.0),
                                       optional=bool(q.is_optional or q.or_group), page_no=q.page_no,
                                       line_no=q.line_no, needs_review=bool(q.needs_review)))
            excluded = []
            for e in s.execute(select(Exam).where(Exam.course_id == course_id, Exam.include_in_analysis.is_(False),
                                                  Exam.source.notin_(NON_HISTORICAL_SOURCES))
                               .order_by(Exam.order_index, Exam.id)).scalars():
                kind = "duplicate" if e.duplicate_of_id else "undated" if e.year is None else "excluded"
                reason = e.exclusion_reason or ("Excluded from the analysis by you." if kind == "excluded" else "")
                excluded.append(ExcludedPaper(e.id, _exam_label(e), e.year, kind, reason, e.duplicate_of_id))
            snapshot = [(e.id, e.order_index, e.structure, e.full_marks, e.duration,
                         [(q.id, q.parent_id, q.label, q.marks, q.or_group, q.is_optional, q.is_leaf, q.text, q.section_id)
                          for q in rows]) for e, rows in exam_rows]
            fingerprint = data_fingerprint(s, course_id, settings)
        return settings, tree, historical, infos, leaves, snapshot, synthetic, papers, excluded, fingerprint

    # ------------------------------------------------------------------- main
    def _run(self, run_id: int, started: float) -> dict[str, Any]:
        with self.app.db.session() as s:
            course_id = s.get(AnalysisRun, run_id).course_id
        self._progress(run_id, 0.03, "Loading syllabus and papers")
        (settings, tree, historical, infos, leaves, snapshot, synthetic, papers, excluded_papers,
         fingerprint) = self._load(course_id)
        with self.app.db.session() as s:
            run_row = s.get(AnalysisRun, run_id)
            run_row.data_fingerprint = combine_fingerprint(fingerprint)
            run_row.config = {**(run_row.config or {}), "fingerprint_parts": fingerprint}
        notes: list[str] = []
        cache = DbEmbeddingCache(self.app.db.new_session)
        backend, backend_note = get_backend(settings, cache)
        if backend_note:
            notes.append(backend_note)
        pretrained, pretrained_note = get_pretrained(settings)

        # 2. Alignment ----------------------------------------------------------
        self._progress(run_id, 0.1, "Mapping questions to the syllabus")
        aligner = SyllabusAligner(settings, tree, backend)
        # Causal feedback: each paper is mapped using feedback from earlier papers only.
        results = aligner.align([QuestionItem(q.id, q.text, q.context, float(q.exam_index)) for q in leaves])
        th = thresholds_for(settings, backend.kind)
        for q, r in zip(leaves, results):
            q.alignment = r
            if q.manual:
                q.status = q.manual[0][1]
                kept = [(i + 1, tid) for i, (tid, st) in enumerate(q.manual)
                        if tid is not None and counts_for_prediction(st, 1.0, "manual", settings, backend.kind)
                        and tid in tree.nodes]
                q.counted = [(tid, RANK_WEIGHTS.get(rank, 0.33)) for rank, tid in kept]
                q.ranks = {tid: rank for rank, tid in kept}
            else:
                q.status = r.status
                kept = [m for m in r.matches if counts_for_prediction(m.status, m.score, "auto", settings, backend.kind)]
                q.counted = [(m.topic_id, RANK_WEIGHTS.get(m.rank, 0.33)) for m in kept]
                q.ranks = {m.topic_id: m.rank for m in kept}
                # Matches shown for review but not counted (uncertain, status C, or below the strictness bar).
                q.uncounted = [(m.topic_id, m.status if m.rank > 1 else r.status, float(m.score)) for m in r.matches
                               if m not in kept and (m.status if m.rank > 1 else r.status) != "D"]
        hist_matches = self._historical_matches(settings, historical, backend, leaves)
        self._persist_mappings(leaves, hist_matches)

        # 3. Types ------------------------------------------------------------
        self._progress(run_id, 0.2, "Classifying question types")
        classifier = QuestionTypeClassifier.load(self.app.data_dir)
        type_updates = {}
        for q in leaves:
            if not q.type_user_edited:
                tr = classifier.classify(q.text, marks=q.marks, options=q.options,
                                         context=q.context if q.context != q.text else "")
                q.types = tr.types
                q.format = tr.format
                type_updates[q.id] = (tr.types, {"scores": tr.scores, "format": tr.format,
                                                 "confidence": tr.confidence, "evidence": tr.evidence})
            else:
                q.format = classifier.format_of(q.types[0]) if q.types else "theory"
            if q.format not in FORMATS:
                q.format = "theory"
            q.labels = format_labels(q.text, q.types, q.marks, q.context if q.context != q.text else "")
            q.families = families_of(q.labels)
        self._persist_types(type_updates)

        # 4. Recurrence ---------------------------------------------------------
        self._progress(run_id, 0.27, "Finding repeated and paraphrased questions")
        primary_node = {q.id: (q.counted[0][0] if q.counted else None) for q in leaves}
        fam_qs = [FamilyQuestion(q.id, q.exam_index, q.text,
                                 tree.topic_of(primary_node[q.id]) if primary_node[q.id] else None,
                                 primary_node[q.id]) for q in leaves]
        # Character n-grams found reworded repeats better than the bundled static embeddings on the
        # demo and paraphrase checks, so only a full sentence-transformer replaces them here.
        recurrence = find_recurrence(fam_qs, settings, backend if backend.kind == "neural" else None)

        # 5. Panels ---------------------------------------------------------------
        self._progress(run_id, 0.32, "Building the exam x topic panel")
        topic_ids = tree.topic_ids()
        unit_ids = tree.unit_ids()
        concept_ids = tree.concept_ids()
        semantic_vectors = pretrained or backend
        panel_args = dict(aligner=aligner, semantic_vectors=semantic_vectors,
                          temperature=float(settings.alignment.semantic_temperature))
        topic_panel, cell_q = self._panel(tree, infos, leaves, recurrence, topic_ids, unit_ids, "topic", filtered=True,
                                          **panel_args)
        concept_panel, _ = self._panel(tree, infos, leaves, recurrence, concept_ids, unit_ids, "concept", filtered=True,
                                       **panel_args)
        unfiltered_panel, _ = self._panel(tree, infos, leaves, recurrence, topic_ids, unit_ids, "topic", filtered=False,
                                          **panel_args)
        for panel in (topic_panel, concept_panel, unfiltered_panel):
            panel.meta["semantic_source"] = getattr(backend, "name", "alignment")
            panel.meta["pretrained"] = pretrained is not None
        # Per-topic history from the same counted mappings as the topic panel (checked against it below).
        history = build_history(papers, excluded_papers, _question_infos(leaves, recurrence), tree.topic_of, topic_ids)
        _check_history(history, topic_panel, topic_ids)

        # 6. Backtest ---------------------------------------------------------------
        self._progress(run_id, 0.38, "Backtesting every component on past exams")
        with self.app.db.session() as s:
            general, general_info = general_model_for(s, course_id,
                                                      use_simulated=bool(settings.models.use_simulated_prior))
            run_row = s.get(AnalysisRun, run_id)
            run_row.config = {**(run_row.config or {}), "general_model": general_info}
        engine = BacktestEngine(topic_panel, settings, general=general)
        report = engine.run()
        self._progress(run_id, 0.55, "Backtesting the concept layer")
        concept_report = BacktestEngine(concept_panel, settings, general=general).run(audit_leakage=False,
                                                                                     calibrate_scores=False)
        unf_report = BacktestEngine(unfiltered_panel, settings, general=general).run(audit_leakage=False,
                                                                                   calibrate_scores=False)

        # 7. Evidence, uncertainty, ablation ---------------------------------------------
        self._progress(run_id, 0.62, "Measuring evidence, uncertainty and the ablation study")
        profile = build_profile(topic_panel, questions=sum(1 for q in leaves if q.counted), questions_total=len(leaves),
                                questions_with_marks=sum(1 for q in leaves if q.marks),
                                syllabus_weights=syllabus_share(topic_panel)[1], pretrained=pretrained is not None,
                                pretrained_note=pretrained_note, repository_courses=general_info["real_courses"],
                                report=report)
        status_rows = component_status(report, profile)
        component_status(concept_report, profile)
        mode, mode_message = low_data_message(profile, report)
        uncertainty = topic_uncertainty(topic_panel, settings, report, engine.models, general=general,
                                        seed=int(settings.models.random_seed))
        col_of_topic = {tid: i for i, tid in enumerate(topic_ids)}
        q_exam = {q.id: q.exam_index for q in leaves}
        q_topic = {q.id: col_of_topic.get(tree.topic_of(primary_node[q.id])) if primary_node[q.id] else None
                   for q in leaves}
        half_life = float(settings.temporal.default_half_life)
        fam_k = max(5, report.k)

        def recurrence_recall(preds: dict[int, np.ndarray]) -> dict[int, float]:
            res = backtest_families(recurrence, q_exam, q_topic, preds, report.targets, fam_k, half_life)
            return res.get("per_fold", {})

        ablation = run_ablation(report, concept_report, recurrence_recall)
        filter_check = syllabus_filter_check(report.predictions.get(report.selected, {}),
                                             unf_report.predictions.get(report.selected, {}), topic_panel,
                                             unfiltered_panel, report.targets, report.k, report.primary,
                                             report.models[report.selected].display)

        # 8. Layer 2 ---------------------------------------------------------------------
        self._progress(run_id, 0.72, "Forecasting question formats and recurring questions")
        fine_types: dict[int, Counter] = defaultdict(Counter)
        for q in leaves:
            for nid, _ in q.counted[:1]:
                fine_types[tree.topic_of(nid)].update(q.types[:1])
        type_report = run_type_forecast(topic_panel, settings, report.targets, {k: dict(v) for k, v in fine_types.items()})
        families_bt = backtest_families(recurrence, q_exam, q_topic, report.predictions[report.selected],
                                        report.targets, fam_k, half_life)
        families_bt.pop("per_fold", None)
        family_preds = predict_families(recurrence, q_exam, q_topic, report.final_scores, topic_panel.T, half_life)

        # 9. Explanations -----------------------------------------------------------------
        self._progress(run_id, 0.8, "Explaining predictions")
        fm_final = engine.store.at(topic_panel.T)
        explain_output = report.final_outputs.get("logistic") or report.final_outputs.get("general")
        mapping_conf = np.zeros(len(topic_ids))
        conf_lists: dict[int, list[float]] = defaultdict(list)
        for q in leaves:
            if q.alignment is None:
                continue
            for nid, _ in q.counted[:1]:
                col = col_of_topic.get(tree.topic_of(nid))
                match = next((m for m in q.alignment.matches if m.topic_id == nid), None)
                if col is not None:
                    conf_lists[col].append(1.0 if q.manual else (match.confidence if match else 0.5))
        for col, vals in conf_lists.items():
            mapping_conf[col] = float(np.mean(vals))
        rotation = {}
        for r in rotation_tests(topic_panel.Y, min_gaps=int(settings.temporal.rotation_min_gaps),
                                alpha=float(settings.temporal.rotation_p_value),
                                permutations=int(settings.temporal.rotation_permutations),
                                seed=int(settings.models.random_seed)):
            rotation[r.item] = {"gaps": r.gaps, "cv": r.cv, "p_value": r.p_value, "period": r.period,
                                "significant": r.significant}
        locations = {tid: _location(tree, tid) for tid in topic_ids}
        semantic_evidence = self._semantic_evidence(tree, topic_ids, leaves, infos, semantic_vectors)
        preds = build_topic_predictions(topic_panel, report, fm_final, settings, explain_output=explain_output,
                                        mapping_confidence=mapping_conf, rotation=rotation,
                                        type_forecast=type_report.per_topic, syllabus_location=locations,
                                        lab_items={t for t in topic_ids if _is_lab(tree, t)}, uncertainty=uncertainty,
                                        semantic_evidence=semantic_evidence)

        # Format guide for every topic (what kind of question to prepare), from its own counted questions.
        verifier = FormulationVerifier(tree, aligner, classifier, pretrained)
        guides = self._format_guides(tree, topic_ids, leaves, infos, history, type_report, verifier)

        # Illustrative practice questions for every topic, in the order of its format guide, each verified.
        history_by_topic: dict[int, list[HistoricalQuestion]] = defaultdict(list)
        course_hist: list[HistoricalQuestion] = []
        for q in leaves:
            hq = HistoricalQuestion(q.id, q.text, q.format, q.types, q.marks, infos[q.exam_index].label, q.exam_index,
                                    recurrence.family_of.get(q.id))
            if q.status in ("A", "B") or q.manual:
                course_hist.append(hq)
            for nid, _ in q.counted[:1]:
                history_by_topic[tree.topic_of(nid)].append(hq)
        formulations: dict[int, list[dict[str, Any]]] = {}
        for p in preds:
            dist = type_report.per_topic.get(p.item_id, {}).get("distribution")
            guide = guides.get(p.item_id) or {}
            order = [FAMILY_GENERATOR_FORMAT.get(f) for f in [guide.get("family")]
                     + [alt["family"] for alt in guide.get("alternatives", [])]]
            forms = generate_formulations(tree, p.item_id, history_by_topic.get(p.item_id, []), course_hist, dist,
                                          settings, verifier=verifier, format_order=[f for f in order if f])
            formulations[p.item_id] = [f.as_dict() for f in forms]
            _attach_illustrative(guide, formulations[p.item_id])
        reasons = {p.item_id: priority_reason(category=p.category, rank=p.rank, topics=len(preds),
                                              stats=history["topics"].get(str(p.item_id), {}), facts=p.facts,
                                              guide=guides.get(p.item_id)) for p in preds}

        # 10. Structure, coverage, charts, papers -------------------------------------------
        self._progress(run_id, 0.88, "Analysing paper structure and coverage")
        summaries = self._exam_summaries(snapshot, infos, leaves, tree, col_of_topic, unit_ids)
        structure = discover_structure(summaries)
        unit_labels = [tree.nodes[u].label() for u in unit_ids]
        coverage = {"units": unit_coverage(topic_panel, unit_labels), "topics": topic_coverage(topic_panel),
                    "cooccurrence": cooccurrence(topic_panel)}
        leaf_marks = [q.marks for q in leaves if q.marks]
        charts = build_charts(topic_panel, cell_q, unit_labels, report, leaf_marks, settings)
        unit_share = coverage["units"].get("marks_share") if coverage["units"].get("available") else None
        max_unit_share = float(np.max(unit_share)) if unit_share is not None and np.size(unit_share) else 0.35
        papers = simulate_papers(
            topic_ids=topic_ids, topic_labels=topic_panel.item_labels, scores=_paper_scores(preds, len(topic_ids)),
            units=topic_panel.item_unit, unit_labels=unit_labels, typical=structure.typical, formulations=formulations,
            type_forecast=type_report.per_topic, max_unit_share=max_unit_share,
            variants=int(settings.generation.paper_variants), seed=int(settings.generation.paper_seed))
        excluded = self._excluded_groups(leaves, recurrence, infos, tree)

        # 11. Persist ------------------------------------------------------------------------
        self._progress(run_id, 0.95, "Saving results")
        evidence = {"mode": mode, "message": mode_message, "profile": profile.as_dict(), "components": status_rows,
                    "validation": report.validation, "general_model": general_info,
                    "uncertainty": {"jackknife_replicates": uncertainty["jackknife_replicates"],
                                    "posterior_draws": uncertainty["posterior_draws"],
                                    "held_fixed": uncertainty["held_fixed"],
                                    "levels": dict(Counter(uncertainty["level"])),
                                    "median_rank_range": float(np.median(uncertainty["rank_high"] - uncertainty["rank_low"]))
                                    if len(topic_ids) else None,
                                    "overall": _overall_uncertainty(uncertainty["level"])},
                    "evidence_quality": _evidence_quality(preds),
                    "ensemble": _ensemble_summary(report), "generation_checks": verifier.summary(),
                    "synthetic_course": synthetic, "notes": profile.notes}
        # Kept under its old key for older clients: the same evidence status, never "disabled".
        sufficiency = {"exams": topic_panel.T, "folds": len(report.targets), "mode": mode, "message": mode_message,
                       "models": [{"model": r["display"], "enabled": r["status"] != "UNAVAILABLE",
                                   "status": r["status"], "reason": r["reason"]} for r in status_rows],
                       "notes": report.notes}
        summary = {
            "exams": topic_panel.T, "questions": len(leaves), "topics": len(topic_ids), "concepts": len(concept_ids),
            "units": len(unit_ids), "k": report.k, "primary_metric": report.primary,
            "selected_model": report.selected, "selected_display": report.models[report.selected].display,
            "selection_reason": report.selection_reason, "best_by_mean": report.best_by_mean,
            "calibrated": bool(report.calibration and report.calibration.valid),
            "calibration_reason": report.calibration.reason if report.calibration else
            "No backtest folds: probabilities cannot be calibrated.",
            "notes": report.notes + notes, "embedding_backend": backend.name, "disclaimer": DISCLAIMER,
            "status_counts": dict(Counter(q.status for q in leaves)),
            "counted_questions": sum(1 for q in leaves if q.counted),
            "leakage_audit": report.leakage_audit, "sufficiency": sufficiency,
            "inference_mode": mode, "inference_message": mode_message, "validation": report.validation,
            "engine_version": ENGINE_VERSION,
            "concept_layer": _layer_summary(concept_report),
            "type_forecast": {"method": type_report.method, "reason": type_report.reason},
            "families": families_bt, "seconds": round(time.time() - started, 2),
            "fingerprint": _fingerprint(leaves, topic_ids, settings),
            "settings_used": {"strictness": float(settings.alignment.strictness), "top_k": str(settings.models.top_k),
                              "selection_rule": "evidence-aware ensemble unless a single method is reliably better",
                              "alignment_thresholds": th},
        }
        topic_extras = {tid: {"history": _history_summary(history["topics"].get(str(tid), {})),
                              "format_guide": guides.get(tid), "priority_reason": reasons.get(tid)}
                        for tid in topic_ids}
        self._persist_results(run_id, settings, report, concept_report, preds, formulations, family_preds,
                              concept_panel, tree, recurrence, infos, {
                                  "charts": charts,
                                  "structure": {"per_exam": structure.per_exam, "patterns": structure.patterns,
                                                "hypotheses": structure.hypotheses, "typical": structure.typical},
                                  "coverage": coverage, "ablation": ablation, "syllabus_filter": filter_check,
                                  "calibration": report.calibration.as_dict() if report.calibration else {"valid": False},
                                  "type_forecast": {"method": type_report.method, "reason": type_report.reason,
                                                    "accuracy": type_report.accuracy, "gated": type_report.gated},
                                  "families": {"backtest": families_bt, "predictions": family_preds},
                                  "papers": {"papers": papers, "structure_hypotheses": structure.hypotheses},
                                  "excluded": {"groups": excluded},
                                  "topic_history": history,
                                  "rotation": {"items": [{"label": topic_panel.item_labels[k], **v}
                                                         for k, v in rotation.items()]},
                                  "sufficiency": sufficiency, "evidence": evidence,
                                  "models": {"table": report.summary_table(), "selected": report.selected,
                                             "reason": report.selection_reason, "k": report.k,
                                             "primary": report.primary, "targets": [infos[t].label for t in report.targets],
                                             "leakage_audit": report.leakage_audit, "validation": report.validation,
                                             "concept_table": concept_report.summary_table()},
                              }, course_id=course_id, synthetic=synthetic, settings_snapshot=settings,
                              topic_extras=topic_extras)
        log_event(log, "analysis_done", run_id=run_id, seconds=summary["seconds"], selected=report.selected,
                  mode=mode)
        return summary

    # ------------------------------------------------------------- helpers
    def _format_guides(self, tree: TopicTree, topic_ids: list[int], leaves: list[Leaf], infos: list[ExamInfo],
                       history: dict[str, Any], type_report, verifier: FormulationVerifier) -> dict[int, dict[str, Any]]:
        """The question format guide of every topic, from the questions counted for it in ``history``."""
        leaf_by_id = {q.id: q for q in leaves}
        reliability = format_reliability(type_report.accuracy)
        names = proper_nouns([q.text for q in leaves])
        out = {}
        for tid in topic_ids:
            stats = history["topics"].get(str(tid), {})
            qs = []
            for key, role in stats.get("roles", {}).items():
                q = leaf_by_id.get(int(key))
                if q is None:
                    continue
                qs.append({"id": q.id, "text": q.text, "labels": q.labels, "families": q.families, "marks": q.marks,
                           "exam_index": q.exam_index, "exam_label": infos[q.exam_index].label,
                           "path_label": q.path_label, "role": role})
            node = tree.nodes[tid]
            kinds = set(node.kinds)
            concepts = list(node.concepts)
            for d in tree.descendants(tid):
                kinds |= set(tree.nodes[d].kinds)
                concepts += list(tree.nodes[d].concepts)
            def accept(q: dict[str, Any], tid: int = tid) -> bool:
                # A past numerical shown with its real values must pass the same topic check as generated questions.
                return bool(verifier.check(clean_question(q["text"]), tid, "numerical", "historical_variant")["passed"])

            out[tid] = build_format_guide(topic_label=node.label(), topic_title=node.title, kinds=kinds,
                                          concepts=concepts, stats=stats, questions=qs,
                                          course_families=history.get("course_families", []),
                                          papers=history["papers"], reliability=reliability, names=names,
                                          accept=accept)
        return out

    def _historical_matches(self, settings, historical, backend, leaves) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        weak = [q for q in leaves if q.status in ("C", "D") and not q.manual]
        if not weak or not historical:
            return out
        for label, htree in historical:
            from ..embeddings.backends import TfidfBackend
            from ..embeddings.pretrained import HybridBackend

            # A fresh backend per historical syllabus: the aligner fits TF-IDF on the syllabus it is given.
            if backend.kind == "neural":
                hb = backend
            elif backend.kind == "hybrid":
                hb = HybridBackend(settings, backend.pretrained)
            else:
                hb = TfidfBackend(settings)
            res = SyllabusAligner(settings, htree, hb).align([QuestionItem(q.id, q.text, q.context) for q in weak],
                                                              feedback=False)
            for q, r in zip(weak, res):
                if r.status in ("A", "B") and r.matches and q.id not in out:
                    out[q.id] = {"version": label, "topic": htree.nodes[r.matches[0].topic_id].title,
                                 "score": r.matches[0].score}
        return out

    def _persist_mappings(self, leaves: list[Leaf], hist_matches: dict[int, dict[str, Any]]) -> None:
        with self.app.db.session() as s:
            ids = [q.id for q in leaves if not q.manual]
            for start in range(0, len(ids), 500):
                s.execute(delete(QuestionTopicMapping).where(QuestionTopicMapping.question_id.in_(ids[start:start + 500]),
                                                             QuestionTopicMapping.method == "auto"))
            for q in leaves:
                if q.manual or q.alignment is None:
                    continue
                r = q.alignment
                evidence = {"reason": r.reason, "unknown_terms": r.unknown_terms, "status_label": describe_status(r.status)}
                if q.id in hist_matches:
                    evidence["historical_syllabus"] = hist_matches[q.id]
                    evidence["reason"] += (f" It matches the '{hist_matches[q.id]['version']}' syllabus topic "
                                           f"'{hist_matches[q.id]['topic']}', which is not in the current syllabus.")
                if not r.matches:
                    s.add(QuestionTopicMapping(question_id=q.id, topic_id=None, rank=1, confidence=0.0, status=r.status,
                                               evidence_text="", evidence=evidence, method="auto",
                                               unknown_term_ratio=r.unknown_ratio))
                    continue
                for m in r.matches:
                    s.add(QuestionTopicMapping(
                        question_id=q.id, topic_id=m.topic_id, rank=m.rank, confidence=m.confidence,
                        status=m.status if m.rank > 1 else r.status, semantic_similarity=m.semantic,
                        keyword_overlap=m.keyword, unknown_term_ratio=r.unknown_ratio, matched_terms=m.matched_terms,
                        evidence_text=m.evidence_text, evidence={**evidence, "score": m.score}, method="auto"))

    def _persist_types(self, updates: dict[int, tuple[list[str], dict[str, Any]]]) -> None:
        if not updates:
            return
        with self.app.db.session() as s:
            for qid, (types, scores) in updates.items():
                row = s.get(ExamQuestion, qid)
                if row is not None and not row.type_user_edited:
                    row.question_types = types
                    row.type_scores = scores

    def _panel(self, tree: TopicTree, infos: list[ExamInfo], leaves: list[Leaf], recurrence, item_ids: list[int],
               unit_ids: list[int], layer: str, filtered: bool, aligner: SyllabusAligner | None = None,
               semantic_vectors=None, temperature: float = 0.05
               ) -> tuple[Panel, dict[tuple[int, int], list[int]]]:
        col = {nid: i for i, nid in enumerate(item_ids)}
        unit_col = {u: i for i, u in enumerate(unit_ids)}
        K = len(item_ids)

        def to_col(nid: int) -> int | None:
            if layer == "topic":
                return col.get(tree.topic_of(nid))
            if layer == "concept":
                # Only exact concept matches count; a question mapped to a parent node is not a concept.
                return col.get(nid)
            return col.get(tree.unit_of(nid))

        # Map every mappable node the aligner scored to this layer's columns (for semantic soft counts).
        node_col = None
        if aligner is not None and aligner.node_ids:
            node_col = np.array([c if (c := to_col(n)) is not None else -1 for n in aligner.node_ids], dtype=int)

        records, cells = [], defaultdict(list)
        for q in leaves:
            if filtered:
                mapped = q.counted
            else:
                mapped = q.counted or ([(q.alignment.matches[0].topic_id, 1.0)] if q.alignment and q.alignment.matches else [])
            items: dict[int, float] = {}
            for nid, w in mapped:
                c = to_col(nid)
                if c is not None:
                    items[c] = max(items.get(c, 0.0), w)
            soft: dict[int, float] = {}
            if q.alignment is not None and q.status != "D":
                for m in q.alignment.matches:
                    c = to_col(m.topic_id)
                    if c is not None:
                        soft[c] = max(soft.get(c, 0.0), m.score)
            semantic = _semantic_mass(q, items, node_col, K, temperature) if (items or not filtered) else {}
            for c in items:
                cells[(q.exam_index, c)].append(q.id)
            records.append(QuestionRecord(q.id, q.exam_index, q.text, q.marks, q.format, q.types, list(items.items()),
                                          status=q.status, soft=soft, semantic=semantic,
                                          exact_repeat=bool(recurrence.exact_prev.get(q.id)),
                                          para_repeat=bool(recurrence.para_prev.get(q.id)),
                                          mapping_confidence=_mapping_confidence(q), parse_confidence=q.parse_confidence,
                                          optional=q.optional))
        labels = [tree.nodes[i].label() for i in item_ids]
        item_unit = np.array([unit_col.get(tree.unit_of(i), 0) for i in item_ids], dtype=int)
        static = _static_features(tree, item_ids, unit_ids)
        panel = build_panel(infos, item_ids, labels, records, static=static, item_unit=item_unit, unit_ids=unit_ids,
                            layer=layer, item_sim=_item_similarity(tree, item_ids, semantic_vectors),
                            format_prior=_format_prior(tree, item_ids))
        return panel, dict(cells)

    def _semantic_evidence(self, tree: TopicTree, topic_ids: list[int], leaves: list[Leaf], infos: list[ExamInfo],
                           vectors, per_topic: int = 3) -> dict[int, list[dict[str, Any]]]:
        """For each topic, the most similar past in-syllabus questions (semantic evidence for the explanation)."""
        counted = [q for q in leaves if q.counted]
        if vectors is None or not counted or not topic_ids:
            return {}
        try:
            qv = vectors.encode([q.text or q.context for q in counted])
            tv = vectors.encode([tree.document(t) for t in topic_ids])
        except Exception:  # pragma: no cover - semantic evidence is optional
            return {}
        sims = tv @ qv.T
        out: dict[int, list[dict[str, Any]]] = {}
        for i, tid in enumerate(topic_ids):
            order = np.argsort(-sims[i])[:per_topic]
            out[tid] = [{"question_id": counted[j].id, "exam": infos[counted[j].exam_index].label,
                         "text": counted[j].text[:300], "similarity": round(float(sims[i, j]), 3),
                         "mapped_here": any(tree.topic_of(n) == tid for n, _ in counted[j].counted)} for j in order]
        return out

    def _exam_summaries(self, snapshot, infos, leaves, tree, col_of_topic, unit_ids) -> list[ExamSummary]:
        leaf_by_id = {q.id: q for q in leaves}
        unit_col = {u: i for i, u in enumerate(unit_ids)}
        out = []
        for (eid, order, structure, full_marks, duration, rows), info in zip(snapshot, infos):
            children = Counter(r[1] for r in rows if r[1] is not None)
            mains = []
            for (qid, parent, label, marks, or_group, optional, is_leaf, text, section) in rows:
                if parent is None:
                    mains.append(MainQuestion(label, marks, children.get(qid, 0), or_group, optional, section,
                                              "short note" in (text or "").lower()))
            leaf_infos = []
            for r in rows:
                q = leaf_by_id.get(r[0])
                if q is None:
                    continue
                nid = q.counted[0][0] if q.counted else None
                leaf_infos.append(LeafInfo(q.format, q.marks, col_of_topic.get(tree.topic_of(nid)) if nid else None,
                                           unit_col.get(tree.unit_of(nid)) if nid else None))
            sections = len({r[8] for r in rows if r[8] is not None})
            out.append(ExamSummary(info.label, order, full_marks, mains, leaf_infos, sections,
                                   (structure or {}).get("attempt_count"), duration or ""))
        return out

    def _excluded_groups(self, leaves: list[Leaf], recurrence, infos, tree) -> list[dict[str, Any]]:
        groups: dict[str, list[Leaf]] = defaultdict(list)
        for q in leaves:
            if q.status == "D":
                groups[recurrence.family_of.get(q.id, f"q{q.id}")].append(q)
        out = []
        for key, qs in groups.items():
            first = qs[0]
            r = first.alignment
            nearest = tree.nodes[r.matches[0].topic_id].title if r and r.matches else None
            reason = r.reason if r else "Outside the current syllabus."
            if first.manual:
                reason = "Marked outside the syllabus by you."
            if r and r.unknown_terms:
                reason += f" Terms not in the syllabus: {', '.join(r.unknown_terms[:6])}."
            out.append({"label": first.text[:160], "reason": reason, "nearest_topic": nearest,
                        "questions": [q.id for q in qs], "years": sorted({infos[q.exam_index].label for q in qs}),
                        "category": CATEGORY_ORDER[4]})
        out.sort(key=lambda g: -len(g["questions"]))
        return out

    def _persist_results(self, run_id, settings, report: BacktestReport, concept_report: BacktestReport, preds,
                         formulations, family_preds, concept_panel: Panel, tree, recurrence, infos,
                         artifacts: dict[str, Any], *, course_id: int, synthetic: bool, settings_snapshot=None,
                         topic_extras: dict[int, dict[str, Any]] | None = None) -> None:
        with self.app.db.session() as s:
            run = s.get(AnalysisRun, run_id)
            run.engine_version = ENGINE_VERSION
            for layer, rep in (("topic", report), ("concept", concept_report)):
                for name, mr in rep.models.items():
                    if mr.hidden:
                        continue
                    s.add(ModelResult(run_id=run_id, layer=layer, model_name=name, display_name=mr.display,
                                      family=mr.family, complexity=mr.complexity, enabled=mr.enabled,
                                      gate_reason=mr.gate_reason, selected=name == rep.selected,
                                      metrics=_clean(mr.mean), metric_se=_clean(mr.se), notes=" ".join(mr.notes),
                                      role=mr.role, scope=mr.scope, status=mr.status, weight=mr.weight,
                                      reliability=None if mr.reliability is None else float(mr.reliability),
                                      evidence=_jsonable({"status_reason": mr.status_reason, "df": mr.df,
                                                          "skill": mr.skill, "sd": _clean(mr.sd), "ci95": mr.ci95,
                                                          "fallback_folds": mr.fallback_folds,
                                                          "unavailable_folds": mr.unavailable_folds,
                                                          "final": _public_info(mr.final_info)})))
                    if layer == "topic":
                        for t, fm in mr.fold_metrics.items():
                            s.add(BacktestFold(run_id=run_id, layer=layer, model_name=name,
                                               target_exam_id=infos[t].exam_id, target_index=t,
                                               target_label=infos[t].label, n_train_exams=t, metrics=_clean(fm)))
            for p in preds:
                s.add(Prediction(
                    run_id=run_id, layer="topic", topic_id=p.item_id, item_key=str(p.item_id), label=p.label,
                    rank=p.rank, score=p.score, probability=p.probability, prob_low=p.prob_low, prob_high=p.prob_high,
                    calibrated=p.calibrated, category=p.category, confidence=p.confidence,
                    features={"facts": _jsonable(p.facts), "signals_for": p.signals_for,
                              "relative_score": p.relative_score,
                              "signal_contributions": {"values": p.signal_contributions, "source": p.signal_source},
                              **_jsonable((topic_extras or {}).get(p.item_id, {}))},
                    contributions={"values": p.contributions, "source": p.contribution_source},
                    evidence={"lines": p.evidence}, why_not=p.why_not, evidence_strength=p.evidence_strength,
                    uncertainty=_jsonable(p.uncertainty)))
                for f in formulations.get(p.item_id, []):
                    s.add(PredictedQuestion(run_id=run_id, topic_id=p.item_id, text=f["text"],
                                            question_type=f["format"], marks_low=f["marks_low"],
                                            marks_high=f["marks_high"], basis=f["basis"], rank=f["rank"],
                                            evidence_question_ids=f["evidence_question_ids"],
                                            grounding=_jsonable({**f["grounding"], "note": f["note"], "label": f["label"],
                                                                 "kind": f.get("kind")})))
            scores = concept_report.final_scores
            order = np.lexsort((np.arange(len(scores)), -scores))
            for rank, i in enumerate(order, start=1):
                nid = concept_panel.item_ids[i]
                s.add(Prediction(run_id=run_id, layer="concept", topic_id=nid, item_key=str(nid),
                                 label=tree.path_label(nid), rank=rank, score=round(float(scores[i]), 5),
                                 category="", confidence="",
                                 features={"appearances": int(concept_panel.Y[:, i].sum())}))
            for rank, fam in enumerate(family_preds, start=1):
                tid = report_topic_id(fam, preds)
                s.add(Prediction(run_id=run_id, layer="family", topic_id=tid, item_key=fam["family"],
                                 label=f"{len(fam['question_ids'])} question(s) in {fam['appearances']} exam(s)",
                                 rank=rank, score=fam["score"], category="", confidence="",
                                 features={"question_ids": fam["question_ids"],
                                           "exams": [infos[e].label for e in fam["exam_indices"]]}))
            for key, data in artifacts.items():
                s.add(AnalysisArtifact(run_id=run_id, key=key, data=_jsonable(data)))
            # Repository: the course's scale-free rows (for other courses' general model; synthetic courses
            # are stored flagged and never used) and the course-specific model's parameters.
            if report.ctx is not None and settings_snapshot is not None:
                X, y, _ = panel_rows(report.ctx.panel, settings_snapshot)
                save_course_features(s, course_id, run_id, "topic", X, y, report.ctx.panel.T, synthetic)
            lr = report.final_outputs.get("logistic")
            if lr is not None:
                register_course_model(s, course_id, run_id, "logistic",
                                      {"coefficients": lr.info.get("coefficients"), "intercept": lr.info.get("intercept"),
                                       "prior_precision": lr.info.get("prior_precision")},
                                      {"train_rows": lr.info.get("train_rows"), "train_exams": lr.info.get("train_exams"),
                                       "shift_from_general": lr.info.get("shift_from_general")})


# ---------------------------------------------------------------------- utilities
def report_topic_id(fam: dict[str, Any], preds) -> int | None:
    col = fam.get("topic_col")
    for p in preds:
        if p.item_index == col:
            return p.item_id
    return None


def _exam_label(e: Exam) -> str:
    return (e.structure or {}).get("label") or (f"{e.year} {e.session}".strip() if e.year else f"Exam {e.id}")


BAD_PAGE_FLAGS = {"low_ocr_confidence": "low OCR confidence", "garbled_symbols": "garbled characters",
                  "garbled_text_layer": "garbled text layer", "needs_ocr_but_unavailable": "scanned page without OCR",
                  "empty_text": "page with no text", "implausible_words": "many implausible words"}


def _paper_quality(session, exam: Exam, src: SourceFile | None, rows: list[ExamQuestion]) -> list[str]:
    """Reasons a paper's counts may be unreliable (shown with every count it contributes to)."""
    reasons = []
    if exam.year is None:
        reasons.append("year unknown; placed by its order in the list, not by date")
    else:
        conf = ((exam.metadata_confidence or {}).get("year") or {}).get("confidence")
        if conf is not None and conf < 0.6 and "year" not in (exam.user_edited_fields or []):
            reasons.append(f"year detected with low confidence ({conf:.0%})")
    if src is not None:
        bad: dict[str, list[int]] = defaultdict(list)
        for page in session.execute(select(DocumentPage).where(DocumentPage.file_id == src.id)).scalars():
            for flag in page.quality_flags or []:
                if flag in BAD_PAGE_FLAGS:
                    bad[flag].append(page.page_no)
        for flag, pages in bad.items():
            reasons.append(f"{BAD_PAGE_FLAGS[flag]} on page {', '.join(str(p) for p in sorted(set(pages)))}")
    review = sum(1 for q in rows if q.is_leaf and q.needs_review)
    if review:
        reasons.append(f"{review} question(s) flagged for review")
    return reasons


def _question_infos(leaves: list[Leaf], recurrence) -> list[QuestionInfo]:
    out = []
    for q in leaves:
        rel, earlier = relation_of(q.id, recurrence.exact_prev, recurrence.para_prev, recurrence.concept_prev)
        out.append(QuestionInfo(
            id=q.id, exam_index=q.exam_index,
            nodes=[(nid, q.ranks.get(nid, i + 1), w) for i, (nid, w) in enumerate(q.counted)],
            uncounted=list(q.uncounted), status=q.status, manual=bool(q.manual), confidence=_mapping_confidence(q),
            labels=list(q.labels), families=list(q.families), format=q.format, marks=q.marks, relation=rel,
            earlier=earlier, family_key=recurrence.family_of.get(q.id), parse_confidence=q.parse_confidence,
            needs_review=q.needs_review))
    return out


def _check_history(history: dict[str, Any], panel: Panel, topic_ids: list[int]) -> None:
    """The history must count exactly what the model saw; a mismatch is a bug, so it is logged loudly."""
    for col, tid in enumerate(topic_ids):
        st = history["topics"].get(str(tid), {})
        papers = int(panel.Y[:, col].sum()) if panel.T else 0
        questions = int(panel.n_questions[:, col].sum()) if panel.T else 0
        if st.get("exam_frequency", 0) != papers or st.get("question_frequency", 0) != questions:
            log.error("topic history differs from the panel", extra={
                "event": "history_mismatch", "topic": tid, "history": [st.get("exam_frequency"),
                                                                      st.get("question_frequency")],
                "panel": [papers, questions]})


def _history_summary(st: dict[str, Any]) -> dict[str, Any]:
    """The compact history shown on a prediction card (the full record is the topic_history artifact)."""
    fams = st.get("families") or []
    return {k: st.get(k) for k in ("usable_papers", "exam_frequency", "question_frequency", "primary_questions",
                                   "secondary_questions", "secondary_only_papers", "last_label", "last_year",
                                   "recent_window", "recent_hits", "marks", "repetition", "provisional",
                                   "provisional_reasons")} | {
        "top_format": {"family": fams[0]["family"], "display": fams[0]["display"], "papers": fams[0]["papers"],
                       "questions": fams[0]["questions"]} if fams else None,
        "labels": (st.get("labels") or [])[:6]}


def _attach_illustrative(guide: dict[str, Any], forms: list[dict[str, Any]]) -> None:
    """Attach the verified practice question that matches the guide's format (and its alternatives)."""
    if not guide:
        return

    def pick(family: str) -> dict[str, Any] | None:
        want = FAMILY_GENERATOR_FORMAT.get(family)
        for f in forms:
            if want and f.get("kind") == want and f.get("basis") == "template":
                return {"text": f["text"], "basis": "generated", "format": family,
                        "evidence_question_ids": f.get("evidence_question_ids", []),
                        "marks_low": f.get("marks_low"), "marks_high": f.get("marks_high"),
                        "note": "Illustrative practice question built from the syllabus wording and this course's own "
                                "phrasing. It passed the syllabus, topic, semantic and question-type checks. It is not "
                                "a prediction of the exact wording."}
        return None

    if guide.get("illustrative") is None:
        guide["illustrative"] = pick(guide.get("family", ""))
        if guide["illustrative"] is None:
            guide["illustrative_note"] = ("No practice question in this format passed the grounding checks; use the "
                                          "description and the past questions instead.")
    for alt in guide.get("alternatives", []):
        if not alt.get("illustrative"):
            alt["illustrative"] = pick(alt["family"])


def _paper_scores(preds, n: int) -> np.ndarray:
    out = np.zeros(n)
    for p in preds:
        out[p.item_index] = p.probability if p.probability is not None else max(p.relative_score, 0.01) ** 2
    return out


def _main_marks(rows: list[ExamQuestion]) -> float | None:
    mains = [r for r in rows if r.parent_id is None]
    seen, total = set(), 0.0
    for m in mains:
        if m.marks is None:
            continue
        if m.or_group:
            if m.or_group in seen:
                continue
            seen.add(m.or_group)
        total += m.marks
    return total or None


def _static_features(tree: TopicTree, item_ids: list[int], unit_ids: list[int]) -> dict[str, np.ndarray]:
    K = len(item_ids)
    hours = np.full(K, np.nan)
    marks_w = np.full(K, np.nan)
    breadth = np.zeros(K)
    siblings: dict[int, int] = Counter(tree.unit_of(i) for i in item_ids)
    for k, nid in enumerate(item_ids):
        node = tree.nodes[nid]
        if node.hours is not None:
            hours[k] = node.hours
        else:
            unit = tree.nodes[tree.unit_of(nid)]
            if unit.hours is not None:
                hours[k] = unit.hours / max(siblings[unit.id], 1)
        unit = tree.nodes[tree.unit_of(nid)]
        mw = node.marks_weight if node.marks_weight is not None else unit.marks_weight
        if mw is not None:
            marks_w[k] = mw / (1 if node.marks_weight is not None else max(siblings[unit.id], 1))
        breadth[k] = math.log1p(len(tree.descendants(nid)) + len(node.concepts))
    out = {"breadth": breadth}
    for name, arr in (("hours_share", hours), ("marks_weight_share", marks_w)):
        if np.isfinite(arr).sum() >= max(1, K // 2):
            filled = np.where(np.isfinite(arr), arr, np.nanmean(arr))
            out[name] = filled / filled.sum() * K / 10.0
        else:
            out[name] = np.zeros(K)
    return out


def _is_lab(tree: TopicTree, nid: int) -> bool:
    return any("lab" in tree.nodes[i].kinds for i in [nid, *tree.ancestors(nid)])


def _location(tree: TopicTree, nid: int) -> str:
    path = tree.path_label(nid)
    refs = tree.nodes[nid].source_refs
    if refs:
        r = refs[0]
        return f"{path} ({r.get('file')}, page {r.get('page')})"
    return path


def _overall_uncertainty(levels: list[str]) -> str:
    """Prediction uncertainty of the top of the ranking: the most common level among topics."""
    if not levels:
        return "High"
    counts = Counter(levels)
    return max(("High", "Medium", "Low"), key=lambda lv: (counts.get(lv, 0), lv == "High"))


def _evidence_quality(preds) -> str:
    """Overall evidence quality: the typical evidence strength of the topics ranked in the top half."""
    top = [p.evidence_strength for p in preds[: max(1, len(preds) // 2)] if p.evidence_strength]
    if not top:
        return "Minimal"
    order = ["Minimal", "Limited", "Moderate", "Strong"]
    return order[int(np.median([order.index(s) for s in top]))]


def _mapping_confidence(q: Leaf) -> float:
    if q.manual:
        return 1.0
    if q.alignment is None or not q.alignment.matches:
        return 0.0
    return float(q.alignment.matches[0].confidence)


def _semantic_mass(q: Leaf, items: dict[int, float], node_col: np.ndarray | None, K: int,
                   temperature: float) -> dict[int, float]:
    """Spread one unit of evidence of an in-syllabus question over the layer's items by similarity."""
    r = q.alignment
    if r is None or r.score_vector is None or node_col is None or K == 0:
        return {c: 1.0 / len(items) for c in items} if items else {}
    best = np.full(K, -np.inf)
    valid = node_col >= 0
    if not valid.any():
        return {c: 1.0 / len(items) for c in items} if items else {}
    np.maximum.at(best, node_col[valid], r.score_vector[valid].astype(float))
    cols = np.flatnonzero(np.isfinite(best))
    z = (best[cols] - best[cols].max()) / max(temperature, 1e-6)
    p = np.exp(z)
    p /= p.sum()
    keep = {int(c): float(x) for c, x in zip(cols, p) if x >= 0.01}
    total = sum(keep.values()) or 1.0
    return {c: x / total for c, x in keep.items()}


def _item_similarity(tree: TopicTree, item_ids: list[int], vectors) -> np.ndarray | None:
    """Syllabus-only similarity between items, re-centred so unrelated items score about 0."""
    if vectors is None or len(item_ids) < 2:
        return None
    try:
        V = np.asarray(vectors.encode([tree.document(i) for i in item_ids]), dtype=float)
    except Exception:  # pragma: no cover - similarity is optional
        return None
    n = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(n == 0, 1.0, n)
    S = V @ V.T
    off = S[~np.eye(len(item_ids), dtype=bool)]
    S = np.clip(S - float(np.median(off)), 0.0, None)
    np.fill_diagonal(S, 0.0)
    return S


def _format_prior(tree: TopicTree, item_ids: list[int]) -> np.ndarray | None:
    """Question formats suggested by syllabus tags (numerical, derivation, ...) of each item's subtree."""
    F = np.zeros((len(item_ids), len(FORMATS)))
    for k, nid in enumerate(item_ids):
        kinds = set(tree.nodes[nid].kinds)
        for d in tree.descendants(nid):
            kinds |= set(tree.nodes[d].kinds)
        for kind in kinds:
            f = KIND_FORMATS.get(kind)
            if f:
                F[k, FORMATS.index(f)] = 1.0
    return F if F.any() else None


def _public_info(info: dict[str, Any]) -> dict[str, Any]:
    """Final-forecast info of a model without large or non-JSON objects."""
    out = {}
    for k, v in (info or {}).items():
        if k in ("posterior", "coefficients", "summary", "displays"):
            continue
        out[k] = v
    return out


def _ensemble_summary(report: BacktestReport) -> dict[str, Any]:
    out = report.final_outputs.get("ensemble")
    if out is None:
        return {}
    info = out.info
    displays = info.get("displays", {})
    rows = [{"model": m, "display": displays.get(m, m), "weight": w, "reliability": info.get("reliability", {}).get(m),
             "skill": info.get("skill", {}).get(m), "raw_gain": info.get("raw_gain", {}).get(m),
             "folds": info.get("skill_folds", {}).get(m)} for m, w in sorted(info.get("weights", {}).items(),
                                                                            key=lambda kv: -kv[1])]
    return {"members": rows, "excluded": [{"model": m, "display": displays.get(m, m), "reason": r}
                                          for m, r in info.get("excluded", {}).items()],
            "exams": info.get("exams"), "weight_folds": info.get("weight_folds"),
            "is_final": report.selected == "ensemble"}


def _layer_summary(rep: BacktestReport) -> dict[str, Any]:
    sel = rep.models.get(rep.selected)
    return {"selected": rep.selected, "k": rep.k,
            "recall": _clean({"v": sel.mean.get("recall") if sel else None})["v"],
            "ndcg": _clean({"v": sel.mean.get("ndcg") if sel else None})["v"],
            "random_recall": _clean({"v": rep.models["random"].mean.get("recall") if "random" in rep.models
                                     and rep.models["random"].mean else None})["v"]}


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _fingerprint(leaves: list[Leaf], topic_ids: list[int], settings) -> str:
    h = hashlib.sha256()
    for q in leaves:
        h.update(f"{q.id}|{q.exam_index}|{q.text}|{q.marks}|{q.manual}".encode("utf-8"))
    h.update(json.dumps(topic_ids).encode())
    h.update(json.dumps(settings.as_dict(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]
