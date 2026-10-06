"""Historical papers -> prediction, on the synthetic demo course with planted patterns."""

import re

import numpy as np
from sqlalchemy import select

from predictor.database.models import (AnalysisArtifact, CourseTopic, Exam, ExamQuestion, PredictedQuestion,
                                       Prediction, QuestionTopicMapping)
from predictor.demo.bank import TOPIC_NUMBERS
from predictor.export.service import ExportService


def _topics_by_number(s, cid):
    return {t.number: t for t in s.execute(select(CourseTopic).where(CourseTopic.course_id == cid)).scalars()}


def test_ingestion_detects_duplicates(analysed_demo):
    out = analysed_demo["outcomes"]
    copy = next(v for k, v in out.items() if k.endswith("_copy.pdf"))
    assert copy["duplicate"] is True
    retyped = next(v for k, v in out.items() if k.endswith("_retyped.txt"))
    assert retyped["duplicate_of"] is not None  # same paper, different file: excluded from analysis
    assert analysed_demo["summary"]["exams"] == 12


def test_alignment_accuracy_and_out_of_syllabus(analysed_demo):
    app, cid, truth = analysed_demo["app"], analysed_demo["course_id"], analysed_demo["truth"]
    with app.db.session() as s:
        topics = _topics_by_number(s, cid)
        id_to_number = {t.id: t.number for t in topics.values()}
        correct = total = 0
        oos_statuses = []
        for exam in s.execute(select(Exam).where(Exam.course_id == cid, Exam.include_in_analysis.is_(True))).scalars():
            leaves = s.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam.id, ExamQuestion.is_leaf.is_(True))
                               .order_by(ExamQuestion.order_no)).scalars().all()
            expected = truth["questions"][str(exam.year)]
            assert len(leaves) == len(expected)
            for q, exp in zip(leaves, expected):
                primary = next((m for m in q.mappings if m.rank == 1), None)
                if exp.get("topic") is None:
                    oos_statuses.append(primary.status if primary else "none")
                    continue
                total += 1
                got = id_to_number.get(primary.topic_id) if primary else None
                correct += got == TOPIC_NUMBERS[exp["topic"]]
    assert correct / total >= 0.85, f"alignment accuracy {correct}/{total}"
    assert oos_statuses and all(st == "D" for st in oos_statuses)


def test_predictions_respect_the_syllabus(analysed_demo):
    app, run_id, cid = analysed_demo["app"], analysed_demo["run_id"], analysed_demo["course_id"]
    with app.db.session() as s:
        topic_ids = {t.id for t in s.execute(select(CourseTopic).where(CourseTopic.course_id == cid)).scalars()}
        preds = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "topic")).scalars().all()
        assert preds and all(p.topic_id in topic_ids for p in preds)
        forms = s.execute(select(PredictedQuestion).where(PredictedQuestion.run_id == run_id)).scalars().all()
        assert forms
        for f in forms:
            assert f.grounding["grounded"], f.text
            assert not re.search(r"turbine|pelton|kaplan|impeller|centrifugal pump", f.text, re.IGNORECASE)
        excluded = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == run_id,
                                                            AnalysisArtifact.key == "excluded")).scalars().one()
        labels = " ".join(g["label"] for g in excluded.data["groups"]).lower()
        assert "turbine" in labels and "pump" in labels


def test_models_beat_baselines_and_leakage_audit(analysed_demo):
    summary = analysed_demo["summary"]
    assert summary["leakage_audit"]["passed"]
    app, run_id = analysed_demo["app"], analysed_demo["run_id"]
    from predictor.services.results_service import ResultsService

    models = {m["name"]: m for m in ResultsService(app).models(run_id)["models"] if m["layer"] == "topic"}
    selected = next(m for m in models.values() if m["selected"])
    assert selected["metrics"]["ndcg"] > models["random"]["metrics"]["ndcg"] + 0.15
    # Small-data protection: with 12 papers the tree models stay off and say why.
    assert not models["random_forest"]["enabled"] and models["random_forest"]["gate_reason"]


def test_planted_patterns_are_recovered(analysed_demo):
    app, run_id, cid = analysed_demo["app"], analysed_demo["run_id"], analysed_demo["course_id"]
    with app.db.session() as s:
        topics = _topics_by_number(s, cid)
        preds = {p.topic_id: p for p in s.execute(select(Prediction).where(Prediction.run_id == run_id,
                                                                             Prediction.layer == "topic")).scalars()}
    n = len(preds)

    def rank(key):
        return preds[topics[TOPIC_NUMBERS[key]].id].rank

    # Emerging topics appear only in recent papers. Recency must lift them above where all-time frequency
    # alone would put them; the fading topic (frequent early, absent recently) drops to the bottom half.
    appearances = {tid: p.features["facts"]["appearances"] for tid, p in preds.items()}

    def frequency_rank(key):
        mine = appearances[topics[TOPIC_NUMBERS[key]].id]
        return 1 + sum(1 for v in appearances.values() if v > mine)

    assert rank("boundary_layer") <= n / 2
    assert rank("separation_drag") < frequency_rank("separation_drag")
    assert rank("orifices_notches") > n / 2
    # Core topics are near the top.
    assert np.mean([rank(k) for k in ("bernoulli", "darcy", "viscosity", "continuity")]) <= 6


def test_evidence_and_why_not_present(analysed_demo):
    app, run_id = analysed_demo["app"], analysed_demo["run_id"]
    with app.db.session() as s:
        preds = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "topic")
                          .order_by(Prediction.rank)).scalars().all()
    assert all(p.evidence["lines"] for p in preds)
    assert any(p.why_not for p in preds[len(preds) // 2:])
    if preds[0].calibrated:
        assert preds[0].prob_low <= preds[0].probability <= preds[0].prob_high


def test_manual_edits_survive_reanalysis(analysed_demo, tmp_path):
    """Manual mappings and edited text are kept when the analysis runs again (run on a copy-free path)."""
    from predictor.services.analysis_service import AnalysisService
    from predictor.services.review_service import ReviewService

    app, cid = analysed_demo["app"], analysed_demo["course_id"]
    review = ReviewService(app)
    with app.db.session() as s:
        q = s.execute(select(ExamQuestion).join(Exam).where(Exam.course_id == cid, Exam.include_in_analysis.is_(True),
                                                             ExamQuestion.is_leaf.is_(True))).scalars().first()
        target = s.execute(select(CourseTopic).where(CourseTopic.course_id == cid, CourseTopic.number == "7.2")).scalars().one()
        qid, tid = q.id, target.id
    review.set_mapping(qid, [tid], "A")
    review.update_question(qid, {"text": "Explain boundary layer separation and its control."})
    service = AnalysisService(app)
    service.run(service.create_run(cid))
    with app.db.session() as s:
        maps = s.execute(select(QuestionTopicMapping).where(QuestionTopicMapping.question_id == qid)).scalars().all()
        assert [(m.topic_id, m.method) for m in maps] == [(tid, "manual")]
        assert s.get(ExamQuestion, qid).text.startswith("Explain boundary layer separation")


def test_exports(analysed_demo):
    service = ExportService(analysed_demo["app"])
    run_id = analysed_demo["run_id"]
    assert service.pdf(run_id).startswith(b"%PDF")
    assert service.xlsx(run_id)[:2] == b"PK"
    csv_text = service.csv(run_id, "predictions").decode("utf-8-sig")
    assert csv_text.splitlines()[0].startswith("rank,topic,category")
    assert b"PREDICTED QUESTION FORMULATION" in service.csv(run_id, "questions")
