"""SQLAlchemy ORM models.

See docs/ARCHITECTURE.md section 6 for how these map to the entities named in the
specification. JSON columns hold small structured values (flags, evidence, metrics).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Course(Base):
    __tablename__ = "course"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    code: Mapped[str] = mapped_column(String(60), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    files: Mapped[list["SourceFile"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    exams: Mapped[list["Exam"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    topics: Mapped[list["CourseTopic"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    syllabus_versions: Mapped[list["SyllabusVersion"]] = relationship(
        back_populates="course", cascade="all, delete-orphan")
    runs: Mapped[list["AnalysisRun"]] = relationship(back_populates="course", cascade="all, delete-orphan")


class SyllabusVersion(Base):
    __tablename__ = "syllabus_version"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(120), default="current")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    # Exams with order_index >= this value were set under this syllabus (None = unknown).
    effective_from_order: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    course: Mapped[Course] = relationship(back_populates="syllabus_versions")


class SourceFile(Base):
    __tablename__ = "source_file"
    __table_args__ = (UniqueConstraint("course_id", "sha256", name="uq_course_file_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # exam | syllabus
    filename: Mapped[str] = mapped_column(String(400))
    sha256: Mapped[str] = mapped_column(String(64))
    mime: Mapped[str] = mapped_column(String(100), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    stored_path: Mapped[str] = mapped_column(String(800))
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending|processing|done|error
    error: Mapped[str] = mapped_column(Text, default="")
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    extraction_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    syllabus_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("syllabus_version.id", ondelete="SET NULL"), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    course: Mapped[Course] = relationship(back_populates="files")
    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="file", cascade="all, delete-orphan", order_by="DocumentPage.page_no")


class DocumentPage(Base):
    __tablename__ = "document_page"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("source_file.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)  # 1-based
    text: Mapped[str] = mapped_column(Text, default="")
    method: Mapped[str] = mapped_column(String(20), default="native")  # native|ocr|docx|text
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    file: Mapped[SourceFile] = relationship(back_populates="pages")


class CourseTopic(Base):
    """A node of the canonical syllabus tree (unit, topic or concept)."""

    __tablename__ = "course_topic"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), index=True)
    syllabus_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("syllabus_version.id", ondelete="CASCADE"), nullable=True, index=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("course_topic.id", ondelete="CASCADE"), nullable=True, index=True)
    depth: Mapped[int] = mapped_column(Integer, default=1)
    number: Mapped[str] = mapped_column(String(40), default="")
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, default="")
    concepts: Mapped[list[Any]] = mapped_column(JSON, default=list)
    objectives: Mapped[list[Any]] = mapped_column(JSON, default=list)
    hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    marks_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    kinds: Mapped[list[Any]] = mapped_column(JSON, default=list)  # numerical, derivation, theory, lab...
    aliases: Mapped[list[Any]] = mapped_column(JSON, default=list)
    source_refs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    order_no: Mapped[int] = mapped_column(Integer, default=0)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    user_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    course: Mapped[Course] = relationship(back_populates="topics")
    parent: Mapped[Optional["CourseTopic"]] = relationship(remote_side="CourseTopic.id", back_populates="children")
    children: Mapped[list["CourseTopic"]] = relationship(
        back_populates="parent", cascade="all, delete-orphan", order_by="CourseTopic.order_no")


class Exam(Base):
    __tablename__ = "exam"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), index=True)
    source_file_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("source_file.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(400), default="")
    subject: Mapped[str] = mapped_column(String(400), default="")
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    calendar: Mapped[str] = mapped_column(String(10), default="AD")  # AD | BS | unknown
    session: Mapped[str] = mapped_column(String(60), default="")
    exam_type: Mapped[str] = mapped_column(String(60), default="")
    exam_date: Mapped[str] = mapped_column(String(40), default="")
    order_index: Mapped[float] = mapped_column(Float, default=0.0)
    full_marks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pass_marks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration: Mapped[str] = mapped_column(String(40), default="")
    examiner: Mapped[str] = mapped_column(String(200), default="")
    instructions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    structure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_confidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    include_in_analysis: Mapped[bool] = mapped_column(Boolean, default=True)
    exclusion_reason: Mapped[str] = mapped_column(String(400), default="")
    duplicate_of_id: Mapped[Optional[int]] = mapped_column(ForeignKey("exam.id", ondelete="SET NULL"), nullable=True)
    user_edited_fields: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    course: Mapped[Course] = relationship(back_populates="exams")
    sections: Mapped[list["ExamSection"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", order_by="ExamSection.order_no")
    questions: Mapped[list["ExamQuestion"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", order_by="ExamQuestion.order_no")


class ExamSection(Base):
    __tablename__ = "exam_section"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exam.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(60), default="")
    title: Mapped[str] = mapped_column(String(300), default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    attempt_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    order_no: Mapped[int] = mapped_column(Integer, default=0)

    exam: Mapped[Exam] = relationship(back_populates="sections")


class ExamQuestion(Base):
    """A node of an exam's question tree (main question, sub-question, sub-sub-question)."""

    __tablename__ = "exam_question"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exam.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("exam_section.id", ondelete="SET NULL"), nullable=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("exam_question.id", ondelete="CASCADE"), nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(20), default="")
    path_label: Mapped[str] = mapped_column(String(60), default="")
    depth: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text, default="")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    normalized_text: Mapped[str] = mapped_column(Text, default="")
    context_text: Mapped[str] = mapped_column(Text, default="")
    marks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    marks_source: Mapped[str] = mapped_column(String(30), default="")
    or_group: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    is_optional: Mapped[bool] = mapped_column(Boolean, default=False)
    is_leaf: Mapped[bool] = mapped_column(Boolean, default=True)
    order_no: Mapped[int] = mapped_column(Integer, default=0)
    page_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    line_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    question_types: Mapped[list[Any]] = mapped_column(JSON, default=list)
    type_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    type_user_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    options: Mapped[list[Any]] = mapped_column(JSON, default=list)
    equations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    quality_flags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    parse_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    user_edited: Mapped[bool] = mapped_column(Boolean, default=False)

    exam: Mapped[Exam] = relationship(back_populates="questions")
    mappings: Mapped[list["QuestionTopicMapping"]] = relationship(
        back_populates="question", cascade="all, delete-orphan", order_by="QuestionTopicMapping.rank")


class QuestionTopicMapping(Base):
    __tablename__ = "question_topic_mapping"
    __table_args__ = (Index("ix_mapping_topic", "topic_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("exam_question.id", ondelete="CASCADE"), index=True)
    topic_id: Mapped[Optional[int]] = mapped_column(ForeignKey("course_topic.id", ondelete="CASCADE"), nullable=True)
    rank: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(2), default="C")  # A | B | C | D
    semantic_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    keyword_overlap: Mapped[float] = mapped_column(Float, default=0.0)
    unknown_term_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    matched_terms: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence_text: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    method: Mapped[str] = mapped_column(String(10), default="auto")  # auto | manual

    question: Mapped[ExamQuestion] = relationship(back_populates="mappings")
    topic: Mapped[Optional[CourseTopic]] = relationship()


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    backend: Mapped[str] = mapped_column(String(120))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)


class AnalysisRun(Base):
    __tablename__ = "analysis_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    course: Mapped[Course] = relationship(back_populates="runs")
    artifacts: Mapped[list["AnalysisArtifact"]] = relationship(cascade="all, delete-orphan")
    model_results: Mapped[list["ModelResult"]] = relationship(cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(cascade="all, delete-orphan", order_by="Prediction.rank")
    predicted_questions: Mapped[list["PredictedQuestion"]] = relationship(cascade="all, delete-orphan")
    folds: Mapped[list["BacktestFold"]] = relationship(cascade="all, delete-orphan")


class AnalysisArtifact(Base):
    __tablename__ = "analysis_artifact"
    __table_args__ = (UniqueConstraint("run_id", "key", name="uq_artifact_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True)
    key: Mapped[str] = mapped_column(String(80))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelResult(Base):
    __tablename__ = "model_result"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True)
    layer: Mapped[str] = mapped_column(String(20), default="topic")
    model_name: Mapped[str] = mapped_column(String(60))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    family: Mapped[str] = mapped_column(String(30), default="")
    complexity: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    gate_reason: Mapped[str] = mapped_column(Text, default="")
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metric_se: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")


class BacktestFold(Base):
    __tablename__ = "backtest_fold"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True)
    layer: Mapped[str] = mapped_column(String(20), default="topic")
    model_name: Mapped[str] = mapped_column(String(60))
    target_exam_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_index: Mapped[int] = mapped_column(Integer)
    target_label: Mapped[str] = mapped_column(String(120), default="")
    n_train_exams: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Prediction(Base):
    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True)
    layer: Mapped[str] = mapped_column(String(20), default="topic")  # topic | concept | family | unit
    topic_id: Mapped[Optional[int]] = mapped_column(ForeignKey("course_topic.id", ondelete="SET NULL"), nullable=True)
    item_key: Mapped[str] = mapped_column(String(80), default="")
    label: Mapped[str] = mapped_column(String(500), default="")
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    category: Mapped[str] = mapped_column(String(30), default="")
    confidence: Mapped[str] = mapped_column(String(20), default="")
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    contributions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    why_not: Mapped[list[Any]] = mapped_column(JSON, default=list)


class PredictedQuestion(Base):
    __tablename__ = "predicted_question"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_run.id", ondelete="CASCADE"), index=True)
    topic_id: Mapped[Optional[int]] = mapped_column(ForeignKey("course_topic.id", ondelete="SET NULL"), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(60), default="")
    marks_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    marks_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    basis: Mapped[str] = mapped_column(String(40), default="template")  # template | historical_variant
    rank: Mapped[int] = mapped_column(Integer, default=0)
    evidence_question_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    grounding: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
