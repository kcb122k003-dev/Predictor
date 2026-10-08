"""The exam x item panel that every temporal feature and model reads.

Rows are exams in time order, columns are items (topics, concepts or units). The only
way models see history is through ``Panel.until(t)``, which returns a copy holding rows
``0..t-1``. Nothing computed from that copy can depend on exam ``t`` or later, which is how
leakage is prevented by construction (spec section 45).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

FORMATS = ["definition", "theory", "derivation", "numerical", "diagram", "objective", "mixed"]


@dataclass
class ExamInfo:
    exam_id: int
    order: float
    label: str
    total_marks: float
    year: int | None = None
    syllabus_version: str = "current"


@dataclass
class QuestionRecord:
    """A leaf question as it enters the panel."""

    question_id: int
    exam_index: int
    text: str
    marks: float | None
    format: str
    types: list[str]
    items: list[tuple[int, float]]  # (item column, weight) for mappings that count
    concept_node: int | None = None
    topic_node: int | None = None
    status: str = "A"
    soft: dict[int, float] = field(default_factory=dict)  # item column -> similarity
    semantic: dict[int, float] = field(default_factory=dict)  # item column -> share of this question (sums to 1)
    exact_repeat: bool = False
    para_repeat: bool = False
    mapping_confidence: float = 1.0
    parse_confidence: float = 1.0
    optional: bool = False  # optional / OR-alternative question


@dataclass
class Panel:
    exams: list[ExamInfo]
    item_ids: list[int]
    item_labels: list[str]
    Y: np.ndarray  # (T, K) 0/1 appearance
    marks: np.ndarray  # (T, K) marks attributed to the item
    soft: np.ndarray  # (T, K) best semantic similarity of any question in the exam
    formats: np.ndarray  # (T, K, F) question counts by format
    n_questions: np.ndarray  # (T, K)
    exact_repeat: np.ndarray  # (T, K) questions that repeat an earlier question exactly
    para_repeat: np.ndarray  # (T, K) questions that paraphrase an earlier question
    static: dict[str, np.ndarray] = field(default_factory=dict)  # (K,) syllabus properties
    # (T, K) semantic evidence: each in-syllabus question spreads one unit of evidence over the topics
    # by similarity (pretrained/hybrid alignment scores), so near-misses also count a little. Per paper
    # and item the value is 1 - prod(1 - share), at most 1: exam-level, not inflated by long papers.
    semantic: np.ndarray | None = None
    item_sim: np.ndarray | None = None  # (K, K) syllabus-only similarity between items (no exam text)
    format_prior: np.ndarray | None = None  # (K, F) format tendency from syllabus tags (no exam text)
    # (T, K, 4) sums over the questions behind each cell: mapping confidence, parse/OCR confidence,
    # marks known, optional/OR question (divide by n_questions for means).
    quality: np.ndarray | None = None
    item_unit: np.ndarray | None = None  # (K,) unit column index per item (for unit features)
    unit_ids: list[int] = field(default_factory=list)
    layer: str = "topic"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.exams)

    @property
    def K(self) -> int:
        return len(self.item_ids)

    def until(self, t: int) -> "Panel":
        """History strictly before exam index ``t``."""
        t = max(0, min(t, self.T))
        return replace(
            self,
            exams=self.exams[:t],
            Y=self.Y[:t].copy(), marks=self.marks[:t].copy(), soft=self.soft[:t].copy(),
            formats=self.formats[:t].copy(), n_questions=self.n_questions[:t].copy(),
            exact_repeat=self.exact_repeat[:t].copy(), para_repeat=self.para_repeat[:t].copy(),
            semantic=self.semantic_matrix()[:t].copy(),
            quality=self.quality[:t].copy() if self.quality is not None else None,
            meta=dict(self.meta),
        )

    def semantic_matrix(self) -> np.ndarray:
        return self.semantic if self.semantic is not None else np.zeros_like(self.Y)

    def drop_exam(self, j: int) -> "Panel":
        """The panel without exam ``j`` (used for leave-one-exam-out uncertainty)."""
        keep = [i for i in range(self.T) if i != j]
        return replace(
            self, exams=[self.exams[i] for i in keep], Y=self.Y[keep].copy(), marks=self.marks[keep].copy(),
            soft=self.soft[keep].copy(), formats=self.formats[keep].copy(), n_questions=self.n_questions[keep].copy(),
            exact_repeat=self.exact_repeat[keep].copy(), para_repeat=self.para_repeat[keep].copy(),
            semantic=self.semantic_matrix()[keep].copy(),
            quality=self.quality[keep].copy() if self.quality is not None else None, meta=dict(self.meta))

    def total_marks(self) -> np.ndarray:
        return np.array([max(e.total_marks, 1e-9) for e in self.exams], dtype=float)

    def items_per_exam(self) -> np.ndarray:
        return self.Y.sum(axis=1) if self.T else np.zeros(0)


def build_panel(exams: list[ExamInfo], item_ids: list[int], item_labels: list[str],
                questions: list[QuestionRecord], *, static: dict[str, np.ndarray] | None = None,
                item_unit: np.ndarray | None = None, unit_ids: list[int] | None = None,
                layer: str = "topic", item_sim: np.ndarray | None = None,
                format_prior: np.ndarray | None = None) -> Panel:
    T, K, F = len(exams), len(item_ids), len(FORMATS)
    Y = np.zeros((T, K))
    marks = np.zeros((T, K))
    soft = np.zeros((T, K))
    formats = np.zeros((T, K, F))
    nq = np.zeros((T, K))
    exact = np.zeros((T, K))
    para = np.zeros((T, K))
    semantic = np.zeros((T, K))
    quality = np.zeros((T, K, 4))
    f_index = {f: i for i, f in enumerate(FORMATS)}
    for q in questions:
        t = q.exam_index
        if not (0 <= t < T):
            continue
        for col, sim in q.soft.items():
            if 0 <= col < K:
                soft[t, col] = max(soft[t, col], sim)
        for col, mass in q.semantic.items():
            if 0 <= col < K:
                # Stored as sum of log(1 - mass); converted below to 1 - prod(1 - mass): the chance that at least
                # one question of the paper was on the item. A paper counts at most once, however long it is.
                semantic[t, col] += np.log1p(-min(float(mass), 0.999))
        total_w = sum(w for _, w in q.items) or 1.0
        for col, w in q.items:
            if not (0 <= col < K):
                continue
            Y[t, col] = 1.0
            nq[t, col] += 1
            if q.marks is not None:
                marks[t, col] += q.marks * w / total_w
            formats[t, col, f_index.get(q.format, f_index["theory"])] += 1
            quality[t, col] += (q.mapping_confidence, q.parse_confidence, 1.0 if q.marks is not None else 0.0,
                                1.0 if q.optional else 0.0)
            if q.exact_repeat:
                exact[t, col] += 1
            if q.para_repeat:
                para[t, col] += 1
    semantic = 1.0 - np.exp(semantic)
    return Panel(exams=exams, item_ids=list(item_ids), item_labels=list(item_labels), Y=Y, marks=marks,
                 soft=soft, formats=formats, n_questions=nq, exact_repeat=exact, para_repeat=para,
                 static=static or {}, item_unit=item_unit, unit_ids=unit_ids or [], layer=layer, semantic=semantic,
                 item_sim=item_sim, format_prior=format_prior, quality=quality)
