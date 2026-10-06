"""Question recurrence: exact repeats, paraphrases, concept and topic recurrence (spec 9, 14, 38).

Four separate signals, never collapsed early:

* exact     similarity >= exact_similarity AND token-set string ratio >= exact_string_ratio
* paraphrase similarity >= paraphrase_similarity (or string ratio >= paraphrase_string_ratio)
              AND the same syllabus topic; not exact
* concept   same deepest syllabus node, or similarity >= concept_similarity within the topic
* topic     same topic, different concept

Similarity uses either the neural backend or, by default, character n-gram vectors with no
fitted statistics. No corpus statistics means pairwise similarities do not depend on which
exams exist, so recurrence flags computed for exam t only use exams before t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from ..config.settings import Settings
from ..embeddings.backends import EmbeddingBackend, stateless_char_vectors
from ..preprocessing.textnorm import content_terms, normalize_for_matching


@dataclass
class FamilyQuestion:
    question_id: int
    exam_index: int
    text: str
    topic: int | None
    concept: int | None


@dataclass
class RecurrenceResult:
    family_of: dict[int, str]  # question id -> family key (exact + paraphrase components)
    families: dict[str, list[int]]  # family key -> question ids
    exact_prev: dict[int, list[int]]  # question id -> earlier questions it repeats exactly
    para_prev: dict[int, list[int]]  # question id -> earlier questions it paraphrases
    concept_prev: dict[int, list[int]]  # question id -> earlier questions testing the same concept
    pairs: list[tuple[int, int, str, float, float]] = field(default_factory=list)  # (q1, q2, kind, sim, ratio)


def similarity_matrix(texts: list[str], backend: EmbeddingBackend | None = None) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))
    if backend is not None and backend.kind == "neural":
        vecs = backend.encode(texts)
        return np.clip(vecs @ vecs.T, -1.0, 1.0)
    char = stateless_char_vectors(texts)
    sim_char = (char @ char.T).toarray()
    term_sets = [set(content_terms(t)) for t in texts]
    jac = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(i, len(texts)):
            a, b = term_sets[i], term_sets[j]
            val = len(a & b) / len(a | b) if a and b else 0.0
            jac[i, j] = jac[j, i] = val
    return 0.5 * sim_char + 0.5 * jac


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_recurrence(questions: list[FamilyQuestion], settings: Settings,
                    backend: EmbeddingBackend | None = None) -> RecurrenceResult:
    cfg = settings.recurrence
    n = len(questions)
    texts = [q.text for q in questions]
    sims = similarity_matrix(texts, backend)
    norm = [normalize_for_matching(t) for t in texts]
    uf = _UnionFind([q.question_id for q in questions])
    exact_prev: dict[int, list[int]] = {q.question_id: [] for q in questions}
    para_prev: dict[int, list[int]] = {q.question_id: [] for q in questions}
    concept_prev: dict[int, list[int]] = {q.question_id: [] for q in questions}
    pairs: list[tuple[int, int, str, float, float]] = []
    for i in range(n):
        qi = questions[i]
        for j in range(i + 1, n):
            qj = questions[j]
            sim = float(sims[i, j])
            if sim < min(cfg.concept_similarity, cfg.paraphrase_similarity) and (qi.concept is None or qi.concept != qj.concept):
                continue
            ratio = float(fuzz.token_set_ratio(norm[i], norm[j]))
            same_topic = qi.topic is not None and qi.topic == qj.topic
            same_concept = qi.concept is not None and qi.concept == qj.concept
            kind = ""
            if sim >= cfg.exact_similarity and ratio >= cfg.exact_string_ratio:
                kind = "exact"
            elif same_topic and (sim >= cfg.paraphrase_similarity or ratio >= cfg.paraphrase_string_ratio):
                kind = "paraphrase"
            elif same_concept or (same_topic and sim >= cfg.concept_similarity):
                kind = "concept"
            if not kind:
                continue
            pairs.append((qi.question_id, qj.question_id, kind, round(sim, 3), round(ratio, 1)))
            if kind in ("exact", "paraphrase"):
                uf.union(qi.question_id, qj.question_id)
            # Earlier/later relation by exam order (same-exam pairs are alternatives, not repeats).
            if qi.exam_index == qj.exam_index:
                continue
            early, late = (qi, qj) if qi.exam_index < qj.exam_index else (qj, qi)
            target = {"exact": exact_prev, "paraphrase": para_prev, "concept": concept_prev}[kind]
            target[late.question_id].append(early.question_id)
    families: dict[str, list[int]] = {}
    family_of: dict[int, str] = {}
    for q in questions:
        root = uf.find(q.question_id)
        key = f"F{root}"
        families.setdefault(key, []).append(q.question_id)
        family_of[q.question_id] = key
    return RecurrenceResult(family_of, families, exact_prev, para_prev, concept_prev, pairs)


def families_before(result: RecurrenceResult, exam_index_of: dict[int, int], t: int) -> dict[str, list[int]]:
    """Families rebuilt from pairs whose questions both come from exams before ``t``."""
    uf = _UnionFind([q for q, e in exam_index_of.items() if e < t])
    for a, b, kind, _, _ in result.pairs:
        if kind in ("exact", "paraphrase") and exam_index_of.get(a, t) < t and exam_index_of.get(b, t) < t:
            uf.union(a, b)
    out: dict[str, list[int]] = {}
    for q in uf.parent:
        out.setdefault(f"F{uf.find(q)}", []).append(q)
    return out
