"""Question -> syllabus alignment with in/out-of-syllabus status (spec sections 3 and 11).

Score = semantic_weight * cosine(question, topic) + keyword_weight * keyword coverage.

* Keyword coverage is the IDF-weighted share of the question's content terms (stems, with
  instruction words such as "derive" or "explain" removed) that appear in the topic text.
* Unknown-term ratio is the IDF-weighted share of content terms that appear nowhere in the
  syllabus (after a fuzzy check for OCR misspellings). A weak match with many unknown terms
  is strong evidence that the question is outside the syllabus.

Status:
  A clearly in    score >= clearly_in and (keyword coverage high or similarity very high)
  B probably in   score >= probably_in
  C uncertain     everything between
  D outside       score < outside, or below probably_in with mostly unknown terms
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz, process

from ..config.settings import Settings
from ..embeddings.backends import EmbeddingBackend
from ..parsing.question_types import UNIT_RE
from ..preprocessing.textnorm import content_terms, normalize_math, stem, tokenize
from .tree import TopicTree

GENERIC_WORDS_FILE = Path(__file__).resolve().parent.parent / "config" / "generic_words.txt"


def load_generic_stems(data_dir: str | Path | None = None) -> frozenset[str]:
    """Stems of everyday words that are no evidence for or against syllabus membership."""
    words: set[str] = set()
    files = [GENERIC_WORDS_FILE]
    if data_dir:
        files.append(Path(data_dir) / "generic_words.txt")
    for f in files:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    words.add(stem(line))
    return frozenset(words)


@dataclass
class QuestionItem:
    id: int
    text: str
    context: str = ""


@dataclass
class TopicMatch:
    topic_id: int
    rank: int
    score: float
    semantic: float
    keyword: float
    status: str
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    evidence_text: str = ""


@dataclass
class AlignmentResult:
    question_id: int
    status: str
    matches: list[TopicMatch]
    unknown_terms: list[str]
    unknown_ratio: float
    reason: str
    best_score: float = 0.0


def thresholds_for(settings: Settings, backend_kind: str) -> dict[str, float]:
    table = settings.alignment.thresholds
    key = "neural" if backend_kind == "neural" else "tfidf"
    t = table[key]
    return {"clearly_in": float(t.clearly_in), "probably_in": float(t.probably_in), "outside": float(t.outside)}


def match_confidence(score: float, th: dict[str, float]) -> float:
    """Monotone 0-1 'syllabus match strength' derived from the thresholds (not a probability)."""
    o, p, c = th["outside"], th["probably_in"], th["clearly_in"]
    if score <= o:
        return max(0.0, 0.25 * score / max(o, 1e-9))
    if score <= p:
        return 0.25 + 0.25 * (score - o) / max(p - o, 1e-9)
    if score <= c:
        return 0.5 + 0.25 * (score - p) / max(c - p, 1e-9)
    return min(1.0, 0.75 + 0.25 * (score - c) / 0.3)


def counts_for_prediction(status: str, score: float, method: str, settings: Settings, backend_kind: str) -> bool:
    """Whether a mapping may influence predictions under the configured strictness."""
    if method == "manual":
        return status in ("A", "B")
    strictness = float(settings.alignment.strictness)
    th = thresholds_for(settings, backend_kind)
    if status == "A":
        return True
    if status == "B":
        if strictness >= 1.0:
            return False
        shift = (th["clearly_in"] - th["probably_in"]) * max(0.0, strictness - 0.5) * 2.0
        return score >= th["probably_in"] + shift
    if status == "C":
        return strictness <= 0.25
    return False


class SyllabusAligner:
    def __init__(self, settings: Settings, tree: TopicTree, backend: EmbeddingBackend):
        self.settings = settings
        self.tree = tree
        self.backend = backend
        cfg = settings.alignment
        self.w_sem = float(cfg.semantic_weight)
        self.w_kw = float(cfg.keyword_weight)
        self.instruction = list(cfg.instruction_words)
        self.th = thresholds_for(settings, backend.kind)
        self.min_cov_clear = float(cfg.evidence.min_keyword_coverage_clear)
        self.outside_unknown = float(cfg.evidence.outside_unknown_term_ratio)
        self.multi_ratio = float(cfg.multi_topic_ratio)
        self.max_topics = int(cfg.max_topics_per_question)
        self.lab_penalty = float(cfg.lab_penalty)
        self.fb_weight = float(cfg.feedback_weight)
        self.fb_min = int(cfg.feedback_min_questions)
        self.fb_margin = float(cfg.feedback_margin)
        self.generic = load_generic_stems(settings.app.data_dir)

        self.node_ids = tree.mappable_ids()
        docs = [tree.document(i) for i in self.node_ids]
        self.backend.fit([tree.document(i) for i in tree.nodes] or docs)
        self.node_vecs = self.backend.encode(docs) if docs else np.zeros((0, 1), dtype=np.float32)
        # Term statistics over all syllabus nodes.
        self.node_terms: list[set[str]] = []
        df: Counter[str] = Counter()
        for nid in tree.nodes:
            df.update(set(content_terms(tree.term_text(nid), self.instruction)))
        n_docs = max(len(tree.nodes), 1)
        self.idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}
        self.df = dict(df)
        self.max_idf = math.log(1 + n_docs) + 1.0
        for nid in self.node_ids:
            terms = set(content_terms(tree.term_text(nid), self.instruction))
            for anc in tree.ancestors(nid)[:1]:
                terms |= set(content_terms(tree.nodes[anc].title, self.instruction))
            self.node_terms.append(terms)
        self.vocab = sorted(self.idf)
        self.is_lab = np.array([("lab" in tree.nodes[nid].kinds) or any(
            "lab" in tree.nodes[a].kinds for a in tree.ancestors(nid)) for nid in self.node_ids], dtype=bool)
        self.base_vecs = self.node_vecs.copy()
        self.base_terms = [set(t) for t in self.node_terms]
        self.feedback: dict[int, list[int]] = {}

    def _idf(self, term: str) -> float:
        return self.idf.get(term, self.max_idf)

    def _known(self, term: str) -> bool:
        if term in self.idf:
            return True
        if len(term) >= 5 and self.vocab:
            hit = process.extractOne(term, self.vocab, scorer=fuzz.ratio, score_cutoff=86)
            return hit is not None
        return False

    def align(self, questions: list[QuestionItem], *, feedback: bool = True) -> list[AlignmentResult]:
        if not questions:
            return []
        if not self.node_ids:
            return [AlignmentResult(q.id, "C", [], [], 0.0, "The syllabus has no topics yet.") for q in questions]
        leaf_vecs = self.backend.encode([q.text or q.context for q in questions])
        ctx_texts = [q.context if q.context and q.context != q.text else (q.text or "") for q in questions]
        ctx_vecs = self.backend.encode(ctx_texts)
        qvecs = 0.7 * leaf_vecs + 0.3 * ctx_vecs
        norms = np.linalg.norm(qvecs, axis=1, keepdims=True)
        qvecs = qvecs / np.where(norms == 0, 1.0, norms)
        self.node_vecs, self.node_terms, self.feedback = self.base_vecs.copy(), [set(t) for t in self.base_terms], {}
        results = [self._align_one(q, qvecs[i] @ self.node_vecs.T) for i, q in enumerate(questions)]
        if feedback and self.fb_weight > 0 and self._apply_feedback(questions, qvecs, results):
            results = [self._align_one(q, qvecs[i] @ self.node_vecs.T) for i, q in enumerate(questions)]
            for r in results:
                for m in r.matches[:1]:
                    if m.topic_id in self.feedback:
                        r.reason += (f" Topic vocabulary includes terms from {len(self.feedback[m.topic_id])} "
                                     f"confidently mapped question(s).")
        return results

    def _apply_feedback(self, questions: list[QuestionItem], qvecs: np.ndarray,
                        results: list[AlignmentResult]) -> bool:
        """Rocchio-style expansion of topic vectors and term sets from status-A questions."""
        by_node: dict[int, list[int]] = {}
        for i, r in enumerate(results):
            if r.status != "A" or not r.matches:
                continue
            second = r.matches[1].score if len(r.matches) > 1 else 0.0
            if r.matches[0].score - second < self.fb_margin:
                continue
            by_node.setdefault(r.matches[0].topic_id, []).append(i)
        changed = False
        index_of = {nid: j for j, nid in enumerate(self.node_ids)}
        for nid, idxs in by_node.items():
            if len(idxs) < self.fb_min:
                continue
            j = index_of[nid]
            centroid = qvecs[idxs].mean(axis=0)
            vec = self.node_vecs[j] + self.fb_weight * centroid
            self.node_vecs[j] = vec / max(np.linalg.norm(vec), 1e-9)
            counts: Counter[str] = Counter()
            for i in idxs:
                q = questions[i]
                counts.update(set(content_terms(f"{q.context} {q.text}", self.instruction)))
            self.node_terms[j] |= {t for t, c in counts.items() if c >= 2}
            self.feedback[nid] = [questions[i].id for i in idxs]
            changed = True
        return changed

    def _align_one(self, q: QuestionItem, sims: np.ndarray) -> AlignmentResult:
        text = f"{q.context} {q.text}" if q.context and q.context != q.text else q.text
        terms = content_terms(text, self.instruction)
        unique_terms = list(dict.fromkeys(terms))
        total_idf = sum(self._idf(t) for t in unique_terms)
        # Out-of-syllabus evidence uses technical terms only (everyday words are ignored).
        technical = [t for t in unique_terms if t not in self.generic]
        unknown = [t for t in technical if not self._known(t)]
        tech_idf = sum(self._idf(t) for t in technical)
        unknown_ratio = (sum(self._idf(t) for t in unknown) / tech_idf) if tech_idf else 0.0
        numerical = len(UNIT_RE.findall(text)) >= 2
        if numerical and len(unknown) < 2:
            unknown_ratio = min(unknown_ratio, self.outside_unknown - 0.01)

        coverage = np.zeros(len(self.node_ids))
        if total_idf:
            for j, node_terms in enumerate(self.node_terms):
                shared = [t for t in unique_terms if t in node_terms]
                if shared:
                    coverage[j] = sum(self._idf(t) for t in shared) / total_idf
        scores = self.w_sem * np.clip(sims, 0.0, 1.0) + self.w_kw * coverage
        if self.is_lab.any() and not LAB_CUE.search(text):
            scores = np.where(self.is_lab, scores * self.lab_penalty, scores)
        order = np.argsort(-scores)
        best = int(order[0])
        best_score = float(scores[best])

        if not unique_terms:
            status, reason = "C", "The question has no content words to match (check the text or OCR)."
        else:
            # A shared distinctive term (used by at most two syllabus nodes, e.g. "weir") rules out
            # an out-of-syllabus verdict based on unknown terms alone.
            distinctive = any(t in self.node_terms[best] and self.df.get(t, 99) <= 2 for t in unique_terms)
            status, reason = self._status(best_score, float(sims[best]), float(coverage[best]),
                                          0.0 if distinctive else unknown_ratio)

        matches: list[TopicMatch] = []
        chosen: list[int] = []
        surface = _surface_forms(text)
        for rank_pos, j in enumerate(order[: max(self.max_topics * 4, 4)]):
            j = int(j)
            score = float(scores[j])
            nid = self.node_ids[j]
            if matches:
                if len(matches) >= self.max_topics or score < max(self.multi_ratio * best_score, self.th["probably_in"]):
                    break
                related = any(nid in self.tree.ancestors(c) or c in self.tree.ancestors(nid) for c in chosen)
                if related:
                    continue
            m_status = status if not matches else self._status(score, float(sims[j]), float(coverage[j]),
                                                                unknown_ratio)[0]
            shared = [t for t in unique_terms if t in self.node_terms[j]]
            node = self.tree.nodes[nid]
            evidence = node.title
            if node.concepts:
                evidence += ": " + ", ".join(node.concepts[:8])
            matches.append(TopicMatch(
                topic_id=nid, rank=len(matches) + 1, score=round(score, 4), semantic=round(float(sims[j]), 4),
                keyword=round(float(coverage[j]), 4), status=m_status,
                confidence=round(match_confidence(score, self.th), 3),
                matched_terms=[surface.get(t, t) for t in shared][:12], evidence_text=evidence[:400]))
            chosen.append(nid)
        return AlignmentResult(q.id, status, matches, [surface.get(t, t) for t in unknown][:12],
                               round(unknown_ratio, 3), reason, round(best_score, 4))

    def _status(self, score: float, sim: float, cov: float, unknown_ratio: float) -> tuple[str, str]:
        th = self.th
        if score >= th["clearly_in"] and (cov >= self.min_cov_clear or sim >= th["clearly_in"] + 0.15):
            return "A", "Strong semantic and keyword match with the syllabus."
        if score < th["outside"]:
            return "D", "No syllabus topic is similar to this question."
        if score < th["probably_in"] and unknown_ratio >= self.outside_unknown:
            return "D", "Most of the question's technical terms do not appear anywhere in the syllabus."
        if score >= th["probably_in"]:
            if score >= th["clearly_in"]:
                return "B", "High similarity but limited shared terminology; check the mapping."
            return "B", "Reasonable match with some ambiguity."
        return "C", "Weak match: not enough evidence to decide."


def _surface_forms(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in tokenize(normalize_math(text)):
        st = stem(tok)
        out.setdefault(st, tok)
    return out


def describe_status(status: str) -> str:
    return {"A": "Clearly in syllabus", "B": "Probably in syllabus", "C": "Uncertain",
            "D": "Outside syllabus"}.get(status, status)


LAB_CUE = re.compile(r"\b(?:experiment\w*|apparatus|laborator\w*|lab|procedure|observation\w*|calibrat\w*|"
                     r"verif\w*|practical)\b", re.IGNORECASE)
