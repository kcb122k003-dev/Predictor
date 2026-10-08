"""Checks every predicted question formulation must pass before it is shown.

1. Syllabus verification: every content word appears in the topic's syllabus subtree or in
   its past questions (``check_grounding`` in questions.py).
2. Topic verification: aligning the formulation to the whole syllabus puts the intended topic
   among its top two matches, with an in-syllabus status.
3. Semantic consistency: among all syllabus topics, the pretrained model ranks the intended
   topic among the closest to the formulation (top 3, or the top 15% of topics in large
   syllabi). A relative check is used because short template questions have lower absolute
   similarity than full past questions, so an absolute cut-off rejected valid ones.
4. Question-type compatibility: the question-type classifier reads the formulation as the
   intended format (theory and diagram are treated as compatible).

A formulation that fails any check is dropped; the counts are reported with the run.
Formulations are never stored as exam questions and never count as history.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..syllabus.alignment import QuestionItem

COMPATIBLE = {"theory": {"theory", "diagram", "definition"}, "diagram": {"diagram", "theory"},
              "definition": {"definition", "theory"}, "derivation": {"derivation"}, "numerical": {"numerical"},
              "objective": {"objective"}, "mixed": {"mixed", "theory", "numerical", "derivation"}}


class FormulationVerifier:
    def __init__(self, tree, aligner, classifier, pretrained=None):
        self.tree = tree
        self.aligner = aligner
        self.classifier = classifier
        self.pretrained = pretrained
        self.rejected: dict[str, int] = {"topic": 0, "semantic": 0, "type": 0}
        self.checked = 0
        self.examples: list[dict[str, Any]] = []
        self.topic_ids = tree.topic_ids()
        self.max_rank = max(3, int(np.ceil(0.15 * len(self.topic_ids))))
        self._topic_mat = None
        if pretrained is not None and self.topic_ids:
            self._topic_mat = pretrained.encode([tree.document(t) for t in self.topic_ids])

    def _semantic_rank(self, text: str, topic_id: int) -> tuple[int, float]:
        sims = self._topic_mat @ self.pretrained.encode([text])[0]
        target = self.topic_ids.index(topic_id)
        return int((sims > sims[target]).sum()) + 1, float(sims[target])

    def check(self, text: str, topic_id: int, fmt: str, basis: str = "template") -> dict[str, Any]:
        self.checked += 1
        out: dict[str, Any] = {"syllabus": True}
        r = self.aligner.align([QuestionItem(-1, text, text)], feedback=False)[0]
        tops = [self.tree.topic_of(m.topic_id) for m in r.matches[:2]]
        target = self.tree.topic_of(topic_id)
        out["topic"] = bool(target in tops and r.status in ("A", "B"))
        out["topic_detail"] = {"status": r.status, "top_topics": [self.tree.nodes[t].title for t in tops]}
        if basis == "historical_variant":
            # A verbatim past question is real exam text: its semantic fit is the aligner's verdict above.
            out["semantic"] = out["topic"]
            out["semantic_detail"] = {"basis": basis}
        elif self._topic_mat is not None and self.tree.topic_of(topic_id) in self.topic_ids:
            rank, sim = self._semantic_rank(text, self.tree.topic_of(topic_id))
            out["semantic"] = bool(rank <= self.max_rank)
            out["semantic_detail"] = {"rank": rank, "max_rank": self.max_rank, "similarity": round(sim, 3),
                                      "topics": len(self.topic_ids)}
        else:
            out["semantic"] = bool(r.matches and r.matches[0].score >= self.aligner.th["probably_in"])
            out["semantic_detail"] = {"score": round(float(r.best_score), 3), "threshold": self.aligner.th["probably_in"]}
        got = self.classifier.classify(text).format
        out["type"] = bool(got in COMPATIBLE.get(fmt, {fmt}))
        out["type_detail"] = {"intended": fmt, "classified": got}
        out["passed"] = bool(out["topic"] and out["semantic"] and out["type"])
        for key in ("topic", "semantic", "type"):
            if not out[key]:
                self.rejected[key] += 1
        if not out["passed"] and len(self.examples) < 12:
            self.examples.append({"text": text, "topic": self.tree.nodes[topic_id].title, "format": fmt,
                                  "failed": [k for k in ("topic", "semantic", "type") if not out[k]],
                                  "semantic": out.get("semantic_detail"), "topic_check": out.get("topic_detail")})
        return out

    def summary(self) -> dict[str, Any]:
        return {"checked": self.checked, "rejected": dict(self.rejected), "examples": self.examples,
                "semantic_max_rank": self.max_rank if self._topic_mat is not None else None}
