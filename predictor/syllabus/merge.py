"""Merge several parsed course-content documents into one canonical topic tree.

The document with the most structure becomes the base. Nodes from the other documents are
matched by title similarity (and section number when both have one) under an already
matched parent first, then anywhere in the tree. Matches enrich the base node (concepts,
hours, description, source references); unmatched nodes are added and flagged so you can
place them in the Syllabus editor.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .parser import ParsedSyllabus, SyllabusNode


def _norm_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\b(?:unit|chapter|module)\s*[ivx\d]+\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def title_similarity(a: SyllabusNode, b: SyllabusNode) -> float:
    na, nb = _norm_title(a.title), _norm_title(b.title)
    if not na or not nb:
        return 0.0
    score = max(fuzz.token_set_ratio(na, nb), fuzz.ratio(na, nb))
    # token_set_ratio is 100 when one title is a subset of the other; temper that for very short titles.
    if min(len(na.split()), len(nb.split())) == 1 and na != nb:
        score = min(score, fuzz.ratio(na, nb) + 10)
    if a.number and b.number and a.number == b.number:
        score += 6
    return float(min(score, 100.0))


def _merge_into(base: SyllabusNode, other: SyllabusNode) -> None:
    for c in other.concepts:
        if c.lower() not in {x.lower() for x in base.concepts}:
            base.concepts.append(c)
    for d in other.description:
        if d not in base.description:
            base.description.append(d)
    for o in other.objectives:
        if o not in base.objectives:
            base.objectives.append(o)
    if base.hours is None and other.hours is not None:
        base.hours = other.hours
    if base.marks_weight is None and other.marks_weight is not None:
        base.marks_weight = other.marks_weight
    base.kinds = sorted(set(base.kinds) | set(other.kinds))
    base.source_refs.extend(other.source_refs)


def _best_match(node: SyllabusNode, candidates: list[SyllabusNode], threshold: float) -> SyllabusNode | None:
    best, best_score = None, threshold
    for cand in candidates:
        score = title_similarity(node, cand)
        if score >= best_score:
            best, best_score = cand, score
    return best


def merge_syllabi(docs: list[ParsedSyllabus], threshold: float = 86.0) -> ParsedSyllabus:
    docs = [d for d in docs if d.roots]
    if not docs:
        return ParsedSyllabus(roots=[], warnings=["No syllabus structure found in any document."])
    docs = sorted(docs, key=lambda d: len(d.all_nodes()), reverse=True)
    base = docs[0]
    merged = ParsedSyllabus(roots=list(base.roots), course_title=base.course_title,
                            course_code=base.course_code, objectives=list(base.objectives),
                            warnings=list(base.warnings))
    for other in docs[1:]:
        merged.objectives.extend(o for o in other.objectives if o not in merged.objectives)
        merged.course_title = merged.course_title or other.course_title
        merged.course_code = merged.course_code or other.course_code

        def place(node: SyllabusNode, matched_parent: SyllabusNode | None) -> None:
            scope = matched_parent.children if matched_parent is not None else merged.roots
            target = _best_match(node, scope, threshold)
            if target is None:
                target = _best_match(node, merged.all_nodes(), threshold + 4)
            if target is not None:
                _merge_into(target, node)
                for child in node.children:
                    place(child, target)
                return
            new = SyllabusNode(title=node.title, depth=1, number=node.number, description=list(node.description),
                               concepts=list(node.concepts), objectives=list(node.objectives), hours=node.hours,
                               marks_weight=node.marks_weight, kinds=list(node.kinds),
                               source_refs=list(node.source_refs))
            if matched_parent is not None:
                matched_parent.add_child(new)
                new.depth = matched_parent.depth + 1
            else:
                merged.roots.append(new)
                file = node.source_refs[0]["file"] if node.source_refs else "another document"
                merged.warnings.append(f"'{node.title}' from {file} did not match an existing unit and was added "
                                       f"at the top level. Review its position in the Syllabus editor.")
            for child in node.children:
                place(child, new)

        for root in other.roots:
            place(root, None)
    _fix_depths(merged.roots, 1)
    return merged


def _fix_depths(nodes: list[SyllabusNode], depth: int) -> None:
    for n in nodes:
        n.depth = depth
        _fix_depths(n.children, depth + 1)
