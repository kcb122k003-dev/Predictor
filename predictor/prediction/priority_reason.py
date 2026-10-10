"""One plain sentence saying why a topic has its priority, built only from recorded facts.

Every clause is backed by a number shown elsewhere (appearances, recent appearances, syllabus
match, past formats). A topic that never appeared is not described as unlikely: the sentence
says there is no historical evidence and that its position comes from the syllabus and
cross-course patterns.
"""

from __future__ import annotations

from typing import Any


def priority_word(category: str) -> str:
    """'Very High Priority' -> 'Very high'."""
    word = (category or "").replace(" Priority", "").strip()
    if word == "Extremely High":  # runs made before the rename
        word = "Very High"
    return word[:1] + word[1:].lower() if word else ""


def priority_reason(*, category: str, rank: int, topics: int, stats: dict[str, Any], facts: dict[str, Any],
                    guide: dict[str, Any] | None) -> str:
    T = stats.get("usable_papers", 0)
    a = stats.get("exam_frequency", 0)
    head = f"{priority_word(category)} priority (rank {rank} of {topics})"
    if T == 0:
        return (f"{head}: no past papers have been supplied, so the position comes only from the syllabus structure "
                f"and patterns learned across courses.")
    if a == 0:
        return (f"{head}: the topic did not appear in any of the {T} usable paper{'s' if T != 1 else ''}, so its "
                f"position comes from the syllabus structure, related questions and patterns learned across courses. "
                f"Missing from past papers is not evidence that it will not be examined.")
    reasons = []
    w, r = stats.get("recent_window", 0), stats.get("recent_hits", 0)
    hist = f"appeared in {a} of {T} usable paper{'s' if T != 1 else ''}"
    if w and T > w:
        hist += f", including {r} of the last {w}"
    if stats.get("last_label"):
        hist += f" (most recently {stats['last_label']})"
    reasons.append(hist)
    mc = facts.get("mapping_confidence")
    if mc is not None and mc >= 0.75:
        reasons.append("its past questions match the syllabus entry strongly")
    elif mc is not None and mc < 0.55:
        reasons.append("its past questions match the syllabus entry only weakly")
    # The question format barely affects the rank, so it is not given as a reason for the priority (the format guide
    # describes it separately).
    sec = stats.get("secondary_only_papers", 0)
    text = f"{head} because it " + _join(reasons) + "."
    if sec:
        text += (f" In {sec} of those papers it was only the secondary topic of a question that mainly tested "
                 f"another topic.")
    return text


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]
