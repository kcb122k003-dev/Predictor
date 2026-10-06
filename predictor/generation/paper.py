"""Optional predicted-paper simulation (spec section 49).

Builds hypothetical papers from the discovered template (number of main questions, parts
per question, short-notes question, OR alternatives), the topic ranking, the unit balance
seen historically and the grounded formulations. Paper A takes the highest-ranked topics;
papers B and C sample other plausible combinations. None is the real exam.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DISCLAIMER = ("Hypothetical paper built from historical patterns. It is not the actual exam and should be used "
              "for practice and prioritising revision only.")


def _choose_topics(scores: np.ndarray, units: np.ndarray, n: int, unit_cap: int, rng: np.random.Generator | None,
                   penalty: dict[int, float], temperature: float = 0.35) -> list[int]:
    base = np.clip(scores, 1e-6, None)
    adj = np.array([base[i] * penalty.get(i, 1.0) for i in range(len(base))])
    if rng is not None:
        gumbel = -np.log(-np.log(rng.uniform(1e-9, 1 - 1e-9, size=len(adj))))
        key = np.log(adj) / max(temperature, 1e-6) + gumbel
    else:
        key = adj
    chosen: list[int] = []
    per_unit: dict[int, int] = {}
    for i in np.argsort(-key):
        u = int(units[i])
        if per_unit.get(u, 0) >= unit_cap:
            continue
        chosen.append(int(i))
        per_unit[u] = per_unit.get(u, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


def simulate_papers(*, topic_ids: list[int], topic_labels: list[str], scores: np.ndarray, units: np.ndarray,
                    unit_labels: list[str], typical: dict[str, Any], formulations: dict[int, list[dict[str, Any]]],
                    type_forecast: dict[int, dict[str, Any]], max_unit_share: float, variants: int = 3,
                    seed: int = 7) -> list[dict[str, Any]]:
    n_main = int(typical.get("main_questions") or 6)
    parts = max(1, int(typical.get("parts_per_question") or 2))
    short_notes = bool(typical.get("short_notes_question"))
    n_or = int(typical.get("or_alternatives") or 0)
    marks_per_q = typical.get("marks_per_question")
    regular_q = n_main - (1 if short_notes else 0)
    n_slots = max(1, regular_q * parts + (3 if short_notes else 0) + n_or)
    n_slots = min(n_slots, len(topic_ids))
    unit_cap = max(1, math.ceil(max_unit_share * n_slots) + 1)
    rng = np.random.default_rng(seed)
    papers: list[dict[str, Any]] = []
    used_before: dict[int, float] = {}
    # Historical format mix to aim for (numerical / derivation / theory items per paper).
    format_targets = {"numerical": int(typical.get("numerical_leaves") or 0),
                      "derivation": int(typical.get("derivation_leaves") or 0)}
    for v in range(variants):
        chosen = _choose_topics(scores, units, n_slots, unit_cap, None if v == 0 else rng, used_before)
        for i in chosen:
            used_before[i] = used_before.get(i, 1.0) * 0.6
        regular = chosen[: regular_q * parts]
        notes = chosen[regular_q * parts: regular_q * parts + (3 if short_notes else 0)]
        alts = chosen[regular_q * parts + len(notes):]
        regular.sort(key=lambda i: (int(units[i]), -scores[i]))
        questions = []
        qno = 0
        remaining = dict(format_targets)
        for start in range(0, len(regular), parts):
            qno += 1
            group = regular[start:start + parts]
            items = []
            for j, i in enumerate(group):
                items.append(_item(i, j, v, topic_ids, topic_labels, formulations, type_forecast,
                                   (marks_per_q / len(group)) if marks_per_q else None, remaining))
            q = {"number": str(qno), "parts": items}
            if alts and qno == max(1, regular_q // 2):
                i = alts.pop(0)
                q["or_alternative"] = _item(i, 0, v, topic_ids, topic_labels, formulations, type_forecast, marks_per_q)
            questions.append(q)
        if notes:
            qno += 1
            questions.append({"number": str(qno), "short_notes": True,
                              "instruction": "Write short notes on (any two):",
                              "parts": [{"label": r, "topic_id": topic_ids[i], "topic": topic_labels[i],
                                         "text": _short_note(i, topic_labels, formulations, topic_ids)}
                                        for r, i in zip(("i", "ii", "iii"), notes)]})
        papers.append({
            "title": f"Predicted Paper {chr(ord('A') + v)} (hypothetical)",
            "disclaimer": DISCLAIMER,
            "strategy": "Highest-ranked topics with unit balance" if v == 0 else
                        "Alternative plausible combination (sampled by score, avoiding Paper A's choices)",
            "questions": questions,
            "topics_covered": [topic_labels[i] for i in chosen],
            "units_covered": sorted({unit_labels[int(units[i])] for i in chosen}),
        })
    return papers


def _short_note(i: int, topic_labels, formulations, topic_ids) -> str:
    """A short-note item names the topic itself, without its section number."""
    label = topic_labels[i]
    return label.split(" ", 1)[1] if label[:1].isdigit() and " " in label else label


def _item(i: int, j: int, variant: int, topic_ids, topic_labels, formulations, type_forecast, marks,
          remaining: dict[str, int] | None = None) -> dict[str, Any]:
    tid = topic_ids[i]
    forms = formulations.get(tid) or []
    form = None
    if forms:
        preferred = (type_forecast.get(tid) or {}).get("format")
        ordered = forms[variant % len(forms):] + forms[:variant % len(forms)]
        # Fill formats the historical papers usually contain (numerical, derivation) first.
        for f in ordered:
            if remaining and remaining.get(f["format"], 0) > 0:
                form = f
                remaining[f["format"]] -= 1
                break
        if form is None:
            form = next((f for f in ordered if f["format"] == preferred), ordered[0])
    text = form["text"] if form else f"Question on {topic_labels[i]} (no grounded formulation available)."
    out = {"label": chr(ord("a") + j), "topic_id": tid, "topic": topic_labels[i], "text": text,
           "format": form["format"] if form else (type_forecast.get(tid, {}).get("format") or "theory"),
           "basis": form["basis"] if form else "topic only"}
    if form and form.get("marks_low") is not None:
        out["marks"] = form["marks_high"] if form["marks_high"] == form["marks_low"] else \
            f"{form['marks_low']:g}-{form['marks_high']:g}"
    elif marks:
        out["marks"] = round(float(marks))
    return out
