"""Question-format labels shown to students (Syllabus Explorer and Predictions).

The classifier in ``question_types.py`` produces fine types (definition, derivation,
numerical, ...) and one coarse format per question for the models. Students need the
instruction a question gives ("state", "explain", "discuss", "prove", ...), so this module
adds a presentation layer on top of the classifier's types plus the question's own verbs:

* ``format_labels(text, types, marks, context)`` returns every label that applies (a question
  can be, for example, both "Diagram-based" and "Explain"); labels never change model inputs.
* ``FAMILIES`` groups labels into the formats a student prepares for (numerical problem,
  derivation or proof, conceptual explanation, ...). A question can belong to several
  families, so family counts overlap; the UI says so.
"""

from __future__ import annotations

import re

# label id -> display name, in display order
LABELS: dict[str, str] = {
    "definition": "Definition",
    "state_list": "State or list",
    "explain": "Explain",
    "discuss": "Discuss",
    "derive": "Derive",
    "prove": "Prove",
    "compare": "Compare and contrast",
    "numerical": "Numerical calculation",
    "problem_solving": "Problem solving",
    "diagram": "Diagram-based",
    "example": "Explain with an example",
    "application": "Application of a theory",
    "design_analysis": "Design or analytical problem",
    "short_answer": "Short answer",
    "long_form": "Long-form theoretical",
    "objective": "Objective (MCQ, true/false, fill in)",
    "unclassified": "Unclassified",
}

# Fine classifier type -> labels it implies.
TYPE_LABELS: dict[str, tuple[str, ...]] = {
    "definition": ("definition",),
    "derivation": ("derive",),
    "numerical": ("numerical",),
    "problem_solving": ("problem_solving",),
    "design": ("design_analysis",),
    "analytical_reasoning": ("design_analysis",),
    "compare_contrast": ("compare",),
    "diagram": ("diagram",),
    "explain_with_example": ("example",),
    "application": ("application",),
    "conceptual_explanation": ("explain",),
    "short_answer": ("short_answer",),
    "long_theory": ("long_form",),
    "mcq": ("objective",),
    "true_false": ("objective",),
    "fill_blank": ("objective",),
}

# Instruction verbs read from the question itself (the classifier folds several of them into one type).
VERB_LABELS: list[tuple[str, re.Pattern]] = [
    ("state_list", re.compile(r"(?:^|[.;:?]\s*)(?:state|list|enumerate|name|mention|write down)\b", re.I)),
    ("explain", re.compile(r"\b(?:explain|describe|outline|what (?:is|are) the (?:significance|importance|principle|role))\b", re.I)),
    ("discuss", re.compile(r"\b(?:discuss|elaborate|critically|comment on)\b", re.I)),
    ("prove", re.compile(r"\b(?:prove that|show that)\b", re.I)),
    ("derive", re.compile(r"\b(?:derive|derivation|deduce|obtain (?:an |the )?(?:expression|relation|equation|formula))\b", re.I)),
]

# Formats a student prepares for. Each family lists the labels that place a question in it.
FAMILIES: dict[str, dict[str, object]] = {
    "numerical": {"display": "Numerical problem", "labels": ("numerical", "problem_solving")},
    "derivation": {"display": "Derivation or proof", "labels": ("derive", "prove")},
    "explanation": {"display": "Conceptual explanation", "labels": ("explain", "discuss", "example", "application",
                                                                   "long_form")},
    "definition": {"display": "Definition or statement", "labels": ("definition", "state_list")},
    "diagram": {"display": "Diagram-based explanation", "labels": ("diagram",)},
    "compare": {"display": "Compare and contrast", "labels": ("compare",)},
    "design": {"display": "Design or analytical problem", "labels": ("design_analysis",)},
    "short_note": {"display": "Short note", "labels": ("short_answer",)},
    "objective": {"display": "Objective question", "labels": ("objective",)},
}
# Tie-break when two families appear in the same number of papers: the more specific family first.
FAMILY_ORDER = ["numerical", "derivation", "design", "compare", "diagram", "explanation", "definition", "short_note",
                "objective"]

# Family -> the formulation format the question generator uses for it (None: no generator template).
FAMILY_GENERATOR_FORMAT = {"numerical": "numerical", "derivation": "derivation", "explanation": "theory",
                           "definition": "definition", "diagram": "diagram", "compare": "compare",
                           "short_note": "short_note", "design": "numerical", "objective": None}

# Syllabus "kinds" (from the syllabus line) -> family, for topics with no past question.
KIND_FAMILY = {"numerical": "numerical", "derivation": "derivation", "design": "design", "definition": "definition",
               "diagram": "diagram", "theory": "explanation", "process": "explanation", "method": "explanation"}


def display(label: str) -> str:
    return LABELS.get(label, label.replace("_", " ").capitalize())


def family_display(family: str) -> str:
    return str(FAMILIES.get(family, {}).get("display", family))


def format_labels(text: str, types: list[str] | None, marks: float | None = None, context: str = "") -> list[str]:
    """Every format label that applies to a question, in display order."""
    found: set[str] = set()
    for t in types or []:
        found.update(TYPE_LABELS.get(t, ()))
    body = (text or "").strip()
    for label, pattern in VERB_LABELS:
        if pattern.search(body):
            found.add(label)
    # "Prove that" / "Show that" are proofs, not derivations from first principles, unless both are asked.
    if "prove" in found and not re.search(r"\b(?:derive|deduce|obtain)\b", body, re.I):
        found.discard("derive")
    if "short_answer" not in found and re.search(r"\bshort notes?\b", f"{context} {body}", re.I):
        found.add("short_answer")
    if not found:
        found.add("unclassified")
    return [label for label in LABELS if label in found]


def families_of(labels: list[str]) -> list[str]:
    """Format families of a question (several when it combines formats)."""
    out = []
    for fam in FAMILY_ORDER:
        if any(label in FAMILIES[fam]["labels"] for label in labels):
            out.append(fam)
    # A short note is its own format only when nothing more specific was asked.
    if "short_note" in out and len(out) > 1:
        out.remove("short_note")
    # "Explain with a sketch" is a diagram-based explanation; the plain explanation family is kept as well so the
    # overlap is visible in the counts.
    return out
