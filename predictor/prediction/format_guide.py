"""How a topic is likely to be asked: the "question format guide" shown with every prediction.

For each topic the guide separates four things the student needs and never mixes them:

* the suggested format (numerical problem, derivation, conceptual explanation, ...), chosen by
  how many past papers asked the topic in that format;
* a specific description of that format, built from the topic's own past questions (what is
  calculated from what, what is derived, what is explained) or, without history, from the
  syllabus wording;
* for numerical problems, a general template in which every value is a named placeholder;
* why the format was suggested, with the counts behind it, and at most two alternatives that
  the history also supports.

When the topic has no past question, the format is labelled as inferred (from the syllabus
entry or the course-wide mix), never as historically established. The illustrative practice
question is attached by the analysis run from the verified formulations; for numerical
problems it reuses the values of a real past paper (named) because the app cannot check that
newly invented values give a consistent, solvable problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..parsing.format_labels import FAMILIES, FAMILY_ORDER, KIND_FAMILY, family_display

QUANTITY_NOUNS = {
    "diameter", "length", "width", "breadth", "depth", "height", "thickness", "radius", "area", "velocity", "speed",
    "pressure", "temperature", "discharge", "flow", "rate", "head", "density", "viscosity", "mass", "weight", "force",
    "power", "efficiency", "coefficient", "factor", "gradient", "slope", "angle", "volume", "time", "frequency",
    "concentration", "energy", "work", "heat", "stress", "strain", "load", "torque", "current", "voltage",
    "resistance", "charge", "distance", "elevation", "level", "gravity", "conductivity", "capacity", "duration",
    "span", "spacing", "fraction", "ratio", "porosity", "tension", "displacement", "acceleration", "momentum",
    "enthalpy", "entropy", "flux", "yield", "dose", "size", "cost", "interest", "population", "deflection",
}
UNIT_DIMENSIONS = [
    (r"m3/s|m\^3/s|cumecs?|l/s|lps|litres?/s|liters?/s|litres?/min|liters?/min|l/min|litres? per second|"
     r"liters? per second|m3/min", "flow rate"),
    (r"m/s2|m/s\^2", "acceleration"),
    (r"m/s|km/h|km/hr|kmph|cm/s|mm/s|ft/s", "velocity"),
    (r"kg/m3|kg/m\^3|g/cm3", "density"),
    (r"pa\.s|pa-s|n\.?s/m2|poise|centipoise|cp|stokes?|m2/s", "viscosity"),
    (r"n/m2|kn/m2|n/mm2|kpa|mpa|gpa|pa|bar|atm|mm of mercury|mm of hg|cm of mercury|psi", "pressure"),
    (r"m2|m\^2|cm2|mm2|sq\.? ?m|hectares?|ha", "area"),
    (r"m3|m\^3|cm3|litres?|liters?|ml", "volume"),
    (r"km|cm|mm|m|ft|inch(?:es)?|in", "length"),
    (r"°c|deg ?c|degrees? (?:celsius|centigrade)|k|kelvin|°f", "temperature"),
    (r"kg|g|tonnes?|tons?", "mass"),
    (r"kn|n|newtons?|kgf", "force"),
    (r"kw|mw|w|watts?|hp|horse ?power", "power"),
    (r"kj|mj|j|joules?|kwh|cal|kcal", "energy"),
    (r"rpm|rev/min", "rotational speed"),
    (r"seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|years?", "time"),
    (r"%|percent|per cent", "percentage"),
    (r"degrees?|°", "angle"),
    (r"v|volts?|kv", "voltage"),
    (r"a|amps?|amperes?", "current"),
    (r"ohms?|Ω", "resistance"),
]
_UNIT_RE = "|".join(u for u, _ in UNIT_DIMENSIONS)
NUMBER_UNIT = re.compile(rf"(?<![\w.])(-?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^?\s*-?\d+)?)(?:\s*({_UNIT_RE}))?(?![\w/^])",
                         re.IGNORECASE)
SKIP_BACK = {"of", "is", "as", "at", "be", "was", "are", "were", "being", "equal", "to", "=", ":", "a", "an", "the",
             "by", "with", "having", "has", "its", "take", "taking", "assume", "assuming", "if", "and", "about",
             "approximately", "respectively"}
NUM_VERBS = r"calculate|compute|determine|find(?: out)?|estimate|evaluate|work out|obtain"
OPENERS = {
    "derivation": r"^(?:derive|deduce|obtain|prove|show)\b(?:\s+(?:an?|the)\s+(?:expression|relation(?:ship)?|equation|formula)\s+(?:for|of))?",
    "explanation": r"^(?:with (?:a |the )?(?:help of )?(?:neat |suitable |labelled )?(?:sketch|diagram|figure)s?,?\s*)?"
                   r"(?:explain|describe|discuss|elaborate|outline)\b(?:\s+(?:in detail|briefly|with (?:a |an )?(?:neat )?"
                   r"(?:sketch|diagram|example)))?",
    "definition": r"^(?:define|state|list|enumerate|name|mention|what (?:is|are|do you) (?:meant by|mean by|understand by)|"
                  r"what (?:is|are))\b",
    "diagram": r"^(?:with (?:a |the )?(?:help of )?(?:neat |suitable |labelled )?(?:sketch|diagram|figure)s?,?\s*)?"
               r"(?:draw|sketch|explain|describe)\b(?:\s+(?:and explain|a neat sketch of|the))?",
    "compare": r"^(?:differentiate|distinguish|compare|contrast)\b(?:\s+between)?",
    "short_note": r"^(?:write\s+)?(?:a\s+)?short\s+notes?\s+on\b",
    "design": r"^(?:design|size|select|specify|analy[sz]e|justify)\b",
}
LEAD_VERBS = ("obtain|give|write|mention|name|list|state|outline|classify|enumerate|draw|sketch|illustrate|analy[sz]e|"
              "justify|determine|find|prove|show|discuss|describe|explain|define|derive|compare|differentiate")
FAMILY_CUES = {
    "derivation": r"\b(?:derive|deduce|obtain|prove|show that)\b",
    "explanation": r"\b(?:explain|describe|discuss|elaborate|outline)\b",
    "definition": r"\b(?:define|state|list|enumerate|mention|what (?:is|are) meant|what do you (?:mean|understand))\b",
    "diagram": r"\b(?:sketch|diagram|draw|figure)\b",
    "compare": r"\b(?:differentiate|distinguish|compare|contrast)\b",
    "short_note": r"\bshort notes?\b",
    "design": r"\b(?:design|size|select|specify|analy[sz]e)\b",
}
FRAMES = {
    "derivation": "Derivation or proof: {verb} {obj}, working through each step{assumptions}.",
    "explanation": "Conceptual explanation: {verb} {obj}{sketch}{example}.",
    "definition": "Definition or statement: {verb} {obj}.",
    "diagram": "Diagram-based explanation: {verb} {obj} with a labelled sketch.",
    "compare": "Compare and contrast: {verb} {obj}, point by point.",
    "short_note": "Short note: write a short note on {obj}.",
    "design": "Design or analytical problem: {verb} {obj} from the given requirements or data.",
    "objective": "Objective question on {obj}.",
}
DEFAULT_VERBS = {"derivation": "derive", "explanation": "explain", "definition": "define", "compare": "differentiate between",
                 "design": "design", "diagram": "explain"}


# ------------------------------------------------------------------------- text helpers
def clean_question(text: str) -> str:
    """A past question without its number, marks and OR markers (the wording is otherwise unchanged)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    for _ in range(2):  # "Q3. (a) ..." carries two labels
        t = re.sub(r"^(?:Q\.?\s*)?\d+(?:\.\d+)*\s*(?:\(\s*[a-z]{1,4}\s*\))*\s*[.)]?\s+|^\(?[a-z]\)\s*|"
                   r"^\(?[ivx]{1,4}\)\s*", "", t, flags=re.I)
    t = re.sub(r"\s*[\[(]\s*\d+(?:\s*[+x×]\s*\d+)*\s*(?:marks?)?\s*[\])]\s*$", "", t, flags=re.I)
    t = re.sub(r"\s*\d+\s*marks?\s*$", "", t, flags=re.I)
    t = re.sub(r"^(?:or|OR)\s+", "", t)
    return t.strip()


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return parts[0] if parts else text


def _family_sentence(text: str, family: str) -> str:
    """The sentence that carries the family's instruction ("Define X. With a neat sketch, explain Y." -> the second
    sentence for a diagram-based explanation)."""
    sentences = [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]
    cue = FAMILY_CUES.get(family)
    if cue:
        for sentence in sentences:
            if re.search(cue, sentence, flags=re.I):
                return sentence
    return sentences[0] if sentences else text


def _trim_words(text: str, n: int = 16) -> str:
    words = text.split()
    if len(words) <= n:
        return text
    cut = " ".join(words[:n])
    # Cut at the last comma or conjunction inside the limit, so the phrase stays readable.
    m = re.search(r"^(.*\w)(?:,| and | or | which | that )", cut)
    return (m.group(1) if m and len(m.group(1).split()) >= 4 else cut).rstrip(",;: ")


def object_phrase(text: str, family: str) -> tuple[str, str | None]:
    """(object of the instruction, the verb used) for a past question, e.g. ("the working of a Pelton wheel",
    "explain"). The object never contains the question's numbers."""
    t = _family_sentence(clean_question(text), family).rstrip(" .?!")
    verb = None
    pattern = OPENERS.get(family)
    if pattern:
        m = re.match(pattern, t, flags=re.I)
        if m is None:
            # The instruction can follow a lead-in ("Using the Buckingham pi theorem, derive ...", "State the
            # principle and derive ..."): take the text after the instruction verb.
            m = re.search(r"(?:^|[,;:]\s*|\b(?:and|then|hence|also)\s+)" + pattern.lstrip("^"), t, flags=re.I)
        if m:
            verb = re.sub(r"^[,;:\s]*(?:and|then|hence|also)?\s*", "", m.group(0).strip().lower()).rstrip(",")
            t = t[m.end():]
    if verb is None:
        # The question's own instruction verb when it is not one of the family's usual openers ("Obtain the
        # velocity distribution ..." classified as a long theoretical question).
        m = re.match(rf"^({LEAD_VERBS})\b\s*", t, flags=re.I)
        if m:
            verb, t = m.group(1).lower(), t[m.end():]
    t = re.sub(r"^[\s,:;-]+", "", t)
    t = re.sub(r"[,;]?\s+(?:and|&)\s+(?:hence\s+)?(?:state|explain|discuss|list|mention|write|derive|draw|sketch|"
               r"give|comment|also)\b.*$", "", t, flags=re.I)
    t = re.sub(r"\s+(?:with (?:a |the |an )?(?:neat |suitable |labelled |relevant )?(?:sketch|diagram|figure|example)(?:es|s)?|"
               r"in detail|briefly)$", "", t, flags=re.I)
    t = NUMBER_UNIT.sub("", t) if family != "numerical" else t
    return _trim_words(re.sub(r"\s+", " ", t).strip(" ,;:"), 16), verb


def numeric_template(text: str) -> tuple[str, list[dict[str, str]]]:
    """The question with every numerical value replaced by a named placeholder, e.g.
    "a pipe of 300 mm diameter" -> "a pipe of [diameter in mm]". Returns (template, placeholders)."""
    t = clean_question(text)
    out, pos, placeholders = [], 0, []
    named: list[tuple[str, str]] = []  # (unit, name) of values named by a quantity word in the question
    for m in NUMBER_UNIT.finditer(t):
        value, unit = m.group(1), m.group(2)
        start, end = m.span()
        after = t[end:]
        name, consumed = _name_after(after) if unit else (None, 0)
        if not name:
            name = _name_before(t[:start])
        if not name and unit:
            # "A pipe of 15 cm diameter suddenly enlarges to 30 cm": a later value in the same unit with no word of
            # its own is the next value of the same quantity ("second diameter"), not a generic length.
            same = [n for u, n in named if u.lower() == unit.lower()]
            if same:
                base = same[-1].split(" ", 1)[1] if same[-1].split(" ", 1)[0] in ORDINALS else same[-1]
                count = sum(1 for n in same if n == base or n.endswith(f" {base}"))
                name = f"{ORDINALS[min(count, len(ORDINALS)) - 1]} {base}"
        if name and unit:
            named.append((unit, name))
        if not name:
            name = _dimension(unit) if unit else "value"
        label = f"[{name} in {unit}]" if unit else f"[{name}]"
        out.append(t[pos:start])
        out.append(label)
        pos = end + consumed
        placeholders.append({"name": name, "unit": unit or "", "example": value})
    out.append(t[pos:])
    template = re.sub(r"\s+", " ", "".join(out)).strip()
    return template, placeholders


ORDINALS = ["second", "third", "fourth", "fifth"]


def _name_after(after: str) -> tuple[str | None, int]:
    m = re.match(r"\s+([a-z]+)(\s+(?:factor|coefficient|gradient|ratio))?\b", after, flags=re.I)
    if m and m.group(1).lower() in QUANTITY_NOUNS:
        name = m.group(1).lower() + (m.group(2) or "").lower()
        return name.strip(), m.end()
    return None, 0


def _name_before(before: str) -> str | None:
    words = re.findall(r"[A-Za-z]+|=|:", before)[-8:]
    for i in range(len(words) - 1, -1, -1):
        w = words[i].lower()
        # "with diameters 500 mm, 400 mm and 300 mm": skip the units of the earlier values in the same list.
        if w in SKIP_BACK or re.fullmatch(_UNIT_RE, w, flags=re.I):
            continue
        if w.rstrip("s") in QUANTITY_NOUNS and w not in QUANTITY_NOUNS:
            w = w.rstrip("s")
        if w in QUANTITY_NOUNS:
            prev = words[i - 1].lower() if i > 0 else ""
            if prev and prev not in SKIP_BACK and prev.isalpha() and len(prev) > 2 and w in (
                    "factor", "coefficient", "gradient", "ratio", "gravity", "head", "rate", "load", "loss"):
                return f"{prev} {w}"
            return w
        break
    return None


def _dimension(unit: str | None) -> str:
    if not unit:
        return "value"
    for pattern, dim in UNIT_DIMENSIONS:
        if re.fullmatch(pattern, unit, flags=re.I):
            return dim
    return "value"


def numeric_target(text: str) -> tuple[str | None, str | None]:
    """(what is to be calculated, the verb) in a numerical question, without any values."""
    t = clean_question(text)
    m = re.search(rf"\b({NUM_VERBS})\s+(.*)", t, flags=re.I)
    if not m:
        return None, None
    rest = m.group(2)
    stop = re.search(r"\d|\s(?:if|when|given|using|assuming|for which|where)\s|[,;.?]", rest)
    target = rest[: stop.start()] if stop else rest
    target = re.sub(r"\s+(?:of|in|at|for|with|on|by|to|from|a|an|the)$", "", target.strip(), flags=re.I)
    target = re.sub(r"\s+(?:of|in|at|for|with|on|by|to|from|a|an|the)$", "", target.strip(), flags=re.I)
    # "the diameter of a single equivalent pipe of length [value]": drop the quantity whose value was cut off.
    m2 = re.search(r"\s+(?:of|with|at)\s+([a-z]+)$", target, flags=re.I)
    if m2 and m2.group(1).lower() in QUANTITY_NOUNS:
        target = target[: m2.start()]
    if len(target.split()) < 2:
        return None, m.group(1).lower()
    return _trim_words(target, 12), m.group(1).lower()


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _marks_text(marks: list[float]) -> str:
    if not marks:
        return ""
    lo, hi = min(marks), max(marks)
    fmt = (lambda v: f"{v:g}")
    rng = fmt(lo) if lo == hi else f"{fmt(lo)}-{fmt(hi)}"
    return f"Past questions in this format carried {rng} marks ({len(marks)} with marks known)."


# ------------------------------------------------------------------------- the guide
def choose_family(stats: dict[str, Any], kinds: set[str], course_families: list[dict[str, Any]]
                  ) -> tuple[str, str, list[dict[str, Any]]]:
    """(family, basis, alternatives). basis: history | weak_history | syllabus | course_pattern | none."""
    fams = [f for f in stats.get("families", []) if f["family"] in FAMILIES]
    if fams:
        top = fams[0]
        basis = "history" if top["papers"] >= 2 and stats.get("exam_frequency", 0) >= 2 else "weak_history"
        return top["family"], basis, fams[1:3]
    syllabus = [KIND_FAMILY[k] for k in sorted(kinds) if k in KIND_FAMILY]
    syllabus = [f for f in FAMILY_ORDER if f in syllabus]
    if syllabus:
        alts = [{"family": f, "display": family_display(f), "papers": 0, "questions": 0} for f in syllabus[1:3]]
        return syllabus[0], "syllabus", alts
    if course_families:
        return course_families[0]["family"], "course_pattern", []
    return "explanation", "none", []


def build_format_guide(*, topic_label: str, topic_title: str, kinds: set[str], concepts: list[str],
                       stats: dict[str, Any], questions: list[dict[str, Any]], course_families: list[dict[str, Any]],
                       papers: list[dict[str, Any]], reliability: dict[str, Any] | None = None,
                       names: set[str] | None = None, accept=None) -> dict[str, Any]:
    """The format guide for one topic.

    ``questions``: the topic's counted past questions as dicts with id, text, families, labels, marks,
    exam_index, exam_label, path_label and role. ``stats``: ``topic_history.node_stats`` for the topic.
    """
    ctx = _Ctx(set(names or ()), accept)
    family, basis, alternatives = choose_family(stats, kinds, course_families)
    T = len(papers)
    a = stats.get("exam_frequency", 0)
    in_family = [q for q in questions if family in q.get("families", [])]
    # A format "established" by questions that are only probable (status B) matches to the topic is a weak
    # indication: those questions may belong to another topic.
    certain = {q["exam_index"] for q in in_family if q.get("status", "A") != "B" or q.get("manual")}
    probable_only = basis == "history" and len(certain) < 2
    if probable_only:
        basis = "weak_history"
    rep = _representative(in_family, ctx)
    guide: dict[str, Any] = {
        "family": family, "display": family_display(family), "basis": basis,
        "evidence": {"history": "established", "weak_history": "weak"}.get(basis, "inferred"),
        "papers": 0, "questions": 0, "topic_papers": a, "usable_papers": T,
    }
    fam_stats = next((f for f in stats.get("families", []) if f["family"] == family), None)
    if fam_stats:
        guide["papers"], guide["questions"] = fam_stats["papers"], fam_stats["questions"]
        guide["last_label"] = papers[fam_stats["last_index"]]["label"] if fam_stats.get("last_index") is not None else None
    guide["description"] = describe(family, rep, topic_title, concepts, in_family, ctx,
                                    "course" if basis in ("course_pattern", "none") else "syllabus")
    guide["template"] = None
    guide["illustrative"] = None
    if family in ("numerical", "design") and rep is not None:
        guide["template"], guide["illustrative"] = _numerical_parts(rep, family)
    guide["marks_note"] = _marks_text([float(q["marks"]) for q in in_family if q.get("marks") and q.get("role") == "primary"])
    guide["why"] = _why(family, basis, guide, stats, kinds, course_families, T, papers,
                        certain_papers=len(certain) if probable_only else None)
    guide["alternatives"] = [_alternative(alt, questions, topic_title, concepts, ctx, a) for alt in alternatives]
    if reliability:
        guide["reliability"] = reliability
    return guide


def describe(family: str, rep: dict[str, Any] | None, topic_title: str, concepts: list[str],
             in_family: list[dict[str, Any]], ctx: "_Ctx | None" = None, source: str = "syllabus") -> str:
    """A specific, one-sentence description of the question structure for this family."""
    if family in ("numerical",) and rep is not None:
        target, verb = numeric_target(rep["text"])
        _, placeholders = numeric_template(rep["text"])
        names, unnamed = [], False
        for p in placeholders:
            if p["name"] == "value":
                unnamed = True
                continue
            n = f"{p['name']} ({p['unit']})" if p["unit"] else p["name"]
            if n not in names:
                names.append(n)
        if unnamed and names:
            names.append("the other values stated")
        given = f" from the given {_join(names[:6])}" if names else " from the given data"
        if target:
            return f"Numerical problem: {verb or 'calculate'} {target}{given}."
        return f"Numerical problem on {_phrase(topic_title, ctx)}: calculate the required quantity{given}."
    if family == "numerical":
        why = ("the past numerical questions mapped here did not pass the topic check, so none is used as a model"
               if in_family else "no past numerical on this topic exists, so the structure is inferred")
        return (f"Numerical problem on {_phrase(topic_title, ctx)}: apply the relations of this topic to given data and "
                f"calculate the required quantity ({why}).")
    if rep is not None:
        obj, verb = object_phrase(rep["text"], family)
        obj = _phrase(obj, ctx) if obj else _phrase(topic_title, ctx)
        verb = _verb(family, verb)
        assumptions = (", and state the assumptions" if any("assumption" in q["text"].lower() for q in in_family)
                       else "")
        sketch = (", with a labelled sketch" if family == "explanation" and any(
            "diagram" in q.get("labels", []) for q in in_family) else "")
        example = (", with an example" if family == "explanation" and "example" not in obj.lower() and any(
            "example" in q.get("labels", []) for q in in_family) else "")
        frame = FRAMES.get(family, "{verb} {obj}.")
        return frame.format(verb=verb, obj=obj, assumptions=assumptions, sketch=sketch, example=example)
    # No past question in this family: build from the syllabus wording.
    obj = _phrase(concepts[0], ctx) if concepts else _phrase(topic_title, ctx)
    frame = FRAMES.get(family, "{verb} {obj}.")
    text = frame.format(verb=DEFAULT_VERBS.get(family, "explain"), obj=obj, assumptions="", sketch="", example="")
    if source == "course":
        return text.rstrip(".") + (" (format inferred from the course-wide mix of question formats, wording from the "
                                   "syllabus entry; no past question on this topic exists).")
    return text.rstrip(".") + " (structure inferred from the syllabus entry; no past question of this kind exists)."


def _verb(family: str, verb: str | None) -> str:
    if not verb:
        return DEFAULT_VERBS.get(family, "explain")
    if verb.startswith("what"):
        return "explain what is meant by"
    verb = re.sub(r"^with (?:a |the )?(?:help of )?(?:neat |suitable |labelled )?(?:sketch|diagram|figure)s?,?\s*", "",
                  verb)
    return verb or DEFAULT_VERBS.get(family, "explain")


@dataclass
class _Ctx:
    names: set[str] = field(default_factory=set)  # words that are names (kept capitalised mid-sentence)
    accept: Callable[[dict[str, Any]], bool] | None = None  # topic check for questions shown with real values


def _phrase(text: str, ctx: "_Ctx | None" = None) -> str:
    """A syllabus or question phrase for use mid-sentence: number removed, first word lower-cased unless it is a
    name (Bernoulli's, Darcy-Weisbach, Reynolds)."""
    from ..generation.questions import display_phrase

    t = re.sub(r"^\s*[\d.]+\s+", "", text or "").strip()
    return display_phrase(t, ctx.names if ctx else set())


def _representative(questions: list[dict[str, Any]], ctx: "_Ctx | None" = None) -> dict[str, Any] | None:
    """The most recent question in the family, preferring questions where the topic is the primary topic.

    Numerical questions are shown with their real values, so they must also pass the topic check (``accept``); a
    past question that does not is probably mapped to the wrong topic and is skipped.
    """
    ordered = sorted(questions, key=lambda q: (q.get("role") != "primary", -q["exam_index"], q["id"]))
    for q in ordered:
        accept = ctx.accept if ctx else None
        if accept is None or not (set(q.get("families", [])) & {"numerical", "design"}) or accept(q):
            return q
    return None


def _source(q: dict[str, Any]) -> dict[str, Any]:
    return {"question_id": q["id"], "exam": q.get("exam_label"), "question": q.get("path_label")}


def _alternative(alt: dict[str, Any], questions: list[dict[str, Any]], topic_title: str,
                 concepts: list[str], ctx: "_Ctx | None" = None, topic_papers: int = 0) -> dict[str, Any]:
    fam = alt["family"]
    in_family = [q for q in questions if fam in q.get("families", [])]
    rep = _representative(in_family, ctx)
    out = {"family": fam, "display": family_display(fam), "papers": alt.get("papers", 0),
           "questions": alt.get("questions", 0), "description": describe(fam, rep, topic_title, concepts, in_family,
                                                                        ctx)}
    if out["papers"] and topic_papers:
        out["support"] = (f"asked in {out['papers']} of the {_n(topic_papers, 'paper')} containing this topic"
                          if topic_papers > 1 else "asked in the one paper containing this topic")
    elif out["papers"]:
        out["support"] = f"asked in {_n(out['papers'], 'past paper')}"
    else:
        out["support"] = "named in the syllabus entry (inferred)"
    if fam in ("numerical", "design") and rep is not None:
        out["template"], out["illustrative"] = _numerical_parts(rep, fam)
    return out


def _numerical_parts(rep: dict[str, Any], family: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(placeholder template, illustrative question with a real paper's values) for a numerical family."""
    template, placeholders = numeric_template(rep["text"])
    where = f"{rep.get('exam_label')} question {rep.get('path_label') or ''}".rstrip()
    tpl = ({"text": template, "placeholders": placeholders, "source": _source(rep),
            "note": f"Every value is a placeholder; the wording follows {where}."} if placeholders else None)
    # The practice question is the real past question, word for word (so its data are known to be consistent). It
    # is labelled as a past question, never as a generated one.
    illustrative = {
        "text": clean_question(rep["text"]), "basis": "past_values", "format": family, "source": _source(rep),
        "note": (f"This is {where} from a real past paper, word for word. A new paper will ask a similar question "
                 f"with other values. The app does not invent new values because it cannot check that they give a "
                 f"solvable problem."),
    }
    return tpl, illustrative


def _n(n: int, word: str, many: str | None = None) -> str:
    """'1 paper', '3 papers'."""
    return f"{n} {word if n == 1 else (many or word + 's')}"


def _why(family: str, basis: str, guide: dict[str, Any], stats: dict[str, Any], kinds: set[str],
         course_families: list[dict[str, Any]], T: int, papers: list[dict[str, Any]],
         certain_papers: int | None = None) -> str:
    disp = family_display(family)
    a = stats.get("exam_frequency", 0)
    fams = [f for f in stats.get("families", []) if f["family"] in FAMILIES]
    tied = [f for f in fams if f["family"] != family and f["papers"] == guide["papers"]]
    if basis == "history":
        text = (f"{disp} questions appeared in {guide['papers']} of the {a} papers that contained this topic"
                + (f", most recently {guide['last_label']}" if guide.get("last_label") else "") + ".")
        if tied:
            names = _join([f["display"].lower() for f in tied])
            text += (f" {names[0].upper() + names[1:]} appeared in as many papers, so these formats are equally common; "
                     f"{disp.lower()} is listed first only because it is the more specific format.")
        elif len(fams) > 1:
            second = fams[1]
            text += (f" The next most frequent format, {second['display'].lower()}, appeared in "
                     f"{_n(second['papers'], 'paper')}.")
        if stats.get("multi_label_questions"):
            text += " Some questions combine formats, so these counts overlap."
        return text
    if basis == "weak_history":
        if certain_papers is not None:
            certain = ("none of those questions is a certain match to this topic (all are probable matches, mapping "
                       "status B)" if certain_papers == 0 else
                       f"only {_n(certain_papers, 'of those papers', 'of those papers')} had a question that certainly "
                       f"belongs to this topic (the others are probable matches, mapping status B)")
            return (f"{disp} questions appeared in {guide['papers']} of the {a} papers that contained this topic, but "
                    f"{certain}. Treat this format as a weak indication.")
        text = (f"This topic appeared in {a} of {_n(T, 'usable paper')}, and {disp.lower()} was asked in "
                f"{guide['papers']} of them. That is too little history to establish a pattern, so treat this format "
                f"as a weak indication.")
        if tied:
            text += f" {_join([f['display'] for f in tied])} appeared as often."
        return text
    if basis == "syllabus":
        named = ", ".join(sorted(k for k in kinds if k in KIND_FAMILY))
        return (f"No past question on this topic was found in the {_n(T, 'usable paper')}. Its syllabus entry names "
                f"'{named}', so {disp.lower()} is suggested. This format is inferred from the syllabus, not observed "
                f"in past papers.")
    if basis == "course_pattern":
        top = course_families[0]
        return (f"No past question on this topic was found in the {_n(T, 'usable paper')}, and its syllabus entry names "
                f"no format. {disp} is the most common format across this course ({top['questions']} of "
                f"{top['of_questions']} counted questions), so it is shown as a starting point. This format is "
                f"inferred, not observed for this topic.")
    return ("No past papers and no format named in the syllabus: a conceptual explanation is shown as a generic "
            "starting point. This format is inferred.")


def format_reliability(accuracy: dict[str, Any] | None) -> dict[str, Any] | None:
    """How often format forecasts matched a format the topic actually took on earlier held-out papers."""
    if not accuracy:
        return None
    topic = accuracy.get("topic") or {}
    glob = accuracy.get("global") or {}
    folds = topic.get("folds") or glob.get("folds")
    if not folds or topic.get("mean") is None:
        return None
    text = (f"On {_n(folds, 'earlier paper')}, the format a topic was asked in most often before matched a format it was "
            f"actually asked in {topic['mean']:.0%} of the time")
    if glob.get("mean") is not None:
        text += f" (the course-wide format mix: {glob['mean']:.0%})"
    text += ". Use the suggested format as a guide, not a certainty."
    return {"topic_history": topic.get("mean"), "course_mix": glob.get("mean"), "folds": folds, "text": text}
