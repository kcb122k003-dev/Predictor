"""Course-content parsing into a unit -> topic -> sub-topic tree (spec section 5).

Handles "Unit I / Chapter 2 / Module 3" headings, decimal numbering (1, 1.1, 1.1.1),
lettered items, bullets, teaching hours, inline concept lists ("Properties: a, b and c"),
laboratory sections, learning objectives, marks-distribution tables, and skips
references, textbooks and evaluation schemes. Every node keeps its source location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config.settings import Settings
from ..ingestion.types import PageText
from ..preprocessing.textnorm import collapse_whitespace

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
         "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15}

RE_UNIT = re.compile(
    r"^(?:unit|chapter|module|lesson|topic|block|section)\s*[-:.]?\s*(?:no\.?\s*)?([IVXLC]{1,5}|\d{1,2}|[A-H])(?![a-z])"
    r"\s*[:.\-–)]?\s*(.*)$", re.IGNORECASE)
RE_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})(?:\.(?!\d))?\)?\s+(?=\S)(.+)$")
RE_LETTERED = re.compile(r"^\(?([a-z])[).]\s+(.+)$")
RE_ROMAN_ITEM = re.compile(r"^\(?((?:i{1,3}|iv|vi{0,3}|ix|x))[).]\s+(.+)$", re.IGNORECASE)
RE_BULLET = re.compile(r"^[•\-*•▪●○◦·✓✔➢➤►]\s*(.+)$")
RE_HOURS = re.compile(
    r"[\[(]?\s*[-–]?\s*(\d{1,3}(?:\.\d)?)\s*(?:hours?|hrs?\.?|lectures?|lecture hours|periods?|classes|L)\s*[\])]?",
    re.IGNORECASE)
RE_OBJECTIVES = re.compile(r"^\W*(?:objectives?|learning (?:objectives|outcomes)|course outcomes|"
                           r"specific objectives|students will be able to)\b\s*[:\-]?\s*(.*)$", re.IGNORECASE)
RE_OBJECTIVE_LEAD = re.compile(
    r"^\W*(?:to|understand|know|learn|apply|analy[sz]e|explain|describe|develop|identify|demonstrate|gain|"
    r"familiari[sz]e|introduce|provide|enable|equip|impart|acquaint|students?|be able|make|build|use)\b",
    re.IGNORECASE)
RE_CONNECTOR_END = re.compile(r"\b(?:and|or|of|the|in|on|for|to|with|between|a|an|its|their)$", re.IGNORECASE)
LAB_HEADINGS = re.compile(r"^\W*(?:practicals?|laboratory(?: work| works| exercises)?|lab(?:oratory)? work|"
                          r"list of experiments|experiments)\W*$", re.IGNORECASE)
KIND_PATTERNS = {
    "numerical": re.compile(r"\bnumerical|\bproblems?\b|\bcalculations?\b|\bexamples?\b|\bsizing\b", re.IGNORECASE),
    "derivation": re.compile(r"\bderivation|\bderive|\bproof\b|\bexpression for\b|\btheorem\b", re.IGNORECASE),
    "lab": re.compile(r"\bexperiment|\blaborator|\blab\b|\bdetermination of\b", re.IGNORECASE),
    "design": re.compile(r"\bdesign\b", re.IGNORECASE),
    "process": re.compile(r"\bprocess(?:es)?\b|\bcycle\b|\bmechanism\b|\bworking\b", re.IGNORECASE),
    "method": re.compile(r"\bmethod\b|\btechnique\b|\balgorithm\b|\bapproach\b", re.IGNORECASE),
}


@dataclass
class SyllabusNode:
    title: str
    depth: int
    number: str = ""
    description: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    hours: float | None = None
    marks_weight: float | None = None
    kinds: list[str] = field(default_factory=list)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    children: list["SyllabusNode"] = field(default_factory=list)
    parent: "SyllabusNode | None" = field(default=None, repr=False)

    def add_child(self, node: "SyllabusNode") -> "SyllabusNode":
        node.parent = self
        self.children.append(node)
        return node

    def walk(self) -> Iterable["SyllabusNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def full_text(self) -> str:
        parts = [self.title] + self.concepts + self.description
        return collapse_whitespace(" ".join(parts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "depth": self.depth, "number": self.number,
            "description": " ".join(self.description), "concepts": self.concepts,
            "objectives": self.objectives, "hours": self.hours, "marks_weight": self.marks_weight,
            "kinds": self.kinds, "source_refs": self.source_refs,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ParsedSyllabus:
    roots: list[SyllabusNode]
    course_title: str = ""
    course_code: str = ""
    objectives: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def all_nodes(self) -> list[SyllabusNode]:
        out: list[SyllabusNode] = []
        for r in self.roots:
            out.extend(r.walk())
        return out


def _roman_or_int(label: str) -> int | None:
    if label.isdigit():
        return int(label)
    return ROMAN.get(label.lower())


def split_concepts(text: str, max_words: int) -> list[str]:
    """Split "a, b and c" style lists into concept phrases."""
    text = re.sub(r"\s+", " ", text).strip(" .;:")
    if not text:
        return []
    # Do not split inside parentheses.
    depth, buf, parts = 0, [], []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    out: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip(" .")
        if not part:
            continue
        # "x and y" at the end of a list is two concepts; elsewhere "and" may be part of a name.
        if (i == len(parts) - 1 and len(parts) > 1) or len(part.split()) > max_words:
            sub = re.split(r"\s+(?:and|&)\s+", part)
            if len(sub) > 1 and all(len(s.split()) <= max_words for s in sub):
                out.extend(s.strip(" .") for s in sub if s.strip(" ."))
                continue
        out.append(part)
    return [c for c in out if len(c) >= 2]


def _list_item(line: str) -> tuple[str, str] | None:
    """Return (label, text) for bullet, lettered and roman list items."""
    m = RE_ROMAN_ITEM.match(line)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    m = RE_LETTERED.match(line)
    if m:
        return m.group(1), m.group(2).strip()
    m = RE_BULLET.match(line)
    if m and not re.match(r"^-\s*\d", line):
        return "•", m.group(1).strip()
    return None


def _label_style(label: str) -> str:
    if label == "•":
        return "bullet"
    if label.lower() in ROMAN:
        return "roman"
    if len(label) == 1 and label.isalpha():
        return "letter"
    return "number"


def _detect_kinds(text: str) -> list[str]:
    return [kind for kind, pattern in KIND_PATTERNS.items() if pattern.search(text)]


class SyllabusParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = settings.syllabus
        self.stop_headings = [h.lower() for h in cfg.stop_headings]
        self.split_inline = bool(cfg.split_inline_concepts)
        self.max_concept_words = int(cfg.max_concept_words)

    def _is_structural(self, line: str) -> bool:
        return bool(RE_UNIT.match(line) or RE_NUMBERED.match(line) or _list_item(line) is not None
                    or self._is_stop_heading(line) or LAB_HEADINGS.match(line) or RE_OBJECTIVES.match(line))

    def _joined_lines(self, pages: list[PageText]) -> list[tuple[int, int, str]]:
        """Re-join lines that a PDF wrapped ("..., displacement thickness and" + "momentum thickness"),
        including across page breaks. Returns (page, line, text) using the first line's location."""
        out: list[tuple[int, int, str]] = []
        for page in pages:
            for line_no, raw in enumerate(page.text.split("\n"), start=1):
                line = raw.strip()
                if not line:
                    continue
                if out and not self._is_structural(line):
                    prev = out[-1][2].rstrip()
                    if prev and (prev[-1] in ",;&" or RE_CONNECTOR_END.search(prev) or line[0].islower()):
                        out[-1] = (out[-1][0], out[-1][1], f"{prev} {line}")
                        continue
                out.append((page.page_no, line_no, raw))
        return out

    def _is_stop_heading(self, line: str) -> bool:
        core = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if not core or len(core.split()) > 5:
            return False
        return any(core == h or core.startswith(h + " ") or core.rstrip("s") == h.rstrip("s")
                   for h in self.stop_headings)

    def _extract_hours(self, text: str) -> tuple[str, float | None]:
        hours = None
        for m in RE_HOURS.finditer(text):
            try:
                hours = float(m.group(1))
            except ValueError:
                continue
        if hours is not None:
            text = RE_HOURS.sub(" ", text)
        text = re.sub(r"\s*[\[(]\s*[\])]\s*", " ", text)
        return collapse_whitespace(text).strip(" -–:"), hours

    def _make_node(self, raw_title: str, depth: int, number: str, ref: dict[str, Any]) -> SyllabusNode:
        title, hours = self._extract_hours(raw_title)
        concepts: list[str] = []
        if self.split_inline:
            m = re.match(r"^(.{3,90}?)\s*(?::|\s[-–]\s)\s*(.+)$", title)
            if m and len(m.group(1).split()) <= 10:
                title = m.group(1).strip()
                concepts = split_concepts(m.group(2), self.max_concept_words)
            elif "," in title or ";" in title:
                concepts = split_concepts(title, self.max_concept_words)
                if len(concepts) > 1:
                    title = collapse_whitespace(title)
                else:
                    concepts = []
        node = SyllabusNode(title=title.strip(" .;"), depth=depth, number=number, concepts=concepts,
                            hours=hours, source_refs=[ref])
        node.kinds = _detect_kinds(raw_title)
        return node

    def parse(self, pages: list[PageText], filename: str = "", file_id: int | None = None) -> ParsedSyllabus:
        result = ParsedSyllabus(roots=[])
        stack: list[SyllabusNode] = []  # current path, index = depth - 1
        in_stop = False
        stop_kind = ""
        in_unit_scheme = False
        current_unit_no: int | None = None
        pending_title_for: SyllabusNode | None = None
        objectives_target: SyllabusNode | None | str = None  # node, "course" or None
        lab_root: SyllabusNode | None = None
        marks_table: dict[int, tuple[float | None, float]] = {}
        heading_texts = {h["text"].strip().lower(): h.get("style", "") for p in pages
                         for h in p.details.get("headings", []) if h.get("text")}

        def attach(node: SyllabusNode) -> None:
            while len(stack) >= node.depth:
                stack.pop()
            if stack:
                stack[-1].add_child(node)
                node.depth = stack[-1].depth + 1
            else:
                node.depth = 1
                result.roots.append(node)
            stack.append(node)

        for page_no, line_no, raw in self._joined_lines(pages):
            line = raw.strip()
            if not line:
                continue
            ref = {"file": filename, "file_id": file_id, "page": page_no, "line": line_no,
                   "text": line[:240]}

            if LAB_HEADINGS.match(line):
                in_stop = False
                lab_root = SyllabusNode(title="Laboratory / practical work", depth=1, kinds=["lab"],
                                        source_refs=[ref])
                stack.clear()
                attach(lab_root)
                objectives_target = None
                continue
            if self._is_stop_heading(line):
                in_stop = True
                stop_kind = line.lower()
                lowered = line.lower()
                objectives_target = "course" if "objective" in lowered or "outcome" in lowered else None
                continue

            unit_m = RE_UNIT.match(line)
            if unit_m and len(line) < 160 and not RE_HOURS.fullmatch(unit_m.group(2) or ""):
                in_stop = False
                lab_root = None
                in_unit_scheme = True
                current_unit_no = _roman_or_int(unit_m.group(1))
                rest = unit_m.group(2).strip()
                stack.clear()
                node = self._make_node(rest or f"Unit {unit_m.group(1)}", 1, unit_m.group(1), ref)
                attach(node)
                pending_title_for = node if not rest or RE_HOURS.fullmatch(rest.strip()) else None
                objectives_target = None
                continue

            if in_stop and ("objective" in stop_kind or "outcome" in stop_kind):
                # Objectives end where the topic list starts: a numbered line that carries hours
                # or does not read like an objective ("To understand ...").
                num_m = RE_NUMBERED.match(line)
                if num_m and (RE_HOURS.search(line) or not RE_OBJECTIVE_LEAD.match(num_m.group(2))):
                    in_stop = False
                    objectives_target = None
            if in_stop:
                if "objective" in stop_kind or "outcome" in stop_kind:
                    item = re.sub(r"^\s*(?:\d{1,2}[.)]|[a-z][.)]|[•\-*•])\s*", "", line)
                    if item:
                        result.objectives.append(item)
                elif "marks" in stop_kind or "evaluation" in stop_kind:
                    m = re.match(r"^(?:chapter|unit)?\s*(\d{1,2}(?:\s*(?:,|&|and|-)\s*\d{1,2})*)\b.*?"
                                 r"(?:(\d{1,3})\s+)?(\d{1,3})\s*\*?$", line, re.IGNORECASE)
                    if m:
                        chapters = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
                        hours = float(m.group(2)) if m.group(2) else None
                        marks = float(m.group(3))
                        for ch in chapters:
                            marks_table[ch] = (hours, marks / len(chapters))
                continue

            obj_m = RE_OBJECTIVES.match(line)
            if obj_m:
                objectives_target = stack[0] if stack else "course"
                if obj_m.group(1).strip():
                    self._add_objective(result, objectives_target, obj_m.group(1).strip())
                continue

            style = heading_texts.get(line.lower(), "")
            if style.lower() in ("heading 1", "title") and not RE_NUMBERED.match(line):
                stack.clear()
                attach(self._make_node(line, 1, "", ref))
                in_unit_scheme = True
                objectives_target = None
                continue

            num_m = RE_NUMBERED.match(line)
            if num_m and not RE_HOURS.fullmatch(num_m.group(2).strip()):
                number = num_m.group(1)
                parts = number.split(".")
                if in_unit_scheme:
                    if len(parts) > 1 and current_unit_no is not None and int(parts[0]) == current_unit_no:
                        depth = len(parts)
                    else:
                        depth = len(parts) + 1
                else:
                    depth = len(parts)
                if lab_root is not None and stack and stack[0] is lab_root:
                    depth = max(depth, 2)
                node = self._make_node(num_m.group(2), depth, number, ref)
                attach(node)
                if not in_unit_scheme and len(parts) == 1:
                    current_unit_no = int(parts[0])
                pending_title_for = None
                objectives_target = None
                continue

            item = _list_item(line)
            if item is not None and objectives_target is not None:
                self._add_objective(result, objectives_target, item[1])
                continue
            if item is not None and stack:
                label, text = item
                parent = stack[-1]
                # A list item after an item of the same style is a sibling, otherwise a child.
                same_style = parent.number and _label_style(parent.number) == _label_style(label)
                depth = parent.depth if same_style else parent.depth + 1
                attach(self._make_node(text, depth, label, ref))
                continue

            if pending_title_for is not None:
                title, hours = self._extract_hours(line)
                pending_title_for.title = title or pending_title_for.title
                pending_title_for.hours = pending_title_for.hours or hours
                pending_title_for.kinds = sorted(set(pending_title_for.kinds) | set(_detect_kinds(line)))
                pending_title_for.source_refs.append(ref)
                pending_title_for = None
                continue

            if stack:
                node = stack[-1]
                text, hours = self._extract_hours(line)
                if hours is not None and node.hours is None:
                    node.hours = hours
                if text:
                    node.description.append(text)
                    if self.split_inline:
                        for c in split_concepts(text, self.max_concept_words):
                            if c not in node.concepts and len(c.split()) <= self.max_concept_words:
                                node.concepts.append(c)
                    node.kinds = sorted(set(node.kinds) | set(_detect_kinds(text)))
                continue

            # Text before any structure: course title / code.
            if not result.course_title and len(line.split()) <= 14:
                result.course_title = line
            m = re.search(r"\b([A-Z]{2,6}\s*-?\s*\d{3,4}[A-Z]?)\b", line)
            if m and not result.course_code:
                result.course_code = m.group(1).replace(" ", "")

        for root in result.roots:
            no = _roman_or_int(root.number) if root.number else None
            if no is not None and no in marks_table:
                hours, marks = marks_table[no]
                root.marks_weight = marks
                if root.hours is None and hours is not None:
                    root.hours = hours
        self._drop_empty_placeholders(result)
        if not result.roots:
            result.warnings.append("No syllabus structure was recognised. Add topics manually or check the file text.")
        return result

    @staticmethod
    def _add_objective(result: ParsedSyllabus, target: Any, text: str) -> None:
        if isinstance(target, SyllabusNode):
            target.objectives.append(text)
        else:
            result.objectives.append(text)

    @staticmethod
    def _drop_empty_placeholders(result: ParsedSyllabus) -> None:
        def keep(node: SyllabusNode) -> bool:
            return bool(node.title.strip()) or bool(node.children)

        result.roots = [r for r in result.roots if keep(r)]
        for node in result.all_nodes():
            node.children = [c for c in node.children if keep(c)]
