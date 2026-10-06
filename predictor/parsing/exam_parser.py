"""Structure-aware exam paper segmentation (spec sections 7 and 39).

Turns page text into Exam -> Section -> Question -> Sub-question -> Sub-sub-question with
marks, OR alternatives, optional-question instructions, MCQ options and source location.
It does not assume a fixed template: numbering styles (1. / 1) / Q1 / Q.No.1 / 1(a) /
1.1), lettered and roman sub-parts, inline sub-parts on one line, marks in brackets,
parentheses, words or the right margin are all handled, with a confidence per node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..config.settings import Settings
from ..ingestion.types import PageText
from ..preprocessing.quality import assess_text
from ..preprocessing.textnorm import collapse_whitespace, word_count
from .metadata import ExamMetadata, extract_metadata

ROMAN_VALUES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}

RE_OR = re.compile(r"^[\s\-–—_*=.(\[]*\b(?:or|OR|Or)\b[\s\-–—_*=.)\]]*$")
# Section headers are capitalised ("Group A", "SECTION B", "Part II"); "section 2 of the pipe" is text.
RE_SECTION = re.compile(
    r"^(?:Group|GROUP|Section|SECTION|Part|PART)\s*[-–:.]?\s*([A-Z]|[IVX]{1,4}|[ivx]{1,4}|\d{1,2})(?![a-z])"
    r"\s*([:.\-–)(\[].*|(?:[Aa]ttempt|[Aa]nswer|ATTEMPT|All|ALL|Any)\b.*|)$")
RE_INSTRUCTION = re.compile(
    r"^\W*(?:attempt|answer)\s+(?:any|all|the following|question)|^\W*all questions (?:carry|are)|"
    r"^\W*(?:the )?figures? in the (?:right[- ])?margin|^\W*candidates? (?:are|is)|"
    r"^\W*assume suitable|^\W*use of [\w\s,]{0,40}(?:is|are) (?:allowed|permitted|not allowed)|"
    r"^\W*(?:note|instructions?)\s*[:\-]|^\W*symbols? (?:used )?have (?:their )?usual meanings?|"
    r"^\W*q\.?\s*no\.?\s+.*marks?\s*$|^\W*question\s+marks\s*$",
    re.IGNORECASE)
RE_ATTEMPT = re.compile(r"(?:attempt|answer)\s+any\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
RE_PAGE_NO = re.compile(r"^(?:page\s*\d+\s*(?:of\s*\d+)?|-\s*\d+\s*-|\d+\s*/\s*\d+|p\.\s*t\.\s*o\.?|contd\.?\.*|"
                        r"\(?\s*turn over\s*\)?|\(?\s*continued\s*\)?)$", re.IGNORECASE)

RE_Q_PREFIX = re.compile(r"^(?:Q(?:uestion|n|s)?\s*\.?\s*(?:No\.?\s*)?)(\d{1,2})(?!\d)\s*[.):\-]?\s*(.*)$", re.IGNORECASE)
RE_NUM_SUB = re.compile(r"^(?:Q(?:uestion)?\s*\.?\s*(?:No\.?\s*)?)?(\d{1,2})\s*[.)]?\s*\(\s*([a-hA-H]|i{1,3}|iv|v)\s*\)\s*(.*)$")
RE_NUM_SUB2 = re.compile(r"^(\d{1,2})\s*[.)]\s+\(?([a-h])\s*[).]\s+(.*)$")
RE_DOTTED = re.compile(r"^(?:Q\s*\.?\s*)?(\d{1,2})\.(\d{1,2})\s*[.)]?\s+(?=[A-Z(\"'])(.*)$")
RE_NUM = re.compile(r"^(\d{1,2})\s*[.)](?!\d)\s*(.*)$")
RE_UNIT_START = re.compile(r"^(?:k?Pa|MPa|kN|N|J|kJ|W|kW|m|cm|mm|km|kg|g|s|K|bar|atm|L|ml|mol|rpm|V|A|Hz|%)"
                           r"(?:[./^\s\d]|$)")
RE_NUM_BARE = re.compile(r"^(\d{1,2})\s+(?=[A-Z(])(.*)$")
RE_LETTER_PAREN = re.compile(r"^\(\s*([a-zA-Z])\s*\)\s*(.*)$")
RE_LETTER = re.compile(r"^([a-z])\s*[).]\s+(.*)$")
RE_ROMAN_PAREN = re.compile(r"^\(\s*(i{1,3}|iv|vi{0,3}|ix|x)\s*\)\s*(.*)$", re.IGNORECASE)
RE_ROMAN = re.compile(r"^(i{1,3}|iv|vi{0,3}|ix|x)\s*[).]\s+(.*)$", re.IGNORECASE)
RE_BARE_MARKS = re.compile(r"^[\[(]?\s*(\d{1,2}(?:\.5)?(?:\s*[+×xX*]\s*\d{1,2}(?:\.5)?)*)\s*(?:marks?)?\s*[\])]?$", re.IGNORECASE)

_EXPR = r"\d{1,3}(?:\.\d)?(?:\s*[+×xX*]\s*\d{1,3}(?:\.\d)?)*"
MARKS_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (re.compile(rf"\[\s*({_EXPR})\s*(?:marks?|m)?\s*\]\s*[.,;]?\s*$", re.IGNORECASE), "bracket", 1.0),
    (re.compile(rf"\(\s*({_EXPR})\s*marks?\s*\)\s*[.,;]?\s*$", re.IGNORECASE), "paren_word", 1.0),
    (re.compile(rf"(?<![\w.])({_EXPR})\s*marks?\s*[.,;]?\s*$", re.IGNORECASE), "word", 0.95),
    (re.compile(rf"\(\s*({_EXPR})\s*\)\s*[.,;]?\s*$"), "paren", 0.8),
]
RE_MARGIN_MARKS = re.compile(rf"\s{{3,}}({_EXPR})\s*$")

# Inline sub-parts: "a) Define X. [3] b) State Y. [3]" or "(i) ... (ii) ..." on one line.
RE_INLINE_SPLIT = re.compile(
    r"(?<=[\])?.:])\s+(?=(?:\(\s*(?:[b-h]|ii|iii|iv|v|vi)\s*\)|(?:[b-h]|ii|iii|iv|v|vi)\s*\))\s+[A-Z\"'(])")
RE_INLINE_MAIN = re.compile(r"(?<=\])\s+(?=\d{1,2}\s*[.)]\s+[A-Z])")
RE_INLINE_OR = re.compile(r"(?<=[\].?])\s+(OR)\s+(?=(?:\d{1,2}\s*[.)]|\(?[a-h]\s*\)|\(?[ivx]{1,4}\s*\)|[A-Z]))")
RE_MCQ_CUE = re.compile(r"\?|\b(?:choose|select|which of the following|which one|correct (?:answer|option|statement)|"
                        r"tick|identify the|best describes|true)\b", re.IGNORECASE)
RE_INLINE_OPTIONS = re.compile(r"\(\s*([a-dA-D])\s*\)\s*([^()]{1,80}?)(?=\s*\(\s*[a-dA-D]\s*\)|$)")


@dataclass
class Line:
    text: str
    page: int
    line_no: int


@dataclass
class ParsedQuestion:
    label: str
    kind: str  # main | letter | roman | dotted
    depth: int
    page_no: int
    line_no: int
    lines: list[str] = field(default_factory=list)
    marks: float | None = None
    marks_expr: str = ""
    marks_source: str = ""
    or_group: str | None = None
    is_optional: bool = False
    section_index: int | None = None
    children: list["ParsedQuestion"] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    parent: "ParsedQuestion | None" = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return collapse_whitespace(" ".join(self.lines))

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def path_label(self) -> str:
        parts: list[str] = []
        node: ParsedQuestion | None = self
        while node is not None:
            if node.kind == "dotted":
                parts.append(node.label)
                break  # "5.1" already contains the main number
            parts.append(node.label if node.kind == "main" else f"({node.label})")
            node = node.parent
        parts.reverse()
        return "".join(parts)

    def walk(self) -> Iterable["ParsedQuestion"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def leaves(self) -> Iterable["ParsedQuestion"]:
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child.leaves()

    def context_text(self, max_stem_words: int = 60) -> str:
        stems: list[str] = []
        node = self.parent
        while node is not None:
            if node.text:
                words = node.text.split()
                stems.append(" ".join(words[:max_stem_words]))
            node = node.parent
        stems.reverse()
        return collapse_whitespace(" ".join(stems + [self.text]))


@dataclass
class ParsedSection:
    label: str
    title: str
    instructions: list[str] = field(default_factory=list)
    attempt_count: int | None = None


@dataclass
class ParsedExam:
    metadata: ExamMetadata
    header_text: str
    instructions: list[str]
    sections: list[ParsedSection]
    questions: list[ParsedQuestion]
    warnings: list[str] = field(default_factory=list)
    attempt_count: int | None = None

    def all_nodes(self) -> list[ParsedQuestion]:
        out: list[ParsedQuestion] = []
        for q in self.questions:
            out.extend(q.walk())
        return out

    def leaves(self) -> list[ParsedQuestion]:
        out: list[ParsedQuestion] = []
        for q in self.questions:
            out.extend(q.leaves())
        return out


def parse_marks_expr(expr: str) -> float | None:
    expr = expr.replace(" ", "").replace("×", "x").replace("X", "x").replace("*", "x")
    try:
        if "+" in expr and "x" not in expr:
            return float(sum(float(p) for p in expr.split("+")))
        if "x" in expr and "+" not in expr:
            total = 1.0
            for p in expr.split("x"):
                total *= float(p)
            return total
        if "+" in expr and "x" in expr:  # e.g. 2x3+4
            return float(sum(_product(term) for term in expr.split("+")))
        return float(expr)
    except ValueError:
        return None


def _product(term: str) -> float:
    total = 1.0
    for p in term.split("x"):
        total *= float(p)
    return total


def extract_trailing_marks(text: str, *, allow_margin: bool = True, max_marks: float = 100.0
                           ) -> tuple[str, float | None, str, str, float]:
    """Strip marks from the end of ``text``. Returns (text, marks, expr, source, confidence)."""
    stripped = text.rstrip()
    for pattern, source, conf in MARKS_PATTERNS:
        m = pattern.search(stripped)
        if not m:
            continue
        value = parse_marks_expr(m.group(1))
        if value is None or value <= 0 or value > max_marks:
            continue
        if source == "paren" and value > 30:
            continue
        before = stripped[: m.start()].rstrip()
        if source == "paren" and (not before or before.endswith(("=", "of", "in"))):
            continue
        return before, value, m.group(1).replace(" ", ""), source, conf
    if allow_margin:
        m = RE_MARGIN_MARKS.search(text)
        if m:
            value = parse_marks_expr(m.group(1))
            before = text[: m.start()].rstrip()
            if value is not None and 0 < value <= 30 and before and not before.endswith(("=", "x", "*", "+", "-")):
                return before, value, m.group(1).replace(" ", ""), "margin", 0.75
    return text, None, "", "", 0.0


class ExamParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.parsing

    # ------------------------------------------------------------------ lines
    def _collect_lines(self, pages: list[PageText]) -> list[Line]:
        per_page: list[list[Line]] = []
        for page in pages:
            lines = [Line(t, page.page_no, i + 1) for i, t in enumerate(page.text.split("\n"))]
            per_page.append(lines)
        # Repeated running headers/footers: identical (digit-insensitive) lines near the top or
        # bottom of two or more pages.
        if len(per_page) >= 2:
            def key(text: str) -> str:
                return re.sub(r"\d+", "#", text.strip().lower())

            edge_counts: dict[str, int] = {}
            for lines in per_page:
                non_empty = [ln for ln in lines if ln.text.strip()]
                edges = {key(ln.text) for ln in non_empty[:2] + non_empty[-2:]}
                for k in edges:
                    edge_counts[k] = edge_counts.get(k, 0) + 1
            repeated = {k for k, c in edge_counts.items() if c >= 2 and len(k) > 3}
            # Never drop a line that looks like a question start.
            repeated = {k for k in repeated if not re.match(r"^(?:q\.?\s*)?#\s*[.)]", k)}
        else:
            repeated = set()
        out: list[Line] = []
        for lines in per_page:
            non_empty = [ln for ln in lines if ln.text.strip()]
            last = non_empty[-1] if non_empty else None
            first = non_empty[0] if non_empty else None
            for ln in lines:
                stripped = ln.text.strip()
                if not stripped:
                    continue
                k = re.sub(r"\d+", "#", stripped.lower())
                if k in repeated and (ln in non_empty[:2] or ln in non_empty[-2:]):
                    continue
                if RE_PAGE_NO.match(stripped):
                    continue
                if (ln is last or ln is first) and stripped.isdigit() and int(stripped) == ln.page and len(per_page) > 1:
                    continue
                out.append(ln)
        return self._split_inline(out)

    def _split_inline(self, lines: list[Line]) -> list[Line]:
        out: list[Line] = []
        for ln in lines:
            pieces = [ln.text]
            for splitter in (RE_INLINE_OR, RE_INLINE_MAIN, RE_INLINE_SPLIT):
                new: list[str] = []
                for piece in pieces:
                    if splitter is RE_INLINE_OR:
                        parts = splitter.split(piece)
                        new.extend(p for p in parts if p and p.strip())
                    else:
                        new.extend(p for p in splitter.split(piece) if p and p.strip())
                pieces = new
            expanded: list[str] = []
            for piece in pieces:
                head = re.match(r"^((?:Q\.?\s*)?\d{1,2}\s*[.)]?\s*)(\(.*)$", piece.strip())
                if head and RE_LEADING_PAREN_LABEL.match(head.group(2)):
                    parts = split_sequential_labels(head.group(2))
                    parts[0] = head.group(1) + parts[0]
                    expanded.extend(parts)
                else:
                    expanded.extend(split_sequential_labels(piece.strip()))
            for piece in expanded:
                out.append(Line(piece.strip() if piece.strip() == "OR" else piece, ln.page, ln.line_no))
        return out

    # ---------------------------------------------------------- classification
    @staticmethod
    def _classify(text: str) -> tuple[str, dict]:
        t = text.strip()
        if RE_OR.match(t):
            return "or", {}
        m = RE_SECTION.match(t)
        if m and len(t) < 120:
            return "section", {"label": m.group(1).upper(), "rest": m.group(2).strip(" :.-–)")}
        if RE_INSTRUCTION.match(t):
            return "instruction", {}
        m = RE_BARE_MARKS.match(t)
        if m and ("[" in t or "(" in t or "mark" in t.lower() or len(t) <= 3):
            return "marks", {"expr": m.group(1)}
        m = RE_NUM_SUB.match(t)
        if m:
            return "main_sub", {"num": int(m.group(1)), "sub": m.group(2), "rest": m.group(3), "strength": 0.95}
        m = RE_NUM_SUB2.match(t)
        if m:
            return "main_sub", {"num": int(m.group(1)), "sub": m.group(2), "rest": m.group(3), "strength": 0.9}
        m = RE_DOTTED.match(t)
        if m:
            return "dotted", {"num": int(m.group(1)), "sub": m.group(2), "rest": m.group(3), "strength": 0.8}
        m = RE_Q_PREFIX.match(t)
        if m:
            return "main", {"num": int(m.group(1)), "rest": m.group(2), "strength": 1.0}
        m = RE_NUM.match(t)
        if m:
            return "main", {"num": int(m.group(1)), "rest": m.group(2), "strength": 0.9}
        m = RE_ROMAN_PAREN.match(t) or RE_ROMAN.match(t)
        if m:
            return "roman_or_letter", {"label": m.group(1).lower(), "rest": m.group(2)}
        m = RE_LETTER_PAREN.match(t) or RE_LETTER.match(t)
        if m:
            return "letter", {"label": m.group(1).lower(), "rest": m.group(2)}
        m = RE_NUM_BARE.match(t)
        if m:
            return "main", {"num": int(m.group(1)), "rest": m.group(2), "strength": 0.6}
        return "text", {}

    def _find_first_question(self, lines: list[Line]) -> int:
        for i, ln in enumerate(lines):
            kind, info = self._classify(ln.text)
            if kind in ("main", "main_sub", "dotted") and info["strength"] >= 0.8 and info["num"] <= 3:
                # "1." inside an instruction list (e.g. "1. Attempt all questions") is not a question.
                if RE_INSTRUCTION.match(info["rest"]):
                    continue
                j = i - 1
                while j >= 0 and self._classify(lines[j].text)[0] in ("section", "instruction"):
                    j -= 1
                return j + 1 if j + 1 < i and self._classify(lines[j + 1].text)[0] == "section" else i
            if kind == "section":
                for k in range(i + 1, min(i + 6, len(lines))):
                    k_kind, k_info = self._classify(lines[k].text)
                    if k_kind in ("main", "main_sub") and k_info.get("strength", 0) >= 0.9:
                        return i
        # Papers (or fragments) that do not start at question 1.
        for i, ln in enumerate(lines):
            kind, info = self._classify(ln.text)
            if kind in ("main", "main_sub", "dotted") and info["strength"] >= 0.8 and not RE_INSTRUCTION.match(info["rest"]):
                return i
        return len(lines)

    # ------------------------------------------------------------------ parse
    def parse(self, pages: list[PageText], filename: str = "") -> ParsedExam:
        lines = self._collect_lines(pages)
        start = self._find_first_question(lines)
        header_lines = lines[:start]
        body = lines[start:]
        header_text = "\n".join(ln.text for ln in header_lines)
        full_text = "\n".join(ln.text for ln in lines)
        metadata = extract_metadata(header_text, full_text, filename,
                                    min_year=int(self.cfg.min_year), max_year=int(self.cfg.max_year))
        instructions = [collapse_whitespace(ln.text) for ln in header_lines
                        if RE_INSTRUCTION.match(ln.text) or re.match(r"^\s*[✓✔•\-*]\s+", ln.text)]
        exam = ParsedExam(metadata, header_text, instructions, [], [])
        if not body:
            exam.warnings.append("No numbered questions were found. Check the extracted text and split questions manually.")
            return exam
        self._walk(body, exam)
        self._postprocess(exam, pages)
        return exam

    def _walk(self, body: list[Line], exam: ParsedExam) -> None:
        max_q = int(self.cfg.max_question_number)
        allow_margin = bool(self.cfg.detect_margin_marks)
        current_main: ParsedQuestion | None = None
        current_l2: ParsedQuestion | None = None
        current_l3: ParsedQuestion | None = None
        l2_kind: str | None = None
        last_main_num = 0
        section_started = False
        pending_or = False
        or_counter = 0
        section_index: int | None = None

        def link_or(prev: ParsedQuestion | None, new: ParsedQuestion) -> None:
            nonlocal or_counter
            if prev is None:
                return
            if prev.or_group is None:
                or_counter += 1
                prev.or_group = f"OR{or_counter}"
            new.or_group = prev.or_group

        def new_main(num: int, rest: str, ln: Line, strength: float, label: str | None = None) -> ParsedQuestion:
            nonlocal current_main, current_l2, current_l3, l2_kind, last_main_num, pending_or, section_started
            node = ParsedQuestion(label or str(num), "main", 1, ln.page, ln.line_no,
                                  section_index=section_index, confidence=strength)
            prev = exam.questions[-1] if exam.questions else None
            if pending_or or (prev is not None and prev.label == node.label):
                link_or(prev, node)
            if prev is not None and num > last_main_num + 1 and not section_started:
                node.flags.append("numbering_gap")
                node.confidence = min(node.confidence, 0.7)
            exam.questions.append(node)
            current_main, current_l2, current_l3, l2_kind = node, None, None, None
            last_main_num = max(last_main_num, num)
            pending_or = False
            section_started = False
            self._add_text(node, rest, allow_margin)
            return node

        def new_child(parent: ParsedQuestion, label: str, kind: str, rest: str, ln: Line) -> ParsedQuestion:
            nonlocal pending_or
            node = ParsedQuestion(label, kind, parent.depth + 1, ln.page, ln.line_no,
                                  section_index=parent.section_index, parent=parent,
                                  confidence=min(parent.confidence, 0.95))
            prev = parent.children[-1] if parent.children else None
            if pending_or or (prev is not None and prev.label == label):
                link_or(prev, node)
            elif prev is not None and kind in ("letter", "roman") and not _is_next_label(prev.label, label, kind):
                node.flags.append("sub_label_out_of_sequence")
                node.confidence = min(node.confidence, 0.75)
            parent.children.append(node)
            pending_or = False
            self._add_text(node, rest, allow_margin)
            return node

        for ln in body:
            text = ln.text.strip()
            kind, info = self._classify(text)

            if kind == "or":
                pending_or = True
                continue
            if kind == "section":
                attempt = _attempt_count(info["rest"])
                exam.sections.append(ParsedSection(info["label"], info["rest"], [info["rest"]] if attempt else [], attempt))
                section_index = len(exam.sections) - 1
                section_started = True
                current_main = current_l2 = current_l3 = None
                continue
            if kind == "instruction":
                attempt = _attempt_count(text)
                if section_index is not None and (current_main is None or section_started):
                    sec = exam.sections[section_index]
                    sec.instructions.append(collapse_whitespace(text))
                    if attempt:
                        sec.attempt_count = attempt
                elif current_main is None:
                    exam.instructions.append(collapse_whitespace(text))
                    if attempt:
                        exam.attempt_count = attempt
                else:
                    deepest = current_l3 or current_l2 or current_main
                    # "Answer any two" inside a question applies to its sub-parts.
                    if attempt and deepest is current_main:
                        current_main.flags.append(f"attempt_any_{attempt}_subparts")
                    self._add_text(deepest, text, allow_margin)
                continue
            if kind == "marks":
                target = current_l3 or current_l2 or current_main
                value = parse_marks_expr(info["expr"])
                if target is not None and value is not None:
                    if target.marks is None:
                        target.marks, target.marks_expr, target.marks_source = value, info["expr"], "own_line"
                    else:
                        target.flags.append("extra_marks_value")
                continue

            if kind in ("main", "main_sub", "dotted"):
                num = info["num"]
                strength = info["strength"]
                same_q_alternative = pending_or and num == last_main_num
                acceptable = (
                    num <= max_q and (
                        current_main is None and (num == 1 or strength >= 0.9 or not exam.questions)
                        or num == last_main_num + 1
                        or (section_started and (num == 1 or num == last_main_num + 1))
                        or same_q_alternative
                        or (strength >= 0.9 and last_main_num < num <= last_main_num + 3)
                    )
                )
                continuing_same = (current_main is not None and num == last_main_num
                                   and current_main.label == str(num) and not pending_or)
                if kind == "dotted" and continuing_same:
                    # "1.5 Pa.s ..." on a wrapped line is a number, not sub-question 1.5. A dotted label must
                    # continue the sequence (n.1, then n.2, ...) and must not be followed by a unit.
                    prev_dotted = [c for c in current_main.children if c.kind == "dotted"]
                    expected_sub = int(prev_dotted[-1].label.split(".")[-1]) + 1 if prev_dotted else 1
                    if int(info["sub"]) == expected_sub and not RE_UNIT_START.match(info["rest"]):
                        current_l2 = new_child(current_main, f"{num}.{info['sub']}", "dotted", info["rest"], ln)
                        current_l3 = None
                        l2_kind = "dotted"
                        continue
                    kind = "text"
                if kind == "main_sub" and continuing_same:
                    # "1(b)" after "1(a)": another part of the same question.
                    sub = info["sub"].lower()
                    sub_kind = l2_kind or ("roman" if sub in ROMAN_VALUES else "letter")
                    current_l2 = new_child(current_main, sub, sub_kind, info["rest"], ln)
                    current_l3 = None
                    continue
                if acceptable and kind != "text":
                    if kind == "main":
                        new_main(num, info["rest"], ln, strength)
                    else:
                        main = new_main(num, "", ln, strength)
                        if kind == "dotted":
                            sub_kind, sub = "dotted", f"{num}.{info['sub']}"
                        else:
                            sub = info["sub"].lower()
                            sub_kind = "roman" if sub in ROMAN_VALUES else "letter"
                        current_l2 = new_child(main, sub, sub_kind, info["rest"], ln)
                        l2_kind = sub_kind
                    continue
                # Not a plausible new question number: treat as text below.

            if kind in ("letter", "roman_or_letter") and current_main is not None:
                label = info["label"]
                rest = info["rest"]
                if kind == "roman_or_letter":
                    letter_continues = (l2_kind == "letter" and current_l2 is not None
                                        and _is_next_label(current_l2.label, label, "letter"))
                    if letter_continues or (l2_kind == "letter" and current_l2 is not None and current_l2.label == label):
                        current_l2 = new_child(current_main, label, "letter", rest, ln)
                        current_l3 = None
                    elif l2_kind == "letter" and current_l2 is not None:
                        current_l3 = new_child(current_l2, label, "roman", rest, ln)
                    else:
                        current_l2 = new_child(current_main, label, "roman", rest, ln)
                        current_l3 = None
                        l2_kind = "roman"
                    continue
                if l2_kind in (None, "letter"):
                    inner = RE_ROMAN_PAREN.match(rest.strip()) or RE_ROMAN.match(rest.strip())
                    if inner:
                        current_l2 = new_child(current_main, label, "letter", "", ln)
                        current_l3 = new_child(current_l2, inner.group(1).lower(), "roman", inner.group(2), ln)
                    else:
                        current_l2 = new_child(current_main, label, "letter", rest, ln)
                        current_l3 = None
                    l2_kind = "letter"
                    continue
                if l2_kind in ("roman", "dotted") and current_l2 is not None:
                    current_l3 = new_child(current_l2, label, "letter", rest, ln)
                    continue

            # Continuation text.
            target = current_l3 or current_l2 or current_main
            if target is None:
                if section_index is not None:
                    exam.sections[section_index].instructions.append(collapse_whitespace(text))
                else:
                    exam.instructions.append(collapse_whitespace(text))
                continue
            if pending_or:
                # An "OR" followed by plain text: the alternative has no label of its own.
                pending_or = False
                target.flags.append("unlabelled_or_alternative")
            self._add_text(target, text, allow_margin)

    @staticmethod
    def _add_text(node: ParsedQuestion, text: str, allow_margin: bool) -> None:
        text = text.strip()
        if not text:
            return
        cleaned, marks, expr, source, conf = extract_trailing_marks(text, allow_margin=allow_margin)
        if marks is not None:
            if node.marks is None:
                node.marks, node.marks_expr, node.marks_source = marks, expr, source
                if conf < 1.0:
                    node.confidence = min(node.confidence, max(conf, 0.6))
            else:
                node.flags.append("multiple_marks_values")
                node.marks += marks
                node.marks_expr = f"{node.marks_expr}+{expr}"
        if cleaned.strip():
            node.lines.append(cleaned.strip())

    # ------------------------------------------------------------ post-process
    def _postprocess(self, exam: ParsedExam, pages: list[PageText]) -> None:
        page_flags = {p.page_no: p.flags for p in pages}
        page_conf = {p.page_no: p.confidence for p in pages}
        for q in exam.questions:
            for node in list(q.walk()):
                self._detect_mcq(node)
        for q in exam.questions:
            self._reconcile_marks(q)
        self._mark_optional(exam)
        for node in exam.all_nodes():
            if node.is_leaf:
                report = assess_text(node.text, self.settings, context="question")
                has_stem = node.parent is not None and bool(node.parent.text)
                for flag in report.flags:
                    if flag == "very_short_question" and (has_stem or node.options):
                        continue  # "Pitot tube" under "Write short notes on" is complete.
                    if flag not in node.flags:
                        node.flags.append(flag)
            flags = page_flags.get(node.page_no) or []
            if any(f in flags for f in ("low_ocr_confidence", "garbled_symbols", "implausible_words",
                                         "needs_ocr_but_unavailable", "garbled_text_layer")):
                node.flags.append("source_page_low_quality")
                conf = page_conf.get(node.page_no)
                node.confidence = min(node.confidence, (conf / 100.0) if conf else 0.6)
            if not node.text and node.is_leaf and not node.options:
                node.flags.append("empty_question_text")
        # Main question numbering should increase by one.
        nums = [int(q.label) for q in exam.questions if q.label.isdigit()]
        if nums and nums[0] != 1 and not exam.sections:
            exam.warnings.append(f"The first question is numbered {nums[0]}, not 1. Earlier questions may be missing.")

    def _detect_mcq(self, node: ParsedQuestion) -> None:
        max_words = int(self.cfg.mcq_max_option_words)
        if len(node.children) >= 3:
            labels = [c.label for c in node.children]
            # Only the last option may carry marks (the marks of the whole MCQ printed on its line).
            short = all(word_count(c.text) <= max_words and not c.children and (c.marks is None or c is node.children[-1])
                        for c in node.children)
            abcd = labels[:4] in (["a", "b", "c", "d"], ["a", "b", "c"], ["i", "ii", "iii", "iv"])
            if short and abcd and RE_MCQ_CUE.search(node.text or ""):
                last = node.children[-1]
                if last.marks is not None and node.marks is None:
                    node.marks, node.marks_expr, node.marks_source = last.marks, last.marks_expr, last.marks_source
                node.options = [c.text for c in node.children]
                node.children = []
                node.flags.append("mcq_options_collapsed")
                return
        if node.is_leaf and not node.options:
            matches = RE_INLINE_OPTIONS.findall(node.text)
            if len(matches) >= 3 and [m[0].lower() for m in matches[:3]] == ["a", "b", "c"]:
                first = re.search(r"\(\s*[aA]\s*\)", node.text)
                stem = node.text[: first.start()].strip() if first else node.text
                if RE_MCQ_CUE.search(stem) or len(stem.split()) >= 4:
                    node.options = [collapse_whitespace(m[1]) for m in matches]
                    node.lines = [stem]

    def _reconcile_marks(self, node: ParsedQuestion) -> None:
        for child in node.children:
            self._reconcile_marks(child)
        if not node.children:
            return
        slots = _alternative_slots(node.children)
        last = node.children[-1]
        if (node.marks is None and len(node.children) >= 2 and last.marks is not None
                and all(c.marks is None for c in node.children[:-1]) and not last.children):
            # "(a) What is X? (b) How is X related to Y? 10 marks": the marks cover the whole question.
            node.marks, node.marks_expr, node.marks_source = last.marks, last.marks_expr, last.marks_source
            last.marks, last.marks_expr, last.marks_source = None, "", ""
            node.flags.append("marks_moved_to_parent")
        child_marks = [c.marks for c in node.children]
        if node.marks is None:
            if all(m is not None for m in child_marks):
                node.marks = float(sum(max(c.marks for c in slot) for slot in slots))  # type: ignore[type-var]
                node.marks_source = "sum_of_parts"
        else:
            known = [m for m in child_marks if m is not None]
            if len(known) == len(child_marks):
                total = sum(max(c.marks for c in slot) for slot in slots)  # type: ignore[type-var]
                if abs(total - node.marks) > 0.01 and abs(total - node.marks) / max(node.marks, 1) > 0.05:
                    node.flags.append("marks_mismatch_with_parts")
            elif not known:
                share = node.marks / len(slots)
                attempt = _inline_attempt(node.text) or _attempt_from_flags(node.flags)
                factors = [float(x) for x in re.split(r"[x×X*]", node.marks_expr) if x] if re.fullmatch(
                    r"\d+(?:\.\d)?[x×X*]\d+(?:\.\d)?", node.marks_expr or "") else []
                if len(factors) == 2:
                    # "[2x4]" with "any two" means two parts worth 4 marks each.
                    a, b = factors
                    if attempt and int(a) == attempt or int(a) == len(slots):
                        share = b
                    elif attempt and int(b) == attempt or int(b) == len(slots):
                        share = a
                elif attempt and attempt < len(slots):
                    share = node.marks / attempt
                for slot in slots:
                    for c in slot:
                        c.marks = round(share, 2)
                        c.marks_source = "distributed"

    def _mark_optional(self, exam: ParsedExam) -> None:
        for node in exam.all_nodes():
            if node.or_group:
                node.is_optional = True
        total_main = len({q.or_group or id(q) for q in exam.questions})
        if exam.attempt_count and exam.attempt_count < total_main and not exam.sections:
            for q in exam.questions:
                q.is_optional = True
        for idx, sec in enumerate(exam.sections):
            in_sec = [q for q in exam.questions if q.section_index == idx]
            if sec.attempt_count and sec.attempt_count < len(in_sec):
                for q in in_sec:
                    q.is_optional = True
        for node in exam.all_nodes():
            if not node.children:
                continue
            attempt = _attempt_from_flags(node.flags)
            if attempt is None:
                attempt = _inline_attempt(node.text)
                if attempt:
                    node.flags.append(f"attempt_any_{attempt}_subparts")
            if attempt and attempt < len(node.children):
                for c in node.children:
                    c.is_optional = True


def _alternative_slots(children: list[ParsedQuestion]) -> list[list[ParsedQuestion]]:
    slots: list[list[ParsedQuestion]] = []
    by_group: dict[str, list[ParsedQuestion]] = {}
    for c in children:
        if c.or_group:
            if c.or_group not in by_group:
                by_group[c.or_group] = []
                slots.append(by_group[c.or_group])
            by_group[c.or_group].append(c)
        else:
            slots.append([c])
    return slots


def _is_next_label(prev: str, label: str, kind: str) -> bool:
    if kind == "letter":
        return len(prev) == 1 and len(label) == 1 and ord(label) == ord(prev) + 1
    if kind == "roman":
        return prev in ROMAN_VALUES and label in ROMAN_VALUES and ROMAN_VALUES[label] == ROMAN_VALUES[prev] + 1
    return True


def _next_label(label: str) -> str | None:
    if label in ROMAN_VALUES:
        inverse = {v: k for k, v in ROMAN_VALUES.items()}
        return inverse.get(ROMAN_VALUES[label] + 1)
    if len(label) == 1 and label.isalpha() and label.lower() < "z":
        return chr(ord(label) + 1)
    return None


RE_LEADING_PAREN_LABEL = re.compile(r"^\(\s*([a-zA-Z]|i{1,3}|iv|vi{0,3}|ix|x)\s*\)")


def split_sequential_labels(text: str) -> list[str]:
    """Split "(i) Carnot cycle (ii) Clausius inequality" at the next label in sequence."""
    pieces = [text]
    while True:
        last = pieces[-1]
        m = RE_LEADING_PAREN_LABEL.match(last.strip())
        if not m:
            return pieces
        nxt = _next_label(m.group(1).lower())
        if not nxt:
            return pieces
        pos = re.search(rf"\s\(\s*{nxt}\s*\)\s", last, re.IGNORECASE)
        if not pos:
            return pieces
        pieces[-1] = last[: pos.start()].rstrip()
        pieces.append(last[pos.start():].strip())


def _attempt_from_flags(flags: list[str]) -> int | None:
    for flag in flags:
        m = re.match(r"attempt_any_(\d+)_subparts", flag)
        if m:
            return int(m.group(1))
    return None


RE_ANY_INLINE = re.compile(r"\(?\s*(?:attempt|answer)?\s*any\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s*\)?", re.IGNORECASE)


def _inline_attempt(text: str) -> int | None:
    m = RE_ANY_INLINE.search(text or "")
    if not m:
        return None
    raw = m.group(1).lower()
    return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)


def _attempt_count(text: str) -> int | None:
    m = RE_ATTEMPT.search(text or "")
    if not m:
        return None
    raw = m.group(1).lower()
    return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
