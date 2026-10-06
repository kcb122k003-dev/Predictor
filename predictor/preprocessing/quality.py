"""Extraction quality checks (spec section 36).

These checks never change text. They attach flags so that low-quality extraction is
visible in the review screen and is not silently treated as clean data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config.settings import Settings
from .textnorm import word_count

VOWELS = set("aeiouyAEIOUY")
_ALLOWED_SYMBOLS = set(".,;:!?()[]{}'\"-+=*/^_%<>|&@#$~`\\ \n")
_MATH_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ∫∂∑√≈≤≥±∇∞×·÷°∝→⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉•")
_REPEAT = re.compile(r"(.)\1{3,}")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}", re.IGNORECASE)
_DIGIT_INSIDE_WORD = re.compile(r"^[A-Za-z]+[0-9][A-Za-z]+$")
_CHEM_OR_UNIT = re.compile(r"^(?:[A-Z][a-z]?\d*){1,4}$|^[a-z]{1,3}\d$|^\d+[a-zA-Z]{1,4}$")
_WORD = re.compile(r"\S+")


@dataclass
class QualityReport:
    flags: list[str] = field(default_factory=list)
    symbol_ratio: float = 0.0
    implausible_ratio: float = 0.0
    words: int = 0

    @property
    def ok(self) -> bool:
        return not self.flags


def implausible_token(token: str) -> bool:
    """Heuristic for OCR noise tokens: no vowels, long consonant runs, l0ad-style digits."""
    core = token.strip(".,;:!?()[]{}'\"")
    if len(core) < 4 or not any(c.isalpha() for c in core):
        return False
    if any(c in _MATH_CHARS for c in core):
        return False
    if core.isupper() and len(core) <= 6:  # abbreviations such as COP, NPSH, LMTD
        return False
    if _CHEM_OR_UNIT.match(core):
        return False
    letters = [c for c in core if c.isalpha()]
    if letters and not any(c in VOWELS for c in letters) and len(letters) >= 4:
        return True
    if _CONSONANT_RUN.search(core):
        return True
    if _REPEAT.search(core):
        return True
    if _DIGIT_INSIDE_WORD.match(core):
        return True
    # Mixed-case noise such as "tHe" or "dErIvE".
    inner = core[1:]
    if sum(1 for c in inner if c.isupper()) >= 2 and sum(1 for c in inner if c.islower()) >= 2:
        if not re.match(r"^[A-Z][a-z]+[A-Z][a-z]+$", core):  # allow CamelCase names
            return True
    return False


def assess_text(text: str, settings: Settings, *, ocr_confidence: float | None = None,
                context: str = "page") -> QualityReport:
    """Return quality flags for a page or a question."""
    report = QualityReport()
    stripped = (text or "").strip()
    report.words = word_count(stripped)
    q = settings.quality

    if ocr_confidence is not None and ocr_confidence < settings.ocr.low_confidence:
        report.flags.append("low_ocr_confidence")

    if not stripped:
        report.flags.append("empty_text")
        return report

    visible = [c for c in stripped if not c.isspace()]
    odd = sum(1 for c in visible if not c.isalnum() and c not in _ALLOWED_SYMBOLS and c not in _MATH_CHARS)
    odd += stripped.count("�") * 3
    report.symbol_ratio = odd / max(len(visible), 1)
    if report.symbol_ratio > q.suspicious_symbol_ratio:
        report.flags.append("garbled_symbols")

    tokens = [t for t in _WORD.findall(stripped) if any(c.isalpha() for c in t)]
    if tokens:
        bad = sum(1 for t in tokens if implausible_token(t))
        report.implausible_ratio = bad / len(tokens)
        min_tokens = 4 if context == "question" else 8
        if len(tokens) >= min_tokens and report.implausible_ratio > q.suspicious_token_ratio:
            report.flags.append("implausible_words")

    if _unbalanced(stripped):
        report.flags.append("unbalanced_brackets")
    if _broken_math(stripped):
        report.flags.append("possible_broken_equation")

    if context == "question":
        if report.words < q.min_question_words:
            report.flags.append("very_short_question")
        elif report.words > q.max_question_words:
            report.flags.append("very_long_question_possible_merge")
    return report


_LIST_LABEL = re.compile(r"(?:(?<=\s)|^)\(?(?:[a-zA-Z]|[ivxIVX]{1,4}|\d{1,2})\)", re.MULTILINE)


def _unbalanced(text: str) -> bool:
    # List labels such as "a)", "ii)" and "3)" legitimately have no opening bracket.
    text = _LIST_LABEL.sub(" ", text)
    for open_c, close_c in ("()", "[]", "{}"):
        if abs(text.count(open_c) - text.count(close_c)) >= 2:
            return True
    return False


def _broken_math(text: str) -> bool:
    # Operators with nothing on one side, or runs of operators, often come from OCR of equations.
    if re.search(r"(?:^|\s)[=+*/^](?:\s|$)[=+*/^]", text):
        return True
    if re.search(r"[=+*/^]{3,}", text):
        return True
    if re.search(r"=\s*$", text.strip()) and not text.strip().endswith("=?"):
        return True
    return False


def garbage_ratio(text: str) -> float:
    """Share of characters that are replacement/private-use/control characters (PDF text layers)."""
    if not text:
        return 1.0
    bad = 0
    for ch in text:
        code = ord(ch)
        if ch == "�" or 0xE000 <= code <= 0xF8FF or (code < 32 and ch not in "\n\t\r"):
            bad += 1
    visible = sum(1 for c in text if not c.isspace())
    return bad / max(visible, 1)
