"""Text cleaning, math-aware normalisation, tokenisation and stemming.

Two representations are kept for every question (spec section 37):

* ``raw``: the text as extracted, with Unicode symbols (α, ∫, H₂O, x²) intact, for display.
* ``normalized``: symbols spelled out (alpha, integral, H2O, x^2) so keyword matching and
  embeddings see consistent tokens.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("english")
except Exception:  # pragma: no cover - optional dependency missing
    _STEMMER = None

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-",  # minus sign
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "…": "...", "•": "•", "": "•", "": "•", "▪": "•",
    "●": "•", "◦": "•", "‣": "•", "⁃": "•",
}
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "µ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
    "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi",
    "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
}
MATH_SYMBOLS = {
    "∫": " integral ", "∬": " double integral ", "∮": " contour integral ", "∂": " partial ",
    "∇": " nabla ", "∑": " sum ", "∏": " product ", "√": " sqrt ", "∞": " infinity ",
    "≈": " approx ", "≃": " approx ", "≅": " approx ", "≠": " != ", "≤": " <= ", "≥": " >= ",
    "±": " +/- ", "∓": " -/+ ", "×": " x ", "·": " * ", "÷": " / ", "∝": " proportional to ",
    "→": " -> ", "⇒": " => ", "⟶": " -> ", "↔": " <-> ", "⇌": " <=> ", "∈": " in ",
    "∀": " for all ", "∃": " exists ", "°": " deg ", "℃": " degC ", "℉": " degF ", "Å": " angstrom ",
    "‰": " per mille ", "∆": " Delta ", "Ω": " Omega ",
}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ", "0123456789+-=()aeoxhklmnpst")
_SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"
_SUB_CHARS = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ"
_FRACTIONS = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅛": "1/8"}

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can cannot could couldn't did didn't do does doesn't doing don't
down during each few for from further had hadn't has hasn't have haven't having he her here hers
herself him himself his how i if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same shan't
she should shouldn't so some such than that that's the their theirs them themselves then there
there's these they this those through to too under until up very was wasn't we were weren't what
when where which while who whom why with won't would wouldn't you your yours yourself yourselves
also may might must shall will within without upon via per etc ie eg i.e e.g. respectively
using used one two three four five six seven eight nine ten first second third its whose
""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:['’][a-z]+)?|\d+(?:\.\d+)?")


def clean_text(text: str) -> str:
    """Normalise Unicode artefacts while preserving line structure and math symbols."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = _CONTROL.sub(" ", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    # Collapse runs of more than two blank lines.
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip("\n")


def _replace_script_runs(text: str, chars: str, table: dict[int, int] | dict, marker: str) -> str:
    pattern = re.compile(f"[{re.escape(chars)}]+")
    return pattern.sub(lambda m: f"{marker}{m.group(0).translate(table)}", text)


def normalize_math(text: str) -> str:
    """Spell out math notation: Greek letters, operators, super/subscripts, fractions."""
    if not text:
        return ""
    for src, dst in _FRACTIONS.items():
        text = text.replace(src, dst)
    # Chemical formulas: subscript digits directly after letters become plain digits (H₂O -> H2O).
    text = re.sub(f"(?<=[A-Za-z)])([{_SUB_CHARS[:10]}]+)",
                  lambda m: m.group(1).translate(_SUBSCRIPTS), text)
    text = _replace_script_runs(text, _SUP_CHARS, _SUPERSCRIPTS, "^")
    text = _replace_script_runs(text, _SUB_CHARS, _SUBSCRIPTS, "_")
    out = []
    for ch in text:
        if ch in GREEK:
            out.append(f" {GREEK[ch]} ")
        elif ch in MATH_SYMBOLS:
            out.append(MATH_SYMBOLS[ch])
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Single-line, lower-case, math-normalised text for similarity and keyword work."""
    text = normalize_math(clean_text(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


_OPERATOR = re.compile(r"[=∫∂∑√≈≤≥±∇∝]|\^|d[a-zA-Z]/d[a-zA-Z]|[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]")
_MATHY_TOKEN = re.compile(
    r"^(?:[^A-Za-z]+|[A-Za-z]{1,2}[0-9_^'’]*|[A-Za-z]+[0-9_^/()=+\-*]+.*|.*[=∫∂∑√≈≤≥±∇∝^/()+*].*|"
    r"[α-ωΑ-Ω].*|.*[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉].*)$")


def extract_equations(text: str) -> list[dict[str, str]]:
    """Find equation-like spans inside prose and keep raw and normalised forms.

    A span is a maximal run of "math-looking" tokens (symbols, digits, one- or two-letter
    variables, Greek letters) that contains at least one operator, for example
    ``Cp - Cv = R`` inside "Show that Cp - Cv = R for an ideal gas".
    """
    found: list[dict[str, str]] = []
    for line in clean_text(text).split("\n"):
        tokens = line.split()
        run: list[str] = []

        def flush() -> None:
            span = " ".join(run).strip(" ,.;:")
            if len(span) >= 3 and _OPERATOR.search(span) and sum(c.isalnum() for c in span) >= 2:
                found.append({"raw": span, "normalized": normalize_math(span)})

        for tok in tokens:
            if _MATHY_TOKEN.match(tok) and tok.lower() not in {"a", "an", "is", "of", "to", "in", "if", "at", "on", "by", "or"}:
                run.append(tok)
            else:
                flush()
                run = []
        flush()
    return found


def tokenize(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOKEN_RE.findall(text)]


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    token = token.lower()
    if token.endswith("'s"):
        token = token[:-2]
    token = token.replace("'", "")
    if _STEMMER is not None:
        return _STEMMER.stemWord(token)
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):  # pragma: no cover
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token  # pragma: no cover


def content_terms(text: str, extra_stopwords: Iterable[str] = (), *, keep_numbers: bool = False) -> list[str]:
    """Stemmed content words: no stopwords, no instruction words, no bare numbers."""
    extra = {w.lower() for w in extra_stopwords}
    extra_stems = {stem(w) for w in extra}
    terms: list[str] = []
    for tok in tokenize(normalize_math(text)):
        if tok in STOPWORDS or tok in extra:
            continue
        if tok[0].isdigit() and not keep_numbers:
            continue
        if len(tok) < 2:
            continue
        st = stem(tok)
        if st in extra_stems:
            continue
        terms.append(st)
    return terms


def word_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
