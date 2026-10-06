"""DOCX extraction in document order: paragraphs, automatic numbering, tables, equations.

Word's automatic list numbers ("1.", "a)", "(i)") are not part of the paragraph text.
Question papers rely on them, so this reader rebuilds them from ``numbering.xml``.
Page numbers are approximate: they follow explicit and rendered page breaks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..preprocessing.textnorm import clean_text
from ..utils.logging import get_logger, log_event
from .types import ExtractionResult, PageText

log = get_logger("ingestion.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = f"{{{W_NS}}}"
M = f"{{{M_NS}}}"


def _to_roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for value, sym in vals:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def _to_letter(n: int) -> str:
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def _format_number(fmt: str, n: int) -> str:
    if fmt == "lowerLetter":
        return _to_letter(n)
    if fmt == "upperLetter":
        return _to_letter(n).upper()
    if fmt == "lowerRoman":
        return _to_roman(n)
    if fmt == "upperRoman":
        return _to_roman(n).upper()
    if fmt == "bullet":
        return "•"
    if fmt in ("none", ""):
        return ""
    return str(n)


class _Numbering:
    """Resolves (numId, ilvl) to formatted labels, keeping counters like Word does."""

    def __init__(self, document: Any):
        self.levels: dict[str, dict[int, dict[str, Any]]] = {}
        self.counters: dict[str, list[int]] = defaultdict(lambda: [0] * 9)
        try:
            root = document.part.numbering_part.element
        except Exception:
            return
        abstract: dict[str, dict[int, dict[str, Any]]] = {}
        for absn in root.findall(f"{W}abstractNum"):
            aid = absn.get(f"{W}abstractNumId")
            lvls: dict[int, dict[str, Any]] = {}
            for lvl in absn.findall(f"{W}lvl"):
                ilvl = int(lvl.get(f"{W}ilvl", "0"))
                fmt_el = lvl.find(f"{W}numFmt")
                text_el = lvl.find(f"{W}lvlText")
                start_el = lvl.find(f"{W}start")
                lvls[ilvl] = {
                    "fmt": fmt_el.get(f"{W}val") if fmt_el is not None else "decimal",
                    "text": text_el.get(f"{W}val") if text_el is not None else f"%{ilvl + 1}.",
                    "start": int(start_el.get(f"{W}val")) if start_el is not None else 1,
                }
            abstract[aid] = lvls
        for num in root.findall(f"{W}num"):
            nid = num.get(f"{W}numId")
            ref = num.find(f"{W}abstractNumId")
            if ref is not None and ref.get(f"{W}val") in abstract:
                self.levels[nid] = abstract[ref.get(f"{W}val")]

    def label(self, num_id: str, ilvl: int) -> str:
        lvls = self.levels.get(num_id)
        if not lvls or ilvl not in lvls:
            return ""
        counters = self.counters[num_id]
        if counters[ilvl] == 0:
            counters[ilvl] = lvls[ilvl]["start"]
        else:
            counters[ilvl] += 1
        for deeper in range(ilvl + 1, 9):
            counters[deeper] = 0
        text = lvls[ilvl]["text"]
        for level in range(ilvl + 1):
            fmt = lvls.get(level, {"fmt": "decimal"})["fmt"]
            value = counters[level] or lvls.get(level, {"start": 1})["start"]
            text = text.replace(f"%{level + 1}", _format_number(fmt, value))
        return text.strip()


def _paragraph_text(p_el) -> tuple[str, int]:
    """Text of a paragraph element, including OMML math; also counts page breaks inside it."""
    parts: list[str] = []
    breaks = 0
    for node in p_el.iter():
        tag = node.tag
        if tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif tag == f"{M}t" and node.text:
            parts.append(node.text)
        elif tag == f"{W}tab":
            parts.append("    ")
        elif tag == f"{W}br":
            if node.get(f"{W}type") == "page":
                breaks += 1
            else:
                parts.append("\n")
        elif tag == f"{W}lastRenderedPageBreak":
            breaks += 1
    return "".join(parts), breaks


def _numpr(ppr) -> tuple[str | None, int | None]:
    if ppr is None:
        return None, None
    numpr = ppr.find(f"{W}numPr")
    if numpr is None:
        return None, None
    num_id_el = numpr.find(f"{W}numId")
    ilvl_el = numpr.find(f"{W}ilvl")
    return (num_id_el.get(f"{W}val") if num_id_el is not None else None,
            int(ilvl_el.get(f"{W}val")) if ilvl_el is not None else None)


class _StyleNumbering:
    """Numbering defined on paragraph styles ("List Number", "List Number 2"), following basedOn."""

    def __init__(self, document: Any):
        self.styles: dict[str, Any] = {}
        try:
            for st in document.styles.element.findall(f"{W}style"):
                self.styles[st.get(f"{W}styleId")] = st
        except Exception:
            pass

    def lookup(self, style_id: str | None, depth: int = 0) -> tuple[str | None, int | None]:
        if not style_id or style_id not in self.styles or depth > 8:
            return None, None
        st = self.styles[style_id]
        num_id, ilvl = _numpr(st.find(f"{W}pPr"))
        if num_id is not None:
            return num_id, ilvl
        based = st.find(f"{W}basedOn")
        return self.lookup(based.get(f"{W}val") if based is not None else None, depth + 1)


def _numbering_of(p_el, style_numbering: "_StyleNumbering | None" = None) -> tuple[str | None, int]:
    ppr = p_el.find(f"{W}pPr")
    num_id, ilvl = _numpr(ppr)
    if num_id is None and style_numbering is not None and ppr is not None:
        style = ppr.find(f"{W}pStyle")
        s_num, s_ilvl = style_numbering.lookup(style.get(f"{W}val") if style is not None else None)
        num_id = s_num
        ilvl = ilvl if ilvl is not None else s_ilvl
    if num_id is None:
        return None, 0
    return num_id, ilvl or 0


def _style_name(document: Any, p_el) -> str:
    ppr = p_el.find(f"{W}pPr")
    if ppr is None:
        return ""
    style = ppr.find(f"{W}pStyle")
    if style is None:
        return ""
    style_id = style.get(f"{W}val", "")
    try:
        return document.styles.get_by_id(style_id, 1).name or style_id  # 1 = paragraph style
    except Exception:
        return style_id


def extract_docx(path: Path) -> ExtractionResult:
    import docx  # python-docx

    document = docx.Document(str(path))
    numbering = _Numbering(document)
    style_numbering = _StyleNumbering(document)
    pages: dict[int, list[str]] = defaultdict(list)
    headings: list[dict[str, Any]] = []
    page = 1
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == f"{W}p":
            text, breaks = _paragraph_text(child)
            num_id, ilvl = _numbering_of(child, style_numbering)
            label = numbering.label(num_id, ilvl) if num_id and num_id != "0" else ""
            style = _style_name(document, child)
            indent = "    " * ilvl if label else ""
            line = f"{indent}{label} {text}".rstrip() if label else text
            if style.lower().startswith("heading") or style.lower() == "title":
                headings.append({"page": page, "text": text.strip(), "style": style})
            pages[page].append(line)
            page += breaks
        elif child.tag == f"{W}tbl":
            for row in child.iter(f"{W}tr"):
                cells = []
                for cell in row.iter(f"{W}tc"):
                    cell_lines = []
                    for p in cell.iter(f"{W}p"):
                        t, _ = _paragraph_text(p)
                        num_id, ilvl = _numbering_of(p, style_numbering)
                        label = numbering.label(num_id, ilvl) if num_id and num_id != "0" else ""
                        t = f"{label} {t}".strip() if label else t.strip()
                        if t:
                            cell_lines.append(t)
                    cells.append(" ".join(cell_lines))
                cells = [c for c in cells if c]
                if cells:
                    # Wide separators keep a trailing marks column detectable as margin marks.
                    pages[page].append("    ".join(cells))
            pages[page].append("")
        elif child.tag == f"{W}sectPr":
            continue
    result = ExtractionResult(pages=[], file_type="docx")
    for page_no in sorted(pages):
        text = clean_text("\n".join(pages[page_no]))
        if text.strip():
            result.pages.append(PageText(page_no, text, "docx", None, [], {"headings": [
                h for h in headings if h["page"] == page_no]}))
    if not result.pages:
        result.warnings.append("The document contains no extractable text.")
    log_event(log, "docx_extracted", file=path.name, pages=len(result.pages))
    return result
