"""Generate the synthetic demo course (syllabus + 12 past papers) as PDF, DOCX and TXT files.

The papers follow the planted process in ``bank.PROFILES``. ``generate_demo`` returns the
ground truth (which topics each paper covers) so tests can measure what the pipeline
recovers. All content is synthetic and labelled as such.
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import shutil

from .bank import BANK, OUT_OF_SYLLABUS_BANK, PROFILES, SYLLABUS_TEXT, TOPIC_NUMBERS

YEARS = list(range(2014, 2026))
SHORT_NOTES = {
    "fluid_continuum": "Continuum concept", "viscosity": "Newtonian and non-Newtonian fluids",
    "surface_tension": "Capillarity", "compressibility": "Cavitation",
    "pressure_measurement": "Differential manometer", "hydrostatic_forces": "Centre of pressure",
    "buoyancy": "Metacentric height", "flow_types": "Streakline", "continuity": "Continuity equation",
    "potential_flow": "Flow net", "bernoulli": "Assumptions of Bernoulli's equation",
    "flow_measurement": "Pitot tube", "orifices_notches": "Notches and weirs", "momentum": "Impact of jets",
    "laminar_flow": "Hagen-Poiseuille flow", "darcy": "Moody diagram", "minor_losses": "Minor losses in pipes",
    "pipe_networks": "Hydraulic gradient line", "dimensional_analysis": "Buckingham pi theorem",
    "similitude": "Model laws", "boundary_layer": "Displacement thickness",
    "separation_drag": "Boundary layer separation",
}
UNIT_OF = {k: int(v.split(".")[0]) for k, v in TOPIC_NUMBERS.items()}


@dataclass
class DemoExam:
    year: int
    lines: list[str]
    topics: list[str] = field(default_factory=list)
    oos: list[str] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)


def _probability(profile: str, t: int, appeared_last: bool, key: str) -> float:
    if profile == "core":
        return 0.85
    if profile == "regular":
        return 0.45
    if profile == "rare":
        return 0.12
    if profile == "alt_even":
        return 0.9 if t % 2 == 0 else 0.08
    if profile == "alt_odd":
        return 0.9 if t % 2 == 1 else 0.08
    if profile == "every3":
        offset = 0 if key == "pipe_networks" else 1
        return 0.85 if t % 3 == offset else 0.1
    if profile == "cooldown":
        return 0.15 if appeared_last else 0.8
    if profile == "emerging":
        return 0.06 if t < 7 else 0.85
    if profile == "fading":
        return 0.8 if t < 6 else 0.04
    return 0.3


def _pick_question(rng: random.Random, key: str, t: int, used: dict[str, list[int]]) -> int:
    bank = BANK[key]
    weights = []
    for i, (qtype, _text, _marks, _fam) in enumerate(bank):
        w = 1.0
        if key == "bernoulli":  # planted type shift: derivations early, numericals later
            w = (3.0 if qtype == "derivation" else 0.4) if t < 6 else (3.0 if qtype in ("numerical", "application") else 0.4)
        if used.get(key) and i == used[key][-1]:
            w *= 0.35  # exact repeats happen, but less often than fresh questions
        weights.append(w)
    return rng.choices(range(len(bank)), weights=weights, k=1)[0]


def build_exams(seed: int = 2024) -> list[DemoExam]:
    rng = random.Random(seed)
    appeared: dict[str, list[int]] = {k: [] for k in BANK}
    used: dict[str, list[int]] = {}
    exams: list[DemoExam] = []
    for t, year in enumerate(YEARS):
        probs = {k: _probability(PROFILES[k], t, bool(appeared[k] and appeared[k][-1] == t - 1), k) for k in BANK}
        chosen = [k for k in BANK if rng.random() < probs[k]]
        chosen.sort(key=lambda k: -probs[k] + rng.random() * 0.01)
        if len(chosen) > 13:
            chosen = chosen[:13]
        while len(chosen) < 11:
            rest = sorted((k for k in BANK if k not in chosen), key=lambda k: -probs[k] - rng.random() * 0.2)
            chosen.append(rest[0])
        for k in chosen:
            appeared[k].append(t)
        rng.shuffle(chosen)
        # Short-note slot uses three topics; the rest become (a)/(b) parts of five questions.
        notes, parts = chosen[:3], chosen[3:]
        parts.sort(key=lambda k: UNIT_OF[k])
        oos_keys = ["turbines" if t % 2 == 0 else "pumps"] if t < 4 else []
        exam = DemoExam(year=year, lines=[], topics=sorted(chosen), oos=oos_keys)
        lines = [
            "SYNTHETIC DEMO UNIVERSITY (example data, not a real exam)",
            "Faculty of Engineering - Examination Control Office",
            f"{year} Fall",
            "Exam.        Regular",
            "Level   BE      Full Marks   80",
            "Programme  BCE   Pass Marks  32",
            "Year / Part  II / I   Time  3 hrs.",
            "Subject: - Fluid Mechanics (CE 501)",
            "✓ Candidates are required to give their answers in their own words as far as practicable.",
            "✓ Attempt All questions.",
            "✓ The figures in the margin indicate Full Marks.",
            "✓ Assume suitable data if necessary.",
            "",
        ]
        qno = 0
        pairs = [parts[i:i + 2] for i in range(0, len(parts), 2)]
        or_inserted = False
        for pair in pairs:
            qno += 1
            for j, key in enumerate(pair):
                idx = _pick_question(rng, key, t, used)
                used.setdefault(key, []).append(idx)
                qtype, text, marks, fam = BANK[key][idx]
                label = "a" if j == 0 else "b"
                prefix = f"{qno}. {label})" if j == 0 else f"   {label})"
                if len(pair) == 1:
                    prefix = f"{qno}."
                lines.append(f"{prefix} {text} [{marks:g}]")
                exam.questions.append({"q": f"{qno}{'' if len(pair) == 1 else '(' + label + ')'}",
                                       "topic": key, "type": qtype, "marks": marks, "family": f"{key}#{fam}",
                                       "text": text})
            if qno == 2 and oos_keys and not or_inserted:
                # An OR alternative drawn from the old syllabus (out of the current syllabus).
                okey = oos_keys[0]
                qtype, text, marks, fam = OUT_OF_SYLLABUS_BANK[okey][t % len(OUT_OF_SYLLABUS_BANK[okey])]
                lines.append("OR")
                lines.append(f"{qno}. {text} [{marks:g}]")
                exam.questions.append({"q": f"{qno}-OR", "topic": None, "oos": okey, "type": qtype,
                                       "marks": marks, "family": f"{okey}#{fam}", "text": text})
                or_inserted = True
        qno += 1
        lines.append(f"{qno}. Write short notes on (any two):          [2×4]")
        for roman, key in zip(("i", "ii", "iii"), notes):
            lines.append(f"   {roman}) {SHORT_NOTES[key]}")
            exam.questions.append({"q": f"{qno}({roman})", "topic": key, "type": "short_answer", "marks": 4,
                                   "family": f"{key}#note", "text": SHORT_NOTES[key]})
        exam.lines = lines
        exams.append(exam)
    return exams


# --------------------------------------------------------------------------- writers
def _write_pdf(path: Path, lines: list[str]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50
    for line in lines:
        marks = ""
        body = line
        if line.rstrip().endswith("]") and "[" in line:
            cut = line.rfind("[")
            body, marks = line[:cut].rstrip(), line[cut:].strip()
        wrapped = textwrap.wrap(body, 88, subsequent_indent="      ") or [""]
        for i, part in enumerate(wrapped):
            if y < 60:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica", 10)
            c.drawString(50, y, part)
            if marks and i == len(wrapped) - 1:
                c.drawRightString(width - 50, y, marks)
            y -= 15
    c.save()


def _write_docx(path: Path, lines: list[str]) -> None:
    import docx

    doc = docx.Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(str(path))


def generate_demo(out_dir: Path | str, seed: int = 2024) -> dict:
    """Write the demo files into ``out_dir`` and return the ground truth."""
    out = Path(out_dir)
    (out / "syllabus").mkdir(parents=True, exist_ok=True)
    (out / "exams").mkdir(parents=True, exist_ok=True)
    _write_pdf(out / "syllabus" / "fluid_mechanics_syllabus.pdf", SYLLABUS_TEXT.split("\n"))
    exams = build_exams(seed)
    truth = {"years": [], "topics": {}, "oos": {}, "questions": {}}
    for i, exam in enumerate(exams):
        stem = f"fluid_mechanics_{exam.year}"
        if i % 3 == 1:
            _write_docx(out / "exams" / f"{stem}.docx", exam.lines)
        elif i % 3 == 2:
            (out / "exams" / f"{stem}.txt").write_text("\n".join(exam.lines), encoding="utf-8")
        else:
            _write_pdf(out / "exams" / f"{stem}.pdf", exam.lines)
        truth["years"].append(exam.year)
        truth["topics"][str(exam.year)] = exam.topics
        truth["oos"][str(exam.year)] = exam.oos
        truth["questions"][str(exam.year)] = exam.questions
    # A byte-identical copy (duplicate file) and the same paper in another format (duplicate paper).
    first = exams[0]
    shutil.copyfile(out / "exams" / f"fluid_mechanics_{first.year}.pdf",
                    out / "exams" / f"fluid_mechanics_{first.year}_copy.pdf")
    (out / "exams" / f"fluid_mechanics_{exams[3].year}_retyped.txt").write_text(
        "\n".join(exams[3].lines), encoding="utf-8")
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "Synthetic demo data for Exam Predictor. These papers are generated, not real exams.\n"
        "Planted patterns are described in predictor/demo/bank.py (PROFILES).\n", encoding="utf-8")
    return truth
