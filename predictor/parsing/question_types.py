"""Rule-based, extensible question-type classification (spec section 8).

The taxonomy lives in ``config/taxonomy.json``; a ``taxonomy.json`` in the data folder
can add types or override patterns. Each type belongs to a *format* group (definition,
theory, derivation, numerical, diagram, objective) used for type-transition analysis,
because transitions between a dozen fine-grained types would be too sparse to learn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TAXONOMY = Path(__file__).resolve().parent.parent / "config" / "taxonomy.json"
DERIVED_TYPES = {
    "long_theory": {"label": "Long-form theory", "format": "theory"},
    "mixed": {"label": "Mixed", "format": "mixed"},
    "unclassified": {"label": "Unclassified", "format": "theory"},
}
UNIT_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:kg|g|mg|m|cm|mm|km|s|sec|min|hrs?|K|°C|deg\s?C|degC|kPa|MPa|GPa|Pa|bar|atm|"
    r"N|kN|J|kJ|MJ|W|kW|MW|mol|kmol|L|lit(?:re|er)?s?|ml|m3|m\^3|m2|m\^2|m/s|rpm|V|kV|A|mA|ohm|Hz|kHz|%|"
    r"ppm|kcal|cal|BTU|psi|ft|in|lb|kmph|km/h)(?![A-Za-z])")
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")


@dataclass
class TypeResult:
    types: list[str]
    scores: dict[str, float]
    primary: str
    format: str
    confidence: float
    evidence: dict[str, list[str]] = field(default_factory=dict)


class QuestionTypeClassifier:
    def __init__(self, taxonomy: dict[str, Any]):
        self.taxonomy = taxonomy
        self.short_max = float(taxonomy.get("short_answer_max_marks", 4))
        self.long_min = float(taxonomy.get("long_answer_min_marks", 7))
        self.types: list[dict[str, Any]] = []
        for t in taxonomy.get("types", []):
            compiled = [re.compile(p, re.IGNORECASE) for p in t.get("patterns", [])]
            self.types.append({**t, "compiled": compiled})
        self.meta = {t["id"]: {"label": t.get("label", t["id"]), "format": t.get("format", "theory")}
                     for t in self.types}
        self.meta.update({k: v for k, v in DERIVED_TYPES.items() if k not in self.meta})

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> "QuestionTypeClassifier":
        taxonomy = json.loads(DEFAULT_TAXONOMY.read_text(encoding="utf-8"))
        if data_dir:
            user_file = Path(data_dir) / "taxonomy.json"
            if user_file.exists():
                user = json.loads(user_file.read_text(encoding="utf-8"))
                by_id = {t["id"]: t for t in taxonomy["types"]}
                for t in user.get("types", []):
                    by_id[t["id"]] = {**by_id.get(t["id"], {}), **t}
                taxonomy["types"] = list(by_id.values())
                for key in ("short_answer_max_marks", "long_answer_min_marks"):
                    if key in user:
                        taxonomy[key] = user[key]
        return cls(taxonomy)

    def label(self, type_id: str) -> str:
        return self.meta.get(type_id, {}).get("label", type_id.replace("_", " ").title())

    def format_of(self, type_id: str) -> str:
        return self.meta.get(type_id, {}).get("format", "theory")

    def type_ids(self) -> list[str]:
        return list(self.meta)

    def classify(self, text: str, *, marks: float | None = None, options: list[str] | None = None,
                 context: str = "") -> TypeResult:
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        scan = f"{context} {text}".strip() if context and context != text else text
        for t in self.types:
            hits = []
            for pattern in t["compiled"]:
                m = pattern.search(scan)
                if m:
                    hits.append(m.group(0).strip())
            if hits:
                scores[t["id"]] = float(t.get("weight", 1.0)) * (1.0 + 0.25 * (len(hits) - 1))
                evidence[t["id"]] = hits[:3]
        units = UNIT_RE.findall(text)
        numbers = NUMBER_RE.findall(text)
        if len(units) >= 2 or (len(units) >= 1 and len(numbers) >= 3):
            scores["numerical"] = scores.get("numerical", 0.0) + 0.8
            evidence.setdefault("numerical", []).append(f"{len(units)} quantities with units")
        if options:
            scores["mcq"] = scores.get("mcq", 0.0) + 2.0
            evidence.setdefault("mcq", []).append(f"{len(options)} options")
        if not scores:
            if "?" in text or re.match(r"^\W*(?:what|how|why|which|when|where)\b", text, re.IGNORECASE):
                scores["conceptual_explanation"] = 0.4
            else:
                scores["unclassified"] = 0.2
        top_id = max(scores, key=scores.get)
        top_format = self.format_of(top_id)
        if marks is not None:
            if marks <= self.short_max and top_format in ("theory", "definition") and "short_answer" not in scores:
                scores["short_answer"] = 0.5
                evidence.setdefault("short_answer", []).append(f"{marks:g} marks")
            if marks >= self.long_min and top_format == "theory":
                scores["long_theory"] = 0.7
                evidence.setdefault("long_theory", []).append(f"{marks:g} marks")
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_score = ranked[0][1]
        types = [tid for tid, sc in ranked if sc >= 0.5 * top_score]
        formats = {self.format_of(tid) for tid, sc in ranked if sc >= 0.8}
        if len(formats - {"theory"}) >= 2 or (len(formats) >= 2 and "derivation" in formats and "numerical" in formats):
            types.append("mixed")
        primary = ranked[0][0]
        confidence = round(min(1.0, top_score / 1.5), 2)
        return TypeResult(types, {k: round(v, 3) for k, v in scores.items()}, primary,
                          self.format_of(primary), confidence, evidence)
