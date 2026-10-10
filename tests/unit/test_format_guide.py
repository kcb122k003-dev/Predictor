"""Question-format labels, the format guide and the priority sentence."""

from __future__ import annotations

import re

from predictor.analysis.topic_history import PaperInfo, QuestionInfo, build_history
from predictor.parsing.format_labels import families_of, format_labels
from predictor.prediction.format_guide import (build_format_guide, choose_family, clean_question, numeric_template,
                                               object_phrase)
from predictor.prediction.priority_reason import priority_reason

NUMERICAL = ("Calculate the head loss due to friction in a pipe of 300 mm diameter and 450 m length carrying water at "
             "2.5 m/s. Take the friction factor as 0.02. [6]")


def test_labels_read_the_instruction_and_allow_several():
    assert format_labels("State Pascal's law. Explain its applications.", []) == ["state_list", "explain"]
    assert format_labels("Prove that the pressure at a point is equal in all directions.", ["derivation"]) == ["prove"]
    assert format_labels("Derive Euler's equation and state the assumptions made.", ["derivation"]) == ["derive"]
    assert "short_answer" in format_labels("Hydraulic gradient line", [], context="Write short notes on any two:")
    assert set(format_labels("With a neat sketch, explain a Pelton wheel.", ["diagram", "conceptual_explanation"])) \
        == {"diagram", "explain"}
    assert format_labels("", []) == ["unclassified"]


def test_families_group_labels_and_short_note_yields_to_specific_formats():
    assert families_of(["numerical"]) == ["numerical"]
    assert families_of(["diagram", "explain"]) == ["diagram", "explanation"]
    assert families_of(["short_answer", "definition"]) == ["definition"]
    assert families_of(["short_answer"]) == ["short_note"]


def test_numeric_template_replaces_every_value_and_keeps_units():
    template, placeholders = numeric_template(NUMERICAL)
    assert not re.search(r"\d", template), template
    assert "[diameter in mm]" in template and "[length in m]" in template and "[velocity in m/s]" in template
    assert "[friction factor]" in template
    assert [p["example"] for p in placeholders] == ["300", "450", "2.5", "0.02"]
    assert clean_question("4(b) Define viscosity. (4 marks)") == "Define viscosity."


def test_object_phrase_takes_the_instruction_sentence():
    assert object_phrase("Define viscosity. With a neat sketch, explain Newton's law of viscosity.", "diagram")[0] \
        == "Newton's law of viscosity"
    assert object_phrase("Using the Buckingham pi theorem, derive an expression for the drag force.", "derivation")[0] \
        == "the drag force"


def _history(questions, years=(2019, 2020, 2021, 2022)):
    papers = [PaperInfo(i, i + 1, f"{y} Fall", y) for i, y in enumerate(years)]
    return build_history(papers, [], questions, lambda n: n, [1])


def _qdicts(h, texts):
    st = h["topics"]["1"]
    out = []
    for key, role in st["roles"].items():
        hq = h["questions"][key]
        out.append({"id": int(key), "text": texts[int(key)], "labels": hq["labels"], "families": hq["families"],
                    "marks": hq["marks"], "exam_index": hq["e"], "exam_label": h["papers"][hq["e"]]["label"],
                    "path_label": "2(a)", "role": role})
    return st, out


def test_guide_from_history_counts_papers_and_explains_itself():
    texts = {1: NUMERICAL, 2: "Calculate the head loss in a pipe of 200 mm diameter carrying 0.05 m3/s.",
             3: "Derive the Darcy-Weisbach equation.", 4: NUMERICAL}
    qs = [QuestionInfo(1, 0, [(1, 1, 1.0)], labels=["numerical"], families=["numerical"], marks=6.0),
          QuestionInfo(2, 1, [(1, 1, 1.0)], labels=["numerical"], families=["numerical"], marks=8.0),
          QuestionInfo(3, 2, [(1, 1, 1.0)], labels=["derive"], families=["derivation"], marks=8.0),
          QuestionInfo(4, 3, [(1, 1, 1.0)], labels=["numerical"], families=["numerical"], marks=6.0)]
    h = _history(qs)
    st, qd = _qdicts(h, texts)
    g = build_format_guide(topic_label="5.2 Pipes", topic_title="Pipes", kinds=set(), concepts=[], stats=st,
                           questions=qd, course_families=h["course_families"], papers=h["papers"])
    assert g["family"] == "numerical" and g["evidence"] == "established" and g["papers"] == 3
    assert "3 of the 4 papers" in g["why"] and "2022 Fall" in g["why"]
    assert g["description"].startswith("Numerical problem: calculate the head loss due to friction in a pipe")
    assert g["template"] and not re.search(r"\d", g["template"]["text"])
    assert g["illustrative"]["basis"] == "past_values" and "2022 Fall" in g["illustrative"]["note"]
    assert [a["family"] for a in g["alternatives"]] == ["derivation"]
    assert g["alternatives"][0]["papers"] == 1


def test_numerical_examples_must_pass_the_topic_check():
    texts = {1: NUMERICAL, 2: "A jet of water 5 cm in diameter strikes a plate at 20 m/s. Find the force."}
    qs = [QuestionInfo(1, 0, [(1, 1, 1.0)], labels=["numerical"], families=["numerical"]),
          QuestionInfo(2, 1, [(1, 1, 1.0)], labels=["numerical"], families=["numerical"])]
    h = _history(qs, years=(2019, 2020))
    st, qd = _qdicts(h, texts)
    g = build_format_guide(topic_label="5.2 Pipes", topic_title="Pipes", kinds=set(), concepts=[], stats=st,
                           questions=qd, course_families=[], papers=h["papers"], accept=lambda q: q["id"] != 2)
    assert g["illustrative"]["source"]["question_id"] == 1  # the newer, off-topic question is skipped


def test_weak_and_inferred_formats_are_labelled():
    one = _history([QuestionInfo(1, 0, [(1, 1, 1.0)], labels=["derive"], families=["derivation"])])
    st, qd = _qdicts(one, {1: "Derive Bernoulli's equation."})
    g = build_format_guide(topic_label="4.1 B", topic_title="Bernoulli's equation", kinds=set(), concepts=[],
                           stats=st, questions=qd, course_families=[], papers=one["papers"])
    assert g["basis"] == "weak_history" and g["evidence"] == "weak" and "too little history" in g["why"]

    none = _history([])
    st = none["topics"]["1"]
    assert choose_family(st, {"derivation"}, [])[:2] == ("derivation", "syllabus")
    course = [{"family": "explanation", "display": "Conceptual explanation", "questions": 9, "papers": 4,
               "of_questions": 20}]
    assert choose_family(st, set(), course)[:2] == ("explanation", "course_pattern")
    assert choose_family(st, set(), [])[:2] == ("explanation", "none")
    g = build_format_guide(topic_label="1.4 C", topic_title="Compressibility", kinds={"derivation"}, concepts=[],
                           stats=st, questions=[], course_families=course, papers=none["papers"])
    assert g["evidence"] == "inferred" and "inferred" in g["why"] and "inferred" in g["description"]
    assert g["alternatives"] == [] and g["illustrative"] is None


def test_priority_reason_never_calls_a_missing_topic_unlikely():
    text = priority_reason(category="Low Priority", rank=24, topics=25,
                           stats={"usable_papers": 12, "exam_frequency": 0}, facts={}, guide=None)
    assert "unlikely" not in text.lower() and "not evidence that it will not be examined" in text
    text = priority_reason(category="High Priority", rank=3, topics=25,
                           stats={"usable_papers": 12, "exam_frequency": 9, "recent_window": 3, "recent_hits": 3,
                                  "last_label": "2025 Fall"},
                           facts={"mapping_confidence": 0.9},
                           guide={"basis": "history", "papers": 5, "display": "Numerical problem"})
    assert text.startswith("High priority (rank 3 of 25) because it appeared in 9 of 12 usable papers")
    # The format barely affects the rank, so it is not given as a reason for the priority.
    assert "numerical problem" not in text and "format" not in text
    assert "no past papers" in priority_reason(category="Low Priority", rank=1, topics=3, stats={"usable_papers": 0},
                                               facts={}, guide=None)
