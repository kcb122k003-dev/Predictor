from predictor.parsing.exam_parser import ExamParser, extract_trailing_marks, parse_marks_expr

from tests.conftest import pages_from

TU_PAPER = """TRIBHUVAN UNIVERSITY
INSTITUTE OF ENGINEERING
Examination Control Division
2076 Chaitra
Exam.        Regular
Level   BE      Full Marks   80
Programme  BCH   Pass Marks  32
Year / Part  II / I   Time  3 hrs.
Subject: - Fluid Mechanics (CE 501)
✓ Candidates are required to give their answers in their own words as far as practicable.
✓ Attempt All questions.
1. a) Define viscosity. Explain Newton's law of viscosity with a neat sketch. [2+4]
   b) A plate 0.025 mm distant from a fixed plate moves at 60 cm/s and requires a force of 2 N/m2.
      Determine the dynamic viscosity of the fluid between the plates. [6]
2. Derive Bernoulli's equation from Euler's equation of motion along a streamline. [8]
OR
2. Explain the working principle of a venturimeter. [8]
3. (a) What is meant by metacentric height? [3]  (b) Explain the stability of floating bodies. [5]
4. Write short notes on (any two):          [2×4]
   i) Pitot tube
   ii) Notches and weirs
   iii) Boundary layer separation
5. Which of the following is a dimensionless number?
   (a) Reynolds number
   (b) Viscosity
   (c) Density
   (d) Pressure                               [1]
"""


def _nodes(parsed):
    return {n.path_label: n for n in parsed.all_nodes()}


def test_marks_expressions():
    assert parse_marks_expr("2+4") == 6
    assert parse_marks_expr("5x2") == 10
    assert parse_marks_expr("2×4") == 8
    assert parse_marks_expr("2x3+4") == 10
    text, marks, expr, source, _ = extract_trailing_marks("Define entropy. [4]")
    assert (text, marks, source) == ("Define entropy.", 4, "bracket")
    _, marks, _, source, _ = extract_trailing_marks("Explain the cycle.        6")
    assert marks == 6 and source == "margin"
    _, marks, _, _, _ = extract_trailing_marks("Find the velocity at x = 5")
    assert marks is None


def test_tu_paper_structure(settings):
    parsed = ExamParser(settings).parse(pages_from(TU_PAPER), "fluid_2076.pdf")
    md = parsed.metadata.as_dict()
    assert md["year"] == 2076 and md["calendar"] == "BS"
    assert md["session"] == "Chaitra" and md["exam_type"] == "regular"
    assert md["full_marks"] == 80 and md["pass_marks"] == 32
    nodes = _nodes(parsed)
    assert nodes["1(a)"].marks == 6 and nodes["1(b)"].marks == 6
    assert "dynamic viscosity" in nodes["1(b)"].text  # multi-line continuation kept
    assert nodes["1"].marks == 12
    q2 = [q for q in parsed.questions if q.label == "2"]
    assert len(q2) == 2 and q2[0].or_group == q2[1].or_group is not None
    assert all(q.is_optional for q in q2)
    # Inline sub-parts on one line.
    assert nodes["3(a)"].marks == 3 and nodes["3(b)"].marks == 5
    # "any two" of three short notes, [2x4] -> 4 marks each, optional.
    assert nodes["4(i)"].marks == 4 and nodes["4(ii)"].is_optional
    # MCQ options collapsed into the stem.
    assert nodes["5"].options == ["Reynolds number", "Viscosity", "Density", "Pressure"]
    assert nodes["5"].marks == 1 and nodes["5"].is_leaf


def test_q_prefix_sections_and_attempt_any(settings):
    paper = """KATHMANDU UNIVERSITY
End Semester Examination  Spring 2019
Course: CHEM 201 Physical Chemistry           Full Marks: 50
Group A
Attempt any three questions.
Q1. Explain the third law of thermodynamics.     (5 marks)
Q2. Derive the Gibbs-Helmholtz equation.    (5 marks)
Q3. Calculate the entropy change for 2 mol of ideal gas expanding from 10 L to 20 L.     (5 marks)
Q4. Define chemical potential.  (5 marks)
Group B
Q5.1 State and explain Raoult's law.   (4)
Q5.2 Differentiate between ideal and non-ideal solutions.   (6)
"""
    parsed = ExamParser(settings).parse(pages_from(paper), "chem.pdf")
    assert parsed.metadata.get("year") == 2019
    assert parsed.metadata.get("session") == "Spring"
    assert [s.label for s in parsed.sections] == ["A", "B"]
    assert parsed.sections[0].attempt_count == 3
    nodes = _nodes(parsed)
    assert all(nodes[str(i)].is_optional for i in range(1, 5))
    assert nodes["5.1"].marks == 4 and nodes["5.2"].marks == 6
    assert nodes["5"].marks == 10


def test_nested_roman_and_margin_marks(settings):
    paper = """Question 1
(a) Explain the first law of thermodynamics for a closed system.        5
(b) A gas expands from 0.1 m3 to 0.3 m3 at 2.5 bar. Find the work done.        5
Question 2 Explain the following:
(i) Carnot cycle (ii) Clausius inequality      [5+5]
\fPage 2 of 2
3. (a) Explain the concept of entropy. [4]
(b) Prove that entropy is a property. [6]
(c) i. Define availability.  [2]
    ii. Explain irreversibility. [3]
"""
    parsed = ExamParser(settings).parse(pages_from(paper), "x.pdf")
    nodes = _nodes(parsed)
    assert nodes["1(a)"].marks == 5 and nodes["1(a)"].marks_source == "margin"
    assert nodes["2(i)"].text == "Carnot cycle" and nodes["2(ii)"].marks == 5
    assert nodes["3(c)(i)"].marks == 2 and nodes["3(c)(ii)"].marks == 3
    assert nodes["3(c)"].marks == 5
    assert nodes["3(a)"].page_no == 2
    assert "Explain the following" in nodes["2(i)"].context_text()


def test_numbers_inside_text_do_not_start_questions(settings):
    paper = """1. A tank contains water. Answer the following steps:
2.5 kg of steam is added at 1 bar.
1. Find the final temperature.
2. Explain the energy balance.   [6]
"""
    parsed = ExamParser(settings).parse(pages_from(paper), "x.pdf")
    # "2.5 kg" is text, and "1." after question 1 is not a new main question; "2." is question 2.
    labels = [q.label for q in parsed.questions]
    assert labels == ["1", "2"]
    assert "2.5 kg of steam" in parsed.questions[0].text


def test_no_questions_warns(settings):
    parsed = ExamParser(settings).parse(pages_from("Just some text without numbering."), "x.pdf")
    assert parsed.questions == [] and parsed.warnings


def test_wrapped_decimal_is_not_dotted_numbering(settings):
    paper = """1. a) The space between two plates 12 mm apart is filled with oil of viscosity
      1.5 Pa.s. Calculate the shear stress when the plate moves at 2.5 m/s.    [6]
   b) Derive an expression for the metacentric height. [8]
2. Explain the following:
2.1 State Raoult's law. (4)
2.2 Explain osmosis. (6)
"""
    parsed = ExamParser(settings).parse(pages_from(paper), "x.pdf")
    nodes = _nodes(parsed)
    assert "1.5 Pa.s. Calculate the shear stress" in nodes["1(a)"].text and nodes["1(a)"].marks == 6
    assert nodes["2.1"].marks == 4 and nodes["2.2"].marks == 6
