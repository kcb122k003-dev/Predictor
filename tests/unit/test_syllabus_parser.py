from predictor.syllabus.merge import merge_syllabi
from predictor.syllabus.parser import SyllabusParser, split_concepts

from tests.conftest import pages_from

SYLLABUS = """FLUID MECHANICS
CE 501
Course Objectives:
1. To provide basic knowledge of fluid statics and dynamics.
1. Properties of Fluids (4 hours)
1.1 Definition of fluid, continuum concept
1.2 Viscosity: Newton's law of viscosity, dynamic and kinematic viscosity
2. Fluid Statics (8 hours)
2.1 Pressure measurement: piezometer, U-tube manometer, differential manometer
UNIT 4
Flow through pipes [8 hours]
 • Laminar flow: Hagen-Poiseuille equation
 • Darcy-Weisbach equation, Moody diagram
Practical:
1. Determination of coefficient of discharge of a venturimeter
References:
1. Modi and Seth, Hydraulics and Fluid Mechanics, Standard Book House
Marks distribution
Chapter   Hours   Marks
1         4        8
2         8        16
4         8        16
"""


def test_split_concepts():
    assert split_concepts("piezometer, U-tube manometer and differential manometer", 9) == [
        "piezometer", "U-tube manometer", "differential manometer"]
    assert split_concepts("Euler's equation (derivation, applications)", 9) == ["Euler's equation (derivation, applications)"]


def test_syllabus_tree(settings):
    parsed = SyllabusParser(settings).parse(pages_from(SYLLABUS), "syl.txt", file_id=7)
    titles = [(n.depth, n.title) for n in parsed.all_nodes()]
    assert (1, "Properties of Fluids") in titles
    assert (2, "Viscosity") in titles
    assert (1, "Flow through pipes") in titles
    assert (2, "Laminar flow") in titles
    assert parsed.objectives == ["To provide basic knowledge of fluid statics and dynamics."]
    nodes = {n.title: n for n in parsed.all_nodes()}
    assert nodes["Properties of Fluids"].hours == 4
    assert nodes["Flow through pipes"].hours == 8
    assert nodes["Properties of Fluids"].marks_weight == 8
    assert "U-tube manometer" in nodes["Pressure measurement"].concepts
    assert nodes["Laminar flow"].concepts == ["Hagen-Poiseuille equation"]
    # References never become topics; the practical section becomes a lab unit.
    assert not any("Modi" in t for _, t in titles)
    lab = nodes["Laboratory / practical work"]
    assert "lab" in lab.kinds and lab.children[0].title.startswith("Determination")
    ref = nodes["Viscosity"].source_refs[0]
    assert ref["file_id"] == 7 and ref["page"] == 1 and ref["line"] > 1


def test_merge_documents(settings):
    p = SyllabusParser(settings)
    a = p.parse(pages_from(SYLLABUS), "a.txt")
    b = p.parse(pages_from("Unit 1: Properties of fluids\n- Viscosity: Newton's law, viscometers\n"
                           "Unit 7: Turbomachines (6 hrs)\n- Centrifugal pumps"), "b.txt")
    merged = merge_syllabi([a, b])
    by_title = {n.title.lower(): n for n in merged.all_nodes()}
    props = by_title["properties of fluids"]
    assert any(r["file"] == "b.txt" for r in props.source_refs)
    assert "viscometers" in [c.lower() for c in by_title["viscosity"].concepts]
    assert "turbomachines" in by_title
    assert merged.warnings  # the unmatched unit is reported
