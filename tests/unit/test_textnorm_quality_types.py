from predictor.parsing.question_types import QuestionTypeClassifier
from predictor.preprocessing.quality import assess_text, implausible_token
from predictor.preprocessing.textnorm import clean_text, content_terms, extract_equations, normalize_math


def test_clean_and_math_normalisation():
    assert clean_text("ﬁnal  “value” – 5 kg") == 'final  "value" - 5 kg'
    assert normalize_math("H₂O at 25°C, x² + αβ") == "H2O at 25 deg C, x^2 + alpha beta"
    eqs = extract_equations("Show that Cp - Cv = R for an ideal gas.")
    assert eqs[0]["raw"] == "Cp - Cv = R"


def test_content_terms_drop_instruction_words(settings):
    terms = content_terms("Derive Bernoulli's equation from Euler's equation of motion.",
                          settings.alignment.instruction_words)
    assert "deriv" not in terms and "bernoulli" in terms and "euler" in terms


def test_quality_flags(settings):
    assert implausible_token("xqzvbt") and implausible_token("l0ad")
    assert not implausible_token("NPSH") and not implausible_token("H2SO4") and not implausible_token("viscosity")
    noisy = "Th1s qxzv wrkd brrgh tnnnk zzzzzz plmqrst cvbnm"
    assert "implausible_words" in assess_text(noisy, settings).flags
    clean = "a) Define viscosity and explain Newton's law of viscosity."
    assert assess_text(clean, settings).ok


def test_question_types():
    clf = QuestionTypeClassifier.load()
    assert clf.classify("Derive Bernoulli's equation from Euler's equation.").primary == "derivation"
    r = clf.classify("Calculate the work done when 2 kg of air expands from 1 bar to 5 bar at 300 K.")
    assert r.primary == "numerical" and r.format == "numerical"
    assert clf.classify("Differentiate between laminar and turbulent flow.").primary == "compare_contrast"
    assert clf.classify("Define viscosity.", marks=2).primary == "definition"
    assert "short_answer" in clf.classify("Define viscosity.", marks=2).types
    mcq = clf.classify("Which of the following is dimensionless?", options=["a", "b", "c", "d"])
    assert mcq.primary == "mcq"
    mixed = clf.classify("Derive the expression for discharge through a venturimeter and calculate the "
                         "flow rate if d1 = 200 mm, d2 = 100 mm and the head is 0.5 m.")
    assert "mixed" in mixed.types
    assert clf.classify("Pitot tube", context="Write short notes on").primary == "short_answer"
