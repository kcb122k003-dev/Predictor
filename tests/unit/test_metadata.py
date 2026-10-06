from datetime import date

from predictor.parsing.metadata import detect_year, extract_metadata, order_index_for


def test_year_from_header_beats_course_code():
    header = "ABC University\nCourse Code: ENG 2051\nFinal Examination 2019\nFull Marks: 60"
    year, conf, cal, _ = detect_year(header, "paper.pdf", min_year=1950, max_year=2100, today=date(2026, 1, 1))
    assert year == 2019 and cal == "AD" and conf > 0.5


def test_bikram_sambat_detected():
    year, _, cal, _ = detect_year("Examination 2079 Baishakh", "", min_year=1950, max_year=2100,
                                  today=date(2026, 1, 1))
    assert year == 2079 and cal == "BS"


def test_filename_year_used_when_header_missing():
    year, _, _, ev = detect_year("", "thermo_2017_final.pdf", min_year=1950, max_year=2100)
    assert year == 2017 and "file name" in ev


def test_metadata_fields():
    header = "Midterm Exam - Fall 2021\nSubject: Heat Transfer\nF.M.: 40   P.M.: 16\nTime: 1.5 hrs"
    md = extract_metadata(header, header, "x.pdf")
    assert md.get("year") == 2021
    assert md.get("session") == "Fall"
    assert md.get("exam_type") == "mid-term"
    assert md.get("full_marks") == 40 and md.get("pass_marks") == 16
    assert md.get("subject") == "Heat Transfer"
    assert md.get("duration") == "1.5 hrs"


def test_order_index_sorts_sessions_and_calendars():
    spring = order_index_for(2019, "AD", 0.3)
    fall = order_index_for(2019, "AD", 0.8)
    back = order_index_for(2019, "AD", 0.8, "back")
    bs = order_index_for(2076, "BS", 0.95)  # Chaitra 2076 is about March/April 2020
    assert spring < fall < back < bs < order_index_for(2021, "AD", 0.3)
