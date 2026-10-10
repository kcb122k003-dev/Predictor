"""Per-topic history: paper vs question counts, roles, missing and excluded papers, formats, repetition."""

from __future__ import annotations

from predictor.analysis.topic_history import (ExcludedPaper, PaperInfo, QuestionInfo, build_history, node_index,
                                              node_stats, relation_of, timeline)

# Syllabus: unit 1 with topics 10 and 11; unit 2 with topic 20 (and sub-topic 21 under 20).
TOPIC_OF = {10: 10, 11: 11, 20: 20, 21: 20}
TOPICS = [10, 11, 20]


def topic_of(n):
    return TOPIC_OF.get(n)


def papers(years):
    return [PaperInfo(i, 100 + i, f"{y} Fall" if y else "Year? paper", y) for i, y in enumerate(years)]


def q(qid, e, nodes, labels=("explain",), families=("explanation",), rel="new", pc=1.0, marks=5.0, uncounted=()):
    return QuestionInfo(qid, e, list(nodes), uncounted=list(uncounted), labels=list(labels), families=list(families),
                        relation=rel, parse_confidence=pc, marks=marks)


def test_paper_and_question_counts_are_kept_apart():
    """Two questions on a topic in one paper count once for papers and twice for questions."""
    h = build_history(papers([2019, 2020, 2021]), [], [
        q(1, 0, [(10, 1, 1.0)]), q(2, 0, [(10, 1, 1.0)]), q(3, 2, [(10, 1, 1.0)]), q(4, 1, [(11, 1, 1.0)]),
    ], topic_of, TOPICS)
    st = h["topics"]["10"]
    assert st["exam_frequency"] == 2 and st["usable_papers"] == 3
    assert st["question_frequency"] == 3
    assert st["papers"] == [0, 2] and st["last_label"] == "2021 Fall"
    assert h["topics"]["20"]["exam_frequency"] == 0 and h["topics"]["20"]["question_frequency"] == 0


def test_secondary_topics_are_counted_and_labelled():
    h = build_history(papers([2019, 2020]), [], [
        q(1, 0, [(10, 1, 1.0), (11, 2, 0.5)]), q(2, 1, [(11, 1, 1.0)]),
    ], topic_of, TOPICS)
    t11 = h["topics"]["11"]
    assert t11["exam_frequency"] == 2 and t11["primary_questions"] == 1 and t11["secondary_questions"] == 1
    assert t11["secondary_only_papers"] == 1
    assert t11["roles"] == {"1": "secondary", "2": "primary"}
    assert h["topics"]["10"]["secondary_questions"] == 0


def test_sub_topic_mappings_roll_up_and_groups_count_each_question_once():
    h = build_history(papers([2019, 2020]), [], [
        q(1, 0, [(21, 1, 1.0)]),                    # sub-topic of 20
        q(2, 1, [(10, 1, 1.0), (11, 2, 0.5)]),      # two topics of the same unit
    ], topic_of, TOPICS)
    assert h["topics"]["20"]["question_frequency"] == 1
    unit1 = node_stats(h, lambda n: n in {10, 11})
    assert unit1["question_frequency"] == 1 and unit1["exam_frequency"] == 1
    sub = node_stats(h, lambda n: n == 21)
    assert sub["question_frequency"] == 1


def test_candidate_index_never_changes_the_result():
    h = build_history(papers([2019, 2020, 2021]), [], [
        q(1, 0, [(10, 1, 1.0)]), q(2, 1, [(21, 1, 1.0)], uncounted=[(10, "C", 0.2)]), q(3, 2, [(11, 1, 1.0)]),
    ], topic_of, TOPICS)
    index = node_index(h)
    for members in ({10}, {20, 21}, {10, 11}, {21}):
        cands = set().union(*(index.get(n, set()) for n in members))
        assert node_stats(h, lambda n: n in members, cands) == node_stats(h, lambda n: n in members)


def test_uncertain_matches_are_listed_but_not_counted():
    h = build_history(papers([2019]), [], [q(1, 0, [], uncounted=[(10, "C", 0.18)])], topic_of, TOPICS)
    st = h["topics"]["10"]
    assert st["exam_frequency"] == 0 and st["question_frequency"] == 0
    assert st["uncounted_questions"] == [1]


def test_missing_years_are_missing_data_not_absences():
    h = build_history(papers([2018, 2021]), [], [q(1, 0, [(10, 1, 1.0)])], topic_of, TOPICS)
    assert h["missing_years"] == [2019, 2020]
    rows = timeline(h, h["topics"]["10"])
    kinds = [(r["kind"], r["label"], r["appeared"]) for r in rows]
    assert kinds == [("paper", "2018 Fall", True), ("missing", "2019", None), ("missing", "2020", None),
                     ("paper", "2021 Fall", False)]
    assert all("not examined" not in (r.get("reason") or "").lower() for r in rows)


def test_excluded_and_duplicate_papers_are_shown_but_never_counted():
    excluded = [ExcludedPaper(9, "2019 Fall", 2019, "duplicate", "Duplicate of 2019 Fall Regular", 101),
                ExcludedPaper(8, "2020 Fall", 2020, "excluded", "Excluded from the analysis by you."),
                ExcludedPaper(7, "Year? paper", None, "undated", "Exam year not detected")]
    h = build_history(papers([2019, 2021]), excluded, [q(1, 0, [(10, 1, 1.0)])], topic_of, TOPICS)
    assert h["usable_papers"] == 2 and h["topics"]["10"]["exam_frequency"] == 1
    rows = timeline(h, h["topics"]["10"])
    by_kind = {(r["kind"], r["label"]) for r in rows}
    assert ("excluded", "2019 Fall") in by_kind and ("excluded", "2020 Fall") in by_kind
    assert ("missing", "2020") not in by_kind  # the year has a paper, it was excluded
    assert [e["kind"] for e in h["excluded_papers"]] == ["duplicate", "excluded", "undated"]


def test_undated_usable_paper_is_flagged_not_silently_placed():
    h = build_history(papers([None, 2020]), [], [q(1, 0, [(10, 1, 1.0)])], topic_of, TOPICS)
    assert h["undated_papers"] == [0]
    rows = timeline(h, h["topics"]["10"])
    assert rows[0]["undated"] is True and rows[0]["label"] == "Year? paper"


def test_formats_are_multi_label_with_paper_counts():
    h = build_history(papers([2019, 2020, 2021]), [], [
        q(1, 0, [(10, 1, 1.0)], labels=("numerical",), families=("numerical",)),
        q(2, 0, [(10, 1, 1.0)], labels=("numerical",), families=("numerical",)),
        q(3, 1, [(10, 1, 1.0)], labels=("diagram", "explain"), families=("diagram", "explanation")),
        q(4, 2, [(10, 1, 1.0)], labels=("numerical",), families=("numerical",)),
    ], topic_of, TOPICS)
    st = h["topics"]["10"]
    labels = {d["label"]: (d["questions"], d["papers"]) for d in st["labels"]}
    assert labels == {"numerical": (3, 2), "diagram": (1, 1), "explain": (1, 1)}
    assert st["multi_label_questions"] == 1
    assert st["families"][0]["family"] == "numerical" and st["families"][0]["papers"] == 2


def test_repetition_classes_and_same_paper_pairs():
    exact = {5: [1], 6: [], 7: []}
    para = {5: [1], 6: [2], 7: []}
    concept = {7: [3]}
    assert relation_of(5, exact, para, concept) == ("exact", [1])  # exact wins over paraphrase
    assert relation_of(6, exact, para, concept) == ("paraphrase", [2])
    assert relation_of(7, exact, para, concept) == ("concept", [3])
    assert relation_of(8, exact, para, concept) == ("new", [])
    h = build_history(papers([2019, 2020]), [], [
        q(1, 0, [(10, 1, 1.0)]), q(2, 1, [(10, 1, 1.0)], rel="exact"), q(3, 1, [(10, 1, 1.0)], rel="concept"),
    ], topic_of, TOPICS)
    assert h["topics"]["10"]["repetition"] == {"exact": 1, "paraphrase": 0, "concept": 1, "new": 1}


def test_provisional_counts_name_their_reasons():
    ps = papers([2019, 2020])
    ps[1].provisional = ["low OCR confidence on page 2"]
    h = build_history(ps, [], [q(1, 0, [(10, 1, 1.0)], pc=0.5), q(2, 1, [(11, 1, 1.0)])], topic_of, TOPICS)
    assert h["topics"]["10"]["provisional"] and "low parse confidence" in h["topics"]["10"]["provisional_reasons"][0]
    assert h["topics"]["11"]["provisional"]
    assert any("OCR" in r for r in h["topics"]["11"]["provisional_reasons"])
    assert not h["topics"]["20"]["provisional"]


def test_zero_papers():
    h = build_history([], [], [], topic_of, TOPICS)
    assert h["usable_papers"] == 0 and h["missing_years"] == []
    st = h["topics"]["10"]
    assert st["exam_frequency"] == 0 and st["recent_window"] == 0 and st["last_label"] is None
    assert timeline(h, st) == []
