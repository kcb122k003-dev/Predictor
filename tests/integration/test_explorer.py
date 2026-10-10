"""Syllabus Explorer and the redesigned predictions on the synthetic demo course.

The demo is a software fixture: these tests check that counts, sources and labels are correct and
consistent, not that predictions are accurate.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from predictor.app.server import create_app
from predictor.database.models import AnalysisArtifact, BacktestFold, Exam, ExamQuestion, PredictedQuestion, SourceFile
from predictor.generation.questions import LABEL
from predictor.services.context import AppContext


@pytest.fixture(scope="module")
def api(analysed_demo):
    return TestClient(create_app(analysed_demo["app"], run_jobs_inline=True)), analysed_demo["course_id"]


@pytest.fixture(scope="module")
def overview(api):
    c, cid = api
    return c.get(f"/api/courses/{cid}/explorer").json()


def _results(c, cid):
    return c.get(f"/api/courses/{cid}/results").json()


def test_tree_shows_every_syllabus_node_in_its_hierarchy(api, overview):
    c, cid = api
    syllabus = c.get(f"/api/courses/{cid}/syllabus").json()["versions"][0]["topics"]
    flat = {}

    def walk(nodes):
        for n in nodes:
            flat[n["id"]] = n
            walk(n.get("children") or [])

    walk(syllabus)
    nodes = {n["id"]: n for n in overview["nodes"]}
    assert set(nodes) == set(flat)
    for nid, n in nodes.items():
        assert n["parent_id"] == flat[nid]["parent_id"]
        assert n["title"] == flat[nid]["title"]
        assert sorted(n["children"]) == sorted(ch["id"] for ch in flat[nid].get("children") or [])
    assert {n["id"] for n in overview["nodes"] if n["parent_id"] is None} == set(overview["roots"])
    # Original wording is kept next to the normalised title.
    darcy = next(n for n in overview["nodes"] if n["number"] == "5.2")
    assert darcy["original"].startswith("5.2 Darcy-Weisbach")
    # The laboratory heading's name was supplied by the app ("Practical:" in the document).
    lab = next(n for n in overview["nodes"] if n["is_lab"] and n["parent_id"] is None)
    assert lab["inferred"] and "Practical" in lab["inferred_reason"]


def test_counts_match_the_model_panel_and_the_predictions_tab(api, overview, analysed_demo):
    c, cid = api
    res = _results(c, cid)
    preds = {p["topic_id"]: p for p in res["predictions"]}
    nodes = {n["id"]: n for n in overview["nodes"]}
    with analysed_demo["app"].db.session() as s:
        charts = s.execute(select(AnalysisArtifact).where(AnalysisArtifact.run_id == res["run"]["id"],
                                                          AnalysisArtifact.key == "charts")).scalar_one().data
    timeline = {label: (sum(1 for z in row if z), sum(len(q) for q in qs))
                for label, row, qs in zip(charts["timeline"]["y"], charts["timeline"]["z"], charts["timeline"]["questions"])}
    assert len(preds) == 25
    for tid, p in preds.items():
        st = nodes[tid]["stats"]
        facts = p["facts"]
        assert st["exam_frequency"] == facts["appearances"] == p["history"]["exam_frequency"], p["label"]
        assert st["question_frequency"] == facts["questions_total"] == p["history"]["question_frequency"], p["label"]
        assert (st["exam_frequency"], st["question_frequency"]) == timeline[p["label"]]
        assert st["usable_papers"] == 12 == p["history"]["usable_papers"]
        compact = nodes[tid]["prediction"]
        assert (compact["rank"], compact["category"], compact["probability"]) == (p["rank"], p["category"],
                                                                                  p["probability"])
    # Paper counts never exceed question counts, and several questions in one paper count once for papers.
    assert all(n["stats"]["exam_frequency"] <= n["stats"]["question_frequency"] for n in overview["nodes"])
    assert any(n["stats"]["exam_frequency"] < n["stats"]["question_frequency"] for n in overview["nodes"])


def test_node_detail_reuses_the_same_prediction_and_original_wording(api, analysed_demo):
    c, cid = api
    res = _results(c, cid)
    top = res["predictions"][0]
    d = c.get(f"/api/courses/{cid}/explorer/nodes/{top['topic_id']}").json()
    assert d["prediction"] == top  # the exact dict the Predictions tab shows
    assert d["stats"]["exam_frequency"] == top["history"]["exam_frequency"]
    assert d["questions"] and len(d["questions"]) == d["stats"]["question_frequency"]
    with analysed_demo["app"].db.session() as s:
        for q in d["questions"]:
            row = s.get(ExamQuestion, q["id"])
            exam = s.get(Exam, row.exam_id)
            src = s.get(SourceFile, exam.source_file_id)
            assert q["text"] == row.text and q["raw_text"] == row.raw_text
            assert q["page"] == row.page_no and q["file"] == src.filename and q["question"] == row.path_label
    papers = [r for r in d["timeline"] if r["kind"] == "paper"]
    assert len(papers) == 12 and sum(1 for r in papers if r["appeared"]) == d["stats"]["exam_frequency"]


def test_duplicate_paper_is_shown_as_excluded_not_as_an_absence(api, overview):
    c, cid = api
    assert overview["usable_papers"] == 12
    dup = [e for e in overview["excluded_papers"] if e["kind"] == "duplicate"]
    assert len(dup) == 1 and dup[0]["year"] == 2017
    d = c.get(f"/api/courses/{cid}/explorer/nodes/{_results(c, cid)['predictions'][0]['topic_id']}").json()
    rows = [r for r in d["timeline"] if r.get("year") == 2017]
    assert {r["kind"] for r in rows} == {"paper", "excluded"}
    assert all(r["appeared"] is None for r in rows if r["kind"] == "excluded")


def test_group_nodes_aggregate_children_without_a_fake_prediction(api, overview):
    c, cid = api
    group = next(n for n in overview["nodes"] if n["level"] == "group" and n["number"] == "4")
    d = c.get(f"/api/courses/{cid}/explorer/nodes/{group['id']}").json()
    assert d["prediction"] is None and d["child_predictions"]
    kids = d["children"]
    assert d["stats"]["question_frequency"] == sum(k["stats"]["question_frequency"] for k in kids)
    assert d["stats"]["exam_frequency"] <= sum(k["stats"]["exam_frequency"] for k in kids)
    assert d["stats"]["exam_frequency"] >= max(k["stats"]["exam_frequency"] for k in kids)


def test_formats_are_counted_by_questions_and_papers(api):
    c, cid = api
    res = _results(c, cid)
    top = res["predictions"][0]
    d = c.get(f"/api/courses/{cid}/explorer/nodes/{top['topic_id']}").json()
    labels = d["stats"]["labels"]
    assert labels and all(1 <= lb["papers"] <= lb["questions"] for lb in labels)
    assert sum(lb["questions"] for lb in labels) >= d["stats"]["question_frequency"]  # overlapping labels
    guide = top["format_guide"]
    fam = next(f for f in d["stats"]["families"] if f["family"] == guide["family"])
    assert guide["papers"] == fam["papers"] == max(f["papers"] for f in d["stats"]["families"])
    assert str(guide["papers"]) in guide["why"]


def test_probabilities_are_marginal_and_only_shown_when_calibrated(api):
    c, cid = api
    res = _results(c, cid)
    preds = res["predictions"]
    if res["run"]["summary"]["calibrated"]:
        probs = [p["probability"] for p in preds]
        assert all(p is not None and 0 <= p <= 1 for p in probs)
        assert sum(probs) > 1.5  # several topics appear in one paper: probabilities are not normalised to 100%
    else:
        assert all(p["probability"] is None for p in preds)
    assert all(0 <= p["relative_score"] <= 1 for p in preds)


def test_illustrative_questions_are_labelled_grounded_and_never_history(api, analysed_demo):
    c, cid = api
    res = _results(c, cid)
    run_id = res["run"]["id"]
    with analysed_demo["app"].db.session() as s:
        rows = s.execute(select(PredictedQuestion).where(PredictedQuestion.run_id == run_id)).scalars().all()
        assert rows and all((r.grounding or {}).get("label") == LABEL for r in rows)
        assert all((r.grounding or {}).get("grounded") for r in rows)
        assert all(((r.grounding or {}).get("checks") or {}).get("passed", True) for r in rows)
        stored = {q.text for q in s.execute(select(ExamQuestion)).scalars()}
    assert not any(r.basis == "template" and r.text in stored for r in rows)
    for p in res["predictions"]:
        g = p["format_guide"]
        assert g["description"] and g["why"] and g["evidence"] in ("established", "weak", "inferred")
        il = g.get("illustrative")
        if il:
            assert il["note"].startswith("Illustrative practice question")
            if il["basis"] == "past_values":  # a real question counted for this topic, named in the note
                d = c.get(f"/api/courses/{cid}/explorer/nodes/{p['topic_id']}").json()
                assert str(il["source"]["question_id"]) in d["stats"]["roles"]
        if g["template"]:  # every value is a placeholder (units such as m3/s stay inside the brackets)
            assert not re.search(r"\d", re.sub(r"\[[^\]]*\]", "", g["template"]["text"])), g["template"]["text"]
        assert len(g["alternatives"]) <= 2


def test_topics_with_no_history_are_not_called_unlikely(api, overview):
    c, cid = api
    never = [n for n in overview["nodes"] if n["level"] == "topic" and n["stats"]["exam_frequency"] == 0]
    assert never
    preds = {p["topic_id"]: p for p in _results(c, cid)["predictions"]}
    for n in never:
        p = preds[n["id"]]
        text = " ".join([p["priority_reason"], p["facts"]["evidence_summary"], p["format_guide"]["why"]]).lower()
        assert "unlikely" not in text and "rarely tested" not in text
        assert "not evidence that it will not be examined" in p["priority_reason"]
        assert p["format_guide"]["evidence"] == "inferred"


def test_source_passages_and_pages(api, overview, analysed_demo):
    c, cid = api
    darcy = next(n for n in overview["nodes"] if n["number"] == "5.2")
    src = c.get(f"/api/topics/{darcy['id']}/source").json()
    ref = src["refs"][0]
    assert ref["file"].endswith(".pdf") and ref["page"] == 1 and ref["passage"].startswith("5.2 Darcy-Weisbach")
    img = c.get(f"/api/files/{ref['file_id']}/pages/{ref['page']}/image", params={"highlight": ref["text"]})
    assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"
    with analysed_demo["app"].db.session() as s:
        pdf_exam = next(e for e in s.execute(select(Exam).where(Exam.course_id == cid)).scalars()
                        if s.get(SourceFile, e.source_file_id).filename.endswith(".pdf"))
        q = next(q for q in pdf_exam.questions if q.is_leaf)
        qid, page = q.id, q.page_no
    qs = c.get(f"/api/questions/{qid}/source").json()
    assert qs["is_pdf"] and qs["page"] == page and qs["page_text"]
    assert c.get(f"/api/files/{ref['file_id']}/pages/99/image").status_code == 404


# ------------------------------------------------------------------ corrections, leakage, exclusions
@pytest.fixture(scope="module")
def editable(tmp_path_factory, demo_files):
    folder, _ = demo_files
    ctx = AppContext.create(tmp_path_factory.mktemp("explorer_edit"), log=False)
    c = TestClient(create_app(ctx, run_jobs_inline=True))
    cid = c.post("/api/courses", json={"name": "Editable demo"}).json()["id"]
    for kind, sub in (("syllabus", "syllabus"), ("exam", "exams")):
        files = [("files", (p.name, p.read_bytes())) for p in sorted((folder / sub).iterdir())]
        c.post(f"/api/courses/{cid}/files", data={"kind": kind}, files=files)
    run = c.post(f"/api/courses/{cid}/analyze").json()
    assert c.get(f"/api/runs/{run['id']}").json()["status"] == "done"
    return c, cid, ctx


def test_mapping_correction_marks_results_stale_and_updates_counts_without_leakage(editable):
    c, cid, ctx = editable
    before = c.get(f"/api/courses/{cid}/explorer").json()
    assert before["freshness"]["stale"] is False
    run1 = before["run"]["id"]
    last = before["usable_papers"] - 1
    # A primary question of the last paper, moved to a different topic.
    topic = next(n for n in before["nodes"] if n["level"] == "topic" and (n["stats"]["last_index"] == last))
    detail = c.get(f"/api/courses/{cid}/explorer/nodes/{topic['id']}").json()
    q = next(q for q in detail["questions"] if q["paper_index"] == last and q["role"] == "primary")
    target = next(n for n in before["nodes"] if n["level"] == "topic" and n["id"] != topic["id"]
                  and n["stats"]["exam_frequency"] == 0)
    assert c.put(f"/api/questions/{q['id']}/mapping", json={"topic_ids": [target["id"]], "status": "A"}).status_code == 200
    fresh = c.get(f"/api/courses/{cid}/freshness").json()
    assert fresh["stale"] and fresh["changed"] == ["mappings"] and "re-analyse" in fresh["message"].lower()
    assert c.get(f"/api/courses/{cid}/results").json()["freshness"]["stale"]

    run2 = c.post(f"/api/courses/{cid}/analyze").json()["id"]
    after = c.get(f"/api/courses/{cid}/explorer").json()
    assert after["run"]["id"] == run2 and after["freshness"]["stale"] is False
    nodes = {n["id"]: n for n in after["nodes"]}
    assert nodes[target["id"]]["stats"]["question_frequency"] == 1
    assert nodes[topic["id"]]["stats"]["question_frequency"] == topic["stats"]["question_frequency"] - 1
    moved = c.get(f"/api/courses/{cid}/explorer/nodes/{target['id']}").json()["questions"]
    assert [m["id"] for m in moved] == [q["id"]] and moved[0]["manual"]

    # Leakage: correcting a label in the last paper must not change any backtest result for earlier papers.
    with ctx.db.session() as s:
        def folds(run_id):
            return {(f.model_name, f.target_index): f.metrics for f in s.execute(
                select(BacktestFold).where(BacktestFold.run_id == run_id, BacktestFold.target_index < last)).scalars()}
        f1, f2 = folds(run1), folds(run2)
    assert f1 and f1.keys() == f2.keys()
    for key in f1:
        for metric, value in f1[key].items():
            assert value == pytest.approx(f2[key][metric], abs=1e-9), (key, metric)


def test_excluded_topics_and_papers(editable):
    c, cid, _ = editable
    ov = c.get(f"/api/courses/{cid}/explorer").json()
    topic = next(n for n in ov["nodes"] if n["level"] == "topic" and n["number"] == "7.2")
    assert c.patch(f"/api/topics/{topic['id']}", json={"excluded": True}).status_code == 200
    exams = c.get(f"/api/courses/{cid}/exams").json()
    paper = next(e for e in exams if e["year"] == 2019 and e["include_in_analysis"])
    assert c.patch(f"/api/exams/{paper['id']}", json={"include_in_analysis": False}).status_code == 200
    assert set(c.get(f"/api/courses/{cid}/freshness").json()["changed"]) == {"papers", "syllabus"}
    c.post(f"/api/courses/{cid}/analyze")
    ov = c.get(f"/api/courses/{cid}/explorer").json()
    preds = c.get(f"/api/courses/{cid}/results").json()["predictions"]
    assert topic["id"] not in {p["topic_id"] for p in preds}
    node = next(n for n in ov["nodes"] if n["id"] == topic["id"])
    assert node["excluded"] and node["prediction"] is None
    assert ov["usable_papers"] == 11
    assert any(e["year"] == 2019 and e["kind"] == "excluded" for e in ov["excluded_papers"])
    top = c.get(f"/api/courses/{cid}/explorer/nodes/{preds[0]['topic_id']}").json()
    rows_2019 = [r for r in top["timeline"] if r.get("year") == 2019]
    assert [r["kind"] for r in rows_2019] == ["excluded"] and rows_2019[0]["appeared"] is None


def test_course_with_only_a_syllabus(tmp_path, demo_files):
    folder, _ = demo_files
    c = TestClient(create_app(AppContext.create(tmp_path / "data", log=False), run_jobs_inline=True))
    cid = c.post("/api/courses", json={"name": "Syllabus only"}).json()["id"]
    files = [("files", (p.name, p.read_bytes())) for p in (folder / "syllabus").iterdir()]
    c.post(f"/api/courses/{cid}/files", data={"kind": "syllabus"}, files=files)
    # Before any analysis the tree is still shown, without counts.
    ov = c.get(f"/api/courses/{cid}/explorer").json()
    assert ov["nodes"] and ov["run"] is None and all("stats" not in n for n in ov["nodes"])
    nid = ov["nodes"][0]["id"]
    assert c.get(f"/api/courses/{cid}/explorer/nodes/{nid}").json()["history_available"] is False
    c.post(f"/api/courses/{cid}/analyze")
    ov = c.get(f"/api/courses/{cid}/explorer").json()
    assert ov["usable_papers"] == 0 and ov["missing_years"] == []
    topics = [n for n in ov["nodes"] if n["level"] == "topic"]
    assert topics and all(n["stats"]["exam_frequency"] == 0 and n["prediction"]["probability"] is None for n in topics)
    d = c.get(f"/api/courses/{cid}/explorer/nodes/{topics[0]['id']}").json()
    assert d["timeline"] == [] and d["questions"] == []
    guide = d["prediction"]["format_guide"]
    assert guide["evidence"] == "inferred" and guide["papers"] == 0
    assert "no past papers" in d["prediction"]["priority_reason"]
