"""HTTP API: the browser workflow end to end (jobs run inline so the test is deterministic)."""

import pytest
from fastapi.testclient import TestClient

from predictor.app.server import create_app
from predictor.services.context import AppContext


@pytest.fixture(scope="module")
def client(tmp_path_factory, demo_files):
    folder, _ = demo_files
    app = create_app(AppContext.create(tmp_path_factory.mktemp("api"), log=False), run_jobs_inline=True)
    c = TestClient(app)
    cid = c.post("/api/courses", json={"name": "API course"}).json()["id"]
    files = [("files", (p.name, p.read_bytes())) for p in (folder / "syllabus").iterdir()]
    assert c.post(f"/api/courses/{cid}/files", data={"kind": "syllabus"}, files=files).status_code == 200
    exams = sorted((folder / "exams").iterdir())[:9]
    files = [("files", (p.name, p.read_bytes())) for p in exams]
    c.post(f"/api/courses/{cid}/files", data={"kind": "exam"}, files=files)
    run = c.post(f"/api/courses/{cid}/analyze").json()
    return c, cid, run


def test_health_and_static(client):
    c, _, _ = client
    health = c.get("/api/health").json()
    assert health["external_services"] is False and "ocr" in health
    assert c.get("/").status_code == 200
    assert c.get("/static/js/app.js").status_code == 200


def test_results_flow(client):
    c, cid, run = client
    assert c.get(f"/api/runs/{run['id']}").json()["status"] == "done"
    res = c.get(f"/api/courses/{cid}/results").json()
    preds = res["predictions"]
    assert preds and preds[0]["rank"] == 1 and preds[0]["evidence"]
    detail = c.get(f"/api/runs/{run['id']}/topics/{preds[0]['topic_id']}").json()
    assert detail["questions"] and detail["topic"]["path"]
    models = c.get(f"/api/runs/{run['id']}/models").json()
    assert any(m["selected"] for m in models["models"])
    for key in ("charts", "structure", "coverage", "ablation", "calibration", "papers", "sufficiency"):
        assert c.get(f"/api/runs/{run['id']}/artifacts/{key}").status_code == 200
    papers = c.post(f"/api/runs/{run['id']}/papers", json={"seed": 11, "variants": 2}).json()["papers"]
    assert len(papers) == 2 and "hypothetical" in papers[0]["title"]


def test_review_endpoints(client):
    c, cid, _ = client
    exams = c.get(f"/api/courses/{cid}/exams").json()
    eid = exams[0]["id"]
    assert c.patch(f"/api/exams/{eid}", json={"session": "Spring"}).status_code == 200
    assert c.patch(f"/api/exams/{eid}", json={"bogus": 1}).status_code == 400
    tree = c.get(f"/api/exams/{eid}/questions").json()
    leaf = next(q for q in tree["questions"] if q["is_leaf"]) if any(q["is_leaf"] for q in tree["questions"]) \
        else tree["questions"][0]["children"][0]
    split = c.post(f"/api/questions/{leaf['id']}/split", json={"parts": ["First part.", "Second part."]}).json()
    assert [q["text"] for q in split] == ["First part.", "Second part."]
    merged = c.post(f"/api/questions/{split[0]['id']}/merge-next").json()
    assert merged["text"] == "First part. Second part."
    syl = c.get(f"/api/courses/{cid}/syllabus").json()
    unit = syl["versions"][0]["topics"][0]
    assert c.patch(f"/api/topics/{unit['id']}", json={"aliases": ["fluid properties"]}).status_code == 200
    new = c.post(f"/api/courses/{cid}/topics", json={"title": "Open channel flow", "parent_id": None}).json()
    assert c.delete(f"/api/topics/{new['id']}").status_code == 200
    assert c.get("/api/topics/999999").status_code in (404, 405)


def test_search_and_export(client):
    c, cid, run = client
    res = c.get(f"/api/courses/{cid}/search", params={"q": "venturimeter"}).json()
    assert res["questions"] and "venturimeter" in res["questions"][0]["text"].lower()
    assert res["topics"]
    for fmt, magic in (("pdf", b"%PDF"), ("xlsx", b"PK"), ("json", b"{")):
        r = c.get(f"/api/runs/{run['id']}/export", params={"format": fmt})
        assert r.status_code == 200 and r.content.startswith(magic)
    assert c.get(f"/api/runs/{run['id']}/export", params={"format": "doc"}).status_code == 400


def test_settings_validation(client):
    c, _, _ = client
    assert c.put("/api/settings", json={"alignment": {"strictness": 0.7}}).status_code == 200
    assert c.get("/api/settings").json()["effective"]["alignment"]["strictness"] == 0.7
    assert c.put("/api/settings", json={"alignment": {"strictnes": 0.7}}).status_code == 400
    assert c.put("/api/settings", json={"ocr": {"dpi": "high"}}).status_code == 400
    c.put("/api/settings", json={})


def test_missing_syllabus_gives_clear_error(client):
    c, _, _ = client
    cid = c.post("/api/courses", json={"name": "Empty"}).json()["id"]
    run = c.post(f"/api/courses/{cid}/analyze").json()
    status = c.get(f"/api/runs/{run['id']}").json()
    assert status["status"] == "error" and "syllabus" in status["message"].lower()
