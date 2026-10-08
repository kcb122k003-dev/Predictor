from __future__ import annotations

import json

import pytest

from predictor.demo.generator import generate_demo
from predictor.services.analysis_service import AnalysisService
from predictor.services.context import AppContext
from predictor.services.course_service import CourseService
from predictor.services.ingest_service import IngestService


@pytest.fixture(scope="session")
def demo_files(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    truth = generate_demo(out)
    return out, truth


@pytest.fixture(scope="session")
def analysed_demo(tmp_path_factory, demo_files):
    """One fully ingested and analysed demo course shared by the integration tests."""
    folder, truth = demo_files
    app = AppContext.create(tmp_path_factory.mktemp("data"), log=False)
    ingest = IngestService(app)
    cid = CourseService(app).create("Fluid Mechanics demo", is_synthetic=True)
    outcomes = {}
    for kind, sub in (("syllabus", "syllabus"), ("exam", "exams")):
        for path in sorted((folder / sub).iterdir()):
            res = ingest.add_file(cid, path, path.name, kind)
            outcomes[path.name] = {"duplicate": res.duplicate}
            if not res.duplicate:
                outcomes[path.name].update(ingest.process_file(res.file_id))
    service = AnalysisService(app)
    run_id = service.create_run(cid)
    summary = service.run(run_id)
    return {"app": app, "course_id": cid, "run_id": run_id, "summary": summary, "outcomes": outcomes,
            "truth": truth, "folder": folder}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
