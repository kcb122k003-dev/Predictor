"""Builds the regression snapshot of the demo course (run this file to refresh it after an intended change)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT = Path(__file__).with_name("snapshots") / "demo_snapshot.json"


def build_snapshot() -> dict:
    from sqlalchemy import select

    from predictor.database.models import CourseTopic, Prediction
    from predictor.demo.generator import generate_demo
    from predictor.services.analysis_service import AnalysisService
    from predictor.services.context import AppContext
    from predictor.services.course_service import CourseService
    from predictor.services.ingest_service import IngestService
    from predictor.services.results_service import ResultsService

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generate_demo(tmp / "demo")
        app = AppContext.create(tmp / "data", log=False)
        ingest = IngestService(app)
        cid = CourseService(app).create("regression")
        for kind, sub in (("syllabus", "syllabus"), ("exam", "exams")):
            for path in sorted((tmp / "demo" / sub).iterdir()):
                res = ingest.add_file(cid, path, path.name, kind)
                if not res.duplicate:
                    ingest.process_file(res.file_id)
        service = AnalysisService(app)
        run_id = service.create_run(cid)
        summary = service.run(run_id)
        with app.db.session() as s:
            numbers = {t.id: t.number for t in s.execute(select(CourseTopic).where(CourseTopic.course_id == cid)).scalars()}
            preds = s.execute(select(Prediction).where(Prediction.run_id == run_id, Prediction.layer == "topic")
                              .order_by(Prediction.rank)).scalars().all()
            ranking = [numbers[p.topic_id] for p in preds]
            categories = {numbers[p.topic_id]: p.category for p in preds}
        models = {m["name"]: m["metrics"].get("ndcg") for m in ResultsService(app).models(run_id)["models"]
                  if m["layer"] == "topic" and m["enabled"]}
        app.db.dispose()
    return {
        "exams": summary["exams"], "questions": summary["questions"], "topics": summary["topics"], "k": summary["k"],
        "status_counts": summary["status_counts"], "selected_model": summary["selected_model"],
        "calibrated": summary["calibrated"], "ranking": ranking, "categories": categories, "model_ndcg": models,
    }


if __name__ == "__main__":
    data = build_snapshot()
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {SNAPSHOT}")
