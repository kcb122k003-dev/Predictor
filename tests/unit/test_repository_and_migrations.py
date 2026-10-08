"""Cross-course repository isolation, synthetic exclusion and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from sqlalchemy import inspect, select, text

from predictor.database.migrations import SCHEMA_VERSION, current_version
from predictor.database.models import Course, CourseFeatureSet, Exam, ModelRegistry
from predictor.database.session import Database
from predictor.evaluation.backtest import BacktestEngine
from predictor.inference.generic import load_prior, panel_rows
from predictor.inference.repository import general_model_for, save_course_features, training_sets
from tests.unit.test_low_data import planted

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "schema_v1.sql"


def _course(s, name: str, synthetic: bool = False) -> int:
    c = Course(name=name, is_synthetic=synthetic)
    s.add(c)
    s.flush()
    return c.id


def test_general_model_learns_only_from_other_real_courses(settings):
    db = Database(":memory:")
    with db.session() as s:
        a, b, demo = _course(s, "A"), _course(s, "B"), _course(s, "demo", synthetic=True)
        for cid, seed in ((a, 1), (demo, 2)):
            X, y, _ = panel_rows(planted(10, seed=seed), settings)
            save_course_features(s, cid, None, "topic", X, y, 10, is_synthetic=(cid == demo))
    with db.session() as s:
        # Course B learns from A (real) but never from the synthetic demo; A never learns from itself.
        assert [r.course_id for r in training_sets(s, b)] == [a]
        assert training_sets(s, a) == []
        gm_b, info_b = general_model_for(s, b)
        gm_a, info_a = general_model_for(s, a)
        assert info_b["real_courses"] == 1 and info_a["real_courses"] == 0
        assert gm_a is load_prior()
        assert not np.allclose(gm_b.coef, gm_a.coef)
        assert s.execute(select(ModelRegistry).where(ModelRegistry.scope == "global")).scalars().first() is not None
        rows = s.execute(select(CourseFeatureSet)).scalars().all()
        assert {r.course_id: r.is_synthetic for r in rows} == {a: False, demo: True}


def test_other_courses_do_not_change_this_courses_history(settings):
    """Course A's data can change the cross-course model, never course B's frequencies or posteriors."""
    db = Database(":memory:")
    with db.session() as s:
        a, b = _course(s, "A"), _course(s, "B")
        X, y, _ = panel_rows(planted(14, seed=5), settings)
        save_course_features(s, a, None, "topic", X, y, 14, is_synthetic=False)
        gm_with_a, _ = general_model_for(s, b)
    panel_b = planted(6, seed=9)
    alone = BacktestEngine(planted(6, seed=9), settings, general=load_prior()).run(audit_leakage=False)
    shared = BacktestEngine(panel_b, settings, general=gm_with_a).run(audit_leakage=False)
    assert np.array_equal(alone.ctx.panel.Y, shared.ctx.panel.Y)
    for name in ("frequency", "beta_binomial", "recency", "hazard", "semantic", "markov"):
        assert np.allclose(alone.final_outputs[name].scores, shared.final_outputs[name].scores), name
    p1 = alone.final_outputs["beta_binomial"].info["posterior"]
    p2 = shared.final_outputs["beta_binomial"].info["posterior"]
    assert np.allclose(p1.mean, p2.mean) and p1.exams == p2.exams == 6
    assert not np.allclose(alone.final_outputs["general"].scores, shared.final_outputs["general"].scores)


def test_migration_from_version_1_keeps_data(tmp_path):
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(FIXTURE.read_text(encoding="utf-8"))
    con.execute("INSERT INTO course (id, name, code, description, created_at, settings) "
                "VALUES (1, 'Old course', 'X1', '', '2024-01-01', '{}')")
    con.execute("INSERT INTO exam (id, course_id, title, subject, year, calendar, session, exam_type, exam_date, "
                "order_index, duration, examiner, instructions, structure, metadata_confidence, include_in_analysis, "
                "exclusion_reason, user_edited_fields, created_at) VALUES (1, 1, 'Paper', '', 2020, 'AD', '', '', '', "
                "2020.5, '', '', '[]', '{}', '{}', 1, '', '[]', '2024-01-01')")
    con.commit()
    con.close()
    db = Database(path)
    cols = {c["name"] for c in inspect(db.engine).get_columns("exam")}
    assert "source" in cols
    assert "is_synthetic" in {c["name"] for c in inspect(db.engine).get_columns("course")}
    assert {"status", "weight", "reliability", "evidence"} <= {c["name"] for c in inspect(db.engine).get_columns("model_result")}
    assert "model_registry" in inspect(db.engine).get_table_names()
    assert current_version(db.engine) == SCHEMA_VERSION
    with db.session() as s:
        exam = s.get(Exam, 1)
        assert exam.title == "Paper" and exam.year == 2020 and exam.source == "upload"
        assert s.get(Course, 1).is_synthetic is False
    db.dispose()
    # Opening again is a no-op.
    db2 = Database(path)
    assert db2.migrated_columns == []
    with db2.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar() == 1
    db2.dispose()


def test_migration_marks_the_old_demo_course_synthetic(tmp_path):
    path = tmp_path / "old_demo.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(FIXTURE.read_text(encoding="utf-8"))
    con.execute("INSERT INTO course (id, name, code, description, created_at, settings) VALUES "
                "(1, 'Fluid Mechanics (synthetic demo)', 'CE 501', "
                "'Generated example data with planted patterns. Not real exams.', '2024-01-01', '{}')")
    con.execute("INSERT INTO course (id, name, code, description, created_at, settings) VALUES "
                "(2, 'Thermodynamics', 'ME 1', '', '2024-01-01', '{}')")
    for eid, cid in ((1, 1), (2, 2)):
        con.execute("INSERT INTO exam (id, course_id, title, subject, year, calendar, session, exam_type, exam_date, "
                    "order_index, duration, examiner, instructions, structure, metadata_confidence, "
                    "include_in_analysis, exclusion_reason, user_edited_fields, created_at) VALUES "
                    f"({eid}, {cid}, 'P', '', 2020, 'AD', '', '', '', 2020.5, '', '', '[]', '{{}}', '{{}}', 1, '', '[]', "
                    "'2024-01-01')")
    con.commit()
    con.close()
    db = Database(path)
    with db.session() as s:
        assert s.get(Course, 1).is_synthetic is True and s.get(Course, 2).is_synthetic is False
        assert s.get(Exam, 1).source == "demo" and s.get(Exam, 2).source == "upload"
    db.dispose()
