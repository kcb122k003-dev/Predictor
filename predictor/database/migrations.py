"""Schema migrations for existing databases.

``Base.metadata.create_all`` creates missing tables but never changes existing ones, so
columns added in later versions are added here with ``ALTER TABLE ... ADD COLUMN`` and a
default. Existing rows keep their data. Each applied version is recorded in
``schema_version``. Migrations only add; nothing is dropped or rewritten.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

SCHEMA_VERSION = 2

# version -> list of (table, column, SQL type with default)
MIGRATIONS: dict[int, list[tuple[str, str, str]]] = {
    2: [
        ("course", "is_synthetic", "BOOLEAN NOT NULL DEFAULT 0"),
        ("exam", "source", "VARCHAR(20) NOT NULL DEFAULT 'upload'"),
        ("analysis_run", "engine_version", "VARCHAR(20) NOT NULL DEFAULT ''"),
        ("model_result", "role", "VARCHAR(20) NOT NULL DEFAULT ''"),
        ("model_result", "scope", "VARCHAR(20) NOT NULL DEFAULT 'course'"),
        ("model_result", "status", "VARCHAR(20) NOT NULL DEFAULT ''"),
        ("model_result", "weight", "FLOAT"),
        ("model_result", "reliability", "FLOAT"),
        ("model_result", "evidence", "JSON NOT NULL DEFAULT '{}'"),
        ("prediction", "evidence_strength", "VARCHAR(20) NOT NULL DEFAULT ''"),
        ("prediction", "uncertainty", "JSON NOT NULL DEFAULT '{}'"),
    ],
}


def current_version(engine: Engine) -> int:
    insp = inspect(engine)
    if "schema_version" not in insp.get_table_names():
        return 1
    with engine.connect() as conn:
        v = conn.execute(text("SELECT MAX(version) FROM schema_version")).scalar()
    return int(v or 1)


# Courses created by the version 1 demo command, recognisable by their fixed name and description.
OLD_DEMO_NAME = "%(synthetic demo)%"
OLD_DEMO_DESCRIPTION = "Generated example data with planted patterns. Not real exams."


def migrate(engine: Engine) -> list[str]:
    """Bring an existing database up to SCHEMA_VERSION. Returns the columns that were added."""
    added: list[str] = []
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    from_version = current_version(engine) if "course" in tables else SCHEMA_VERSION
    with engine.begin() as conn:
        for version in sorted(MIGRATIONS):
            for table, column, ddl in MIGRATIONS[version]:
                if table not in tables:
                    continue
                existing = {c["name"] for c in inspect(conn).get_columns(table)}
                if column not in existing:
                    conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))
                    added.append(f"{table}.{column}")
        if from_version < 2 and "course" in tables and "course.is_synthetic" in added:
            # Version 1 had no synthetic flag: mark the old demo course (and its papers) so its planted data
            # never trains the cross-course model.
            conn.execute(text("UPDATE course SET is_synthetic = 1 WHERE name LIKE :n OR description = :d"),
                         {"n": OLD_DEMO_NAME, "d": OLD_DEMO_DESCRIPTION})
            if "exam" in tables:
                conn.execute(text("UPDATE exam SET source = 'demo' WHERE course_id IN "
                                  "(SELECT id FROM course WHERE is_synthetic = 1)"))
        have = conn.execute(text("SELECT MAX(version) FROM schema_version")).scalar() if "schema_version" in tables else None
        if have is None or int(have) < SCHEMA_VERSION:
            note = ("added " + ", ".join(added)) if added else "new database"
            conn.execute(text("INSERT INTO schema_version (version, applied_at, note) VALUES (:v, CURRENT_TIMESTAMP, :n)"),
                         {"v": SCHEMA_VERSION, "n": note[:2000]})
    return added
