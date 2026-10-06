"""Database engine, sessions and the FTS5 search index."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    kind UNINDEXED, ref_id UNINDEXED, course_id UNINDEXED, text,
    tokenize = 'porter unicode61'
)
"""


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver hook
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


class Database:
    """Owns the engine for one SQLite file (or an in-memory database for tests)."""

    def __init__(self, path: Path | str | None):
        if path is None or str(path) == ":memory:":
            url = "sqlite+pysqlite:///:memory:"
            from sqlalchemy.pool import StaticPool

            self.engine: Engine = create_engine(
                url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+pysqlite:///{Path(path).as_posix()}"
            self.engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self._sessionmaker = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.fts_available = True
        self.create_all()

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        try:
            with self.engine.begin() as conn:
                conn.execute(text(FTS_DDL))
        except Exception:  # pragma: no cover - SQLite built without FTS5
            self.fts_available = False

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def new_session(self) -> Session:
        return self._sessionmaker()

    def dispose(self) -> None:
        self.engine.dispose()
