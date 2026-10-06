"""Local vector index and the database-backed embedding cache.

A course holds a few thousand vectors at most, so exact search with NumPy takes
milliseconds. FAISS is used when installed and the index is large, but it is not needed.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database.models import EmbeddingCache

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore

    HAVE_FAISS = True
except Exception:
    faiss = None
    HAVE_FAISS = False

FAISS_MIN_SIZE = 50_000


class VectorIndex:
    def __init__(self, vectors: np.ndarray, ids: list):
        self.ids = list(ids)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        if HAVE_FAISS and len(self.ids) >= FAISS_MIN_SIZE:  # pragma: no cover
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    def search(self, query: np.ndarray, k: int = 10) -> list[list[tuple[object, float]]]:
        if query.ndim == 1:
            query = query[None, :]
        if not self.ids:
            return [[] for _ in range(query.shape[0])]
        k = min(k, len(self.ids))
        if self._faiss is not None:  # pragma: no cover
            scores, idx = self._faiss.search(np.ascontiguousarray(query, dtype=np.float32), k)
            return [[(self.ids[j], float(s)) for j, s in zip(row_i, row_s)] for row_i, row_s in zip(idx, scores)]
        sims = query @ self.vectors.T
        out = []
        for row in sims:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            out.append([(self.ids[j], float(row[j])) for j in top])
        return out


class DbEmbeddingCache:
    """Stores neural embeddings in SQLite so unchanged text is never re-encoded."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if not keys:
            return out
        session: Session = self._session_factory()
        try:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                for row in session.execute(select(EmbeddingCache).where(EmbeddingCache.key.in_(chunk))).scalars():
                    out[row.key] = np.frombuffer(row.vector, dtype=np.float32).copy()
        finally:
            session.close()
        return out

    def put_many(self, backend: str, vectors: dict[str, np.ndarray]) -> None:
        if not vectors:
            return
        session: Session = self._session_factory()
        try:
            for key, vec in vectors.items():
                session.merge(EmbeddingCache(key=key, backend=backend, dim=int(vec.shape[0]),
                                             vector=np.asarray(vec, dtype=np.float32).tobytes()))
            session.commit()
        finally:
            session.close()
