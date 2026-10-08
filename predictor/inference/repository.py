"""Local model repository: what one course may learn from the others.

* ``save_course_features`` stores a course's scale-free feature rows (``GENERIC_FEATURES``
  at every backtest cutoff, with the 0/1 outcome). No topic names or raw counts are stored.
* ``general_model_for`` returns the general ranking model for a course: the shipped
  simulated prior, updated with the rows of every *other* real course. Synthetic courses
  (the demo) are excluded, and so is the course itself, so its own history is never counted
  twice and course A's frequencies never become course B's history.

Every general model actually used is recorded in ``model_registry`` (scope "global") with
a fingerprint of the courses it learned from.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sqlalchemy import delete, select

from ..database.models import Course, CourseFeatureSet, ModelRegistry
from ..features.builder import GENERIC_FEATURES
from .generic import VERSION, GeneralModel, load_prior, update_with_rows

_CACHE: dict[str, GeneralModel] = {}


def save_course_features(session, course_id: int, run_id: int | None, layer: str, X: np.ndarray, y: np.ndarray,
                         n_exams: int, is_synthetic: bool) -> None:
    session.execute(delete(CourseFeatureSet).where(CourseFeatureSet.course_id == course_id,
                                                   CourseFeatureSet.layer == layer))
    session.add(CourseFeatureSet(course_id=course_id, run_id=run_id, layer=layer, is_synthetic=is_synthetic,
                                 feature_names=list(GENERIC_FEATURES), n_rows=int(len(y)), n_exams=int(n_exams),
                                 X=np.asarray(X, dtype=np.float32).tobytes(), y=np.asarray(y, dtype=np.float32).tobytes()))


def training_sets(session, exclude_course_id: int | None, layer: str = "topic") -> list[CourseFeatureSet]:
    rows = session.execute(
        select(CourseFeatureSet).join(Course, Course.id == CourseFeatureSet.course_id)
        .where(CourseFeatureSet.layer == layer, CourseFeatureSet.is_synthetic.is_(False), Course.is_synthetic.is_(False))
        .order_by(CourseFeatureSet.course_id)).scalars().all()
    return [r for r in rows if r.course_id != exclude_course_id and r.feature_names == GENERIC_FEATURES and r.n_rows]


def general_model_for(session, course_id: int | None, layer: str = "topic") -> tuple[GeneralModel, dict[str, Any]]:
    prior = load_prior()
    sets = training_sets(session, course_id, layer)
    if not sets:
        info = {"source": prior.source, "real_courses": 0, "real_rows": 0}
        return prior, info
    h = hashlib.sha256()
    for r in sets:
        h.update(f"{r.course_id}:{r.run_id}:{r.n_rows}|".encode())
    key = f"{VERSION}:{layer}:{h.hexdigest()[:16]}"
    model = _CACHE.get(key)
    if model is None:
        p = len(GENERIC_FEATURES)
        X = np.vstack([np.frombuffer(r.X, dtype=np.float32).reshape(r.n_rows, p) for r in sets]).astype(float)
        y = np.concatenate([np.frombuffer(r.y, dtype=np.float32) for r in sets]).astype(float)
        model = update_with_rows(prior, X, y, n_courses=len(sets))
        _CACHE[key] = model
        session.add(ModelRegistry(scope="global", course_id=None, name="general", version=str(VERSION),
                                  source=model.source, fingerprint=key, params=model.to_dict(),
                                  training={"courses": [r.course_id for r in sets], "rows": int(len(y)),
                                            "excluded_course": course_id}))
    info = {"source": model.source, "real_courses": len(sets), "real_rows": int(sum(r.n_rows for r in sets)),
            "fingerprint": key}
    return model, info


def register_course_model(session, course_id: int, run_id: int, name: str, params: dict[str, Any],
                          training: dict[str, Any]) -> None:
    session.add(ModelRegistry(scope="course", course_id=course_id, run_id=run_id, name=name, version=str(VERSION),
                              source="course", params=params, training=training))
