"""Review and manual override: exams, questions, mappings and the syllabus tree."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ..services.ingest_service import IngestService
from ..services.review_service import ReviewService
from .deps import ctx

router = APIRouter()


class QuestionIn(BaseModel):
    parent_id: int | None = None
    label: str
    text: str
    marks: float | None = None


class SplitIn(BaseModel):
    parts: list[str]


class MappingIn(BaseModel):
    topic_ids: list[int] = []
    status: str = "A"


class TopicIn(BaseModel):
    title: str
    parent_id: int | None = None
    number: str = ""
    concepts: list[str] = []
    hours: float | None = None


@router.get("/courses/{course_id}/exams")
def exams(request: Request, course_id: int) -> list[dict[str, Any]]:
    return ReviewService(ctx(request)).exams(course_id)


@router.patch("/exams/{exam_id}")
def patch_exam(request: Request, exam_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    return ReviewService(ctx(request)).update_exam(exam_id, changes)


@router.delete("/exams/{exam_id}")
def delete_exam(request: Request, exam_id: int) -> dict[str, Any]:
    ReviewService(ctx(request)).delete_exam(exam_id)
    return {"ok": True}


@router.get("/exams/{exam_id}/questions")
def question_tree(request: Request, exam_id: int) -> dict[str, Any]:
    return ReviewService(ctx(request)).question_tree(exam_id)


@router.post("/exams/{exam_id}/questions")
def add_question(request: Request, exam_id: int, body: QuestionIn) -> dict[str, Any]:
    return ReviewService(ctx(request)).add_question(exam_id, body.parent_id, body.label, body.text, body.marks)


@router.get("/questions")
def questions_by_ids(request: Request, ids: str = Query("")) -> list[dict[str, Any]]:
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()][:500]
    return ReviewService(ctx(request)).questions_by_ids(id_list)


@router.patch("/questions/{qid}")
def patch_question(request: Request, qid: int, changes: dict[str, Any]) -> dict[str, Any]:
    return ReviewService(ctx(request)).update_question(qid, changes)


@router.delete("/questions/{qid}")
def delete_question(request: Request, qid: int) -> dict[str, Any]:
    ReviewService(ctx(request)).delete_question(qid)
    return {"ok": True}


@router.post("/questions/{qid}/split")
def split_question(request: Request, qid: int, body: SplitIn) -> list[dict[str, Any]]:
    return ReviewService(ctx(request)).split_question(qid, body.parts)


@router.post("/questions/{qid}/merge-next")
def merge_question(request: Request, qid: int) -> dict[str, Any]:
    return ReviewService(ctx(request)).merge_with_next(qid)


@router.put("/questions/{qid}/mapping")
def set_mapping(request: Request, qid: int, body: MappingIn) -> dict[str, Any]:
    return ReviewService(ctx(request)).set_mapping(qid, body.topic_ids, body.status)


@router.delete("/questions/{qid}/mapping")
def clear_mapping(request: Request, qid: int) -> dict[str, Any]:
    ReviewService(ctx(request)).clear_manual_mapping(qid)
    return {"ok": True}


@router.get("/courses/{course_id}/mappings")
def mappings(request: Request, course_id: int, status: str | None = None,
             exam_id: int | None = None) -> list[dict[str, Any]]:
    return ReviewService(ctx(request)).mappings(course_id, status, exam_id)


@router.get("/courses/{course_id}/syllabus")
def syllabus(request: Request, course_id: int) -> dict[str, Any]:
    return ReviewService(ctx(request)).syllabus(course_id)


@router.post("/courses/{course_id}/syllabus/rebuild")
def rebuild(request: Request, course_id: int) -> dict[str, Any]:
    return IngestService(ctx(request)).rebuild_syllabus(course_id)


@router.post("/courses/{course_id}/topics")
def add_topic(request: Request, course_id: int, body: TopicIn) -> dict[str, Any]:
    return ReviewService(ctx(request)).add_topic(course_id, body.title, body.parent_id, body.number, body.concepts,
                                                 body.hours)


@router.patch("/topics/{topic_id}")
def patch_topic(request: Request, topic_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    return ReviewService(ctx(request)).update_topic(topic_id, changes)


@router.delete("/topics/{topic_id}")
def delete_topic(request: Request, topic_id: int) -> dict[str, Any]:
    ReviewService(ctx(request)).delete_topic(topic_id)
    return {"ok": True}
