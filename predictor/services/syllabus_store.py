"""Persisting syllabus trees and loading them back as ``TopicTree`` objects."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database.models import CourseTopic, SyllabusVersion
from ..syllabus.parser import ParsedSyllabus, SyllabusNode
from ..syllabus.tree import TopicNode, TopicTree


def current_version(session: Session, course_id: int, create: bool = True) -> SyllabusVersion | None:
    version = session.execute(select(SyllabusVersion).where(SyllabusVersion.course_id == course_id,
                                                             SyllabusVersion.is_current.is_(True))).scalars().first()
    if version is None and create:
        version = SyllabusVersion(course_id=course_id, label="current", is_current=True)
        session.add(version)
        session.flush()
    return version


def get_or_create_version(session: Session, course_id: int, label: str | None) -> SyllabusVersion:
    if not label or label == "current":
        return current_version(session, course_id)
    version = session.execute(select(SyllabusVersion).where(SyllabusVersion.course_id == course_id,
                                                             SyllabusVersion.label == label)).scalars().first()
    if version is None:
        version = SyllabusVersion(course_id=course_id, label=label, is_current=False)
        session.add(version)
        session.flush()
    return version


def topics_of(session: Session, course_id: int, version_id: int | None) -> list[CourseTopic]:
    q = select(CourseTopic).where(CourseTopic.course_id == course_id)
    q = q.where(CourseTopic.syllabus_version_id == version_id) if version_id else q
    return list(session.execute(q.order_by(CourseTopic.depth, CourseTopic.order_no, CourseTopic.id)).scalars())


def to_tree(topics: list[CourseTopic]) -> TopicTree:
    nodes = [TopicNode(id=t.id, parent_id=t.parent_id, depth=t.depth, title=t.title, number=t.number or "",
                       concepts=list(t.concepts or []), description=t.description or "",
                       aliases=list(t.aliases or []), kinds=list(t.kinds or []), hours=t.hours,
                       marks_weight=t.marks_weight, excluded=t.excluded, order_no=t.order_no,
                       source_refs=list(t.source_refs or [])) for t in topics]
    return TopicTree(nodes)


def tree_from_db_as_parsed(topics: list[CourseTopic]) -> ParsedSyllabus:
    """Existing DB topics as a ParsedSyllabus (so new documents can be merged into them)."""
    by_id: dict[int, SyllabusNode] = {}
    roots: list[SyllabusNode] = []
    for t in sorted(topics, key=lambda x: (x.depth, x.order_no, x.id)):
        node = SyllabusNode(title=t.title, depth=t.depth, number=t.number or "",
                            description=[t.description] if t.description else [], concepts=list(t.concepts or []),
                            objectives=list(t.objectives or []), hours=t.hours, marks_weight=t.marks_weight,
                            kinds=list(t.kinds or []), source_refs=list(t.source_refs or []))
        node._db_id = t.id  # type: ignore[attr-defined]
        by_id[t.id] = node
        if t.parent_id and t.parent_id in by_id:
            by_id[t.parent_id].add_child(node)
        else:
            roots.append(node)
    return ParsedSyllabus(roots=roots)


def persist_tree(session: Session, course_id: int, version_id: int, parsed: ParsedSyllabus) -> int:
    """Insert new nodes and update merged ones. Nodes created from DB keep their ids."""
    count = 0
    order = 0

    def save(node: SyllabusNode, parent_id: int | None, depth: int) -> None:
        nonlocal count, order
        order += 1
        db_id = getattr(node, "_db_id", None)
        if db_id:
            row = session.get(CourseTopic, db_id)
            if row is not None and not row.user_edited:
                row.concepts = list(node.concepts)
                row.description = " ".join(node.description)
                row.hours = node.hours if node.hours is not None else row.hours
                row.marks_weight = node.marks_weight if node.marks_weight is not None else row.marks_weight
                row.kinds = list(node.kinds)
                row.source_refs = list(node.source_refs)
                row.objectives = list(node.objectives)
        else:
            row = CourseTopic(course_id=course_id, syllabus_version_id=version_id, parent_id=parent_id, depth=depth,
                              number=node.number or "", title=node.title[:500], description=" ".join(node.description),
                              concepts=list(node.concepts), objectives=list(node.objectives), hours=node.hours,
                              marks_weight=node.marks_weight, kinds=list(node.kinds), aliases=[],
                              source_refs=list(node.source_refs), order_no=order)
            session.add(row)
            session.flush()
            count += 1
        for child in node.children:
            save(child, row.id if row is not None else parent_id, depth + 1)

    for root in parsed.roots:
        save(root, None, 1)
    return count
