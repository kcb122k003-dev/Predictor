"""In-memory syllabus tree with the three prediction levels: unit, topic and concept.

* Unit: depth-1 nodes.
* Topic (primary prediction unit): depth-2 nodes, plus depth-1 nodes without children.
  In a flat syllabus (only one level) units and topics coincide.
* Concept: leaves (the finest nodes). A topic without children is its own concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TopicNode:
    id: int
    parent_id: int | None
    depth: int
    title: str
    number: str = ""
    concepts: list[str] = field(default_factory=list)
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    hours: float | None = None
    marks_weight: float | None = None
    excluded: bool = False
    order_no: int = 0
    source_refs: list[dict] = field(default_factory=list)

    def label(self) -> str:
        return f"{self.number} {self.title}".strip() if self.number and self.number != "•" else self.title


class TopicTree:
    def __init__(self, nodes: list[TopicNode], topic_level: int | None = None):
        self.nodes = {n.id: n for n in nodes}
        self.children: dict[int | None, list[int]] = {}
        for n in sorted(nodes, key=lambda x: (x.depth, x.order_no, x.id)):
            parent = n.parent_id if n.parent_id in self.nodes else None
            self.children.setdefault(parent, []).append(n.id)
        self.max_depth = max((n.depth for n in nodes), default=0)
        self.topic_level = topic_level or (2 if self.max_depth >= 2 else 1)

    def __len__(self) -> int:
        return len(self.nodes)

    def roots(self) -> list[int]:
        return list(self.children.get(None, []))

    def kids(self, node_id: int) -> list[int]:
        return list(self.children.get(node_id, []))

    def ancestors(self, node_id: int) -> list[int]:
        out = []
        node = self.nodes.get(node_id)
        while node is not None and node.parent_id in self.nodes:
            out.append(node.parent_id)
            node = self.nodes[node.parent_id]
        return out

    def descendants(self, node_id: int) -> list[int]:
        out: list[int] = []
        stack = list(self.kids(node_id))
        while stack:
            nid = stack.pop()
            out.append(nid)
            stack.extend(self.kids(nid))
        return out

    def is_excluded(self, node_id: int) -> bool:
        return any(self.nodes[i].excluded for i in [node_id, *self.ancestors(node_id)] if i in self.nodes)

    def path_titles(self, node_id: int) -> list[str]:
        ids = list(reversed(self.ancestors(node_id))) + [node_id]
        return [self.nodes[i].title for i in ids]

    def path_label(self, node_id: int) -> str:
        return " > ".join(self.path_titles(node_id))

    # -- levels -----------------------------------------------------------------
    def unit_of(self, node_id: int) -> int:
        anc = self.ancestors(node_id)
        return anc[-1] if anc else node_id

    def topic_of(self, node_id: int) -> int:
        """The topic-level ancestor (or the node itself when it is at or above topic level)."""
        node = self.nodes[node_id]
        if node.depth <= self.topic_level:
            return node_id
        for anc in self.ancestors(node_id):
            if self.nodes[anc].depth == self.topic_level:
                return anc
        return node_id

    def topic_ids(self) -> list[int]:
        out = []
        for nid in self._ordered():
            n = self.nodes[nid]
            if self.is_excluded(nid):
                continue
            if n.depth == self.topic_level or (n.depth < self.topic_level and not self.kids(nid)):
                out.append(nid)
        return out

    def unit_ids(self) -> list[int]:
        return [nid for nid in self.roots() if not self.is_excluded(nid)]

    def concept_ids(self) -> list[int]:
        return [nid for nid in self._ordered() if not self.kids(nid) and not self.is_excluded(nid)]

    def mappable_ids(self) -> list[int]:
        """Nodes a question can be mapped to: topic level and below, plus childless units."""
        return [nid for nid in self._ordered() if not self.is_excluded(nid)
                and (self.nodes[nid].depth >= self.topic_level or not self.kids(nid))]

    def _ordered(self) -> list[int]:
        out: list[int] = []

        def visit(nid: int) -> None:
            out.append(nid)
            for c in self.kids(nid):
                visit(c)

        for r in self.roots():
            visit(r)
        return out

    def document(self, node_id: int) -> str:
        """Text used to represent a node for alignment: title, aliases, concepts, description, parent title."""
        n = self.nodes[node_id]
        parts = [n.title, n.title] + [a for a in n.aliases for _ in (0, 1)] + list(n.concepts)
        if n.description:
            parts.append(" ".join(n.description.split()[:80]))
        anc = self.ancestors(node_id)
        if anc:
            parts.append(self.nodes[anc[0]].title)
        return " . ".join(p for p in parts if p)

    def term_text(self, node_id: int) -> str:
        n = self.nodes[node_id]
        return " ".join([n.title, *n.aliases, *n.concepts, n.description])
