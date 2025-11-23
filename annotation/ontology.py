from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Competency:
    id: str
    label: str
    parent: str | None
    description: str | None = None


class Ontology:
    def __init__(
        self,
        data: dict,
        focus_parents: Sequence[str] | None = None,
        leaves_only: bool = True,
    ) -> None:
        self._nodes = data["nodes"]
        self._links = data["links"]
        self._focus_parents = set(focus_parents) if focus_parents else set()
        self._leaves_only = leaves_only
        self._children_map = self._build_children_map()
        self._competencies = self._extract_competencies()

    @property
    def competencies(self) -> Sequence[Competency]:
        return self._competencies

    def _build_children_map(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = defaultdict(list)
        for link in self._links:
            mapping[link["source"]].append(link["target"])
        return mapping

    def _extract_competencies(self) -> List[Competency]:
        competencies: List[Competency] = []

        if self._focus_parents:
            candidate_ids: set[str] = set()
            for parent in self._focus_parents:
                candidate_ids.update(self.descendants_of(parent))
        else:
            candidate_ids = {
                node["id"]
                for node in self._nodes
                if isinstance(node["id"], str)
                and node["id"].startswith("http://example.org/competencies#")
            }

        if self._focus_parents:
            for node in self._nodes:
                node_id = node["id"]
                if node_id not in candidate_ids:
                    continue
                if self._leaves_only and self.has_children(node_id):
                    continue
                    competencies.append(
                        Competency(
                            id=node_id,
                            label=node["label"],
                            parent=self.parent_of(node_id),
                            description=node.get("description"),
                        )
                    )
        else:
            for node in self._nodes:
                node_id = node["id"]
                if node_id not in candidate_ids:
                    continue
                if self._leaves_only and self.has_children(node_id):
                    continue
                competencies.append(
                    Competency(
                        id=node_id,
                        label=node["label"],
                        parent=self.parent_of(node_id),
                        description=node.get("description"),
                    )
                )

        return competencies

    def parent_of(self, node_id: str) -> str | None:
        for link in self._links:
            if link["target"] == node_id:
                return link["source"]
        return None

    def children_of(self, node_id: str) -> List[str]:
        return [link["target"] for link in self._links if link["source"] == node_id]

    def has_children(self, node_id: str) -> bool:
        return bool(self._children_map.get(node_id))

    def descendants_of(self, root_id: str) -> set[str]:
        visited: set[str] = set()
        queue: deque[str] = deque([root_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for child in self._children_map.get(current, []):
                if child not in visited:
                    queue.append(child)
        return visited

