from typing import List, Optional
from ..cognition.contracts import EvidenceItem, EvidenceSource, EvidenceRelation


class TaxonomyBuilder:
    """BFS no CGE/WordNet gerando EvidenceItem(source=taxonomy).

    max_depth_soft=4: profundidade padrao de busca
    max_depth_hard=8: limite absoluto
    """

    def __init__(self, cge, max_depth_soft: int = 4, max_depth_hard: int = 8):
        self.cge = cge
        self.max_depth_soft = max_depth_soft
        self.max_depth_hard = max_depth_hard

    def build(self, concept: str, max_depth: Optional[int] = None) -> List[EvidenceItem]:
        """Busca relacoes taxonomicas (is_a, has_property) no CGE.

        Returns lista de EvidenceItem ordenada por confianca (decrescente).
        """
        depth = min(max_depth or self.max_depth_soft, self.max_depth_hard)
        items: List[EvidenceItem] = []
        visited = set()

        if concept not in self.cge.nodes:
            return items

        queue = [(concept, 0)]
        while queue and len(items) < 100:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            edges = self.cge.query_edges(current)
            for edge in edges:
                neighbor_data = self._resolve_neighbor(current, edge)
                if neighbor_data is None:
                    continue
                neighbor_id, relation_hint = neighbor_data
                if neighbor_id in visited:
                    continue

                rel = self._classify_relation(edge, relation_hint)
                confidence = float(edge.get("confidence", 0.5))
                weight = float(edge.get("weight", 0.5))
                combined = min(1.0, (confidence + weight) / 2 + 0.1 * max(0, 1 - d / depth))

                items.append(EvidenceItem(
                    source=EvidenceSource.TAXONOMY,
                    relation=rel,
                    confidence=combined,
                    raw_score=combined,
                    trace=f"CGE: {current} --[{rel.value}]--> {neighbor_id} (depth={d})",
                    metadata={"from": current, "to": neighbor_id, "depth": d},
                ))

                if d < depth:
                    queue.append((neighbor_id, d + 1))

        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    def build_hierarchy(self, root: str) -> List[EvidenceItem]:
        """Usa get_concept_hierarchy do CGE para construir arvore taxonomica."""
        hierarchy = self.cge.get_concept_hierarchy(root, max_depth=self.max_depth_hard)
        items: List[EvidenceItem] = []

        def walk(node: dict, depth: int = 0):
            if depth > self.max_depth_hard:
                return
            name = node.get("name", "")
            confidence = float(node.get("confidence", 0.5))
            if name and name != root:
                items.append(EvidenceItem(
                    source=EvidenceSource.TAXONOMY,
                    relation=EvidenceRelation.IS_A,
                    confidence=min(1.0, confidence + 0.05 * max(0, 1 - depth / self.max_depth_hard)),
                    raw_score=confidence,
                    trace=f"Hierarchy: {root} -> ... -> {name} (depth={depth})",
                    metadata={"root": root, "node": name, "depth": depth},
                ))
            for child in node.get("children", []):
                walk(child, depth + 1)

        walk(hierarchy)
        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    def _resolve_neighbor(self, current: str, edge: dict):
        edge_from = edge.get("from", "")
        edge_to = edge.get("to", "")
        if edge_from == current and edge_to in self.cge.nodes:
            return edge_to, "forward"
        if edge_to == current and edge_from in self.cge.nodes:
            return edge_from, "reverse"
        return None

    def _classify_relation(self, edge: dict, hint: str) -> EvidenceRelation:
        edge_type_str = str(edge.get("edge_type", "")).lower()
        if "is_a" in edge_type_str or "hypernym" in edge_type_str:
            return EvidenceRelation.IS_A
        if "has" in edge_type_str or "property" in edge_type_str:
            return EvidenceRelation.HAS_PROPERTY
        if "causes" in edge_type_str or "caused_by" in edge_type_str:
            return EvidenceRelation.CAUSES
        if "similar" in edge_type_str or "synonym" in edge_type_str:
            return EvidenceRelation.SIMILAR
        return EvidenceRelation.RELATED
