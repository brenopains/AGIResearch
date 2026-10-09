from typing import List, Optional
from ..cognition.contracts import EvidenceItem, EvidenceSource, EvidenceRelation


class PerceptualBuilder:
    """CGE activation levels -> EvidenceItem(source=perceptual).

    Mede quao ativo um conceito esta no CGE e usa isso como evidencia.
    """

    def __init__(self, cge, activation_decay: float = 0.5, max_hops: int = 3):
        self.cge = cge
        self.activation_decay = activation_decay
        self.max_hops = max_hops

    def build(self, seed_concepts: List[str], min_activation: float = 0.05) -> List[EvidenceItem]:
        """Propaga ativacao no CGE a partir de seed_concepts.

        Returns EvidenceItem para cada no com ativacao > min_activation.
        """
        activation_map = self.cge.spread_activation(
            seed_ids=seed_concepts,
            decay=self.activation_decay,
            max_hops=self.max_hops,
        )

        items: List[EvidenceItem] = []
        for node_id, activation in activation_map.items():
            if activation < min_activation:
                continue
            if node_id in seed_concepts and activation == 1.0:
                continue

            items.append(EvidenceItem(
                source=EvidenceSource.PERCEPTUAL,
                relation=self._infer_relation(node_id, activation),
                confidence=min(1.0, activation * 1.2),
                raw_score=activation,
                trace=f"Activation: {seed_concepts} -> {node_id} ({activation:.3f})",
                metadata={
                    "node_id": node_id,
                    "activation": round(activation, 4),
                    "seed": seed_concepts,
                    "hops": self._estimate_hops(activation),
                },
            ))

        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    def build_from_landscape(self, seed_ids: List[str], landscape: Optional[dict] = None) -> List[EvidenceItem]:
        """Usa activate_with_landscape para propagacao ciente do gradiente."""
        activation_map = self.cge.activate_with_landscape(
            seed_ids=seed_ids,
            landscape=landscape,
            temperature=0.5,
            max_energy=10.0,
        )

        items: List[EvidenceItem] = []
        for node_id, activation in activation_map.items():
            if activation < 0.05 or node_id in seed_ids:
                continue

            items.append(EvidenceItem(
                source=EvidenceSource.PERCEPTUAL,
                relation=EvidenceRelation.RELATED,
                confidence=min(1.0, activation),
                raw_score=activation,
                trace=f"Landscape: {seed_ids} -> {node_id} ({activation:.3f})",
                metadata={"node_id": node_id, "activation": round(activation, 4), "seed": seed_ids},
            ))

        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    def _infer_relation(self, node_id: str, activation: float) -> EvidenceRelation:
        node = self.cge.nodes.get(node_id)
        if node is None:
            return EvidenceRelation.RELATED
        confidence = float(node.get("confidence", 0))
        if confidence > 0.6 and activation > 0.5:
            return EvidenceRelation.IS_A
        if confidence > 0.4:
            return EvidenceRelation.HAS_PROPERTY
        return EvidenceRelation.RELATED

    def _estimate_hops(self, activation: float) -> int:
        if activation > 0.7:
            return 1
        if activation > 0.4:
            return 2
        return 3
