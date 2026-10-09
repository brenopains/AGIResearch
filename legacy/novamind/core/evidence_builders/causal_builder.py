from typing import List, Optional, Dict, Any
from ..cognition.contracts import EvidenceItem, EvidenceSource, EvidenceRelation


class CausalBuilder:
    """WorldModel + ScienceCurriculum -> EvidenceItem(source=causal).

    Consulta relacoes causais nos 3 niveis de Pearl:
    - associacao (correlacao)
    - intervencao (causa direta)
    - contrafactual (e se...)
    """

    def __init__(self, world_model, science_curriculum: Optional[Dict[str, Any]] = None):
        self.world_model = world_model
        self.science_curriculum = science_curriculum or {}

    def build_association(self, target: str, evidence: Dict[str, float]) -> List[EvidenceItem]:
        """Nivel 1: associacao (correlacao observacional)."""
        result = self.world_model.query_association(target, evidence)
        items: List[EvidenceItem] = []

        if result and result.get("probability", 0) > 0:
            items.append(EvidenceItem(
                source=EvidenceSource.CAUSAL,
                relation=EvidenceRelation.CAUSES,
                confidence=min(1.0, float(result.get("confidence", 0.5))),
                raw_score=float(result.get("probability", 0)),
                trace=f"Association: P({target}|{evidence}) = {result.get('probability', 0):.3f}",
                metadata={
                    "query_type": "association",
                    "target": target,
                    "evidence": evidence,
                    "probability": result.get("probability", 0),
                    "path": result.get("path", ""),
                },
            ))

        return items

    def build_intervention(self, target: str, interventions: Dict[str, float]) -> List[EvidenceItem]:
        """Nivel 2: intervencao (do-calculus)."""
        result = self.world_model.query_intervention(target, interventions)
        items: List[EvidenceItem] = []

        if result and result.get("causal_effect", 0) != 0:
            items.append(EvidenceItem(
                source=EvidenceSource.CAUSAL,
                relation=EvidenceRelation.CAUSES,
                confidence=min(1.0, abs(float(result.get("causal_effect", 0)))),
                raw_score=float(result.get("probability", 0)),
                trace=f"Intervention: do({interventions}) -> P({target}) = {result.get('probability', 0):.3f}",
                metadata={
                    "query_type": "intervention",
                    "target": target,
                    "interventions": interventions,
                    "causal_effect": result.get("causal_effect", 0),
                    "probability": result.get("probability", 0),
                },
            ))

        return items

    def build_counterfactual(self, target: str, actual: Dict[str, float],
                             hypothetical: Dict[str, float]) -> List[EvidenceItem]:
        """Nivel 3: contrafactual."""
        result = self.world_model.query_counterfactual(target, actual, hypothetical)
        items: List[EvidenceItem] = []

        if result and result.get("causal_attribution", 0) != 0:
            attr = float(result.get("causal_attribution", 0))
            items.append(EvidenceItem(
                source=EvidenceSource.CAUSAL,
                relation=EvidenceRelation.CAUSES,
                confidence=min(1.0, abs(attr)),
                raw_score=attr,
                trace=f"Counterfactual: if {hypothetical} then {result.get('counterfactual_outcome', '?')}",
                metadata={
                    "query_type": "counterfactual",
                    "target": target,
                    "actual": actual,
                    "hypothetical": hypothetical,
                    "actual_outcome": result.get("actual_outcome", ""),
                    "counterfactual_outcome": result.get("counterfactual_outcome", ""),
                    "causal_attribution": attr,
                },
            ))

        return items

    def build_causal_chains(self, min_length: int = 2) -> List[EvidenceItem]:
        """Extrai cadeias causais inteiras do WorldModel."""
        chains = self.world_model.get_causal_chains(min_length=min_length)
        items: List[EvidenceItem] = []

        for chain in chains:
            strength = self._chain_strength(chain)
            if strength < 0.1:
                continue
            items.append(EvidenceItem(
                source=EvidenceSource.CAUSAL,
                relation=EvidenceRelation.CAUSES,
                confidence=min(1.0, strength),
                raw_score=strength,
                trace=f"CausalChain: {' -> '.join(chain)}",
                metadata={"chain": chain, "length": len(chain), "strength": round(strength, 3)},
            ))

        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    def build_from_curriculum(self, concept: str) -> List[EvidenceItem]:
        """Busca evidencias causais no science_curriculum."""
        items: List[EvidenceItem] = []
        for topic, facts in self.science_curriculum.items():
            if concept.lower() not in topic.lower():
                continue
            if isinstance(facts, dict):
                for rel, val in facts.items():
                    items.append(EvidenceItem(
                        source=EvidenceSource.CAUSAL,
                        relation=EvidenceRelation.CAUSES if "cause" in rel.lower() else EvidenceRelation.RELATED,
                        confidence=min(1.0, float(val) if isinstance(val, (int, float)) else 0.5),
                        raw_score=float(val) if isinstance(val, (int, float)) else 0.5,
                        trace=f"Curriculum[{topic}]: {rel}",
                        metadata={"topic": topic, "relation": rel, "value": val},
                    ))
        return items

    def _chain_strength(self, chain: List[str]) -> float:
        if len(chain) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(chain) - 1):
            edge_key = (chain[i], chain[i + 1])
            edge = self.world_model.edges.get(edge_key)
            if edge:
                total += float(getattr(edge, "strength", 0.5))
                count += 1
        return total / max(1, count) * (1 - 0.1 * max(0, len(chain) - 3))
