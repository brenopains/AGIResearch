from typing import List, Dict, Tuple
from .contracts import Hypothesis, EvidenceItem, EvidenceRelation


class ConsistencyChecker:
    """Detecta conflitos logicos entre hipoteses e evidencias.

    Ex: "whale is mammal" AND "whale is fish" -> INCONSISTENTE
    """

    def __init__(self):
        self.conflicts: List[Dict] = []

    def check_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Dict]:
        """Varre todas as hipoteses em busca de contradicoes.

        Returns lista de conflitos encontrados.
        """
        self.conflicts = []
        if len(hypotheses) < 2:
            return self.conflicts

        # Conflitos diretos: mesma entidade, categorias mutuamente exclusivas
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i + 1:]:
                conflict = self._check_pair(h1, h2)
                if conflict:
                    self.conflicts.append(conflict)

        # Conflitos internos: evidencias a favor vs contra na mesma hipotese
        for h in hypotheses:
            internal = self._check_internal(h)
            if internal:
                self.conflicts.append(internal)

        return self.conflicts

    def _check_pair(self, h1: Hypothesis, h2: Hypothesis) -> Dict:
        """Checa se h1 e h2 sao logicamente incompativeis."""
        # Mesmo nome com categorias diferentes
        if h1.name.lower() == h2.name.lower() and h1.category != h2.category:
            return {
                "type": "DIRECT_CONTRADICTION",
                "entity": h1.name,
                "statements": [f"{h1.name} is {h1.category}", f"{h2.name} is {h2.category}"],
                "severity": "high",
                "resolution": f"resolve '{h1.name}' category: evidence supports {self._pick_winner(h1, h2)}",
            }

        # Mesma categoria para entidades mutualmente exclusivas
        cats = [h1.category, h2.category]
        if h1.name.lower() != h2.name.lower() and len(set(cats)) == 1 and cats[0] in (
            "is_a", "has_property"
        ):
            isa1 = [e for e in h1.evidence_for if e.relation == EvidenceRelation.IS_A]
            isa2 = [e for e in h2.evidence_for if e.relation == EvidenceRelation.IS_A]
            for e1 in isa1:
                for e2 in isa2:
                    if self._are_exclusive(e1.trace, e2.trace):
                        return {
                            "type": "EXCLUSIVE_CATEGORIES",
                            "entity_a": h1.name,
                            "entity_b": h2.name,
                            "conflict": f"both claim '{cats[0]}' but types are exclusive",
                            "severity": "medium",
                            "resolution": "check taxonomy depth or refine categories",
                        }

        return {}

    def _check_internal(self, h: Hypothesis) -> Dict:
        """Checa se hipotese tem evidencias contradizendo sua categoria."""
        for_isa = [e for e in h.evidence_for if e.relation == EvidenceRelation.IS_A]
        against_isa = [e for e in h.evidence_against if e.relation == EvidenceRelation.IS_A]

        if for_isa and against_isa:
            strongest_for = max(for_isa, key=lambda e: e.confidence)
            strongest_against = max(against_isa, key=lambda e: e.confidence)
            if strongest_against.confidence > 0.3:
                return {
                    "type": "INTERNAL_CONTRADICTION",
                    "entity": h.name,
                    "detail": f"{strongest_for.trace} (conf={strongest_for.confidence:.2f}) vs "
                              f"{strongest_against.trace} (conf={strongest_against.confidence:.2f})",
                    "severity": "medium",
                    "resolution": "weigh evidence by source reliability",
                }
        return {}

    def check_evidence_consistency(self, evidence_list: List[EvidenceItem]) -> List[Dict]:
        """Checa consistencia entre si de uma lista de evidencias."""
        issues: List[Dict] = []
        for i, e1 in enumerate(evidence_list):
            for e2 in evidence_list[i + 1:]:
                if e1.relation == EvidenceRelation.IS_A and e2.relation == EvidenceRelation.IS_A:
                    if self._are_exclusive(e1.trace, e2.trace):
                        issues.append({
                            "type": "EVIDENCE_CONFLICT",
                            "trace_a": e1.trace[:60],
                            "trace_b": e2.trace[:60],
                            "severity": "low",
                        })
        return issues

    def _are_exclusive(self, trace_a: str, trace_b: str) -> bool:
        """Heuristica: duas categorias finais sao mutuamente exclusivas?"""
        a = trace_a.lower()
        b = trace_b.lower()
        exclusive_pairs = [
            ("mammal", "fish"), ("mammal", "bird"), ("mammal", "reptile"),
            ("bird", "fish"), ("reptile", "fish"), ("reptile", "bird"),
            ("animal", "mineral"), ("living", "nonliving"),
            ("solid", "liquid"), ("liquid", "gas"),
        ]
        for x, y in exclusive_pairs:
            if (x in a and y in b) or (x in b and y in a):
                return True
        return False

    def _pick_winner(self, h1: Hypothesis, h2: Hypothesis) -> str:
        return h1.name if h1.evidence_total > h2.evidence_total else h2.name
