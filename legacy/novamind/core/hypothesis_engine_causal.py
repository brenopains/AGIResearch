"""
HypothesisEngineCausal — Gera hipóteses sobre relações causais a partir do KnowledgeGraph.

Dado um grafo de conhecimento com relações causais (X causa Y, X inibe Y),
este motor:

  1. Detecta gaps: "Y tem causa desconhecida"
  2. Gera hipóteses: "talvez X cause Y" baseado em similaridade semântica
  3. Testa hipóteses contra conhecimento existente
  4. Atribui confiança baseada em evidências
  5. Sugere experimentos para validar

Uso:
    hc = HypothesisEngineCausal(kg, encoder)
    hypotheses = hc.generate_hypotheses("Alzheimer")
    for h in hypotheses:
        print(f"{h.cause} → {h.effect} (conf={h.confidence})")
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time


@dataclass
class CausalHypothesis:
    """Hipótese sobre uma relação causal."""
    cause: str
    effect: str
    relation_type: str  # "causa", "inibe", "ativa", "previne"
    confidence: float
    evidence: List[str] = field(default_factory=list)
    source_hdv_similarity: float = 0.0
    contradicting_evidence: List[str] = field(default_factory=list)
    suggested_experiment: str = ""
    is_novel: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "type": self.relation_type,
            "confidence": round(self.confidence, 3),
            "evidence_count": len(self.evidence),
            "novel": self.is_novel,
            "experiment": self.suggested_experiment[:80] if self.suggested_experiment else "",
        }


class HypothesisEngineCausal:
    """Gera e valida hipóteses causais a partir do grafo de conhecimento.

    Uso:
        eng = HypothesisEngineCausal(kg, encoder)
        hyps = eng.generate_hypotheses(target="Alzheimer")
        for h in hyps:
            print(f"{h.cause} → {h.effect}")
    """

    def __init__(self, knowledge_graph=None, encoder=None):
        self.kg = knowledge_graph
        self.encoder = encoder
        self.hypotheses: List[CausalHypothesis] = []
        self._known_relations: Set[Tuple[str, str]] = set()

        if self.kg:
            for rel in self.kg.relations:
                self._known_relations.add((rel.source, rel.target))

    def generate_hypotheses(self, target_concept: str,
                            max_hypotheses: int = 10,
                            min_confidence: float = 0.3) -> List[CausalHypothesis]:
        """Gera hipóteses sobre causas de um conceito alvo."""
        candidates = []
        if not self.kg:
            return candidates

        seen_signatures = set()

        def _is_duplicate(cause: str, effect: str) -> bool:
            sig = f"{cause}::{effect}"
            if sig in seen_signatures:
                return True
            seen_signatures.add(sig)
            return False

        # Pre-computa indices para busca O(1)
        relations_by_target = {}
        relations_by_source = {}
        for rel in self.kg.relations:
            relations_by_target.setdefault(rel.target, []).append(rel)
            relations_by_source.setdefault(rel.source, []).append(rel)

        targets_set = set(relations_by_target.keys())

        def _matches(concept: str, target: str) -> bool:
            """Match flexivel: substring ou correspondencia exata."""
            if not concept or not target:
                return False
            concept_lower = concept.lower()
            target_lower = target.lower()
            return (concept_lower in target_lower or
                    target_lower in concept_lower or
                    concept_lower == target_lower)

        # 1. Busca cadeia causal (transitividade)
        direct_relations = relations_by_target.get(target_concept, [])

        # Tambem busca por substring nos targets
        for target_in_kg, rels_list in relations_by_target.items():
            if _matches(target_concept, target_in_kg):
                direct_relations.extend(rels_list)
        # Remove duplicatas e ordena
        seen_rels = set()
        unique_relations = []
        for rel in direct_relations:
            key = (rel.source, rel.target)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relations.append(rel)

        for rel in unique_relations:
            if not _is_duplicate(rel.source, rel.target):
                candidates.append(CausalHypothesis(
                    cause=rel.source,
                    effect=rel.target,
                    relation_type=rel.relation_type,
                    confidence=rel.confidence,
                    evidence=[rel.evidence],
                    is_novel=False,
                ))

            # Cadeia: se X → Y e Y → target, entao X → target (inferencia)
            if rel.source in targets_set:
                for chain_rel in relations_by_target.get(rel.source, []):
                    if not _is_duplicate(chain_rel.source, rel.source):
                        candidates.append(CausalHypothesis(
                            cause=chain_rel.source,
                            effect=rel.source,
                            relation_type=chain_rel.relation_type,
                            confidence=chain_rel.confidence * 0.7,
                            evidence=[f"chain: {chain_rel.source} → {chain_rel.source}",
                                      f"and: {chain_rel.source} → {rel.source}"],
                            is_novel=True,
                        ))

        # 2. Busca por similaridade textual (limitada para performance)
        # (desativada sem encoder HDV — ativar quando encoder estiver disponivel)
        # if self.encoder and self.kg:
        #     ...

        # Ordena por confianca e retorna
        candidates.sort(key=lambda h: -h.confidence)
        self.hypotheses.extend(candidates)
        return candidates[:max_hypotheses]

    def get_hypotheses_by_domain(self, domain: str) -> List[CausalHypothesis]:
        """Retorna hipóteses de um domínio específico."""
        if not self.kg:
            return []
        return [h for h in self.hypotheses
                if (self.kg.get_concept(h.cause) and
                    self.kg.get_concept(h.cause).domain == domain) or
                (self.kg.get_concept(h.effect) and
                 self.kg.get_concept(h.effect).domain == domain)]

    def get_stats(self) -> dict:
        # Conta hipoteses unicas
        unique = set()
        for h in self.hypotheses:
            unique.add((h.cause, h.effect))
        return {
            "total_hypotheses": len(self.hypotheses),
            "unique_hypotheses": len(unique),
            "novel_hypotheses": sum(1 for h in self.hypotheses if h.is_novel),
            "direct_causes": sum(1 for h in self.hypotheses if not h.is_novel),
            "chain_inferences": sum(1 for h in self.hypotheses if h.is_novel and h.relation_type not in ("associado", "interage")),
            "hdv_similarity": sum(1 for h in self.hypotheses if h.source_hdv_similarity > 0),
            "avg_confidence": round(
                sum(h.confidence for h in self.hypotheses) / max(1, len(self.hypotheses)), 3
            ) if self.hypotheses else 0,
        }
