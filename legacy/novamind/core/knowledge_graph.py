"""
KnowledgeGraph — Grafo de conhecimento causal com metadados.

Wrappa o CGE com:
  - Ingestão de conceitos textuais (via MultilingualEncoder)
  - Relações causais (X causa Y, X inibe Y, X ativa Y)
  - Metadados por domínio
  - Consultas por domínio, relação, confiança
  - Estatísticas de cobertura

Uso:
    kg = KnowledgeGraph(encoder, cge)
    kg.add_concept("Alzheimer", ["neurodegeneração", "perda de memória"], domain="medicine")
    kg.add_causal_relation("proteína_tau", "emaranhados_neurofibrilares", "causa")
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time


@dataclass
class KnowledgeConcept:
    """Um conceito no grafo de conhecimento."""
    name: str
    domain: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.5
    source: str = ""
    hdv_vector: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class CausalRelation:
    """Uma relação causal entre dois conceitos."""
    source: str
    target: str
    relation_type: str  # "causa", "inibe", "ativa", "previne", "trata", "interage"
    confidence: float = 0.5
    evidence: str = ""
    domain: str = ""
    source_text: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence[:60] if self.evidence else "",
            "domain": self.domain,
        }


class KnowledgeGraph:
    """Grafo de conhecimento que integra com CGE.

    Mantém:
      - Conceitos com vetores HDV
      - Relações causais com metadados
      - Índices por domínio e tipo de relação
    """

    def __init__(self, encoder=None, cge=None):
        self.encoder = encoder
        self.cge = cge

        # Índices locais (além do CGE)
        self.concepts: Dict[str, KnowledgeConcept] = {}
        self.relations: List[CausalRelation] = []
        self._by_domain: Dict[str, List[str]] = defaultdict(list)
        self._by_relation_type: Dict[str, List[CausalRelation]] = defaultdict(list)

        # Estatísticas
        self.total_ingested: int = 0
        self.total_relations: int = 0

    def add_concept(self, name: str, aliases: List[str] = None,
                    domain: str = "general", description: str = "",
                    confidence: float = 0.5, source: str = "",
                    hdv_vector: np.ndarray = None) -> KnowledgeConcept:
        """Adiciona um conceito ao grafo.

        Se encoder e CGE estiverem disponíveis, também cria nó no CGE.
        """
        aliases = aliases or []

        concept = KnowledgeConcept(
            name=name,
            domain=domain,
            aliases=aliases,
            description=description,
            confidence=confidence,
            source=source,
            hdv_vector=hdv_vector,
        )

        # Se já existe, atualiza
        if name in self.concepts:
            existing = self.concepts[name]
            existing.aliases = list(set(existing.aliases + aliases))
            existing.confidence = max(existing.confidence, confidence)
            existing.last_accessed = time.time()
            existing.access_count += 1
            return existing

        self.concepts[name] = concept
        self._by_domain[domain].append(name)
        self.total_ingested += 1

        # Cria vetor HDV e nó no CGE
        if self.encoder and hdv_vector is None:
            try:
                hdv = self.encoder.encode_bipolar(f"{name}: {description}")
                concept.hdv_vector = hdv.cpu().numpy()

                if self.cge:
                    self.cge.create_node(
                        node_id=name,
                        semantic_vec=hdv,
                        description=description,
                        domain=domain,
                    )
            except Exception as e:
                pass

        return concept

    def add_causal_relation(self, source: str, target: str,
                            relation_type: str,
                            confidence: float = 0.5,
                            evidence: str = "",
                            domain: str = "",
                            source_text: str = "") -> Optional[CausalRelation]:
        """Adiciona uma relação causal entre dois conceitos.

        Se o conceito alvo nao existir, cria auto-magicamente.
        """
        # Auto-cria conceitos se necessario
        if source not in self.concepts:
            self.add_concept(source, domain=domain or "general",
                             description=f"discovered from: {evidence[:60]}",
                             confidence=confidence * 0.8)
        if target not in self.concepts:
            self.add_concept(target, domain=domain or "general",
                             description=f"discovered from: {evidence[:60]}",
                             confidence=confidence * 0.8)

        rel = CausalRelation(
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=confidence,
            evidence=evidence,
            domain=domain or self.concepts[source].domain,
            source_text=source_text,
        )
        self.relations.append(rel)
        self._by_relation_type[relation_type].append(rel)
        self.total_relations += 1

        # Adiciona aresta causal no CGE
        if self.cge:
            try:
                self.cge.add_causal_edge(
                    from_id=source,
                    to_id=target,
                    relation_type=relation_type,
                    weight=confidence,
                )
            except Exception:
                pass

        return rel

    def get_concept(self, name: str) -> Optional[KnowledgeConcept]:
        """Busca um conceito pelo nome."""
        concept = self.concepts.get(name)
        if concept:
            concept.last_accessed = time.time()
            concept.access_count += 1
        return concept

    def search_concepts(self, query: str, top_k: int = 10) -> List[KnowledgeConcept]:
        """Busca conceitos similares textualmente."""
        query_lower = query.lower()
        results = []
        for name, concept in self.concepts.items():
            if query_lower in name.lower():
                results.append(concept)
            elif any(query_lower in a.lower() for a in concept.aliases):
                results.append(concept)
            elif query_lower in concept.description.lower():
                results.append(concept)

        results.sort(key=lambda c: c.confidence, reverse=True)
        return results[:top_k]

    def search_relations(self, source: str = None, target: str = None,
                         relation_type: str = None,
                         domain: str = None,
                         min_confidence: float = 0.0) -> List[CausalRelation]:
        """Busca relações causais com filtros."""
        results = []
        for rel in self.relations:
            if source and rel.source != source:
                continue
            if target and rel.target != target:
                continue
            if relation_type and rel.relation_type != relation_type:
                continue
            if domain and rel.domain != domain:
                continue
            if rel.confidence < min_confidence:
                continue
            results.append(rel)
        return results

    def get_concepts_by_domain(self, domain: str) -> List[KnowledgeConcept]:
        """Retorna todos os conceitos de um domínio."""
        names = self._by_domain.get(domain, [])
        return [self.concepts[n] for n in names if n in self.concepts]

    def get_domains(self) -> List[str]:
        """Lista todos os domínios registrados."""
        return sorted(self._by_domain.keys())

    def get_stats(self) -> dict:
        return {
            "total_concepts": self.total_ingested,
            "total_relations": self.total_relations,
            "domains": len(self._by_domain),
            "relation_types": dict(
                (t, len(rs)) for t, rs in self._by_relation_type.items()
            ),
            "concepts_per_domain": {
                d: len(concepts) for d, concepts in self._by_domain.items()
            },
            "cge_online": self.cge is not None,
        }
