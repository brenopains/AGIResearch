"""
Concept Formation — Constroi CONCEITOS ESTRUTURADOS a partir de invariantes.

Diferenca entre Invariant e Concept:
  Invariant: "trajetoria inverte direcao apos colisao" (descricao)
  Concept: REFLECTION com condicoes, efeitos, restricoes, contraexemplos

Cada Concept tem:
  - nome: identificador unico
  - condicoes: o que precisa ser verdade no mundo para o conceito se aplicar
  - efeitos: o que muda no mundo quando aplicado
  - restricoes: o que NAO pode ser verdade
  - contraexemplos: casos onde o conceito NAO se aplica
  - dominios: onde foi observado
  - metricas: as 5 metricas objetivas para validar existencia
"""
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
import time


@dataclass
class Concept:
    """Um conceito estruturado.
    
    So existe se melhora pelo menos 2 das 5 metricas objetivas.
    """
    name: str
    description: str = ""
    
    # Estrutura do conceito
    conditions: List[str] = field(default_factory=list)     # "existe_objeto", "ha_colisao"
    effects: List[str] = field(default_factory=list)        # "inverte_direcao", "preserva_area"
    constraints: List[str] = field(default_factory=list)    # "nao_altera_cor", "nao_cria_objeto"
    counterexamples: List[str] = field(default_factory=list) # "nao_funciona_com_objeto_movel"
    
    # Metadados
    domains: Set[str] = field(default_factory=set)
    invariant_types: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    n_observations: int = 0
    
    # 5 metricas objetivas (validadas periodicamente)
    metric_predictive_error: float = 1.0   # 1 = max error
    metric_search_space: float = 1.0       # 1 = max candidates needed
    metric_program_length: float = 1.0     # 1 = max avg program length
    metric_reuse_count: int = 0           # quantas vezes foi reutilizado
    metric_compression: float = 0.0        # 1 = max compression contribution
    
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    age: float = 0.0
    
    @property
    def viability(self) -> float:
        """O conceito merece existir?
        
        Baseado nas 5 metricas. Precisa de pelo menos 2 com melhoria > 0.1.
        """
        metrics = [
            1.0 - self.metric_predictive_error,
            1.0 - self.metric_search_space,
            1.0 - self.metric_program_length,
            min(1.0, self.metric_reuse_count / 50),
            self.metric_compression,
        ]
        improvements = sum(1 for m in metrics if m > 0.1)
        return improvements / 5.0
    
    @property
    def is_viable(self) -> bool:
        """Pelo menos 2 das 5 metricas precisam mostrar melhoria."""
        metrics = [
            1.0 - self.metric_predictive_error,
            1.0 - self.metric_search_space,
            1.0 - self.metric_program_length,
            min(1.0, self.metric_reuse_count / 50),
            self.metric_compression,
        ]
        return sum(1 for m in metrics if m > 0.1) >= 2
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domains": list(self.domains),
            "conditions": self.conditions[:3],
            "effects": self.effects[:3],
            "constraints": self.constraints[:3],
            "confidence": round(self.confidence, 3),
            "viability": round(self.viability, 3),
            "metrics": {
                "predictive_error": round(self.metric_predictive_error, 3),
                "search_space": round(self.metric_search_space, 3),
                "program_length": round(self.metric_program_length, 3),
                "reuse_count": self.metric_reuse_count,
                "compression": round(self.metric_compression, 3),
            },
        }


class ConceptFormation:
    """Constroi conceitos estruturados a partir de invariantes e observacoes.
    
    Uso:
        cf = ConceptFormation()
        inv = Invariant(type="reflection", domains={"physics", "audio"})
        concept = cf.build(inv)
        print(concept.name, concept.conditions, concept.effects)
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
    
    def build(self, invariant_type: str, domains: Set[str],
              signature: dict = None) -> Concept:
        """Constroi ou reforca um conceito a partir de um invariante."""
        signature = signature or {}
        
        # Mapeia tipo de invariante para estrutura de conceito
        structure = self._get_concept_structure(invariant_type, domains, signature)
        
        if invariant_type in self.concepts:
            concept = self.concepts[invariant_type]
            concept.domains |= domains
            concept.n_observations += 1
            concept.confidence = min(1.0, concept.n_observations * 0.02)
            concept.last_used = time.time()
            return concept
        
        concept = Concept(
            name=invariant_type,
            description=structure.get("description", f"concept: {invariant_type}"),
            conditions=structure.get("conditions", []),
            effects=structure.get("effects", []),
            constraints=structure.get("constraints", []),
            counterexamples=structure.get("counterexamples", []),
            domains=set(domains),
            confidence=0.3,
            n_observations=1,
        )
        self.concepts[invariant_type] = concept
        return concept
    
    def _get_concept_structure(self, inv_type: str, domains: Set[str],
                                sig: dict) -> dict:
        """Gera estrutura do conceito baseado no tipo de invariante."""
        structures = {
            "reflection": {
                "description": "trajetoria ou estado inverte apos interacao com barreira",
                "conditions": ["existe_barreira", "existe_trajetoria", "ha_colisao"],
                "effects": ["inverte_direcao", "preserva_energia", "muda_orientacao"],
                "constraints": ["nao_altera_magnitude", "nao_altera_cor", "nao_cria_objeto"],
                "counterexamples": ["barreira_absorve_tudo", "trajetoria_continua_em_frente"],
            },
            "expansion": {
                "description": "algo aumenta de escala mantendo estrutura",
                "conditions": ["existe_estrutura_base", "ha_espaco_para_crescer"],
                "effects": ["aumenta_area", "preserva_proporcao", "duplica_padrao"],
                "constraints": ["nao_altera_topologia", "nao_perde_informacao"],
                "counterexamples": ["compressao", "delecao", "corte"],
            },
            "contraction": {
                "description": "algo diminui de escala mantendo estrutura essencial",
                "conditions": ["existe_estrutura_grande", "ha_restricao_de_espaco"],
                "effects": ["reduz_area", "preserva_conteudo_principal", "remove_redundancia"],
                "constraints": ["preserva_caracteristicas_essenciais", "nao_perde_identidade"],
                "counterexamples": ["expansao", "delecao_total"],
            },
            "causal_chain": {
                "description": "sequencia de eventos onde A causa B que causa C",
                "conditions": ["existe_evento_A", "existe_evento_B", "A_precede_B"],
                "effects": ["propaga_influencia", "preserva_ordem_temporal"],
                "constraints": ["nao_inverte_ordem", "nao_remove_elos"],
                "counterexamples": ["eventos_independentes", "causalidade_reversa"],
            },
            "symmetry": {
                "description": "estrutura preservada apos transformacao",
                "conditions": ["existe_estrutura_referencia", "existe_eixo_de_transformacao"],
                "effects": ["preserva_identidade", "mantem_relacoes"],
                "constraints": ["nao_altera_contagem", "nao_altera_conectividade"],
                "counterexamples": ["transformacao_assimetrica", "deformacao"],
            },
            "spatial_transformation": {
                "description": "objetos mudam de posicao mantendo identidade",
                "conditions": ["existem_objetos", "existe_operador_espacial"],
                "effects": ["move_objetos", "preserva_identidade", "muda_posicao_relativa"],
                "constraints": ["nao_altera_cor", "nao_altera_tamanho"],
                "counterexamples": ["delecao", "criacao"],
            },
        }
        
        result = structures.get(inv_type, {
            "description": f"conceito descoberto: {inv_type}",
            "conditions": ["observacao_existe"],
            "effects": ["algo_muda"],
            "constraints": ["algo_nao_muda"],
            "counterexamples": [],
        })
        
        # Enriquece com dados da assinatura
        if sig:
            if inv_type == "expansion":
                ratio = sig.get("object_ratio", 0)
                if ratio > 0:
                    result["effects"].append(f"multiplica_objetos_{ratio:.1f}x")
                h_ratio = sig.get("shape_h_ratio", 0)
                if h_ratio > 0:
                    result["effects"].append(f"expande_grid_{h_ratio:.1f}x")
        
        return result
    
    def get_concepts(self, min_viability: float = 0.0) -> List[Concept]:
        return [c for c in self.concepts.values() if c.viability >= min_viability]
    
    def get_viable_concepts(self) -> List[Concept]:
        """Conceitos que merecem existir (2+ metricas com melhoria)."""
        return [c for c in self.concepts.values() if c.is_viable]
    
    def get_by_domain(self, domain: str) -> List[Concept]:
        return [c for c in self.concepts.values() if domain in c.domains]
    
    def get_stats(self) -> dict:
        return {
            "total_concepts": len(self.concepts),
            "viable": len(self.get_viable_concepts()),
            "avg_confidence": round(
                sum(c.confidence for c in self.concepts.values()) / max(1, len(self.concepts)), 3
            ),
            "concepts": {n: c.to_dict() for n, c in self.concepts.items()},
        }
