"""
Universal Imagination Simulator — Instancia conceitos em diferentes dominios.

Dado um conceito (ex: BRANCHING), o simulador consegue "imaginar" como
ele se manifesta em:
  - arvores (botanica)
  - vasos sanguineos (medicina)
  - rios (geografia)
  - fractais (geometria)
  - arvore sintatica (computacao)

Isso testa se o conceito e genuinamente abstrato:
se ele se instancia em multiplos dominios, a abstracao e real.
Se so funciona em um dominio, e apenas um padrao local.
"""
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ImaginedInstance:
    """Instancia imaginada de um conceito em um dominio."""
    concept: str
    domain: str
    description: str
    plausibility: float = 0.0  # 0-1, quao plausivel e a instancia
    constraints_satisfied: List[str] = field(default_factory=list)
    constraints_violated: List[str] = field(default_factory=list)


# Mapeamento de conceitos -> dominios onde podem se manifestar
CONCEPT_DOMAIN_MAP = {
    "reflection": {
        "physics": "bola quicando na parede, luz refletindo em espelho",
        "audio": "eco de som em canhao, reverberacao em sala",
        "geometry": "simetria de reflexao, flip horizontal/vertical",
        "programming": "recursao que retorna ao estado anterior, tail recursion",
        "psychology": "introspeccao, auto-observacao do proprio pensamento",
    },
    "expansion": {
        "biology": "crescimento de arvore, expansao de colonia de fungos",
        "geometry": "tile de padrao, escala de figura",
        "programming": "loop que acumula, macro expansion",
        "physics": "dilatacao termica, expansao de gas",
        "sociology": "crescimento populacional, expansao urbana",
    },
    "branching": {
        "biology": "arvore com galhos, sistema vascular, neuronios",
        "geography": "rios com afluentes, delta, rede de drenagem",
        "geometry": "fractal, arvore de Pitagoras, floco de neve",
        "programming": "arvore binaria, estrutura de diretorios, git branching",
        "physics": "raios, descarga eletrica, cristalizacao dendritica",
        "engineering": "rede de distribuicao, sistema ferroviario",
    },
    "causal_chain": {
        "physics": "bola que bate em pinos que derrubam outros pinos",
        "programming": "pipeline de dados, chain of responsibility",
        "biology": "cadeia alimentar, cascata bioquimica",
        "sociology": "efeito dominho, propagacao de noticia",
    },
    "symmetry": {
        "geometry": "rotacao, reflexao, translacao",
        "biology": "simetria bilateral em animais, radial em estrelas do mar",
        "physics": "conservacao de momentum, simetria de gauge",
        "art": "padroes simetricos em mosaicos, mandalas",
    },
    "spatial_transformation": {
        "geometry": "translacao, rotacao, escala de objetos",
        "programming": "transformacao de coordenadas em graficos",
        "physics": "movimento de corpos no espaco",
        "robotics": "cinematica de bracos roboticos",
    },
}


class ImaginationSimulator:
    """Simulador universal que instancia conceitos em dominios.
    
    O mapa de conceitos e populado DINAMICAMENTE pelo InvariantDiscovery.
    O mapa estatico CONCEPT_DOMAIN_MAP e fallback para conceitos conhecidos.
    
    Uso:
        sim = ImaginationSimulator()
        sim.learn_domain("branching", "geometry")  # descoberto pelo CFE
        instances = sim.imagine("branching")
    """
    
    def __init__(self, cge=None):
        self.cge = cge
        # Mapa DINAMICO: conceito -> set de dominios
        self._dynamic_map: Dict[str, set] = {}
        # Mapa estatico como fallback
        self._static_map = CONCEPT_DOMAIN_MAP
    
    def learn_domain(self, concept_name: str, domain: str):
        """Aprende que um conceito se manifesta em um dominio.
        
        Chamado pelo CFE quando um invariante e detectado em novo dominio.
        """
        if concept_name not in self._dynamic_map:
            self._dynamic_map[concept_name] = set()
        self._dynamic_map[concept_name].add(domain)
    
    def learn_domains_from_invariants(self, invariants: list):
        """Aprende dominios a partir de invariantes descobertos.
        
        Cada invariante tem .structure_type e .domains.
        """
        for inv in invariants:
            for dom in inv.domains:
                self.learn_domain(inv.structure_type, dom)
    
    def get_domains(self, concept_name: str) -> set:
        """Retorna dominios onde um conceito se manifesta.
        
        Combina mapa dinamico (descoberto) com estatico (fallback).
        """
        dynamic = self._dynamic_map.get(concept_name, set())
        static_keys = set(self._static_map.get(concept_name, {}).keys())
        return dynamic | static_keys
    
    def imagine(self, concept_name: str, target_domain: Optional[str] = None
                ) -> List[ImaginedInstance]:
        """Instancia um conceito em um ou todos os dominios possiveis."""
        # Usa mapa dinamico primeiro, depois estatico como fallback
        domains = self.get_domains(concept_name)
        
        if not domains:
            # Tenta encontrar conceito no CGE
            if self.cge is not None:
                try:
                    results = self.cge.query_attention(concept_name, top_k=5)
                    if results:
                        for nid, _, _ in results[:3]:
                            self._register_domain(nid, "unknown")
                except Exception:
                    pass
            # Se ainda assim não encontrou, tenta mapa estatico
            if concept_name in self._static_map:
                domains = set(self._static_map[concept_name].keys())
            else:
                return []
        
        # Constroi descricoes: prioriza mapa estatico, depois dinamico
        domain_map = {}
        static_data = self._static_map.get(concept_name, {})
        for d in domains:
            if d in static_data:
                domain_map[d] = static_data[d]
            else:
                domain_map[d] = f"{concept_name} manifests in {d}"
        
        if target_domain and target_domain not in domain_map:
            return []
        instances = []
        
        for domain, description in domain_map.items():
            if target_domain and domain != target_domain:
                continue
            
            plausibility = self._compute_plausibility(concept_name, domain)
            constraints = self._get_constraints(concept_name)
            
            inst = ImaginedInstance(
                concept=concept_name,
                domain=domain,
                description=description,
                plausibility=plausibility,
                constraints_satisfied=constraints.get("satisfied", []),
                constraints_violated=constraints.get("violated", []),
            )
            instances.append(inst)
        
        return instances
    
    def imagine_cross_modal(self, concept_name: str) -> List[ImaginedInstance]:
        """Instancia um conceito em TODOS os dominios possiveis (teste de abstracao)."""
        return self.imagine(concept_name)
    
    def get_domains_for_concept(self, concept_name: str) -> List[str]:
        """Retorna dominios onde um conceito se manifesta."""
        return list(self.get_domains(concept_name))
    
    def get_transfer_score(self, concept_name: str) -> float:
        """Quao transferivel e este conceito (0-1).
        
        So conta dominios DINAMICAMENTE descobertos (nao estaticos).
        """
        dynamic_domains = self._dynamic_map.get(concept_name, set())
        if not dynamic_domains:
            return 0.0
        return min(1.0, len(dynamic_domains) / 5.0)
    
    def _compute_plausibility(self, concept: str, domain: str) -> float:
        """Computa quao plausivel e um conceito em um dominio."""
        # Baseado em: quantas invariantes do conceito se aplicam ao dominio
        base = 0.7  # partida alta (se esta no mapa, e viavel)
        
        # Bonus para dominios com mais instancias conhecidas
        if domain in ("physics", "geometry", "programming"):
            base += 0.15
        if domain in ("biology",):
            base += 0.1
        
        return min(1.0, base)
    
    def _get_constraints(self, concept: str) -> dict:
        """Retorna constraints do conceito."""
        constraints = {
            "reflection": {
                "satisfied": ["preserva_energia", "inverte_direcao"],
                "violated": ["cria_objeto_novo", "altera_magnitude"],
            },
            "expansion": {
                "satisfied": ["aumenta_escala", "preserva_estrutura"],
                "violated": ["reduz_area", "perde_informacao"],
            },
            "branching": {
                "satisfied": ["ramifica", "preserva_conectividade"],
                "violated": ["unifica", "linear"],
            },
        }
        return constraints.get(concept, {"satisfied": [], "violated": []})
    
    def _register_domain(self, concept_name: str, domain: str):
        """Registra um novo dominio para um conceito no mapa DINAMICO."""
        self.learn_domain(concept_name, domain)
    
    def get_stats(self) -> dict:
        # Combinacao de conceitos estaticos + dinamicos
        all_concepts = set(self._static_map.keys()) | set(self._dynamic_map.keys())
        total = len(all_concepts)
        dynamic_concepts = len(self._dynamic_map)
        avg_transfer = sum(self.get_transfer_score(c) for c in all_concepts) / max(1, total)
        return {
            "total_concepts": total,
            "dynamic_concepts": dynamic_concepts,
            "static_concepts": total - dynamic_concepts,
            "avg_transfer_score": round(avg_transfer, 3),
            "high_transfer": sum(1 for c in all_concepts if self.get_transfer_score(c) > 0.6),
            "concepts": list(all_concepts),
        }
