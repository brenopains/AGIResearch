"""
WorldLaws — Nivel acima de conceitos.

Principios universais que explicam MULTIPLOS conceitos em MULTIPLOS dominios.

Diferenca entre Concept e WorldLaw:
  Concept: "branching" (estrutura se divide)
  WorldLaw: "propagation" (tudo que se propaga: branching, epidemia, fogo, rios)

Cada WorldLaw:
  - Explains: quais conceitos sao casos particulares
  - Predicts: propriedades esperadas em novos dominios
  - Conflicts: que outras leis contradiz
  - Compression: quantos conceitos substitui
"""
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class WorldLaw:
    """Principio universal que explica multiplos conceitos."""
    name: str
    description: str = ""

    # Conceitos que esta lei explica
    explains: Set[str] = field(default_factory=set)

    # Dominios onde foi observada
    domains: Set[str] = field(default_factory=set)

    # Predicoes: o que esperar em novos dominios
    predictions: Dict[str, float] = field(default_factory=dict)

    # Metricas
    compression: float = 0.0       # 1 - (1/n_conceitos_explicados)
    coverage: float = 0.0          # % de conceitos conhecidos que explica
    parsimony: float = 1.0         # simplicidade
    confidence: float = 0.3
    n_observations: int = 0
    falsified: bool = False
    falsification_evidence: List[str] = field(default_factory=list)

    # Relacoes com outras leis
    conflicts_with: Set[str] = field(default_factory=set)
    specializes: List[str] = field(default_factory=list)   # leis mais gerais
    generalizes: List[str] = field(default_factory=list)   # leis mais especificas

    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def economy(self) -> float:
        """Cobertura / complexidade."""
        if self.parsimony == 0:
            return 0.0
        return self.coverage * self.parsimony

    def explain(self, concept_name: str) -> bool:
        """Esta lei explica este conceito?"""
        return concept_name in self.explains

    def predict(self, domain: str) -> Dict[str, float]:
        """Faz predicoes para um novo dominio."""
        return dict(self.predictions)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "explains": list(self.explains)[:5],
            "domains": list(self.domains),
            "compression": round(self.compression, 3),
            "coverage": round(self.coverage, 3),
            "economy": round(self.economy, 3),
            "confidence": round(self.confidence, 3),
            "n_observations": self.n_observations,
            "falsified": self.falsified,
            "conflicts_with": list(self.conflicts_with),
        }


# Leis fundamentais pre-definidas (crescem com experiencia)
SEED_LAWS = {
    "propagation": WorldLaw(
        name="propagation",
        description="Tudo que se propaga: branching, cascade, diffusion, spread",
        explains={"branching", "diffusion", "cascade", "spread", "flow"},
        domains={"biology", "geography", "physics", "epidemiology", "networks"},
        predictions={"increases_over_space": 0.8, "needs_medium": 0.6, "has_source": 0.9},
        compression=0.8, coverage=0.15, parsimony=0.5,
    ),
    "conservation": WorldLaw(
        name="conservation",
        description="Algo se preserva apos transformacao: symmetry, reflection, equilibrium",
        explains={"symmetry", "reflection", "equilibrium", "invariance", "reversibility"},
        domains={"physics", "geometry", "chemistry", "biology"},
        predictions={"quantity_preserved": 0.9, "structure_preserved": 0.7},
        compression=0.8, coverage=0.15, parsimony=0.5,
    ),
    "competition": WorldLaw(
        name="competition",
        description="Selecao por recursos limitados: selection, optimization, evolution",
        explains={"selection", "optimization", "evolution", "adaptation", "survival"},
        domains={"biology", "economics", "machine_learning", "sociology"},
        predictions={"fittest_survives": 0.7, "resources_limited": 0.9},
        compression=0.8, coverage=0.15, parsimony=0.5,
    ),
    "feedback": WorldLaw(
        name="feedback",
        description="Saida influencia entrada: causal_chain, oscillation, regulation",
        explains={"causal_chain", "oscillation", "regulation", "homeostasis", "reinforcement"},
        domains={"cybernetics", "biology", "engineering", "psychology"},
        predictions={"output_affects_input": 0.9, "can_oscillate": 0.6},
        compression=0.8, coverage=0.15, parsimony=0.5,
    ),
    "scale": WorldLaw(
        name="scale",
        description="Mudanca de escala mantendo estrutura: expansion, contraction, fractal",
        explains={"expansion", "contraction", "fractal", "scaling", "zoom"},
        domains={"geometry", "physics", "biology"},
        predictions={"proportions_preserved": 0.7, "structure_recursive": 0.6},
        compression=0.75, coverage=0.12, parsimony=0.5,
    ),
    "hierarchy": WorldLaw(
        name="hierarchy",
        description="Organizacao em niveis: branching, containment, taxonomy",
        explains={"branching", "containment", "taxonomy", "nesting", "layers"},
        domains={"biology", "organizations", "computer_science", "linguistics"},
        predictions={"levels_exist": 0.8, "each_level_contains": 0.7},
        compression=0.75, coverage=0.12, parsimony=0.5,
    ),
}


class WorldLawRegistry:
    """Registro central de WorldLaws.

    Cresce com a experiencia do sistema.
    Novas leis sao descobertas quando conceitos se unificam.
    """

    def __init__(self):
        self.laws: Dict[str, WorldLaw] = {}
        self._concept_to_law: Dict[str, List[str]] = defaultdict(list)
        self._law_co_occurrence: Dict[tuple, int] = defaultdict(int)

        # Semente inicial
        for name, law in SEED_LAWS.items():
            self.register(law)

    def register(self, law: WorldLaw):
        """Registra ou atualiza uma lei."""
        if law.name in self.laws:
            existing = self.laws[law.name]
            existing.domains |= law.domains
            existing.explains |= law.explains
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.n_observations += law.n_observations
            existing.last_updated = time.time()
            return existing

        self.laws[law.name] = law
        for concept in law.explains:
            self._concept_to_law[concept].append(law.name)
        return law

    def get_law(self, name: str) -> Optional[WorldLaw]:
        return self.laws.get(name)

    def get_all(self, min_confidence: float = 0.0) -> List[WorldLaw]:
        return [l for l in self.laws.values() if l.confidence >= min_confidence]

    def get_laws_for_concept(self, concept: str) -> List[WorldLaw]:
        """Retorna leis que explicam um conceito."""
        law_names = self._concept_to_law.get(concept, [])
        return [self.laws[n] for n in law_names if n in self.laws]

    def get_laws_for_domain(self, domain: str) -> List[WorldLaw]:
        """Retorna leis ativas em um dominio."""
        return [l for l in self.laws.values() if domain in l.domains]

    def discover_law(self, concept1: str, concept2: str,
                     shared_domains: Set[str],
                     constraint_network=None,
                     min_domains: int = 2,
                     max_domain_fields: int = 2) -> Optional[WorldLaw]:
        """Tenta descobrir uma nova lei que unifica dois conceitos.

        Args:
            constraint_network: se fornecido, valida a lei antes de registrar
            min_domains: dominios compartilhados minimos para criar lei
            max_domain_fields: maximo de campos semanticos permitidos

        Returns: nova lei, ou None se rejeitada.
        """
        laws_for_c1 = set(self._concept_to_law.get(concept1, []))
        laws_for_c2 = set(self._concept_to_law.get(concept2, []))

        # Se ja existe lei que explica ambos → nao precisa
        shared_laws = laws_for_c1 & laws_for_c2
        if shared_laws:
            return None

        # Threshold: precisa compartilhar pelo menos min_domains dominios
        if len(shared_domains) < min_domains:
            return None

        # Candidata a nova lei
        name = f"{concept1}_{concept2}_law"
        if name in self.laws:
            return None

        new_law = WorldLaw(
            name=name,
            description=f"unifies {concept1} and {concept2}",
            explains={concept1, concept2},
            domains=set(shared_domains),
            compression=0.5,
            coverage=0.05,
            parsimony=0.5,
            confidence=0.2,
            n_observations=1,
        )

        # Valida contra constraint network antes de registrar
        if constraint_network is not None:
            verdicts = constraint_network.validate(
                name, concept_type="world_law",
                domains=set(shared_domains),
            )
            approved = sum(1 for v in verdicts if v.approved)
            required = max(1, int(len(verdicts) * constraint_network.strictness))
            if approved < required:
                return None  # Lei rejeitada

        self.register(new_law)
        return new_law

    def check_consistency(self, domain: str, observation: dict) -> List[dict]:
        """Verifica se todas as leis ativas em um dominio sao consistentes.

        Returns: lista de conflitos encontrados.
        """
        conflicts = []
        domain_laws = self.get_laws_for_domain(domain)
        for i, l1 in enumerate(domain_laws):
            for l2 in domain_laws[i + 1:]:
                if l2.name in l1.conflicts_with:
                    conflicts.append({
                        "law_a": l1.name,
                        "law_b": l2.name,
                        "domain": domain,
                        "confidence_a": l1.confidence,
                        "confidence_b": l2.confidence,
                    })
        return conflicts

    def get_stats(self) -> dict:
        return {
            "total_laws": len(self.laws),
            "seed_laws": len(SEED_LAWS),
            "discovered_laws": len(self.laws) - len(SEED_LAWS),
            "total_explanations": sum(len(l.explains) for l in self.laws.values()),
            "avg_compression": round(
                sum(l.compression for l in self.laws.values()) / max(1, len(self.laws)), 3
            ),
            "laws": {n: l.to_dict() for n, l in self.laws.items()},
        }
