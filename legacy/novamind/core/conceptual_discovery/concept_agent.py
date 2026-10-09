"""
ConceptAgent — Conceito como agente cognitivo com comportamento preditivo.

Um conceito deixa de ser um rotulo com confianca e passa a ser:
  - Preditivo: dado X, preve Y via o conceito
  - Cross-modal: sabe como se manifesta em cada modalidade
  - Relacional: conhece suas relacoes com outros conceitos
  - Gerativo: produz hipoteses a partir de observacoes
  - Testavel: verifica se suas predicoes se confirmam

Isso inverte a arquitetura:
  ANTES: Observacao → Programa → Conceito (rotulo)
  AGORA: Observacao → Conceito → Hipotese → Programa
"""
from typing import List, Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time


@dataclass
class ModalSignature:
    """Como um conceito se manifesta em uma modalidade especifica.

    Cada modalidade tem uma assinatura unica do mesmo conceito.
    Exemplo: REFLECTION em geometria = flip, em audio = eco.
    """
    modality: str
    observable_features: List[str] = field(default_factory=list)
    invariant_properties: List[str] = field(default_factory=list)
    typical_transform: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Prediction:
    """Predicao que um conceito faz sobre uma situacao."""
    feature: str
    expected_value: Any
    actual_value: Any = None
    confirmed: Optional[bool] = None
    confidence: float = 0.0

    @property
    def is_correct(self) -> bool:
        if self.confirmed is None:
            return False
        return self.confirmed


class ConceptAgent:
    """Conceito como agente cognitivo.

    Uso:
        reflection = ConceptAgent("reflection", "spatial_transformation")
        reflection.add_prediction("orientation_inverted", True)
        reflection.add_modal_signature("geometry", ["flip", "rotate"])
        reflection.compatible_with["rotation"] = 0.8

        obs = {"grid": np.array(...), "type": "transformation"}
        hypothesis = reflection.generate_hypothesis(obs)
    """

    def __init__(self, name: str, invariant_type: str = "",
                 description: str = ""):
        self.name = name
        self.invariant_type = invariant_type or name
        self.description = description or f"concept: {name}"

        # Predicoes: o que este conceito diz sobre o mundo
        self.predictions: Dict[str, Any] = {}

        # Assinaturas por modalidade
        self.modal_signatures: Dict[str, ModalSignature] = {}

        # Relacoes com outros conceitos
        self.compatible_with: Dict[str, float] = {}  # conceito -> compatibilidade
        self.contradicts: Set[str] = set()
        self.specializes: List[str] = []   # conceitos mais gerais
        self.generalizes: List[str] = []   # conceitos mais especificos

        # Metricas
        self.confidence: float = 0.3
        self.n_observations: int = 0
        self.n_successful_predictions: int = 0
        self.n_failed_predictions: int = 0
        self.domains: Set[str] = set()
        self.explanatory_power: float = 0.0
        self.parsimony: float = 1.0  # simplicidade (1 = muito simples)

        # Geracao de hipoteses
        self._hypothesis_fn: Optional[Callable] = None

        self.created_at: float = time.time()
        self.last_used: float = time.time()

    # ---- Predicoes ----

    def add_prediction(self, feature: str, expected_value: Any):
        """Adiciona uma predicao que este conceito faz."""
        self.predictions[feature] = expected_value

    def test_prediction(self, feature: str, actual_value: Any) -> bool:
        """Testa se uma predicao se confirma."""
        if feature not in self.predictions:
            return False
        expected = self.predictions[feature]
        if isinstance(expected, np.ndarray):
            confirmed = np.array_equal(expected, actual_value)
        else:
            confirmed = expected == actual_value
        if confirmed:
            self.n_successful_predictions += 1
        else:
            self.n_failed_predictions += 1
        self.confidence = (self.n_successful_predictions + 1) / max(
            1, self.n_successful_predictions + self.n_failed_predictions + 1
        )
        return confirmed

    def get_prediction_accuracy(self) -> float:
        total = self.n_successful_predictions + self.n_failed_predictions
        if total == 0:
            return 0.0
        return self.n_successful_predictions / total

    # ---- Assinaturas Modais ----

    def add_modal_signature(self, modality: str, features: List[str],
                            invariants: Optional[List[str]] = None,
                            transform: Optional[str] = None):
        """Registra como este conceito aparece em uma modalidade."""
        self.modal_signatures[modality] = ModalSignature(
            modality=modality,
            observable_features=features,
            invariant_properties=invariants or [],
            typical_transform=transform,
            confidence=0.5,
        )

    def get_modality_signature(self, modality: str) -> Optional[ModalSignature]:
        return self.modal_signatures.get(modality)

    def has_modality(self, modality: str) -> bool:
        return modality in self.modal_signatures

    # ---- Relacoes ----

    def add_compatibility(self, other_concept: str, score: float):
        self.compatible_with[other_concept] = score

    def add_contradiction(self, other_concept: str):
        self.contradicts.add(other_concept)

    def add_specialization(self, parent_concept: str):
        if parent_concept not in self.specializes:
            self.specializes.append(parent_concept)

    def add_generalization(self, child_concept: str):
        if child_concept not in self.generalizes:
            self.generalizes.append(child_concept)

    # ---- Geracao de Hipoteses ----

    def set_hypothesis_fn(self, fn: Callable):
        """Define funcao que gera hipoteses a partir de observacoes."""
        self._hypothesis_fn = fn

    def generate_hypothesis(self, observation: Any,
                            domain: str = "general") -> Optional[Dict]:
        """Gera uma hipotese conceitual a partir de uma observacao.

        A hipotese diz: "esta observacao e uma instancia de {self.name}
        com estas propriedades especificas".

        Returns: dict com explicacao conceitual, ou None.
        """
        self.last_used = time.time()

        if self._hypothesis_fn is not None:
            custom_hyp = self._hypothesis_fn(observation)
            if custom_hyp and isinstance(custom_hyp, dict):
                custom_hyp.setdefault("confidence", self.confidence)
                custom_hyp.setdefault("timestamp", time.time())
                custom_hyp.setdefault("compatible_concepts", list(self.compatible_with.keys()))
                custom_hyp.setdefault("modality_signatures", {
                    m: sig.observable_features for m, sig in self.modal_signatures.items()
                })
            return custom_hyp

        # Hipoteses genericas baseadas nas predicoes do conceito
        hypothesis = {
            "concept": self.name,
            "invariant_type": self.invariant_type,
            "domain": domain,
            "predictions": dict(self.predictions),
            "confidence": self.confidence,
            "explanation": f"observation matches concept '{self.name}'",
            "modality_signatures": {
                m: sig.observable_features
                for m, sig in self.modal_signatures.items()
            },
            "compatible_concepts": list(self.compatible_with.keys()),
            "timestamp": time.time(),
        }

        # Verifica se observacao confirma predicoes
        if isinstance(observation, dict):
            confirmed = 0
            total = 0
            for key, expected in self.predictions.items():
                if key in observation:
                    total += 1
                    if self.test_prediction(key, observation[key]):
                        confirmed += 1
            if total > 0:
                hypothesis["predictions_confirmed"] = confirmed / total
                hypothesis["predictions_total"] = total
            else:
                hypothesis["predictions_confirmed"] = 0.0
                hypothesis["predictions_total"] = 0

        return hypothesis

    # ---- Observacao ----

    def observe(self, domain: str):
        """Registra uma observacao deste conceito em um dominio."""
        self.n_observations += 1
        self.domains.add(domain)
        self.last_used = time.time()
        # Confianca aumenta com observacoes
        self.confidence = min(1.0, self.confidence + 0.02)

    @staticmethod
    def create_for_pattern(pattern_type: str, domain: str,
                           registry: 'ConceptAgentRegistry' = None):
        """Cria um novo ConceptAgent para um padrao que nao existe ainda.

        Chamado quando o sistema encontra um invariante que nao corresponde
        a nenhum agente existente.
        """
        name = f"{pattern_type}_{domain}_{int(time.time())}"
        agent = ConceptAgent(
            name=name,
            invariant_type=pattern_type,
            description=f"dynamically created for {pattern_type} in {domain}",
        )
        agent.add_prediction(f"{pattern_type}_detected", True)
        agent.add_modal_signature(domain, [pattern_type],
                                  invariants=[f"{pattern_type}_preserved"])
        agent.observe(domain)
        agent.confidence = 0.3

        if registry:
            registry.register(agent)

        return agent

    # ---- Estado ----

    @property
    def is_viable(self) -> bool:
        """Um conceito e viavel se tem predicoes e observacoes."""
        return (len(self.predictions) > 0 or len(self.modal_signatures) > 0) \
            and self.n_observations > 0

    @property
    def cross_domain_count(self) -> int:
        return len(self.domains)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.invariant_type,
            "domains": list(self.domains),
            "n_observations": self.n_observations,
            "confidence": round(self.confidence, 3),
            "predictions": len(self.predictions),
            "prediction_accuracy": round(self.get_prediction_accuracy(), 3),
            "modalities": list(self.modal_signatures.keys()),
            "compatible_with": list(self.compatible_with.keys()),
            "specializes": self.specializes,
            "generalizes": self.generalizes,
        }


class ConceptAgentRegistry:
    """Registro central de ConceptAgents.

    Similar ao PrimitiveRegistry, mas para conceitos.
    """

    def __init__(self, compatibility_temperature: float = 0.5):
        self._agents: Dict[str, ConceptAgent] = {}
        self._relations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.compatibility_temperature = max(0.0, min(1.0, compatibility_temperature))

    def set_temperature(self, temperature: float):
        """Ajusta temperatura de compatibilidade entre conceitos.

        Temperatura alta: conceitos distantes podem se conectar
        Temperatura baixa: so conceitos fortemente relacionados se conectam
        """
        self.compatibility_temperature = max(0.0, min(1.0, temperature))

    def register(self, agent: ConceptAgent):
        if agent.name in self._agents:
            existing = self._agents[agent.name]
            existing.n_observations += agent.n_observations
            existing.confidence = max(existing.confidence, agent.confidence)
            existing.domains |= agent.domains
            for m, sig in agent.modal_signatures.items():
                existing.modal_signatures[m] = sig
        else:
            self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[ConceptAgent]:
        return self._agents.get(name)

    def get_all(self, min_confidence: float = 0.0) -> List[ConceptAgent]:
        """Retorna agentes, filtrando por temperatura.

        Temperatura alta: inclui mais agentes (limiar de compatibilidade menor)
        Temperatura baixa: so agentes com alta confianca
        """
        effective_min = min_confidence * (1.0 + (1.0 - self.compatibility_temperature))
        return [a for a in self._agents.values()
                if a.confidence >= effective_min]

    def get_compatible(self, concept_name: str) -> List[ConceptAgent]:
        """Retorna agentes compativeis com um conceito, ajustado por temperatura.

        Temperatura alta: inclui compatibilidades fracas
        Temperatura baixa: so compatibilidades fortes
        """
        agent = self._agents.get(concept_name)
        if not agent:
            return []

        min_compat = 0.7 - self.compatibility_temperature * 0.5
        compatible = []
        for name, score in agent.compatible_with.items():
            if score >= min_compat:
                other = self._agents.get(name)
                if other:
                    compatible.append(other)
        return compatible
        return [a for a in self._agents.values() if domain in a.domains]

    def get_by_modality(self, modality: str) -> List[ConceptAgent]:
        return [a for a in self._agents.values() if a.has_modality(modality)]

    def get_cross_domain(self, min_domains: int = 2) -> List[ConceptAgent]:
        return [a for a in self._agents.values()
                if len(a.domains) >= min_domains]

    def build_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """Matriz de compatibilidade entre todos os conceitos."""
        matrix = {}
        names = list(self._agents.keys())
        for n1 in names:
            matrix[n1] = {}
            for n2 in names:
                if n1 == n2:
                    matrix[n1][n2] = 1.0
                else:
                    a1 = self._agents[n1]
                    score = a1.compatible_with.get(n2, 0.0)
                    matrix[n1][n2] = score
        return matrix

    @property
    def count(self) -> int:
        return len(self._agents)

    def get_stats(self) -> dict:
        return {
            "total_agents": self.count,
            "cross_domain": len(self.get_cross_domain()),
            "avg_confidence": round(
                sum(a.confidence for a in self._agents.values()) / max(1, self.count), 3
            ),
            "agents": {n: a.to_dict() for n, a in self._agents.items()},
        }
