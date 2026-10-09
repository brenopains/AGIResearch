"""
Theory Builder v2 — Economia de Teorias.

Cada teoria unifica multiplos conceitos sob uma explicacao comum.
Metricas:
  - Cobertura explicativa: quantos fenomenos/observacoes a teoria explica
  - Parsimonia: quantos conceitos por teoria (Occam)
  - Falsificabilidade: quantas predicoes testaveis a teoria gera
  - Compressao: reducao no numero de conceitos necessarios para explicar tudo

Uma teoria e preferivel quando:
  coverage / complexity > threshold
"""
from typing import List, Dict, Set, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


@dataclass
class Theory:
    """Uma teoria que unifica conceitos sob uma explicacao comum."""
    name: str
    description: str = ""

    # Conceitos que a teoria unifica
    concepts: Set[str] = field(default_factory=set)

    # Dominios onde a teoria se aplica
    domains: Set[str] = field(default_factory=set)

    # Metricas da economia de teorias
    coverage: float = 0.0        # quantos fenomenos explica
    parsimony: float = 1.0       # simplicidade (1/size)
    falsifiability: float = 0.0  # predicoes testaveis
    compression: float = 0.0     # reducao explicativa

    # Predicoes da teoria (o que ela preve sobre novas observacoes)
    predictions: Dict[str, float] = field(default_factory=dict)

    # Metadados
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_tested: Optional[float] = None
    test_count: int = 0
    success_count: int = 0

    @property
    def economy(self) -> float:
        """Economia da teoria: cobertura / complexidade."""
        if self.parsimony == 0:
            return 0.0
        return self.coverage * self.parsimony

    @property
    def simplicity(self) -> float:
        return 1.0 / max(1, len(self.concepts))

    def apply(self, observation: dict) -> Optional[Dict]:
        """Aplica a teoria a uma observacao.

        Se a observacao se encaixa na teoria, retorna explicacao.
        """
        obs_concepts = set(observation.get("concepts", []))
        match = obs_concepts & self.concepts
        if len(match) >= 2:
            return {
                "theory": self.name,
                "matching_concepts": list(match),
                "confidence": len(match) / max(1, len(self.concepts)),
                "predictions": self.predictions,
            }
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "concepts": list(self.concepts)[:5],
            "domains": list(self.domains),
            "coverage": round(self.coverage, 3),
            "parsimony": round(self.parsimony, 3),
            "falsifiability": round(self.falsifiability, 3),
            "compression": round(self.compression, 3),
            "economy": round(self.economy, 3),
            "predictions": len(self.predictions),
            "confidence": round(self.confidence, 3),
            "test_accuracy": round(
                self.success_count / max(1, self.test_count), 3
            ) if self.test_count > 0 else 0.0,
        }


class TheoryBuilder:
    """Constroi e avalia teorias usando economia de teorias.

    Uso:
        tb = TheoryBuilder()
        tb.observe_concept("reflection", {"physics", "audio", "geometry"})
        tb.observe_concept("rotation", {"geometry", "physics"})
        theories = tb.build_theories()
        best = tb.select_best_theory(theories)
    """

    def __init__(self, economy_threshold: float = 0.3):
        self.theories: Dict[str, Theory] = {}
        self._concept_domains: Dict[str, Set[str]] = {}
        self._concept_pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        self._concept_observations: Dict[str, int] = defaultdict(int)
        self.economy_threshold = economy_threshold

    def observe_concept(self, concept_name: str, domains: Set[str],
                        n_observations: int = 1):
        """Observa um conceito e atualiza potenciais teorias."""
        if concept_name in self._concept_domains:
            self._concept_domains[concept_name] |= domains
        else:
            self._concept_domains[concept_name] = set(domains)
        self._concept_observations[concept_name] += n_observations

        # Atualiza co-ocorrencia
        for existing in self._concept_domains:
            if existing != concept_name:
                pair = tuple(sorted([existing, concept_name]))
                self._concept_pairs[pair] += 1

    def build_theories(self) -> List[Theory]:
        """Constroi/atualiza teorias. Retorna novas teorias."""
        theory_candidates = defaultdict(set)
        for (c1, c2), count in self._concept_pairs.items():
            if count >= 3:
                shared = self._concept_domains.get(c1, set()) & \
                         self._concept_domains.get(c2, set())
                if shared:
                    theory_name = self._name_theory(c1, c2)
                    theory_candidates[theory_name].add(c1)
                    theory_candidates[theory_name].add(c2)

        new_theories = []
        for name, concepts in theory_candidates.items():
            if name not in self.theories:
                domains = set()
                total_obs = 0
                for c in concepts:
                    domains |= self._concept_domains.get(c, set())
                    total_obs += self._concept_observations.get(c, 0)

                n_concepts = len(concepts)
                theory = Theory(
                    name=name,
                    description=f"unifies: {', '.join(list(concepts)[:4])}",
                    concepts=concepts,
                    domains=domains,
                    coverage=total_obs / max(1, total_obs + 10),
                    parsimony=1.0 / max(1, n_concepts),
                    falsifiability=n_concepts / max(1, n_concepts + 3),
                    compression=1.0 - (1.0 / max(1, n_concepts)),
                    confidence=min(1.0, n_concepts * 0.15),
                    predictions=self._generate_predictions(concepts),
                )

                # So adiciona se a economia for boa
                if theory.economy >= self.economy_threshold:
                    self.theories[name] = theory
                    new_theories.append(theory)

        return new_theories

    def _generate_predictions(self, concepts: Set[str]) -> Dict[str, float]:
        """Gera predicoes da teoria baseadas nos conceitos unificados."""
        preds = {}
        for c in concepts:
            domains = self._concept_domains.get(c, set())
            for d in domains:
                key = f"{c}_appears_in_{d}"
                preds[key] = 0.7
        return preds

    def select_best_theory(self, theories: Optional[List[Theory]] = None) -> Optional[Theory]:
        """Seleciona a melhor teoria pela economia."""
        candidates = theories or list(self.theories.values())
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.economy)

    def get_theories_by_domain(self, domain: str) -> List[Theory]:
        return [t for t in self.theories.values() if domain in t.domains]

    def get_theories_by_concept(self, concept: str) -> List[Theory]:
        return [t for t in self.theories.values() if concept in t.concepts]

    def record_test(self, theory_name: str, success: bool):
        if theory_name in self.theories:
            t = self.theories[theory_name]
            t.test_count += 1
            if success:
                t.success_count += 1
            t.last_tested = time.time()
            t.confidence = t.success_count / max(1, t.test_count)

    def get_stats(self) -> dict:
        return {
            "total_theories": len(self.theories),
            "total_concepts": len(self._concept_domains),
            "avg_economy": round(
                sum(t.economy for t in self.theories.values()) / max(1, len(self.theories)), 3
            ) if self.theories else 0,
            "best_theory": self.select_best_theory().to_dict() if self.theories else None,
            "theories": {n: t.to_dict() for n, t in self.theories.items()},
        }

    def _name_theory(self, c1: str, c2: str) -> str:
        common_prefix = ""
        for i in range(min(len(c1), len(c2))):
            if c1[i] == c2[i]:
                common_prefix += c1[i]
            else:
                break
        if len(common_prefix) >= 3:
            return f"{common_prefix}_theory"
        parts = []
        for c in [c1, c2]:
            words = c.replace("_", " ").split()
            if words:
                parts.append(words[0].capitalize())
        if parts:
            return f"{''.join(parts)}Theory"
        return f"theory_{c1}_{c2}"
