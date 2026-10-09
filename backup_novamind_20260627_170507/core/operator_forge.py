"""
OperatorForge — Descobre Novos Operadores Cognitivos

Quando uma sequencia de operadores produz consistentemente
bons resultados, a Forja pode abstrai-la em um NOVO operador.

Exemplo:
  observe + predict + compare  →  ANTICIPATE (novo operador)
  
Novos operadores entram em QUARENTENA:
  - sao testados por N episodios
  - se melhoram previsoes → permanentes
  - se nao → descartados
"""
import time
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from .cognitive_operators import CognitiveOperator, OPERATOR_EFFECTS, OPERATOR_COST


@dataclass
class CandidateOperator:
    """Um operador candidato (em quarentena)."""
    name: str
    sequence: List[str]
    effect_profile: Dict[str, float]
    cost: float
    test_count: int = 0
    success_count: int = 0
    total_utility: float = 0.0
    created_at: float = field(default_factory=time.time)
    quarantine_period: int = 10
    is_permanent: bool = False
    multi_metric: Dict = field(default_factory=dict)  # metricas multi-dim

    @property
    def avg_utility(self) -> float:
        return self.total_utility / max(1, self.test_count)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.test_count)

    def test(self, utility: float):
        """Um teste do candidato."""
        self.test_count += 1
        self.total_utility += utility
        if utility > 0.6:
            self.success_count += 1

    def should_promote(self) -> bool:
        """Decide se deve virar operador permanente."""
        if self.test_count < self.quarantine_period:
            return False
        return self.success_rate > 0.6 and self.avg_utility > 0.5

    def should_discard(self) -> bool:
        if self.test_count < self.quarantine_period:
            return False
        return self.success_rate < 0.3 or self.avg_utility < 0.3


class OperatorForge:
    """Forja de operadores: detecta padroes e cria novos operadores.

    Funciona em 3 etapas:
      1. DETECT: encontra n-grams na sequencia e calcula utilidade MULTI-DIMENSIONAL
      2. FORGE: cria candidato com perfil de efeito combinado + hierarquia
      3. QUARENTINE: testa o candidato, promove ou descarta

    Utilidade multi-dimensional:
      reward: recompensa imediata
      prediction_gain: quanto melhora a acuracia do modelo
      future_reuse: estimativa de quantas vezes sera usado
      compression_gain: quantos operadores substitui (reducao de custo)
      energy_saved: reducao de custo computacional
      planning_gain: quantas decisoes futuras simplifica
    """

    def __init__(self, operator_system=None):
        self.operator_system = operator_system
        self.candidates: Dict[str, CandidateOperator] = {}
        self.promoted: List[str] = []
        self.discarded: List[str] = []
        self.discovery_log: List[Dict] = []

    def detect_candidates(self, sequence: List[str],
                          utility_by_step: Optional[List[float]] = None) -> List[str]:
        """Analisa sequencia de operadores com utilidade MULTI-DIMENSIONAL.

        Nao usa apenas frequencia. Calcula:
          reward: utilidade media
          compression_gain: quantos operadores a sequencia substitui
          energy_saved: reducao de custo ao compor
          future_reuse: frequencia relativa
        """
        if len(sequence) < 3:
            return []

        # Custo total dos operadores individuais
        total_individual_cost = 0.0
        for op_name in set(sequence):
            if self.operator_system:
                op = self.operator_system.operators.get(op_name)
                if op:
                    total_individual_cost += op.cost

        # Detectar n-grams (3 e 4)
        ngrams = {}
        for n in [3, 4]:
            for i in range(len(sequence) - n + 1):
                gram = tuple(sequence[i:i+n])
                if gram not in ngrams:
                    ngrams[gram] = {"count": 0, "utilities": [],
                                    "cost_individual": 0.0}
                ngrams[gram]["count"] += 1
                if utility_by_step and i + n <= len(utility_by_step):
                    avg_u = sum(utility_by_step[i:i+n]) / n
                    ngrams[gram]["utilities"].append(avg_u)
                # Custo individual da sequencia
                for op_name in gram:
                    if self.operator_system:
                        op = self.operator_system.operators.get(op_name)
                        if op:
                            ngrams[gram]["cost_individual"] += op.cost

        # Pontuar cada n-gram com utilidade MULTI-DIMENSIONAL
        promising = []
        for gram, data in ngrams.items():
            n = len(gram)

            # reward: utilidade media observada
            reward = sum(data["utilities"]) / max(1, len(data["utilities"]))

            # compression_gain: quantos operadores substitui
            compression_gain = n / max(1, n + 1)  # max 1.0 pra n-grande

            # energy_saved: reducao de custo ao compor
            composed_cost = data["cost_individual"] * 0.7
            energy_saved = max(0, data["cost_individual"] - composed_cost)

            # future_reuse: frequencia relativa
            future_reuse = data["count"] / max(1, len(sequence) // 3)

            # planning_gain: profundidade na arvore
            planning_gain = 0.2 * n  # sequencias maiores tem mais ganho

            # UTILIDADE MULTI-DIMENSIONAL
            utility = (
                reward * 0.30 +
                compression_gain * 0.20 +
                energy_saved * 0.20 +
                future_reuse * 0.15 +
                planning_gain * 0.15
            )

            if reward > 0.3 and utility > 0.3:
                promising.append({
                    "sequence": list(gram),
                    "utility": utility,
                    "reward": reward,
                    "compression_gain": compression_gain,
                    "energy_saved": energy_saved,
                    "future_reuse": future_reuse,
                    "planning_gain": planning_gain,
                    "count": data["count"],
                })

        # Ordenar por utilidade multi-dimensional
        promising.sort(key=lambda x: -x["utility"])

        new_candidates = []
        for p in promising[:8]:  # max 8 por ciclo
            name = self._generate_name(p["sequence"])
            if name not in self.candidates and name not in self.promoted:
                candidate = self._forge_candidate(name, p["sequence"])
                # Registrar metrica multi-dimensional
                candidate.multi_metric = p
                self.candidates[name] = candidate
                new_candidates.append(name)
                self.discovery_log.append({
                    "time": time.time(),
                    "name": name,
                    "sequence": p["sequence"],
                    "utility": round(p["utility"], 3),
                    "reward": round(p["reward"], 3),
                    "compression_gain": round(p["compression_gain"], 3),
                    "energy_saved": round(p["energy_saved"], 3),
                    "future_reuse": round(p["future_reuse"], 3),
                    "planning_gain": round(p["planning_gain"], 3),
                    "status": "quarantine",
                })

        return new_candidates

    def _generate_name(self, sequence: List[str]) -> str:
        """Gera um nome para o novo operador baseado na sequencia."""
        # Nomes pre-definidos para sequencias comuns
        pattern_names = {
            ("observe", "predict", "compare"): "anticipate",
            ("hypothesize", "predict", "test", "compare"): "experiment",
            ("explore", "observe", "compare"): "survey",
            ("focus", "observe", "predict"): "scrutinize",
            ("generalize", "specialize", "test"): "adapt",
            ("chunk", "compose", "generalize"): "synthesize",
            ("ignore", "wait", "observe"): "bide",
            ("reflect", "generalize", "compose"): "theorize",
            ("explore", "exploit", "compare"): "calibrate",
            ("predict", "imagine", "compare"): "simulate",
        }
        key = tuple(sequence)
        if key in pattern_names:
            return pattern_names[key]

        # Nome generico baseado nos operadores
        prefix = sequence[0][:4]
        suffix = sequence[-1][:4]
        return f"{prefix}_{suffix}_{int(time.time())}"

    def _forge_candidate(self, name: str, sequence: List[str]) -> CandidateOperator:
        """Cria um operador candidato combinando efeitos da sequencia."""
        combined_effects = {}
        total_cost = 0.0
        for op_name in sequence:
            effects = OPERATOR_EFFECTS.get(op_name, {})
            for key, val in effects.items():
                combined_effects[key] = combined_effects.get(key, 0.0) + val
            total_cost += OPERATOR_COST.get(op_name, 0.1)

        # Normalizar efeitos para nao explodir
        for key in combined_effects:
            combined_effects[key] = max(-1.0, min(1.0, combined_effects[key] * 0.7))

        cost = min(1.0, total_cost * 0.6)

        return CandidateOperator(
            name=name,
            sequence=sequence,
            effect_profile=combined_effects,
            cost=cost,
        )

    def report_utility(self, name: str, utility: float):
        """Reporta utilidade de um operador candidato."""
        candidate = self.candidates.get(name)
        if candidate is None:
            return

        candidate.test(utility)

        # Verificar se deve promover ou descartar
        if candidate.should_promote():
            self._promote(candidate)
        elif candidate.should_discard():
            self._discard(candidate)

    def _promote(self, candidate: CandidateOperator):
        """Promove candidato a operador permanente (hierarquico)."""
        if self.operator_system is None:
            return

        # Criar operador hierarquico com sub-operadores
        self.operator_system.compose_hierarchical(
            name=candidate.name,
            sequence=candidate.sequence,
            combined_effects=candidate.effect_profile,
            combined_cost=candidate.cost,
        )

        self.promoted.append(candidate.name)
        if candidate.name in self.candidates:
            del self.candidates[candidate.name]

        disc_log = self.discovery_log[-1] if self.discovery_log else {}
        disc_log["status"] = "promoted"
        disc_log["depth"] = self.operator_system.operators[candidate.name].depth
        disc_log["sub_operators"] = candidate.sequence

    def _discard(self, candidate: CandidateOperator):
        """Descarta candidato com baixo desempenho."""
        self.discarded.append(candidate.name)
        if candidate.name in self.candidates:
            del self.candidates[candidate.name]

        self.discovery_log.append({
            "time": time.time(),
            "name": candidate.name,
            "status": "discarded",
        })

    def get_state(self) -> Dict:
        return {
            "candidates": len(self.candidates),
            "promoted": len(self.promoted),
            "discarded": len(self.discarded),
            "recent_discoveries": self.discovery_log[-5:] if self.discovery_log else [],
        }
