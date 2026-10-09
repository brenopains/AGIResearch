"""
Operadores Cognitivos — Primitivas Fundamentais do Pensamento

15 operadores base que compoem todo pensamento.
Programas sao sequencias de operadores.
Operadores sao as primitivas — programas sao composicoes.

Os 15:
  Observe / Compare / Predict / Imagine / Generalize
  Specialize / Ignore / Wait / Focus / Explore
  Exploit / Reflect / Chunk / Compose / Test

Cada operador tem:
  - efeito no estado cognitivo (atencao, erro, simulacao)
  - custo computacional
  - utilidade esperada
  - historico de performance
  - confianca

Novos operadores podem ser criados pelo OperatorForge
quando uma sequencia consistente e bem-sucedida e detectada.
"""
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class OperatorID(Enum):
    OBSERVE = "observe"
    COMPARE = "compare"
    PREDICT = "predict"
    IMAGINE = "imagine"
    GENERALIZE = "generalize"
    SPECIALIZE = "specialize"
    IGNORE = "ignore"
    WAIT = "wait"
    FOCUS = "focus"
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    REFLECT = "reflect"
    CHUNK = "chunk"
    COMPOSE = "compose"
    TEST = "test"


@dataclass
class CognitiveOperator:
    """Um operador cognitivo — primitiva fundamental do pensamento."""
    id: str
    name: str
    description: str = ""
    effect_profile: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.1  # custo computacional 0..1
    utility: float = 0.5  # utilidade esperada 0..1
    history: List[float] = field(default_factory=list)
    confidence: float = 0.5
    created_by_forge: bool = False
    parent_sequence: Optional[List[str]] = None  # se criado por forge, quais ops compoem
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    times_used: int = 0
    retired: bool = False
    # Hierarquia: operadores que este operador chama internamente
    sub_operators: List[str] = field(default_factory=list)
    depth: int = 0  # profundidade na arvore (0=primitivo)

    @property
    def avg_utility(self) -> float:
        if not self.history:
            return self.utility
        return sum(self.history[-30:]) / max(1, len(self.history[-30:]))

    def apply(self, cognitive_state: Dict[str, float]) -> Dict[str, float]:
        """Aplica este operador ao estado cognitivo, retorna novo estado."""
        new_state = dict(cognitive_state)
        for key, delta in self.effect_profile.items():
            current = new_state.get(key, 0.5)
            new_state[key] = max(0.0, min(1.0, current + delta))
        self.times_used += 1
        self.last_used = time.time()
        return new_state

    def record_outcome(self, reward: float):
        """Registra resultado da aplicacao."""
        self.history.append(reward)
        n = len(self.history[-20:])
        avg = sum(self.history[-20:]) / max(1, n)
        self.confidence = self.confidence * 0.9 + avg * 0.1

    def total_subtree_cost(self, all_operators: Dict = None) -> float:
        """Custo total incluindo sub-operadores."""
        total = self.cost
        if all_operators:
            for sub_id in self.sub_operators:
                sub = all_operators.get(sub_id)
                if sub:
                    total += sub.cost * 0.5  # sub-operadores custam menos
        return total


# Perfis de efeito para cada operador
OPERATOR_EFFECTS = {
    "observe":   {"attention_width": +0.2, "exploration_weight": +0.1, "simulation_depth": -0.1},
    "compare":   {"error_sensitivity": +0.2, "certainty_threshold": -0.1},
    "predict":   {"simulation_depth": +0.2, "certainty_threshold": +0.1},
    "imagine":   {"simulation_depth": +0.3, "exploration_weight": +0.1, "attention_width": -0.1},
    "generalize": {"certainty_threshold": +0.2, "attention_width": +0.1},
    "specialize": {"attention_width": -0.2, "simulation_depth": +0.1},
    "ignore":    {"attention_width": -0.3, "exploration_weight": -0.1},
    "wait":      {"simulation_depth": +0.1, "exploration_weight": -0.1},
    "focus":     {"attention_width": -0.2, "simulation_depth": +0.2},
    "explore":   {"exploration_weight": +0.3, "attention_width": +0.1, "certainty_threshold": -0.2},
    "exploit":   {"exploration_weight": -0.2, "certainty_threshold": +0.2},
    "reflect":   {"error_sensitivity": +0.1, "certainty_threshold": -0.1, "simulation_depth": +0.1},
    "chunk":     {"attention_width": +0.1, "simulation_depth": -0.1},
    "compose":   {"simulation_depth": +0.1, "attention_width": -0.1},
    "test":      {"exploration_weight": -0.1, "error_sensitivity": +0.2, "certainty_threshold": -0.1},
}

OPERATOR_COST = {
    "observe": 0.05, "compare": 0.1, "predict": 0.15, "imagine": 0.3,
    "generalize": 0.2, "specialize": 0.15, "ignore": 0.02, "wait": 0.01,
    "focus": 0.05, "explore": 0.15, "exploit": 0.1, "reflect": 0.2,
    "chunk": 0.15, "compose": 0.2, "test": 0.15,
}


def create_base_operators() -> Dict[str, CognitiveOperator]:
    """Cria os 15 operadores base."""
    ops = {}
    for op_id in OperatorID:
        name = op_id.value
        ops[name] = CognitiveOperator(
            id=name, name=name.capitalize(),
            description=_generate_description(name),
            effect_profile=OPERATOR_EFFECTS.get(name, {}),
            cost=OPERATOR_COST.get(name, 0.1),
        )
    return ops


def _generate_description(name: str) -> str:
    descs = {
        "observe": "Absorver dados sensoriais sem agir",
        "compare": "Confrontar predicao com realidade observada",
        "predict": "Gerar estado futuro esperado",
        "imagine": "Simular sem agir, explorar possibilidades",
        "generalize": "Extrair padrao comum de multiplas instancias",
        "specialize": "Adaptar conceito geral a contexto especifico",
        "ignore": "Suprimir atencao de um estimulo",
        "wait": "Pausar processamento, permitir acumulo de informacao",
        "focus": "Estreitar atencao em um alvo especifico",
        "explore": "Buscar novidade e informacao desconhecida",
        "exploit": "Usar conhecimento consolidado para acao",
        "reflect": "Reavaliar estrategia e crencas a luz de novos dados",
        "chunk": "Agrupar conceitos em unidade maior",
        "compose": "Combinar conceitos em novo significado composto",
        "test": "Executar acao para refutar ou confirmar hipotese",
    }
    return descs.get(name, "")


class OperatorSystem:
    """Sistema que mantem e executa operadores cognitivos.

    O Maestro nao controla diretamente — ele arbitra entre operadores
    que competem por prioridade baseada no estado cognitivo atual.
    """

    def __init__(self):
        self.operators: Dict[str, CognitiveOperator] = create_base_operators()
        self.execution_history: List[Dict] = deque(maxlen=200)
        self.active_sequence: List[str] = []  # operadores usados na sessao atual

    def get_applicable(self, cognitive_state: Dict[str, float],
                       min_confidence: float = 0.1) -> List[Tuple[float, str]]:
        """Retorna operadores viaveis ordenados por prioridade."""
        scored = []
        for op_id, op in self.operators.items():
            if op.retired or op.confidence < min_confidence:
                continue
            # Pontuacao: utilidade media + adequacao ao estado
            utility = op.avg_utility
            cost_penalty = op.cost * 0.5
            score = utility - cost_penalty + random.uniform(-0.1, 0.1)
            scored.append((score, op_id))

        scored.sort(key=lambda x: -x[0])
        return scored

    def select(self, cognitive_state: Dict[str, float],
               strategy: str = "best") -> Optional[CognitiveOperator]:
        """Seleciona o melhor operador para o estado atual."""
        scored = self.get_applicable(cognitive_state)

        if not scored:
            return None

        if strategy == "explore":
            # As vezes escolher um operador menos usado
            least_used = min(scored[:5], key=lambda x: self.operators[x[1]].times_used)
            return self.operators[least_used[1]]

        # Melhor pontuacao
        best_id = scored[0][1]
        return self.operators.get(best_id)

    def execute(self, op_id: str, cognitive_state: Dict[str, float]) -> Dict[str, float]:
        """Executa um operador e registra."""
        op = self.operators.get(op_id)
        if op is None:
            return cognitive_state

        new_state = op.apply(cognitive_state)
        self.active_sequence.append(op_id)

        self.execution_history.append({
            "time": time.time(),
            "operator": op_id,
            "state_before": dict(cognitive_state),
            "state_after": dict(new_state),
        })

        return new_state

    def compose_hierarchical(self, name: str, sequence: List[str],
                             combined_effects: Dict[str, float],
                             combined_cost: float) -> CognitiveOperator:
        """Cria um operador hierarquico que encapsula sub-operadores."""
        max_depth = 0
        for op_name in sequence:
            sub = self.operators.get(op_name)
            if sub:
                max_depth = max(max_depth, sub.depth)

        new_op = CognitiveOperator(
            id=name, name=name.capitalize(),
            description=f"Composto de: {' -> '.join(sequence)}",
            effect_profile=combined_effects,
            cost=combined_cost * 0.7,
            confidence=0.3,
            created_by_forge=True,
            parent_sequence=sequence,
            sub_operators=sequence,
            depth=max_depth + 1,
        )
        self.operators[name] = new_op
        return new_op

    def get_sequence_pattern(self) -> Dict[str, Any]:
        """Analisa a sequencia ativa em busca de padroes uteis.

        Retorna dict com:
          - repeated_ngrams: n-grams que aparecem 2+ vezes
          - high_reward_sequences: sequencias que levaram a recompensa alta
        """
        seq = self.active_sequence
        if len(seq) < 4:
            return {"repeated_ngrams": [], "high_reward_sequences": []}

        # Detectar trigramas repetidos
        ngrams = {}
        for i in range(len(seq) - 2):
            gram = tuple(seq[i:i+3])
            ngrams[gram] = ngrams.get(gram, 0) + 1

        repeated = [gram for gram, count in ngrams.items() if count >= 2]

        # Detectar sequencias com recompensa alta no historico
        high_reward = []
        for i in range(len(self.execution_history) - 2):
            ops = [self.execution_history[j]["operator"]
                   for j in range(i, min(i+3, len(self.execution_history)))]
            high_reward.append(tuple(ops))

        return {
            "repeated_ngrams": repeated[:5],
            "high_reward_sequences": high_reward[:5],
        }

    def clear_sequence(self):
        """Limpa a sequencia ativa (inicio de novo ciclo)."""
        self.active_sequence = []

    def get_state(self) -> Dict:
        active = [op for op in self.operators.values() if not op.retired]
        forged = [op for op in self.operators.values() if op.created_by_forge]
        return {
            "total": len(self.operators),
            "active": len(active),
            "forged": len(forged),
            "retired": sum(1 for op in self.operators.values() if op.retired),
            "total_executions": len(self.execution_history),
            "active_sequence_len": len(self.active_sequence),
            "best_operator": max(active, key=lambda op: op.avg_utility).id
                             if active else None,
        }
