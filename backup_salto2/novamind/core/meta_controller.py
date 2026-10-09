"""
MetaController — Sistema de Modos Cognitivos e Meta-Policy

O Maestro nao executa pensamentos. Ele escolhe COMO pensar.

Modos cognitivos:
  EXPLORE: busca novidade, expande conhecimento
  HYPOTHESIZE: gera candidatos de explicacao
  PREDICT: simula consequencias de cada hipotese
  TEST: executa acao para refutar hipotese
  CONSOLIDATE: comprime, abstrai, esquece irrelevante
  EXPLOIT: usa conhecimento consolidado
  REFLECT: reavalia estrategia atual

O meta-controller aprende (via reforco) quais modos funcionam
melhor em quais contextos — policy gradient sobre modos cognitivos.
"""
import time
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum


class CognitiveMode(Enum):
    EXPLORE = "explore"
    HYPOTHESIZE = "hypothesize"
    PREDICT = "predict"
    TEST = "test"
    CONSOLIDATE = "consolidate"
    EXPLOIT = "exploit"
    REFLECT = "reflect"
    IDLE = "idle"

    @property
    def is_active(self) -> bool:
        return self != CognitiveMode.IDLE

    @property
    def requires_external_input(self) -> bool:
        return self in (CognitiveMode.EXPLORE, CognitiveMode.TEST)

    @property
    def generates_knowledge(self) -> bool:
        return self in (CognitiveMode.HYPOTHESIZE, CognitiveMode.CONSOLIDATE,
                        CognitiveMode.REFLECT)


# Transicoes naturais entre modos (sequencias que fazem sentido)
CANONICAL_LOOPS = {
    "scientific": [
        CognitiveMode.EXPLORE,
        CognitiveMode.HYPOTHESIZE,
        CognitiveMode.PREDICT,
        CognitiveMode.TEST,
        CognitiveMode.CONSOLIDATE,
    ],
    "execution": [
        CognitiveMode.EXPLOIT,
        CognitiveMode.PREDICT,
        CognitiveMode.EXPLOIT,
    ],
    "impasse": [
        CognitiveMode.EXPLORE,
        CognitiveMode.REFLECT,
        CognitiveMode.EXPLORE,
    ],
    "deep_reflect": [
        CognitiveMode.CONSOLIDATE,
        CognitiveMode.REFLECT,
        CognitiveMode.HYPOTHESIZE,
        CognitiveMode.PREDICT,
        CognitiveMode.TEST,
    ],
    "rapid_explore": [
        CognitiveMode.EXPLORE,
        CognitiveMode.EXPLOIT,
        CognitiveMode.EXPLORE,
    ],
}


class MetaController:
    """Controla o modo cognitivo atual e a meta-policy.

    O MetaController:
      - Mantem o modo cognitivo atual
      - Decide quando trocar de modo baseado em sinais de erro
      - Cria e modifica loops dinamicamente
      - Aprende quais modos funcionam melhor em quais contextos
    """

    def __init__(self, error_tracker=None, attention_controller=None,
                 cge=None, goal_manager=None):
        self.mode: CognitiveMode = CognitiveMode.EXPLORE  # comeca explorando
        self.mode_history: List[Dict] = []
        self.mode_duration: Dict[str, float] = {}
        self.mode_start_time: float = time.time()
        self.mode_performance: Dict[str, List[float]] = {}

        # Dependencias opcionais
        self.error_tracker = error_tracker
        self.attention = attention_controller
        self.cge = cge
        self.goal_manager = goal_manager

        # Meta-policy: Q-table para (contexto, modo) -> valor
        self._q_table: Dict[Tuple[str, str], float] = {}
        self._context_history: List[Dict] = []
        self._loop_stack: List[List[CognitiveMode]] = []  # pilha de loops
        self._current_loop: Optional[List[CognitiveMode]] = None
        self._loop_position: int = 0
        self._loop_register: Dict[str, List[CognitiveMode]] = dict(CANONICAL_LOOPS)

        # Hiperparametros (ajustados para mundo procedural rapido)
        self.min_mode_time: float = 1.0  # segundos minimos no modo
        self.exploration_rate: float = 0.15  # chance de experimentar modo novo
        self.q_learning_rate: float = 0.1
        self.q_discount: float = 0.9

    # ─── Contexto Atual ───────────────────────────────────

    def _get_context_key(self) -> str:
        """Gera chave de contexto para a Q-table."""
        et = self.error_tracker
        if et is None:
            return "default"

        state = et.get_state()
        parts = []
        if state["drives_exploration"]:
            parts.append("explore")
        if state["drives_exploitation"]:
            parts.append("exploit")
        if state["drives_reflection"]:
            parts.append("reflect")
        if state["avg_uncertainty"] > 0.6:
            parts.append("uncertain")
        elif state["avg_uncertainty"] < 0.3:
            parts.append("certain")
        if state["avg_novelty"] > 0.5:
            parts.append("novel")
        if state["avg_prediction_error"] > 0.5:
            parts.append("error")

        return "+".join(parts) if parts else "default"

    def _get_allowed_modes(self, context_key: str) -> List[CognitiveMode]:
        """Modos permitidos no contexto atual."""
        # Sempre pode ficar parado
        modes = [CognitiveMode.IDLE]

        if "explore" in context_key or "novel" in context_key:
            modes.append(CognitiveMode.EXPLORE)
            modes.append(CognitiveMode.HYPOTHESIZE)

        if "exploit" in context_key or "certain" in context_key:
            modes.append(CognitiveMode.EXPLOIT)
            modes.append(CognitiveMode.PREDICT)

        if "reflect" in context_key or "error" in context_key:
            modes.append(CognitiveMode.REFLECT)
            modes.append(CognitiveMode.HYPOTHESIZE)

        if "uncertain" in context_key:
            modes.append(CognitiveMode.TEST)
            modes.append(CognitiveMode.EXPLORE)

        # Modos universais
        modes.append(CognitiveMode.CONSOLIDATE)
        modes.append(CognitiveMode.PREDICT)
        modes.append(CognitiveMode.HYPOTHESIZE)
        modes.append(CognitiveMode.TEST)

        return list(set(modes))

    # ─── Selecao de Modo (Meta-Policy) ────────────────────

    def _select_mode(self, context_key: str) -> CognitiveMode:
        """Seleciona o proximo modo via Q-learning + exploracao."""
        allowed = self._get_allowed_modes(context_key)

        # Se tem um loop ativo, seguir o loop
        if self._current_loop is not None:
            if self._loop_position < len(self._current_loop):
                mode = self._current_loop[self._loop_position]
                if mode in allowed:
                    return mode
                # Se o modo nao e permitido, avancar no loop
                self._loop_position += 1
                if self._loop_position < len(self._current_loop):
                    return self._current_loop[self._loop_position]
                # Loop terminou
                self._current_loop = None

        # Q-learning: escolher melhor modo conhecido
        if random.random() < self.exploration_rate:
            return random.choice(allowed)

        best_mode = CognitiveMode.IDLE
        best_value = float("-inf")
        for mode in allowed:
            key = (context_key, mode.value)
            value = self._q_table.get(key, 0.0)
            if value > best_value:
                best_value = value
                best_mode = mode

        return best_mode

    def _update_q_value(self, context_key: str, mode: CognitiveMode,
                        reward: float):
        """Atualiza Q-table com o resultado do modo."""
        key = (context_key, mode.value)
        old = self._q_table.get(key, 0.0)
        self._q_table[key] = old + self.q_learning_rate * (
            reward + self.q_discount * old - old
        )

    # ─── Loop Management ──────────────────────────────────

    def push_loop(self, loop: List[CognitiveMode], name: str = "custom"):
        """Empilha um loop personalizado."""
        self._loop_stack.append(loop)
        if self._current_loop is None:
            self._current_loop = loop
            self._loop_position = 0

    def create_loop(self, modes: List[str], name: str = "dynamic") -> str:
        """Cria um novo loop registrado."""
        loop = []
        for m in modes:
            try:
                loop.append(CognitiveMode(m))
            except ValueError:
                pass
        if len(loop) >= 2:
            self._loop_register[name] = loop
        return name

    def start_loop(self, name: str) -> bool:
        """Inicia um loop registrado."""
        if name in self._loop_register:
            self._current_loop = list(self._loop_register[name])
            self._loop_position = 0
            return True
        return False

    # ─── Ciclo Principal ──────────────────────────────────

    def tick(self, reward_signal: Optional[float] = None) -> CognitiveMode:
        """Um ciclo do meta-controller.

        Args:
            reward_signal: sinal de recompensa do ciclo anterior (opcional)

        Returns:
            Modo cognitivo escolhido para este ciclo
        """
        context = self._get_context_key()
        elapsed = time.time() - self.mode_start_time

        # Atualizar Q-table com recompensa do modo anterior
        if reward_signal is not None and self.mode != CognitiveMode.IDLE:
            self._update_q_value(context, self.mode, reward_signal)
            self.mode_performance.setdefault(self.mode.value, []).append(reward_signal)
            # Atualizar loop register se tem performance alta
            if reward_signal > 0.7 and self._current_loop:
                loop_key = f"learned_loop_{int(time.time())}"
                self._loop_register[loop_key] = list(self._current_loop)

        # So troca se passou tempo minimo no modo atual
        if elapsed < self.min_mode_time and self.mode.is_active:
            # Avancar no loop se completou um sub-ciclo
            if self._current_loop is not None:
                self._loop_position += 1
                if self._loop_position < len(self._current_loop):
                    self.mode = self._current_loop[self._loop_position]
                    self.mode_start_time = time.time()
            return self.mode

        # Registrar performance do modo anterior
        if reward_signal is not None and self.mode.is_active:
            self.mode_history.append({
                "mode": self.mode.value,
                "duration": elapsed,
                "context": context,
                "reward": reward_signal,
                "time": time.time(),
            })

        # Selecionar novo modo
        new_mode = self._select_mode(context)
        self.mode = new_mode
        self.mode_start_time = time.time()
        self.mode_duration[new_mode.value] = self.mode_duration.get(
            new_mode.value, 0.0) + elapsed

        # Log
        self._context_history.append({
            "time": time.time(),
            "context": context,
            "mode": new_mode.value,
        })

        return new_mode

    def get_state(self) -> Dict:
        """Estado completo para debugging."""
        return {
            "mode": self.mode.value,
            "mode_duration": round(time.time() - self.mode_start_time, 1),
            "context": self._get_context_key(),
            "exploration_rate": self.exploration_rate,
            "loop_active": self._current_loop is not None,
            "loop_name": str(self._current_loop[0].value if self._current_loop else None),
            "q_table_size": len(self._q_table),
            "mode_performance": {
                k: round(sum(v) / len(v), 2) if v else 0
                for k, v in self.mode_performance.items()
            },
            "registered_loops": list(self._loop_register.keys()),
        }
