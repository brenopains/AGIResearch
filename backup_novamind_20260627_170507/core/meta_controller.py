"""
MetaController — Sistema de Modos Cognitivos DINAMICOS e Meta-Policy

O Maestro nao executa pensamentos. Ele escolhe COMO pensar.

Modos NAO sao um enum fixo.
A ForjaEvolution pode criar novos modos em runtime via ModeRegistry.
O MetaController opera sobre o registro, nao sobre constantes.

Modos primordiais:
  EXPLORE, HYPOTHESIZE, PREDICT, TEST, CONSOLIDATE, EXPLOIT, REFLECT
"""
import time
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable

from .mode_registry import MODE_REGISTRY, ModeProfile


class MetaController:
    """Controla o modo cognitivo atual e a meta-policy.

    O MetaController:
      - Mantem o modo cognitivo atual (string, nao enum)
      - Decide quando trocar de modo baseado em sinais de erro
      - Cria e modifica loops dinamicamente
      - Aprende quais modos funcionam melhor em quais contextos
      - Usa ModeRegistry — novos modos podem surgir em runtime
    """

    def __init__(self, error_tracker=None, attention_controller=None,
                 cge=None, goal_manager=None):
        self.mode: str = "explore"  # comeca explorando (string, nao enum)
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
        self._loop_stack: List[List[str]] = []  # pilha de loops (strings)
        self._current_loop: Optional[List[str]] = None
        self._loop_position: int = 0
        self._loop_register: Dict[str, List[str]] = {
            "scientific": ["explore", "hypothesize", "predict", "test", "consolidate"],
            "execution": ["exploit", "predict", "exploit"],
            "impasse": ["explore", "reflect", "explore"],
            "deep_reflect": ["consolidate", "reflect", "hypothesize", "predict", "test"],
            "rapid_explore": ["explore", "exploit", "explore"],
        }

        # Hiperparametros (ajustados para mundo procedural rapido)
        self.min_mode_time: float = 1.0  # segundos minimos no modo
        self.exploration_rate: float = 0.15  # chance de experimentar modo novo
        self.q_learning_rate: float = 0.1
        self.q_discount: float = 0.9

        # ─── Deteccao de Impasse ────────────────────────────
        self.impasse_counter: int = 0
        self.impasse_threshold: int = 4  # N ciclos sem melhora = impasse
        self.last_reward: float = 0.0
        self.reward_history: List[float] = []
        self.stagnation_counter: int = 0
        self.stagnation_threshold: int = 6  # N ciclos no mesmo modo = estagnacao
        self.mode_repeat_counter: int = 0
        self.last_mode: Optional[str] = None

        # ─── Alternancia Foco/Difuso ────────────────────────
        self.search_radius: float = 0.5  # 0=foco local, 1=busca global
        self.radius_increment: float = 0.2

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

    def _get_allowed_modes(self, context_key: str) -> List[str]:
        """Modos permitidos no contexto atual."""
        all_modes = MODE_REGISTRY.get_active_names()
        modes = ["idle"]

        if "explore" in context_key or "novel" in context_key:
            modes.extend(["explore", "hypothesize"])

        if "exploit" in context_key or "certain" in context_key:
            modes.extend(["exploit", "predict"])

        if "reflect" in context_key or "error" in context_key:
            modes.extend(["reflect", "hypothesize"])

        if "uncertain" in context_key:
            modes.extend(["test", "explore"])

        # Adicionar modos descobertos pela ForjaEvolution
        discovered = MODE_REGISTRY.get_discovered_names()
        modes.extend(discovered)

        # Modos universais
        for u in ["consolidate", "predict", "hypothesize", "test"]:
            if u not in modes:
                modes.append(u)

        return [m for m in set(modes) if m in all_modes or m == "idle"]

    # ─── Selecao de Modo (Meta-Policy) ────────────────────

    def _select_mode(self, context_key: str) -> str:
        """Seleciona o proximo modo (string) via Q-learning + exploracao."""
        allowed = self._get_allowed_modes(context_key)

        if self._current_loop is not None:
            if self._loop_position < len(self._current_loop):
                mode = self._current_loop[self._loop_position]
                if mode in allowed:
                    return mode
                self._loop_position += 1
                if self._loop_position < len(self._current_loop):
                    return self._current_loop[self._loop_position]
                self._current_loop = None

        if random.random() < self.exploration_rate:
            return random.choice(allowed)

        best_mode = "idle"
        best_value = float("-inf")
        for mode in allowed:
            key = (context_key, mode)
            value = self._q_table.get(key, 0.0)
            if value > best_value:
                best_value = value
                best_mode = mode

        return best_mode

    def _update_q_value(self, context_key: str, mode: str, reward: float):
        key = (context_key, mode)
        old = self._q_table.get(key, 0.0)
        self._q_table[key] = old + self.q_learning_rate * (
            reward + self.q_discount * old - old
        )

    # ─── Loop Management ──────────────────────────────────

    def push_loop(self, loop: List[str], name: str = "custom"):
        self._loop_stack.append(loop)
        if self._current_loop is None:
            self._current_loop = loop
            self._loop_position = 0

    def create_loop(self, modes: List[str], name: str = "dynamic") -> str:
        loop = [m for m in modes if MODE_REGISTRY.get(m) is not None]
        if len(loop) >= 2:
            self._loop_register[name] = loop
        return name

    def start_loop(self, name: str) -> bool:
        if name in self._loop_register:
            self._current_loop = list(self._loop_register[name])
            self._loop_position = 0
            return True
        return False

    # ─── Deteccao de Impasse ──────────────────────────────

    def _detect_impasse(self, reward_signal: float) -> bool:
        """Detecta se o sistema esta em impasse (recompensa estagnada)."""
        self.reward_history.append(reward_signal)
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)
        if len(self.reward_history) < 4:
            return False
        recent = self.reward_history[-3:]
        improvement = recent[-1] - recent[0]
        if improvement < 0.05 and all(r < 0.4 for r in recent):
            self.impasse_counter += 1
        else:
            self.impasse_counter = max(0, self.impasse_counter - 1)
        return self.impasse_counter >= self.impasse_threshold

    def _detect_stagnation(self) -> bool:
        """Detecta se ficou no mesmo modo tempo demais."""
        mode_val = self.mode.value if self.mode else None
        if mode_val == self.last_mode:
            self.mode_repeat_counter += 1
        else:
            self.mode_repeat_counter = 0
        self.last_mode = mode_val
        return self.mode_repeat_counter >= self.stagnation_threshold

    def _expand_search(self) -> str:
        """Expande busca quando detecta estagnacao/impasse."""
        self.search_radius = min(1.0, self.search_radius + self.radius_increment)
        self.exploration_rate = min(0.8, self.exploration_rate + 0.1)

        active = MODE_REGISTRY.get_active_names()
        if self.search_radius > 0.7 and "explore" in active:
            return "explore"
        elif self.mode in ("explore", "exploit"):
            return random.choice([m for m in ["reflect", "hypothesize"] if m in active])
        else:
            return random.choice([m for m in ["explore", "predict"] if m in active])

    def _mode_is_active(self, mode_name: str) -> bool:
        prof = MODE_REGISTRY.get(mode_name)
        return prof is not None and prof.is_active

    # ─── Ciclo Principal ──────────────────────────────────

    def tick(self, reward_signal: Optional[float] = None) -> str:
        context = self._get_context_key()
        elapsed = time.time() - self.mode_start_time

        if reward_signal is not None and self.mode != "idle":
            self._update_q_value(context, self.mode, reward_signal)
            self.mode_performance.setdefault(self.mode, []).append(reward_signal)
            if reward_signal > 0.7 and self._current_loop:
                loop_key = f"learned_loop_{int(time.time())}"
                self._loop_register[loop_key] = list(self._current_loop)

        stuck = False
        if reward_signal is not None:
            stuck = self._detect_impasse(reward_signal)
            self.last_reward = reward_signal
        stagnant = self._detect_stagnation()

        if stuck or stagnant:
            self.impasse_counter = 0
            self.mode_repeat_counter = 0
            new_mode = self._expand_search()
            self.mode = new_mode
            self.mode_start_time = time.time()
            self._context_history.append({
                "time": time.time(), "context": context,
                "mode": new_mode, "trigger": "impasse" if stuck else "stagnation",
            })
            return new_mode
        elif self.search_radius > 0.5:
            self.search_radius = max(0.5, self.search_radius - 0.05)

        if elapsed < self.min_mode_time and self._mode_is_active(self.mode):
            if self._current_loop is not None:
                self._loop_position += 1
                if self._loop_position < len(self._current_loop):
                    self.mode = self._current_loop[self._loop_position]
                    self.mode_start_time = time.time()
            return self.mode

        if reward_signal is not None and self._mode_is_active(self.mode):
            self.mode_history.append({
                "mode": self.mode, "duration": elapsed,
                "context": context, "reward": reward_signal,
                "time": time.time(),
            })

        new_mode = self._select_mode(context)
        self.mode = new_mode
        self.mode_start_time = time.time()
        self.mode_duration[new_mode] = self.mode_duration.get(new_mode, 0.0) + elapsed
        self._context_history.append({
            "time": time.time(), "context": context, "mode": new_mode,
        })
        return new_mode

    def get_state(self) -> Dict:
        return {
            "mode": self.mode,
            "mode_duration": round(time.time() - self.mode_start_time, 1),
            "context": self._get_context_key(),
            "exploration_rate": self.exploration_rate,
            "search_radius": round(self.search_radius, 2),
            "impasse_counter": self.impasse_counter,
            "stagnation_counter": self.mode_repeat_counter,
            "loop_active": self._current_loop is not None,
            "loop_name": str(self._current_loop[0] if self._current_loop else None),
            "q_table_size": len(self._q_table),
            "mode_performance": {
                k: round(sum(v) / len(v), 2) if v else 0
                for k, v in self.mode_performance.items()
            },
            "registered_loops": list(self._loop_register.keys()),
        }
