"""
MentalState — Dynamic Blackboard (Workspace Global).

Nao e uma classe fixa com 200 atributos.
E um espaco de nomes onde modulos registram/seu estado dinamicamente.

Uso:
    ms = get_mental_state()
    ms.register("world_model", {"branches": [], "error": 0.5})
    ms.register("attention", {"focus": "gato", "salience": 0.8})

    wm_state = ms.read("world_model")  # returns dict or None
    ms.write("world_model", {"branches": [], "error": 0.3})

    # Conveniencia para atributos comuns:
    ms.goal = "entender gatos"
    ms.da = 0.7
    ms.prediction_error = 0.1

Modulos nunca conversam diretamente entre si.
Todos leem/escrevem apenas no MentalState.
"""
import time
from typing import Dict, Any, Optional, List


class MentalState:
    """Blackboard cognitivo. Modulos registram estado dinamicamente."""

    def __init__(self):
        # ─── Blackboard dinamico ──────────────────────────
        self._blackboard: Dict[str, Any] = {}

        # ─── Atributos canonicos (conveniencia) ───────────
        self._input_text: str = ""
        self._focus: str = ""
        self._goal: str = ""
        self._da: float = 0.5
        self._na: float = 0.5
        self._ach: float = 0.5
        self._ht5: float = 0.5
        self._prediction_error: float = 0.0
        self._confidence: float = 0.5
        self._creativity: float = 0.5
        self._tick: int = 0
        self._total_cycles: int = 0
        self._last_cycle_duration: float = 0.0
        self._cycle_iterations: int = 0  # quantas iteracoes no loop interno
        self._timestamp: float = time.time()

        # ─── Modulos registrados ──────────────────────────
        self._modules: Dict[str, Any] = {}

    # ─── Blackboard API ────────────────────────────────────

    def register(self, module: str, state: dict):
        """Registra estado de um modulo no blackboard."""
        self._blackboard[module] = state

    def read(self, module: str) -> Optional[dict]:
        """Le estado de um modulo no blackboard."""
        return self._blackboard.get(module)

    def write(self, module: str, state: dict):
        """Atualiza estado de um modulo no blackboard."""
        self._blackboard[module] = state

    def unregister(self, module: str):
        """Remove modulo do blackboard."""
        self._blackboard.pop(module, None)

    def has(self, module: str) -> bool:
        return module in self._blackboard

    @property
    def blackboard(self) -> Dict[str, dict]:
        return dict(self._blackboard)

    # ─── Atributos canonicos ───────────────────────────────

    @property
    def current_input(self) -> str: return self._input_text
    @current_input.setter
    def current_input(self, v: str): self._input_text = v

    @property
    def focus(self) -> str: return self._focus
    @focus.setter
    def focus(self, v: str): self._focus = v

    @property
    def current_goal(self) -> str: return self._goal
    @current_goal.setter
    def current_goal(self, v: str): self._goal = v

    @property
    def da(self) -> float: return self._da
    @da.setter
    def da(self, v: float): self._da = max(0.0, min(1.0, v))

    @property
    def na(self) -> float: return self._na
    @na.setter
    def na(self, v: float): self._na = max(0.0, min(1.0, v))

    @property
    def ach(self) -> float: return self._ach
    @ach.setter
    def ach(self, v: float): self._ach = max(0.0, min(1.0, v))

    @property
    def ht5(self) -> float: return self._ht5
    @ht5.setter
    def ht5(self, v: float): self._ht5 = max(0.0, min(1.0, v))

    @property
    def nt_context(self) -> dict:
        return {"DA": self._da, "NA": self._na, "ACh": self._ach, "5HT": self._ht5}

    @nt_context.setter
    def nt_context(self, d: dict):
        self._da = d.get("DA", self._da)
        self._na = d.get("NA", self._na)
        self._ach = d.get("ACh", self._ach)
        self._ht5 = d.get("5HT", self._ht5)

    @property
    def prediction_error(self) -> float: return self._prediction_error
    @prediction_error.setter
    def prediction_error(self, v: float): self._prediction_error = v

    @property
    def confidence(self) -> float: return self._confidence
    @confidence.setter
    def confidence(self, v: float): self._confidence = max(0.0, min(1.0, v))

    @property
    def creativity(self) -> float: return self._creativity
    @creativity.setter
    def creativity(self, v: float): self._creativity = max(0.0, min(1.0, v))

    @property
    def tick(self) -> int: return self._tick
    @tick.setter
    def tick(self, v: int): self._tick = v

    @property
    def total_cycles(self) -> int: return self._total_cycles
    @total_cycles.setter
    def total_cycles(self, v: int): self._total_cycles = v

    @property
    def last_cycle_duration(self) -> float: return self._last_cycle_duration
    @last_cycle_duration.setter
    def last_cycle_duration(self, v: float): self._last_cycle_duration = v

    @property
    def cycle_iterations(self) -> int: return self._cycle_iterations
    @cycle_iterations.setter
    def cycle_iterations(self, v: int): self._cycle_iterations = v

    @property
    def timestamp(self) -> float: return self._timestamp
    @timestamp.setter
    def timestamp(self, v: float): self._timestamp = v

    # ─── Estado completo para logs ─────────────────────────

    def to_dict(self) -> dict:
        d = {
            "focus": self.focus,
            "goal": self.current_goal,
            "da": round(self.da, 3),
            "na": round(self.na, 3),
            "ach": round(self.ach, 3),
            "5ht": round(self.ht5, 3),
            "prediction_error": round(self.prediction_error, 4),
            "confidence": round(self.confidence, 3),
            "cycle": self.total_cycles,
            "tick": self.tick,
            "inner_iterations": self.cycle_iterations,
            "duration": round(self.last_cycle_duration, 3),
            "current_input": self.current_input[:80] if self.current_input else "",
        }
        # Inclui blackboard (apenas informativo)
        d["blackboard_modules"] = list(self._blackboard.keys())
        for mod, state in self._blackboard.items():
            if isinstance(state, dict):
                d[mod] = {k: str(v)[:60] for k, v in list(state.items())[:5]}
        return d


# ─── Singleton ──────────────────────────────────────────────
_INSTANCE: Optional[MentalState] = None


def get_mental_state() -> MentalState:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MentalState()
    return _INSTANCE


def reset_mental_state():
    global _INSTANCE
    _INSTANCE = MentalState()
