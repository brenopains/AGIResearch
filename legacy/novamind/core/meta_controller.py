"""
MetaController — Estado Cognitivo Continuo (Nao Controlador Discreto)

Antes: Q-learning selecionava modo entre 7 opcoes (EXPLORE, CONSOLIDATE...)
Agora: O NeuroField produz parametros continuos. O MetaController APENAS
traduz em rotulos legiveis para humanos.

O controle REAL esta:
  - NeuroField: gating_temperature, plasticity, consolidation_pressure, etc.
  - Gating Field (CognitiveDAG): quais portas abertas baseado na temperatura
  - CGE: paisagem de energia com atratores e repulsao (anti-loop)

MetaController nao decide. Ele descreve.
"""
import time, random
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


class MetaController:
    """Estado cognitivo continuo.

    O modo (label) emerge dos parametros do NeuroField.
    Nao ha Q-learning. Nao ha selecao de modo.
    """

    def __init__(self, error_tracker=None, attention_controller=None,
                 cge=None, goal_manager=None, neurofield=None):
        self.mode: str = "observe"
        self.mode_history: List[Dict] = []
        self.mode_start_time: float = time.time()
        self.neurofield = neurofield

        self.error_tracker = error_tracker
        self.attention = attention_controller
        self.cge = cge
        self.goal_manager = goal_manager

        self._context_history: List[Dict] = []
        self._loop_register: Dict[str, List[str]] = {
            "scientific": ["explore", "hypothesize", "predict", "test", "consolidate"],
            "execution": ["exploit", "predict", "exploit"],
        }

        # Impasse detection (ultimo resquicio de seguranca)
        self.impasse_counter: int = 0
        self.last_reward: float = 0.0
        self.reward_history: List[float] = []

    def _mode_label_from_neurofield(self) -> str:
        """Rotulo legivel do estado atual. EMERGE dos parametros continuos.

        Todos os thresholds sao DESCRITIVOS (nao prescritivos).
        O NeuroField produz os valores; este metodo apenas nomeia.
        """
        if self.neurofield is None:
            return self.mode

        ed = self.neurofield.exploration_drive
        fi = self.neurofield.focus_intensity
        pl = self.neurofield.plasticity
        cp = self.neurofield.consolidation_pressure
        cr = self.neurofield.creativity

        if cp > 0.7 and pl < 0.4:
            return "consolidate"
        elif ed > 0.7 and fi < 0.4:
            return "explore"
        elif cr > 0.6 and ed > 0.4:
            return "reflect"
        elif ed > 0.4 and pl > 0.6:
            return "hypothesize"
        elif fi > 0.7 and pl < 0.3:
            return "exploit"
        elif fi > 0.5:
            return "predict"
        else:
            return "observe"

    def _detect_impasse(self, reward_signal: float) -> bool:
        """Detecta estagnacao (rede de seguranca anti-loop)."""
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
        return self.impasse_counter >= 4

    def tick(self, reward_signal: Optional[float] = None) -> str:
        """Ciclo principal.

        NAO usa Q-learning. NAO usa selecao de modo discreta.
        O modo emerge do NeuroField. Se impasse, injeta novidade.
        """
        elapsed = time.time() - self.mode_start_time

        stuck = False
        if reward_signal is not None:
            stuck = self._detect_impasse(reward_signal)
            self.last_reward = reward_signal

        if stuck and self.neurofield:
            self.neurofield.update(novelty=0.8, prediction_error=0.6)
            new_mode = "explore"
            self.mode = new_mode
            self.mode_start_time = time.time()
            self._context_history.append({
                "time": time.time(), "mode": new_mode, "trigger": "impasse"
            })
            return new_mode

        if self.neurofield:
            new_mode = self._mode_label_from_neurofield()
        else:
            new_mode = self.mode
            if elapsed > 5:
                fallback = ["observe", "hypothesize", "consolidate", "reflect"]
                idx = int(time.time()) % len(fallback)
                new_mode = fallback[idx]

        if new_mode != self.mode:
            self.mode = new_mode
            self.mode_start_time = time.time()
            self._context_history.append({
                "time": time.time(), "mode": new_mode, "trigger": "neurofield"
            })

        return self.mode

    def get_state(self) -> Dict:
        nf = self.neurofield.get_state() if self.neurofield else {}
        return {
            "mode": self.mode,
            "mode_duration": round(time.time() - self.mode_start_time, 1),
            "registered_loops": list(self._loop_register.keys()),
            "neurofield": nf,
        }
