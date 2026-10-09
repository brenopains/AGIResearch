"""
BasalGanglia — Seleção de Ação Go/NoGo Modulada por Dopamina
==============================================================
Implementa o circuito dos gânglios da base para SELEÇÃO DE AÇÃO:

  Córtex → Putamen → GPe → GPi/SNr → Tálamo → Córtex (volta)

- Via Direta (Go, D1): facilita a ação selecionada
- Via Indireta (NoGo, D2): suprime ações concorrentes
- Via Hiperdireta (STN): atrasa decisão em caso de conflito

A dopamina fásica (erro de predição de recompensa) modula a
plasticidade nas vias Go (D1) e NoGo (D2).

Dopamina alta → mais Go (exploração, risco)
Dopamina baixa → mais NoGo (conservador, inibição)

Referência: Frank (2006), "From RL models of BG to pathophysiology"
Mink (2018) "Basal ganglia mechanisms in action selection"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class BasalGanglia:
    def __init__(self, learning_rate: float = 0.1, dopamine_tonic: float = 0.5):
        self.lr = learning_rate
        self.dopamine_tonic = dopamine_tonic  # nível tônico de dopamina

        # Go weights: context → action → weight (via direta D1)
        self.go_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # NoGo weights: context → action → weight (via indireta D2)
        self.nogo_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Expected reward for RPE calculation
        self.expected_reward: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Histórico de seleções
        self.selection_history: List[Dict] = []
        self._max_history = 100
        self.stn_threshold = 0.3  # threshold de conflito para ativar STN

    def select(self, context: str, candidates: List[str],
               forced: Optional[str] = None) -> Optional[str]:
        """
        Seleciona a melhor ação baseado em Go/NoGo competition.

        Se o conflito entre ações for alto > stn_threshold,
        o núcleo subtalâmico (STN) atrasa a decisão (retorna None).

        Args:
            context: representação do estado atual
            candidates: lista de ações disponíveis
            forced: ação forçada (ignora competição)

        Returns:
            ação selecionada ou None (STN delay)
        """
        if forced:
            return forced

        if not candidates:
            return None

        scores = {}
        for action in candidates:
            go = self.go_weights[context].get(action, 0.1)
            nogo = self.nogo_weights[context].get(action, 0.0)
            scores[action] = go * (1.0 + self.dopamine_tonic) - nogo * (1.0 - self.dopamine_tonic * 0.5)

        sorted_actions = sorted(scores.items(), key=lambda x: -x[1])

        if len(sorted_actions) < 2:
            return sorted_actions[0][0]

        best_action, best_score = sorted_actions[0]
        second_score = sorted_actions[1][1]
        conflict = best_score - second_score

        if conflict < self.stn_threshold and best_score < 0.3:
            return None

        if best_score < -0.5:
            return None

        # Adiciona ruído para exploração (especialmente no início)
        if np.random.random() < max(0.05, 0.3 - self.dopamine_tonic * 0.3):
            return np.random.choice(candidates)

        self.selection_history.append({
            "context": context[:30],
            "selected": best_action,
            "score": best_score,
            "conflict": conflict,
        })
        if len(self.selection_history) > self._max_history:
            self.selection_history = self.selection_history[-self._max_history:]

        return best_action

    def update(self, context: str, action: str, reward: float,
               dopamine_phasic: Optional[float] = None):
        """
        Atualiza pesos via erro de predição de recompensa (RPE).

        RPE positivo → fortalece Go (D1), enfraquece NoGo (D2)
        RPE negativo → enfraquece Go (D1), fortalece NoGo (D2)

        Args:
            context: estado em que a ação foi tomada
            action: ação executada
            reward: recompensa recebida (0-1)
            dopamine_phasic: sinal dopaminérgico fásico (se None, calcula de reward)
        """
        expected = self.expected_reward[context].get(action, 0.3)
        rpe = reward - expected

        if dopamine_phasic is None:
            dopamine_phasic = rpe

        # Update expected reward
        self.expected_reward[context][action] = expected + self.lr * rpe

        # D1: Go pathway — fortalece com RPE positivo
        go_delta = self.lr * max(0, dopamine_phasic)
        self.go_weights[context][action] += go_delta

        # D2: NoGo pathway — fortalece com RPE negativo
        nogo_delta = self.lr * max(0, -dopamine_phasic)
        self.nogo_weights[context][action] += nogo_delta

        # Normaliza pesos para evitar crescimento infinito
        self.go_weights[context][action] = min(5.0, max(-5.0, self.go_weights[context][action]))
        self.nogo_weights[context][action] = min(5.0, max(-5.0, self.nogo_weights[context][action]))

        # Limpeza periódica para evitar vazamento de memória
        if np.random.random() < 0.01:
            self._cleanup()

    def tonic_update(self):
        """Update tonic dopamine level with slow drift toward baseline."""
        self.dopamine_tonic = np.clip(
            self.dopamine_tonic + np.random.uniform(-0.02, 0.02),
            0.0, 1.0
        )

    def get_action_value(self, context: str, action: str) -> float:
        """Retorna o valor líquido (Go - NoGo) para uma ação num contexto."""
        go = self.go_weights[context].get(action, 0.1)
        nogo = self.nogo_weights[context].get(action, 0.0)
        return go - nogo

    def set_dopamine_tonic(self, value: float):
        """Ajusta nível tônico de dopamina (exploração vs explotação)."""
        self.dopamine_tonic = np.clip(value, 0.0, 1.0)

    def _cleanup(self):
        """Remove entradas antigas para controlar memória."""
        if len(self.go_weights) > 200:
            keys = list(self.go_weights.keys())
            for k in keys[:50]:
                del self.go_weights[k]
        if len(self.nogo_weights) > 200:
            keys = list(self.nogo_weights.keys())
            for k in keys[:50]:
                del self.nogo_weights[k]

    def get_state(self) -> Dict:
        total_go = sum(len(v) for v in self.go_weights.values())
        total_nogo = sum(len(v) for v in self.nogo_weights.values())
        return {
            "dopamine_tonic": self.dopamine_tonic,
            "go_entries": total_go,
            "nogo_entries": total_nogo,
            "contexts_known": len(self.go_weights),
            "last_selection": self.selection_history[-1] if self.selection_history else None,
            "selections": len(self.selection_history),
        }

    def decide(self, action: str, context: Optional[Dict] = None) -> Dict:
        """
        Decide se uma ÚNICA ação deve ser executada (Go) ou inibida (NoGo).

        Versão simplificada para integração com MotorCortex.imitate_sound().

        Args:
            action: nome da ação candidata
            context: dict opcional com informações de contexto

        Returns:
            dict com {"go": bool, "selected": action, "score": float}
        """
        ctx_str = str(context or {})[:50]
        go = self.go_weights[ctx_str].get(action, 0.1)
        nogo = self.nogo_weights[ctx_str].get(action, 0.0)
        score = go * (1.0 + self.dopamine_tonic) - nogo * (1.0 - self.dopamine_tonic * 0.5)

        conflict_threshold = 0.1  # baixo para síntese de fala
        go_decision = score > conflict_threshold

        return {
            "go": bool(go_decision),
            "selected": action if go_decision else None,
            "score": float(score),
        }
