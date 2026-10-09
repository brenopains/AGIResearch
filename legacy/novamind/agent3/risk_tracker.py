"""Risco por acao: taxa de falha (fim de jogo imediato) observada por
action_id, com prior fraco (Beta(1,1)) para nao travar numa unica
observacao.

Desenhado depois de ler o codigo-fonte de bd01 (revisitar celula = derrota
instantanea) e bn01 (marcar celula errada = derrota instantanea): nos dois
casos, a informacao util nao e "esse estado exato e perigoso" (o estado
quase nunca se repete) — e "essa acao, neste jogo, tem historico de matar".
Por isso o rastreamento e por action_id, nao por (estado, acao).
"""
from __future__ import annotations

from collections import defaultdict


class ActionRiskTracker:
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.attempts: dict = defaultdict(int)
        self.failures: dict = defaultdict(int)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def record_attempt(self, action_id: int) -> None:
        self.attempts[action_id] += 1

    def record_failure(self, action_id: int) -> None:
        self.failures[action_id] += 1

    def failure_rate(self, action_id: int) -> float:
        a = self.failures[action_id] + self.prior_alpha
        b = (self.attempts[action_id] - self.failures[action_id]) + self.prior_beta
        return a / (a + b)
