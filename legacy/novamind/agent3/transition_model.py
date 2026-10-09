"""Modelo de transicao count-based: (assinatura_de_estado, acao) -> (assinatura_seguinte, contagem).

Nao e rede neural. A metrica que importa aqui e eficiencia amostral, entao
o modelo e uma tabela — mais barato, mais interpretavel, e o suficiente
para orientar exploracao por ganho de informacao (ver action_selection.py).
"""
from __future__ import annotations

from collections import defaultdict


class TransitionModel:
    def __init__(self):
        # (state_sig, action_id) -> contagem de vezes tentado
        self.visit_counts: dict = defaultdict(int)
        # (state_sig, action_id) -> {next_state_sig: contagem}
        self.outcomes: dict = defaultdict(lambda: defaultdict(int))

    def record(self, state_sig: tuple, action_id: int, next_state_sig: tuple) -> None:
        key = (state_sig, action_id)
        self.visit_counts[key] += 1
        self.outcomes[key][next_state_sig] += 1

    def visits(self, state_sig: tuple, action_id: int) -> int:
        return self.visit_counts.get((state_sig, action_id), 0)

    def outcome_entropy(self, state_sig: tuple, action_id: int) -> float:
        """Entropia das transicoes observadas para (estado, acao). Alta
        entropia = acao tem efeito imprevisivel/variado nesse estado =
        ainda ha informacao a ganhar observando de novo."""
        dist = self.outcomes.get((state_sig, action_id))
        if not dist:
            return 1.0  # nunca visto: incerteza maxima
        total = sum(dist.values())
        if total <= 1:
            return 1.0
        import math
        h = 0.0
        for count in dist.values():
            p = count / total
            h -= p * math.log2(p)
        max_h = math.log2(len(dist)) if len(dist) > 1 else 1e-9
        return h / max_h if max_h > 0 else 0.0

    def changed_frame_rate(self, state_sig: tuple, action_id: int) -> float:
        """Fracao de vezes que essa acao, nesse estado, mudou a cena."""
        dist = self.outcomes.get((state_sig, action_id))
        if not dist:
            return 0.5  # desconhecido: nem assume que muda nem que nao
        total = sum(dist.values())
        same = dist.get(state_sig, 0)
        return 1.0 - (same / total)
