"""Memoria de skills: assinatura-de-estado -> acao que, no passado, fez
parte de uma sequencia que completou um nivel.

Indexada por scene_signature (nao por game_id) de proposito: se dois jogos
diferentes produzem a mesma assinatura coarse de cena, a skill aprendida
num pode sugerir acao no outro. Isso e a forma mais simples possivel de
testar transferencia — sem prometer que ela vai acontecer.
"""
from __future__ import annotations

from collections import Counter, defaultdict


class SkillMemory:
    def __init__(self, min_confidence: int = 1):
        # state_sig -> Counter({action_id: vezes que apareceu num episodio de sucesso})
        self._votes: dict = defaultdict(Counter)
        # (state_sig, action_id) -> ultimo par de coordenadas usado com sucesso (para ACTION6)
        self._coord_hint: dict = {}
        self.min_confidence = min_confidence

    def record_success_episode(self, episode: list) -> None:
        """episode: lista de (state_sig, action_id, coord_dict) que terminou
        em incremento de levels_completed."""
        for state_sig, action_id, coord in episode:
            self._votes[state_sig][action_id] += 1
            if coord:
                self._coord_hint[(state_sig, action_id)] = (coord.get("x"), coord.get("y"))

    def suggest(self, state_sig: tuple):
        votes = self._votes.get(state_sig)
        if not votes:
            return None
        action_id, count = votes.most_common(1)[0]
        if count < self.min_confidence:
            return None
        coord = self._coord_hint.get((state_sig, action_id))
        return action_id, coord

    def n_skills(self) -> int:
        return sum(len(v) for v in self._votes.values())
