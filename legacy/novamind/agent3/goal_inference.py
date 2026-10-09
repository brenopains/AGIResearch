"""Inferencia de objetivo por atribuicao de credito retroativa.

O jogo nunca diz o objetivo. O unico sinal e `levels_completed` subindo.
Quando sobe, a sequencia de (estado, acao) desde o ultimo incremento e a
evidencia mais direta que existe de "isso funcionou" — devolvida para que
skill_memory a grave. E a unica forma de credito disponivel neste jogo.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoalTracker:
    last_levels_completed: int = 0
    episode: list = field(default_factory=list)  # [(state_sig, action_id, coord), ...]

    def observe(self, state_sig: tuple, action_id: int, coord: dict,
                new_levels_completed: int) -> list | None:
        """Chamar apos cada passo. Retorna o episodio de sucesso (lista de
        (state_sig, action_id, coord)) se um nivel acabou de ser completado,
        senao None. O episodio ja inclui o passo atual."""
        self.episode.append((state_sig, action_id, coord))

        if new_levels_completed > self.last_levels_completed:
            success_episode = list(self.episode)
            self.episode.clear()
            self.last_levels_completed = new_levels_completed
            return success_episode

        return None
