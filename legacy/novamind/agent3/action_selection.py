"""Selecao de acao por ganho de informacao esperado (EIG) sobre o modelo
de transicao, com desvio para skill_memory quando ha uma sugestao confiavel.

E o coracao da eficiencia medida pelo RHAE: nao adianta resolver o jogo,
tem que resolver gastando poucas acoes. Isso significa nao repetir provas
que ja deram a mesma resposta (baixa entropia, muitas visitas) e preferir
acoes cujo efeito ainda e incerto.
"""
from __future__ import annotations

import random

ACTION6_ID = 6  # arcengine.GameAction.ACTION6 — unica acao que carrega (x, y)


def _coordinate_for(action_id: int, rng: random.Random, hint: tuple | None) -> dict:
    if action_id != ACTION6_ID:
        return {}
    if hint is not None:
        x, y = hint
        return {"x": x, "y": y}
    return {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}


def eig_score(transition_model, state_sig: tuple, action_id: int,
              risk_tracker=None, risk_weight: float = 0.6) -> float:
    visits = transition_model.visits(state_sig, action_id)
    novelty = 1.0 / (1.0 + visits)
    entropy = transition_model.outcome_entropy(state_sig, action_id)
    # nunca tentado pesa mais que "ja tentei, mas o efeito varia" — informacao
    # nova vale mais que refinar uma estimativa existente.
    score = 0.7 * novelty + 0.3 * entropy
    if risk_tracker is not None:
        # ganho de informacao por unidade de risco, nao ganho de informacao
        # bruto — uma acao com alta taxa de fim-de-jogo precisa de MUITO
        # mais novidade pra justificar ser tentada de novo.
        score -= risk_weight * risk_tracker.failure_rate(action_id)
    return score


def select_action(
    frame_available_actions: list[int],
    state_sig: tuple,
    transition_model,
    skill_memory,
    rng: random.Random,
    risk_tracker=None,
    exploit_epsilon: float = 0.15,
) -> tuple[int, dict, str]:
    """Retorna (action_id, data, motivo) — motivo vai pro trace.py."""
    avail = list(frame_available_actions) or [1]

    suggestion = skill_memory.suggest(state_sig) if skill_memory is not None else None
    if suggestion is not None and suggestion[0] in avail and rng.random() > exploit_epsilon:
        action_id, coord_hint = suggestion
        return action_id, _coordinate_for(action_id, rng, coord_hint), "skill_replay"

    scored = [(eig_score(transition_model, state_sig, a, risk_tracker), a) for a in avail]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score = scored[0][0]
    top = [a for s, a in scored if s >= best_score - 1e-9]
    action_id = rng.choice(top)

    reason = "explore_untried" if transition_model.visits(state_sig, action_id) == 0 else "explore_entropy"
    return action_id, _coordinate_for(action_id, rng, None), reason
