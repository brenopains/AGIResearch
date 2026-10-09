"""NovaMindAgent — agente minimo da Fase 1 do plano.

Percepcao object-centric -> modelo de transicao count-based -> selecao de
acao por ganho de informacao esperado PONDERADO POR RISCO -> inferencia de
objetivo por atribuicao de credito -> memoria de skills. Sete mecanismos
agora (risco entrou depois do diagnostico em bd01/bn01: morte instantanea
por acao mal ponderada). Mesma interface de RandomAgent/GreedyAgent
(`.act(frame) -> (action_id, data)`) para que bench.py possa comparar os
tres sem tratamento especial.
"""
from __future__ import annotations

import random

from agent3.action_selection import select_action
from agent3.goal_inference import GoalTracker
from agent3.perception import extract_scene, scene_signature
from agent3.risk_tracker import ActionRiskTracker
from agent3.skill_memory import SkillMemory
from agent3.trace import TraceLogger
from agent3.transition_model import TransitionModel

# SkillMemory pode ser compartilhada entre instancias (entre jogos) para
# medir transferencia real: se uma assinatura de cena aparece em dois
# jogos diferentes, a skill aprendida num pode ser sugerida no outro.
# bench.py recria o agente a cada jogo — por isso a skill memory vive
# num escopo de modulo, nao na instancia. Risco NAO e compartilhado: o
# significado de ACTION6 (e o quao perigoso ele e) e especifico de cada
# jogo, misturar entre jogos so adicionaria ruido.
_SHARED_SKILL_MEMORY = SkillMemory()


class NovaMindAgent:
    name = "novamind"

    def __init__(self, seed: int = 0, trace: bool = False, shared_skills: bool = True):
        self.rng = random.Random(seed)
        self.transition_model = TransitionModel()
        self.skill_memory = _SHARED_SKILL_MEMORY if shared_skills else SkillMemory()
        self.risk_tracker = ActionRiskTracker()
        self.goal_tracker = GoalTracker()
        self.trace = TraceLogger() if trace else None

        self._prev_state_sig = None
        self._prev_action_id = None
        self._prev_coord = None
        self._done = False

    def _process_outcome(self, frame) -> None:
        """Fecha o loop causal do passo anterior: (estado, acao) -> o que
        aconteceu. Chamado tanto de act() (para o proximo passo normal)
        quanto de observe_terminal() (para o ultimo passo de um episodio,
        que bench.py nao teria como entregar via act() — o loop para
        assim que WIN/GAME_OVER chega, entao sem este gancho a ultima
        acao nunca seria creditada nem penalizada)."""
        if self._prev_state_sig is None or self._done:
            return

        state_sig = scene_signature(extract_scene(frame.frame))
        self.transition_model.record(self._prev_state_sig, self._prev_action_id, state_sig)

        success_episode = self.goal_tracker.observe(
            self._prev_state_sig, self._prev_action_id, self._prev_coord,
            frame.levels_completed,
        )
        if success_episode:
            self.skill_memory.record_success_episode(success_episode)

        if frame.state.name == "GAME_OVER":
            self.risk_tracker.record_failure(self._prev_action_id)

    def act(self, frame) -> tuple[int, dict]:
        self._process_outcome(frame)

        scene = extract_scene(frame.frame)
        state_sig = scene_signature(scene)

        action_id, data, reason = select_action(
            frame.available_actions, state_sig,
            self.transition_model, self.skill_memory, self.rng,
            risk_tracker=self.risk_tracker,
        )
        self.risk_tracker.record_attempt(action_id)

        if self.trace is not None:
            self.trace.record(
                state_sig_len=len(state_sig), n_objects=sum(len(s) for s in scene),
                action=action_id, data=data, reason=reason,
                levels_completed=frame.levels_completed,
                n_skills=self.skill_memory.n_skills(),
                risk=self.risk_tracker.failure_rate(action_id),
            )

        self._prev_state_sig = state_sig
        self._prev_action_id = action_id
        self._prev_coord = data
        return action_id, data

    def observe_terminal(self, frame) -> None:
        """bench.py chama isso apos o loop de step() terminar (frame final
        em WIN ou GAME_OVER), para que a ultima acao do episodio nao fique
        sem credito nem sem penalidade."""
        self._process_outcome(frame)
        self._done = True
