"""
MTB-Sandbox-Salto2 — Salto 2: Auto-Arquitetura Cognitiva

Adiciona sobre o Salto 1:
  - CognitiveProgram: programas como entidades no CGE
  - ModelCompetition: modelos competem com colapso epistemico
  - LatentVector: variaveis latentes descobertas pelo sistema
"""
import time
import random
from typing import Dict, List, Optional, Tuple

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.goal_system import GoalManager
from core.bge_encoder import BGEEncoder
from core.error_signal import ErrorSignal, MultiErrorTracker
from core.attention_controller import (SalienceMap, ChangeDetector,
                                        SpotlightController)
from core.meta_controller import MetaController, CognitiveMode
from core.cognitive_program import (CognitiveProgram, ProgramForge,
                                     LoopEvaluator)
from core.model_competition import Model, ModelEcosystem
from core.latent_vector import LatentSpace
from mtb_world_generator import MTBWorld, MTBMaterial
from mtb_sandbox_salto import MTBSandboxSalto


class MTBSandboxSalto2(MTBSandboxSalto):
    """Sandbox com Salto 2: Auto-Arquitetura Cognitiva."""

    def __init__(self, cge: CognitiveGraphEngine, gm: GoalManager,
                 bge: Optional[BGEEncoder] = None):
        super().__init__(cge, gm, bge)

        # Salto 2 — Modulo A: Programas Cognitivos
        self.program_forge = ProgramForge(cge=cge)
        self.program_evaluator = LoopEvaluator(max_programs=32)

        # Salto 2 — Modulo B: Competicao de Modelos
        self.model_ecosystem = ModelEcosystem(max_models=12)
        self._setup_initial_models()

        # Salto 2 — Modulo C: Variaveis Latentes
        self.latent_space = LatentSpace(n_dims=6, cge=cge)

        # Rastreamento de episodios para descoberta de programas
        self._episode_buffer: List[Dict] = []
        self._current_episode: List[str] = []

    def _setup_initial_models(self):
        """Cria modelos iniciais sobre comportamento de materiais."""
        # Modelo: VORIN conduz bem
        def vorin_pred(obs):
            return {"target_energy": obs.get("source", 80) * 0.75 * (0.75 ** obs.get("cells", 4))}
        m1 = Model("vorin_model", "VORIN conduz com C=0.75", confidence=0.5,
                    domain="mtb_energy")
        m1.predictor = vorin_pred
        self.model_ecosystem.add_model(m1)

        # Modelo: NEXIUM conduz melhor
        def nexium_pred(obs):
            return {"target_energy": obs.get("source", 80) * 0.91 * (0.75 ** obs.get("cells", 4))}
        m2 = Model("nexium_model", "NEXIUM conduz com C=0.91", confidence=0.5,
                    domain="mtb_energy")
        m2.predictor = nexium_pred
        self.model_ecosystem.add_model(m2)

        # Modelo: material bloqueante nao conduz
        def blocking_pred(obs):
            return {"target_energy": 0.0}
        m3 = Model("blocking_model", "Materiais bloqueantes nao conduzem",
                    confidence=0.3, domain="mtb_energy", parsimony=0.8)
        m3.predictor = blocking_pred
        self.model_ecosystem.add_model(m3)

    def load_world(self, world: MTBWorld):
        """Carrega mundo com Salto 2 ativo."""
        super().load_world(world)

        # Resetar Salto 2
        self._episode_buffer = []
        self._current_episode = []

        # Criar estados latentes para materiais deste mundo
        for name, mat in world.materials.items():
            self.latent_space.get_or_create(
                f"mat_{name}",
                None  # observable_vec viria do BGE
            )

    def step_agi(self, max_steps: int = 50) -> Dict:
        """Ciclo AGI com CognitiveProgram + ModelCompetition + LatentVector."""
        if self.world is None:
            return {"error": "no world"}

        self.cycle += 1
        x, y = self.agi_pos
        energy_before = self.world.get_energy_at_target()

        # PERCEBER
        self._perceptual_state = self._build_perceptual_state()
        self.spotlight.update(self._perceptual_state, dt=0.2)
        focus = self.spotlight.current_focus

        # MOVER para observar
        action_result = self.world.interact("MOVE", x, y)

        # ERROR (Salto 1)
        self._update_error_signals(action_result, self.total_reward)

        # META (Salto 1)
        reward = self._get_reward_signal(energy_before, self.world.get_energy_at_target())
        self._cognitive_mode = self.meta.tick(reward_signal=reward)
        self._mode_counts[self._cognitive_mode.value] = \
            self._mode_counts.get(self._cognitive_mode.value, 0) + 1

        # ─── SALTO 2: COMPETICAO DE MODELOS ───
        obs_for_model = {
            "source": 80,
            "conductance": max(self._explored_materials.values()) if self._explored_materials else 0.5,
            "cells": len(self._path_plan) if self._path_plan else 4,
            "target_energy": self.world.get_energy_at_target(),
        }
        comp_result = self.model_ecosystem.compete(
            obs_for_model,
            predictor_fn=lambda obs: self._model_predict(obs)
        )

        # ─── SALTO 2: VARIAVEIS LATENTES ───
        for mat_name, cond in list(self._explored_materials.items())[:3]:
            self.latent_space.observe_and_update(
                f"mat_{mat_name}",
                {"conductance": cond, "energy": self.world.get_energy_at_target()}
            )

        # ─── SALTO 2: PROGRAMA COGNITIVO ───
        selected_program = self.program_evaluator.select({
            "novelty": self.error_tracker.get_state().get("avg_novelty", 0.3),
            "error": self.error_tracker.get_state().get("avg_prediction_error", 0.3),
            "uncertainty": self.error_tracker.get_state().get("avg_uncertainty", 0.5),
        })

        # DECIDIR acao (usando programa cognitivo se disponivel)
        if selected_program and random.random() < selected_program.confidence:
            action, ax, ay, material = self._decide_action_program(
                selected_program, self.world.get_energy_at_target()
            )
        else:
            action, ax, ay, material = self._decide_action(
                self.world.get_energy_at_target()
            )

        # AGIR
        result = self.world.interact(action, ax, ay, material)
        if action == "MOVE" and result["success"]:
            self.agi_pos = (ax, ay)

        # OBSERVAR
        energy_after = self.world.get_energy_at_target()
        step_reward = energy_after - energy_before
        self.total_reward += step_reward

        # ATUALIZAR programa cognitivo com resultado
        if selected_program:
            prog_reward = min(1.0, max(0.0, energy_after / 20.0))
            self.program_evaluator.evaluate_outcome(selected_program.id, prog_reward)

        # ATUALIZAR modelo vencedor
        winner = self.model_ecosystem.get_best_model(domain="mtb_energy")
        if winner:
            winner.update(abs(energy_after - energy_before) / max(1, energy_before + 1))

        # REGISTRAR episodio
        self._current_episode.append(action)
        if self.cycle % 5 == 0 and self._current_episode:
            self.program_forge.record_episode(
                context={"novelty": self.error_tracker.get_state().get("avg_novelty", 0.3),
                         "error": self.error_tracker.get_state().get("avg_prediction_error", 0.3)},
                actions=list(self._current_episode),
                reward=step_reward / max(1, abs(energy_before) + 1),
                outcome={"energy": energy_after}
            )
            self._current_episode = []

        # DETECTAR padroes (periodicamente)
        if self.cycle % 10 == 0:
            patterns = self.program_forge.detect_patterns()
            for pattern in patterns:
                new_prog = self.program_forge.forge_program(pattern)
                self.program_evaluator.register(new_prog)
                if self.cycle < 30:
                    print(f"  [SALTO2] Novo programa: {new_prog.id} "
                          f"(reward={pattern['avg_reward']:.2f})")

        # CONSOLIDACAO
        if self.cycle % 5 == 0 and self._cognitive_mode == CognitiveMode.CONSOLIDATE:
            try:
                self.cge.consolidation_cycle()
            except Exception:
                pass

        # Log
        entry = {
            "cycle": self.cycle,
            "pos": self.agi_pos,
            "action": action,
            "material": material,
            "reward": round(step_reward, 2),
            "energy_at_target": round(energy_after, 2),
            "cognitive_mode": self._cognitive_mode.value,
            "focus": focus,
            "program": selected_program.id if selected_program else None,
            "best_model": winner.id if winner else None,
            "model_error": round(comp_result.get("avg_error", 0), 2),
        }
        self.log.append(entry)

        # Verificar goal
        active_goal = self.gm.get_active_goal()
        if active_goal:
            obs = {"energy_at_target": energy_after,
                   "stability": len(self.world.interaction_log)}
            score = self.gm.evaluate(active_goal.id, obs)
            if active_goal.state == "completed":
                self.solved_worlds += 1
                print(f"\n[MTB:SALTO2] MUNDO {self.world.world_id} RESOLVIDO! "
                      f"Energia={energy_after:.2f}")

        self.world.step(dt=0.1)
        return entry

    def _model_predict(self, obs: Dict) -> Dict:
        """Predicao do modelo vencedor da competicao."""
        winner = self.model_ecosystem.get_best_model(domain="mtb_energy")
        if winner and hasattr(winner, 'predictor') and winner.predictor:
            try:
                return winner.predictor(obs)
            except Exception:
                pass
        # Fallback: modelo simples
        return {"target_energy": obs.get("source", 80) * 0.75 * (0.75 ** obs.get("cells", 4))}

    def _decide_action_program(self, program: CognitiveProgram,
                                energy_at_target: float) -> Tuple:
        """Decide acao baseada em um programa cognitivo."""
        effects = program.effect_profile
        explore_w = effects.get("exploration_weight", 0.5)

        if explore_w > 0.6:
            # Explorar mais
            return self._explore_action(
                self.agi_pos[0], self.agi_pos[1],
                self.world.width, self.world.height,
                self.world.target[0], self.world.target[1]
            )
        else:
            # Construir caminho
            return self._build_path_action(
                self.agi_pos[0], self.agi_pos[1],
                self.world.width, self.world.height,
                self.world.source[0], self.world.source[1],
                self.world.target[0], self.world.target[1]
            )

    def run_world(self, world: MTBWorld, max_cycles: int = 60) -> Dict:
        """Executa mundo com Salto 2."""
        self.load_world(world)
        for _ in range(max_cycles):
            result = self.step_agi()
            if result.get("energy_at_target", 0) > 5.0:
                break

        try:
            stats = self.cge.consolidation_cycle()
        except Exception:
            stats = {"edges_created": 0, "concepts_created": 0, "pruned_nodes": 0}

        # Recompensar programa
        energy = self.world.get_energy_at_target()
        if energy > 3.0:
            best_prog = self.program_evaluator.select(
                {"novelty": 0.1, "error": 0.1, "uncertainty": 0.1}
            )
            if best_prog:
                self.program_evaluator.evaluate_outcome(best_prog.id, 1.0)

        model_state = self.model_ecosystem.get_state()
        latent_state = self.latent_space.get_state()
        prog_state = self.program_evaluator.get_state()

        print(f"  [SALTO2] Programas: {prog_state['active']} ativos, "
              f"Modelos: {model_state['active']} ativos, "
              f"Latente: {latent_state['n_dims']} dims, {latent_state['discoveries']} descobertas")

        return {
            "world_id": world.world_id,
            "cycles": self.cycle,
            "solved": self.world.get_energy_at_target() > 3.0,
            "final_energy": self.world.get_energy_at_target(),
            "n_cge_nodes": len(self.cge.nodes),
            "n_cge_edges": len(self.cge.edges),
            "cognitive_modes": dict(self._mode_counts),
            "programs_active": prog_state["active"],
            "models_active": model_state["active"],
            "latent_dims": latent_state["n_dims"],
        }
