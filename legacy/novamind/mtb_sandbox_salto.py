"""
MTB-Sandbox-Salto — Sandbox com Salto AGI integrado (Fases 1-5)

Usa o MetaController para escolher modo cognitivo em tempo real,
MultiErrorTracker para erros multi-dimensionais, e AttentionController
para foco competitivo.

Uso: mesma interface que MTBSandbox, substituir no mtb_runner.py
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
from mtb_world_generator import MTBWorld, MTBMaterial
from mtb_sandbox import MTBSandbox


class MTBSandboxSalto(MTBSandbox):
    """Sandbox com Salto AGI (Fases 1-5) integrado."""

    def __init__(self, cge: CognitiveGraphEngine, gm: GoalManager,
                 bge: Optional[BGEEncoder] = None):
        super().__init__(cge, gm, bge)

        # Fase 1: Erro Multi-Dimensional
        self.error_tracker = MultiErrorTracker(max_errors=16)

        # Fase 2: Atencao Competitiva
        self.salience_map = SalienceMap(
            lateral_inhibition_fn=lambda nid: []  # integrado com CGE depois
        )
        self.change_detector = ChangeDetector(threshold=0.1)
        self.spotlight = SpotlightController(self.salience_map,
                                              self.change_detector)

        # Fase 3: Meta-Controller
        self.meta = MetaController(
            error_tracker=self.error_tracker,
            attention_controller=self.spotlight,
        )

        # Fase 4: Simulador Interno (modelos preditivos)
        self._models = {}  # nome -> funcao predict

        # Fase 5: Hipoteses
        self._hypotheses = []  # lista de dicts

        # Estado do spotlight para o mundo atual
        self._perceptual_state: Dict[str, float] = {}

        # Modo cognitivo atual (do MetaController)
        self._cognitive_mode = CognitiveMode.EXPLORE

        # Contadores para logging
        self._mode_counts: Dict[str, int] = {}

    def load_world(self, world: MTBWorld):
        """Carrega um mundo com os novos sistemas ativos."""
        super().load_world(world)

        # Resetar sistemas do Salto AGI
        self.error_tracker = MultiErrorTracker(max_errors=16)
        self.salience_map = SalienceMap()
        self.change_detector = ChangeDetector(threshold=0.1)
        self.spotlight = SpotlightController(self.salience_map,
                                              self.change_detector)
        self.meta = MetaController(
            error_tracker=self.error_tracker,
            attention_controller=self.spotlight,
        )
        self._perceptual_state = {}
        self._mode_counts = {}
        self._hypotheses = []

        # Criar erros iniciais para este mundo
        self.error_tracker.create_error(
            f"world_{world.world_id}_energy",
            prediction_error=0.5, salience=0.9,
            novelty=0.3, uncertainty=0.7,
            context={"world_id": world.world_id, "type": "energy"}
        )
        self.error_tracker.create_error(
            f"world_{world.world_id}_materials",
            prediction_error=0.0, salience=0.5,
            novelty=0.8, uncertainty=0.9,
            context={"world_id": world.world_id, "type": "materials"}
        )

        # Registrar modelos preditivos
        self._register_models()

    def _register_models(self):
        """Registra modelos preditivos para o simulador interno."""
        self._models["energy_flow"] = lambda state: {
            "target_energy": (
                state.get("source_energy", 80) *
                state.get("avg_conductance", 0.75) *
                (state.get("decay", 0.75) ** state.get("path_cells", 4))
            )
        }
        self._models["material_conductance"] = lambda state: {
            "effective_conductance": (
                state.get("base_conductance", 0.5) *
                (0.5 if state.get("is_blocking", False) else 1.0)
            )
        }

    def _build_perceptual_state(self) -> Dict[str, float]:
        """Constroi estado perceptual atual para atencao."""
        w = self.world
        if w is None:
            return {}

        state = {
            "source_energy": w.energy_grid[w.source[1]][w.source[0]],
            "target_energy": w.get_energy_at_target(),
        }

        # Materiais no grid (amostra limitada)
        for y in range(min(w.height, 4)):
            for x in range(min(w.width, 6)):
                mat = w.grid[y][x]
                if mat:
                    m = w.materials.get(mat)
                    if m:
                        state[f"mat_{x}_{y}"] = m.get_conductance()
                    else:
                        state[f"mat_{x}_{y}"] = -1.0

        return state

    def _update_error_signals(self, action_result: Dict, reward: float):
        """Atualiza sinais de erro multi-dimensionais."""
        w = self.world
        if w is None:
            return

        energy = w.get_energy_at_target()

        # Erro de energia (prediction error)
        energy_error = max(0, min(1, 1.0 - energy / 10.0))
        self.error_tracker.update_error(
            f"world_{w.world_id}_energy",
            prediction_error=energy_error,
            salience=0.5 + energy_error * 0.5,
            novelty=max(0, 0.3 - energy * 0.02),
        )

        # Erro de materiais (novidade/incerteza)
        explored = len(self._explored_materials)
        total = len(w.materials)
        uncertainty = 1.0 - (explored / max(1, total))
        novelty = 0.5 if explored < total * 0.7 else 0.1
        self.error_tracker.update_error(
            f"world_{w.world_id}_materials",
            prediction_error=1.0 - uncertainty,
            uncertainty=uncertainty,
            novelty=novelty,
            salience=0.3 + uncertainty * 0.7,
        )

        # Se acao falhou, criar erro de violacao
        if not action_result.get("success", True):
            self.error_tracker.create_error(
                f"world_{w.world_id}_action_fail",
                prediction_error=0.8,
                violation=0.7,
                salience=0.9,
                novelty=0.2,
                context={"action": action_result.get("action", "unknown")}
            )

    def _get_reward_signal(self, energy_before: float,
                           energy_after: float) -> float:
        """Calcula sinal de recompensa para o meta-controller."""
        if self.world is None:
            return 0.0

        delta = energy_after - energy_before
        if delta > 0:
            return min(1.0, 0.5 + delta * 0.1)
        elif delta < 0:
            return max(0.0, 0.5 + delta * 0.1)
        return 0.5

    def step_agi(self, max_steps: int = 50) -> Dict:
        """Um ciclo AGI com meta-controller e atencao."""
        if self.world is None:
            return {"error": "no world"}

        self.cycle += 1
        x, y = self.agi_pos
        energy_before = self.world.get_energy_at_target()

        # PERCEBER: atualizar estado perceptual
        self._perceptual_state = self._build_perceptual_state()

        # ATENCAO: spotlight detecta mudancas e foca
        self.spotlight.update(self._perceptual_state, dt=0.2)
        focus = self.spotlight.current_focus

        # ERROR: atualizar sinais de erro
        action_result = self.world.interact("MOVE", x, y)
        self._update_error_signals(action_result, self.total_reward)

        # META: meta-controller decide modo cognitivo
        reward = self._get_reward_signal(energy_before,
                                          self.world.get_energy_at_target())
        self._cognitive_mode = self.meta.tick(reward_signal=reward)
        self._mode_counts[self._cognitive_mode.value] = \
            self._mode_counts.get(self._cognitive_mode.value, 0) + 1

        # DECIDIR: acao baseada no modo cognitivo
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

        # CONSOLIDAR (periodicamente)
        if self.cycle % 5 == 0 and self._cognitive_mode == CognitiveMode.CONSOLIDATE:
            try:
                self.cge.consolidation_cycle()
            except Exception:
                pass

        # GERAR HIPOTESE (se no modo HYPOTHESIZE)
        if self._cognitive_mode == CognitiveMode.HYPOTHESIZE and self.cycle % 3 == 0:
            self._generate_hypothesis()

        # PREDIZER (se no modo PREDICT)
        if self._cognitive_mode == CognitiveMode.PREDICT and self.cycle % 3 == 0:
            self._run_predictions()

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
        }
        self.log.append(entry)

        # Verificar goal completion
        active_goal = self.gm.get_active_goal()
        if active_goal:
            obs = {"energy_at_target": energy_after,
                   "stability": len(self.world.interaction_log)}
            score = self.gm.evaluate(active_goal.id, obs)
            if active_goal.state == "completed":
                self.solved_worlds += 1
                print(f"\n[MTB:SALTO] MUNDO {self.world.world_id} RESOLVIDO! "
                      f"Energia={energy_after:.2f}, modo={self._cognitive_mode.value}")

        # Simular fisica
        self.world.step(dt=0.1)

        return entry

    def _generate_hypothesis(self):
        """Gera uma hipotese sobre materiais/caminho."""
        if self.world is None:
            return

        # Hipotese sobre melhor material
        if self._explored_materials:
            best = max(self._explored_materials, key=lambda m: self._explored_materials[m])
            best_cond = self._explored_materials[best]
            m = self.world.materials.get(best)
            is_blocking = m.is_blocking() if m else False
            penalty = 0.5 if is_blocking else 1.0

            hypo = {
                "type": "material",
                "material": best,
                "predicted_conductance": best_cond * penalty,
                "is_blocking": is_blocking,
                "confidence": 0.5 + best_cond * 0.3,
                "generated_at": self.cycle,
            }
            self._hypotheses.append(hypo)

        # Hipotese sobre caminho ideal
        sx, sy = self.world.source
        tx, ty = self.world.target
        cells = self._get_line_path((sx, sy), (tx, ty))
        path_len = len(cells) - 2  # exclui source/target

        if self._explored_materials:
            best = max(self._explored_materials, key=lambda m: self._explored_materials[m])
            best_cond = self._explored_materials[best]
            predicted = 80 * best_cond * (0.75 ** path_len)
            hypo = {
                "type": "path",
                "path_cells": path_len,
                "best_material": best,
                "predicted_energy": round(predicted, 2),
                "confidence": 0.4,
                "generated_at": self.cycle,
            }
            self._hypotheses.append(hypo)

    def _run_predictions(self):
        """Roda modelos preditivos e compara com realidade."""
        if self.world is None:
            return

        energy_fn = self._models.get("energy_flow")
        if energy_fn and self._explored_materials:
            best = max(self._explored_materials, key=lambda m: self._explored_materials[m])
            sx, sy = self.world.source
            tx, ty = self.world.target
            cells = self._get_line_path((sx, sy), (tx, ty))
            path_len = len(cells) - 2

            pred = energy_fn({
                "source_energy": 80,
                "avg_conductance": self._explored_materials[best],
                "decay": 0.75,
                "path_cells": max(1, path_len),
            })
            actual = self.world.get_energy_at_target()
            error = abs(pred["target_energy"] - actual)

    def _decide_action(self, energy_at_target: float) -> Tuple[str, int, int, Optional[str]]:
        """Decide acao baseada no MODO COGNITIVO atual.

        Todos os modos ativos constroem caminho, mas com estrategias diferentes.
        Apenas REFLECT e IDLE sao passivos.
        """
        x, y = self.agi_pos
        w, h = self.world.width, self.world.height
        tx, ty = self.world.target
        sx, sy = self.world.source
        mode = self._cognitive_mode

        # Se energia ja no target, colher
        if energy_at_target > 5.0:
            return ("MEASURE", tx, ty, None)

        # IDLE / REFLECT: passivo, nao constroi caminho
        if mode in (CognitiveMode.IDLE, CognitiveMode.REFLECT):
            if len(self._explored_materials) < len(self.world.materials) * 0.5:
                return self._explore_action(x, y, w, h, tx, ty)
            return ("MEASURE", x, y, None)

        # ─── EXPLORE: explora materiais primeiro, depois constroi ───
        if mode == CognitiveMode.EXPLORE:
            if len(self._explored_materials) < len(self.world.materials) * 0.5:
                return self._explore_action(x, y, w, h, tx, ty)
            return self._build_path_action(x, y, w, h, sx, sy, tx, ty)

        # ─── HYPOTHESIZE: testa materiais alternativos ───
        if mode == CognitiveMode.HYPOTHESIZE:
            return self._hypothesize_action(x, y, w, h, sx, sy, tx, ty)

        # ─── PREDICT / TEST / CONSOLIDATE / EXPLOIT:
        #      constroem caminho com melhor material conhecido ───
        return self._build_path_action(x, y, w, h, sx, sy, tx, ty)

    def _explore_action(self, x: int, y: int, w: int, h: int,
                        tx: int, ty: int) -> Tuple:
        """Explorar: medir material atual, mover para nao explorado."""
        # Medir material atual
        mat = self.world.grid[y][x]
        if mat and mat not in self._explored_materials:
            m = self.world.materials.get(mat)
            if m:
                cond = m.get_conductance()
                self._explored_materials[mat] = cond
                print(f"  [SALTO:EXPLORE] {mat}: cond={cond:.2f}, "
                      f"blocking={m.is_blocking()}")

        # Mover para celula nao explorada
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                m = self.world.grid[ny][nx]
                if m and m not in self._explored_materials:
                    return ("MOVE", nx, ny, None)

        if len(self._explored_materials) >= len(self.world.materials) * 0.5:
            return self._move_toward(tx, ty)
        return self._move_toward(tx or 0, ty or 0)

    def _build_path_action(self, x: int, y: int, w: int, h: int,
                           sx: int, sy: int, tx: int, ty: int) -> Tuple:
        """Construir caminho condutor."""
        if not hasattr(self, '_path_plan') or not self._path_plan:
            self._path_plan = self._get_line_path((sx, sy), (tx, ty))
            self._path_plan = [(px, py) for px, py in self._path_plan
                              if (px, py) != (sx, sy) and (px, py) != (tx, ty)]

        if not self._path_plan:
            return self._move_toward(tx, ty)

        # Melhor material conhecido
        if self._explored_materials:
            good = {m: c for m, c in self._explored_materials.items()
                    if c > 0.1 and m in self.world.materials
                    and not self.world.materials[m].is_blocking()}
            if good:
                best = max(good, key=lambda m: good[m])
                best_cond = good[best]
            else:
                # Nenhum material condutor encontrado: continuar explorando
                if self._explored_materials:
                    # Tentar o que tem maior (mesmo que zero)
                    best = max(self._explored_materials,
                              key=lambda m: self._explored_materials[m])
                    best_cond = self._explored_materials.get(best, 0)
                    # Se o melhor tem C=0, explorar mais
                    if best_cond < 0.1:
                        return self._explore_action(x, y, w, h, tx, ty)
                else:
                    best = list(self.world.materials.keys())[0]
                    best_cond = 0.0
        else:
            best = list(self.world.materials.keys())[0]
            best_cond = 0.0

        # Se nenhum material condutor foi encontrado, usar threshold minimo
        cond_threshold = max(best_cond * 0.5, 0.1)

        # Verificar se ja construiu tudo
        all_done = True
        for px, py in self._path_plan:
            mat = self.world.grid[py][px]
            if mat is None:
                all_done = False
                break
            m = self.world.materials.get(mat)
            if m:
                if m.is_blocking():
                    if good:
                        all_done = False
                        break
                elif m.get_conductance() < cond_threshold:
                    all_done = False
                    break

        if all_done:
            self._path_plan = []
            return self._move_toward(sx, sy)

        # Colocar material na proxima celula necessaria
        for px, py in self._path_plan:
            mat = self.world.grid[py][px]
            if mat is None:
                good_target = True
            else:
                m = self.world.materials.get(mat)
                if m is None:
                    good_target = True
                elif m.is_blocking() and good:
                    good_target = True
                elif not m.is_blocking() and m.get_conductance() < cond_threshold:
                    good_target = True
                else:
                    good_target = False

            if good_target:
                dx, dy = px - x, py - y
                dist = abs(dx) + abs(dy)
                if dist == 0:
                    return ("PLACE", px, py, best)
                elif dist == 1:
                    return ("MOVE", px, py, None)
                else:
                    return self._move_toward(px, py)

        return self._move_toward(tx, ty)

    def _hypothesize_action(self, x: int, y: int, w: int, h: int,
                            sx: int, sy: int, tx: int, ty: int) -> Tuple:
        """HYPOTHESIZE: testa material alternativo ao melhor conhecido."""
        # Se tem materiais nao testados, tentar um diferente do melhor
        if self._explored_materials:
            best = max(self._explored_materials, key=lambda m: self._explored_materials[m])
            # Encontrar material com condutividade intermediaria
            sorted_mats = sorted(self._explored_materials.items(),
                                key=lambda x: -x[1])
            target_mat = sorted_mats[min(1, len(sorted_mats)-1)][0]  # segundo melhor
            # Verificar se celula atual precisa de material e colocar o alternativo
            if self.world.grid[y][x] is None or not self.world.materials.get(
                    self.world.grid[y][x], MTBMaterial("", {})).get_conductance():
                return ("PLACE", x, y, target_mat)
        return self._build_path_action(x, y, w, h, sx, sy, tx, ty)

    def _test_action(self, x: int, y: int, w: int, h: int,
                     tx: int, ty: int) -> Tuple:
        """Testar: medir ao redor para validar/refutar hipotese."""
        return ("MEASURE", x, y, None)

    def run_world(self, world: MTBWorld, max_cycles: int = 60) -> Dict:
        """Executa um mundo completo com Salto AGI."""
        self.load_world(world)
        for _ in range(max_cycles):
            result = self.step_agi()
            if result.get("energy_at_target", 0) > 5.0:
                break

        # Consolidacao final
        try:
            stats = self.cge.consolidation_cycle()
        except Exception:
            stats = {"edges_created": 0, "concepts_created": 0, "pruned_nodes": 0}

        mode_counts_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self._mode_counts.items())
        )
        print(f"  [SALTO] Modos: {mode_counts_str}")

        return {
            "world_id": world.world_id,
            "cycles": self.cycle,
            "solved": self.world.get_energy_at_target() > 3.0,
            "final_energy": self.world.get_energy_at_target(),
            "cognitive_modes": dict(self._mode_counts),
            "hypotheses_generated": len(self._hypotheses),
            "n_cge_nodes": len(self.cge.nodes),
            "n_cge_edges": len(self.cge.edges),
        }
