"""
MTB-1 Sandbox — AGI-world interface (Salto AGI integrado)

Conecta o mundo procedural com:
  - CGE (grafo cognitivo)
  - GoalManager (metas)
  - MultiErrorTracker (erro multi-dimensional - Fase 1)
  - AttentionController (atencao competitiva - Fase 2)
  - MetaController (modos cognitivos - Fase 3)
  - Simulador Interno (predicao - Fase 4)
  - Gerador de Hipoteses (inferencia ativa - Fase 5)
  - BGE encoder (percepcao textual)
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


class MTBSandbox:
    """Sandbox que conecta AGI ao mundo MTB."""

    def __init__(self, cge: CognitiveGraphEngine, gm: GoalManager,
                 bge: Optional[BGEEncoder] = None):
        self.cge = cge
        self.gm = gm
        self.bge = bge
        self.world: Optional[MTBWorld] = None
        self.agi_pos = (0, 0)
        self.cycle = 0
        self.total_reward = 0.0
        self.solved_worlds = 0
        self._path_plan: List[Tuple[int, int]] = []  # plano de caminho
        self._path_idx = 0
        self._phase = "explore"  # explore | build_path | harvest | done

        # Metricas
        self.log: List[Dict] = []
        self.world_results: List[Dict] = []
        self._explored_materials: Dict[str, float] = {}  # material -> conductance conhecida

    def load_world(self, world: MTBWorld):
        """Carrega um mundo para a AGI explorar."""
        self.world = world
        self.agi_pos = world.source
        self.cycle = 0
        self.total_reward = 0.0
        self._path_plan = []
        self._phase = "explore"
        self._explored_materials = {}

        # Resetar estado AGI para este mundo
        self.cge.clear()

        # Criar nos para materiais no CGE com HDVs via BGE ou hash
        for name, mat in world.materials.items():
            if self.bge:
                vec = self.bge.encode_bipolar([name]).squeeze(0)
            else:
                import torch
                import hashlib
                h = hashlib.md5(name.encode()).hexdigest()
                seed = int(h[:8], 16)
                rng_t = torch.Generator(device=self.cge.device)
                rng_t.manual_seed(seed)
                vec = torch.sign(torch.randn(self.cge.dim, generator=rng_t, device=self.cge.device))
            self.cge.create_node(name, vec)

        # Meta: transportar energia da fonte ao destino
        criteria = {"energy_at_target": 5.0, "stability": 3}
        gid = self.gm.create_goal(
            f"transport_energy_w{world.world_id}",
            criteria,
            priority=8
        )

        # Alimentar descricao inicial do mundo
        desc = world.get_state_text()
        if self.bge:
            self.cge.write_text_concepts(desc, emotional_valence=0.3, emotional_arousal=0.5)

        print(f"\n[MTB] Mundo {world.world_id} carregado. "
              f"Fisica: {world.physics['energy_flow']}+{world.physics['gravity']}. "
              f"Materiais: {list(world.materials.keys())}")

    def step_agi(self, max_steps: int = 50) -> Dict:
        """Um ciclo AGI: percebe -> raciocina -> age -> observa -> aprende."""
        if self.world is None:
            return {"error": "no world"}

        self.cycle += 1
        active_goal = self.gm.get_active_goal()
        if active_goal is None:
            self.gm.activate_next()
            active_goal = self.gm.get_active_goal()

        # PERCEBER: estado atual do mundo
        x, y = self.agi_pos
        percep = self.world.interact("MOVE", x, y)
        energy_at_target = self.world.get_energy_at_target()
        state_desc = (
            f"Cycle {self.cycle}: AGI at ({x},{y}). "
            f"{percep['observation']}. "
            f"Energy at target: {energy_at_target:.2f}"
        )

        # Escrever no CGE
        nm = getattr(self, '_neuromodulator', None)
        valence = 0.3 + (energy_at_target / 20.0)
        self.cge.write_text_concepts(state_desc, emotional_valence=valence, emotional_arousal=0.4)

        # RACIOCINAR: planejar proxima acao
        action, ax, ay, material = self._decide_action(energy_at_target)

        # AGIR
        result = self.world.interact(action, ax, ay, material)

        # Se MOVE, atualizar posicao
        if action == "MOVE" and result["success"]:
            self.agi_pos = (ax, ay)

        # OBSERVAR resultado
        new_energy = self.world.get_energy_at_target()
        reward = new_energy - energy_at_target
        self.total_reward += reward

        obs = {
            "energy_at_target": new_energy,
            "stability": len(self.world.interaction_log),
            "action": action,
            "success": result["success"],
        }

        # AVALIAR progresso na meta
        if active_goal:
            score = self.gm.evaluate(active_goal.id, obs)

            # ERROR-DRIVEN LEARNING
            path = [(e["from"], e["to"]) for e in self.cge.edges[-5:]]
            self.gm.apply_error_signal(active_goal.id, score, path_taken=path)

            # Se completou, marcar
            if active_goal.state == "completed":
                self.solved_worlds += 1
                print(f"\n[MTB] MUNDO {self.world.world_id} RESOLVIDO! "
                      f"Energia no target: {new_energy:.2f}")
                self.world_results.append({
                    "world_id": self.world.world_id,
                    "cycles": self.cycle,
                    "final_energy": new_energy,
                    "materials_used": len(self.world.interaction_log),
                })

        # Simular fisica
        self.world.step(dt=0.1)

        # Log
        entry = {
            "cycle": self.cycle,
            "pos": self.agi_pos,
            "action": action,
            "action_target": (ax, ay),
            "material": material,
            "reward": round(reward, 2),
            "energy_at_target": round(new_energy, 2),
            "goal_score": round(score, 2) if active_goal else 0,
            "goal_state": active_goal.state if active_goal else "none",
        }
        self.log.append(entry)

        return entry

    def _decide_action(self, energy_at_target: float) -> Tuple[str, int, int, Optional[str]]:
        """AGI decide acao baseada no estado atual."""
        x, y = self.agi_pos
        w, h = self.world.width, self.world.height
        tx, ty = self.world.target
        sx, sy = self.world.source

        # Se energia ja chegando ao target, colher
        if energy_at_target > 5.0:
            self._phase = "done"
            return ("MEASURE", tx, ty, None)

        # --- FASE 1: EXPLORAR materiais via MOVE ---
        if self._phase == "explore":
            mat = self.world.grid[y][x]
            if mat and mat not in self._explored_materials:
                m = self.world.materials.get(mat)
                if m:
                    cond = m.get_conductance()
                    self._explored_materials[mat] = cond
                    print(f"  [AGI] Explorou {mat}: conduct={cond:.2f}, "
                          f"props={list(m.properties.keys())}")

            # Mover para celula nao explorada
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    m = self.world.grid[ny][nx]
                    if m and m not in self._explored_materials:
                        return ("MOVE", nx, ny, None)

            if len(self._explored_materials) >= len(self.world.materials) * 0.5:
                print(f"  [AGI] Fase explore completa ({len(self._explored_materials)}/"
                      f"{len(self.world.materials)} materiais) -> build_path")
                self._phase = "build_path"
                self._path_plan = self._get_line_path((sx, sy), (tx, ty))
                # Excluir source e target do plano (ja tem energia / e o destino)
                self._path_plan = [(px, py) for px, py in self._path_plan
                                   if (px, py) != (sx, sy) and (px, py) != (tx, ty)]
                self._path_idx = 0
                return self._move_toward(sx, sy)
            return self._move_toward(tx, ty)

        # --- FASE 2: CONSTRUIR CAMINHO ---
        if self._phase == "build_path":
            if self._explored_materials:
                # Filtrar materiais condutores NAO bloqueantes
                good_mats = {m: c for m, c in self._explored_materials.items()
                             if c > 0.1 and m in self.world.materials
                             and not self.world.materials[m].is_blocking()}
                if good_mats:
                    best_mat = max(good_mats, key=lambda m: good_mats[m])
                    best_conduct = good_mats[best_mat]
                else:
                    # Fallback: qualquer material com maior condutancia (mesmo bloqueante)
                    best_mat = max(self._explored_materials, key=lambda m: self._explored_materials[m])
                    best_conduct = self._explored_materials[best_mat]
            else:
                best_mat = list(self.world.materials.keys())[0]
                best_conduct = 0.0

            # Verificar se jah terminou de construir o caminho
            all_done = True
            has_non_blocking = good_mats and len(good_mats) > 0
            for px, py in self._path_plan:
                mat = self.world.grid[py][px]
                if mat is None:
                    all_done = False
                    break
                m = self.world.materials.get(mat)
                if m is None:
                    all_done = False
                    break
                if m.is_blocking():
                    # So aceita blocking se nao tem alternativa nao-blocking
                    if has_non_blocking:
                        all_done = False
                        break
                    elif m.get_conductance() < best_conduct * 0.3:
                        all_done = False
                        break
                elif m.get_conductance() < best_conduct * 0.5:
                    all_done = False
                    break

            if all_done:
                print(f"  [AGI] Caminho construido com {best_mat}! Indo colher.")
                self._phase = "harvest"
                return self._move_toward(sx, sy)

            # Ir ate a proxima celula que precisa de material melhor e colocar
            for px, py in self._path_plan:
                mat = self.world.grid[py][px]
                needs_replacement = False
                if mat is None:
                    needs_replacement = True
                else:
                    m = self.world.materials.get(mat)
                    if m is None:
                        needs_replacement = True
                    elif m.is_blocking():
                        if has_non_blocking:
                            needs_replacement = True
                        elif m.get_conductance() < best_conduct * 0.3:
                            needs_replacement = True
                    elif m.get_conductance() < best_conduct * 0.5:
                        needs_replacement = True

                if needs_replacement:
                    dx, dy = px - x, py - y
                    dist = abs(dx) + abs(dy)
                    if dist == 0:
                        return ("PLACE", px, py, best_mat)
                    elif dist == 1:
                        return ("MOVE", px, py, None)
                    else:
                        return self._move_toward(px, py)

            # Fallback
            return self._move_toward(tx, ty)

        # --- FASE 3: HARVEST ---
        if self._phase == "harvest":
            if (x, y) != (sx, sy):
                return self._move_toward(sx, sy)
            return ("MOVE", tx, ty, None)

        # Fallback
        return ("MEASURE", x, y, None)

    def _move_toward(self, tx: int, ty: int) -> Tuple[str, int, int, Optional[str]]:
        """Move um passo em direcao ao target (A* simplificado sem colisoes)."""
        x, y = self.agi_pos
        w = self.world.width
        h = self.world.height
        dx = 1 if x < tx else -1 if x > tx else 0
        dy = 1 if y < ty else -1 if y > ty else 0

        if dx != 0:
            nx, ny = x + dx, y
            if 0 <= nx < w:
                return ("MOVE", nx, ny, None)
        if dy != 0:
            nx, ny = x, y + dy
            if 0 <= ny < h:
                return ("MOVE", nx, ny, None)
        return ("MEASURE", x, y, None)

    def _get_line_path(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Retorna celulas entre p1 e p2 (linha de Bresenham)."""
        x1, y1 = p1
        x2, y2 = p2
        cells = []
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            cells.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
        return cells

    def run_world(self, world: MTBWorld, max_cycles: int = 60) -> Dict:
        """Executa um mundo completo."""
        self.load_world(world)
        for _ in range(max_cycles):
            result = self.step_agi(max_steps=1)
            if result.get("goal_state") == "completed":
                break

        # Consolidacao apos cada mundo
        stats = self.cge.consolidation_cycle()
        print(f"[MTB] Consolidacao: +{stats['edges_created']} edges, "
              f"+{stats['concepts_created']} conceitos, "
              f"poda de {stats['pruned_nodes']} nos")

        return {
            "world_id": world.world_id,
            "cycles": self.cycle,
            "solved": self.world.get_energy_at_target() > 3.0,
            "final_energy": self.world.get_energy_at_target(),
            "goal_state": self.gm.get_active_goal().state if self.gm.get_active_goal() else "none",
            "n_cge_nodes": len(self.cge.nodes),
            "n_cge_edges": len(self.cge.edges),
        }

    def run_multi_world(self, worlds: List[MTBWorld]) -> Dict:
        """Executa multiplos mundos em sequencia com transferencia."""
        results = []
        prev_concepts = set()

        for i, w in enumerate(worlds):
            r = self.run_world(w)
            results.append(r)

            # Medir transferencia: conceitos que persistem entre mundos
            current_concepts = set(self.cge.nodes.keys())
            transferred = current_concepts & prev_concepts if i > 0 else set()
            r["concepts_transferred"] = len(transferred)

            if transferred:
                print(f"[MTB] Transferidos {len(transferred)} conceitos: "
                      f"{list(transferred)[:5]}")

            prev_concepts = current_concepts

        # Calcular metricas agregadas
        solved = sum(1 for r in results if r["solved"])
        avg_cycles = sum(r["cycles"] for r in results) / len(results)

        return {
            "n_worlds": len(worlds),
            "solved": solved,
            "solved_pct": round(solved / len(worlds) * 100, 1),
            "avg_cycles": round(avg_cycles, 1),
            "world_results": results,
        }
