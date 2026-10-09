"""
CognitiveExecutive — Orquestrador integrado com CognitiveModeManager e WorldModel.

Agora o executivo alterna entre modos (FOCUSED, EXPLORATION, INCUBATION, DREAMING,
CONSOLIDATION) baseado no estado interno e surpresa do WorldModel.

Cada modo altera o comportamento de TODOS os modulos:
  - temperature → hypothesis_engine
  - beam_width/max_depth → synthesizer
  - constraint_strictness → constraint_network
  - concept_threshold → concept_registry
  - energy_cost → unified_self
"""
from typing import Optional
import numpy as np
import time

from .goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from .unified_self import UnifiedSelf
from .intrinsic_motivation import IntrinsicMotivationEngine, IntrinsicGoal
from .cognitive_modes import CognitiveModeManager, CognitiveMode


class CognitiveExecutive:
    """Orquestrador cognitivo com modos operacionais.

    Uso:
        exec = CognitiveExecutive(engine, self_model, mode_mgr)
        result = exec.cycle(input_grid, target_grid)
        idle = exec.idle_cycle()  # quando sem tarefa
    """

    def __init__(self, goal_engine: GoalDrivenEngine,
                 cfe_bridge=None,
                 self_model: Optional[UnifiedSelf] = None,
                 motivation: Optional[IntrinsicMotivationEngine] = None,
                 mode_manager: Optional[CognitiveModeManager] = None,
                 world_model=None,
                 hypothesis_engine=None,
                 synthesizer=None,
                 constraint_network=None,
                 concept_registry=None,
                 law_registry=None,
                 concept_economy=None,
                 conflict_resolver=None,
                 knowledge_graph=None,
                 inference_engine=None,
                 virtual_lab=None,
                 hdv_memory=None):
        self.engine = goal_engine
        self.cfe = cfe_bridge
        self.self_model = self_model or UnifiedSelf()
        self.motivation = motivation or IntrinsicMotivationEngine(
            self_model=self.self_model
        )
        self.mode_mgr = mode_manager or CognitiveModeManager()
        self.world = world_model
        self.hypothesis_engine = hypothesis_engine
        self.synthesizer = synthesizer
        self.constraint_network = constraint_network
        self.concept_registry = concept_registry
        self.law_registry = law_registry
        self.concept_economy = concept_economy
        self.conflict_resolver = conflict_resolver
        self.knowledge_graph = knowledge_graph
        self.inference_engine = inference_engine
        self.virtual_lab = virtual_lab
        self.hdv_memory = hdv_memory

        self.cmp = GoalComparator()
        self.history: list = []
        self._cycle = 0
        self._idle_cycles = 0

    # ─── METODOS AUXILIARES ───

    def _apply_mode_config(self):
        """Aplica configuracao do modo atual a todos os modulos."""
        config = self.mode_mgr.get_current_config()

        # Self: energia
        self.self_model.apply_mode_energy_cost(config.energy_cost)
        self.self_model.current_mode = config.name.lower()

        # Synthesizer: beam_width e max_depth
        if self.synthesizer is not None and hasattr(self.synthesizer, 'set_search_params'):
            self.synthesizer.set_search_params(
                beam_width=config.beam_width,
                max_depth=config.max_search_depth,
            )

        # HypothesisEngine: temperatura
        if self.hypothesis_engine is not None and hasattr(self.hypothesis_engine, 'set_temperature'):
            self.hypothesis_engine.set_temperature(config.temperature)

        # ConstraintNetwork: strictness
        if self.constraint_network is not None and hasattr(self.constraint_network, 'set_strictness'):
            self.constraint_network.set_strictness(config.constraint_strictness)

        # ConceptRegistry: temperatura de compatibilidade
        if self.concept_registry is not None and hasattr(self.concept_registry, 'set_temperature'):
            self.concept_registry.set_temperature(config.temperature)

    def _compute_surprise(self, input_grid, target_grid) -> float:
        """Computa surpresa via WorldModel."""
        if self.world is None or input_grid is None or target_grid is None:
            return 0.0
        pred = self.world.predict(input_grid)
        error = self.world.update(input_grid, target_grid, pred)
        return error.surprise

    # ─── CICLO PRINCIPAL ───

    def cycle(self, input_grid: np.ndarray = None,
              target_grid: np.ndarray = None) -> dict:
        """Um ciclo com tarefa externa.

        1. Atualiza modo cognitivo baseado em surpresa + energia
        2. Aplica configuracao do modo a todos os modulos
        3. Executa GoalDrivenEngine
        4. Atualiza self, world model, historico
        """
        self._cycle += 1
        self.self_model.tick()
        has_task = target_grid is not None

        # 1. Computa surpresa
        surprise = self._compute_surprise(input_grid, target_grid)

        # 2. Mode manager decide o modo
        self.mode_mgr.tick(
            has_task=has_task,
            success_rate=self._get_success_rate(),
            surprise=surprise,
            energy=self.self_model.energy,
        )

        # 3. Aplica configuracao do modo
        self._apply_mode_config()

        # 4. Metas intrinsecas
        intrinsic_goals = self.motivation.generate_goals()

        # 5. KnowledgeGraph: enriquece estado com conceitos relevantes
        if self.knowledge_graph and has_task and input_grid is not None:
            # Tenta extrair conceitos do estado para enriquecer
            pass

        # 6. Executar
        start_time = time.time()
        success = False
        exact = False
        acc = 0.0
        gap_reduction = 0.0
        goal_source = "external"

        if target_grid is not None and input_grid is not None:
            self.engine.set_goal_from_target(target_grid)
            self.engine.perceive(input_grid)
            self.engine.run(max_steps=3)

            final = self.engine.current_state
            if final.shape == target_grid.shape:
                exact = bool(np.array_equal(final, target_grid))
                acc = 1.0 - int((final != target_grid).sum()) / max(1, int(target_grid.size))
                gap_before = self.cmp.compute(input_grid, GoalState(target=target_grid)).total
                gap_after = self.cmp.compute(final, GoalState(target=target_grid)).total
                gap_reduction = gap_before - gap_after
                success = exact or acc > 0.8
        else:
            # Meta intrinseca
            goal = intrinsic_goals[0] if intrinsic_goals else None
            if goal:
                drive = self.self_model.drives.get(goal.drive)
                if drive:
                    drive.satisfy()
                goal_source = f"intrinsic:{goal.drive}"
                success = True

        elapsed = time.time() - start_time

        # 6. Atualizar self
        self.self_model.update_mood(success, 1.0 - acc)
        if acc > 0:
            self.self_model.update_capability("arc_solving", acc * 0.1)
        if success:
            self.self_model.n_successes += 1
        else:
            self.self_model.n_failures += 1

        self.self_model.memory.record(
            event_type="success" if success else "failure",
            description=f"cycle {self._cycle}: acc={acc:.2%} exact={exact} mode={self.mode_mgr.current_mode.value}",
            concepts=["arc", goal_source.split(":")[0]],
            valence=0.5 if success else -0.3,
            conf_delta=acc - 0.5 if acc > 0 else 0,
        )

        if self.cfe and input_grid is not None and target_grid is not None:
            self.cfe.discovery.observe_transformation(input_grid, target_grid,
                                                       f"cycle_{self._cycle}",
                                                       domain="arc")

        result = {
            "cycle": self._cycle,
            "exact": exact,
            "accuracy": round(acc, 4),
            "gap_reduction": round(gap_reduction, 4),
            "success": success,
            "goal_source": goal_source,
            "mode": self.mode_mgr.current_mode.value,
            "time_ms": round(elapsed * 1000, 1),
            "self": {
                "mood": self.self_model.current_mood,
                "energy": round(self.self_model.energy, 2),
            },
            "intrinsic_goals": [g.to_dict() for g in intrinsic_goals],
        }
        self.history.append(result)
        return result

    # ─── CICLO SEM TAREFA (IDLE / BACKGROUND) ───

    def idle_cycle(self) -> dict:
        """Ciclo executado quando nao ha tarefa externa.

        O modo cognitivo determina o que acontece:
          INCUBATION → recombina conceitos, gera teorias
          DREAMING → combinacoes aleatorias, simulacao
          CONSOLIDATION → memory replay, compressao
          EXPLORATION → busca ampla sem target
        """
        self._idle_cycles += 1
        self.self_model.tick()

        # Mode manager decide (sem tarefa)
        self.mode_mgr.tick(
            has_task=False,
            success_rate=self._get_success_rate(),
            surprise=self.world.get_surprise() if self.world else 0.0,
            energy=self.self_model.energy,
        )

        self._apply_mode_config()
        mode = self.mode_mgr.current_mode
        config = self.mode_mgr.get_current_config()

        report = {
            "idle_cycle": self._idle_cycles,
            "mode": mode.value,
            "temperature": config.temperature,
            "energy": round(self.self_model.energy, 2),
        }

        # INCUBATION: recombina conceitos
        if mode == CognitiveMode.INCUBATION:
            if self.concept_registry:
                agents = self.concept_registry.get_all(min_confidence=0.1)
                report["agents_available"] = len(agents)
                # Cross-pollination: busca compativeis a temperatura atual
                cross = []
                for a in agents:
                    compat = self.concept_registry.get_compatible(a.name)
                    if compat:
                        cross.append((a.name, [c.name for c in compat[:3]]))
                report["cross_pollinations"] = cross[:5]

        # DREAMING: geracao de hipoteses sem observacao
        if mode == CognitiveMode.DREAMING:
            if self.hypothesis_engine:
                obs = {"mode": "dreaming", "temperature": config.temperature}
                hyps = self.hypothesis_engine.generate_hypotheses(
                    obs, domain="internal", modality="abstract"
                )
                report["dream_hypotheses"] = [h.concept for h in hyps[:5]]

            # DREAMING: inference engine gera cadeias causais novas
            if self.inference_engine:
                novel = self.inference_engine.run_all()
                report["inference_chains"] = len(novel)
                if self.knowledge_graph and novel:
                    for h in novel[:5]:
                        self.knowledge_graph.add_causal_relation(
                            source=h.cause,
                            target=h.effect,
                            relation_type=h.inference_type,
                            confidence=h.confidence,
                        )

            # DREAMING: HDV brain fornece contexto associativo
            if self.hdv_memory:
                dream_context = self.hdv_memory.get_context(
                    f"dream cycle {self._idle_cycles}", top_k=5
                )
                if dream_context:
                    report["hdv_dream_context"] = dream_context[:200]

        # CONSOLIDATION: memory replay + integracao no KG
        if mode == CognitiveMode.CONSOLIDATION:
            report["consolidation"] = True
            # Forca o mode manager a resetar contador
            self.mode_mgr.cycles_since_consolidation = 0

            # CONSOLIDATION: InferenceEngine gera novo conhecimento
            if self.inference_engine and self.knowledge_graph:
                novel = self.inference_engine.run_all()
                for h in novel[:10]:
                    try:
                        self.knowledge_graph.add_causal_relation(
                            source=h.cause,
                            target=h.effect,
                            relation_type="inferred_consolidated",
                            confidence=h.confidence,
                        )
                    except:
                        pass
                report["consolidated_inferences"] = len(novel)

            # CONSOLIDATION: HDV brain alimenta KG
            if self.hdv_memory and self.knowledge_graph:
                # Pega conceitos aleatorios do HDV e adiciona ao KG
                for domain in ["schizophrenia", "alzheimer", "parkinson", "depression"]:
                    ctx = self.hdv_memory.get_context(domain, top_k=2)
                    if ctx:
                        report[f"hdv_{domain}"] = ctx[:100]

        # EXPLORATION: generate curiosity + simular + integrar
        if mode == CognitiveMode.EXPLORATION:
            goals = self.motivation.generate_goals()
            report["exploration_goals"] = [g.to_dict() for g in goals[:3]]
            # Law discovery: tenta unificar conceitos em novas leis
            if self.law_registry and self.concept_registry:
                agents = self.concept_registry.get_all(min_confidence=0.1)
                discoveries = []
                for i in range(len(agents)):
                    for j in range(i + 1, len(agents)):
                        a, b = agents[i], agents[j]
                        shared = a.domains & b.domains
                        if len(shared) >= 2:
                            new_law = self.law_registry.discover_law(a.name, b.name, shared)
                            if new_law:
                                discoveries.append(new_law.name)
                report["law_discoveries"] = discoveries

            # EXPLORATION: VirtualLab simula intervencoes
            if self.virtual_lab:
                sim_results = []
                for disease in ["Alzheimer", "Parkinson"]:
                    try:
                        self.virtual_lab.load_disease_model(disease)
                        result = self.virtual_lab.simulate_intervention("baseline", dose=0)
                        sim_results.append({
                            "disease": disease,
                            "biomarkers": result.biomarker_change if hasattr(result, 'biomarker_change') else {},
                            "confidence": result.confidence if hasattr(result, 'confidence') else 0.5,
                        })
                    except Exception:
                        pass
                report["virtual_lab_simulations"] = sim_results

            # EXPLORATION: HDV brain como contexto
            if self.hdv_memory:
                ctx = self.hdv_memory.get_context(
                    f"exploration mode cycle {self._idle_cycles}", top_k=5
                )
                if ctx:
                    report["hdv_context"] = ctx[:200]

            # EXPLORATION: ConceptEconomy faz merge/split/prune
            if self.concept_economy:
                try:
                    economy_result = self.concept_economy.run()
                    report["concept_economy"] = economy_result
                except:
                    pass

        # SURPRESA: alimenta neuromodulacao
        if self.world:
            self.self_model.noradrenaline = min(1.0, self.world.get_surprise())

        # Consistency check every 5 idle cycles
        if self._idle_cycles % 5 == 0 and self.world and self.law_registry:
            conflicts = self.world.check_consistency("general", {"cycle": self._idle_cycles})
            if conflicts:
                report["consistency_conflicts"] = len(conflicts)
                if self.conflict_resolver:
                    for c in conflicts[:3]:
                        conflict = self.conflict_resolver.detect_conflict(
                            law_name=c["law_a"],
                            domain=c["domain"],
                            observation=c,
                            prediction=None,
                            severity=c.get("confidence_a", 0.5),
                        )
                        hypotheses = self.conflict_resolver.resolve(conflict)
                        report["resolved_conflicts"] = [
                            h.to_dict() for h in hypotheses if h.passed
                        ]

        return report

    # ─── HELPERS ───

    def _get_success_rate(self) -> float:
        """Taxa de sucesso recente."""
        recent = [h for h in self.history[-20:] if h.get("success") is not None]
        if not recent:
            return 0.5
        return sum(1 for h in recent if h["success"]) / len(recent)

    def get_stats(self) -> dict:
        return {
            "total_cycles": self._cycle,
            "total_idle_cycles": self._idle_cycles,
            "mode": self.mode_mgr.get_stats(),
            "self": self.self_model.get_state(),
            "recent_history": self.history[-5:],
        }

    def summarize_self(self) -> str:
        mode = self.mode_mgr.current_mode.value
        energy = self.self_model.energy
        mood = self.self_model.current_mood
        extras = []
        if self.knowledge_graph:
            extras.append("KG")
        if self.inference_engine:
            extras.append("INF")
        if self.virtual_lab:
            extras.append("VLAB")
        if self.hdv_memory:
            extras.append("HDV")
        tag = f"[{'|'.join(extras)}]" if extras else ""
        return f"[{mode.upper()}] {mood}, energy={energy:.0%} {tag}"
