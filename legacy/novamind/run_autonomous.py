"""
Autonomous Run — Deixa a AGI existir por N ciclos sem tarefa externa.

O sistema:
  1. Carrega dataset multimodal expandido
  2. A cada ciclo: observa exemplos, alimenta CFE, gera hipoteses,
     executa economia de conceitos, descobre leis, detecta conflitos
  3. Periodicamente (a cada snapshot_interval): salva metricas
  4. Modos cognitivos alternam automaticamente (INCUBATION, DREAMING, etc)
  5. Ao final: gera relatorio de auto-organizacao

Uso:
    python run_autonomous.py --cycles 1000 --snapshot 50
"""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.goal_driven_engine import GoalDrivenEngine, GoalState
from core.world_model import WorldModel
from core.hypothesis_engine import HypothesisEngine
from core.conceptual_discovery.constraint_network import ConstraintNetwork
from core.conceptual_discovery.concept_agent import ConceptAgent, ConceptAgentRegistry
from core.conceptual_discovery.concept_economy import ConceptEconomy
from core.conceptual_discovery.world_laws import WorldLaw, WorldLawRegistry
from core.conceptual_discovery.conflict_resolver import ConflictResolver
from core.invariant_dataset import MultimodalInvariantDataset
from core.traceability import TraceabilityAnalyzer
from core.law_guided_synthesizer import LawGuidedSynthesizer
from core.adversarial_reality import AdversarialRealityGenerator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitive import Primitive


def setup_concept_agents(registry: ConceptAgentRegistry):
    """Cria e registra concept agents de exemplo."""
    agents = []

    # Reflection
    ref = ConceptAgent("reflection", "spatial_transformation")
    ref.add_prediction("orientation_inverted", True)
    ref.add_prediction("area_preserved", True)
    ref.add_prediction("color_preserved", True)
    ref.add_modal_signature("grid", ["flip", "rotate", "mirror"],
                            invariants=["shape_preserved"])
    ref.add_modal_signature("audio", ["echo", "reverb"],
                            invariants=["amplitude_decays"])
    ref.add_compatibility("symmetry", 0.8)
    ref.add_compatibility("rotation", 0.7)
    ref.add_contradiction("expansion")
    ref.observe("geometry"); ref.observe("physics"); ref.observe("audio")
    agents.append(ref)

    # Symmetry
    sym = ConceptAgent("symmetry", "structural_preservation")
    sym.add_prediction("structure_preserved", True)
    sym.add_prediction("count_preserved", True)
    sym.add_prediction("connectivity_preserved", True)
    sym.add_modal_signature("grid", ["bilateral", "rotational"],
                            invariants=["shape_preserved"])
    sym.add_modal_signature("biology", ["body_plan"],
                            invariants=["organization_preserved"])
    sym.add_compatibility("reflection", 0.9)
    sym.add_compatibility("rotation", 0.8)
    sym.add_contradiction("randomness")
    sym.observe("geometry"); sym.observe("biology")
    agents.append(sym)

    # Expansion
    exp = ConceptAgent("expansion", "scale_increase")
    exp.add_prediction("object_count_increases", True)
    exp.add_prediction("pattern_repeated", True)
    exp.add_prediction("proportions_preserved", True)
    exp.add_modal_signature("grid", ["tile", "scale", "copy"],
                            invariants=["pattern_preserved"])
    exp.add_modal_signature("physics", ["thermal_expansion"],
                            invariants=["proportions_preserved"])
    exp.add_compatibility("branching", 0.6)
    exp.add_contradiction("contraction")
    exp.observe("geometry"); exp.observe("physics")
    agents.append(exp)

    # Contraction
    con = ConceptAgent("contraction", "scale_decrease")
    con.add_prediction("object_count_decreases", True)
    con.add_prediction("area_decreases", True)
    con.add_modal_signature("grid", ["crop", "compress"],
                            invariants=["content_preserved"])
    con.add_compatibility("filtering", 0.7)
    con.add_contradiction("expansion")
    con.observe("geometry"); con.observe("physics")
    agents.append(con)

    # Branching
    bra = ConceptAgent("branching", "structural_division")
    bra.add_prediction("object_count_increases", True)
    bra.add_prediction("parent_structure_preserved", True)
    bra.add_prediction("recursive_pattern", True)
    bra.add_modal_signature("grid", ["fractal", "recursive_tile"],
                            invariants=["hierarchy_preserved"])
    bra.add_modal_signature("biology", ["tree", "vessels"],
                            invariants=["hierarchy_preserved"])
    bra.add_compatibility("expansion", 0.6)
    bra.add_contradiction("contraction")
    bra.observe("biology"); bra.observe("geography"); bra.observe("geometry")
    agents.append(bra)

    # Causal chain
    causal = ConceptAgent("causal_chain", "sequential_transformation")
    causal.add_prediction("order_preserved", True)
    causal.add_prediction("intermediate_steps_possible", True)
    causal.add_modal_signature("events", ["pipeline", "chain"],
                               invariants=["order_preserved"])
    causal.add_compatibility("spatial_transformation", 0.5)
    causal.observe("physics"); causal.observe("programming")
    agents.append(causal)

    for a in agents:
        registry.register(a)
    return agents


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous AGI Run")
    parser.add_argument("--cycles", type=int, default=10000,
                        help="Numero de ciclos")
    parser.add_argument("--snapshot", type=int, default=100,
                        help="Intervalo entre snapshots")
    args = parser.parse_args()

    print("=" * 60)
    print(f"AUTONOMOUS AGI RUN — {args.cycles} ciclos")
    print("=" * 60)

    # ─── Setup ───
    concept_registry = ConceptAgentRegistry()
    setup_concept_agents(concept_registry)
    print(f"\nConceptAgents: {concept_registry.count}")

    hypothesis_engine = HypothesisEngine(concept_registry=concept_registry)
    for agent in concept_registry.get_all():
        hypothesis_engine.register_agent(agent)

    world_model = WorldModel()
    law_registry = WorldLawRegistry()
    concept_economy = ConceptEconomy(prune_threshold=0.05)
    constraint_network = ConstraintNetwork(
        law_registry=law_registry,
        concept_registry=concept_registry,
    )
    conflict_resolver = ConflictResolver(
        law_registry=law_registry,
        constraint_network=constraint_network,
        hypothesis_engine=hypothesis_engine,
    )
    world_model.law_registry = law_registry

    self_model = UnifiedSelf()
    motivation = IntrinsicMotivationEngine(self_model=self_model)
    mode_mgr = CognitiveModeManager()

    engine = GoalDrivenEngine(use_planner=False)
    exec_ = CognitiveExecutive(
        goal_engine=engine,
        self_model=self_model,
        motivation=motivation,
        mode_manager=mode_mgr,
        world_model=world_model,
        hypothesis_engine=hypothesis_engine,
        constraint_network=constraint_network,
        concept_registry=concept_registry,
        law_registry=law_registry,
        concept_economy=concept_economy,
        conflict_resolver=conflict_resolver,
    )

    # Synthesizer setup (para law-guided search)
    synth_registry = PrimitiveRegistry()
    from core.program_induction.primitives import (
        create_recolor_primitive, create_remove_primitive,
        create_move_primitive, create_add_primitive,
    )
    from core.program_induction.operator_proposals import get_all_proposals
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        synth_registry.register(fn())
    for op in get_all_proposals():
        fn = op["function"]
        prim = Primitive(name=op["name"], structure_type=op["name"])
        def _me(f):
            def _e(g, p):
                try: return f(g)
                except: return g.copy()
            return _e
        prim._execute = _me(fn)
        synth_registry.register(prim)

    cg = CompositionGraph()
    og = OperatorGroundingBridge(synth_registry, cg)
    pie = ProgramSynthesizer(synth_registry, composition_graph=cg, grounding_bridge=og, max_depth=4)
    law_synthesizer = LawGuidedSynthesizer(
        pie, law_registry=law_registry,
        conflict_resolver=conflict_resolver,
        world_model=world_model,
    )

    # Adversarial Reality Generator
    adv_gen = AdversarialRealityGenerator(
        law_registry=law_registry,
        concept_registry=concept_registry,
    )

    # Dataset multimodal
    dataset_path = r"D:\novamind_data\multimodal_world\strategic_dataset.json"
    if os.path.exists(dataset_path):
        dataset = MultimodalInvariantDataset(dataset_path)
        print(f"Dataset: {dataset.total_examples} exemplos")
    else:
        dataset = MultimodalInvariantDataset()
        print(f"Dataset base: {dataset.total_examples} exemplos (usando fallback)")

    # Traceability
    analyzer = TraceabilityAnalyzer()

    # ─── Loop Principal ───
    start_time = time.time()
    examples = list(dataset._examples)
    n_examples = len(examples)

    for cycle in range(1, args.cycles + 1):
        # 1. Seleciona um exemplo do dataset (round-robin)
        ex = examples[(cycle - 1) % n_examples]
        inp = ex.input
        out = ex.output

        # 2. OBSERVA: alimenta TODOS os modulos com este exemplo
        # 2a. ConceptAgents: expande dominios + cria agentes dinamicamente
        matched = False
        for agent in concept_registry.get_all():
            invariant_match = (
                ex.invariant_type == agent.name or
                ex.invariant_type == agent.invariant_type or
                (ex.invariant_type in agent.name) or
                (agent.name in ex.invariant_type)
            )
            if invariant_match or ex.domain in agent.domains:
                agent.observe(ex.domain)
                matched = True

        # FEATURE 1: Cria novo ConceptAgent se padrao nao reconhecido
        if not matched:
            new_agent = ConceptAgent.create_for_pattern(
                ex.invariant_type, ex.domain,
                registry=concept_registry
            )
            hypothesis_engine.register_agent(new_agent)
            analyzer.log_event("new_agent",
                               f"new agent: {new_agent.name}",
                               {"type": ex.invariant_type, "domain": ex.domain})

        # 2b. ConceptEconomy: registra o conceito observado
        concept_economy.observe_concept(
            ex.invariant_type, {ex.domain},
            confidence=0.4 + (cycle % 5) * 0.1,
            reuse=cycle // n_examples + 1,
        )

        # 2c. WorldModel: faz predicao do exemplo e mede erro
        if isinstance(inp, np.ndarray) and isinstance(out, np.ndarray):
            pred = world_model.predict(inp)
            error = world_model.update(inp, out, pred)

            # FEATURE 3: Law-guided synthesis + verificacao de leis
            law_results = law_synthesizer.synthesize(inp, out, domain=ex.domain)
            if error.surprise > 0.3:
                analyzer.log_event("high_surprise",
                                   f"surprise={error.surprise:.2f} in {ex.domain}",
                                   {"domain": ex.domain, "type": ex.invariant_type})
        else:
            pred = world_model.predict(
                {"domain": ex.domain, "type": ex.invariant_type}
            )
            error = world_model.update(
                {"domain": ex.domain, "type": ex.invariant_type},
                {"confirmed": True, "domain": ex.domain},
                pred,
            )

        # 2d. HypothesisEngine: alimenta com observacao
        obs = {"domain": ex.domain, "invariant": ex.invariant_type,
               "modality": ex.modality}
        if isinstance(inp, np.ndarray):
            obs["grid"] = inp
            obs["target"] = out
        hypotheses = hypothesis_engine.generate_hypotheses(obs, ex.domain, ex.modality)

        # 3. A CADA N CICLOS: processamento autonomo
        if cycle % 5 == 0:
            # 3a. Economia de conceitos
            economy_actions = concept_economy.run()
            for a in economy_actions:
                analyzer.log_event(f"economy_{a.action_type}",
                                   f"{a.action_type}: {a.concept_a}",
                                   {"reason": a.reason})

            # 3b. Descoberta de leis entre conceitos co-ocorrentes
            #    Agora com constraint_network para filtrar leis ruins
            constraint_network.set_strictness(0.6)  # 50% dos checkers precisam aprovar
            agents = concept_registry.get_all(min_confidence=0.1)
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    shared = agents[i].domains & agents[j].domains
                    if len(shared) >= 2:
                        new_law = law_registry.discover_law(
                            agents[i].name, agents[j].name, shared,
                            constraint_network=constraint_network,
                            min_domains=2,
                        )
                        if new_law:
                            analyzer.log_event("law_discovery",
                                               f"new law: {new_law.name}",
                                               {"explains": list(new_law.explains),
                                                "domains": list(new_law.domains)})

            # 3c. Consistencia do modelo do mundo
            consistency_conflicts = world_model.check_consistency(
                ex.domain, {"type": ex.invariant_type}
            )
            for c in consistency_conflicts:
                conflict = conflict_resolver.detect_conflict(
                    law_name=c.get("law_a", "unknown"),
                    domain=ex.domain,
                    observation={"type": ex.invariant_type},
                    prediction=None,
                )
                hypotheses = conflict_resolver.resolve(conflict)

            # FEATURE 2: Pressao adversarial periodica
            if cycle % 25 == 0:
                adv_examples = adv_gen.generate_pressure_batch(n=5)
                for adv_ex in adv_examples:
                    analyzer.log_event("adversarial",
                                       f"{adv_ex.type} in {adv_ex.domain}",
                                       {"challenge": adv_ex.challenge[:60]})
                    # Alimenta conflict resolver
                    conflict = conflict_resolver.detect_conflict(
                        law_name=adv_ex.target_law or "unknown",
                        domain=adv_ex.domain,
                        observation=adv_ex.observation,
                        prediction=adv_ex.expected_prediction,
                        description=adv_ex.challenge,
                    )
                    resolved = conflict_resolver.resolve(conflict)
                    if any(h.passed for h in resolved):
                        adv_gen.mark_resolved(adv_ex)
                        analyzer.log_event("adversarial_resolved",
                                           f"resolved: {adv_ex.type}",
                                           {"domain": adv_ex.domain})

            # 3d. Ciclo idle (modos cognitivos)
            idle_result = exec_.idle_cycle()

        # 4. SNAPSHOT
        if cycle % args.snapshot == 0:
            snap = analyzer.snapshot(cycle, exec_)

            # Inclui stats do law_synthesizer e adversarial
            law_stats = law_synthesizer.get_stats()
            adv_stats = adv_gen.get_stats()

            if mode_mgr.mode_history:
                lt = mode_mgr.mode_history[-1]
                analyzer.log_event("mode_transition",
                                   f"{lt.get('from', '?')} → {lt.get('to', '?')}")
            if exec_.history:
                recent = exec_.history[-1] if exec_.history else {}
                ig = recent.get("intrinsic_goals", [])
                if ig:
                    for g in ig:
                        analyzer.log_event("curiosity_goal", g.get("description", ""),
                                           {"drive": g.get("drive", "")})

            elapsed = time.time() - start_time
            total_agents = concept_registry.count
            print(f"  [{cycle}/{args.cycles}] {elapsed:.0f}s | "
                  f"mode={snap.current_mode} "
                  f"energy={snap.energy:.2f} "
                  f"surprise={snap.world_surprise:.2f} "
                  f"laws={snap.n_laws} "
                  f"agents={total_agents} "
                  f"compression={snap.compression_ratio:.2f} "
                  f"conflicts={snap.n_conflicts} "
                  f"adversarial={adv_stats['total_generated']} "
                  f"law_OK={law_stats['law_confirmed']} "
                  f"law_BAD={law_stats['law_violated']}")

    # ─── Relatorio Final ───
    elapsed = time.time() - start_time
    analyzer.snapshot(args.cycles, exec_)
    report = analyzer.generate_report()
    analyzer.print_summary(report)

    # Salva
    report_path = r"D:\novamind_data\traceability\autonomous_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatorio completo: {report_path}")
    print(f"Duracao total: {elapsed:.0f}s")

    # Print extra: law synthesizer + adversarial stats
    law_stats = law_synthesizer.get_stats()
    adv_stats = adv_gen.get_stats()
    print(f"\nLaw-Guided Synthesizer:")
    for k, v in law_stats.items():
        print(f"  {k}: {v}")
    print(f"\nAdversarial Reality Generator:")
    for k, v in adv_stats.items():
        print(f"  {k}: {v}")
    print(f"\nTotal agents: {concept_registry.count}")


if __name__ == "__main__":
    main()
