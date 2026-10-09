"""
Pipeline de experiencia em massa — alimenta o sistema com centenas de tasks,
acumula composicoes, descobre macros, cria skills, e mede evolucao.

Ciclo:
  1. Roda 400 tasks ARC pelo GoalDrivenEngine (coleta composicoes)
  2. CompositionGraph acumula edges (aprende quais primitives compoem bem)
  3. MacroDiscovery escaneia o grafo por n-grams frequentes
  4. SkillEngine agrupa macros por intencao (estado intermediario)
  5. StrategyMemory armazena planos que funcionaram
  6. Re-testa ARC com registry expandido
  7. Relatorio: evolucao de composicoes, macros, skills, accuracy
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.strategy_memory import StrategyMemory, ProblemSignature
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.macro_discovery import MacroDiscovery
from core.macro_registry import MacroRegistry
from core.skill_engine import SkillDiscovery, SkillRegistry
from core.arc_dataset import ARCDataset

RESULTS_DIR = "D:/novamind_data/experience"
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_registry():
    registry = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        registry.register(fn())
    for op in get_all_proposals():
        fn = op["function"]
        prim = Primitive(name=op["name"], structure_type=op["name"],
                         description=op.get("description", ""))
        def _me(f):
            def _e(g, p):
                try: return f(g)
                except: return g.copy()
            return _e
        prim._execute = _me(fn)
        registry.register(prim)
    return registry


def run_batch(dataset, registry, graph, strat, n_tasks: int) -> dict:
    """Processa N tasks, alimenta grafo e memoria de estrategias."""
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()

    results = {"exact": 0, "accs": [], "gaps": [], "compositions": [], "strategies": 0}

    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        engine.set_goal_from_target(out, description=name)
        engine.perceive(inp)
        engine.step()
        final = engine.current_state

        exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
        acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0
        gap = cmp.compute(final, GoalState(target=out)).total if final.shape == out.shape else 1.0

        results["exact"] += 1 if exact else 0
        results["accs"].append(acc)
        results["gaps"].append(gap)

        # StrategyMemory: armazena se acc > 0.7
        if acc > 0.7:
            sig = ProblemSignature.from_grids(inp, out, gap_type="ARC", domain="ARC")
            seq = []
            for h in engine.history:
                if h.get("event") == "action_executed":
                    seq.append(h.get("action", "")[:40])
            if seq:
                strat.store(sig, seq, [{} for _ in seq], score=acc, task_id=name)
                results["strategies"] += 1

    results["avg_acc"] = round(np.mean(results["accs"]), 4) if results["accs"] else 0
    results["avg_gap"] = round(np.mean(results["gaps"]), 4) if results["gaps"] else 0
    results["n_tasks"] = len(results["accs"])
    return results


def discover_macros(graph) -> MacroRegistry:
    """Descobre macros do CompositionGraph."""
    disc = MacroDiscovery(composition_graph=graph)
    disc.scan(min_frequency=3, min_success_rate=0.3)
    macros = disc.get_macros(min_confidence=0.1)

    macro_reg = MacroRegistry()
    for m in macros:
        macro_reg.register(
            name=m.name, sequence=m.primitive_sequence,
            frequency=m.frequency, success_rate=m.success_rate,
            domain="ARC",
        )
    return macro_reg


def discover_skills(macro_reg: MacroRegistry) -> SkillRegistry:
    """Descobre skills de macros."""
    seqs = [m.primitive_sequence for m in macro_reg.get_best(n=100, min_confidence=0.0)]
    disc = SkillDiscovery()
    return disc.discover(seqs, domain="ARC")


def apply_macros_to_registry(registry, macro_reg):
    """Adiciona macros ao registry como primitives executaveis."""
    n_added = 0
    for macro in macro_reg.get_best(n=50, min_confidence=0.3):
        if registry.get(macro.name):
            continue
        seq = macro.primitive_sequence
        prim = Primitive(name=macro.name, structure_type="macro", description=macro.description)
        def _make_exec(s):
            def _exec(g, p):
                r = g.copy()
                for pname in s:
                    pr = registry.get(pname)
                    if pr:
                        r = pr.apply(r, {})
                return r
            return _exec
        prim._execute = _make_exec(seq)
        prim.confidence = macro.confidence
        prim._solver_fn = lambda inp, out: {}
        registry.register(prim)
        n_added += 1
    return n_added


def apply_skills_to_registry(registry, skill_reg):
    """Adiciona skills ao registry."""
    n_added = 0
    for skill in skill_reg.get_best(n=20, min_confidence=0.3):
        if registry.get(skill.name):
            continue
        prim = Primitive(name=skill.name, structure_type="skill", description=skill.description)
        # Skill escolhe a melhor implementacao no momento da execucao
        def _make_skill_exec(sk):
            def _exec(g, p):
                return sk.execute(g, registry)
            return _exec
        prim._execute = _make_skill_exec(skill)
        prim.confidence = skill.confidence
        prim._solver_fn = lambda inp, out: {}
        registry.register(prim)
        n_added += 1
    return n_added


def evaluate(registry, dataset, n_tasks: int = 100, label: str = "") -> dict:
    """Avalia performance no ARC."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()

    exacts, accs, gaps = 0, [], []
    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)
        engine.set_goal_from_target(out, description=name)
        engine.perceive(inp)
        engine.step()
        final = engine.current_state
        exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
        acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0
        gap = cmp.compute(final, GoalState(target=out)).total if final.shape == out.shape else 1.0
        exacts += 1 if exact else 0
        accs.append(acc)
        gaps.append(gap)

    return {
        "label": label,
        "n_tasks": len(accs),
        "exact": exacts,
        "avg_acc": round(np.mean(accs), 4) if accs else 0,
        "avg_gap": round(np.mean(gaps), 4) if gaps else 0,
        "n_primitives": registry.count,
    }


def main():
    print("=" * 70)
    print("PIPELINE DE EXPERIENCIA EM MASSA")
    print("=" * 70)

    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

    # ─── FASE 0: Baseline ───
    print("\n[FASE 0] Baseline (11 primitives, sem experiencia)")
    reg_base = setup_registry()
    baseline = evaluate(reg_base, dataset, n_tasks=100, label="BASELINE")
    print(f"  Acc: {baseline['avg_acc']:.2%}  Exact: {baseline['exact']}  Primitives: {baseline['n_primitives']}")

    # ─── FASE 1: Experiencia em massa (400 tasks) ───
    print("\n[FASE 1] Coletando experiencia (400 tasks ARC)...")
    graph = CompositionGraph()
    strat = StrategyMemory()
    registry = setup_registry()

    # Divide em batches para medir progresso
    batch_size = 100
    milestones = []

    for batch_start in range(0, 400, batch_size):
        batch_n = min(batch_size, 400 - batch_start)
        print(f"\n  Batch {batch_start//batch_size + 1}: tasks {batch_start}-{batch_start+batch_n}...")

        result = run_batch(dataset, registry, graph, strat, n_tasks=batch_n)

        # Macro discovery apos cada batch
        macro_reg = discover_macros(graph)
        skill_reg = discover_skills(macro_reg)

        # Conta composicoes e estrategias
        gs = graph.get_stats()
        ms = macro_reg.get_stats()
        ss = skill_reg.get_stats()

        milestone = {
            "batch": batch_start // batch_size + 1,
            "tasks": batch_n,
            "graph_edges": gs.get("n_edges", 0),
            "graph_nodes": gs.get("n_nodes", 0),
            "macros": ms.get("total", 0),
            "skills": ss.get("total", 0),
            "strategies": result.get("strategies", 0),
            "avg_acc": result["avg_acc"],
            "exact": result["exact"],
        }
        milestones.append(milestone)
        print(f"    Acc: {milestone['avg_acc']:.2%}  "
              f"Graph: {milestone['graph_edges']} edges  "
              f"Macros: {milestone['macros']}  "
              f"Skills: {milestone['skills']}")

    # ─── FASE 2: Macro + Skill Discovery final ───
    print("\n[FASE 2] Descoberta final de macros e skills...")
    macro_reg = discover_macros(graph)
    skill_reg = discover_skills(macro_reg)

    macros_final = macro_reg.get_stats()
    skills_final = skill_reg.get_stats()
    print(f"  Macros: {macros_final}")
    print(f"  Skills: {skills_final}")

    # ─── FASE 3: Expandir registry com macros + skills ───
    print("\n[FASE 3] Expandindo registry...")
    n_macros = apply_macros_to_registry(registry, macro_reg)
    n_skills = apply_skills_to_registry(registry, skill_reg)
    print(f"  +{n_macros} macros, +{n_skills} skills = {registry.count} primitives totais")
    for p in registry.get_all():
        if p.structure_type in ("macro", "skill"):
            print(f"    [{p.structure_type.upper()}] {p.name}: {p.description[:50]}")

    # ─── FASE 4: Re-avaliacao ───
    print("\n[FASE 4] Re-avaliando ARC com registry expandido...")
    final = evaluate(registry, dataset, n_tasks=100, label="POS_EXPERIENCIA")

    # ─── RESUMO ───
    print("\n" + "=" * 70)
    print("RESUMO DA EVOLUCAO")
    print("=" * 70)
    print(f"\n  {'Metrica':<30s} {'Antes':>12s} {'Depois':>12s}")
    print(f"  {'-'*55}")
    print(f"  {'Pixel Accuracy':<30s} {baseline['avg_acc']:.2%}     {final['avg_acc']:.2%}")
    print(f"  {'Exact Match':<30s} {baseline['exact']}           {final['exact']}")
    print(f"  {'Primitives no registry':<30s} {baseline['n_primitives']}           {final['n_primitives']}")

    # Milestones
    print(f"\n  Milestones por batch:")
    for m in milestones:
        print(f"    Batch {m['batch']}: graph={m['graph_edges']} edges, "
              f"{m['macros']} macros, {m['skills']} skills, acc={m['avg_acc']:.2%}")

    # Salva resultados
    report = {
        "baseline": baseline,
        "final": final,
        "milestones": milestones,
        "graph": graph.get_stats(),
        "macros": macros_final,
        "skills": skills_final,
    }
    path = os.path.join(RESULTS_DIR, "experience_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\n  Relatorio salvo: {path}")

    # Amostra de macros encontradas
    print(f"\n  Amostra de macros ({macros_final.get('total', 0)} totais):")
    for m in macro_reg.get_best(n=5, min_confidence=0.3):
        print(f"    {m.name:30s} {' -> '.join(m.primitive_sequence):40s} conf={m.confidence:.2f}")

    print(f"\n  Amostra de skills ({skills_final.get('total', 0)} totais):")
    for s in skill_reg.get_best(n=5, min_confidence=0.3):
        print(f"    {s.name:30s} intent={s.intent:30s} impls={len(s.implementations)}")


if __name__ == "__main__":
    main()
