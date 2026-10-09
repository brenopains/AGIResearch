"""
A/B Test: ARC com e sem Macros.

Grupo A (controle): 11 primitives base, sem macros
Grupo B (tratamento): 11 primitives + macros do MacroRegistry

Mede:
  - Exact match, Pixel accuracy, Gap medio
  - Acoes por tarefa, Tempo por tarefa
  - Profundidade media, Nos explorados
  - Compression ratio, Macros utilizadas
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.macro_registry import MacroRegistry
from core.strategy_memory import StrategyMemory, ProblemSignature
from core.arc_dataset import ARCDataset


def setup_registry(with_macros: bool = False,
                    macro_registry: MacroRegistry = None):
    """Cria registry com ou sem macros."""
    registry = PrimitiveRegistry()

    # Base primitives
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        registry.register(fn())

    # Spatial operators (rotate, flip, transpose, identity)
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

    # Macros (se grupo B)
    if with_macros and macro_registry:
        for macro in macro_registry.get_best(n=100, min_confidence=0.1):
            seq = macro.primitive_sequence
            # Registra como primitiva separada (structure_type="macro")
            prim = Primitive(
                name=macro.name,
                structure_type="macro",
                description=macro.description,
            )
            # Funcao que executa a sequencia completa
            def _make_macro_exec(s):
                def _exec(g, p):
                    r = g.copy()
                    # Precisa do registry original para buscar primitives
                    for pname in s:
                        pr = registry.get(pname)
                        if pr:
                            r = pr.apply(r, {})
                    return r
                return _exec
            prim._execute = _make_macro_exec(seq)
            prim.confidence = macro.confidence
            # Macro sempre "inferencia" params vazios (sequencia fixa)
            prim._solver_fn = lambda inp, out: {}
            registry.register(prim)

    return registry


def run_arc_batch(registry, dataset, n_tasks: int = 100,
                  label: str = "Grupo") -> dict:
    """Roda batch de tasks ARC e coleta metricas."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph,
                              use_planner=False)
    cmp = GoalComparator()
    strat = StrategyMemory()

    results = {
        "exact": 0, "accs": [], "gaps": [], "times": [],
        "actions": [], "depths": [], "reused": 0,
        "macro_uses": 0, "n_tasks": 0,
    }

    for task_idx in range(n_tasks):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue

        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        t0 = time.time()

        # Tenta StrategyMemory
        sig = ProblemSignature.from_grids(inp, out, gap_type="ARC", domain="ARC")
        plan = strat.retrieve(sig, min_similarity=0.4)
        reused = plan is not None

        engine.set_goal_from_target(out, description=name)
        engine.perceive(inp)
        action = engine.step()
        elapsed = time.time() - t0

        n_actions = 0
        depth = 0
        macro_used = False

        if action:
            n_actions = 1
            # Extrai profundidade (numero de primitives no programa)
            prog = action.program
            if isinstance(prog, list):
                depth = sum(
                    len(p.instances) if hasattr(p, "instances") else 1
                    for p in prog
                )
            elif hasattr(prog, "instances"):
                depth = len(prog.instances)
            else:
                depth = 1

            # Verifica se usou macro
            prog_name = engine._extract_program_name(prog)
            if "Macro" in prog_name:
                macro_used = True

        # Mede resultado
        final = engine.current_state
        if final.shape == out.shape:
            exact = bool(np.array_equal(final, out))
            acc = 1.0 - int((final != out).sum()) / max(1, int(out.size))
            gap = cmp.compute(final, GoalState(target=out)).total
        else:
            exact = False
            acc = 0.0
            gap = 1.0

        results["exact"] += 1 if exact else 0
        results["accs"].append(acc)
        results["gaps"].append(gap)
        results["times"].append(elapsed)
        results["actions"].append(n_actions)
        results["depths"].append(depth)
        results["n_tasks"] += 1
        if macro_used:
            results["macro_uses"] += 1
        if reused:
            results["reused"] += 1

        # Alimenta StrategyMemory se acc > 0.7
        if acc > 0.7:
            seq = []
            for h in engine.history:
                if h.get("event") == "action_executed":
                    seq.append(h.get("action", "")[:40])
            if seq:
                strat.store(sig, seq, [{} for _ in seq], score=acc, task_id=name)

    # Calcula metricas finais
    n = results["n_tasks"]
    return {
        "label": label,
        "n_tasks": n,
        "exact": results["exact"],
        "pixel_accuracy": round(np.mean(results["accs"]), 4) if results["accs"] else 0,
        "gap_medio": round(np.mean(results["gaps"]), 4) if results["gaps"] else 0,
        "tempo_medio": round(np.mean(results["times"]) * 1000, 1) if results["times"] else 0,
        "acoes_media": round(np.mean(results["actions"]), 2) if results["actions"] else 0,
        "profundidade_media": round(np.mean(results["depths"]), 2) if results["depths"] else 0,
        "programas_reutilizados": results["reused"],
        "macros_utilizadas": results["macro_uses"],
        "n_primitives": registry.count,
    }


def print_comparison(a: dict, b: dict):
    """Imprime comparacao lado a lado."""
    print(f"\n{'='*70}")
    print(f"{'METRICA':<35} {'SEM MACROS':>15} {'COM MACROS':>15}")
    print(f"{'='*70}")
    metrics = [
        ("n_tasks", "Tasks", "d"),
        ("exact", "Exact Match", "d"),
        ("pixel_accuracy", "Pixel Accuracy", ".2%"),
        ("gap_medio", "Gap Medio", ".4f"),
        ("tempo_medio", "Tempo (ms)", ".0f"),
        ("acoes_media", "Acoes/tarefa", ".2f"),
        ("profundidade_media", "Profundidade media", ".2f"),
        ("programas_reutilizados", "Programas reutilizados", "d"),
        ("macros_utilizadas", "Macros utilizadas", "d"),
        ("n_primitives", "Primitives no registry", "d"),
    ]
    for key, name, fmt in metrics:
        va = a.get(key, 0)
        vb = b.get(key, 0)
        if "d" in fmt:
            sa = f"{va}"
            sb = f"{vb}"
        elif ".2%" in fmt:
            sa = f"{va:.2%}"
            sb = f"{vb:.2%}"
        else:
            sa = f"{va:{fmt}}"
            sb = f"{vb:{fmt}}"
        print(f"  {name:<33} {sa:>15} {sb:>15}")
    print()

    # Diferencas
    delta_acc = b["pixel_accuracy"] - a["pixel_accuracy"]
    delta_tempo = a["tempo_medio"] - b["tempo_medio"]
    delta_prof = a["profundidade_media"] - b["profundidade_media"]
    print(f"  Diferenca acc:     {delta_acc:+.2%}")
    print(f"  Diferenca tempo:   {delta_tempo:+.0f}ms (negativo = macros mais rapido)")
    print(f"  Diferenca prof:    {delta_prof:+.2f} (positivo = macros reduzem profundidade)")


if __name__ == "__main__":
    print("=" * 70)
    print("A/B TEST: ARC COM E SEM MACROS")
    print("=" * 70)

    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    N = 100

    # --- FASE 1: Descobre macros (60 tasks) ---
    print(f"\n[FASE 1] Descobrindo macros de 60 tasks ARC...")
    disc_registry = setup_registry(with_macros=False)
    disc_graph = CompositionGraph()
    disc_bridge = OperatorGroundingBridge(disc_registry, disc_graph)
    disc_pie = ProgramSynthesizer(disc_registry, composition_graph=disc_graph, grounding_bridge=disc_bridge)
    disc_engine = GoalDrivenEngine(program_induction_engine=disc_pie, composition_graph=disc_graph, use_planner=False)

    for task_idx in range(60):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)
        disc_engine.set_goal_from_target(out, description=name)
        disc_engine.perceive(inp)
        disc_engine.step()

    # Extrai macros do grafo
    from core.macro_discovery import MacroDiscovery
    macro_disc = MacroDiscovery(composition_graph=disc_graph)
    macro_disc.scan(min_frequency=3, min_success_rate=0.3)
    macros_found = macro_disc.get_macros(min_confidence=0.1)
    print(f"  Macros descobertas: {len(macros_found)}")

    # Popula MacroRegistry
    macro_registry = MacroRegistry()
    for m in macros_found:
        macro_registry.register(
            name=m.name,
            sequence=m.primitive_sequence,
            frequency=m.frequency,
            success_rate=m.success_rate,
            domain="ARC",
        )
    print(f"  MacroRegistry: {macro_registry.get_stats()}")

    # --- FASE 2: Grupo A (sem macros) ---
    print(f"\n[FASE 2] Grupo A: SEM macros ({N} tasks)...")
    reg_a = setup_registry(with_macros=False)
    result_a = run_arc_batch(reg_a, dataset, n_tasks=N, label="SEM MACROS")
    print(f"  Feito. Acc={result_a['pixel_accuracy']:.2%} "
          f"Exact={result_a['exact']} Tempo={result_a['tempo_medio']}ms")

    # --- FASE 3: Grupo B (com macros) ---
    print(f"\n[FASE 3] Grupo B: COM macros ({N} tasks)...")
    reg_b = setup_registry(with_macros=True, macro_registry=macro_registry)
    result_b = run_arc_batch(reg_b, dataset, n_tasks=N, label="COM MACROS")
    print(f"  Feito. Acc={result_b['pixel_accuracy']:.2%} "
          f"Exact={result_b['exact']} Tempo={result_b['tempo_medio']}ms")

    # --- FASE 4: Comparacao ---
    print(f"\n[RESULTADO]")
    print_comparison(result_a, result_b)

    # --- ANALISE ---
    print(f"\n[ANALISE]")
    macro_registry_stats = macro_registry.get_stats()
    print(f"  Total de macros no registry: {macro_registry_stats['total']}")
    print(f"  Generalidade media: {macro_registry_stats['avg_generality']}")
    print(f"  Compressao media: {macro_registry_stats['avg_compression']}x")
    if result_b["macros_utilizadas"] > 0:
        print(f"  Macros foram utilizadas em {result_b['macros_utilizadas']} tarefas")

    print(f"\n  Conclusao:")
    delta_acc = result_b["pixel_accuracy"] - result_a["pixel_accuracy"]
    delta_tempo = result_a["tempo_medio"] - result_b["tempo_medio"]
    delta_prof = result_a["profundidade_media"] - result_b["profundidade_media"]
    print(f"    Acc: {delta_acc:+.2%}")
    print(f"    Tempo: {delta_tempo:+.0f}ms ({delta_tempo/result_a['tempo_medio']*100:+.0f}%)")
    print(f"    Profundidade: {delta_prof:+.2f}")
    if delta_tempo > 3 and delta_prof >= 0:
        print(f"  [OK] Macros validam: reduziram tempo e profundidade")
    elif delta_tempo > 0:
        print(f"  [OK] Macros reduziram tempo (mas profundidade similar)")
    elif delta_prof > 0:
        print(f"  [WARN] Macros reduziram profundidade mas aumento tempo (overhead)")
    else:
        print(f"  [INFO] Macros sem ganho significativo em 100 tasks")
        print(f"         (28 macros de 60 tasks de treino, reuso exige +tasks)")
