"""
Teste definitivo: GoalDrivenEngine no ARC-AGI.

Uma AGI de verdade, dado um par input/output:
  1. Percebe o input (extrai objetos)
  2. Define o output como GOAL
  3. Computa o GAP entre input e output
  4. Induz PROGRAMAS para reduzir o gap
  5. Executa a melhor acao
  6. Verifica se o goal foi atingido
  7. Se nao, tenta novamente (multi-step)

O teste NAO e especializado em ARC. Usa o pipeline generico:
  GoalDrivenEngine -> ProgramInductionEngine -> CompositionGraph -> Pruning

Se passar, e porque o mecanismo de inducao + agencia e GENERICO.
Se falhar, o gap analysis mostra EXATAMENTE onde.
"""
import sys, numpy as np, json, time
sys.path.insert(0, ".")

from core.goal_driven_engine import (
    GoalDrivenEngine, GoalState, GoalComparator, ActionSelector,
)
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.arc_dataset import ARCDataset
from core.arc_object_centric import ARCGridParser


def run_arc_test(n_tasks: int = 50, output_path: str = "arc_goal_results.json"):
    """Roda o teste ARC-AGI com GoalDrivenEngine.

    Returns: dict com resultados detalhados.
    """
    print("=" * 70)
    print("ARC-AGI GOAL-DRIVEN TEST")
    print("=" * 70)

    # ─── Setup ───
    device = "cuda"
    registry = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        registry.register(fn())

    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph,
                             grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie,
                              composition_graph=graph)
    cmp = GoalComparator()
    parser = ARCGridParser()

    dataset = ARCDataset(data_dir="data/ARC/data/evaluation",
                         max_h=30, max_w=30, pad_val=10, dev=device)

    results = []
    t_start = time.time()

    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        if not train_pairs:
            continue

        t_task = time.time()

        # Usa o primeiro par de treino como exemplo
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        # ─── Goal-Driven Loop ───
        engine.set_goal_from_target(out, description=f"ARC {name}", priority=0.9)
        engine.perceive(inp)

        # Extrai objetos para analise
        input_objects = parser.extract_objects(inp)
        output_objects = parser.extract_objects(out)

        # Computa gap inicial
        initial_gap = cmp.compute(inp, GoalState(target=out))

        # Roda o loop (max 5 steps)
        achieved = engine.run(max_steps=5)

        # Computa gap final
        final_state = engine.current_state
        final_gap = cmp.compute(final_state, GoalState(target=out))
        exact_match = bool(np.array_equal(final_state, out)) if final_state.shape == out.shape else False

        # Pixel accuracy
        pixel_acc = 0.0
        if final_state.shape == out.shape:
            total = int(out.size)
            wrong = int((final_state != out).sum())
            pixel_acc = 1.0 - wrong / max(1, total)

        elapsed = time.time() - t_task

        result = {
            "task": name,
            "n_input_objects": len(input_objects),
            "n_output_objects": len(output_objects),
            "input_shape": list(inp.shape),
            "output_shape": list(out.shape),
            "initial_gap": initial_gap.total,
            "initial_gap_details": {
                "structural": initial_gap.structural_gap,
                "semantic": initial_gap.semantic_gap,
            },
            "final_gap": final_gap.total,
            "final_gap_details": {
                "structural": final_gap.structural_gap,
                "semantic": final_gap.semantic_gap,
            },
            "pixel_accuracy": round(pixel_acc, 4),
            "exact_match": exact_match,
            "goal_achieved": bool(achieved),
            "n_actions": engine.get_stats()["n_actions"],
            "time_ms": round(elapsed * 1000, 1),
        }
        results.append(result)

        # Log
        status = "OK" if exact_match else "PARTIAL" if pixel_acc > 0.5 else "FAIL"
        print(f"  [{task_idx+1:3d}/{n_tasks}] {name}: gap={initial_gap.total:.3f}->{final_gap.total:.3f} "
              f"acc={pixel_acc:.2%} actions={result['n_actions']} {status}")

    total_time = time.time() - t_start

    # ─── Summary ───
    n_exact = sum(1 for r in results if r["exact_match"])
    n_goal = sum(1 for r in results if r["goal_achieved"])
    n_some_progress = sum(1 for r in results if r["pixel_accuracy"] > 0.5)
    avg_gap_reduction = np.mean([r["initial_gap"] - r["final_gap"] for r in results]) if results else 0
    avg_acc = np.mean([r["pixel_accuracy"] for r in results]) if results else 0
    avg_actions = np.mean([r["n_actions"] for r in results]) if results else 0

    # Analise de gargalos
    gaps_by_type = {}
    for r in results:
        # Classifica o tipo de gap predominante
        if r["exact_match"]:
            t = "SOLVED"
        elif r["final_gap"] < 0.3:
            t = "NEAR_MISS"
        elif r["initial_gap"] == r["final_gap"]:
            t = "NO_PROGRESS"
        elif r["pixel_accuracy"] > 0.5:
            t = "PARTIAL"
        else:
            t = "STRUGGLING"
        gaps_by_type[t] = gaps_by_type.get(t, 0) + 1

    summary = {
        "n_tasks": len(results),
        "exact_match": n_exact,
        "goal_achieved": n_goal,
        "some_progress": n_some_progress,
        "avg_pixel_accuracy": round(avg_acc, 4),
        "avg_gap_reduction": round(avg_gap_reduction, 4),
        "avg_actions": round(avg_actions, 2),
        "total_time_s": round(total_time, 1),
        "gap_types": gaps_by_type,
        "composition_graph": graph.get_stats(),
        "n_primitives": registry.count,
    }

    # ─── Detail: top 10 hardest tasks ───
    hardest = sorted(results, key=lambda r: r["final_gap"] - r["initial_gap"])[:10]

    print(f"\n{'='*70}")
    print(f"RESUMO: {summary['n_tasks']} tasks ARC")
    print(f"{'='*70}")
    print(f"  Exact match:     {n_exact}/{len(results)} ({n_exact/max(1,len(results))*100:.1f}%)")
    print(f"  Goal achieved:   {n_goal}/{len(results)} ({n_goal/max(1,len(results))*100:.1f}%)")
    print(f"  Algum progresso: {n_some_progress}/{len(results)} ({n_some_progress/max(1,len(results))*100:.1f}%)")
    print(f"  Pixel acc media: {avg_acc:.2%}")
    print(f"  Reducao de gap:  {avg_gap_reduction:.3f}")
    print(f"  Acoes por task:  {avg_actions:.1f}")
    print(f"  Tempo total:     {summary['total_time_s']:.1f}s")
    print(f"\n  Distribuicao de gargalos:")
    for t, c in sorted(gaps_by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:15s}: {c:3d} ({c/max(1,len(results))*100:.0f}%)")

    print(f"\n  CompositionGraph: {graph.get_stats()}")
    print(f"\n  Top 10 tarefas mais dificeis:")
    for r in hardest[:5]:
        print(f"    {r['task']}: gap {r['initial_gap']:.3f} -> {r['final_gap']:.3f} "
              f"acc={r['pixel_accuracy']:.2%} objs={r['n_input_objects']}->{r['n_output_objects']}")

    # Salva
    output = {
        "summary": summary,
        "results": results,
        "hardest": hardest[:10],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\nResultados salvos: {output_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=50, help="Numero de tasks ARC")
    parser.add_argument("--output", type=str, default="arc_goal_results.json")
    args = parser.parse_args()
    run_arc_test(n_tasks=args.tasks, output_path=args.output)
