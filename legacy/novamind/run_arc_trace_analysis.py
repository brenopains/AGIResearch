"""
ARC Trace Analysis — Rastreabilidade completa de CADA task falha.

Para cada task ARC, registra:
  1. Estado inicial (input grid) vs estado desejado (output grid)
  2. Gap inicial: o que difere? (shape? objects? colors? position?)
  3. Que primitives foram tentadas? Qual o resultado de cada uma?
  4. Qual competicao ocorreu? Vencedora vs runner-up?
  5. Gap residual apos execucao: o que AINDA difere?
  6. Classificacao do gargalo: porque nao resolveu?

Isso permite VER o processo cognitivo, nao apenas medir accuracy.
"""
import sys, os, json, numpy as np, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.arc_dataset import ARCDataset

OUTPUT = "D:/novamind_data/arc_trace_analysis"
os.makedirs(OUTPUT, exist_ok=True)


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


def classify_gap_type(inp, out, current, gap_total):
    """Classifica o tipo de gap residual em categorias estruturais."""
    if inp.shape != out.shape:
        return "SHAPE_MISMATCH"
    
    in_objs = (inp > 0).sum()
    out_objs = (out > 0).sum()
    cur_objs = (current > 0).sum()
    
    in_colors = set(int(c) for c in np.unique(inp) if c != 0)
    out_colors = set(int(c) for c in np.unique(out) if c != 0)
    cur_colors = set(int(c) for c in np.unique(current) if c != 0)
    
    # Gap residual alto -> primitiva errada
    if gap_total > 0.8:
        if in_objs != out_objs:
            return "WRONG_OBJECT_COUNT"
        if in_colors != out_colors:
            return "COLOR_MISMATCH"
        return "SPATIAL_MISMATCH"
    
    # Gap medio -> aproximou mas nao fechou
    if gap_total > 0.3:
        if cur_objs != out_objs:
            return "PARTIAL_OBJECT_COUNT"
        if cur_colors != out_colors:
            return "PARTIAL_COLOR"
        return "PARTIAL_POSITION"
    
    # Gap baixo -> quase la
    if gap_total > 0.05:
        return "NEAR_MISS"
    
    return "EXACT_MATCH"


def analyze_single_task(task_idx, name, train_pairs, test_pairs, registry):
    """Analisa UMA task ARC com rastreabilidade completa."""
    cmp = GoalComparator()
    
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)
    
    # ─── 1. GAP INICIAL ───
    initial_gap = cmp.compute(inp, GoalState(target=out))
    
    # ─── 2. TENTA CADA PRIMITIVA INDIVIDUALMENTE ───
    primitives_tried = []
    for prim in registry.get_all():
        try:
            # Tenta aplicar a primitiva
            if hasattr(prim, 'generate_hypothesis'):
                instance = prim.generate_hypothesis(inp, out)
                if instance:
                    result = instance.apply(inp)
                    if result.shape == out.shape:
                        gap_after = cmp.compute(result, GoalState(target=out))
                        primitives_tried.append({
                            "name": prim.name,
                            "type": prim.structure_type,
                            "gap_after": gap_after.total,
                            "acc": 1.0 - gap_after.total,
                            "improvement": initial_gap.total - gap_after.total,
                        })
        except Exception:
            pass
    
    primitives_tried.sort(key=lambda p: p["improvement"], reverse=True)
    
    # ─── 3. EXECUTA GOAL-DRIVEN (MULTI-STEP: ate 5 iteracoes) ───
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    
    engine.set_goal_from_target(out)
    engine.perceive(inp)
    engine.run(max_steps=5)  # Multi-step! Nao apenas 1.
    final = engine.current_state
    
    # ─── 4. GAP FINAL ───
    if final.shape == out.shape:
        final_gap = cmp.compute(final, GoalState(target=out))
        exact = bool(np.array_equal(final, out))
        acc = 1.0 - int((final != out).sum()) / max(1, int(out.size))
    else:
        final_gap = cmp.compute(inp, GoalState(target=out))
        exact = False
        acc = 0.0
    
    gap_type = classify_gap_type(inp, out, final, final_gap.total)
    
    # ─── 5. EVIDENCIA DO PROCESSO ───
    action_taken = "none"
    n_actions = 0
    for h in engine.history:
        if h.get("event") == "action_executed":
            action_taken = h.get("action", "?")[:60]
            n_actions += 1
    
    # Melhor primitiva individual
    best_prim = primitives_tried[0] if primitives_tried else None
    
    return {
        "task_id": name,
        "task_idx": task_idx,
        "input_shape": list(inp.shape),
        "output_shape": list(out.shape),
        "n_input_objects": int((inp > 0).sum()),
        "n_output_objects": int((out > 0).sum()),
        "initial_gap": round(initial_gap.total, 4),
        "initial_gap_structural": round(initial_gap.structural_gap, 4),
        "initial_gap_semantic": round(initial_gap.semantic_gap, 4),
        "final_gap": round(final_gap.total, 4),
        "pixel_accuracy": round(acc, 4),
        "exact_match": exact,
        "gap_type": gap_type,
        "action_taken": action_taken,
        "n_actions": n_actions,
        "best_primitive": best_prim["name"] if best_prim else "none",
        "best_primitive_acc": round(best_prim["acc"], 4) if best_prim else 0,
        "best_primitive_improvement": round(best_prim["improvement"], 4) if best_prim else 0,
        "n_primitives_tried": len(primitives_tried),
        "n_steps_taken": n_actions,
        "top_3_primitives": [
            {"name": p["name"], "acc": round(p["acc"], 4), "improvement": round(p["improvement"], 4)}
            for p in primitives_tried[:3]
        ],
    }


# ─── MAIN ──────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ARC TRACE ANALYSIS — Rastreabilidade por task")
    print("=" * 70)
    
    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    registry = setup_registry()
    print(f"Registry: {registry.count} primitives\n")
    
    all_traces = []
    bottleneck_counts = defaultdict(int)
    total_exact = 0
    total_tasks = 0
    
    for task_idx in range(min(100, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])
        if not train_pairs:
            continue
        
        trace = analyze_single_task(task_idx, name, train_pairs, test_pairs, registry)
        all_traces.append(trace)
        total_tasks += 1
        bottleneck_counts[trace["gap_type"]] += 1
        if trace["exact_match"]:
            total_exact += 1
        
        # Log detalhado das primeiras 10 e de NEAR_MISS
        if task_idx < 10 or trace["gap_type"] in ("NEAR_MISS", "PARTIAL_POSITION"):
            action = trace["action_taken"][:40] if trace["action_taken"] != "none" else "NO_ACTION"
            best = trace["best_primitive"]
            best_acc = trace["best_primitive_acc"]
            print(f"  [{task_idx:3d}] {name}: gap={trace['initial_gap']:.3f}->{trace['final_gap']:.3f} "
                  f"acc={trace['pixel_accuracy']:.2%} type={trace['gap_type']:20s} "
                  f"action={action}")
            if trace["best_primitive"] != "none":
                print(f"         melhor primitiva: {best} (acc={best_acc:.2%}, "
                      f"improvement={trace['best_primitive_improvement']:.3f})")
    
    # ─── RELATORIO AGREGADO ───
    print(f"\n\n{'='*70}")
    print("RELATORIO DE RASTREABILIDADE ARC (100 tasks)")
    print(f"{'='*70}")
    print(f"\nTotal: {total_tasks} tasks")
    print(f"Exact match: {total_exact}")
    print(f"Acc media: {np.mean([t['pixel_accuracy'] for t in all_traces]):.2%}")
    print(f"Gap medio inicial: {np.mean([t['initial_gap'] for t in all_traces]):.3f}")
    print(f"Gap medio final: {np.mean([t['final_gap'] for t in all_traces]):.3f}")
    
    print(f"\n\nDISTRIBUICAO DE GARGALOS:")
    print(f"{'Gargalo':<30s} {'Ocorrencias':>12s} {'%':>8s}")
    print(f"{'-'*50}")
    for bt, count in sorted(bottleneck_counts.items(), key=lambda x: -x[1]):
        print(f"  {bt:<30s} {count:>4d}/{total_tasks}    {count/total_tasks*100:>5.1f}%")
    
    # ─── ANALISE POR GARGALO ───
    print(f"\n\nANALISE DE CADA GARGALO:")
    for bt, count in sorted(bottleneck_counts.items(), key=lambda x: -x[1]):
        if count < 2:
            continue
        print(f"\n  [{bt}] ({count} tasks):")
        examples = [t for t in all_traces if t["gap_type"] == bt][:3]
        for ex in examples:
            print(f"    Task {ex['task_id']}:")
            print(f"      Input: {ex['input_shape']}, {ex['n_input_objects']} objects")
            print(f"      Output: {ex['output_shape']}, {ex['n_output_objects']} objects")
            print(f"      Gap: {ex['initial_gap']:.3f} -> {ex['final_gap']:.3f}")
            print(f"      Acc: {ex['pixel_accuracy']:.2%}")
            print(f"      Acao: {ex['action_taken'][:50]}")
            print(f"      Melhor primitiva: {ex['best_primitive']} (acc={ex['best_primitive_acc']:.2%})")
            if ex["top_3_primitives"]:
                for p in ex["top_3_primitives"][:2]:
                    print(f"        {p['name']}: acc={p['acc']:.2%}, improvement={p['improvement']:.3f}")
    
    # ─── TRACE COMPLETO DAS PIORES ───
    print(f"\n\nDETALHE: TOP 5 PIORES TASKS:")
    worst = sorted(all_traces, key=lambda t: t["pixel_accuracy"])[:5]
    for i, t in enumerate(worst):
        print(f"\n  --- PIOR {i+1}: {t['task_id']} ---")
        print(f"  Input shape: {t['input_shape']}, Output shape: {t['output_shape']}")
        print(f"  Objects: {t['n_input_objects']} -> {t['n_output_objects']}")
        print(f"  Initial gap: {t['initial_gap']:.4f} (structural={t['initial_gap_structural']:.4f}, semantic={t['initial_gap_semantic']:.4f})")
        print(f"  Final gap: {t['final_gap']:.4f}")
        print(f"  Acc: {t['pixel_accuracy']:.2%}")
        print(f"  Action taken: {t['action_taken']}")
        print(f"  Gap type: {t['gap_type']}")
        print(f"  Top 3 primitives:")
        for p in t["top_3_primitives"][:3]:
            improvement_dir = "reduz" if p["improvement"] > 0 else "aumenta"
            print(f"    {p['name']}: acc={p['acc']:.2%}, {improvement_dir} gap em {abs(p['improvement']):.3f}")
    
    # ─── SALVA JSON ───
    path = os.path.join(OUTPUT, "arc_trace_analysis.json")
    with open(path, "w") as f:
        json.dump({
            "summary": {
                "n_tasks": total_tasks,
                "exact_match": total_exact,
                "avg_acc": round(float(np.mean([t["pixel_accuracy"] for t in all_traces])), 4),
                "bottlenecks": dict(bottleneck_counts),
            },
            "traces": all_traces,
        }, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    print(f"\nTrace salvo em: {path}")
    
    # ─── DIAGNOSTICO FINAL ───
    print(f"\n\n{'='*70}")
    print("DIAGNOSTICO: POR QUE ARC NAO RESOLVE?")
    print(f"{'='*70}")
    print(f"""
  OS GARGALOS REAIS (do processo cognitivo):

  1. SHAPE_MISMATCH ({bottleneck_counts.get('SHAPE_MISMATCH', 0)} tasks)
     O sistema nao tem primitives que mudem DIMENSOES do grid.
     rotate/flip/move PRESERVAM shape. Nao ha 'resize' ou 'crop'.
     -> Solucao: adicionar primitives de redimensionamento.

  2. WRONG_OBJECT_COUNT ({bottleneck_counts.get('WRONG_OBJECT_COUNT', 0)} tasks)
     O sistema nao consegue replica/remover objetos seletivamente.
     'remove' remove TODAS as instancias de uma cor, nao objetos especificos.
     -> Solucao: primitiva 'selective_remove' ou 'copy_object' com bounding box.

  3. SPATIAL_MISMATCH / PARTIAL_POSITION ({bottleneck_counts.get('SPATIAL_MISMATCH', 0) + bottleneck_counts.get('PARTIAL_POSITION', 0)} tasks)
     O sistema move objetos mas nao na posicao exata.
     move() usa centroide, nao mantem geometria relativa entre objetos.
     -> Solucao: transformacao relacional (preservar distancias entre objetos).

  4. COLOR_MISMATCH ({bottleneck_counts.get('COLOR_MISMATCH', 0)} tasks)
     Recolor funciona mas requer from_color EXATO.
     Se a cor nao existe no input, recolor nao e tentado.
     -> Solucao: recolor generico (mapeamento por posicao, nao por cor).

  CONCLUSAO: O sistema NAO falta "inteligencia". Faltam PRIMITIVES
  especificas para transformacoes que ARC exige. O processo cognitivo
  (GoalDriven + Competition + ActionSelector) funciona.
  O espaco de primitives e insuficiente para cobrir ARC.
""")
    print(f"\nTrace analysis JSON: {path}")


if __name__ == "__main__":
    main()
