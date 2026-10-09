"""
Descobre primitives de escala/dimensao do dataset generico e re-testa ARC.

Fluxo:
  1. Gera dataset de transformacoes de dimensao (tile, crop, resize, extract_bbox, pad)
  2. Alimenta PrimitiveDiscovery com os exemplos
  3. Sistema descobre novas primitives (sem hardcoding)
  4. Adiciona ao registry
  5. Re-testa ARC
  6. Mede: SHAPE_MISMATCH caiu? Acc subiu?
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
    PrimitiveDiscovery,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive
from core.arc_dataset import ARCDataset
from core.scale_transforms_dataset import ScaleTransformDataset

RESULTS = "D:/novamind_data/scale_discovery"
os.makedirs(RESULTS, exist_ok=True)


def setup_base_registry():
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


def run_arc(registry, dataset, n_tasks=100):
    """Roda ARC e retorna metricas."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()
    
    exacts, accs, shapes = 0, [], 0
    for idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[idx]
        train = task.get("train", [])
        if not train: continue
        inp = np.array(train[0]["input"], dtype=int)
        out = np.array(train[0]["output"], dtype=int)
        engine.set_goal_from_target(out)
        engine.perceive(inp)
        engine.run(max_steps=5)
        final = engine.current_state
        if final.shape == out.shape:
            exacts += 1 if np.array_equal(final, out) else 0
            accs.append(1.0 - int((final != out).sum()) / max(1, int(out.size)))
        else:
            shapes += 1
    avg_acc = np.mean(accs) if accs else 0
    return {"exact": exacts, "acc": round(avg_acc, 4), "shape_mismatch": shapes, "n": len(accs)}


if __name__ == "__main__":
    print("=" * 70)
    print("SCALE TRANSFORM SELF-DISCOVERY")
    print("Dataset generico -> PrimitiveDiscovery -> ARC retest")
    print("=" * 70)
    
    registry = setup_base_registry()
    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    
    # ─── 1. BASELINE ───
    print(f"\n[1] Baseline ARC com {registry.count} primitives...")
    base = run_arc(registry, dataset)
    print(f"  Acc: {base['acc']:.2%}  Exact: {base['exact']}  ShapeMis: {base['shape_mismatch']}")
    
    # ─── 2. GERA DATASET DE ESCALA ───
    print(f"\n[2] Gerando dataset de transformacoes de dimensao...")
    scale_data = ScaleTransformDataset(size=2000, max_h=10, max_w=10)
    dist = scale_data.get_label_distribution()
    print(f"  Total: {len(scale_data)} exemplos")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")
    
    # ─── 3. ALIMENTA PRIMITIVE DISCOVERY ───
    print(f"\n[3] Alimentando PrimitiveDiscovery...")
    discovery = PrimitiveDiscovery(registry=registry)
    n_before = registry.count
    
    for idx in range(len(scale_data)):
        inp, out, label = scale_data[idx]
        discovery.observe(inp, out)
    
    n_after = registry.count
    n_new = n_after - n_before
    
    print(f"  Novas primitives descobertas: {n_new}")
    if n_new > 0:
        new_prims = list(registry.get_all())[-n_new:]
        for p in new_prims:
            print(f"    {p.name}: {p.description[:60] if p.description else 'generic'}")
    
    print(f"  Registry: {n_before} -> {n_after}")
    
    # Prime descobrir se as novas primitives TÊM _execute function
    print(f"\n  Verificando se as novas primitives sao executaveis...")
    for p in registry.get_all():
        if p.structure_type not in ["recolor", "remove", "move", "add", 
                                      "rotate_90", "rotate_180", "rotate_270",
                                      "flip_horizontal", "flip_vertical", "transpose", "identity"]:
            has_exec = hasattr(p, '_execute') and p._execute is not None
            print(f"    {p.name}: type={p.structure_type}, has_execute={has_exec}")
    
    # ─── 4. RETESTA ARC ───
    print(f"\n[4] Retest ARC com {registry.count} primitives...")
    pos = run_arc(registry, dataset)
    
    print(f"\n  Acc: {pos['acc']:.2%}  Exact: {pos['exact']}  ShapeMis: {pos['shape_mismatch']}")
    
    # ─── 5. RELATORIO ───
    delta_acc = pos["acc"] - base["acc"]
    delta_shape = base["shape_mismatch"] - pos["shape_mismatch"]
    
    print(f"\n{'='*70}")
    print("RELATORIO FINAL")
    print(f"{'='*70}")
    print(f"\n{'Metrica':<30s} {'Antes':>12s} {'Depois':>12s} {'Delta':>12s}")
    print(f"{'-'*66}")
    print(f"  {'Pixel Acc':<30s} {base['acc']:.2%}     {pos['acc']:.2%}     {delta_acc:+.2%}")
    print(f"  {'Exact Match':<30s} {base['exact']:>6d}      {pos['exact']:>6d}      {pos['exact']-base['exact']:>+6d}")
    print(f"  {'SHAPE_MISMATCH':<30s} {base['shape_mismatch']:>6d}      {pos['shape_mismatch']:>6d}      {delta_shape:>+6d}")
    print(f"  {'Primitives':<30s} {n_before:>6d}      {n_after:>6d}      +{n_new}")
    
    print(f"\n  Diagnostico:")
    if delta_shape > 0:
        print(f"    SHAPE_MISMATCH reduziu em {delta_shape} tasks ({delta_shape/base['shape_mismatch']*100:.0f}%)")
    else:
        print(f"    SHAPE_MISMATCH nao mudou — as novas primitives precisam ser testadas")
    
    if delta_acc > 0.01:
        print(f"    Acc subiu {delta_acc:.2%} — melhoria modesta mas positiva")
    else:
        print(f"    Acc estavel — pode ser que as novas primitives precisem de solver melhor")
    
    # Salva
    report = {"baseline": base, "pos": pos, "n_new": n_new}
    with open(os.path.join(RESULTS, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRelatorio salvo em: {RESULTS}/report.json")
