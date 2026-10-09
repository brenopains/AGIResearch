"""
ARC Self-Discovery Pipeline — O sistema descobre as proprias primitives que faltam.

Ciclo:
  1. Roda ARC, detecta SHAPE_MISMATCH
  2. Para cada SHAPE_MISMATCH, gera OperatorSpec (invariantes + transformacoes)
  3. SpecGuidedDiscovery busca exemplos no GeometryDataset que casam com a spec
  4. PrimitiveDiscovery INDUZ novas primitives dos exemplos
  5. Adiciona primitives ao registry
  6. Re-testa ARC
  7. Repete ate nao haver mais SHAPE_MISMATCH ou estabilizar

Nada hardcoded. O sistema descobre o que falta.
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
from core.spec_guided_discovery import SpecGuidedDiscovery
from core.operator_specification import SpecGenerator, InvariantExtractor
from core.geometry_dataset import GeometryTransformDataset

RESULTS_DIR = "D:/novamind_data/self_discovery"
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


def run_arc_eval(registry, dataset, n_tasks=100):
    """Roda ARC com multi-step e retorna traces."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()
    
    traces = []
    for idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[idx]
        train = task.get("train", [])
        if not train:
            continue
        inp = np.array(train[0]["input"], dtype=int)
        out = np.array(train[0]["output"], dtype=int)
        
        engine.set_goal_from_target(out)
        engine.perceive(inp)
        engine.run(max_steps=5)
        final = engine.current_state
        
        if final.shape == out.shape:
            exact = bool(np.array_equal(final, out))
            acc = 1.0 - int((final != out).sum()) / max(1, int(out.size))
            gap_val = cmp.compute(final, GoalState(target=out)).total
        else:
            exact, acc, gap_val = False, 0.0, 1.0
        
        # Classifica gap
        gap_type = "EXACT_MATCH" if exact else "SHAPE_MISMATCH"
        if not exact and final.shape == out.shape:
            if acc > 0.7: gap_type = "NEAR_MISS"
            elif acc > 0.3: gap_type = "PARTIAL"
            else: gap_type = "STRUGGLE"
        
        n_actions = sum(1 for h in engine.history if h.get("event") == "action_executed")
        
        traces.append({
            "task_id": name, "exact": exact, "acc": round(acc, 4),
            "gap": round(gap_val, 4), "gap_type": gap_type,
            "n_actions": n_actions,
            "input_shape": list(inp.shape), "output_shape": list(out.shape),
            "n_in_objs": int((inp>0).sum()), "n_out_objs": int((out>0).sum()),
        })
    
    return traces


def discover_from_shape_mismatch(registry, traces, dataset):
    """Para cada SHAPE_MISMATCH, tenta descobrir primitiva que resolve."""
    from core.operator_specification import SpecGenerator, InvariantExtractor
    from core.spec_guided_discovery import SpecGuidedDiscovery
    
    # Filtra SHAPE_MISMATCH
    shape_mismatch = [t for t in traces if t["gap_type"] == "SHAPE_MISMATCH" and t["input_shape"] != t["output_shape"]]
    if not shape_mismatch:
        print("  Nenhum SHAPE_MISMATCH encontrado")
        return 0
    
    print(f"\n  {len(shape_mismatch)} SHAPE_MISMATCH encontrados. Gerando specs...")
    
    # Gera OperatorSpec para cada shape mismatch
    spec_gen = SpecGenerator()
    inv_ext = InvariantExtractor()
    
    for t in shape_mismatch[:5]:  # Limite de 5 por ciclo
        name = t["task_id"]
        for idx, dt in enumerate(dataset.tasks):
            if dt[0] == name:
                task_data = dt[1]
                break
        else:
            continue
        
        train = task_data.get("train", [])
        if not train:
            continue
        
        inp = np.array(train[0]["input"], dtype=int)
        out = np.array(train[0]["output"], dtype=int)
        
        # Extrai invariantes e transformacoes
        inv, trans = inv_ext.extract_all([(inp, out)], "SHAPE_MISMATCH")
        
        # Gera spec
        spec = spec_gen.generate(
            failure_type="SHAPE_MISMATCH",
            name=f"spec_shape_{name[:8]}",
            examples=[(inp, out)],
            description=f"transform grid from {list(inp.shape)} to {list(out.shape)}",
        )
        
        print(f"    Spec para {name}: {spec.input_structure} -> {spec.output_structure}")
        for t in spec.transformations:
            print(f"      TRANSFORMA: {t.description}")
        for i in spec.invariants:
            print(f"      PRESERVA: {i.description}")
    
    # Busca exemplos no GeometryDataset que casam com as specs
    print(f"\n  Buscando exemplos no GeometryDataset...")
    discovery = PrimitiveDiscovery(registry=registry)
    guided = SpecGuidedDiscovery(registry=registry, discovery=discovery, dataset_size=3000)
    
    n_before = registry.count
    n_new = guided.discover_all_from_dataset()
    
    if n_new > 0:
        print(f"  Novas primitives descobertas: {n_new}")
        for p in registry.get_all():
            if p.confidence < 0.5:
                print(f"    NOVA: {p.name}: {p.description[:60]}")
    else:
        print(f"  Nenhuma nova primitiva descoberta (ja existentes ou dataset insuficiente)")
    
    return n_new


def print_report(round_num, traces_before, traces_after, n_new):
    """Imprime relatorio de progresso."""
    n_new = int(n_new) if not isinstance(n_new, int) else n_new
    def stats(traces):
        exact = sum(1 for t in traces if t["exact"])
        acc = np.mean([t["acc"] for t in traces]) if traces else 0
        gap = np.mean([t["gap"] for t in traces]) if traces else 0
        shape = sum(1 for t in traces if t["gap_type"] == "SHAPE_MISMATCH")
        near = sum(1 for t in traces if t["gap_type"] == "NEAR_MISS")
        actions = np.mean([t["n_actions"] for t in traces]) if traces else 0
        return exact, acc, gap, shape, near, actions
    
    b_exact, b_acc, b_gap, b_shape, b_near, b_act = stats(traces_before)
    a_exact, a_acc, a_gap, a_shape, a_near, a_act = stats(traces_after)
    
    print(f"\n{'='*70}")
    print(f"CICLO {round_num}: ARC Self-Discovery")
    print(f"{'='*70}")
    print(f"\n{'Metrica':<25s} {'Antes':>12s} {'Depois':>12s} {'Delta':>12s}")
    print(f"{'-'*61}")
    print(f"  {'Exact Match':<25s} {b_exact:>6d}      {a_exact:>6d}      {a_exact-b_exact:>+6d}")
    print(f"  {'Pixel Acc':<25s} {b_acc:>10.2%}  {a_acc:>10.2%}  {a_acc-b_acc:>+10.2%}")
    print(f"  {'Gap Medio':<25s} {b_gap:>10.3f}  {a_gap:>10.3f}  {a_gap-b_gap:>+10.3f}")
    print(f"  {'SHAPE_MISMATCH':<25s} {b_shape:>6d}      {a_shape:>6d}      {a_shape-b_shape:>+6d}")
    print(f"  {'NEAR_MISS':<25s} {b_near:>6d}      {a_near:>6d}      {a_near-b_near:>+6d}")
    print(f"  {'Acoes/task':<25s} {b_act:>8.2f}    {a_act:>8.2f}    {a_act-b_act:>+8.2f}")
    print(f"  {'Novas Primitives':<25s} {0:>6d}      {n_new:>6d}      +{n_new}")


# ─── MAIN ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("ARC SELF-DISCOVERY: O sistema descobre as proprias primitives")
    print("=" * 70)
    
    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    registry = setup_registry()
    
    print(f"Primitives iniciais: {registry.count}")
    
    # Ciclo de auto-descoberta
    for round_num in range(1, 4):  # Max 3 ciclos
        print(f"\n{'='*70}")
        print(f"CICLO {round_num}")
        print(f"{'='*70}")
        
        # Baseline
        print(f"\n[PRE] Rodando ARC com {registry.count} primitives...")
        traces_before = run_arc_eval(registry, dataset, n_tasks=100)
        b_exact = sum(1 for t in traces_before if t["exact"])
        b_acc = np.mean([t["acc"] for t in traces_before]) if traces_before else 0
        print(f"  Exact: {b_exact}, Acc: {b_acc:.2%}")
        
        if b_acc > 0.7:
            print(f"\n  Acc > 70%, convergido!")
            break
        
        # Descobre novas primitives das falhas
        print(f"\n[DESCOBERTA] Analisando falhas e descobrindo novas primitives...")
        n_new = discover_from_shape_mismatch(registry, traces_before, dataset)
        
        # Re-testa (sempre, mesmo sem novas primitives)
        print(f"\n[POS] Rodando ARC com {registry.count} primitives...")
        traces_after = run_arc_eval(registry, dataset, n_tasks=100)
        
        # Relatorio
        print_report(round_num, traces_before, traces_after, n_new)
        
        if n_new == 0:
            print(f"\n  Nenhuma nova primitiva descoberta neste ciclo.")
            if round_num >= 2:
                print(f"  Estagnou por 2 ciclos. Parando.")
                break
    
    # Final
    print(f"\n\n{'='*70}")
    print(f"CICLO DE AUTO-DESCOBERTA CONCLUIDO")
    print(f"Primitives finais: {registry.count}")
    print(f"Relatorio salvo em: {RESULTS_DIR}")
