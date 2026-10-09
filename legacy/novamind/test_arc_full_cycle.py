"""
Teste completo: ARC -> falha -> spec -> dataset -> nova primitiva -> retest.

O ciclo AGI completo:
  1. Roda ARC com GoalDrivenEngine
  2. Falha gera OperatorSpec
  3. SpecGuidedDiscovery busca exemplos no GeometryDataset
  4. PrimitiveDiscovery induz nova primitiva dos exemplos
  5. Nova primitiva adicionada ao registry
  6. Retesta ARC com registry expandido
  7. Mede quantas tasks a mais foram resolvidas
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.failure_driven_discovery import FailureDrivenLoop, FailureAnalyzer
from core.operator_specification import SpecGenerator, integrate_with_failure_loop
from core.spec_guided_discovery import SpecGuidedDiscovery
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge, PrimitiveDiscovery,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.arc_dataset import ARCDataset

print("=" * 70)
print("CICLO COMPLETO: ARC -> FALHA -> SPEC -> DATASET -> PRIMITIVA -> RETEST")
print("=" * 70)

# ─── Setup inicial ───
device = "cuda"
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

print(f"\nPrimitives iniciais: {registry.count}")
cmp = GoalComparator()

# ─── FASE 1: RODA ARC E COLETA FALHAS ───
print("\n[FASE 1] Rodando ARC...")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10, dev=device)

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
loop = FailureDrivenLoop(registry=registry, goal_engine=engine, min_cluster_size=3)
spec_gen = SpecGenerator()
integrate_with_failure_loop(loop, spec_gen)

baseline = {"total": 0, "exact": 0, "accs": [], "gaps": []}

for task_idx in range(30):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)

    engine.set_goal_from_target(out, description=name)
    engine.perceive(inp)
    achieved = engine.run(max_steps=5)
    final = engine.current_state
    gap = cmp.compute(final, GoalState(target=out))
    exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
    acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0

    baseline["total"] += 1
    if exact:
        baseline["exact"] += 1
    baseline["accs"].append(acc)
    baseline["gaps"].append(gap.total)

    if gap.total > 0.01:
        loop.on_failure(inp, out, final, gap, task_id=name)

print(f"\n  Baseline: {baseline['exact']}/{baseline['total']} exact, "
      f"acc media={np.mean(baseline['accs']):.2%}")

# ─── FASE 2: GERA SPECS DAS FALHAS ───
print("\n[FASE 2] Specs geradas das falhas:")
specs = spec_gen.get_specs()
for s in specs:
    print(f"  [{s.failure_type}] {s.name}:")
    for t in s.transformations:
        print(f"    TRANSFORMA: {t.description}")
    for i in s.invariants:
        print(f"    PRESERVA:   {i.description}")

# ─── FASE 3: SPEC -> DATASET -> NOVA PRIMITIVA ───
print("\n[FASE 3] SpecGuidedDiscovery: buscando implementacoes no dataset...")

discovery_engine = PrimitiveDiscovery(registry=registry)
guided = SpecGuidedDiscovery(registry=registry, discovery=discovery_engine, dataset_size=5000)

print(f"\n  Descobrindo TODAS as primitives do dataset...")
n_new = guided.discover_all_from_dataset()
print(f"  Total de novas primitives: {n_new} (agora {registry.count} no total)")
for p in registry.get_all():
    print(f"    {p.name}: {p.description[:60]}")

# ─── FASE 4: RETEST ARC ───
print("\n[FASE 4] Retest ARC com registry expandido...")

graph2 = CompositionGraph()
bridge2 = OperatorGroundingBridge(registry, graph2)
pie2 = ProgramSynthesizer(registry, composition_graph=graph2, grounding_bridge=bridge2)
engine2 = GoalDrivenEngine(program_induction_engine=pie2, composition_graph=graph2)

improved = {"total": 0, "exact": 0, "accs": [], "gaps": [], "better": 0, "worse": 0}

for task_idx in range(30):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)

    engine2.set_goal_from_target(out, description=name)
    engine2.perceive(inp)
    achieved = engine2.run(max_steps=5)
    final = engine2.current_state
    gap = cmp.compute(final, GoalState(target=out))
    exact = bool(np.array_equal(final, out)) if final.shape == out.shape else False
    acc = 1.0 - int((final != out).sum()) / max(1, int(out.size)) if final.shape == out.shape else 0.0

    improved["total"] += 1
    if exact:
        improved["exact"] += 1
    improved["accs"].append(acc)
    improved["gaps"].append(gap.total)

    # Compara com baseline
    baseline_acc = baseline["accs"][task_idx] if task_idx < len(baseline["accs"]) else 0
    if acc > baseline_acc + 0.05:
        improved["better"] += 1
    elif acc < baseline_acc - 0.05:
        improved["worse"] += 1

print(f"\n[RESULTADOS]")
print(f"  Baseline:        {baseline['exact']}/{baseline['total']} exact, "
      f"acc={np.mean(baseline['accs']):.2%}")
print(f"  Apos descoberta: {improved['exact']}/{improved['total']} exact, "
      f"acc={np.mean(improved['accs']):.2%}")
print(f"  Melhoraram: {improved['better']} | Pioraram: {improved['worse']}")

print(f"\n[RESUMO]")
delta_acc = np.mean(improved['accs']) - np.mean(baseline['accs'])
print(f"  Primitives: 4 -> {registry.count} (+{registry.count - 4})")
print(f"  Acc media:   {np.mean(baseline['accs']):.2%} -> {np.mean(improved['accs']):.2%} "
      f"({'+' if delta_acc > 0 else ''}{delta_acc:.1%})")
print(f"  Exact match: {baseline['exact']} -> {improved['exact']}")
print(f"  Especificacoes geradas: {len(specs)}")
print(f"  GeometryDataset experiencias: {guided.get_stats()}")
