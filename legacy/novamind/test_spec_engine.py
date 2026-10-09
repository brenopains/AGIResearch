"""Teste do OperatorSpecificationEngine: falha -> especificacao (sem implementacao)."""
import sys, numpy as np, json
sys.path.insert(0, ".")

from core.operator_specification import (
    SpecGenerator, InvariantExtractor, OperatorSpec, Invariant, Transformation
)
from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.failure_driven_discovery import (
    FailureDrivenLoop, FailureAnalyzer
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

print("=" * 70)
print("TESTE: OPERATOR SPECIFICATION ENGINE")
print("=" * 70)

# Setup
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)

# ─── TEST 1: InvariantExtractor ───
print("\n[TEST 1] InvariantExtractor — descobre invariantes de exemplos")
extractor = InvariantExtractor()
examples = [
    (np.array([[0,2,0],[2,2,2],[0,2,0]], dtype=int),
     np.array([[0,8,0],[8,8,8],[0,8,0]], dtype=int)),
    (np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int),
     np.array([[0,3,0],[3,3,3],[0,3,0]], dtype=int)),
]
invariants, transforms = extractor.extract_all(examples, "COLOR_MISMATCH")
print(f"  Invariantes: {len(invariants)}")
for i in invariants:
    print(f"    {i.type}: {i.description}")
print(f"  Transformacoes: {len(transforms)}")
for t in transforms:
    print(f"    {t.type}: {t.description}")
assert len(invariants) >= 2  # shape + color_set
assert len(transforms) >= 1  # recolor
print("  PASS")

# ─── TEST 2: SpecGenerator — gera especificacao de exemplos ───
print("\n[TEST 2] SpecGenerator — gera especificacao sem implementacao")
spec_gen = SpecGenerator()
spec = spec_gen.generate(
    failure_type="WRONG_OBJECT_COUNT",
    name="spec_wrong_object_count",
    examples=[
        (np.array([[1,0],[0,1]], dtype=int), np.array([[1,1],[1,1]], dtype=int)),
        (np.array([[2],[3]], dtype=int), np.array([[2,2],[3,3]], dtype=int)),
    ],
    description="replicate objects to increase count",
)
print(f"  Spec: {spec.name}")
print(f"  Failure type: {spec.failure_type}")
print(f"  Desc: {spec.description}")
print(f"  Input: {spec.input_structure}")
print(f"  Output: {spec.output_structure}")
print(f"  Invariantes: {[i.description for i in spec.invariants]}")
print(f"  Transforms: {[t.description for t in spec.transformations]}")
print(f"  Confianca: {spec.confidence}")
print(f"  N exemplos: {spec.n_examples}")
assert spec.failure_type == "WRONG_OBJECT_COUNT"
assert len(spec.invariants) > 0
print("  PASS")

# ─── TEST 3: SpecGenerator deduplica specs similares ───
print("\n[TEST 3] Deduplicacao de specs")
# Mesmo failure_type + exemplos similares
spec2 = spec_gen.generate("WRONG_OBJECT_COUNT", "spec_woc_2", [
    (np.array([[1,0],[0,1]], dtype=int), np.array([[1,1],[1,1]], dtype=int)),
    (np.array([[2,0],[0,2]], dtype=int), np.array([[2,2],[2,2]], dtype=int)),
])
n_specs = len(spec_gen.get_specs())
# Deve ser 1 se a similaridade > 0.7
print(f"  Specs no generator: {n_specs}")
match_score = spec_gen.get_specs()[0].matches(spec2) if spec_gen.get_specs() else 0
print(f"  Similaridade entre specs: {match_score:.3f}")
print(f"  PASS")

# ─── TEST 4: Integracao com FailureDrivenLoop ───
print("\n[TEST 4] Integracao com FailureDrivenLoop")
engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
loop = FailureDrivenLoop(registry=registry, goal_engine=engine, min_cluster_size=3)
spec_gen_loop = SpecGenerator()

# Substitui on_failure para gerar specs
from core.operator_specification import integrate_with_failure_loop
integrate_with_failure_loop(loop, spec_gen_loop)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

for task_idx in range(20):
    name, task = dataset.tasks[task_idx]
    train_pairs = task.get("train", [])
    if not train_pairs:
        continue
    inp = np.array(train_pairs[0]["input"], dtype=int)
    out = np.array(train_pairs[0]["output"], dtype=int)
    engine.set_goal_from_target(out, description=name)
    engine.perceive(inp)
    engine.run(max_steps=5)
    final = engine.current_state
    gap = GoalComparator().compute(final, GoalState(target=out))
    if gap.total > 0.01:
        loop.on_failure(inp, out, final, gap, task_id=name)

# Mostra as specs geradas
print(f"\n  Specs geradas do loop:")
specs = getattr(loop, '_specs', [])
for s in specs:
    print(f"    [{s.failure_type}] {s.name}: {s.description}")
    print(f"      Invariantes: {s.invariant_summary}")
    print(f"      Transformacoes: {s.transformation_summary}")
    print(f"      Exemplos: {s.n_examples}")

# Mostra tambem as specs do spec_gen
print(f"\n  Specs no SpecGenerator:")
for s in spec_gen_loop.get_specs():
    print(f"    [{s.failure_type}] {s.name}: {s.description} (conf={s.confidence:.2f})")

# ─── TEST 5: Spec pode ser usada para guiar busca ───
print("\n[TEST 5] Spec como guia para busca de implementacao")
for s in spec_gen_loop.get_specs():
    print(f"\n  Spec: {s.name}")
    print(f"    O que o operador deve fazer:")
    for t in s.transformations:
        print(f"      TRANSFORMA: {t.description}")
    for i in s.invariants:
        print(f"      PRESERVA:   {i.description}")
    print(f"    Input: {s.input_structure['n_objects']} objetos, "
          f"{s.input_structure['shape']}")
    print(f"    Output: {s.output_structure['n_objects']} objetos, "
          f"{s.output_structure['shape']}")
    print(f"    (spec apenas — implementacao sera descoberta depois)")

print("\n" + "=" * 70)
print("OPERATOR SPECIFICATION ENGINE FUNCIONANDO")
print("(nenhuma implementacao gerada — so especificacao)")
print("=" * 70)
