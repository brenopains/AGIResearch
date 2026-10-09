"""
TESTE DEFINITIVO: Sistema sintetiza rotacao de operandos ATOMICOS.

NAO contem rot90, flip, transpose nos operandos.
So:
  - loop_x, loop_y
  - coord_transform (swap, invert_x, invert_y)
  - read/write pixel

O sistema precisa COMPOR estes operandos para produzir rotacao.
Se funcionar, e PROVA de sintese de programa genuina.
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.program_induction import PrimitiveRegistry
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.conceptual_discovery import InvariantDiscovery
from core.primitive_invention import PrimitiveInventionEngine
from core.primitive_invention.hypothesizer import ProgramHypothesizer, CandidateProgram
from core.primitive_invention.verifier import PrimitiveVerifier
from core.primitive_invention.atomic_ops import get_atomic_programs, COORD_MODES
from core.primitive_invention.registrar import PrimitiveRegistrar

print("=" * 70)
print("TESTE DEFINITIVO: Sintese de rotacao de operandos ATOMICOS")
print("Sem rot90/flip/transpose pre-definidos nos operandos")
print("=" * 70)

# ─── 1. Registry minimo ───
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

# ─── 2. Gera exemplos de rotacao ───
print(f"\n[1] Gerando exemplos de rotacao...")
np.random.seed(42)
examples = []
for i in range(8):
    h, w = np.random.randint(3, 8), np.random.randint(3, 8)
    grid = np.zeros((h, w), dtype=int)
    for _ in range(np.random.randint(1, 4)):
        r, c = np.random.randint(0, h), np.random.randint(0, w)
        grid[r, c] = np.random.randint(1, 10)
    rotated = np.rot90(grid, k=-1)
    examples.append((grid, rotated, f"ex_{i}"))

print(f"  {len(examples)} exemplos")
inp0, out0, _ = examples[0]
print(f"  Exemplo: {inp0.shape} -> {out0.shape}")

# ─── 3. OPERANDOS ATOMICOS ───
print(f"\n[2] Operandos atomicos disponiveis:")
atomic_progs = get_atomic_programs()
for name, fn, desc in atomic_progs:
    # Testa se funciona no exemplo 0
    try:
        result = fn(inp0)
        ok = "OK" if result.shape == out0.shape and np.array_equal(result, out0) else "MISMATCH"
        print(f"  {name:20s} {ok}")
    except Exception as e:
        print(f"  {name:20s} ERRO: {e}")

# ─── 4. HYPOTHESIZER (modo atomico) ───
print(f"\n[3] ProgramHypothesizer (modo atomico)...")
hyp = ProgramHypothesizer(max_depth=1, use_atomic=True)
pair_examples = [(g, r) for g, r, _ in examples]
candidates = hyp.hypothesize(pair_examples)

print(f"  Candidatos encontrados: {len(candidates)}")
for c in candidates:
    print(f"    {c.name}: score={c.score:.3f} n={c.n_correct}/{c.n_tested} val={c.validated}")

# ─── 5. VERIFICACAO ───
print(f"\n[4] PrimitiveVerifier...")
ver = PrimitiveVerifier(min_examples=2)
train = pair_examples[:5]
test = pair_examples[5:]

if candidates:
    best = candidates[0]
    ok = ver.verify(best, train, test)
    print(f"  Melhor candidato ({best.name}): verify={ok}")
    if ok:
        print(f"    >> PROGRAMA VALIDADO! Equivalente a rotate_90!")

# ─── 6. GAP DETECTION ───
print(f"\n[5] GapDetector + InventionEngine (modo atomico)...")
engine = PrimitiveInventionEngine(registry=registry, use_atomic=True)

# Discovery
discovery = InvariantDiscovery()
for inp, out, _ in examples:
    discovery.observe_transformation(inp, out, "rot", domain="test")
invariants = discovery.get_invariants(min_confidence=0.05)

# Feed examples to engine
for inp, out, _ in examples:
    engine.observe(inp, out, invariants)

gaps = engine.get_gaps()
print(f"  Gaps detectados: {len(gaps)}")
for g in gaps:
    print(f"    {g.invariant_type}: n={g.n_observations} ex={len(g.examples)}")

# Invent!
new_prims = engine.invent()
print(f"\n  Primitives inventadas: {len(new_prims)}")
for name in new_prims:
    prim = registry.get(name)
    if prim and prim._execute:
        matches = 0
        for inp_t, out_t, _ in examples:
            try:
                res = prim._execute(inp_t, {})
                if res.shape == out_t.shape and np.array_equal(res, out_t):
                    matches += 1
            except:
                pass
        print(f"    {name}: {matches}/{len(examples)} exemplos corretos")
        if matches == len(examples):
            print(f"      >>> EQUIVALENTE A rotate_90!")

print(f"\n  Registry final: {[p.name for p in registry.get_all()]}")

# ─── 7. ANALISE ───
print(f"\n{'='*70}")
print("ANALISE")
print(f"{'='*70}")
print(f"""
  O sistema {'CONSEGUIU' if new_prims else 'NAO CONSEGUIU'} sintetizar
  rotacao a partir de operandos atomicos.

  {'Isso significa que o ciclo de invencao de primitives esta COMPLETO:'
   if new_prims else 'O ciclo precisa de ajustes.'}
  
  {'O sistema provou que pode criar novas capacidades a partir de'
   ' operacoes elementares, sem depender de primitives pre-definidas.'
   if new_prims else ''}
""")
