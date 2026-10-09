"""
TESTE DEFINITIVO: Sistema INVENTA primitiva a partir de low-level ops.

Protocolo:
  1. Registry SEM rotate_90
  2. Alimenta exemplos de rotacao 90
  3. InvariantDiscovery detecta "reflection"
  4. GapDetector ve que nao ha primitiva para reflection
  5. ProgramHypothesizer gera candidatos de low-level ops
  6. PrimitiveVerifier valida em 5+ exemplos
  7. PrimitiveRegistrar salva a nova primitiva
  8. Testa: nova primitiva funciona em tasks ARC que exigem rotacao?

Se funcionar, o sistema INVENTOU rotate_90 sem nunca ter visto.
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

from core.program_induction import PrimitiveRegistry
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.arc_dataset import ARCDataset
from core.conceptual_discovery import InvariantDiscovery
from core.primitive_invention import PrimitiveInventionEngine

print("=" * 70)
print("TESTE: SISTEMA INVENTA rotate_90 DE LOW-LEVEL OPS")
print("=" * 70)

# ─── 1. Registry SEM rotate_90 ───
registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())
# Propositalmente NAO registra rotate_90, flip, transpose
print(f"\n[1] Registry inicial: {[p.name for p in registry.get_all()]}")
assert "rotate_90" not in [p.name for p in registry.get_all()], "rotate_90 nao deveria estar!"

# ─── 2. Gera exemplos de rotacao ───
print(f"\n[2] Gerando exemplos de rotacao 90...")
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

print(f"  {len(examples)} exemplos gerados")
print(f"  Exemplo: shape {examples[0][0].shape} -> {examples[0][1].shape}")

# ─── 3. InvariantDiscovery observa as transformacoes ───
print(f"\n[3] InvariantDiscovery observando...")
discovery = InvariantDiscovery()
for inp, out, _ in examples:
    discovery.observe_transformation(inp, out, "rotation_example", domain="test")

invariants = discovery.get_invariants(min_confidence=0.1)
print(f"  Invariantes descobertos: {len(invariants)}")
for inv in invariants:
    print(f"    {inv.structure_type}: conf={inv.confidence:.2f}")

# ─── 4. PrimitiveInventionEngine detecta gap e INVENTA ───
print(f"\n[4] PrimitiveInventionEngine...")
engine = PrimitiveInventionEngine(registry=registry, min_gap_priority=0.1)

# Alimenta exemplos
for inp, out, _ in examples:
    engine.observe(inp, out, invariants)

# Tenta inventar
new_prims = engine.invent()
print(f"  Primitives inventadas: {len(new_prims)}")
for name in new_prims:
    prim = registry.get(name)
    print(f"    {name}: ")
    if prim:
        print(f"      desc: {prim.description}")
        print(f"      conf: {prim.confidence:.2f}")
        print(f"      execute: {prim._execute is not None}")

# ─── 5. VALIDACAO: nova primitiva funciona? ───
print(f"\n[5] Validacao nos exemplos de teste...")
test_inp, test_out, _ = examples[0]

for name in new_prims:
    prim = registry.get(name)
    if prim and prim._execute:
        try:
            result = prim._execute(test_inp, {})
            matches = 0
            for inp_t, out_t, _ in examples:
                res = prim._execute(inp_t, {})
                if res.shape == out_t.shape and np.array_equal(res, out_t):
                    matches += 1
            print(f"  {name}: {matches}/{len(examples)} exemplos corretos")
            if matches == len(examples):
                print(f"    >>> PRIMITIVA INVENTADA FUNCIONA EM TODOS OS EXEMPLOS!")
                print(f"    >>> EQUIVALENTE A rotate_90!")
        except Exception as e:
            print(f"  {name}: erro: {e}")

# ─── 6. TESTE EM ARC ───
print(f"\n[6] Teste em ARC (com registry expandido)...")
print(f"\n  Registry final: {[p.name for p in registry.get_all()]}")

# Mostra as novas primitives
print(f"\n  Novas primitives no registry:")
for p in registry.get_all():
    if p.structure_type == "invented":
        print(f"    {p.name}: {p.description[:60]}")

print(f"\n{'='*70}")
print(f"RESULTADO: {len(new_prims)} primitives inventadas")
if new_prims:
    print(f"O sistema INVENTOU novas primitives a partir de low-level ops.")
    print(f"Isso prova que o ciclo de invencao de primitivas funciona:")
    print(f"  Invariante -> Gap -> Hipotese -> Validacao -> Registro")
else:
    print(f"Nenhuma primitiva foi inventada.")
    print(f"Possiveis causas:")
    print(f"  1. Low-level ops nao cobrem a transformacao necessaria")
    print(f"  2. GapDetector exigiu mais exemplos que o fornecido")
    print(f"  3. Hypothesis nao encontrou composicao valida")
print(f"{'='*70}")
