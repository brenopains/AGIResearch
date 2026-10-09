"""
ARC-AGI Multi-Example Learning — usa TODOS os train examples.
"""
import sys, numpy as np, time
sys.path.insert(0, ".")

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


def setup_registry():
    reg = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        reg.register(fn())
    for op in get_all_proposals():
        fn = op["function"]
        prim = Primitive(name=op["name"], structure_type=op["name"])
        def _me(f):
            def _e(g, p):
                try: return f(g)
                except: return g.copy()
            return _e
        prim._execute = _me(fn)
        reg.register(prim)
    return reg


def extract_sig(program):
    """(structure_type, params) tuple como assinatura unica."""
    return tuple((inst.primitive.structure_type, str(sorted(inst.params.items())))
                 for inst in program.instances)


def evaluate(reg, inp, out):
    """Roda synthesizer e retorna (validated, score, program)."""
    cg = CompositionGraph()
    og = OperatorGroundingBridge(reg, cg)
    pie = ProgramSynthesizer(reg, composition_graph=cg, grounding_bridge=og, max_depth=4)
    try:
        programs = pie.synthesize(inp, out)
    except Exception:
        programs = []
    if not programs:
        return False, 0.0, None
    prog = programs[0]
    return prog.validated, prog.score, prog


print("=" * 70)
print("ARC-AGI Multi-Example — 400 tasks")
print("=" * 70)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
registry = setup_registry()

total = 400
results = {
    "baseline": {"exact": 0, "accs": []},
    "multi": {"exact": 0, "accs": []},
}

start = time.time()
for idx in range(total):
    name, task = dataset.tasks[idx]
    train = task.get("train", [])
    test = task.get("test", [])
    if not train or not test:
        continue

    train_pairs = [(np.array(ex["input"], dtype=int), np.array(ex["output"], dtype=int))
                   for ex in train]
    test_inp = np.array(test[0]["input"], dtype=int)
    test_out = np.array(test[0]["output"], dtype=int)

    # --- BASELINE: usa apenas train[0] ---
    _, _, prog = evaluate(registry, train_pairs[0][0], train_pairs[0][1])
    if prog:
        try:
            result = prog.execute(test_inp)
            if result.shape == test_out.shape:
                results["baseline"]["exact"] += 1 if np.array_equal(result, test_out) else 0
                results["baseline"]["accs"].append(
                    1.0 - int((result != test_out).sum()) / max(1, int(test_out.size)))
        except:
            pass

    # --- MULTI-EXAMPLE: coleta programas de CADA par, filtra por consistencia ---
    all_by_sig = {}  # sig -> [(score, validated, program)]
    for inp, out in train_pairs:
        val, score, prog = evaluate(registry, inp, out)
        if prog:
            sig = extract_sig(prog)
            if sig not in all_by_sig:
                all_by_sig[sig] = []
            all_by_sig[sig].append((score, val, prog))

    best_prog = None
    best_score = -1

    for sig, entries in all_by_sig.items():
        # Consistencia = quantos exemplos este programa explica
        n_pairs = len(train_pairs)
        n_found = len(entries)
        consistency = n_found / n_pairs  # fracao de pares onde aparece

        # Score medio
        avg_score = np.mean([e[0] for e in entries])

        # Penalidade se nao explica todos
        combined = avg_score * consistency

        if combined > best_score and consistency >= 0.5:
            best_score = combined
            best_prog = entries[0][2]  # primeiro programa com esta sig

    if best_prog:
        try:
            result = best_prog.execute(test_inp)
            if result.shape == test_out.shape:
                results["multi"]["exact"] += 1 if np.array_equal(result, test_out) else 0
                results["multi"]["accs"].append(
                    1.0 - int((result != test_out).sum()) / max(1, int(test_out.size)))
        except:
            pass

    if (idx + 1) % 50 == 0:
        el = time.time() - start
        print(f"  [{idx+1}/{total}] {el:.0f}s | Base:{results['baseline']['exact']} Multi:{results['multi']['exact']}")

elapsed = time.time() - start

b_exact = results["baseline"]["exact"]
m_exact = results["multi"]["exact"]
b_pa = (np.mean(results["baseline"]["accs"]) * 100) if results["baseline"]["accs"] else 0
m_pa = (np.mean(results["multi"]["accs"]) * 100) if results["multi"]["accs"] else 0

print(f"\n{'='*70}")
print(f"RESULTADOS ({total} tasks, {elapsed:.0f}s)")
print(f"{'='*70}")
print(f"\n{'Metrica':<35s} {'Baseline':>10s} {'Multi':>10s}")
print(f"{'-'*57}")
print(f"  {'Exact Match (test)':<35s} {b_exact:>6d}       {m_exact:>6d}")
print(f"  {'Pixel Acc % (test)':<35s} {b_pa:>7.2f}%    {m_pa:>7.2f}%")
delta_exact = m_exact - b_exact
delta_pa = m_pa - b_pa
print(f"  {'Delta':<35s} {'+'+str(delta_exact) if delta_exact>0 else str(delta_exact):>10s} {'+'+f'{delta_pa:.2f}%' if delta_pa>0 else f'{delta_pa:.2f}%':>10s}")

out_path = r"D:\novamind_data\arc_multi_example_results.txt"
with open(out_path, "w") as f:
    f.write(f"ARC Multi-Example ({total} tasks, {elapsed:.0f}s)\n")
    f.write(f"Baseline: exact={b_exact}, pixel_acc={b_pa:.2f}%\n")
    f.write(f"Multi: exact={m_exact}, pixel_acc={m_pa:.2f}%\n")
print(f"\nResultados: {out_path}")
