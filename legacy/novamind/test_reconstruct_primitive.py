"""
Experimento Definitivo: Sistema RECONSTROI primitiva removida.

2 modos:
  A (bridge): operator proposals disponiveis, discovery detecta
  B (blind):  sem proposals, sistema deve INDUZIR programa dos exemplos

Se o sistema passar no modo B, provou que INDUZ programas de exemplos.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph, OperatorGroundingBridge,
    PrimitiveDiscovery, Program
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.primitive import Primitive, PrimitiveInstance
from core.arc_dataset import ARCDataset

OUTPUT = "D:/novamind_data/reconstruct_test"
os.makedirs(OUTPUT, exist_ok=True)


def make_minimal_registry(include_proposals=True):
    """Registry MINIMO: recolor, remove, move, add + opcionais propostas."""
    registry = PrimitiveRegistry()
    for fn in [create_recolor_primitive, create_remove_primitive,
               create_move_primitive, create_add_primitive]:
        registry.register(fn())
    
    if include_proposals:
        from core.program_induction.operator_proposals import get_all_proposals
        for op in get_all_proposals():
            if op["name"] in ("rotate_90", "rotate_180", "rotate_270",
                             "flip_horizontal", "flip_vertical",
                             "transpose", "identity"):
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


def gen_examples(transform_fn, n=100, max_h=10, max_w=10, label="transform"):
    """Gera exemplos input->output para uma funcao de transformacao."""
    np.random.seed(42)
    examples = []
    for _ in range(n):
        h, w = np.random.randint(3, max_h), np.random.randint(3, max_w)
        grid = np.zeros((h, w), dtype=int)
        for _ in range(np.random.randint(1, 5)):
            color = np.random.randint(1, 10)
            r, c = np.random.randint(0, h), np.random.randint(0, w)
            grid[r, c] = color
        result = transform_fn(grid)
        examples.append((grid, result, label))
    return examples


def validate_program(prog, examples, n_test=30):
    """Testa programa em N exemplos, retorna fracao de acerto."""
    correct = 0
    total = min(n_test, len(examples))
    for inp, out, _ in examples[:total]:
        try:
            result = prog.execute(inp)
            if result.shape == out.shape and np.array_equal(result, out):
                correct += 1
        except:
            pass
    return correct / total


def run_arc(registry, dataset, n=100):
    """Benchmark ARC."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()
    exacts, accs = 0, []
    for idx in range(min(n, len(dataset.tasks))):
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
    return {"exact": exacts, "acc": round(np.mean(accs) if accs else 0, 4)}


def test_reconstruction(transform_name, transform_fn, registry, dataset, mode):
    """Testa se o sistema RECONSTROI a primitiva."""
    print(f"\n  --- {transform_name} (modo {mode}) ---")
    
    # Gera exemplos
    examples = gen_examples(transform_fn, n=100)
    print(f"  Exemplos: {len(examples)} (ex: {examples[0][0].shape} -> {examples[0][1].shape})")
    
    # Discovery
    discovery = PrimitiveDiscovery(registry=registry)
    for inp, out, _ in examples:
        discovery.observe(inp, out)
    
    # Novas primitives descobertas
    new_types = discovery.get_signatures()
    print(f"  Tipos descobertos: {len(new_types)}")
    for name, sig in new_types.items():
        print(f"    {name}")
    
    # INDUCAO: usa o synthesizer para encontrar programa que funciona em TODOS
    pie = ProgramSynthesizer(registry, composition_graph=CompositionGraph(),
                            grounding_bridge=OperatorGroundingBridge(registry, CompositionGraph()))
    
    inp0, out0, _ = examples[0]
    candidates = pie.synthesize(inp0, out0)
    
    best_prog = None
    best_score = 0
    for prog in candidates[:15]:
        score = validate_program(prog, examples)
        prog_name = prog.name if hasattr(prog, 'name') else str(prog)[:40]
        if score > best_score:
            best_score = score
            best_prog = prog
            if score > 0.8:
                print(f"  >> PROGRAMA ENCONTRADO: {prog_name} (score={score:.0%})")
    
    if best_score < 0.8:
        print(f"  >> Melhor programa: score={best_score:.0%} (INSUFICIENTE)")
    
    # Registra se validado
    reconstructed = best_score >= 0.8 and best_prog is not None
    if reconstructed:
        induced_name = f"induced_{transform_name}"
        induced = Primitive(name=induced_name, structure_type=transform_name,
                          description=f"system-reconstructed: {transform_name}")
        # Extrai funcao do programa
        if hasattr(best_prog, 'instances') and best_prog.instances:
            induced._execute = best_prog.instances[0].primitive._execute if best_prog.instances[0].primitive._execute else None
            induced._solver_fn = lambda i, o: {}
            induced.confidence = 0.9
            registry.register(induced)
            print(f"  >> Registrada: {induced_name}")
    
    # ARC
    arc = run_arc(registry, dataset)
    print(f"  ARC: acc={arc['acc']:.2%} exact={arc['exact']}")
    
    return {
        "transform": transform_name,
        "mode": mode,
        "reconstructed": reconstructed,
        "best_score": best_score,
        "n_types": len(new_types),
        "arc_acc": arc["acc"],
        "arc_exact": arc["exact"],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("RECONSTRUCAO DE PRIMITIVA — Experimento Controlado")
    print("=" * 70)
    
    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    results = []
    
    for mode_name, use_proposals in [("A (bridge)", True), ("B (blind)", False)]:
        print(f"\n{'='*70}")
        print(f"MODO {mode_name}")
        print(f"{'='*70}")
        
        transforms = [
            ("rotate_90", lambda g: np.rot90(g, k=-1)),
            ("flip_horizontal", lambda g: np.fliplr(g)),
        ]
        
        mode_results = []
        for tname, tfn in transforms:
            registry = make_minimal_registry(include_proposals=use_proposals)
            r = test_reconstruction(tname, tfn, registry, dataset, mode_name)
            mode_results.append(r)
            results.append(r)
        
        # Summary per mode
        n_rec = sum(1 for r in mode_results if r["reconstructed"])
        print(f"\n  Modo {mode_name}: {n_rec}/{len(mode_results)} reconstruidas")
    
    # FINAL
    print(f"\n\n{'='*70}")
    print("RELATORIO FINAL")
    print(f"{'='*70}")
    for r in results:
        status = "RECONSTRUIDA" if r["reconstructed"] else "FALHOU"
        print(f"  [{r['mode']}] {r['transform']}: {status} (score={r['best_score']:.0%})")
    
    # Conclusao
    all_rec = all(r["reconstructed"] for r in results)
    print(f"\nConclusao:")
    if all_rec:
        print(f"  O sistema RECONSTROIU primitivas em ambos os modos.")
        print(f"  Modo A: descobriu via bridge de propostas espaciais.")
        print(f"  Modo B: descobriu via inducao de programa dos exemplos.")
    else:
        a_ok = sum(1 for r in results if r["mode"] == "A (bridge)" and r["reconstructed"])
        b_ok = sum(1 for r in results if r["mode"] == "B (blind)" and r["reconstructed"])
        print(f"  Modo A: {a_ok}/2 | Modo B: {b_ok}/2")
    
    jp = os.path.join(OUTPUT, "results.json")
    with open(jp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvo: {jp}")
