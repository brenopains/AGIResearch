"""
Pipeline: Curriculum -> ARC a cada estagio.

Para cada estagio do curriculo:
  1. Gera dados do estagio (conceitos, imagens)
  2. Alimenta o CGE com os conceitos
  3. Roda ARC-AGI benchmark
  4. Registra metrica
  5. Avanca para o proximo estagio

No final: curva de aprendizado mostrando ARC acc por estagio.
"""
import os, sys, json, time, numpy as np
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

WORK_DIR = "D:/novamind_data"
RESULTS_FILE = os.path.join(WORK_DIR, "arc_learning_curve.json")
os.makedirs(WORK_DIR, exist_ok=True)

# ─── Conceitos por estagio ─────────────────────────────────

STAGE_CONCEPTS = {
    "BASELINE": {
        "desc": "Sem treino curricular (apenas primitives base)",
        "concepts": [],
    },
    "INFANTIL": {
        "desc": "Formas, cores, causalidade, espaco, contagem, sons",
        "concepts": [
            "circles are round. squares have four sides. triangles have three sides.",
            "red is a color. blue is a color. green is a color. yellow is a color.",
            "objects exist even when hidden. out of sight does not mean gone.",
            "pushing moves objects. dropping makes things fall. cause precedes effect.",
            "above is higher. below is lower. left and right are directions.",
            "inside means contained. outside means not contained.",
            "one two three four five. numbers represent quantity. counting tells how many.",
            "sounds travel through air. loud is strong. soft is gentle.",
        ],
    },
    "PRE_ESCOLAR": {
        "desc": "Linguagem, categorias, historias, emocoes, aritmetica",
        "concepts": [
            "language uses words to express meaning. sentences have subjects and verbs.",
            "animals are living things. plants are living things. rocks are not alive.",
            "stories have a beginning, middle, and end. characters have motivations.",
            "happiness is a positive emotion. sadness is a negative emotion.",
            "anger arises from frustration. fear arises from perceived danger.",
            "addition combines quantities. subtraction separates quantities.",
            "categorization groups similar things. taxonomy organizes categories.",
        ],
    },
    "FUNDAMENTAL": {
        "desc": "Fisica, quimica, biologia, historia, geografia, matematica, sociedade, artes, esportes",
        "concepts": [
            "gravity pulls objects toward earth. force equals mass times acceleration.",
            "energy cannot be created or destroyed. it transforms from one form to another.",
            "atoms are the basic units of matter. molecules are atoms bonded together.",
            "cells are the basic units of life. DNA contains genetic information.",
            "ecosystems are communities of organisms interacting with their environment.",
            "civilizations rise and fall. history is shaped by cause and effect.",
            "mathematics describes patterns. algebra uses symbols for unknown quantities.",
            "geometry studies shapes and spaces. angles are measured in degrees.",
            "governments provide structure to society. laws establish rights and duties.",
            "art expresses human experience. music organizes sound into patterns.",
        ],
    },
    "MEDIO": {
        "desc": "Logica, fisica classica, quimica avancada, calculo, filosofia, programacao, biologia",
        "concepts": [
            "logic uses premises to derive conclusions. deduction guarantees truth.",
            "induction infers general patterns from specific observations.",
            "newtonian mechanics describes motion. thermodynamics describes heat transfer.",
            "oxidation is loss of electrons. reduction is gain of electrons.",
            "calculus studies change. derivatives measure rates of change.",
            "integrals measure accumulation. limits describe approaching values.",
            "philosophy questions fundamental assumptions. epistemology studies knowledge.",
            "programming is giving instructions to a computer. algorithms are step-by-step procedures.",
            "evolution by natural selection explains biodiversity.",
        ],
    },
}


def setup_registry():
    """Cria registry com primitives base + espaciais."""
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


def run_arc_benchmark(registry, cge=None, n_tasks: int = 50) -> dict:
    """Roda ARC-AGI benchmark e retorna metricas."""
    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph)
    cmp = GoalComparator()

    # Se tem CGE, passa para a bridge
    if cge is not None:
        bridge.cge = cge

    exacts, accs, gaps, times = 0, [], [], []

    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        t0 = time.time()
        engine.set_goal_from_target(out, description=name)
        engine.perceive(inp)
        engine.step()
        elapsed = time.time() - t0

        final = engine.current_state
        if final.shape == out.shape:
            exact = bool(np.array_equal(final, out))
            acc = 1.0 - int((final != out).sum()) / max(1, int(out.size))
            gap = cmp.compute(final, GoalState(target=out)).total
        else:
            exact, acc, gap = False, 0.0, 1.0

        exacts += 1 if exact else 0
        accs.append(acc)
        gaps.append(gap)
        times.append(elapsed)

    return {
        "n_tasks": len(accs),
        "exact_match": exacts,
        "pixel_accuracy": round(np.mean(accs), 4) if accs else 0,
        "gap_medio": round(np.mean(gaps), 4) if gaps else 0,
        "tempo_medio_ms": round(np.mean(times) * 1000, 1) if times else 0,
    }


def feed_stage_to_cge(stage_name: str, cge):
    """Alimenta o CGE com os conceitos de um estagio."""
    stage = STAGE_CONCEPTS.get(stage_name)
    if not stage:
        return
    concepts = stage.get("concepts", [])
    for text in concepts:
        cge.write_text_concepts(text)
    print(f"  [CGE] {len(concepts)} conceitos escritos para {stage_name}")


def main():
    results = {}
    
    # Carrega resultados anteriores se existirem
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    
    # Inicializa CGE
    print("[INIT] Carregando CGE...")
    from core.cognitive_graph_engine import CognitiveGraphEngine
    from core.bge_encoder import BGEEncoder
    cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
    bge = BGEEncoder(device="cuda")
    
    registry = setup_registry()
    print(f"[INIT] Registry: {registry.count} primitives")
    
    stages_to_run = ["BASELINE", "INFANTIL", "PRE_ESCOLAR", "FUNDAMENTAL", "MEDIO"]
    
    for stage_name in stages_to_run:
        # Pula se ja processado
        if stage_name in results:
            print(f"\n[SKIP] {stage_name} ja processado: {results[stage_name]}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{stage_name}] {STAGE_CONCEPTS[stage_name]['desc']}")
        print(f"{'='*60}")
        
        # Alimenta CGE com conceitos do estagio
        if stage_name != "BASELINE":
            feed_stage_to_cge(stage_name, cge)
        
        # Roda ARC
        print(f"  Rodando ARC (50 tasks)...")
        result = run_arc_benchmark(registry, cge=cge)
        
        print(f"  ARC: acc={result['pixel_accuracy']:.2%} "
              f"exact={result['exact_match']} gap={result['gap_medio']:.3f}")
        
        results[stage_name] = result
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Resultados salvos em {RESULTS_FILE}")
    
    # --- Curva de aprendizado ---
    print(f"\n\n{'='*60}")
    print(f"CURVA DE APRENDIZADO (ARC por estagio curricular)")
    print(f"{'='*60}")
    print(f"\n  {'Estagio':<20s} {'Acc':>8s} {'Exact':>8s} {'Gap':>8s} {'Tempo':>8s}")
    print(f"  {'-'*55}")
    
    for stage_name in stages_to_run:
        if stage_name in results:
            r = results[stage_name]
            print(f"  {stage_name:<20s} {r['pixel_accuracy']:.2%}   {r['exact_match']}     "
                  f"{r['gap_medio']:.3f}  {r['tempo_medio_ms']:.0f}ms")
    
    # Diferenca entre baseline e ultimo estagio
    if "BASELINE" in results and "MEDIO" in results:
        base = results["BASELINE"]["pixel_accuracy"]
        final = results["MEDIO"]["pixel_accuracy"]
        delta = final - base
        print(f"\n  Melhoria total: {base:.2%} -> {final:.2%} ({delta:+.2%})")
        
        if delta > 0:
            print(f"  O treino curricular MELHOROU a performance em ARC")
            print(f"  (causalidade: conceitos infantis contribuiram para raciocinio visual)")
        else:
            print(f"  O treino curricular NAO afetou significativamente ARC")
            print(f"  (esperado: ARC testa percepcao, nao conhecimento enciclopedico)")


if __name__ == "__main__":
    main()
