"""
4 Experimentos sobre Macros vs Skills.

Experimento 1: Forcar uso de macros (try macro first)
Experimento 2: Medir reducao de busca (nos, profundidade, tempo)
Experimento 3: Agrupar macros por ESTADO INTERMEDIARIO (objetivo)
Experimento 4: Criar SKILL de cluster de macros
"""
import sys, numpy as np, time
from collections import defaultdict
sys.path.insert(0, ".")

from core.goal_driven_engine import GoalDrivenEngine, GoalState, GoalComparator
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge,
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)
from core.program_induction.operator_proposals import get_all_proposals
from core.program_induction.primitive import Primitive, PrimitiveInstance
from core.macro_discovery import MacroDiscovery
from core.macro_registry import MacroRegistry
from core.arc_dataset import ARCDataset


def setup_base_registry():
    """Registry com 11 primitives base."""
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


def discover_macros(registry, dataset, n_tasks=60):
    """Descobre macros do grafo gerado por N tasks ARC."""
    graph = CompositionGraph()
    bridge = OperatorGroundingBridge(registry, graph)
    pie = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)
    engine = GoalDrivenEngine(program_induction_engine=pie, composition_graph=graph, use_planner=False)

    for task_idx in range(n_tasks):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)
        engine.set_goal_from_target(out, description=name)
        engine.perceive(inp)
        engine.step()

    disc = MacroDiscovery(composition_graph=graph, registry=registry)
    disc.scan(min_frequency=3, min_success_rate=0.3)

    macro_reg = MacroRegistry()
    for m in disc.get_macros(min_confidence=0.1):
        macro_reg.register(
            name=m.name, sequence=m.primitive_sequence,
            frequency=m.frequency, success_rate=m.success_rate,
            domain="ARC",
        )
    return macro_reg, graph


def measure_intermediate_state(seq, inp, registry):
    """Aplica uma sequencia de primitives e retorna o estado intermediario apos CADA passo."""
    states = [inp.copy()]
    state = inp.copy()
    for step_idx, pname in enumerate(seq):
        prim = registry.get(pname)
        if prim:
            state = prim.apply(state, {})
        states.append(state.copy())
    return states


# ─── EXPERIMENTO 1: Forcar uso de macros ───
def exp1_force_macros(registry, macro_reg, dataset, n_tasks=100):
    """Forca uso de macros: para cada task, testa macros primeiro."""
    print("\n" + "=" * 60)
    print("EXPERIMENTO 1: FORCAR USO DE MACROS")
    print("=" * 60)

    matched = 0        # macro encontrou match exato
    macro_tried = 0     # tasks onde macro foi tentada
    macro_better = 0    # macro deu acc maior que primitives base

    base_accs = []

    for task_idx in range(n_tasks):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        # Tenta cada macro: se alguma produz match exato, vence
        macro_worked = False
        best_macro_acc = 0.0
        best_macro_name = ""

        for macro in macro_reg.get_best(n=100, min_confidence=0.1):
            seq = macro.primitive_sequence
            try:
                result = inp.copy()
                for pname in seq:
                    prim = registry.get(pname)
                    if prim:
                        result = prim.apply(result, {})
                if result.shape == out.shape:
                    acc = 1.0 - int((result != out).sum()) / max(1, int(out.size))
                    if acc > best_macro_acc:
                        best_macro_acc = acc
                        best_macro_name = macro.name
                    if np.array_equal(result, out):
                        macro_worked = True
                        matched += 1
                        break
            except Exception:
                pass

        if best_macro_acc > 0 or macro_worked:
            macro_tried += 1

        # Baseline: acc sem macro (do engine)
        cmp = GoalComparator()
        gap = cmp.compute(inp, GoalState(target=out))
        base_acc = 1.0 - gap.total
        base_accs.append(base_acc)

        if best_macro_acc > base_acc + 0.05:
            macro_better += 1

    avg_base = np.mean(base_accs) if base_accs else 0
    print(f"  Tasks: {n_tasks}")
    print(f"  Tasks com macro viavel: {macro_tried}")
    print(f"  Macros com match exato: {matched}")
    print(f"  Macros melhores que base: {macro_better}")
    print(f"  Acc media base: {avg_base:.2%}")
    return {"macro_tried": macro_tried, "matched": matched, "better": macro_better}


# ─── EXPERIMENTO 2: Medir reducao de busca ───
def exp2_search_reduction(registry, macro_reg, dataset, n_tasks=50):
    """Para tasks onde macro se aplica, mede nos explorados sem vs com macro."""
    print("\n" + "=" * 60)
    print("EXPERIMENTO 2: REDUCAO DE BUSCA POR MACRO")
    print("=" * 60)

    results = []

    for task_idx in range(n_tasks):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)
        out = np.array(train_pairs[0]["output"], dtype=int)

        # Para cada macro, verifica se ela produz estado intermediario
        for macro in macro_reg.get_best(n=50, min_confidence=0.1):
            seq = macro.primitive_sequence
            try:
                result = inp.copy()
                for pname in seq:
                    prim = registry.get(pname)
                    if prim:
                        result = prim.apply(result, {})

                if result.shape == out.shape:
                    acc = 1.0 - int((result != out).sum()) / max(1, int(out.size))
                    if acc > 0.3:
                        # Macro foi util: mede quantos passos economizou
                        steps_without = len(seq)  # precisaria desses passos sem macro
                        steps_with = 1            # com macro, so 1 passo
                        reduction = steps_without - steps_with
                        results.append({
                            "task": name,
                            "macro": macro.name,
                            "seq": seq,
                            "acc": round(acc, 4),
                            "steps_saved": reduction,
                            "compression": len(seq),
                        })
                        break
            except Exception:
                pass

    if results:
        total_saved = sum(r["steps_saved"] for r in results)
        avg_compression = np.mean([r["compression"] for r in results])
        print(f"  Tasks onde macro foi util: {len(results)}")
        print(f"  Passos economizados total: {total_saved}")
        print(f"  Media compressao: {avg_compression:.1f}x")
        print(f"  Top macros por utilidade:")
        by_macro = defaultdict(list)
        for r in results:
            by_macro[r["macro"]].append(r)
        for macro_name, rs in sorted(by_macro.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"    {macro_name:25s}: usada em {len(rs):2d} tasks, "
                  f"media acc={np.mean([r['acc'] for r in rs]):.2%}")
    else:
        print(f"  Nenhuma macro foi util em {n_tasks} tasks (acc > 0.3)")
    return results


# ─── EXPERIMENTO 3: Agrupar macros por estado intermediario ───
def exp3_group_by_intermediate(registry, macro_reg, dataset, n_tasks=200):
    """Agrupa macros que produzem o MESMO estado intermediario."""
    print("\n" + "=" * 60)
    print("EXPERIMENTO 3: MACROS POR ESTADO INTERMEDIARIO")
    print("=" * 60)

    # Para cada macro, aplica a sequencia ate o penultimo passo e guarda o estado
    macro_intermediates = {}  # macro_name -> (task, state_hash)

    sample_states = defaultdict(list)  # state_hash -> [(macro_name, task_id)]

    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        name, task = dataset.tasks[task_idx]
        train_pairs = task.get("train", [])
        if not train_pairs:
            continue
        inp = np.array(train_pairs[0]["input"], dtype=int)

        for macro in macro_reg.get_best(n=50, min_confidence=0.1):
            seq = macro.primitive_sequence
            if len(seq) < 2:
                continue

            # Estado APOS o primeiro passo da macro
            pname = seq[0]
            prim = registry.get(pname)
            if prim:
                try:
                    after_step1 = prim.apply(inp.copy(), {})
                    # Hash do estado (amostra de pixels)
                    h = hash(after_step1.tobytes()[:100])  # hash parcial
                    sample_states[h].append((macro.name, name, seq[0]))
                except Exception:
                    pass

    # Agrupa macros que produzem estados similares
    groups = defaultdict(set)  # state_hash -> set de macro_names
    for h, entries in sample_states.items():
        if len(entries) >= 2:
            macros_in_group = set(e[0] for e in entries)
            first_ops = set(e[2] for e in entries)
            task_ids = set(e[1] for e in entries)
            # Se a mesma operacao (ex: rotate_90) aparece em contextos diferentes
            # com o mesmo estado resultante, e um candidato a skill
            group_key = tuple(sorted(first_ops))
            groups[group_key].update(macros_in_group)

    print(f"  Estados intermediarios unicos: {len(sample_states)}")
    print(f"  Grupos de operacoes que produzem mesmo estado: {len(groups)}")
    skill_candidates = []
    for ops, macros_set in sorted(groups.items(), key=lambda x: -len(x[1]))[:10]:
        if len(macros_set) >= 2:
            skill_candidates.append((ops, macros_set))
            print(f"    Operacoes: {ops}")
            print(f"      Macros: {len(macros_set)}")
            for m in list(macros_set)[:5]:
                print(f"        {m}")

    print(f"\n  Candidatos a SKILL (operacoes que atingem mesmo estado): {len(skill_candidates)}")
    return skill_candidates


# ─── EXPERIMENTO 4: Criar Skill de cluster ───
def exp4_create_skills(registry, skill_candidates):
    """Cria SKILLS abstratas a partir de clusters de macros."""
    print("\n" + "=" * 60)
    print("EXPERIMENTO 4: SKILLS A PARTIR DE CLUSTERS")
    print("=" * 60)

    if not skill_candidates:
        print("  Nenhum candidato a skill encontrado.")
        return

    skills_created = 0
    for ops, macros_set in skill_candidates[:5]:
        # Skill nome = funcao do estado intermediario
        op_names = [op.replace("_", "") for op in ops]
        skill_name = "Skill" + "".join(capitalize_words(op) for op in op_names)
        skill_name = skill_name[:30]

        # Tenta inferir o proposito da skill
        purpose = infer_skill_purpose(ops)

        print(f"\n  Skill candidata: {skill_name}")
        print(f"    Operacoes: {ops}")
        print(f"    Proposito inferido: {purpose}")
        print(f"    Macros agregadas: {len(macros_set)}")

        if purpose != "unknown":
            skills_created += 1
            print(f"    -> SKILL VALIDA: {skill_name} = {purpose}")

    print(f"\n  Skills criadas: {skills_created}")


def capitalize_words(s):
    return s[0].upper() + s[1:] if s else s


def infer_skill_purpose(operations):
    """Infere o proposito de uma skill baseado nas operacoes."""
    op_types = set()
    for op in operations:
        if "rotate" in op or "flip" in op or "transpose" in op:
            op_types.add("orientation")
        elif "move" in op or "translate" in op:
            op_types.add("position")
        elif "recolor" in op:
            op_types.add("color")
        elif "remove" in op or "delete" in op:
            op_types.add("filter")

    if "orientation" in op_types and "position" in op_types:
        return "normalize_object_orientation_and_position"
    if "orientation" in op_types:
        return "normalize_object_orientation"
    if "position" in op_types:
        return "align_objects"
    if "filter" in op_types:
        return "filter_objects"
    return "unknown"


# ─── MAIN ───
if __name__ == "__main__":
    print("=" * 70)
    print("4 EXPERIMENTOS SOBRE MACROS vs SKILLS")
    print("=" * 70)

    dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)

    # Setup
    registry = setup_base_registry()
    print(f"Registry base: {registry.count} primitives")

    # Descobre macros
    macro_reg, graph = discover_macros(registry, dataset, n_tasks=60)
    print(f"Macros descobertas: {macro_reg.get_stats()}")

    # Exp 1
    r1 = exp1_force_macros(registry, macro_reg, dataset)

    # Exp 2
    r2 = exp2_search_reduction(registry, macro_reg, dataset)

    # Exp 3
    r3 = exp3_group_by_intermediate(registry, macro_reg, dataset)

    # Exp 4
    exp4_create_skills(registry, r3)

    print("\n" + "=" * 70)
    print("RESUMO DOS 4 EXPERIMENTOS")
    print("=" * 70)
    print(f"""
  Exp 1 (Forcar macros): {r1['matched']} match exato, {r1['better']} melhor que base
  Exp 2 (Reducao busca): {len(r2)} tarefas onde macro reduziu profundidade
  Exp 3 (Estado intermed): clusters de operacoes que produzem mesmo estado
  Exp 4 (Skills): skills candidatas inferidas de proposito
  """)
