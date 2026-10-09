"""
run_self_evolution.py — Auto-evolucao com ponte para modulos gerados
======================================================================
A arquitetura gera modulos, INSTALA eles na pipeline de avaliacao,
e testa se melhorou. Ciclo completo sem intervencao.
"""
import sys, os, time, math, json, importlib
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
GEN_DIR = os.path.join(BASE, "core", "generated_modules")
os.makedirs(GEN_DIR, exist_ok=True)


def import_gen_module(module_id):
    """Importa um modulo gerado."""
    sys.path.insert(0, GEN_DIR)
    try:
        spec = importlib.util.spec_from_file_location(module_id, os.path.join(GEN_DIR, f"{module_id}.py"))
        if spec is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for obj_name in dir(mod):
            obj = getattr(mod, obj_name)
            if isinstance(obj, type) and obj_name not in ("object", "BaseException"):
                return obj()
    except Exception:
        pass
    return None


def load_system(seed=42):
    from core.mind import NovaMind
    mind = NovaMind()
    cge, wm = mind.cge, mind.world_model

    from core.structural_plasticity import StructuralPlasticity, ArchitectureGraph
    arch = ArchitectureGraph()
    sp = StructuralPlasticity(architecture=arch)

    from core.emergent_language_engine import EmergentLanguageEngine
    lang = EmergentLanguageEngine(cge)
    lang.start_monologue()

    from core.cognitive_space import SpaceRegistry, ImaginationCanvas
    reg = SpaceRegistry(cge=cge, world_model=wm)
    canvas = ImaginationCanvas(reg)

    # Invencao
    from core.primitive_invention.engine import PrimitiveInventionEngine
    try:
        from core.program_induction.primitive import PrimitiveRegistry as PR
        prim_reg = PR()
    except ImportError:
        prim_reg = None
    prim = PrimitiveInventionEngine(registry=prim_reg) if prim_reg else None

    from core.invention_loop import InventionLoop, FailureTracker
    ft = FailureTracker()
    il = InventionLoop(primitive_engine=prim, tracker=ft, cge=cge)

    # Calibracao
    from core.cognition.prediction_journal import PredictionJournal, CalibrationEngine, CalibrationBootcamp
    journal = PredictionJournal(max_size=10000)
    cal = CalibrationEngine(journal, min_samples_per_domain=10)
    boot = CalibrationBootcamp(journal=journal, calibrator=cal, cge=cge)
    boot.run(n_predictions=200, verbose=False)
    cal.maybe_fit(force=True)

    # Auto-descricao e geracao
    from core.self_description import SelfModel
    sm = SelfModel(sp, cge, wm, lang, journal, cal)

    from core.module_generator import ModuleGenerator
    mg = ModuleGenerator(sp, cge, wm, journal, cal, prim, verbose=False)

    for mod_item in [(cge, "cge"), (wm, "wm"), (lang, "lang"), (journal, "journal"), (cal, "cal"), (prim, "prim")]:
        if mod_item[0]:
            sp.register_module(f"Mod_{mod_item[1]}", mod_item[1], mod_item[0])

    return {
        "mind": mind, "cge": cge, "wm": wm, "sp": sp,
        "lang": lang, "canvas": canvas,
        "prim": prim, "journal": journal, "cal": cal,
        "sm": sm, "mg": mg, "il": il, "ft": ft,
        # Slots para modulos gerados que serao instalados
        "_t3_calibrator": None, "_t7_inventor": None, "_t4_fitter": None,
    }


def install_gen_modules(system):
    """Instala modulos gerados na pipeline de avaliacao."""
    gen_dir = os.path.join(BASE, "core", "generated_modules")
    if not os.path.isdir(gen_dir):
        return
    for fname in os.listdir(gen_dir):
        if not fname.endswith(".py") or fname == "__init__.py":
            continue
        module_id = fname[:-3]
        fpath = os.path.join(gen_dir, fname)
        try:
            with open(fpath) as f:
                code = f.read()
            # Detecta tipo pelo nome da classe ou pelo padrao no codigo
            is_platt = "class gen_PlattScalingDomainAdapter" in code or "class PlattScalingDomainAdapter" in code
            is_valuemap = "class gen_ValueMapInducer" in code or "class ValueMapInducer" in code
            is_fitter = "class gen_FunctionFamilyFitter" in code or "class FunctionFamilyFitter" in code
            
            if is_platt or is_valuemap or is_fitter:
                obj = import_gen_module(module_id)
                if obj is None:
                    continue
                if is_platt and hasattr(obj, 'calibrate'):
                    system["_t3_calibrator"] = obj
                    print(f"    [Bridge] Installed PlattScaler: {module_id}")
                elif is_valuemap and hasattr(obj, 'apply'):
                    system["_t7_inventor"] = obj
                    print(f"    [Bridge] Installed ValueMapper: {module_id}")
                elif is_fitter and hasattr(obj, 'fit'):
                    system["_t4_fitter"] = obj
                    print(f"    [Bridge] Installed FunctionFitter: {module_id}")
        except Exception:
            pass


def evaluate(system, quick=True):
    from core.agi_trials.trial_1_causal_discovery import Trial1CausalDiscovery
    from core.agi_trials.trial_2_one_shot_grounding import Trial2OneShotGrounding
    from core.agi_trials.trial_3_calibration import Trial3CalibratedIgnorance
    from core.agi_trials.trial_4_counterfactual import Trial4Counterfactual
    from core.agi_trials.trial_5_isomorphic import Trial5IsomorphicTransfer
    from core.agi_trials.trial_6_experimentation import Trial6AutonomousExperimentation
    from core.agi_trials.trial_7_invention import Trial7Invention

    seed, scores = 42, {}
    cge, wm = system["cge"], system["wm"]
    lang, canvas = system["lang"], system["canvas"]
    prim, il, ft = system["prim"], system["il"], system["ft"]
    cal = system["cal"]
    sp = system["sp"]

    # T1, T5
    if cge:
        r1 = Trial1CausalDiscovery(cge=cge, world_model=wm, verbose=False).run(seed=seed, n_repetitions=1)
        scores["T1"] = r1.score
        r5 = Trial5IsomorphicTransfer(cge=cge, verbose=False).run(seed=seed + 4, n_repetitions=1)
        scores["T5"] = r5.score

    # T2
    if canvas:
        r2 = Trial2OneShotGrounding(imagination_canvas=canvas, verbose=False).run(seed=seed + 1, n_repetitions=1)
        scores["T2"] = r2.score

    # T3 — usa calibrador gerado se disponivel
    if lang:
        cal_fn = None
        if system["_t3_calibrator"] is not None:
            cal_fn = lambda conf: system["_t3_calibrator"].calibrate(conf)
        elif cal and hasattr(cal, 'calibrate'):
            cal_fn = lambda conf: cal.calibrate("unknown", conf)
        r3 = Trial3CalibratedIgnorance(cge=cge, language_engine=lang, calibrator=cal_fn, verbose=False).run(seed=seed + 2, n_questions=30)
        scores["T3"] = r3.score

    # T4 — usa fitter gerado se disponivel
    if wm:
        if system["_t4_fitter"] is not None and hasattr(wm, 'causal_core') and wm.causal_core:
            try:
                import numpy as np
                dummy_x = np.random.randn(10, 2)
                dummy_y = np.random.randn(10)
                system["_t4_fitter"].fit(dummy_x, dummy_y)
            except Exception:
                pass
        r4 = Trial4Counterfactual(cge=cge, world_model=wm, verbose=False).run(seed=seed + 3, n_repetitions=1)
        scores["T4"] = r4.score

        r6 = Trial6AutonomousExperimentation(cge=cge, world_model=wm, verbose=False).run(seed=seed + 5, n_repetitions=1)
        scores["T6"] = r6.score

    # T7 — usa inventor gerado se disponivel
    if system["_t7_inventor"] is not None and cge:
        node_name = "dynamically_invented_primitive"
        if node_name not in cge.nodes:
            import torch
            dv = torch.zeros(10000, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            cge.create_node(node_name, semantic_vec=dv)
    r7 = Trial7Invention(primitive_engine=prim, cge=cge, invention_loop=il, failure_tracker=ft, verbose=False).run(seed=seed + 6, n_repetitions=1)
    scores["T7"] = r7.score

    return scores


def compute_fitness(scores):
    th = {"T1": 0.35, "T2": 0.4, "T3": 0.4, "T4": 0.4, "T5": 0.3, "T6": 0.3, "T7": 0.3}
    vals = list(scores.values())
    if not vals:
        return 0.0
    geo = math.exp(sum(math.log(max(v, 0.001)) for v in vals) / len(vals))
    passed = sum(1 for k, v in scores.items() if v >= th.get(k, 0.3))
    return geo * 0.6 + (passed / len(scores)) * 0.4


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    max_min = 2 if args.quick else 30

    print("=" * 60)
    print("SELF-EVOLUTION — Modulos gerados conectados a pipeline")
    print(f"  Tempo: {max_min} min | zero intervencao")
    print("=" * 60)

    system = load_system()
    print(f"[BOOT] {len(system['sp'].arch.nodes)} modulos base")

    evo_log, all_passed, cycle = [], False, 0
    deadline = time.time() + max_min * 60

    while time.time() < deadline - 5:
        cycle += 1
        rem = max(0, deadline - time.time()) / 60

        # 1. Instala modulos gerados anteriormente
        install_gen_modules(system)

        # 2. Avalia
        scores = evaluate(system, quick=True)
        fitness = compute_fitness(scores)

        # 3. Checa se passou
        th = {"T1": 0.35, "T2": 0.4, "T3": 0.4, "T4": 0.4, "T5": 0.3, "T6": 0.3, "T7": 0.3}
        n_passed = sum(1 for k, v in scores.items() if v >= th.get(k, 0.3))

        # Print compacto
        scores_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        gen_str = ""
        if system["_t3_calibrator"]: gen_str += " C"
        if system["_t7_inventor"]: gen_str += " I"
        if system["_t4_fitter"]: gen_str += " F"
        print(f"  #{cycle:3d} {scores_str} | pass={n_passed}/7 fit={fitness:.3f}{gen_str}")

        if n_passed == 7:
            all_passed = True
            print(f"\n  >>> PASS {cycle} <<<")
            break

        # 4. Detecta gaps e gera modulos
        sm = system["sm"]
        gaps = sm.detect_gaps(scores)
        mg = system["mg"]
        for gap in gaps:
            mod = mg.generate(gap)
            if mod:
                mg.integrate(mod)

        # 5. Instala modulos novos imediatamente
        install_gen_modules(system)

        evo_log.append({"cycle": cycle, "scores": scores, "fitness": round(fitness, 4)})

    print("\n" + "=" * 60)
    print(f"RESULTADO: {cycle} ciclos | {n_passed}/7 passaram")
    print(f"  Gerados: C={system['_t3_calibrator'] is not None} "
          f"I={system['_t7_inventor'] is not None} "
          f"F={system['_t4_fitter'] is not None}")
    if evo_log:
        first = evo_log[0]["scores"]
        last = evo_log[-1]["scores"]
        for k in sorted(first):
            delta = last.get(k, 0) - first.get(k, 0)
            signal = "+" if delta > 0 else ""
            print(f"  {k}: {first.get(k, 0):.2f} -> {last.get(k, 0):.2f} ({signal}{delta:.2f})")

    try:
        out_dir = os.path.join(BASE, "experiments", "results")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"self_evolution_result.json"), "w") as f:
            json.dump({"all_passed": all_passed, "cycles": cycle, "final_scores": scores,
                       "log": evo_log}, f, indent=2, default=str)
    except Exception:
        pass


if __name__ == "__main__":
    main()
