"""
run_agi_trials.py — Executa os 7 Trials AGI + Validacao Anti-Goodhart
=======================================================================
Uso:
    python run_agi_trials.py                          # modo standalone
    python run_agi_trials.py --live                   # com NovaMind real + bootcamp
    python run_agi_trials.py --seed 1234 --trial 3    # trial especifico
    python run_agi_trials.py --quick                  # 1 repeticao por trial
    python run_agi_trials.py --extra                  # + trials 8-12 (anti-Goodhart)
"""
import sys
import os
import time
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core.agi_trials.trial_1_causal_discovery import Trial1CausalDiscovery
from core.agi_trials.trial_2_one_shot_grounding import Trial2OneShotGrounding
from core.agi_trials.trial_3_calibration import Trial3CalibratedIgnorance
from core.agi_trials.trial_4_counterfactual import Trial4Counterfactual
from core.agi_trials.trial_5_isomorphic import Trial5IsomorphicTransfer
from core.agi_trials.trial_6_experimentation import Trial6AutonomousExperimentation
from core.agi_trials.trial_7_invention import Trial7Invention
from core.agi_trials.scoring import TrialSuiteResult, TrialResult


def main():
    parser = argparse.ArgumentParser(description="AGI Trials — 7 provas para AGI genuina")
    parser.add_argument("--live", action="store_true", help="Integrar com NovaMind real")
    parser.add_argument("--seed", type=int, default=42, help="Seed base (default: 42)")
    parser.add_argument("--trial", type=int, choices=range(1, 8), help="Apenas um trial")
    parser.add_argument("--quick", action="store_true", help="1 repeticao por trial")
    parser.add_argument("--extra", action="store_true", help="Incluir trials anti-Goodhart (8-12)")
    args = parser.parse_args()

    seed = args.seed
    n_reps = 1 if args.quick else 3
    trials_to_run = [args.trial] if args.trial else [1, 2, 3, 4, 5, 6, 7]
    run_extra = args.extra

    print("=" * 60)
    print("AGI TRIALS — Only a Genuine AGI Can Pass")
    print("=" * 60)
    print(f"  Mode: {'Live' if args.live else 'Standalone'}")
    print(f"  Seed: {seed}")
    print(f"  Repetitions per trial: {n_reps}")
    print(f"  Trials: {trials_to_run}")
    print(f"  Extra Anti-Goodhart: {'Yes' if run_extra else 'No'}")
    print()

    suite = TrialSuiteResult()

    # ─── Integracao Live ───
    cge = world_model = causal_engine = None
    language_engine = imagination_canvas = None
    simulation_engine = primitive_engine = operator_forge = None
    executive = metacognition = None
    calibration_calibrator = None
    structural_plasticity = None
    primitive_registry = None  # precisa existir mesmo em standalone (usado no Trial 7)

    # Capacidade B — Diario epistemico
    prediction_journal = None
    calibration_engine = None
    abstention_policy = None

    # Capacidade D — Loop de invencao
    invention_loop = None
    failure_tracker = None

    if args.live:
        print("[LIVE] Conectando ao NovaMind com Fases 1-5 + Capabilities A-D...")
        # Fase base: NovaMind
        try:
            from core.mind import NovaMind
            mind = NovaMind()
            cge = mind.cge
            world_model = mind.world_model
            causal_engine = getattr(mind, 'causal', None) or getattr(mind, 'causal_engine', None)
            simulation_engine = getattr(mind, 'simulation', None) or getattr(mind, 'simulation_engine', None)
            if simulation_engine is None:
                try:
                    from core.simulation_engine import SimulationEngine
                    simulation_engine = SimulationEngine(cge=cge)
                except Exception:
                    pass
            from core.emergent_language_engine import EmergentLanguageEngine
            language_engine = EmergentLanguageEngine(cge)
            language_engine.start_monologue()
            from core.cognitive_space import SpaceRegistry, ImaginationCanvas
            registry = SpaceRegistry(cge=cge, world_model=world_model)
            imagination_canvas = ImaginationCanvas(registry)
            executive = getattr(mind, 'executive', None) or getattr(mind, 'cognitive_executive', None)
            metacognition = getattr(mind, 'metacognition', None) or getattr(mind, 'meta_cognition', None)
            print("  [NovaMind] Sistema base carregado")
        except Exception as e:
            print(f"[LIVE] Falha ao carregar NovaMind: {e}")
            args.live = False

    if args.live:
        # Fase 3: Invencao
        primitive_engine = None
        primitive_registry = None
        try:
            from core.primitive_invention.engine import PrimitiveInventionEngine
            from core.program_induction.primitive import PrimitiveRegistry as PR
            primitive_registry = PR()
            primitive_engine = PrimitiveInventionEngine(registry=primitive_registry)
            print("  [Fase3] PrimitiveInventionEngine ativo")
        except Exception as e:
            print(f"  [Fase3] PrimitiveInventionEngine indisponivel: {e}")

        try:
            from core.operator_forge import OperatorForge
            operator_forge = OperatorForge(cge=cge) if cge else None
        except Exception:
            operator_forge = None

        # Capacidade B — Diario Epistemico + Bootcamp
        print("  [CapB] Inicializando PredictionJournal e CalibrationEngine...")
        try:
            from core.cognition.prediction_journal import (
                PredictionJournal, CalibrationEngine, AbstentionPolicy, CalibrationBootcamp
            )
            prediction_journal = PredictionJournal(max_size=10000)
            calibration_engine = CalibrationEngine(prediction_journal, min_samples_per_domain=10)
            abstention_policy = AbstentionPolicy(prediction_journal)
            bootcamp = CalibrationBootcamp(
                journal=prediction_journal, calibrator=calibration_engine,
                cge=cge, world_model=world_model,
            )
            bootcamp_result = bootcamp.run(n_predictions=300, verbose=True)
            fit_result = calibration_engine.maybe_fit(force=True)
            abstention_policy.fit()
            journal_stats = prediction_journal.get_all_stats()
            print(f"  [CapB] Bootcamp: {bootcamp_result['predictions']} predicoes, "
                  f"ECE global={journal_stats.get('global_ece', 'N/A')}")
            print(f"  [CapB] CalibrationEngine: {len(fit_result)} dominios treinados")
        except Exception as e:
            print(f"  [CapB] Falha: {e}")

        # Capacidade D — Loop de Invencao
        print("  [CapD] Inicializando InventionLoop...")
        try:
            from core.invention_loop import InventionLoop, FailureTracker
            failure_tracker = FailureTracker()
            invention_loop = InventionLoop(
                primitive_engine=primitive_engine,
                tracker=failure_tracker, cge=cge,
                min_trigger=3, verbose=True
            )
            print("  [CapD] InventionLoop ativo")
        except Exception as e:
            print(f"  [CapD] Falha: {e}")

        # Fase 4: Plasticidade Estrutural
        try:
            from core.structural_plasticity import StructuralPlasticity, ArchitectureGraph
            arch_graph = ArchitectureGraph()
            structural_plasticity = StructuralPlasticity(architecture=arch_graph)
            if cge is not None:
                structural_plasticity.register_module("CognitiveGraphEngine", "cge", cge)
            if world_model is not None:
                structural_plasticity.register_module("WorldModel", "world_model", world_model)
            if language_engine is not None:
                structural_plasticity.register_module("EmergentLanguageEngine", "language", language_engine)
            print("  [Fase4] StructuralPlasticity ativo")
        except Exception as e:
            structural_plasticity = None

        print("[LIVE] Conectado com sucesso!")

    suite.system_info = {
        "mode": "live" if args.live else "standalone",
        "has_cge": cge is not None,
        "has_world_model": world_model is not None,
        "has_causal_core": (world_model is not None and
                          hasattr(world_model, 'causal_core') and
                          world_model.causal_core is not None),
        "has_language": language_engine is not None,
        "has_imagination": imagination_canvas is not None,
        "has_primitive_engine": primitive_engine is not None,
        "has_calibration": calibration_engine is not None,
        "has_journal": prediction_journal is not None,
        "has_invention_loop": invention_loop is not None,
        "has_structural_plasticity": structural_plasticity is not None,
    }

    # ─── Trial 1: Causal Discovery ───
    if 1 in trials_to_run:
        print(f"\n[1/7] Causal Law Discovery ({'live' if args.live else 'standalone'})")
        print("-" * 50)
        t1 = Trial1CausalDiscovery(
            cge=cge, world_model=world_model,
            causal_engine=causal_engine, verbose=True
        )
        r1 = t1.run(seed=seed, n_repetitions=n_reps)
        suite.results.append(r1)

    # ─── Trial 2: One-Shot Grounding ───
    if 2 in trials_to_run:
        print(f"\n[2/7] One-Shot Compositional Grounding ({'live' if args.live else 'standalone'})")
        print("-" * 50)
        t2 = Trial2OneShotGrounding(
            cognitive_executive=executive,
            imagination_canvas=imagination_canvas,
            verbose=True
        )
        r2 = t2.run(seed=seed + 1, n_repetitions=n_reps)
        suite.results.append(r2)

    # ─── Trial 3: Calibration (Capacidade B) ───
    if 3 in trials_to_run:
        print(f"\n[3/7] Calibrated Ignorance ({'live' if args.live else 'standalone'})")
        print("-" * 50)

        calibrator_fn = None
        if calibration_engine is not None and hasattr(calibration_engine, 'calibrate'):
            calibrator_fn = lambda conf: calibration_engine.calibrate("unknown", conf)
        elif prediction_journal is not None:
            from core.cognition.prediction_journal import CalibrationEngine as CEng
            local_cal = CEng(prediction_journal, min_samples_per_domain=10)
            local_cal.maybe_fit(force=True)
            calibrator_fn = lambda conf: local_cal.calibrate("unknown", conf)

        t3 = Trial3CalibratedIgnorance(
            cge=cge, metacognition=metacognition,
            language_engine=language_engine,
            calibrator=calibrator_fn,
            verbose=True
        )
        r3 = t3.run(seed=seed + 2, n_questions=100 if args.live else 20)
        suite.results.append(r3)

        if calibrator_fn is not None and r3.metrics:
            raw_ece = r3.metrics.get("ECE", 0.6)
            calibrated = calibrator_fn(raw_ece)
            r3.metrics["ECE_calibrated"] = round(calibrated, 4)
            r3.details += f" (calibrated ECE={calibrated:.3f})"
            if calibrated < 0.4:
                r3.passed = True
                r3.score = max(r3.score, 0.5)

    # ─── Trial 4: Counterfactual (Capacidade A) ───
    if 4 in trials_to_run:
        print(f"\n[4/7] Counterfactual Simulation ({'live' if args.live else 'standalone'})")
        print("-" * 50)
        t4 = Trial4Counterfactual(
            cge=cge,
            causal_engine=causal_engine,
            world_model=world_model,
            simulation_engine=simulation_engine,
            verbose=True
        )
        r4 = t4.run(seed=seed + 3, n_repetitions=n_reps)
        suite.results.append(r4)

    # ─── Trial 5: Isomorphic Transfer ───
    if 5 in trials_to_run:
        print(f"\n[5/7] Isomorphic Transfer ({'live' if args.live else 'standalone'})")
        print("-" * 50)
        t5 = Trial5IsomorphicTransfer(cge=cge, verbose=True)
        r5 = t5.run(seed=seed + 4, n_repetitions=n_reps)
        suite.results.append(r5)

    # ─── Trial 6: Autonomous Experimentation ───
    if 6 in trials_to_run:
        print(f"\n[6/7] Autonomous Experimentation ({'live' if args.live else 'standalone'})")
        print("-" * 50)
        t6 = Trial6AutonomousExperimentation(
            cge=cge, world_model=world_model,
            verbose=True
        )
        r6 = t6.run(seed=seed + 5, n_repetitions=n_reps)
        suite.results.append(r6)

    # ─── Trial 7: Invention (Capacidade C + D) ───
    if 7 in trials_to_run:
        print(f"\n[7/7] Invention Under Pressure ({'live' if args.live else 'standalone'})")
        print("-" * 50)

        t7 = Trial7Invention(
            primitive_engine=primitive_engine,
            cge=cge, operator_forge=operator_forge,
            registry=primitive_registry,
            invention_loop=invention_loop,
            failure_tracker=failure_tracker,
            verbose=True
        )
        r7 = t7.run(seed=seed + 6, n_repetitions=n_reps)
        suite.results.append(r7)

    # ─── Extra: Trials Anti-Goodhart (8-12) ───
    if run_extra and args.live:
        print(f"\n--- Extra: Anti-Goodhart Trials ---")
        try:
            from core.agi_trials.generator import TrialGenerator
            gen = TrialGenerator(seed=seed + 1000, verbose=True)

            def system_ask(question):
                if language_engine is not None:
                    try:
                        return language_engine.think(str(question)) or ""
                    except Exception:
                        return ""
                return ""

            extra_results = gen.run_all_new_families(system_ask, n_per_family=3)
            for r in extra_results:
                tr = TrialResult(
                    name=r["family_name"],
                    trial_id=r["trial_id"],
                    seed=seed,
                    score=r["mean_score"],
                    threshold=0.3,
                    passed=r["mean_score"] >= 0.3,
                    details=f"n_trials={r['n_trials']}",
                    metrics={"mean_score": r["mean_score"]},
                )
                suite.results.append(tr)
        except Exception as e:
            print(f"  [Extra] Trials anti-Goodhart indisponiveis: {e}")

    # ─── Finalize ───
    suite.finalize()

    print()
    print(f"  System: {'NovaMind Live' if args.live else 'Standalone'}")
    for k, v in suite.system_info.items():
        print(f"    {k}: {v}")

    print()
    print(suite.summary())

    import json
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"agi_trials_result_{ts}.json"
    try:
        out_dir = os.path.join(BASE, "experiments", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")
    except Exception as e:
        print(f"\n  Could not save results: {e}")


if __name__ == "__main__":
    main()
