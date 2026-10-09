'\nrun_evolution.py — Evolucao arquitetural de 10 minutos\n=========================================================\nGenoma = dicionario de parametros arquiteturais chave.\nCada individuo e uma configuracao diferente.\nA evolucao testa, seleciona, muta, e LOGGA o que funciona.\n\nIsso nao e treino para os trials. E descoberta de quais\nparametros arquiteturais produzem capacidades gerais.\n\nUso:\n    python run_evolution.py                    # 10 min, populacao 6, 5 geracoes\n    python run_evolution.py --quick            # 3 min, 3 geracoes\n\nQUARENTENA (2026-07-28): auditoria de codigo encontrou dois defeitos que\ntornam a saida deste script NAO CONFIAVEL ate serem corrigidos:\n\n1. `evolution_run` le `result["geo_mean"]` e `result["pass_rate"]`\n   (por volta da linha 399), mas `evaluate_fitness` nao retorna essas\n   chaves — todo individuo cai no `except` e recebe fitness=0.0. A\n   evolucao roda como deriva aleatoria; `best_genome` fica sempre None.\n2. Os 12 parametros do genoma sao write-only: sao escritos em\n   `function_params["_config"]` mas nenhum modulo em core/ le essa\n   chave. Nenhuma mutacao tem efeito observavel no sistema.\n\nAte corrigir os dois pontos, resultados deste script (incluindo\n`evolution_result_*.json` salvos) sao ruido, nao evidencia de melhoria\narquitetural. Ver plano em\nC:\\Users\\usuario\\.claude\\plans\\c-users-usuario-downloads-leia-arquitetur-floofy-crab.md\n'
import sys, os, time, random, copy, json, math
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


# ─── GENOMA: parametros arquiteturais chave ───

DEFAULT_GENOME = {
    # CausalCore (T1, T4)
    "causal_poly_degree": 3,         # 2, 3, 4, 5
    "causal_noise_std_factor": 1.5,  # 0.5 - 3.0
    "causal_confidence_power": 0.5,  # 0.1 - 1.0 (peso da predicao vs slope)

    # Calibration (T3)
    "calibrator_min_samples": 10,    # 5 - 50
    "calibration_bootcamp_n": 300,   # 100 - 500
    "abstention_threshold": 0.3,     # 0.1 - 0.5

    # Invention (T7)
    "value_map_min_pixels": 5,       # 3 - 20
    "invention_trigger": 3,          # 2 - 5

    # General
    "world_model_n_obs": 100,        # 50 - 200
    "simulation_n_branches": 5,      # 3 - 10
    "prediction_confidence_boost": 0.1,  # 0.0 - 0.5
    "curiosity_temperature": 0.5,    # 0.2 - 1.0
}

PARAM_RANGES = {
    "causal_poly_degree": (2, 6),
    "causal_noise_std_factor": (0.5, 3.0),
    "causal_confidence_power": (0.1, 1.0),
    "calibrator_min_samples": (5, 50),
    "calibration_bootcamp_n": (100, 500),
    "abstention_threshold": (0.1, 0.5),
    "value_map_min_pixels": (3, 20),
    "invention_trigger": (2, 5),
    "world_model_n_obs": (50, 200),
    "simulation_n_branches": (3, 10),
    "prediction_confidence_boost": (0.0, 0.5),
    "curiosity_temperature": (0.2, 1.0),
}


def mutate_genome(genome: dict, rate: float = 0.3) -> dict:
    """Muta 1-2 parametros do genoma."""
    new = copy.deepcopy(genome)
    keys = list(PARAM_RANGES.keys())
    n_mutations = random.randint(1, 2)
    mutated = []
    for k in random.sample(keys, min(n_mutations, len(keys))):
        lo, hi = PARAM_RANGES[k]
        old_val = new[k]
        if isinstance(lo, int) and isinstance(hi, int):
            new_val = int(random.gauss(old_val, (hi - lo) * 0.15))
            new_val = max(lo, min(hi, new_val))
        else:
            new_val = old_val + random.uniform(-(hi - lo) * 0.15, (hi - lo) * 0.15)
            new_val = max(lo, min(hi, new_val))
        new[k] = new_val
        mutated.append((k, old_val, new_val))
    return new, mutated


def crossover_genome(a: dict, b: dict) -> dict:
    """Crossover uniforme entre dois genomas."""
    child = {}
    for k in PARAM_RANGES.keys():
        child[k] = random.choice([a[k], b[k]])
    return child


# ─── INTERFACE DE AVALIACAO ───

def _load_system_for_genome(genome: dict):
    """
    Carrega o sistema NovaMind base e aplica o genoma, retornando um dict
    com handles para os motores de fitness interna (FEP, curiosidade,
    compressao). Usado tanto pela fitness interna quanto pela held-out.
    """
    from core.mind import NovaMind
    mind = NovaMind()
    cge = mind.cge
    world_model = mind.world_model
    causal_engine = getattr(mind, 'causal', None)

    # Aplica genoma ao sistema (configuracoes arquiteturais)
    if world_model is not None and hasattr(world_model, 'causal_core') and world_model.causal_core is not None:
        world_model.causal_core.function_params["_config"] = genome

    # Monta os 3 motores de fitness INTERNA
    try:
        from core.math.information import FreeEnergyMinimizer
        fem = FreeEnergyMinimizer()
    except Exception:
        fem = None

    try:
        from core.compression_engine import CompressionEngine
        compressor = CompressionEngine(cge=cge)
    except Exception:
        compressor = None

    return {
        "mind": mind, "cge": cge, "world_model": world_model,
        "causal_engine": causal_engine, "fem": fem, "compressor": compressor,
        "genome": genome,
    }


def evaluate_fitness(genome: dict, trial_ids: list = None,
                     verbose: bool = False, seed: int = 42) -> dict:
    """
    FITNESS INTERNA — o que efetivamente seleciona individuos na evolucao.

    NAO usa os 7 trials AGI. Usa mundos procedurais DISJUNTOS dos trials
    e mede as 3 capacidades internas: FEP, curiosidade (learning progress),
    compressao (MDL). Os trials ficam 100% held-out (ver evaluate_held_out).

    Esta separacao e o que separa "AGI genuina" de "sistema que decorou
    benchmark". Se um individuo sobe no fitness interno E nos trials held-out,
    isso e evidencia real; se sobe so no fitness mas nao nos trials, foi
    overfit no treino.
    """
    from core.agi_trials.internal_fitness import compute_internal_fitness

    try:
        system = _load_system_for_genome(genome)
    except Exception as e:
        return {"fitness": 0.0, "error": f"system_load: {e}",
                "components": {}, "trial_scores": {}}

    # Pesos dos componentes podem fazer parte do genoma (evoluem), mas
    # sempre renormalizados para nao colapsar num so.
    weights = {
        "fep": 1.0 + genome.get("w_fep_bonus", 0.0),
        "curiosity": 1.0 + genome.get("w_curiosity_bonus", 0.0),
        "mdl": 1.0 + genome.get("w_mdl_bonus", 0.0),
    }

    result = compute_internal_fitness(
        system=system, n_worlds=8, seed=900_000, weights=weights
    )

    return {
        "fitness": result["fitness"],
        "components": result["components"],
        "weights": result["weights"],
        "trial_scores": {},  # DELIBERADAMENTE vazio: trials NAO entram no fitness
        "held_out": False,   # marcador explicito
    }


def evaluate_held_out(genome: dict, seed: int = 42,
                      trial_ids: list = None, quick: bool = True) -> dict:
    """
    HELD-OUT — roda os 7 trials AGI SEM retroalimentar a evolucao.

    Esta funcao NUNCA influencia a selecao/mutacao. Serve apenas para
    medir, periodicamente, se o fitness interna esta se traduzindo em
    capacidade AGI real. Rodar a cada K geracoes ou so no fim.

    Retorna o AGI score (geometrica) + breakdown por trial.
    """
    if trial_ids is None:
        trial_ids = [1, 2, 3, 4, 5, 6, 7]

    from core.agi_trials.trial_1_causal_discovery import Trial1CausalDiscovery
    from core.agi_trials.trial_2_one_shot_grounding import Trial2OneShotGrounding
    from core.agi_trials.trial_3_calibration import Trial3CalibratedIgnorance
    from core.agi_trials.trial_4_counterfactual import Trial4Counterfactual
    from core.agi_trials.trial_5_isomorphic import Trial5IsomorphicTransfer
    from core.agi_trials.trial_6_experimentation import Trial6AutonomousExperimentation
    from core.agi_trials.trial_7_invention import Trial7Invention
    from core.agi_trials.scoring import compute_agi_score

    try:
        system = _load_system_for_genome(genome)
    except Exception as e:
        return {"agi_score": 0.0, "error": f"system_load: {e}",
                "breakdown": {}, "held_out": True}

    cge = system["cge"]
    world_model = system["world_model"]
    causal_engine = system["causal_engine"]

    results = []
    scores = {}

    if 1 in trial_ids:
        t = Trial1CausalDiscovery(cge=cge, world_model=world_model, causal_engine=causal_engine, verbose=False)
        r = t.run(seed=seed, n_repetitions=1)
        results.append(r); scores["T1"] = r.score
    if 2 in trial_ids:
        t = Trial2OneShotGrounding(imagination_canvas=None, verbose=False)
        r = t.run(seed=seed + 1, n_repetitions=1)
        results.append(r); scores["T2"] = r.score
    if 3 in trial_ids:
        t = Trial3CalibratedIgnorance(cge=cge, language_engine=None, calibrator=None, verbose=False)
        r = t.run(seed=seed + 2, n_questions=20)
        results.append(r); scores["T3"] = r.score
    if 4 in trial_ids:
        t = Trial4Counterfactual(cge=cge, causal_engine=causal_engine, world_model=world_model, verbose=False)
        r = t.run(seed=seed + 3, n_repetitions=1)
        results.append(r); scores["T4"] = r.score
    if 5 in trial_ids:
        t = Trial5IsomorphicTransfer(cge=cge, verbose=False)
        r = t.run(seed=seed + 4, n_repetitions=1)
        results.append(r); scores["T5"] = r.score
    if 6 in trial_ids:
        t = Trial6AutonomousExperimentation(cge=cge, world_model=world_model, verbose=False)
        r = t.run(seed=seed + 5, n_repetitions=1)
        results.append(r); scores["T6"] = r.score
    if 7 in trial_ids:
        t = Trial7Invention(primitive_engine=None, cge=cge, verbose=False)
        r = t.run(seed=seed + 6, n_repetitions=1)
        results.append(r); scores["T7"] = r.score

    agi = compute_agi_score(results)
    return {
        "agi_score": agi["score"],
        "passed": agi["passed_count"],
        "total": agi["total_count"],
        "breakdown": {k: round(v, 4) for k, v in scores.items()},
        "held_out": True,
        "retro_feeds_evolution": False,  # contrato explicito
    }

    seed = 42

    # Carrega sistema base (uma vez, reusa)
    try:
        from core.mind import NovaMind
        mind = NovaMind()
        cge = mind.cge
        world_model = mind.world_model
        causal_engine = getattr(mind, 'causal', None)

        # Aplica genoma ao sistema
        if world_model is not None and hasattr(world_model, 'causal_core') and world_model.causal_core is not None:
            world_model.causal_core.function_params["_config"] = genome

        from core.emergent_language_engine import EmergentLanguageEngine
        language_engine = EmergentLanguageEngine(cge)
        language_engine.start_monologue()

        from core.cognitive_space import SpaceRegistry, ImaginationCanvas
        registry = SpaceRegistry(cge=cge, world_model=world_model)
        imagination_canvas = ImaginationCanvas(registry)

        from core.primitive_invention.engine import PrimitiveInventionEngine
        try:
            from core.program_induction.primitive import PrimitiveRegistry as PR
            primitive_registry = PR()
        except ImportError:
            primitive_registry = None
        primitive_engine = PrimitiveInventionEngine(registry=primitive_registry) if primitive_registry else None

        from core.invention_loop import InventionLoop, FailureTracker
        failure_tracker = FailureTracker()
        invention_loop = InventionLoop(primitive_engine=primitive_engine, tracker=failure_tracker, cge=cge)

        from core.cognition.prediction_journal import PredictionJournal, CalibrationEngine, CalibrationBootcamp
        journal = PredictionJournal(max_size=10000)
        cal_engine = CalibrationEngine(journal, min_samples_per_domain=max(5, int(genome["calibrator_min_samples"])))
        bootcamp = CalibrationBootcamp(journal=journal, calibrator=cal_engine, cge=cge)
        bootcamp.run(n_predictions=int(genome["calibration_bootcamp_n"]), verbose=False)
        cal_engine.maybe_fit(force=True)

    except Exception as e:
        return {"fitness": 0.0, "error": str(e), "trial_scores": {}}

    results = []
    trial_map = {}

    if 1 in trial_ids:
        t1 = Trial1CausalDiscovery(cge=cge, world_model=world_model, causal_engine=causal_engine, verbose=False)
        r1 = t1.run(seed=seed, n_repetitions=1)
        results.append(r1)
        trial_map["T1"] = r1.score

    if 2 in trial_ids:
        t2 = Trial2OneShotGrounding(imagination_canvas=imagination_canvas, verbose=False)
        r2 = t2.run(seed=seed + 1, n_repetitions=1)
        results.append(r2)
        trial_map["T2"] = r2.score

    if 3 in trial_ids:
        from core.cognition.prediction_journal import CalibrationEngine as CEng
        calibrator_fn = None
        if cal_engine is not None:
            calibrator_fn = lambda conf: cal_engine.calibrate("unknown", conf)
        t3 = Trial3CalibratedIgnorance(cge=cge, language_engine=language_engine, calibrator=calibrator_fn, verbose=False)
        r3 = t3.run(seed=seed + 2, n_questions=30)
        results.append(r3)
        trial_map["T3"] = r3.score

    if 4 in trial_ids:
        t4 = Trial4Counterfactual(cge=cge, causal_engine=causal_engine, world_model=world_model, verbose=False)
        r4 = t4.run(seed=seed + 3, n_repetitions=1)
        results.append(r4)
        trial_map["T4"] = r4.score

    if 5 in trial_ids:
        t5 = Trial5IsomorphicTransfer(cge=cge, verbose=False)
        r5 = t5.run(seed=seed + 4, n_repetitions=1)
        results.append(r5)
        trial_map["T5"] = r5.score

    if 6 in trial_ids:
        t6 = Trial6AutonomousExperimentation(cge=cge, world_model=world_model, verbose=False)
        r6 = t6.run(seed=seed + 5, n_repetitions=1)
        results.append(r6)
        trial_map["T6"] = r6.score

    if 7 in trial_ids:
        t7 = Trial7Invention(primitive_engine=primitive_engine, cge=cge, invention_loop=invention_loop, failure_tracker=failure_tracker, verbose=False)
        r7 = t7.run(seed=seed + 6, n_repetitions=1)
        results.append(r7)
        trial_map["T7"] = r7.score

    # Fitness = geometric mean com bonus por passar threshold
    scores = [r.score for r in results]
    passed = [r.passed for r in results]
    geo_mean = math.exp(sum(math.log(max(s, 0.001)) for s in scores) / len(scores))
    pass_rate = sum(passed) / len(passed)
    fitness = geo_mean * 0.6 + pass_rate * 0.4

    return {
        "fitness": round(fitness, 4),
        "geo_mean": round(geo_mean, 4),
        "pass_rate": round(pass_rate, 3),
        "trial_scores": trial_map,
    }


# ─── LOOP EVOLUCIONARIO ───

def evolution_run(pop_size: int = 6, generations: int = 5,
                  seed: int = 2026, max_minutes: float = 10) -> dict:
    """Loop principal de evolucao. Retorna ledger completo."""
    random.seed(seed)
    start_time = time.time()
    deadline = start_time + max_minutes * 60

    # Populacao inicial
    population = []
    for i in range(pop_size):
        if i == 0:
            genome = copy.deepcopy(DEFAULT_GENOME)
        else:
            genome, _ = mutate_genome(DEFAULT_GENOME, rate=0.5)
        population.append({"id": f"ind_{i:04d}", "genome": genome,
                           "fitness": 0.0, "eval_time": 0.0, "trial_scores": {}})

    mutation_ledger = []
    generation_log = []
    best_fitness = 0.0
    best_genome = None

    for gen in range(generations):
        elapsed = time.time() - start_time
        if elapsed > max_minutes * 60 - 30:  # 30s buffer
            print(f"  Tempo esgotado na geracao {gen}. Parando.")
            break

        print(f"\n  GERACAO {gen + 1} — tempo restante: {max(0, deadline - time.time())/60:.1f}min")
        gen_start = time.time()

        # Avalia fitness
        for ind in population:
            if ind["fitness"] > 0:
                continue
            try:
                t0 = time.time()
                result = evaluate_fitness(ind["genome"], verbose=False)
                ind["fitness"] = result["fitness"]
                ind["trial_scores"] = result.get("trial_scores", {})
                ind["eval_time"] = time.time() - t0
                ts = {k: round(v, 3) for k, v in ind["trial_scores"].items()}
                print(f"    {ind['id']}: fitness={result['fitness']:.4f} "
                      f"geo={result['geo_mean']:.4f} pass={result['pass_rate']:.2f} "
                      f"scores={ts} ({ind['eval_time']:.1f}s)")

                if result["fitness"] > best_fitness:
                    best_fitness = result["fitness"]
                    best_genome = copy.deepcopy(ind["genome"])
            except Exception as e:
                ind["fitness"] = 0.0
                ind["trial_scores"] = {}
                print(f"    {ind['id']}: ERROR {e}")

        # Se for ultima geracao, para aqui
        if gen >= generations - 1:
            break

        # Selecao por torneio
        tournament_size = min(3, pop_size)
        selected = []
        for _ in range(pop_size):
            pool = random.sample(population, tournament_size)
            pool.sort(key=lambda x: -x["fitness"])
            selected.append(pool[0])

        # Proxima geracao: metade filhos, metade mutation dos melhores
        next_pop = []
        sorted_pop = sorted(population, key=lambda x: -x["fitness"])
        next_pop.append(copy.deepcopy(sorted_pop[0]))  # elitismo (melhor)

        while len(next_pop) < pop_size:
            parent_a = random.choice(selected[:pop_size // 2 + 1])
            parent_b = random.choice(selected[:pop_size // 2 + 1])
            child_genome = crossover_genome(parent_a["genome"], parent_b["genome"])
            child_genome, changes = mutate_genome(child_genome, rate=0.3)
            for param, old_v, new_v in changes:
                mutation_ledger.append({
                    "generation": gen, "param": param,
                    "old": old_v, "new": new_v,
                    "parent_fitness": parent_a["fitness"],
                })
            next_pop.append({
                "id": f"ind_{gen}_{len(next_pop):04d}",
                "genome": child_genome, "fitness": 0.0,
                "eval_time": 0.0, "trial_scores": {},
            })

        population = next_pop[:pop_size]
        gen_time = time.time() - gen_start
        gen_avg = sum(i["fitness"] for i in population) / len(population)
        gen_log = {
            "generation": gen,
            "best_fitness": max(i["fitness"] for i in population),
            "avg_fitness": round(gen_avg, 4),
            "time": round(gen_time, 1),
            "n_individuals": len(population),
            "best_scores": {k: round(v, 3) for k, v in sorted_pop[0].get("trial_scores", {}).items()}
        }
        generation_log.append(gen_log)
        print(f"    >> Melhor fitness: {gen_log['best_fitness']:.4f} | "
              f"Media: {gen_log['avg_fitness']:.4f} | "
              f"Tempo: {gen_log['time']:.1f}s")

    # Resultado final
    result = {
        "runtime": round(time.time() - start_time, 1),
        "generations": len(generation_log),
        "best_fitness": best_fitness,
        "initial_fitness": 0.0,
        "final_best_scores": {},
        "mutation_ledger": mutation_ledger,
        "generation_log": generation_log,
    }
    if generation_log:
        result["initial_fitness"] = generation_log[0]["best_fitness"]
        result["final_best_scores"] = generation_log[-1].get("best_scores", {})

    return result


def print_ledger(ledger: list):
    """Imprime o mutation ledger."""
    if not ledger:
        print("    (sem mutacoes registradas)")
        return
    # Agrupa por parametro
    from collections import defaultdict
    by_param = defaultdict(list)
    for m in ledger:
        by_param[m["param"]].append(m)
    print("\n  --- MUTATION LEDGER (parametros mais mutados e seus deltas) ---")
    for param, mutations in sorted(by_param.items(), key=lambda x: -len(x[1])):
        freq = len(mutations)
        avg_delta = sum(m["new"] - m["old"] for m in mutations if isinstance(m["new"], (int, float))) / freq
        print(f"    {param}: {freq} mutacoes, delta medio {avg_delta:+.2f}")

    # Parametro com maior impacto
    print("\n  --- MELHORES MUTACOES (maior potencial) ---")
    # (Nao temos o fitness apos cada mutacao individual, apenas mostramos as que mais aconteceram)
    top_params = sorted(by_param.items(), key=lambda x: -len(x[1]))[:3]
    for param, mutations in top_params:
        vals = [(m["old"], m["new"]) for m in mutations]
        print(f"    {param}: {len(mutations)}x — ex: {vals[0]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evolucao arquitetural de 10 min")
    parser.add_argument("--quick", action="store_true", help="3 min, 3 geracoes")
    args = parser.parse_args()

    max_min = 3 if args.quick else 10
    pop = 4 if args.quick else 6
    gens = 4 if args.quick else 5

    print("=" * 60)
    print("EVOLUTION — Busca arquitetural por parametros AGI")
    print("=" * 60)
    print(f"  Populacao: {pop} | Geracoes: {gens} | Tempo max: {max_min}min")
    print(f"  Genoma: {len(DEFAULT_GENOME)} parametros arquiteturais")
    print()

    t0 = time.time()
    result = evolution_run(pop_size=pop, generations=gens, max_minutes=max_min)
    runtime = time.time() - t0

    print("\n" + "=" * 60)
    print("RESULTADOS DA EVOLUCAO")
    print("=" * 60)
    print(f"  Runtime: {result['runtime']:.1f}s / {max_min * 60:.0f}s alocados")
    print(f"  Geracoes completadas: {result['generations']}")
    print(f"  Fitness inicial: {result['initial_fitness']:.4f}")
    print(f"  Fitness final:   {result['best_fitness']:.4f}")
    delta = result["best_fitness"] - result["initial_fitness"]
    direcao = "sobe" if delta > 0 else "desce"
    print(f"  Delta: {delta:+.4f} ({direcao})")
    print(f"  Trials no melhor individuo: {result.get('final_best_scores', {})}")
    print()

    print_ledger(result["mutation_ledger"])

    print("\n" + "-" * 60)
    print("GENERATION LOG")
    for glog in result["generation_log"]:
        print(f"  Gen {glog['generation']+1}: best={glog['best_fitness']:.4f} "
              f"avg={glog['avg_fitness']:.4f} "
              f"trials={glog['best_scores']} {glog['time']:.1f}s")

    # Salva resultados
    try:
        out_dir = os.path.join(BASE, "experiments", "results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"evolution_result_{ts}.json"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Resultados salvos em: {fname}")
    except Exception as e:
        print(f"\n  (nao foi possivel salvar: {e})")


if __name__ == "__main__":
    main()
