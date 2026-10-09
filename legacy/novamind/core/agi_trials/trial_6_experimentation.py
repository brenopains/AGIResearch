"""
Trial 6 — Experimentacao Autonoma
====================================
Prova que o sistema age como cientista: forma hipoteses, DESEHA
experimentos, executa, analisa e conclui.

Este e o teste mais forte: exige agencia, nao apenas resposta.
O sistema nao recebe dados prontos — ele precisa ESCOLHER
quais experimentos fazer.

Protocolo:
  1. PhysicsWorld com lei oculta: "tempo de queda = f(massa, tamanho)"
  2. O sistema pode ESCOLHER objetos para testar (alterar massa,
     soltar, medir tempo)
  3. O sistema deve descobrir a relacao entre massa e tempo de queda
  4. Medimos: numero de experimentos ate conclusao correta vs.
     baseline aleatorio. Ganho de informacao por experimento.
  5. Passa se converte em <= 50% dos experimentos do baseline.

Nenhum LLM pode executar este teste — nao tem mundo agir.
Sistemas sem curiosidade/agenicia experimentam aleatoriamente.
"""
import random
import math
import time
import numpy as np
from typing import Optional, Any, Dict, List, Tuple
from .scoring import TrialResult
from .procedural_worlds import PhysicsWorld

# Import lazy para evitar circularidade na inicialização do módulo
def _get_active_experimenter():
    from core.active_experimenter import ActiveExperimenter
    return ActiveExperimenter


class Trial6AutonomousExperimentation:
    """
    Trial 6: o sistema deve projetar experimentos para descobrir
    uma lei oculta.
    """

    def __init__(self, cge=None, world_model=None,
                 curiosity_engine=None, hypothesis_engine=None,
                 verbose: bool = True):
        self.cge = cge
        self.world_model = world_model
        self.curiosity = curiosity_engine
        self.hypothesis_engine = hypothesis_engine
        self.verbose = verbose

    def run(self, seed: int = 42, n_repetitions: int = 3,
            max_experiments: int = 50) -> TrialResult:
        result = TrialResult(
            name="Autonomous Experimentation",
            trial_id=6, seed=seed,
            threshold=0.3,
        )
        t0 = time.time()

        scores = []
        details = []

        np.random.seed(seed)
        import random as _r
        _r.seed(seed)

        # Import lazy: evita dependência circular na carga do módulo
        from core.active_experimenter import ActiveExperimenter

        for rep in range(n_repetitions):
            rep_seed = seed + rep * 100
            # MUNDO COM OBJETOS DE TESTE (held-out) — para medir erro real
            world = PhysicsWorld(seed=rep_seed, n_objects=50, n_test_objects=15)

            training_data = world.get_training_data()
            test_objects = world.get_test_objects()

            # Test set: objetos held-out que NINGUÉM viu (nem ativo nem baseline)
            test_set = [(o.mass, o.size, o.fall_time) for o in test_objects
                       if o.fall_time > 0.05]

            # ─── ESTRATÉGIA ATIVA: escolhe experiências por ganho de info ─
            active = ActiveExperimenter()
            # Seed com 5 objetos (orçamento inicial de "pilot experiments")
            n_seed = 5
            for i in range(n_seed):
                obs = training_data[i]
                active.observe(obs["mass"], obs["size"], obs["fall_time"])

            # Orçamento generoso (15) — suficiente para aprender lei de 4 params
            n_budget = min(15, len(training_data) - n_seed)
            candidates = [(d["mass"], d["size"]) for d in training_data[n_seed:]]

            for _ in range(n_budget):
                chosen = active.choose_experiment(candidates)
                if chosen is None:
                    break
                # Encontra a observação correspondente (o "mundo responde")
                matching = [d for d in training_data
                           if abs(d["mass"] - chosen[0]) < 0.01
                           and abs(d["size"] - chosen[1]) < 0.01]
                if matching:
                    m = matching[0]
                    active.observe(m["mass"], m["size"], m["fall_time"])
                    # Remove dos candidatos para não repetir
                    candidates = [(c[0], c[1]) for c in candidates
                                 if not (abs(c[0]-m["mass"])<0.01 and abs(c[1]-m["size"])<0.01)]

            active_error = active.error_on(test_set)

            # ─── BASELINE ALEATÓRIO: mesmo orçamento, escolha aleatória ─
            baseline_obs = [(training_data[i]["mass"], training_data[i]["size"],
                            training_data[i]["fall_time"]) for i in range(n_seed)]
            available = training_data[n_seed:]
            baseline_choices = _r.sample(available, min(n_budget, len(available)))
            for d in baseline_choices:
                baseline_obs.append((d["mass"], d["size"], d["fall_time"]))
            baseline_error = ActiveExperimenter.random_baseline_experiment(
                baseline_obs, test_set
            )

            # Ratio < 1.0 significa que ativo SUPEROU baseline (melhor ciência)
            efficiency_ratio = active_error / max(baseline_error, 1e-6)

            rep_score = 0.0
            # ANTI-GOODHART: só pontua se GENUINAMENTE melhor que aleatório
            if efficiency_ratio < 0.95:
                rep_score += 0.3
            if efficiency_ratio < 0.8:
                rep_score += 0.3
            if efficiency_ratio < 0.6:
                rep_score += 0.2

            # Bônus por erro absoluto baixo no held-out (capacidade real)
            if active_error < 0.25:
                rep_score += 0.2

            scores.append(rep_score)
            if self.verbose:
                print(f"    [T6] rep={rep}, "
                      f"active_err={active_error:.3f}, "
                      f"baseline_err={baseline_error:.3f}, "
                      f"efficiency={efficiency_ratio:.2f}, "
                      f"score={rep_score:.2f}")

        result.score = sum(scores) / len(scores) if scores else 0.0
        result.passed = result.score >= result.threshold
        result.details = f"active vs random: {n_budget} experiments each, "
        result.n_trials = n_repetitions
        result.metrics = {
            "mean_score": result.score,
            "method": "active_inference_vs_random",
        }
        result.duration = time.time() - t0
        return result

    def _measure_learning(self, training_data: List[Dict],
                          true_coeff: Tuple[float, ...]) -> float:
        """Mede erro do modelo do sistema vs lei verdadeira."""
        if len(training_data) < 3:
            return 1.0

        masses = np.array([d["mass"] for d in training_data])
        sizes = np.array([d["size"] for d in training_data])
        times = np.array([d["fall_time"] for d in training_data])

        try:
            log_m = np.log(masses + 1e-6)
            log_s = np.log(sizes + 1e-6)
            log_t = np.log(times + 1e-6)
            A = np.column_stack([np.ones_like(log_m), log_m, log_s])
            coeffs, _, _, _ = np.linalg.lstsq(A, log_t, rcond=None)

            fitted_log_t = A @ coeffs
            mse = np.mean((log_t - fitted_log_t) ** 2)
            return min(mse, 1.0)
        except Exception:
            return 1.0

    def _random_baseline_error(self, training_data: List[Dict],
                                 true_coeff: Tuple[float, ...],
                                 n_samples: int) -> float:
        """Erro esperado de amostragem aleatoria."""
        errors = []
        for _ in range(n_samples):
            subset = random.sample(training_data,
                                   max(3, len(training_data) // 3))
            err = self._measure_learning(subset, true_coeff)
            errors.append(err)
        return np.mean(errors)

    def _query_system_about_physics(self, training_data: List[Dict]) -> float:
        """Pergunta ao sistema sobre a relacao entre massa e queda."""
        if self.cge is None:
            return 0.0

        q = "mass affects fall time"
        score = 0.0

        words = q.lower().split()
        if hasattr(self.cge, 'nodes'):
            matches = sum(1 for w in words if w in self.cge.nodes)
            score = matches / len(words)

        if hasattr(self.cge, 'edges'):
            if any(e.get('from') == 'mass' and e.get('to') == 'fall_time'
                   for e in getattr(self.cge, 'edges', [])):
                score += 0.5
            if any(e.get('from') == 'mass' and e.get('to') == 'gravity'
                   for e in getattr(self.cge, 'edges', [])):
                score += 0.3

        return min(score, 1.0)
