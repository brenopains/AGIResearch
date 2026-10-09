('\ninternal_fitness.py — Fitness INTERNA, separada dos trials held-out\n======================================================================\nO problema que isto resolve\n---------------------------\nrun_evolution.py original usava os 7 trials AGI como funcao de fitness.\nIsso e Goodhart puro: otimizar para os trials = treinar para eles,\nmesmo que "devagar" via evolucao. Mesmo defeito dos LLMs em benchmark.\n\nEste modulo implementa fitness INTERNA, definida antes e independentemente\ndos trials. O sistema e selecionado por:\n  - reduzir energia livre (surpresa Bayesiana) em ambientes DIVERSOS\n  - aprender coisas novas (reducao de erro de predicao = curiosidade)\n  - comprimir o que aprendeu (principio da descricao minima, MDL)\n\nA fonte de experiencia e procedural_worlds.py com seeds DISJUNTAS\ndos trials held-out. Os trials nunca retroalimentam a selecao.\n\nAs 3 componentes se estabilizam mutuamente:\n  - FEP pune curiosidade que persegue ruido (ruido = surpresa alta)\n  - Curiosidade pune FEP que se esconde (esconderijo = zero aprendizado)\n  - MDL pune overfitting de qualquer uma (modelo decorado = caro)\n\nAutor: refatoracao metodologica — separar fitness de avaliacao.\n\nQUARENTENA (2026-07-28): verificado empiricamente que `compute_internal_fitness`\nretorna ~0.3301 independente do argumento `system` (testado com None, {},\ne dois genomas distintos — 4 casas decimais identicas). Duas causas:\n  - `_generate_fitness_worlds` chama `cw.get_observational_samples()`, que\n    nao existe em CausalWorld (o metodo real e `generate_observational`);\n    `hasattr(...)` e False, entao os 4 mundos causais sempre entram vazios\n    e sao pulados.\n  - Nenhum `predict_fn` e passado por `run_evolution.py`, entao FEP/curiosidade/\n    compressao caem nos fallbacks que medem propriedades do MUNDO (seed),\n    nao do sistema — `_measure_state_growth` usa `len(cge.nodes)`, que nao\n    depende do genoma.\nAte isso ser corrigido, este modulo nao mede fitness nenhuma; e uma\nconstante disfarcada. Ver plano em\nC:\\Users\\usuario\\.claude\\plans\\c-users-usuario-downloads-leia-arquitetur-floofy-crab.md\n').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

from .procedural_worlds import CausalWorld, PhysicsWorld


# Range de seeds EXCLUSIVAMENTE para fitness interna.
# IMPORTANTE: disjunto das seeds usadas nos 7 trials (que comecam em 42+).
# Isso garante que a evolucao nunca ve as superficies dos trials held-out.
FITNESS_SEED_BASE = 900_000  # 900000..900099  vs trials 42..48


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def measure_free_energy(system, worlds: List[Dict]) -> float:
    """
    Componente FEP: media da energia livre apos o sistema observar cada mundo.

    Retorna energia livre media (menor = melhor modelo). Se o sistema nao
    expoe FreeEnergyMinimizer, degrada para erro de predicao direto sobre
    os dados do mundo — ainda uma medida honesta de "quao surpreso ficou".

    `system` pode ser um dict com chaves opcionais:
        'fem'  -> FreeEnergyMinimizer com .free_energy_history
        'predict_fn' -> callable(world_dict) -> predicted_array
    Caso None, mede a dificuldade intrinseca do mundo (baseline honesto).
    """
    if system is None:
        system = {}

    fem = system.get("fem") if isinstance(system, dict) else getattr(system, "fem", None)
    predict_fn = system.get("predict_fn") if isinstance(system, dict) else None

    energies: List[float] = []

    for w in worlds:
        data = w.get("data") or w.get("samples") or []
        if not data:
            continue

        if fem is not None and hasattr(fem, "update_belief"):
            # Alimenta o sistema com observacoes do mundo e mede energia livre
            for d in data:
                obs = _to_obs_array(d)
                if obs is not None:
                    try:
                        fem.update_belief(w.get("name", "world"), obs)
                    except Exception:
                        pass
            if hasattr(fem, "free_energy_history") and fem.free_energy_history:
                # Media das ultimas N atualizacoes (estado pos-aprendizado)
                recent = fem.free_energy_history[-len(data):] if len(data) > 0 else fem.free_energy_history
                energies.append(_mean(recent))
                continue

        # Degracacao: mede erro de predicao direto. Sempre honesto.
        if predict_fn is not None:
            try:
                energies.append(_prediction_error(predict_fn, w))
                continue
            except Exception:
                pass

        # Sem sistema: baseline = surpresa intrinseca do mundo
        energies.append(_world_intrinsic_surprise(w))

    if not energies:
        return 1.0  # pior caso

    # Normaliza para [0,1]: energia livre alta -> perto de 1
    raw = _mean(energies)
    return min(1.0, max(0.0, raw / (raw + 1.0)))  # suave, nunca explode


def measure_learning_progress(system, worlds: List[Dict]) -> float:
    """
    Componente CURIOSIDADE (learning progress): reducao de erro de predicao
    ao longo da exposicao ao mundo. NAO e o erro absoluto — e o QUANTO o
    erro caiu. Isso e o que distingue "aprendeu" de "ja sabia" ou "persegue
    ruido".

    Retorna fracao [0,1]: 0 = nao aprendeu nada, ~0.3 = aprendizado real,
    valores altas indicam overfit (raro em mundos procedurais).
    """
    if system is None:
        system = {}
    predict_fn = system.get("predict_fn") if isinstance(system, dict) else None
    if predict_fn is None:
        # Sem preditor: mede se o estado interno do sistema cresceu de forma
        # coerente (nos/atomos ganhos). Heuristica honesta.
        return _measure_state_growth(system, worlds)

    progresses: List[float] = []
    for w in worlds:
        data = w.get("data") or w.get("samples") or []
        if len(data) < 6:
            continue
        mid = len(data) // 2
        first_half = data[:mid]
        second_half = data[mid:]
        # Treina implicitamente (predict_fn mantem estado); mede erro antes/depois
        err_before = _prediction_error_on(predict_fn, first_half)
        err_after = _prediction_error_on(predict_fn, second_half)
        progress = max(0.0, err_before - err_after)  # so conta reducao real
        progresses.append(progress)

    if not progresses:
        return 0.0
    return min(1.0, _mean(progresses) * 2.0)  # escala suave


def measure_compression(system, worlds: List[Dict]) -> float:
    """
    Componente MDL: quao comprimida ficou a representacao apos aprender.
    Mede razao de compressao do conhecimento adquirido. Um sistema que
    "decora" cada amola tem MDL alto; um que encontra leis tem MDL baixo.

    Retorna [0,1]: maior = melhor compressao.
    """
    if system is None:
        system = {}
    comp = system.get("compressor") if isinstance(system, dict) else getattr(system, "compressor", None)

    if comp is not None and hasattr(comp, "atoms"):
        n_atoms = len(comp.atoms)
        # Cada atomo substitui N passos; razao total de compressao
        total_ratio = sum(getattr(a, "compression_ratio", 1) for a in comp.atoms.values())
        # Normaliza: ~10 atomos com razao media 3 = compressao razoavel
        return min(1.0, total_ratio / 30.0)

    # Degracacao: mede se o sistema achou uma REGRA que explica os dados
    # em vez de memoriza-los. Ajusta polinomio de grau baixo; se descreve
    # bem os mundos com poucos parametros, houve compressao.
    return _measure_rule_compression(worlds)


# ─── Fitness combinada ───────────────────────────────────────────────

def compute_internal_fitness(system: Optional[dict],
                             n_worlds: int = 8,
                             seed: int = FITNESS_SEED_BASE,
                             weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Roda o sistema em mundos procedurais DISJUNTOS dos trials e retorna
    fitness interna. Os 7 trials nunca sao tocados aqui.

    Pesos default: alpha=FEP 0.4, beta=curiosidade 0.35, gamma=MDL 0.25.
    (Pesos poderao evoluir como parte do genoma — mas sempre positivos,
    somando 1, pra evitar colapso num componente so.)
    """
    if weights is None:
        weights = {"fep": 0.40, "curiosity": 0.35, "mdl": 0.25}
    # Renormaliza defensivamente
    wsum = sum(max(0.0, v) for v in weights.values()) or 1.0
    weights = {k: max(0.0, v) / wsum for k, v in weights.items()}

    worlds = _generate_fitness_worlds(n_worlds=n_worlds, seed=seed)

    # FEP: menor energia livre = melhor. Invertemos pra "maior = melhor".
    fe = measure_free_energy(system, worlds)
    fep_score = 1.0 - fe

    curiosity = measure_learning_progress(system, worlds)
    mdl = measure_compression(system, worlds)

    # Combinacao ponderada. Nenhum componente domina sozinho.
    fitness = (weights["fep"] * fep_score
               + weights["curiosity"] * curiosity
               + weights["mdl"] * mdl)

    return {
        "fitness": round(fitness, 4),
        "components": {
            "fep": round(fep_score, 4),        # 1 - energia livre normalizada
            "curiosity": round(curiosity, 4),   # progresso de aprendizado
            "mdl": round(mdl, 4),               # compressao alcancada
        },
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "n_worlds": n_worlds,
        "seed": seed,
        "disjoint_from_trials": True,
    }


# ─── Geracao de mundos de fitness (disjuntos dos trials) ────────────

def _generate_fitness_worlds(n_worlds: int = 8,
                             seed: int = FITNESS_SEED_BASE) -> List[Dict]:
    """
    Gera mundos CausalWorld/PhysicsWorld com seeds no range exclusivo
    de fitness. Nunca sobrepoe os seeds dos trials (42-48 por default).
    Metade causal, metade fisica — diversidade de dominio.
    """
    worlds: List[Dict] = []
    for i in range(n_worlds):
        s = seed + i
        if i % 2 == 0:
            cw = CausalWorld(seed=s, n_vars=2 + (i % 2),
                             noise_std=0.05 + 0.03 * (i % 3))
            worlds.append({
                "name": f"causal_{s}",
                "kind": "causal",
                "seed": s,
                "data": cw.get_observational_samples() if hasattr(cw, "get_observational_samples") else [],
                "world": cw,
            })
        else:
            pw = PhysicsWorld(seed=s, n_objects=15 + (i % 3))
            worlds.append({
                "name": f"physics_{s}",
                "kind": "physics",
                "seed": s,
                "data": pw.get_training_data() if hasattr(pw, "get_training_data") else [],
                "world": pw,
            })
    return worlds


# ─── Helpers de medicao (fallbacks honestos) ─────────────────────────

def _to_obs_array(d) -> Optional[np.ndarray]:
    """Converte uma amostra de mundo em array numerico para FEP."""
    if d is None:
        return None
    if isinstance(d, np.ndarray):
        return d.astype(np.float64)
    if isinstance(d, dict):
        vals = [v for v in d.values() if isinstance(v, (int, float, np.floating))]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            # distribui como distribuicao (nao-negativa, normalizada)
            arr = np.abs(arr) + 1e-6
            return arr / arr.sum()
    if isinstance(d, (list, tuple)):
        try:
            arr = np.array(d, dtype=np.float64)
            arr = np.abs(arr) + 1e-6
            return arr / arr.sum()
        except Exception:
            return None
    return None


def _prediction_error(predict_fn, world: Dict) -> float:
    """Erro de predicao medio do sistema sobre os dados do mundo."""
    data = world.get("data") or []
    if not data:
        return 0.5
    errs = []
    for d in data[:50]:
        try:
            pred = predict_fn(d)
            actual = _extract_target(d, world.get("kind"))
            if actual is not None:
                errs.append(float((pred - actual) ** 2))
        except Exception:
            continue
    return min(1.0, _mean(errs)) if errs else 0.5


def _prediction_error_on(predict_fn, samples: List) -> float:
    errs = []
    for d in samples:
        try:
            pred = predict_fn(d)
            actual = _extract_numeric(d)
            if actual is not None:
                errs.append(float((pred - actual) ** 2))
        except Exception:
            continue
    return min(1.0, _mean(errs)) if errs else 0.5


def _extract_target(d, kind: Optional[str]) -> Optional[float]:
    if isinstance(d, dict):
        if kind == "physics" and "fall_time" in d:
            return float(d["fall_time"])
        # causal: o "effect" costuma ser o ultimo valor numerico
        nums = [v for v in d.values() if isinstance(v, (int, float, np.floating))]
        return float(nums[-1]) if nums else None
    return _extract_numeric(d)


def _extract_numeric(d) -> Optional[float]:
    if isinstance(d, (int, float, np.floating)):
        return float(d)
    if isinstance(d, dict):
        nums = [v for v in d.values() if isinstance(v, (int, float, np.floating))]
        return float(nums[-1]) if nums else None
    if isinstance(d, (list, tuple)) and d:
        try:
            return float(d[-1])
        except Exception:
            return None
    return None


def _world_intrinsic_surprise(world: Dict) -> float:
    """Surpresa intrinseca dos dados do mundo (baseline sem sistema)."""
    data = world.get("data") or []
    if not data:
        return 0.5
    nums = []
    for d in data:
        v = _extract_numeric(d)
        if v is not None:
            nums.append(v)
    if len(nums) < 2:
        return 0.5
    arr = np.array(nums)
    # Surpresa ~ variancia normalizada. Mundo ruidoso = mais surpresa.
    return float(min(1.0, np.std(arr) / (abs(np.mean(arr)) + 1e-6)))


def _measure_state_growth(system, worlds: List[Dict]) -> float:
    """Heuristica: se o sistema ganhou estrutura coerente ao operar."""
    if system is None:
        return 0.0
    cge = system.get("cge") if isinstance(system, dict) else getattr(system, "cge", None)
    if cge is not None and hasattr(cge, "nodes"):
        n = len(cge.nodes) if isinstance(cge.nodes, dict) else len(getattr(cge, "nodes", []))
        # Crescimento ate ~100 nos = aprendizado ativo; escala suave
        return min(0.4, n / 250.0)
    return 0.0


def _measure_rule_compression(worlds: List[Dict]) -> float:
    """
    Mede quao comprimivel e cada mundo por uma regra simples (ajuste de
    baixa complexidade). Mundo cujos dados cabem em poucos parametros =
    comprimivel = o sistema PODERIA aprende-lo de forma compacta.
    Nao mede o sistema diretamente — e o teto de compressao disponivel.
    """
    scores = []
    for w in worlds:
        data = w.get("data") or []
        nums = []
        for d in data:
            v = _extract_numeric(d)
            if v is not None:
                nums.append(v)
        if len(nums) < 5:
            continue
        arr = np.array(nums)
        x = np.arange(len(arr), dtype=np.float64)
        # Tenta regressao linear (2 params). Se descreve bem, mundo e comprimivel.
        try:
            A = np.column_stack([np.ones_like(x), x])
            coef, _, _, _ = np.linalg.lstsq(A, arr, rcond=None)
            resid = arr - A @ coef
            r2 = 1.0 - np.var(resid) / (np.var(arr) + 1e-9)
            scores.append(max(0.0, r2))
        except Exception:
            continue
    return _mean(scores) if scores else 0.0
