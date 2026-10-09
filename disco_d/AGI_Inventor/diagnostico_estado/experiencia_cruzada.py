"""Contratos da bancada cruzada; nenhuma regra/variável de jogo no aprendiz."""

import json

import numpy as np

from avaliar_leituras_por_programa import ARMS, fitted_model, predict, validate_dependencies
from piloto_planos_propostos import HERE, sha

TRAIN_SEED, TEST_SEED = 131, 137
PREFIXES = (1, 2, 4, 6, 8, 10, 12, 16)
CHECKPOINTS = (0, 2, 4, 6)
PAIRS = ((0, 1), (2, 3))
PARTITIONS = {"known_programs_new_contexts": PAIRS, "new_programs_new_contexts": ((4, 5), (6, 7))}
PROPOSAL = HERE / "resultados" / "experiencia_cruzada_propostas.json"
SOURCES = ("experiencia_cruzada.py", "coletar_experiencia_cruzada.py", "avaliar_experiencia_cruzada.py",
           "piloto_experiencia_cruzada.py", "auditar_experiencia_cruzada.py", "testes_experiencia_cruzada.py",
           "EXPERIENCIA_CRUZADA.md", "leituras_por_programa.py", "contrastes_programas.py", "avaliar.py",
           "avaliar_leituras_por_programa.py", "coleta.py", "coletar_transicoes_mistas.py")


def base(game, seed):
    return HERE / "resultados" / f"{game}_experiencia_cruzada_{seed}"


def manifest_path(game):
    return base(game, TRAIN_SEED).with_name(base(game, TRAIN_SEED).name + "_manifesto.json")


def result_path(game):
    return base(game, TEST_SEED).with_name(base(game, TEST_SEED).name + "_avaliacao.json")


def save_json(path, obj):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare():
    if PROPOSAL.exists():
        raise RuntimeError("proposta existe; preservar")
    for game in ("breakout", "pong"):
        validate_dependencies(game)
        for seed in (TRAIN_SEED, TEST_SEED):
            if any(base(game, seed).with_suffix(s).exists() for s in (".npz", ".json", ".jsonl")):
                raise RuntimeError("sementes já têm banco; preservar")
    seen, groups = set(), []
    for seed in (9831, 9833):
        rng, rows = np.random.default_rng(seed), []
        while len(rows) < 4:
            word = tuple(int(a) for a in rng.integers(0, 4, 16))
            if word not in seen:
                seen.add(word)
                rows.append(word)
        groups.extend(rows)
    report = {"status": "fixed_before_any_new_outcome", "programs": [list(p) for p in groups],
              "world_seeds": [TRAIN_SEED, TEST_SEED], "program_seeds": [9831, 9833],
              "source_hashes": {p: sha(HERE / p) for p in SOURCES}, "maximum_ale_calls_per_bank": 50000,
              "maximum_wall_seconds": 1800, "maximum_aggregate_rss_bytes": 6 * 1024 ** 3,
              "maximum_processes_including_monitor": 2, "primary": "quadratic", "human_level_goal_achieved": False}
    save_json(PROPOSAL, report)
    return report


def verify_proposal():
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    for p, expected in proposal["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"fonte/protocolo mudou: {p}")
    return proposal


def training_data(game):
    with np.load(base(game, TRAIN_SEED).with_suffix(".npz")) as archive:
        histories, programs, observed = archive["histories"], archive["programs"], archive["rewards"].cumsum(-1)
    h, left, right, yl, yr, roots = [], [], [], [], [], []
    for i in range(32):
        for p, q in PAIRS:
            for length in PREFIXES:
                h.append(histories[i])
                left.append(programs[p, :length].tolist())
                right.append(programs[q, :length].tolist())
                yl.append(observed[i, p, length - 1])
                yr.append(observed[i, q, length - 1])
                roots.append(i)
    return np.asarray(h), left, right, np.asarray(yl), np.asarray(yr), np.asarray(roots)


def paired_gain(primary, other, episodes):
    values = np.asarray([(other - primary)[episodes == e].mean() for e in range(8)])
    boot = values[np.random.default_rng(9837).integers(0, 8, (10000, 8))].mean(1)
    return {"mean": float(values.mean()), "bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "positive_episodes": int(np.sum(values > 1e-12)), "per_episode": values.tolist()}


def context_identification(real, episodes):
    # real[32 roots,2 repeated pairs,2 sides]. Never select roots with this.
    delta = real[:, :, 0] - real[:, :, 1]
    reversed_pairs = [bool(np.any(delta[:, p] < -1e-12) and np.any(delta[:, p] > 1e-12)) for p in range(2)]
    active = [int(e) for e in range(8) if np.any(np.abs(delta[episodes == e]) > 1e-12)]
    return {"opposite_preferences_per_repeated_pair": reversed_pairs, "episodes_with_real_contrast": active,
            "non_tied_comparisons": int(np.sum(np.abs(delta) > 1e-12)),
            "sufficient_contextual_contrast": bool(all(reversed_pairs) and len(active) >= 3)}


def gate(rows, real, episodes):
    identification = context_identification(real, episodes)
    primary = np.asarray(rows["quadratic"]["per_root_regret"])
    gains = {arm: paired_gain(primary, np.asarray(rows[arm]["per_root_regret"]), episodes)
             for arm in ARMS if arm != "quadratic"}
    checks = {"sufficient_contextual_contrast": identification["sufficient_contextual_contrast"]}
    for arm, gain in gains.items():
        checks[f"mean_gain_vs_{arm}"] = bool(gain["mean"] > 1e-12)
        checks[f"positive_lower_vs_{arm}"] = bool(gain["bootstrap_95"][0] > 1e-12)
        checks[f"two_episodes_vs_{arm}"] = bool(gain["positive_episodes"] >= 2)
    passed = bool(all(checks.values()))
    status = "positive_reserved_contextual_test" if passed else "failed_capability_gate"
    if not identification["sufficient_contextual_contrast"]:
        status = "inconclusive_insufficient_contextual_contrast"
    return {"status": status, "positive": passed, "identification": identification, "comparisons": gains, "checks": checks}
