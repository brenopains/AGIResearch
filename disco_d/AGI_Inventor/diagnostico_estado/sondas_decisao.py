"""Constrói leituras de pares com preferências de ação incompatíveis.

Apenas episódios 0--3 de 97 geram os pares. Palavras de ações são construídas
por composição. Nenhuma sonda é aceita aqui: seleção e decisão ficam em outro
processo. Os programas e as projeções sensoriais são armazenados integralmente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar_estado_preditivo import encode, load as load_states
from avaliar_sondas import ScaledRidge
from sondas_contraexemplo import AleRunner, Budget, words, MAX_STEPS, QUANTUM

HERE = Path(__file__).resolve().parent
DEPTH = 2
MAX_PAIRS = 4


def choose_pairs(game):
    _, _, signature, _, episode = load_states(game, 97)
    base = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature[:, 0, 0], base["mean"], base["basis"])
    horizon = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    reward = horizon["rewards"][:, :, -1]
    optimum = reward == reward.max(1, keepdims=True)
    span = np.ptp(reward, axis=1, keepdims=True)
    preferences = np.divide(reward - reward.mean(1, keepdims=True), span,
                            out=np.zeros_like(reward), where=span > 0)
    predicted = np.zeros((32, 4))
    for e in range(4):
        train = np.flatnonzero((episode < 4) & (episode != e))
        test = np.flatnonzero(episode == e)
        predicted[test] = ScaledRidge().fit(z[train], preferences[train]).predict(z[test])
    mistaken = ~optimum[np.arange(32), np.argmax(predicted, axis=1)]
    distance = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    eligible = []
    for i in range(32):
        for j in range(i + 1, 32):
            if (episode[i] == episode[j] or np.any(optimum[i] & optimum[j])
                    or not (mistaken[i] or mistaken[j])):
                continue
            eligible.append((float(distance[i, j]), i, j))
    eligible.sort()
    result, used = [], set()
    for gap, first, second in eligible:
        if first in used or second in used:
            continue
        used.update((first, second))
        result.append({"first": first, "second": second, "latent_distance": gap,
                       "disjoint_optimal_action_sets": True,
                       "cross_episode_prediction_wrong": [bool(mistaken[first]), bool(mistaken[second])],
                       "scope": "near_states_not_exact_identity"})
        if len(result) == MAX_PAIRS:
            break
    return result


def collect(game, seed):
    path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    data = np.load(path)
    expected = data["histories"]
    budget = Budget(MAX_STEPS)
    runner = AleRunner(game, seed, expected, budget)
    definition_path = HERE / "resultados" / f"{game}_sondas_decisao_definicoes.json"
    if seed == 97:
        pairs = choose_pairs(game)
        programs = list(words(4, DEPTH))
    else:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        pairs = definition["pairs"]
        programs = [tuple(program) for program in definition["programs"]]
    fields = []
    for program in programs:
        fields.append(np.stack([runner.observe(root, program) for root in range(64)]))
        print(json.dumps({"game": game, "seed": seed, "program": list(program),
                          "steps": budget.used}), flush=True)
    fields = np.stack(fields)
    if seed == 97:
        probes = []
        rng = np.random.default_rng(9704)
        for p, program in enumerate(programs):
            for pair in pairs:
                delta = fields[p, pair["first"]] - fields[p, pair["second"]]
                length = float(np.linalg.norm(delta))
                if length <= 1e-12:
                    continue
                random_vector = rng.normal(size=336)
                random_vector /= np.linalg.norm(random_vector)
                probes.append({"id": f"decision_probe_{len(probes)}", "program_index": p,
                               "actions": list(program), "quantum": QUANTUM,
                               "source_pair": [pair["first"], pair["second"]],
                               "readout": (delta / length).tolist(),
                               "random_readout": random_vector.tolist(),
                               "witness_separation": length, "status": "candidate"})
        for pair in pairs:
            for root in (pair["first"], pair["second"]):
                runner.certify(root, programs[-1])
        definition = {"status": "generated_not_selected", "game": game, "seed": 97,
                      "pairs": pairs, "programs": [list(p) for p in programs],
                      "probes": probes,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    else:
        probes = definition["probes"]
    response = np.zeros((64, len(probes)))
    random_response = np.zeros_like(response)
    for i, probe in enumerate(probes):
        f = fields[probe["program_index"]]
        response[:, i] = f @ np.asarray(probe["readout"])
        random_response[:, i] = f @ np.asarray(probe["random_readout"])
    output = HERE / "resultados" / f"{game}_sondas_decisao_{seed}.npz"
    np.savez_compressed(output, response=response, random_response=random_response,
                        episode=data["episode"], effects=fields)
    report = {"status": "no_pairs" if not pairs else "unvalidated_candidates",
              "game": game, "seed": seed, "programs": len(programs),
              "pairs": len(pairs), "probes": len(probes), "steps_used": budget.used,
              "roots_certified": len(runner.states), "replays_certified": runner.replay_count,
              "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(97, 103), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
