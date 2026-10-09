"""Sondas executáveis construídas de divergências observadas, sem objetos.

Uma sonda contém um programa de ações anônimas e um vetor de leitura extraído
do resíduo sensorial de um par. Separação gera candidata, nunca promoção.
Os alvos distantes selecionam testemunhas apenas no desenvolvimento (97).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np

from coleta import gray, make_ale, small
from avaliar_composicao import load as load_horizons
from avaliar_estado_preditivo import encode, load as load_states

HERE = Path(__file__).resolve().parent
QUANTUM = 8
MAX_DEPTH = 2
WITNESSES = 4
MAX_STEPS = 50000


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, limit):
        self.limit, self.used = int(limit), 0

    def take(self):
        if self.used >= self.limit:
            raise BudgetExhausted("orçamento de intervenções esgotado")
        self.used += 1


def words(alphabet, depth):
    for length in range(1, depth + 1):
        yield from itertools.product(range(alphabet), repeat=length)


def discover_pair(observe, first, second, alphabet, depth):
    """Todas as palavras curtas; retém a maior separação por comprimento.

    observe devolve o efeito sensorial do mesmo programa desde uma raiz.
    O vetor de leitura é o resíduo do par; não vem de um cadastro de atributos.
    Separação zero não gera sonda. Os corpos completos ficam no resultado.
    """
    best = {}
    events = []
    for program in words(alphabet, depth):
        left = np.asarray(observe(first, program), dtype=np.float64).reshape(-1)
        right = np.asarray(observe(second, program), dtype=np.float64).reshape(-1)
        delta = left - right
        separation = float(np.linalg.norm(delta))
        events.append({"type": "program_generated", "pair": [first, second],
                       "actions": list(program), "separation": separation})
        if separation <= 1e-12:
            continue
        length = len(program)
        if length not in best or separation > best[length]["separation"]:
            best[length] = {"actions": list(program),
                            "readout": (delta / separation).tolist(),
                            "source_pair": [int(first), int(second)],
                            "separation": separation,
                            "status": "candidate_unvalidated"}
    return [best[length] for length in sorted(best)], events


class AleRunner:
    def __init__(self, game, seed, expected, budget):
        self.ale = make_ale(game, seed)
        self.actions = list(self.ale.getMinimalActionSet()[:4])
        self.budget = budget
        self.states, self.initial = [], []
        self.cache = {}
        self.replay_count = 0
        rng = np.random.default_rng(seed)
        for episode in range(8):
            self.ale.reset_game()
            frame = gray(self.ale)
            history = deque([frame.copy() for _ in range(16)], maxlen=16)
            for checkpoint in range(8):
                for _ in range(128):
                    budget.take()
                    self.ale.act(self.actions[int(rng.integers(0, 4))])
                    if self.ale.game_over():
                        self.ale.reset_game()
                        frame = gray(self.ale)
                        history = deque([frame.copy() for _ in range(16)], maxlen=16)
                    else:
                        history.append(gray(self.ale))
                index = episode * 8 + checkpoint
                current = np.stack(history)
                if not np.array_equal(current, expected[index]):
                    raise RuntimeError(f"histórico da raiz {index} divergiu")
                self.states.append(self.ale.cloneSystemState())
                self.initial.append(small(current[-1], 21).reshape(-1))

    def execute(self, root, program):
        self.ale.restoreSystemState(self.states[root])
        elapsed = 0
        for command in program:
            for _ in range(QUANTUM):
                self.budget.take()
                self.ale.act(self.actions[int(command)])
                elapsed += 1
                if self.ale.game_over():
                    return (small(gray(self.ale), 21).reshape(-1) - self.initial[root],
                            elapsed, True)
        return (small(gray(self.ale), 21).reshape(-1) - self.initial[root],
                elapsed, False)

    def observe(self, root, program):
        key = (int(root), tuple(program))
        if key not in self.cache:
            self.cache[key] = self.execute(root, program)
        return self.cache[key][0]

    def certify(self, root, program):
        prior = self.observe(root, program)
        again, elapsed, terminal = self.execute(root, program)
        expected = self.cache[(int(root), tuple(program))]
        if not (np.array_equal(prior, again) and (elapsed, terminal) == expected[1:]):
            raise RuntimeError("replay da sonda divergiu")
        self.replay_count += 1


def witness_pairs(game):
    _, _, future, episode, terminal = load_horizons(game, 97)
    _, _, signature, valid, _ = load_states(game, 97)
    base = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature[:, 0, 0], base["mean"], base["basis"])
    distant = future[:, :, -1, 1]
    contrasts = np.stack([distant[:, a] - distant[:, b]
                          for a, b in itertools.combinations(range(4), 2)], axis=1)
    contrasts = contrasts.reshape(64, -1)
    distance = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    target_distance = np.linalg.norm(contrasts[:, None] - contrasts[None, :], axis=-1)
    proposals = []
    for index in range(64):
        eligible = np.flatnonzero(episode != episode[index])
        other = int(eligible[np.argmin(distance[index, eligible])])
        if target_distance[index, other] <= 1e-12:
            continue
        score = target_distance[index, other] / max(distance[index, other], 1e-6)
        proposals.append((float(score), min(index, other), max(index, other)))
    proposals.sort(reverse=True)
    selected, seen = [], set()
    for ratio, first, second in proposals:
        if (first, second) in seen:
            continue
        seen.add((first, second))
        selected.append({"first": first, "second": second,
                         "state_distance": float(distance[first, second]),
                         "future_distance": float(target_distance[first, second]),
                         "divergence_over_distance": ratio,
                         "scope": "near_states_not_exact_identity"})
        if len(selected) == WITNESSES:
            break
    return selected


def collect(game, seed):
    horizon_path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    horizon = np.load(horizon_path)
    expected = horizon["histories"]
    budget = Budget(MAX_STEPS)
    runner = AleRunner(game, seed, expected, budget)
    definition_path = HERE / "resultados" / f"{game}_sondas_definicoes.json"
    events = []
    if seed == 97:
        witnesses = witness_pairs(game)
        probes = []
        for witness in witnesses:
            found, generated = discover_pair(runner.observe, witness["first"],
                                              witness["second"], 4, MAX_DEPTH)
            events.extend(generated)
            for candidate in found:
                runner.certify(witness["first"], candidate["actions"])
                runner.certify(witness["second"], candidate["actions"])
                candidate["id"] = f"probe_{len(probes)}"
                candidate["quantum"] = QUANTUM
                events.append({"type": "probe_candidate", **candidate})
                probes.append(candidate)
        rng = np.random.default_rng(9702)
        for probe in probes:
            vector = rng.normal(size=336)
            vector /= np.linalg.norm(vector)
            probe["random_readout"] = vector.tolist()
        definition = {"status": "unvalidated_development_candidates", "game": game,
                      "source_seed": 97, "witnesses": witnesses, "probes": probes,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    else:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        probes = definition["probes"]
        if definition["source_seed"] != 97:
            raise RuntimeError("definições não são do desenvolvimento")
    response, random_response = [], []
    for index, probe in enumerate(probes):
        all_effects = np.stack([runner.observe(root, probe["actions"]) for root in range(64)])
        response.append(all_effects @ np.asarray(probe["readout"]))
        random_response.append(all_effects @ np.asarray(probe["random_readout"]))
        events.append({"type": "probe_observed", "probe": probe["id"],
                       "roots": 64, "steps_used": budget.used})
        print(json.dumps({"game": game, "seed": seed, "probe": index,
                          "actions": probe["actions"], "steps_used": budget.used}), flush=True)
    if not probes:
        raise RuntimeError("nenhuma sonda gerada; diagnóstico: sem_separacao_no_orcamento")
    output = HERE / "resultados" / f"{game}_sondas_{seed}.npz"
    np.savez_compressed(output, response=np.asarray(response).T,
                        random_response=np.asarray(random_response).T,
                        episode=horizon["episode"])
    event_path = output.with_suffix(".jsonl")
    with event_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = {"status": "candidate_generation_not_promotion", "game": game, "seed": seed,
              "probes": len(probes), "max_depth": MAX_DEPTH, "quantum": QUANTUM,
              "roots_certified": len(runner.states), "replays_certified": runner.replay_count,
              "steps_used": budget.used, "budget": budget.limit,
              "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "horizon_sha256": hashlib.sha256(horizon_path.read_bytes()).hexdigest(),
              "responses_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(97, 103, 109), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
