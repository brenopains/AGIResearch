"""Extrator de respostas a programas espaciais construídos dos resíduos."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import load as load_states
from extrator_experiencias import ExperienceExtractor, stratified
from leituras_variaveis import execute, make_readout

HERE = Path(__file__).resolve().parent


class VariableExtractor:
    def __init__(self, game):
        self.source = ExperienceExtractor(game)
        self.probes = []
        for index, parent in enumerate(self.source.probes):
            self.probes.append({"id": f"variable_probe_{index}", "parent_id": parent["id"],
                                "actions": parent["actions"], "quantum": parent["quantum"],
                                "program_index": parent["program_index"], "source_pair": parent["source_pair"],
                                "readout_program": make_readout(np.asarray(parent["readout"]).reshape(21, 16), 9740 + index)})
        self.n = len(self.probes) * 4
        self.episode, self.root_x = self.source.episode, self.source.root_x
        self.single_ids = self.source.single_ids
        self.single_regular = self.read(self.source.single_s, self.single_ids)
        self.single_random = self.read(self.source.single_s, self.single_ids, random=True)
        self.double = {}
        for action, data in self.source.double.items():
            self.double[action] = {**data,
                                   "regular": self.read(data["effects"], data["ids"], double=True),
                                   "random": self.read(data["effects"], data["ids"], double=True, random=True)}
        _, _, signature, _, _ = load_states(game, 97)
        actual = np.load(HERE / "resultados" / f"{game}_sondas_decisao_97.npz")["effects"]
        errors = []
        for probe in self.probes:
            program = probe["actions"]
            if len(program) == 1:
                derived = signature[:, 0, 0, program[0]]
            else:
                first, second = program
                derived = signature[:, first, 0, first] + signature[:, first, 1, second]
            observed = actual[probe["program_index"]]
            for random in (False, True):
                difference = np.abs(execute(probe["readout_program"], derived, random=random)
                                    - execute(probe["readout_program"], observed, random=random))
                errors.append(float(difference.max()))
        self.certified_response_max_error = max(errors, default=0.0)
        if self.certified_response_max_error > 1e-6:
            raise RuntimeError(f"leitura não linear divergiu entre composição e execução: {self.certified_response_max_error}")

    def read(self, effects, ids, double=False, random=False):
        result = np.empty((len(effects), len(ids) * 4))
        for column, index in enumerate(ids):
            probe = self.probes[index]
            action = probe["actions"][1] if double else probe["actions"][0]
            result[:, column * 4:(column + 1) * 4] = execute(probe["readout_program"], effects[:, action], random=random)
        return result

    def columns(self, ids):
        return np.asarray([4 * index + channel for index in ids for channel in range(4)])

    def predict(self, query, allowed):
        result, random = np.zeros((len(query), self.n)), np.zeros((len(query), self.n))
        selection = stratified(self.source.single_e, allowed, 512)
        targets = np.column_stack((self.single_regular, self.single_random))
        prediction = KernelRidge().fit(self.source.single_x[selection], targets[selection]).predict(query)
        count = len(self.single_ids) * 4
        result[:, self.columns(self.single_ids)] = prediction[:, :count]
        random[:, self.columns(self.single_ids)] = prediction[:, count:]
        counts = {"single": int(len(selection))}
        for action, data in self.double.items():
            ids = data["ids"]
            if not ids:
                continue
            selection = stratified(data["episode"], allowed, 256)
            targets = np.column_stack((data["regular"], data["random"]))
            prediction = KernelRidge().fit(data["x"][selection], targets[selection]).predict(query)
            count = len(ids) * 4
            result[:, self.columns(ids)] = prediction[:, :count]
            random[:, self.columns(ids)] = prediction[:, count:]
            counts[str(action)] = int(len(selection))
        return result, random, counts

    def real_responses(self, game, seed, random=False):
        fields = np.load(HERE / "resultados" / f"{game}_sondas_decisao_{seed}.npz")["effects"]
        result = np.zeros((64, self.n))
        for index, probe in enumerate(self.probes):
            result[:, index * 4:(index + 1) * 4] = execute(
                probe["readout_program"], fields[probe["program_index"]], random=random)
        return result


def collect(game):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    extractor = VariableExtractor(game)
    _, test_pairs, _, _, test_groups = load_states(game, 103)
    test_x = features(test_pairs[:, 0, 0])
    cv, random_cv = np.zeros((64, extractor.n)), np.zeros((64, extractor.n))
    for episode in range(8):
        query = extractor.episode == episode
        cv[query], random_cv[query], _ = extractor.predict(extractor.root_x[query], set(range(8)) - {episode})
        print(json.dumps({"game": game, "fold": episode, "cpu_seconds": time.process_time() - started_cpu}), flush=True)
    predicted, random_predicted, counts = extractor.predict(test_x, set(range(8)))
    real_train, real_test = extractor.real_responses(game, 97), extractor.real_responses(game, 103)
    denominator = np.mean((real_test - real_train.mean(0)) ** 2, axis=0)
    error = np.mean((real_test - predicted) ** 2, axis=0)
    ratio = np.divide(error, denominator, out=np.full_like(error, np.nan), where=denominator > 1e-12)
    stem = HERE / "resultados" / f"{game}_leituras_variaveis"
    definitions = stem.with_name(stem.name + "_definicoes.json")
    definitions.write_text(json.dumps({"status": "candidate_readouts_not_promoted", "source_seed": 97,
                                       "parent_definition_sha256": hashlib.sha256(extractor.source.definition_path.read_bytes()).hexdigest(),
                                       "probes": extractor.probes}, ensure_ascii=False, indent=2), encoding="utf-8")
    output = stem.with_suffix(".npz")
    np.savez_compressed(output, cv=cv, predicted=predicted, random_cv=random_cv,
                        random_predicted=random_predicted, real_train=real_train, real_test=real_test,
                        random_real_train=extractor.real_responses(game, 97, random=True),
                        random_real_test=extractor.real_responses(game, 103, random=True),
                        train_episode=extractor.episode, test_episode=test_groups)
    report = {"status": "development_only", "game": game, "train_seed": 97, "development_seed": 103,
              "probes": len(extractor.probes), "columns": extractor.n,
              "training_counts": counts, "probe_error_to_constant": ratio.tolist(),
              "certified_response_max_absolute_error": extractor.certified_response_max_error,
              "cpu_seconds": time.process_time() - started_cpu, "wall_seconds": time.perf_counter() - started_wall,
              "new_ale_steps": 0, "bank_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "definition_sha256": hashlib.sha256(definitions.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "executor_sha256": hashlib.sha256((HERE / "leituras_variaveis.py").read_bytes()).hexdigest()}
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = collect(args.game)
    print(json.dumps({"game": args.game, "probes": result["probes"],
                      "certification_error": result["certified_response_max_absolute_error"],
                      "mean_extraction_error": float(np.nanmean(result["probe_error_to_constant"])),
                      "cpu_seconds": result["cpu_seconds"], "wall_seconds": result["wall_seconds"]},
                     ensure_ascii=False, indent=2))
