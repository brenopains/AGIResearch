"""Testa atualizações de estado preditivo em sequências de ações inéditas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import decode, encode, fit_updates, load, score
from coleta import small

HERE = Path(__file__).resolve().parent


def evaluate(game, test_seed):
    train_path, train_pairs, train_sig, train_valid, _ = load(game, 97)
    horizon_path = HERE / "resultados" / f"{game}_horizontes_{test_seed}.npz"
    horizons = np.load(horizon_path)
    test_pairs = np.stack([small(frame, 21) for frame in
                           horizons["histories"][:, -2:].reshape(-1, *horizons["histories"].shape[-2:])])
    test_pairs = test_pairs.reshape(64, 2, 336)
    basis_path = HERE / "resultados" / f"{game}_base_estado_preditivo.npz"
    basis_data = np.load(basis_path)
    mean, basis = basis_data["mean"], basis_data["basis"]
    train_z = encode(train_sig, mean, basis)
    models, examples = fit_updates(train_z, train_valid)
    source = HERE / "resultados" / f"{game}_planos_mistos_{test_seed}.npz"
    mixed = np.load(source)
    actual = mixed["signatures"].reshape(64, 3, 9, 4, 336)
    valid = mixed["valid"]
    plans = mixed["plans"]
    episodes = mixed["episode"]
    if test_seed == 103:
        _, _, test_sig, _, _ = load(game, 103)
        if not np.array_equal(actual[:, 0, 0], test_sig[:, 0, 0]):
            raise RuntimeError("assinaturas das raízes divergem")
    xtrain = features(train_pairs[:, 0, 0])
    xtest = features(test_pairs)
    encoder = KernelRidge().fit(xtrain, train_sig[:, 0, 0].reshape(64, -1))
    predicted_root = encoder.predict(xtest).reshape(64, 4, 336)
    states = {
        "estado_previsto": np.repeat(encode(predicted_root, mean, basis)[:, None], 3, axis=1),
        "inicio_real": np.repeat(encode(actual[:, 0, 0], mean, basis)[:, None], 3, axis=1),
        "acoes_permutadas": np.repeat(encode(predicted_root, mean, basis)[:, None], 3, axis=1),
    }
    persistence = decode(states["estado_previsto"], mean, basis)
    curves = {}
    for stage in range(1, 9):
        for name in states:
            next_states = np.empty_like(states[name])
            for path in range(3):
                action = int(plans[path, stage - 1])
                model_index = (action + 1) % 4 if name == "acoes_permutadas" else action
                old = states[name][:, path]
                next_states[:, path] = old + models[model_index].predict(old)
            states[name] = next_states
        truth = actual[:, :, stage]
        mask = valid[:, :, stage]
        row = {}
        predictions = {name: decode(value, mean, basis)
                       for name, value in states.items()}
        predictions["assinatura_persistente"] = persistence
        for name, prediction in predictions.items():
            entry = score(truth, prediction, mask, episodes)
            entry["per_plan"] = [score(truth[:, path:path + 1],
                                       prediction[:, path:path + 1],
                                       mask[:, path:path + 1], episodes)["mean"]
                                 for path in range(3)]
            row[name] = entry
        curves[str(stage * 8)] = row
    report = {
        "status": "exploratorio_semente_103_ja_inspecionada",
        "game": game, "train_seed": 97, "test_seed": test_seed,
        "training_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_horizon_bank_sha256": hashlib.sha256(horizon_path.read_bytes()).hexdigest(),
        "mixed_bank_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "basis_sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_examples_by_action": examples,
        "plans": plans.tolist(),
        "valid_by_stage": [int(valid[:, :, s].sum()) for s in range(9)],
        "curves": curves,
    }
    output = HERE / "resultados" / f"{game}_avaliacao_planos_mistos_{test_seed}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(103, 109), required=True)
    args = parser.parse_args()
    result = evaluate(args.game, args.seed)
    print(json.dumps({"game": args.game, "valid_by_stage": result["valid_by_stage"],
                      "curves": {stage: {name: value["mean"] for name, value in row.items()}
                                 for stage, row in result["curves"].items()}},
                     ensure_ascii=False, indent=2))
