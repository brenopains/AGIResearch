"""Compõe assinaturas de consequências, em vez de imagens previstas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_efeitos_locais import contrast_loss

HERE = Path(__file__).resolve().parent
RANK = 32
STAGES = 9


def load(game, seed):
    path = HERE / "resultados" / f"{game}_estado_preditivo_{seed}.npz"
    data = np.load(path)
    pairs = data["pairs"].reshape(64, 4, STAGES, 2, 336).copy()
    signature = data["signatures"].reshape(64, 4, STAGES, 4, 336).copy()
    valid = data["valid"].copy()
    episode = data["episode"].copy()
    if not (np.array_equal(pairs[:, 0, 0], pairs[:, 1, 0])
            and np.array_equal(signature[:, 0, 0], signature[:, 1, 0])):
        raise RuntimeError("raiz não é comum às quatro trajetórias")
    return path, pairs, signature, valid, episode


def learn_basis(train_signature, train_valid):
    root = train_signature[:, 0, 0].reshape(64, -1)
    later = train_signature[:, :, 1:].reshape(-1, 1344)
    mask = train_valid[:, :, 1:].reshape(-1)
    samples = np.concatenate((root, later[mask])).astype(np.float32)
    mean = samples.mean(0)
    centered = samples - mean
    rng = np.random.default_rng(9732)
    probe = rng.normal(size=(1344, 64)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ probe)
    for _ in range(2):
        q, _ = np.linalg.qr(centered @ (centered.T @ q))
    _, singular, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    basis = vt[:RANK].T
    return mean, basis, singular[:RANK], len(samples)


def encode(signature, mean, basis):
    return (signature.reshape(*signature.shape[:-2], 1344) - mean) @ basis


def decode(z, mean, basis):
    return (z @ basis.T + mean).reshape(*z.shape[:-1], 4, 336)


def score(actual, predicted, valid, episode):
    episode_ratios, counts = [], []
    for group in range(8):
        selected = valid & (episode[:, None] == group)
        real = actual[selected]
        forecast = predicted[selected]
        counts.append(len(real))
        if not len(real):
            episode_ratios.append(None)
            continue
        error, baseline = contrast_loss(real, forecast)
        denominator = float(baseline.sum())
        episode_ratios.append(float(error.sum() / denominator)
                              if denominator > 1e-12 else None)
    available = [value for value in episode_ratios if value is not None]
    return {"per_episode": episode_ratios,
            "mean": float(np.mean(available)) if available else None,
            "states_per_episode": counts}


def fit_updates(z, valid):
    models, examples = [], []
    for action in range(4):
        source = z[:, action, :-1].reshape(-1, RANK)
        next_state = z[:, action, 1:].reshape(-1, RANK)
        usable = (valid[:, action, :-1] & valid[:, action, 1:]).reshape(-1)
        model = KernelRidge().fit(source[usable], next_state[usable] - source[usable])
        models.append(model)
        examples.append(int(usable.sum()))
    return models, examples


def update(models, states):
    out = np.empty_like(states)
    for action in range(4):
        out[:, action] = states[:, action] + models[action].predict(states[:, action])
    return out


def evaluate(game):
    train_path, train_pairs, train_sig, train_valid, _ = load(game, 97)
    test_path, test_pairs, test_sig, test_valid, episode = load(game, 103)
    mean, basis, singular, basis_count = learn_basis(train_sig, train_valid)
    ztrain = encode(train_sig, mean, basis)
    ztest_true = encode(test_sig, mean, basis)
    models, transition_counts = fit_updates(ztrain, train_valid)
    xtrain = features(train_pairs[:, 0, 0])
    xtest = features(test_pairs[:, 0, 0])
    local = KernelRidge().fit(xtrain, train_sig[:, 0, 0].reshape(64, -1))
    predicted_root_signature = local.predict(xtest).reshape(64, 4, 336)
    initial_pred = encode(predicted_root_signature, mean, basis)
    initial_oracle = ztest_true[:, 0, 0]
    states = {"previsto": np.repeat(initial_pred[:, None], 4, axis=1),
              "inicio_oracular": np.repeat(initial_oracle[:, None], 4, axis=1)}
    persistence = np.repeat(initial_pred[:, None], 4, axis=1)
    curves = {}
    for stage in range(1, STAGES):
        for name, state in states.items():
            states[name] = update(models, state)
        actual = test_sig[:, :, stage]
        valid = test_valid[:, :, stage]
        direct = np.empty((64, 4, 4, 336), dtype=np.float32)
        for action in range(4):
            usable = train_valid[:, action, stage]
            target = train_sig[usable, action, stage].reshape(int(usable.sum()), -1)
            direct_model = KernelRidge().fit(xtrain[usable], target)
            direct[:, action] = direct_model.predict(xtest).reshape(64, 4, 336)
        curves[str(stage * 8)] = {
            "estado_previsto_composto": score(actual, decode(states["previsto"], mean, basis), valid, episode),
            "inicio_oracular_composto": score(actual, decode(states["inicio_oracular"], mean, basis), valid, episode),
            "assinatura_persistente": score(actual, decode(persistence, mean, basis), valid, episode),
            "pixels_diretos": score(actual, direct, valid, episode),
        }
    reconstructed_root = decode(ztest_true[:, 0, 0], mean, basis)
    root_valid = np.ones((64, 1), bool)
    root_episode = episode
    root_reconstruction = score(test_sig[:, :1, 0], reconstructed_root[:, None],
                                root_valid, root_episode)
    root_prediction = score(test_sig[:, :1, 0], predicted_root_signature[:, None],
                            root_valid, root_episode)
    teacher = []
    for stage in range(1, STAGES):
        prior = ztest_true[:, :, stage - 1]
        next_prediction = decode(update(models, prior), mean, basis)
        teacher.append(score(test_sig[:, :, stage], next_prediction,
                             test_valid[:, :, stage], episode)["mean"])
    basis_path = HERE / "resultados" / f"{game}_base_estado_preditivo.npz"
    np.savez_compressed(basis_path, mean=mean, basis=basis, singular=singular)
    report = {"status": "exploratorio_semente_103_ja_inspecionada",
              "game": game, "train_seed": 97, "test_seed": 103,
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "basis_sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
              "basis_rank": RANK, "basis_training_states": basis_count,
              "transition_training_examples_by_action": transition_counts,
              "root_reconstruction": root_reconstruction,
              "root_prediction": root_prediction,
              "teacher_forced_update_error": teacher,
              "curves": curves}
    output = HERE / "resultados" / f"{game}_avaliacao_estado_preditivo.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = evaluate(args.game)
    print(json.dumps({"game": args.game,
                      "basis_states": result["basis_training_states"],
                      "training_pairs": result["transition_training_examples_by_action"],
                      "root_reconstruction": result["root_reconstruction"]["mean"],
                      "root_prediction": result["root_prediction"]["mean"],
                      "teacher_forced": result["teacher_forced_update_error"],
                      "curves": {h: {name: value["mean"] for name, value in row.items()}
                                 for h, row in result["curves"].items()}},
                     ensure_ascii=False, indent=2))
