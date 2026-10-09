"""Avalia candidatas sem promover por separação do par que as gerou.

Treina apenas em 97. 103 já foi usado em outros ensaios: este diagnóstico
é exploratório. A assinatura e a sonda reais são braços oraculares explícitos.
Os braços utilizáveis recebem somente os dois quadros iniciais.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features, load as load_horizons, score
from avaliar_estado_preditivo import encode, load as load_states

HERE = Path(__file__).resolve().parent


class ScaledRidge:
    def fit(self, x, y):
        self.center = x.mean(0)
        deviation = x.std(0)
        positive = deviation[deviation > 1e-10]
        floor = 0.1 * float(np.median(positive)) if len(positive) else 1.0
        self.scale = np.maximum(deviation, max(floor, 1e-10))
        self.model = KernelRidge().fit((x - self.center) / self.scale, y)
        return self

    def predict(self, x):
        return self.model.predict((x - self.center) / self.scale)


def cross_predict(x, y, episode):
    predicted = np.empty_like(y, dtype=np.float64)
    for group in range(8):
        train = episode != group
        test = episode == group
        predicted[test] = KernelRidge().fit(x[train], y[train]).predict(x[test])
    return predicted


def extended_basis(signature, valid):
    root = signature[:, 0, 0].reshape(64, -1)
    later = signature[:, :, 1:].reshape(-1, 1344)
    samples = np.concatenate((root, later[valid[:, :, 1:].reshape(-1)])).astype(np.float32)
    mean = samples.mean(0)
    centered = samples - mean
    rng = np.random.default_rng(9732)
    probe = rng.normal(size=(1344, 64)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ probe)
    for _ in range(2):
        q, _ = np.linalg.qr(centered @ (centered.T @ q))
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    return mean, vt[:40].T


def reward_metrics(actual, predicted, train_reward, episode):
    choice = np.argmax(predicted, axis=1)
    constant = int(np.argmax(train_reward.mean(0)))
    optimum = actual.max(1)
    regret = optimum - actual[np.arange(len(actual)), choice]
    constant_regret = optimum - actual[:, constant]
    contrast = np.ptp(actual, axis=1) > 0
    return {"mean_regret": float(regret.mean()),
            "constant_action": constant,
            "constant_regret": float(constant_regret.mean()),
            "roots_with_reward_contrast": int(contrast.sum()),
            "choices": choice.tolist(),
            "per_episode_regret": [float(regret[episode == e].mean()) for e in range(8)],
            "per_episode_constant_regret": [float(constant_regret[episode == e].mean()) for e in range(8)]}


def evaluate(game, seed):
    train_path, train_pairs, train_sig, train_valid, episode = load_states(game, 97)
    _, test_pairs, test_sig, _, test_episode = load_states(game, seed)
    train_h_path, _, train_future, _, _ = load_horizons(game, 97)
    test_h_path, _, test_future, _, _ = load_horizons(game, seed)
    train_answers_path = HERE / "resultados" / f"{game}_sondas_97.npz"
    test_answers_path = HERE / "resultados" / f"{game}_sondas_{seed}.npz"
    answers = np.load(train_answers_path)
    test_answers = np.load(test_answers_path)
    responses, random_responses = answers["response"], answers["random_response"]
    test_responses = test_answers["response"]
    xtrain = features(train_pairs[:, 0, 0])
    xtest = features(test_pairs[:, 0, 0])
    mean, basis40 = extended_basis(train_sig, train_valid)
    root_signature = train_sig[:, 0, 0].reshape(64, -1)
    signature_cv = cross_predict(xtrain, root_signature, episode).reshape(64, 4, 336)
    signature_pred = KernelRidge().fit(xtrain, root_signature).predict(xtest).reshape(64, 4, 336)
    z_cv = encode(signature_cv, mean, basis40)
    z_pred = encode(signature_pred, mean, basis40)
    q_cv = cross_predict(xtrain, responses, episode)
    q_pred = KernelRidge().fit(xtrain, responses).predict(xtest)
    random_cv = cross_predict(xtrain, random_responses, episode)
    random_pred = KernelRidge().fit(xtrain, random_responses).predict(xtest)
    z_real = encode(train_sig[:, 0, 0], mean, basis40)[:, :32]
    z_test_real = encode(test_sig[:, 0, 0], mean, basis40)[:, :32]
    rng = np.random.default_rng(9703)
    shuffled = responses[rng.permutation(64)]
    shuffled_cv = cross_predict(xtrain, shuffled, episode)
    shuffled_pred = KernelRidge().fit(xtrain, shuffled).predict(xtest)
    arms = {
        "estado32_previsto": (z_cv[:, :32], z_pred[:, :32]),
        "sondas_previstas": (np.column_stack((z_cv[:, :32], q_cv)), np.column_stack((z_pred[:, :32], q_pred))),
        "leituras_aleatorias": (np.column_stack((z_cv[:, :32], random_cv)), np.column_stack((z_pred[:, :32], random_pred))),
        "pca40_previsto": (z_cv, z_pred),
        "sondas_embaralhadas": (np.column_stack((z_cv[:, :32], shuffled_cv)), np.column_stack((z_pred[:, :32], shuffled_pred))),
        "pixels_diretos": (xtrain, xtest),
        "estado32_real_oracular": (z_real, z_test_real),
        "sondas_reais_oracular": (np.column_stack((z_real, responses)), np.column_stack((z_test_real, test_responses))),
    }
    actual = test_future[:, :, -1, 1]
    target = train_future[:, :, -1, 1].reshape(64, -1)
    train_reward = np.load(train_h_path)["rewards"][:, :, -1]
    test_reward = np.load(test_h_path)["rewards"][:, :, -1]
    results = {}
    for name, (fit_x, test_x) in arms.items():
        predicted = ScaledRidge().fit(fit_x, target).predict(test_x).reshape(64, 4, 336)
        predicted_reward = ScaledRidge().fit(fit_x, train_reward).predict(test_x)
        results[name] = {"features": fit_x.shape[1],
                         "visual_error": score(actual, predicted, test_episode),
                         "reward": reward_metrics(test_reward, predicted_reward,
                                                  train_reward, test_episode)}
    constant_q_error = np.mean((test_responses - responses.mean(0)) ** 2, axis=0)
    q_error = np.mean((test_responses - q_pred) ** 2, axis=0)
    reconstruction = np.divide(q_error, constant_q_error,
                               out=np.full_like(q_error, np.nan), where=constant_q_error > 1e-12)
    report = {"status": "exploratory_not_promoted", "game": game,
              "train_seed": 97, "test_seed": seed, "horizon": 128,
              "probe_extraction_error_to_constant": reconstruction.tolist(),
              "arms": results,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_state_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "train_answers_sha256": hashlib.sha256(train_answers_path.read_bytes()).hexdigest(),
              "test_answers_sha256": hashlib.sha256(test_answers_path.read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_avaliacao_sondas_{seed}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(103,), required=True)
    args = parser.parse_args()
    result = evaluate(args.game, args.seed)
    print(json.dumps({"game": args.game,
                      "probe_extraction_error": result["probe_extraction_error_to_constant"],
                      "arms": {name: {"visual_error": value["visual_error"]["mean"],
                                      "reward_regret": value["reward"]["mean_regret"],
                                      "constant_regret": value["reward"]["constant_regret"]}
                               for name, value in result["arms"].items()}},
                     ensure_ascii=False, indent=2))
