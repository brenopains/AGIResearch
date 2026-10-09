"""Liga dinâmica preditiva local a recompensa e sobrevivência locais.

Somente 97 treina. 103 e 113 já foram inspecionados: resultados exploratórios.
Não há regra do jogo, objeto ou tabela de valor; crédito vem de incrementos
de retornos realmente medidos em blocos de oito passos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features, reduce_frames
from avaliar_estado_preditivo import encode, fit_updates, load as load_states
from avaliar_sondas import ScaledRidge
from extrator_experiencias import ExperienceExtractor
from selecionar_sondas_decisao import measure, preferences

HERE = Path(__file__).resolve().parent


def increments(cumulative):
    return np.diff(cumulative, axis=-1, prepend=np.zeros_like(cumulative[..., :1]))


def expected_return(reward, continuation):
    alive = np.ones(reward.shape[:-1])
    total = np.zeros_like(alive)
    for stage in range(reward.shape[-1]):
        total += alive * reward[..., stage]
        alive *= continuation[..., stage]
    return total


def fit_credit(z, valid, cumulative, shuffled=False):
    delta = increments(cumulative)
    if not np.array_equal(delta.cumsum(-1), cumulative):
        raise RuntimeError("crédito local não reconstrói retorno observado")
    models, ranges, counts = [], [], []
    rng = np.random.default_rng(9750)
    for action in range(4):
        x = z[:, action, :-1].reshape(-1, 32)
        mask = valid[:, action, :-1].reshape(-1)
        target = np.column_stack((delta[:, action, :8].reshape(-1), valid[:, action, 1:].reshape(-1)))
        x, target = x[mask], target[mask]
        if shuffled:
            target = target[rng.permutation(len(target))]
        models.append(KernelRidge().fit(x, target))
        ranges.append((float(target[:, 0].min()), float(target[:, 0].max())))
        counts.append({"states": len(x), "nonzero_reward": int(np.sum(target[:, 0] != 0)),
                       "terminations": int(np.sum(target[:, 1] == 0))})
    return models, ranges, counts


def rollout(initial, transitions, credit, ranges, frozen_state=False, permuted=False):
    state = np.repeat(initial[:, None], 4, axis=1)
    alive, total = np.ones(state.shape[:2]), np.zeros(state.shape[:2])
    values = []
    for _ in range(16):
        for action in range(4):
            index = (action + 1) % 4 if permuted else action
            forecast = credit[index].predict(state[:, action])
            reward = np.clip(forecast[:, 0], *ranges[index])
            continuation = np.clip(forecast[:, 1], 0, 1)
            total[:, action] += alive[:, action] * reward
            alive[:, action] *= continuation
            if not frozen_state:
                state[:, action] += transitions[index].predict(state[:, action])
        values.append(total.copy())
    return np.stack(values, axis=-1)


def evaluate(game, seed):
    start = time.process_time()
    train_path, _, signature, valid, _ = load_states(game, 97)
    source = ExperienceExtractor(game)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    z = encode(signature, basis["mean"], basis["basis"])
    train_horizon_path = HERE / "resultados" / f"{game}_horizontes_97.npz"
    test_horizon_path = HERE / "resultados" / f"{game}_horizontes_{seed}.npz"
    train_horizon, test_horizon = np.load(train_horizon_path), np.load(test_horizon_path)
    reward, actual = train_horizon["rewards"], test_horizon["rewards"]
    episode = test_horizon["episode"]
    initial_pair = reduce_frames(test_horizon["histories"][:, -2:])
    root_x = features(initial_pair)
    initial, _, _, _ = source.predict(root_x, set(range(8)))
    short = reduce_frames(test_horizon["pairs"][:, :, 0, 1]) - initial_pair[:, None, 1]
    oracle_initial = encode(short, basis["mean"], basis["basis"])
    transitions, _ = fit_updates(z, valid)
    credit, ranges, training_counts = fit_credit(z, valid, reward)
    shuffled, shuffled_ranges, _ = fit_credit(z, valid, reward, shuffled=True)
    forecasts = {
        "composed": rollout(initial, transitions, credit, ranges),
        "state_frozen": rollout(initial, transitions, credit, ranges, frozen_state=True),
        "action_models_permuted": rollout(initial, transitions, credit, ranges, permuted=True),
        "reward_association_shuffled": rollout(initial, transitions, shuffled, shuffled_ranges),
        "oracle_initial_composed": rollout(oracle_initial, transitions, credit, ranges),
    }
    prior = np.load(HERE / "resultados" / f"{game}_predicoes_sondas_experiencias.npz")
    curves = {}
    for horizon in (8, 32, 64, 128):
        stage = horizon // 8 - 1
        target = reward[:, :, stage]
        truth = actual[:, :, stage]
        constant = int(np.argmax(target.mean(0)))
        row = {}
        for name, forecast in forecasts.items():
            # measure() treina um decoder: para escolhas já previstas, usar
            # diretamente o argmax, sem ajustar com retornos de teste.
            choices = np.argmax(forecast[:, :, stage], axis=1)
            regret = truth.max(1) - truth[np.arange(64), choices]
            row[name] = {"mean_regret": float(regret.mean()),
                         "per_episode": [float(regret[episode == e].mean()) for e in range(8)],
                         "choices": choices.tolist()}
        prediction = ScaledRidge().fit(prior["full_zcv"], preferences(target)).predict(initial)
        choices = prediction.argmax(1)
        regret = truth.max(1) - truth[np.arange(64), choices]
        row["direct_preferences"] = {"mean_regret": float(regret.mean()),
                                     "per_episode": [float(regret[episode == e].mean()) for e in range(8)]}
        row["constant"] = {"mean_regret": float((truth.max(1) - truth[:, constant]).mean()), "action": constant}
        row["roots_with_reward_contrast"] = int(np.sum(np.ptp(truth, axis=1) > 0))
        curves[str(horizon)] = row
    local_diagnostic = None
    if seed == 103:
        _, _, real_signature, real_valid, _ = load_states(game, 103)
        real_z = encode(real_signature, basis["mean"], basis["basis"])
        truth_delta = increments(actual)
        rows = []
        for action in range(4):
            mask = real_valid[:, action, :-1].reshape(-1)
            points = real_z[:, action, :-1].reshape(-1, 32)[mask]
            truth = truth_delta[:, action, :8].reshape(-1)[mask]
            forecast = credit[action].predict(points)[:, 0]
            training_mask = valid[:, action, :-1].reshape(-1)
            training_truth = increments(reward)[:, action, :8].reshape(-1)[training_mask]
            denominator = np.mean((truth - training_truth.mean()) ** 2)
            rows.append({"action": action, "states": len(truth), "reward_events": int(np.sum(truth != 0)),
                         "reward_mse": float(np.mean((truth - forecast) ** 2)),
                         "action_only_mse": float(denominator),
                         "ratio": float(np.mean((truth - forecast) ** 2) / denominator) if denominator > 0 else None})
        local_diagnostic = rows
    report = {"status": "exploratory_not_confirmation", "game": game, "train_seed": 97, "test_seed": seed,
              "training_counts": training_counts, "curves": curves, "local_reward_with_real_state": local_diagnostic,
              "new_ale_steps": 0, "cpu_seconds": time.process_time() - start,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_state_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_valor_composto_{seed}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("--seed", type=int, choices=(103, 113), required=True)
    args = parser.parse_args()
    r = evaluate(args.game, args.seed)
    print(json.dumps({"game": args.game, "seed": args.seed, "counts": r["training_counts"],
                      "local": r["local_reward_with_real_state"],
                      "curves": {h: {name: data["mean_regret"] for name, data in row.items() if isinstance(data, dict)}
                                 for h, row in r["curves"].items()}}, ensure_ascii=False, indent=2))
