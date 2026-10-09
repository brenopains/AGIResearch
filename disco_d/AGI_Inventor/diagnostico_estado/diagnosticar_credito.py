"""Substituições por dados reais localizam a falha de crédito/composição.

Todos os modelos treinam somente em 97. As trajetórias de 103 são oráculos
diagnósticos, nunca observações disponíveis à política na raiz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from avaliar_composicao import features
from avaliar_estado_preditivo import encode, load
from valor_composto import increments

HERE = Path(__file__).resolve().parent


def summarize(values, truth, episode):
    choices = np.argmax(values, axis=1)
    regret = truth.max(1) - truth[np.arange(len(truth)), choices]
    contrast = truth - truth.mean(1, keepdims=True)
    predicted = values - values.mean(1, keepdims=True)
    denominator = float(np.mean(contrast ** 2))
    return {"mean_regret": float(regret.mean()), "choices": choices.tolist(),
            "per_episode": [float(regret[episode == e].mean()) for e in range(8)],
            "contrast_mse": float(np.mean((contrast - predicted) ** 2)),
            "zero_contrast_mse": denominator,
            "contrast_error_ratio": float(np.mean((contrast - predicted) ** 2) / denominator)
            if denominator > 0 else None}


def diagnose(game):
    train_path, train_pairs, train_s, train_valid, _ = load(game, 97)
    test_path, test_pairs, test_s, test_valid, episode = load(game, 103)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    train_h = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")
    test_h = np.load(HERE / "resultados" / f"{game}_horizontes_103.npz")
    train_r, test_r = increments(train_h["rewards"]), increments(test_h["rewards"])
    representations = {
        "true_latent_state": (encode(train_s, basis["mean"], basis["basis"]),
                              encode(test_s, basis["mean"], basis["basis"])),
        "true_full_signature": (train_s.reshape(64, 4, 9, -1), test_s.reshape(64, 4, 9, -1)),
        "true_pixel_pair": (features(train_pairs), features(test_pairs)),
    }
    curves = {str(h): {} for h in (8, 32, 64)}
    event_errors = {}
    for name, (train_x, test_x) in representations.items():
        prediction = np.zeros((64, 4, 8))
        action_rows = []
        for action in range(4):
            train_mask = train_valid[:, action, :-1].reshape(-1)
            test_mask = test_valid[:, action, :-1].reshape(-1)
            x = train_x[:, action, :-1].reshape(-1, train_x.shape[-1])[train_mask]
            target = train_r[:, action, :8].reshape(-1)[train_mask]
            query = test_x[:, action, :-1].reshape(-1, test_x.shape[-1])[test_mask]
            forecast = KernelRidge().fit(x, target[:, None]).predict(query).ravel()
            forecast = np.clip(forecast, target.min(), target.max())
            flattened = prediction[:, action].reshape(-1).copy()
            flattened[test_mask] = forecast
            prediction[:, action] = flattened.reshape(64, 8)
            actual = test_r[:, action, :8].reshape(-1)[test_mask]
            events = actual != 0
            action_rows.append({"action": action, "events": int(events.sum()),
                                "event_mse": float(np.mean((actual[events] - forecast[events]) ** 2))
                                if events.any() else None,
                                "zero_event_mse": float(np.mean(forecast[~events] ** 2))
                                if (~events).any() else None})
        event_errors[name] = action_rows
        cumulative = prediction.cumsum(-1)
        for horizon in (8, 32, 64):
            index = horizon // 8 - 1
            curves[str(horizon)][name] = summarize(cumulative[:, :, index],
                                                   test_h["rewards"][:, :, index], episode)
    for horizon in (8, 32, 64):
        index = horizon // 8 - 1
        truth = test_h["rewards"][:, :, index]
        constant = int(train_h["rewards"][:, :, index].mean(0).argmax())
        values = np.zeros_like(truth)
        values[:, constant] = 1
        curves[str(horizon)]["constant"] = summarize(values, truth, episode)
        # Controle positivo: recompensa observada em cada bloco reconstrói o
        # retorno. Não permitir uma implementação ruim do oráculo parecer gap.
        perfect = test_r[:, :, :index + 1].sum(-1)
        if not np.array_equal(perfect, truth):
            raise RuntimeError("incrementos não reproduzem o retorno")
        curves[str(horizon)]["true_reward_oracle"] = summarize(perfect, truth, episode)
    report = {"status": "exploratory_oracle_diagnostic_not_agent", "game": game,
              "train_seed": 97, "test_seed": 103, "curves": curves,
              "event_errors": event_errors, "new_ale_steps": 0,
              "test_future_visible_only_to_diagnostic": True,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_bank_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "test_bank_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
    output = HERE / "resultados" / f"{game}_diagnostico_credito.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    report = diagnose(args.game)
    print(json.dumps({"game": args.game,
                      "curves": {h: {k: r["mean_regret"] for k, r in row.items()}
                                 for h, row in report["curves"].items()},
                      "events": report["event_errors"]}, ensure_ascii=False, indent=2))
