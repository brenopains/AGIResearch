"""Metade fixa (funções/transporte), metade resolvida (pesos de retorno).

Pesos ajustados por consequências de sequências não são automaticamente
recompensas locais corretas. Esta distinção fica no corpo exportado.
"""

import copy
import json
from pathlib import Path
import numpy as np


def credits(model, history, words):
    phi = model.features(history)
    if model.mode == "no_context":
        phi = np.broadcast_to(model.initial_prior, phi.shape)
    if len(phi) != len(words):
        raise ValueError("histórias e programas incompatíveis")
    width = len(model.initial_prior)
    output = np.zeros((len(phi), model.actions, width))
    for i, word in enumerate(words):
        model.word_value(word)  # Validate language.
        state = phi[i].copy()
        for action in word:
            output[i, action] += state
            if model.mode != "no_transport":
                state = state @ model.transport[action]
    return output.reshape(len(phi), -1)


def fit_credit(base, history, left, right, target, lengths, arm="sequence_credit", alpha=.05):
    if arm not in ("sequence_credit", "shuffled_targets", "no_context", "no_transport", "shuffled_histories"):
        raise ValueError("controle inválido")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("regularização inválida")
    model = copy.deepcopy(base)
    if arm in ("no_context", "no_transport"):
        model.mode = arm
    history, target = np.asarray(history, float), np.asarray(target, float).copy()
    lengths = np.asarray(lengths)
    if target.shape != (len(history),) or lengths.shape != target.shape or not np.isfinite(target).all():
        raise ValueError("alvos inválidos")
    if arm == "shuffled_histories":
        # Each group is one root repeated for the16 prefix lengths.
        roots = len(history) // 16
        if len(history) != roots * 16:
            raise ValueError("estrutura de episódios inválida")
        order = np.random.default_rng(9873).permutation(roots)
        history = history.reshape(roots, 16, *history.shape[1:])[order].reshape(history.shape)
    design = credits(model, history, left) - credits(model, history, right)
    identical = np.asarray([p == q for p, q in zip(left, right)])
    if np.any(np.abs(target[identical]) > 1e-12):
        raise ValueError("mesmo programa com respostas contraditórias")
    if arm == "shuffled_targets":
        rng = np.random.default_rng(9871)
        for length in sorted(set(lengths)):
            for same in (False, True):
                ids = np.flatnonzero((lengths == length) & (identical == same))
                target[ids] = target[rng.permutation(ids)]
    previous = model.reward.reshape(-1).copy()
    scale = float(np.sqrt(np.mean(np.sum(design * design, axis=1))))
    if scale < 1e-12:
        scale = 1.
    x = np.concatenate((design, -design)) / scale
    residual = np.concatenate((target - design @ previous, -target + design @ previous))
    coefficient = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ residual)
    model.reward = (previous + coefficient / scale).reshape(model.reward.shape)
    model.mode = arm if arm in ("no_context", "no_transport") else "transport"
    minimum, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-10)
    report = {"observed_prefix_pairs": len(target), "mirrored_rows": len(x), "solved_parameters": x.shape[1],
              "feature_scale": scale, "alpha": alpha, "numerical_rank_rcond_1e_10": int(rank),
              "training_delta_mse": float(np.mean((design @ model.reward.reshape(-1) - target) ** 2)),
              "unregularized_training_delta_mse_rcond_1e_10": float(np.mean((design @ minimum - target) ** 2)),
              "maximum_singular_value": float(singular.max()), "minimum_singular_value": float(singular.min()),
              "feature_body_unchanged": bool(np.array_equal(model.support, base.support)),
              "transport_body_unchanged": bool(np.array_equal(model.transport, base.transport))}
    return model, report, design, target


def save_body(path, model, arm, report):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise RuntimeError("corpo existente; preservar")
    # Array schema is compatible with the frozen executor, metadata is accurate
    # about the new sequence target rather than copying the old local target.
    np.savez_compressed(path, support=model.support, scale=model.scale, reward=model.reward,
                        transport=model.transport, initial_prior=model.initial_prior, mode=model.mode,
                        actions=model.actions, alpha=model.alpha, supports_limit=model.supports_limit,
                        action_counts=model.action_counts, local_residual_mse=model.local_residual_mse)
    definition = {"operator": "sequence_return_credit_with_frozen_observable_transport", "arm": arm,
                  "body": "phi(history) @ recursively_composed_value(word)",
                  "learned_weights_target": "observed_prefix_return_differences_not_local_rewards",
                  "preserved_arrays": ["support", "scale", "transport", "initial_prior"],
                  "legacy_local_residual_mse_array": "original_local_fit_metadata_not_new_head_accuracy",
                  "parameters": path.name, "fit": report,
                  "scope": "effective_sequence_weights_may_compensate_dynamics_error_not_verified_causal_rewards",
                  "human_level_goal_achieved": False}
    with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(definition, stream, indent=2, allow_nan=False)
