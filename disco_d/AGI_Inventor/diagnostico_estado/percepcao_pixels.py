"""Extrai posições e movimento de quadros, sem consultar o estado físico."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from avaliar import group_split
from geral import _foreground, _sensor_values
from oraculo import HERE, TIMES, metrics, reproduce_oracle
from prefixo import predict


def components(frame):
    mask = _foreground(frame).astype(bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    blobs = []
    for label in range(1, count + 1):
        y, x = np.nonzero(labels == label)
        if len(x) >= 4:
            blobs.append((len(x), float(x.mean()), float(y.mean())))
    blobs.sort(key=lambda b: -b[0])
    return np.asarray([[x, y] for _, x, y in blobs[:3]], np.float32).reshape(-1, 2)


def associate(current, old):
    """Menor custo injetivo para até três pontos, sem identidades do simulador."""
    if len(current) == 0 or len(old) == 0:
        return np.full((len(current), 2), np.nan, np.float32)
    best_cost, best_pairs = float("inf"), ()
    m = min(len(current), len(old))
    for chosen_now in itertools.combinations(range(len(current)), m):
        for chosen_old in itertools.permutations(range(len(old)), m):
            cost = sum(np.linalg.norm(current[i] - old[j]) for i, j in zip(chosen_now, chosen_old))
            if cost < best_cost:
                best_cost, best_pairs = cost, tuple(zip(chosen_now, chosen_old))
    result = np.full((len(current), 2), np.nan, np.float32)
    for i, j in best_pairs:
        result[i] = old[j]
    return result


def describe(frames, t):
    now = components(frames[t])
    order = np.argsort(now[:, 0], kind="stable")
    now = now[order]
    past3 = associate(now, components(frames[t - 3]))
    past8 = associate(now, components(frames[t - 8]))
    entries = []
    for i in range(3):
        if i < len(now):
            v3 = (now[i] - past3[i]) / 3 if np.isfinite(past3[i]).all() else np.zeros(2)
            v8 = (now[i] - past8[i]) / 8 if np.isfinite(past8[i]).all() else np.zeros(2)
            entries.extend((now[i, 0] / 128, now[i, 1] / 128,
                            v3[0] / 4, v3[1] / 4, v8[0] / 4, v8[1] / 4, 1.0))
        else:
            entries.extend((0.5, 0.5, 0, 0, 0, 0, 0))
    return np.asarray(entries, np.float32), now


def describe_tracking(frames, t):
    """Liga componentes em 16 quadros e propaga trajetórias durante oclusões."""
    tracks = []
    for k in range(t + 1):
        observed = components(frames[k])
        assigned_tracks, assigned_points = set(), set()
        if tracks and len(observed):
            expected = np.stack([track["pos"] + track["velocity"] for track in tracks])
            distance = np.linalg.norm(expected[:, None] - observed[None], axis=2)
            rows, cols = linear_sum_assignment(distance)
            for row, col in zip(rows, cols):
                if distance[row, col] > 8:
                    continue
                track = tracks[row]
                elapsed = k - track["last_seen"]
                measured = (observed[col] - track["last_observed"]) / elapsed
                track["velocity"] = 0.6 * measured + 0.4 * track["velocity"]
                track["pos"] = observed[col]
                track["last_observed"] = observed[col].copy()
                track["last_seen"] = k
                assigned_tracks.add(row)
                assigned_points.add(col)
        for row, track in enumerate(tracks):
            if row not in assigned_tracks:
                track["pos"] = track["pos"] + track["velocity"]
        for col, point in enumerate(observed):
            if col not in assigned_points and len(tracks) < 3:
                tracks.append({"pos": point.copy(), "last_observed": point.copy(),
                               "velocity": np.zeros(2, np.float32), "last_seen": k})
    tracks.sort(key=lambda track: float(track["pos"][0]))
    entries = []
    for i in range(3):
        if i < len(tracks):
            track = tracks[i]
            age = t - track["last_seen"]
            entries.extend((track["pos"][0] / 128, track["pos"][1] / 128,
                            track["velocity"][0] / 4, track["velocity"][1] / 4,
                            min(age, 16) / 16, 1.0, 1.0))
        else:
            entries.extend((0.5, 0.5, 0, 0, 1, 0, 0))
    return np.asarray(entries, np.float32), np.asarray(
        [track["pos"] for track in tracks], np.float32).reshape(-1, 2)


def replay_pixels(bank, seed=53):
    sys.path.insert(0, str(HERE.parent / "representacao"))
    import mundo_orbitas as world

    world.DUR = 5.0
    rng = np.random.default_rng(seed)
    camera = world.camera(rng)
    features, detections, tracks, tracked_detections, checks, truths = [], [], [], [], [], []
    for episode in range(8):
        frames, _, truth = world.episodio(rng, camera, "A" if episode < 6 else "B", True)
        for t in TIMES:
            feature, found = describe(frames, t)
            trajectory, tracked = describe_tracking(frames, t)
            features.append(feature)
            detections.append(found)
            tracks.append(trajectory)
            tracked_detections.append(tracked)
            truths.append(truth["pix"][:, t])
            checks.append(_sensor_values(frames[t + 45]))
    data = np.load(bank)
    if not np.array_equal(np.asarray(checks, np.float32), data["values"]):
        raise RuntimeError("quadros regenerados não coincidem com o banco congelado")
    return np.asarray(features), detections, np.asarray(tracks), tracked_detections, truths


def location_audit(found, true, indices):
    errors, counts = [], []
    for i in indices:
        obs, target = found[i], true[i]
        counts.append(len(obs))
        if len(obs) == len(target):
            errors.append(min(sum(np.linalg.norm(obs[k] - target[p[k]]) for k in range(3))
                              for p in itertools.permutations(range(3))) / 3)
    return {"mean_components": float(np.mean(counts)),
            "fraction_three_components": float(np.mean(np.asarray(counts) == 3)),
            "mean_matching_error_when_three_pixels": float(np.mean(errors)) if errors else None}


def run(bank, nulls=32, seed=53):
    data = np.load(bank)
    if str(data["domain"]) != "orbitas":
        raise ValueError("este extrator só foi avaliado em órbitas")
    features, detections, tracks, tracked_detections, true = replay_pixels(bank, seed=seed)
    h = data["histories"].astype(np.float32)
    pixel = np.concatenate((h[:, -1], h[:, -1] - h[:, -2]), axis=1)
    values, episode = data["values"], data["episode"]
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    oracle = reproduce_oracle(bank, seed=seed).reshape(-1, 3, 4)
    tracked = tracks.reshape(-1, 3, 7)
    tracked_position = tracked[..., :2]
    tracked_velocity = tracked[..., 2:4]
    score = {}
    for name, x in (("pixels", pixel), ("entidades_extraidas", features),
                    ("rastreador_temporal", tracks),
                    ("oraculo_posicao", oracle[..., :2].reshape(len(oracle), -1)),
                    ("oraculo_velocidade", oracle[..., 2:].reshape(len(oracle), -1)),
                    ("posicao_extraida_velocidade_oracular",
                     np.concatenate((tracked_position, oracle[..., 2:]), axis=2).reshape(len(oracle), -1)),
                    ("posicao_oracular_velocidade_extraida",
                     np.concatenate((oracle[..., :2], tracked_velocity), axis=2).reshape(len(oracle), -1))):
        score[name] = metrics(values, predict(x, values, fit, test), episode, test)
    rng = np.random.default_rng(2712)
    shuffled_regrets = []
    shuffled_tracking_regrets = []
    for _ in range(nulls):
        shuffled = features.copy()
        shuffled_tracking = tracks.copy()
        for split in (fit, test):
            permutation = rng.permutation(split)
            shuffled[split] = features[permutation]
            shuffled_tracking[split] = tracks[permutation]
        shuffled_regrets.append(metrics(values, predict(shuffled, values, fit, test),
                                        episode, test)["overall"]["normalized_regret"])
        shuffled_tracking_regrets.append(metrics(values, predict(shuffled_tracking, values, fit, test),
                                                 episode, test)["overall"]["normalized_regret"])
    observed = score["entidades_extraidas"]["overall"]["normalized_regret"]
    observed_tracking = score["rastreador_temporal"]["overall"]["normalized_regret"]
    flat_oracle = oracle.reshape(len(oracle), -1)
    score["oraculo"] = metrics(values, predict(flat_oracle, values, fit, test), episode, test)
    return {"bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
            "feature_dim": int(features.shape[1]), "scores": score,
            "localization_fit": location_audit(detections, true, fit),
            "localization_test": location_audit(detections, true, test),
            "tracking_fit": location_audit(tracked_detections, true, fit),
            "tracking_test": location_audit(tracked_detections, true, test),
            "shuffled_regrets": shuffled_regrets,
            "shuffled_equal_or_better": sum(x <= observed for x in shuffled_regrets),
            "shuffled_tracking_regrets": shuffled_tracking_regrets,
            "shuffled_tracking_equal_or_better": sum(
                x <= observed_tracking for x in shuffled_tracking_regrets),
            "null_count": nulls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    result = run(args.bank, seed=args.seed)
    output = args.bank.with_name(args.bank.stem + "_percepcao_pixels.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("shuffled_regrets", "shuffled_tracking_regrets")},
                     ensure_ascii=False, indent=2))
