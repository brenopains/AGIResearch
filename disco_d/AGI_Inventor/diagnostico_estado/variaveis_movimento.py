"""Converte história visual em trilhas de mudanças, sem semântica do domínio.

Operações admitidas: fundo temporal robusto, componentes conexos de inovação,
associação por continuidade, posição/velocidade/tamanho e relações entre trilhas.
Nenhuma informação de recompensa, ação, jogo ou rótulo é usada na extração.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

SLOTS = 4
THRESHOLD = 16.0
MAX_GAP = 3


def components(frame, background, polarity):
    change = polarity * (frame.astype(np.float32) - background)
    mask = change >= THRESHOLD
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
    objects = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        area = len(xx)
        if area == 0 or area > frame.size // 20:
            continue
        mass = change[yy, xx]
        objects.append({"pos": np.array([np.average(xx, weights=mass),
                                           np.average(yy, weights=mass)], np.float32),
                        "area": float(area), "contrast": float(np.mean(mass))})
    return objects


def track(history, polarity):
    frames = np.asarray(history, np.uint8)
    background = np.median(frames.astype(np.float32), axis=0)
    tracks = []
    for t, frame in enumerate(frames):
        found = components(frame, background, polarity)
        eligible = [i for i, item in enumerate(tracks) if t - item["last"] <= MAX_GAP]
        assigned_tracks, assigned_objects = set(), set()
        if eligible and found:
            cost = np.zeros((len(eligible), len(found)), np.float32)
            for row, ix in enumerate(eligible):
                item = tracks[ix]
                elapsed = t - item["last"]
                predicted = item["pos"] + item["velocity"] * elapsed
                for col, observation in enumerate(found):
                    cost[row, col] = np.linalg.norm(predicted - observation["pos"])
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                ix = eligible[row]
                if cost[row, col] > 24:
                    continue
                item = tracks[ix]
                observation = found[col]
                elapsed = t - item["last"]
                measured = (observation["pos"] - item["pos"]) / elapsed
                item["velocity"] = 0.5 * measured + 0.5 * item["velocity"]
                item["pos"] = observation["pos"]
                item["area"] = observation["area"]
                item["contrast"] = observation["contrast"]
                item["seen"] += 1
                item["last"] = t
                assigned_tracks.add(ix)
                assigned_objects.add(col)
        for col, observation in enumerate(found):
            if col not in assigned_objects:
                tracks.append({"pos": observation["pos"].copy(),
                               "velocity": np.zeros(2, np.float32),
                               "area": observation["area"],
                               "contrast": observation["contrast"],
                               "seen": 1, "last": t})
    recent = [item for item in tracks if len(frames) - 1 - item["last"] <= MAX_GAP]
    recent.sort(key=lambda item: (
        -(item["seen"] / len(frames)
          + 0.5 * min(float(np.linalg.norm(item["velocity"])) / 16, 1)
          + 0.1 * np.log1p(item["area"])),
        float(item["pos"][1]), float(item["pos"][0])))
    return recent[:SLOTS]


def describe_tracks(tracks, width, height, duration):
    result = []
    positions = []
    velocities = []
    for slot in range(SLOTS):
        if slot < len(tracks):
            item = tracks[slot]
            pos = item["pos"] / np.array([width, height], np.float32)
            velocity = item["velocity"] / np.array([width, height], np.float32) * 4
            positions.append(pos)
            velocities.append(velocity)
            result.extend((float(pos[0]), float(pos[1]),
                           float(velocity[0]), float(velocity[1]),
                           float(np.log1p(item["area"]) / 8),
                           float(item["contrast"] / 255),
                           float(item["seen"] / duration),
                           float((duration - 1 - item["last"]) / duration), 1.0))
        else:
            positions.append(np.zeros(2, np.float32))
            velocities.append(np.zeros(2, np.float32))
            result.extend((0.0,) * 9)
    for left in range(SLOTS):
        for right in range(left + 1, SLOTS):
            dp = positions[left] - positions[right]
            dv = velocities[left] - velocities[right]
            result.extend((float(dp[0]), float(dp[1]), float(dv[0]), float(dv[1])))
    return result


def describe(history):
    frames = np.asarray(history)
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("esperada história de quadros monocromáticos")
    height, width = frames.shape[1:]
    vector = []
    for polarity in (1, -1):
        vector.extend(describe_tracks(track(frames, polarity), width, height, len(frames)))
    return np.asarray(vector, np.float32)


def batch(histories):
    return np.stack([describe(history) for history in histories])
