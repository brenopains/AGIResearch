"""Rastreio perceptivo com hipóteses alternativas de identidade e oclusão."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from percepcao_pixels import components


@dataclass
class Track:
    pos: np.ndarray
    velocity: np.ndarray
    last_observed: np.ndarray
    last_seen: int


@dataclass
class Hypothesis:
    tracks: list[Track]
    cost: float


def clone(track):
    return Track(track.pos.copy(), track.velocity.copy(),
                 track.last_observed.copy(), track.last_seen)


def assignments(count_tracks, count_points):
    """Mapeia cada trilha a uma observação distinta ou a uma falta (-1)."""
    for mapping in itertools.product(range(-1, count_points), repeat=count_tracks):
        used = [i for i in mapping if i >= 0]
        if len(used) == len(set(used)):
            yield mapping


def missing_cost(frame, prediction, observed):
    x, y = np.rint(prediction).astype(int)
    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
        rgb = frame[y, x].astype(np.int16)
        if rgb.min() > 50 and np.ptp(rgb) < 8:
            return 0.2
    if len(observed) and np.min(np.linalg.norm(observed - prediction, axis=1)) < 6:
        return 0.5
    return 3.0


def advance(hypothesis, points, frame, t):
    old = hypothesis.tracks
    for mapping in assignments(len(old), len(points)):
        updated = [clone(track) for track in old]
        cost = hypothesis.cost
        used = set()
        valid = True
        for i, j in enumerate(mapping):
            track = updated[i]
            prediction = track.pos + track.velocity
            if j < 0:
                cost += missing_cost(frame, prediction, points)
                track.pos = prediction
                continue
            distance = float(np.linalg.norm(prediction - points[j]))
            if distance > 12:
                valid = False
                break
            cost += distance**2 / 12.5
            elapsed = t - track.last_seen
            measured = (points[j] - track.last_observed) / elapsed
            track.velocity = 0.6 * measured + 0.4 * track.velocity
            track.pos = points[j].copy()
            track.last_observed = points[j].copy()
            track.last_seen = t
            used.add(j)
        if not valid:
            continue
        for j, point in enumerate(points):
            if j not in used:
                if len(updated) < 3:
                    updated.append(Track(point.copy(), np.zeros(2, np.float32),
                                         point.copy(), t))
                    cost += 0.8
                else:
                    cost += 4.0
        yield Hypothesis(updated, cost)


def signature(hypothesis):
    rows = sorted((round(float(track.pos[0]) / 2),
                   round(float(track.pos[1]) / 2),
                   round(float(track.velocity[0]) * 2),
                   round(float(track.velocity[1]) * 2))
                  for track in hypothesis.tracks)
    return tuple(rows)


def filter_times(frames, times, beam_size=16):
    beam = [Hypothesis([], 0.0)]
    output = {}
    wanted = set(times)
    for t in range(max(times) + 1):
        points = components(frames[t])
        candidates = []
        for hypothesis in beam:
            candidates.extend(advance(hypothesis, points, frames[t], t))
        candidates.sort(key=lambda h: h.cost)
        selected, seen = [], set()
        for candidate in candidates:
            key = signature(candidate)
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= beam_size:
                break
        if not selected:
            raise RuntimeError("todas as hipóteses foram descartadas")
        minimum = selected[0].cost
        for candidate in selected:
            candidate.cost -= minimum
        beam = selected
        if t in wanted:
            output[t] = beam
    return output


def filter_frames(frames, until, beam_size=16):
    return filter_times(frames, (until,), beam_size)[until]


def describe_hypothesis(hypothesis, t):
    tracks = sorted(hypothesis.tracks, key=lambda track: float(track.pos[0]))
    entries = []
    for i in range(3):
        if i < len(tracks):
            track = tracks[i]
            entries.extend((track.pos[0] / 128, track.pos[1] / 128,
                            track.velocity[0] / 4, track.velocity[1] / 4,
                            min(t - track.last_seen, 16) / 16, 1.0, 1.0))
        else:
            entries.extend((0.5, 0.5, 0, 0, 1, 0, 0))
    return np.asarray(entries, np.float32)


def describe_beam(beam, t):
    descriptions = np.stack([describe_hypothesis(hypothesis, t) for hypothesis in beam])
    scores = -np.asarray([hypothesis.cost for hypothesis in beam]) / 2
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    mean = weights @ descriptions
    uncertainty = float(np.mean(np.var(descriptions.reshape(len(beam), 3, 7)[..., :2], axis=0)))
    return descriptions[0], mean.astype(np.float32), uncertainty
