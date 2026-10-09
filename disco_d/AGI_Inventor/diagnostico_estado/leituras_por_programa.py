"""Funções sensoriais construídas por programa, com corpo completo e sem LLM."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from contrastes_programas import squared_scale, word_features


def flatten_history(history, two_frames=False):
    h = np.asarray(history, float)
    if h.ndim < 2 or not len(h) or not np.isfinite(h).all():
        raise ValueError("histórias inválidas")
    if two_frames:
        if h.ndim < 3 or h.shape[1] < 2:
            raise ValueError("história não contém dois quadros")
        h = h[:, -2:]
    return h.reshape(len(h), -1)


@dataclass
class ReadingBody:
    mean: np.ndarray
    scale: float
    templates: np.ndarray
    linear: np.ndarray
    bias: np.ndarray
    quadratic: np.ndarray
    two_frames: bool = False

    def apply(self, history):
        h = (flatten_history(history, self.two_frames) - self.mean) / self.scale
        if len(self.bias) not in (1, len(h)):
            raise ValueError("número de funções e histórias incompatível")
        linear = np.broadcast_to(self.linear, h.shape)
        q = np.broadcast_to(self.quadratic, (len(h), len(self.templates)))
        return self.bias + np.sum(h * linear, axis=1) + np.sum((h @ self.templates.T) ** 2 * q, axis=1)

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, mean=self.mean, scale=np.asarray(self.scale), templates=self.templates,
                            linear=self.linear, bias=self.bias, quadratic=self.quadratic,
                            two_frames=np.asarray(self.two_frames))
        body = {"operator": "raw_history_reading", "input": "past_numeric_observation_sequence",
                "body": "bias + dot(normalized_history,linear) + sum(quadratic * dot(normalized_history,templates)^2)",
                "parameters": path.name, "two_frames": self.two_frames,
                "scope": "learned_numeric_function_not_named_human_concept"}
        path.with_suffix(".json").write_text(json.dumps(body, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["mean"], float(data["scale"]), data["templates"], data["linear"],
                   data["bias"], data["quadratic"], bool(data["two_frames"]))


class ConditionalReading:
    def __init__(self, alpha=.05, actions=4, maximum=16):
        if not np.isfinite(alpha) or alpha <= 0 or actions < 1 or maximum < 1:
            raise ValueError("regularização/linguagem inválida")
        self.alpha, self.actions, self.maximum = alpha, actions, maximum

    def _word_kernel(self, a, b):
        return np.exp(-KernelRidge.distances(a, b) / self.word_scale)

    def _sensory_kernel(self, history):
        if self.mode == "word_only":
            return np.ones((len(history), len(self.support)))
        h = (flatten_history(history, self.mode == "two_frames") - self.mean) / self.scale
        u = h @ self.support.T
        return 1 + u + (u ** 2 if self.degree == 2 else 0)

    def fit(self, history, left, right, yl, yr, roots, mode="quadratic"):
        if mode not in ("quadratic", "affine", "two_frames", "shuffled_targets", "shuffled_histories", "word_only"):
            raise ValueError("modo inválido")
        self.mode, self.degree = mode, 1 if mode == "affine" else 2
        h = flatten_history(history, mode == "two_frames")
        roots, yl, yr = np.asarray(roots), np.asarray(yl, float), np.asarray(yr, float)
        self.p, self.q = word_features(left, self.actions, self.maximum), word_features(right, self.actions, self.maximum)
        if len(h) != len(self.p) or len(self.p) != len(self.q) or yl.shape != yr.shape or yl.shape != (len(h),):
            raise ValueError("dados de pares incompatíveis")
        if not np.isfinite(yl).all() or not np.isfinite(yr).all() or len(roots) != len(h):
            raise ValueError("alvos/origens inválidos")
        same = np.all(self.p == self.q, axis=1)
        if np.any((yl - yr)[same] != 0):
            raise ValueError("par determinístico contraditório")
        self.mean = h.mean(0)
        self.scale = float(np.sqrt(np.mean(np.sum((h - self.mean) ** 2, axis=1))))
        if self.scale < 1e-12:
            self.scale = 1.0
        rng = np.random.default_rng(9823)
        if mode == "shuffled_histories":
            unique = sorted(set(roots))
            first = {root: int(np.flatnonzero(roots == root)[0]) for root in unique}
            mapping = dict(zip(unique, rng.permutation(unique)))
            h = np.stack([h[first[mapping[root]]] for root in roots])
        self.support = (h - self.mean) / self.scale
        self.templates, self.origin = np.unique(self.support, axis=0, return_inverse=True)
        self.word_scale = squared_scale(np.concatenate((self.p, self.q)))
        target = yl - yr
        if mode == "shuffled_targets":
            lengths = np.asarray([len(p) for p in left])
            for length in sorted(set(lengths)):
                for identical in (False, True):
                    ids = np.flatnonzero((lengths == length) & (same == identical))
                    target[ids] = target[rng.permutation(ids)]
        if mode == "word_only":
            sensory = np.ones((len(h), len(h)))
        else:
            u = self.support @ self.support.T
            sensory = 1 + u + (u ** 2 if self.degree == 2 else 0)
        words = (self._word_kernel(self.p, self.p) - self._word_kernel(self.p, self.q)
                 - self._word_kernel(self.q, self.p) + self._word_kernel(self.q, self.q))
        base = sensory * words
        gram = np.block([[base, -base], [-base, base]])
        self.coeff = np.linalg.solve(gram + self.alpha * np.eye(len(gram)), np.concatenate((target, -target)))
        self.fit_target, self.actual_difference = target.copy(), (yl - yr).copy()
        self.system_size = len(gram)
        return self

    def construct(self, words):
        p = word_features(words, self.actions, self.maximum)
        n = len(self.p)
        coefficients = (self._word_kernel(p, self.p) - self._word_kernel(p, self.q)) * (self.coeff[:n] - self.coeff[n:])
        grouped = np.zeros((len(p), len(self.templates)))
        for source in range(n):
            grouped[:, self.origin[source]] += coefficients[:, source]
        bias = grouped.sum(1)
        if self.mode == "word_only":
            linear, quadratic = np.zeros((len(p), len(self.mean))), np.zeros_like(grouped)
        else:
            linear = grouped @ self.templates
            quadratic = grouped if self.degree == 2 else np.zeros_like(grouped)
        return ReadingBody(self.mean, self.scale, self.templates, linear, bias, quadratic, self.mode == "two_frames")

    def predict(self, history, left, right):
        return self.construct(left).apply(history) - self.construct(right).apply(history)

    def predict_dual(self, history, left, right):
        p, q = word_features(left, self.actions, self.maximum), word_features(right, self.actions, self.maximum)
        kernel = self._sensory_kernel(history) * (
            self._word_kernel(p, self.p) - self._word_kernel(p, self.q)
            - self._word_kernel(q, self.p) + self._word_kernel(q, self.q))
        n = len(self.p)
        return kernel @ (self.coeff[:n] - self.coeff[n:])

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, mean=self.mean, scale=np.asarray(self.scale), support=self.support,
                            templates=self.templates, origin=self.origin, p=self.p, q=self.q, coeff=self.coeff,
                            word_scale=np.asarray(self.word_scale), mode=np.asarray(self.mode), degree=np.asarray(self.degree),
                            alpha=np.asarray(self.alpha), actions=np.asarray(self.actions), maximum=np.asarray(self.maximum),
                            fit_target=self.fit_target, actual_difference=self.actual_difference, system_size=np.asarray(self.system_size))
        definition = {"operator": "construct_sensory_function_for_action_word", "parameters": path.name,
                      "mode": self.mode, "language": {"anonymous_actions": self.actions, "maximum_blocks": self.maximum},
                      "reading_family": "affine_plus_weighted_squared_projections_from_observed_history_templates",
                      "body": "word -> coefficients -> complete ReadingBody; history -> ReadingBody.apply",
                      "equivalent_fixed_features": "1, normalized_history coordinates, squared projections onto observed templates",
                      "scope": "known_finite_numeric_family_not_unrestricted_representation_invention"}
        path.with_suffix(".json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(float(data["alpha"]), int(data["actions"]), int(data["maximum"]))
        for name in ("mean", "support", "templates", "origin", "p", "q", "coeff", "fit_target", "actual_difference"):
            setattr(obj, name, data[name].copy())
        obj.mode, obj.degree = str(data["mode"]), int(data["degree"])
        obj.scale, obj.word_scale, obj.system_size = float(data["scale"]), float(data["word_scale"]), int(data["system_size"])
        return obj
