"""Comparação de programas com corpo completo e simetrias verificáveis."""

import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge


def word_features(words, actions=4, maximum=16):
    if not len(words):
        raise ValueError("nenhum programa")
    rows = []
    for word in words:
        if not len(word) or len(word) > maximum or any(not isinstance(a, (int, np.integer)) or not 0 <= a < actions for a in word):
            raise ValueError("programa fora da linguagem")
        symbols = list(word) + [actions] * (maximum - len(word))
        rows.append(np.eye(actions + 1)[symbols].reshape(-1))
    return np.asarray(rows)


def squared_scale(x):
    d = KernelRidge.distances(x, x)
    nonzero = d[np.triu_indices(len(d), 1)]
    nonzero = nonzero[nonzero > 1e-14]
    return float(np.median(nonzero)) if len(nonzero) else 1.0


class ProgramComparison:
    def __init__(self, alpha=.05, actions=4, maximum=16):
        if not np.isfinite(alpha) or alpha <= 0 or actions < 1 or maximum < 1:
            raise ValueError("regularização ou linguagem inválida")
        self.alpha, self.actions, self.maximum = alpha, actions, maximum

    def _state_kernel(self, left, right):
        if self.mode == "word_only":
            return np.ones((len(left), len(right)))
        return np.exp(-KernelRidge.distances(left, right) / self.state_scale)

    def _word_kernel(self, left, right):
        return np.exp(-KernelRidge.distances(left, right) / self.word_scale)

    def _pair_kernel(self, x, p, q):
        return self._state_kernel(x, self.x) * (
            self._word_kernel(p, self.p) - self._word_kernel(p, self.q)
            - self._word_kernel(q, self.p) + self._word_kernel(q, self.q))

    def fit(self, x, left, right, returns_left, returns_right, roots, mode="conditional"):
        if mode not in ("conditional", "shuffled_targets", "shuffled_states", "word_only", "absolute_returns"):
            raise ValueError("modo inválido")
        self.mode = mode
        roots = np.asarray(roots)
        self.x = np.asarray(x, float).copy()
        self.p, self.q = word_features(left, self.actions, self.maximum), word_features(right, self.actions, self.maximum)
        yl, yr = np.asarray(returns_left, float), np.asarray(returns_right, float)
        if (self.x.ndim != 2 or len(self.x) != len(yl) or yl.shape != yr.shape
                or yl.shape != (len(left),) or len(right) != len(left)
                or len(roots) != len(left) or not all(np.isfinite(v).all() for v in (self.x, yl, yr))):
            raise ValueError("pares inválidos")
        same = np.all(self.p == self.q, axis=1)
        if np.any((yl - yr)[same] != 0):
            raise ValueError("experiência contraditória: mesmo programa com resultados diferentes")
        self.state_scale = squared_scale(self.x)
        self.word_scale = squared_scale(np.concatenate((self.p, self.q)))
        rng = np.random.default_rng(9813)
        if mode == "shuffled_states":
            unique = sorted(set(roots))
            first = {root: int(np.flatnonzero(roots == root)[0]) for root in unique}
            mapping = dict(zip(unique, rng.permutation(unique)))
            self.x = np.stack([x[first[mapping[root]]] for root in roots])
        target = yl - yr
        if mode == "shuffled_targets":
            lengths = np.asarray([len(p) for p in left])
            for length in sorted(set(lengths)):
                for identical in (False, True):
                    ids = np.flatnonzero((lengths == length) & (same == identical))
                    target[ids] = target[rng.permutation(ids)]
        if mode == "absolute_returns":
            states, programs = np.concatenate((self.x, self.x)), np.concatenate((self.p, self.q))
            gram = self._state_kernel(states, states) * self._word_kernel(programs, programs)
            target = np.concatenate((yl, yr))
            self.mean_return = float(target.mean())
            self.coeff = np.linalg.solve(gram + self.alpha * np.eye(len(gram)), target - self.mean_return)
        else:
            base = self._pair_kernel(self.x, self.p, self.q)
            gram = np.block([[base, -base], [-base, base]])
            mirrored = np.concatenate((target, -target))
            self.mean_return = 0.0
            self.coeff = np.linalg.solve(gram + self.alpha * np.eye(len(gram)), mirrored)
        self.training_difference = (yl - yr).copy()
        self.fit_target = target.copy()
        self.system_size = len(gram)
        return self

    def predict(self, x, left, right):
        x = np.asarray(x, float)
        p, q = word_features(left, self.actions, self.maximum), word_features(right, self.actions, self.maximum)
        if len(x) != len(p) or len(p) != len(q):
            raise ValueError("consultas incompatíveis")
        if self.mode == "absolute_returns":
            states, programs = np.concatenate((self.x, self.x)), np.concatenate((self.p, self.q))
            difference = self._word_kernel(p, programs) - self._word_kernel(q, programs)
            return (self._state_kernel(x, states) * difference) @ self.coeff
        kernel = self._pair_kernel(x, p, q)
        n = len(self.x)
        return kernel @ (self.coeff[:n] - self.coeff[n:])

    def save(self, path):
        path = Path(path)
        np.savez_compressed(path, x=self.x, p=self.p, q=self.q, coeff=self.coeff,
                            mode=np.asarray(self.mode), alpha=np.asarray(self.alpha),
                            state_scale=np.asarray(self.state_scale), word_scale=np.asarray(self.word_scale),
                            mean_return=np.asarray(self.mean_return), fit_target=self.fit_target,
                            training_difference=self.training_difference, system_size=np.asarray(self.system_size))
        body = {"operator": "program_return_difference", "input": "state_vector_and_two_action_words",
                "mode": self.mode, "parameters": path.name,
                "body": "state_RBF * (word_RBF(P,L)-word_RBF(P,R)-word_RBF(Q,L)+word_RBF(Q,R)) @ mirrored_coefficients",
                "word_language": {"maximum_blocks": self.maximum, "anonymous_actions": self.actions, "end_symbol": self.actions},
                "guarantees": ["self_zero", "swap_negates", "additive_differences"],
                "does_not_guarantee": "correct_difference_or_human_intelligence"}
        if self.mode == "absolute_returns":
            body["body"] = "(joint_kernel(state,P,support)-joint_kernel(state,Q,support)) @ absolute_return_coefficients"
        path.with_suffix(".json").write_text(json.dumps(body, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        definition = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        language = definition["word_language"]
        obj = cls(float(data["alpha"]), language["anonymous_actions"], language["maximum_blocks"])
        for name in ("x", "p", "q", "coeff", "fit_target", "training_difference"):
            setattr(obj, name, data[name].copy())
        obj.mode = str(data["mode"])
        obj.state_scale, obj.word_scale = float(data["state_scale"]), float(data["word_scale"])
        obj.mean_return, obj.system_size = float(data["mean_return"]), int(data["system_size"])
        return obj
