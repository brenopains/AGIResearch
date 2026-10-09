"""Leituras executáveis de consequências locais aprendidas da experiência.

Nenhum objeto, regra ou nome de domínio. Os parâmetros completos ficam
exportáveis; a função usa um kernel numérico, sem executar código arbitrário.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge


class ConsequenceReadout:
    def fit(self, edges, shuffled=False):
        self.models, self.reward_ranges = [], []
        rng = np.random.default_rng(9769)
        for e in edges:
            target = np.column_stack((e["reward"], e["continuation"]))
            if shuffled:
                target = target[rng.permutation(len(target))]
            self.models.append(KernelRidge().fit(e["source"], target))
            self.reward_ranges.append((float(target[:, 0].min()), float(target[:, 0].max())))
        return self

    def predict_action(self, state, action):
        result = self.models[action].predict(state)
        result[:, 0] = np.clip(result[:, 0], *self.reward_ranges[action])
        result[:, 1] = np.clip(result[:, 1], 0, 1)
        return result

    def coordinates(self, state):
        return np.column_stack([self.predict_action(state, a) for a in range(len(self.models))])

    def save(self, path):
        path = Path(path)
        parameters = {}
        for a, model in enumerate(self.models):
            for key in ("mean", "x", "ym", "scale", "coeff"):
                parameters[f"action_{a}_{key}"] = np.asarray(getattr(model, key))
        parameters["reward_ranges"] = np.asarray(self.reward_ranges)
        np.savez_compressed(path, **parameters)
        definition = {"operator": "rbf_weighted_sum_then_observed_range_clip",
                      "actions": len(self.models), "input_type": "finite_real_vector",
                      "output_type": "reward_and_continuation_per_action",
                      "body": "exp(-squared_distance(state-mean,x)/scale) @ coeff + ym",
                      "parameters": path.name, "regularization": 0.05,
                      "scope": "local_predictions_not_objects_or_general_intelligence"}
        path.with_suffix(".json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = cls()
        data = np.load(path)
        obj.reward_ranges = data["reward_ranges"].tolist()
        obj.models = []
        for a in range(len(obj.reward_ranges)):
            model = KernelRidge()
            for key in ("mean", "x", "ym", "scale", "coeff"):
                value = data[f"action_{a}_{key}"]
                setattr(model, key, float(value) if key == "scale" else value.copy())
            obj.models.append(model)
        return obj


class ConsequenceGeometry:
    def __init__(self, readout, support, random=False):
        self.readout = readout
        self.projection = None
        if random:
            rng = np.random.default_rng(9770)
            self.projection = rng.normal(size=(support.shape[1], 2 * len(readout.models)))
        q = self.response(support)
        self.center_z, self.center_q = support.mean(0), q.mean(0)
        # Amostra determinística limita custo de escala, sem olhar teste.
        selected = np.linspace(0, len(support) - 1, min(512, len(support))).astype(int)
        self.scale_z = self.distance_scale(support[selected])
        self.scale_q = self.distance_scale(q[selected])

    @staticmethod
    def distance_scale(x):
        distance = KernelRidge.distances(x, x)
        nonzero = distance[np.triu_indices(len(x), 1)]
        nonzero = nonzero[nonzero > 1e-12]
        return float(np.sqrt(np.median(nonzero))) if len(nonzero) else 1.0

    def response(self, state):
        return state @ self.projection if self.projection is not None else self.readout.coordinates(state)

    def transform(self, state):
        return np.column_stack(((state - self.center_z) / self.scale_z,
                                (self.response(state) - self.center_q) / self.scale_q)) / np.sqrt(2)


def prepare_edges(edges, credit, geometry=None):
    output = []
    for a, e in enumerate(edges):
        result = {k: np.asarray(v).copy() for k, v in e.items()}
        result["reward"] = credit.predict_action(e["source"], a)[:, 0]
        if geometry is not None:
            result["source"] = geometry.transform(e["source"])
            result["successor"] = geometry.transform(e["successor"])
        # O término continua sendo a transição observada, junto do sucessor.
        output.append(result)
    return output
