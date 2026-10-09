"""Planejamento em bancos de experiências, sem tipos de objeto ou jogo.

Cada aresta conserva estado, sucessor, recompensa e término juntos. Planos
são palavras de índices de ação. A busca guarda futuros alternativos reais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from futuros_distribuidos import weights


@dataclass
class Node:
    plan: tuple[int, ...]
    value: float
    payload: np.ndarray
    alive: float = 1.0


class ExperiencePlanner:
    def __init__(self, edges, neighbors=8, alteration=None):
        if not edges or neighbors < 1:
            raise ValueError("banco e número de vizinhos devem ser positivos")
        self.neighbors = neighbors
        self.edges = []
        rng = np.random.default_rng(9753)
        for action in range(len(edges)):
            e = edges[(action + 1) % len(edges) if alteration == "action" else action]
            copied = {k: np.asarray(e[k], float).copy() for k in ("source", "successor", "reward", "continuation")}
            source = copied["source"]
            n = len(source)
            if (source.ndim != 2 or not n or copied["successor"].shape != source.shape
                    or copied["reward"].shape != (n,) or copied["continuation"].shape != (n,)):
                raise ValueError("aresta com forma inconsistente")
            if not all(np.isfinite(v).all() for v in copied.values()):
                raise ValueError("aresta não finita")
            if np.any((copied["continuation"] < 0) | (copied["continuation"] > 1)):
                raise ValueError("continuação fora de [0,1]")
            if self.edges and source.shape[1] != self.edges[0]["source"].shape[1]:
                raise ValueError("ações usam dimensões de estado diferentes")
            if alteration == "successor":
                permutation = rng.permutation(n)
                copied["successor"] = copied["successor"][permutation]
                copied["continuation"] = copied["continuation"][permutation]
            elif alteration == "reward":
                copied["reward"] = copied["reward"][rng.permutation(n)]
            self.edges.append(copied)
        self.transition_cache = {}

    def transition(self, first, second):
        key = (first, second)
        if key not in self.transition_cache:
            self.transition_cache[key] = csr_matrix(weights(
                self.edges[first]["successor"], self.edges[second]["source"], self.neighbors))
        return self.transition_cache[key]

    def validate_plan(self, plan):
        if not plan or any(type(a) not in (int, np.int64, np.int32) or not 0 <= a < len(self.edges) for a in plan):
            raise ValueError("plano vazio ou ação inválida")

    def plan_values(self, root, plans):
        values = []
        for plan in plans:
            self.validate_plan(plan)
            tail = self.edges[plan[-1]]["reward"].copy()
            for first, second in zip(plan[-2::-1], plan[:0:-1]):
                e = self.edges[first]
                tail = e["reward"] + e["continuation"] * self.transition(first, second).dot(tail)
            probability = weights(root, self.edges[plan[0]]["source"], self.neighbors)
            values.append(probability @ tail)
        return np.stack(values, axis=1)

    def search(self, root, horizon=16, width=8, collapse=False):
        root = np.asarray(root, float)
        if root.ndim != 2 or not np.isfinite(root).all() or horizon < 1 or width < 1:
            raise ValueError("raiz, horizonte ou largura inválidos")
        n = len(root)
        beams = [[] for _ in range(n)]
        expansions = np.zeros(n, int)
        for action, e in enumerate(self.edges):
            probability = weights(root, e["source"], self.neighbors)
            returns = probability @ e["reward"]
            if collapse:
                after = probability @ e["successor"]
                alive = probability @ e["continuation"]
            else:
                after = probability * e["continuation"]
                alive = after.sum(1)
            for i in range(n):
                beams[i].append(Node((action,), float(returns[i]), after[i].copy(), float(alive[i])))
                expansions[i] += 1
        beams = [sorted(b, key=lambda v: (-v.value, v.plan))[:width] for b in beams]
        for _ in range(1, horizon):
            candidates = [[] for _ in range(n)]
            groups = {}
            for i, beam in enumerate(beams):
                for node in beam:
                    groups.setdefault(node.plan[-1], []).append((i, node))
            for last, group in groups.items():
                payload = np.stack([v.payload for _, v in group])
                old_value = np.asarray([v.value for _, v in group])
                old_alive = np.asarray([v.alive for _, v in group])
                for action, e in enumerate(self.edges):
                    if collapse:
                        probability = weights(payload, e["source"], self.neighbors)
                        returns = old_value + old_alive * (probability @ e["reward"])
                        after = probability @ e["successor"]
                        alive = old_alive * (probability @ e["continuation"])
                    else:
                        probability = self.transition(last, action).T.dot(payload.T).T
                        returns = old_value + probability @ e["reward"]
                        after = probability * e["continuation"]
                        alive = after.sum(1)
                    for j, (i, node) in enumerate(group):
                        candidates[i].append(Node(node.plan + (action,), float(returns[j]),
                                                  after[j].copy(), float(alive[j])))
                        expansions[i] += 1
            beams = [sorted(b, key=lambda v: (-v.value, v.plan))[:width] for b in candidates]
        traces = []
        for beam in beams:
            rows = []
            for node in beam:
                if collapse or node.alive < 1e-15:
                    entropy = None
                else:
                    probability = node.payload / node.alive
                    positive = probability > 0
                    entropy = float(-np.sum(probability[positive] * np.log(probability[positive])))
                rows.append({"plan": list(node.plan), "predicted_return": node.value,
                             "predicted_survival": node.alive, "state_entropy_nats": entropy})
            traces.append(rows)
        return {"plans": [list(b[0].plan) for b in beams], "values": [b[0].value for b in beams],
                "expansions_per_root": expansions.tolist(), "traces": traces,
                "scope": "predicted_hypotheses_not_observed_returns"}
