"""Categorias conjuntas de consequência, extraídas do banco de experiências.

A categoria guarda sucessor observado, recompensa e continuação juntos.
Sua probabilidade é aprendida a partir do estado anterior. Não há rótulo de
objeto, regra do mundo ou execução de código arbitrário no corpo exportado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avaliar import KernelRidge
from planejador_distribuido import Node


def prototypes(edge, per_group=16):
    if not isinstance(per_group, int) or per_group < 1:
        raise ValueError("limite de categorias deve ser inteiro positivo")
    reward, continuation = np.asarray(edge["reward"]), np.asarray(edge["continuation"])
    target = np.asarray(edge["successor"], float)
    source = np.asarray(edge["source"], float)
    if (target.ndim != 2 or source.shape != target.shape or not len(target)
            or reward.shape != (len(target),) or continuation.shape != (len(target),)
            or not all(np.isfinite(v).all() for v in (target, source, reward, continuation))
            or not np.all((continuation == 0) | (continuation == 1))):
        raise ValueError("experiências inválidas ou término não binário")
    groups = sorted(set(zip(reward.tolist(), continuation.tolist())))
    labels = np.empty(len(target), int)
    chosen = []
    for r, c in groups:
        ids = np.flatnonzero((reward == r) & (continuation == c))
        y = target[ids]
        first = int(KernelRidge.distances(y, y.mean(0, keepdims=True)).argmin())
        selected = [first]
        distance = KernelRidge.distances(y, y[first:first + 1]).ravel()
        for _ in range(1, min(per_group, len(ids))):
            next_id = int(distance.argmax())
            if distance[next_id] <= 1e-14:
                break
            selected.append(next_id)
            distance = np.minimum(distance, KernelRidge.distances(y, y[next_id:next_id + 1]).ravel())
        assignment = KernelRidge.distances(y, y[selected]).argmin(1)
        labels[ids] = len(chosen) + assignment
        chosen.extend(ids[selected].tolist())
    return np.asarray(chosen), labels


class OutcomeModel:
    def fit(self, edges, per_group=16, shuffled=False, prior_only=False):
        if not edges:
            raise ValueError("nenhuma ação observada")
        self.models, self.outcomes, self.priors, self.origins = [], [], [], []
        self.prior_only = prior_only
        rng = np.random.default_rng(9781)
        for e in edges:
            ids, labels = prototypes(e, per_group)
            target = np.eye(len(ids))[labels]
            prior = target.mean(0)
            if shuffled:
                target = target[rng.permutation(len(target))]
            self.models.append(KernelRidge().fit(e["source"], target))
            self.outcomes.append({k: np.asarray(e[k], float)[ids].copy()
                                  for k in ("successor", "reward", "continuation")})
            self.priors.append(prior)
            self.origins.append(ids)
        return self

    def posterior(self, state, action):
        if self.prior_only:
            return np.repeat(self.priors[action][None], len(state), axis=0)
        raw = np.maximum(self.models[action].predict(state), 0)
        total = raw.sum(1, keepdims=True)
        missing = total[:, 0] < 1e-12
        raw[missing] = self.priors[action]
        total = raw.sum(1, keepdims=True)
        return raw / total

    def save(self, path):
        parameters = {"actions": np.asarray(len(self.models)), "prior_only": np.asarray(self.prior_only)}
        for a, model in enumerate(self.models):
            for name in ("mean", "x", "ym", "scale", "coeff"):
                parameters[f"{a}_{name}"] = np.asarray(getattr(model, name))
            for name, value in self.outcomes[a].items():
                parameters[f"{a}_outcome_{name}"] = value
            parameters[f"{a}_prior"] = self.priors[a]
            parameters[f"{a}_origin"] = self.origins[a]
        path = Path(path)
        np.savez_compressed(path, **parameters)
        definition = {"operator": "joint_outcome_posterior", "input_type": "finite_real_state_vector",
                      "output_type": "distribution_over_observed_successor_reward_continuation",
                      "body": "normalize(nonnegative(exp(-distance/scale) @ coeff + ym)); fallback=training_prior",
                      "parameters": path.name, "categories_by_action": [len(o["reward"]) for o in self.outcomes],
                      "scope": "learned_numeric_outcomes_not_universal_concepts"}
        path.with_suffix(".json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls()
        obj.models, obj.outcomes, obj.priors, obj.origins = [], [], [], []
        obj.prior_only = bool(data["prior_only"])
        for a in range(int(data["actions"])):
            model = KernelRidge()
            for name in ("mean", "x", "ym", "scale", "coeff"):
                value = data[f"{a}_{name}"]
                setattr(model, name, float(value) if name == "scale" else value.copy())
            obj.models.append(model)
            obj.outcomes.append({name: data[f"{a}_outcome_{name}"].copy()
                                 for name in ("successor", "reward", "continuation")})
            obj.priors.append(data[f"{a}_prior"].copy())
            obj.origins.append(data[f"{a}_origin"].copy())
        return obj


class OutcomePlanner:
    def __init__(self, model):
        self.model = model
        self.transitions = {}

    def transition(self, a, b):
        key = (a, b)
        if key not in self.transitions:
            self.transitions[key] = self.model.posterior(self.model.outcomes[a]["successor"], b)
        return self.transitions[key]

    def plan_values(self, roots, plans):
        columns = []
        for plan in plans:
            if not plan or any(not 0 <= a < len(self.model.models) for a in plan):
                raise ValueError("plano vazio ou ação inválida")
            value = self.model.outcomes[plan[-1]]["reward"].copy()
            for a, b in zip(plan[-2::-1], plan[:0:-1]):
                o = self.model.outcomes[a]
                value = o["reward"] + o["continuation"] * (self.transition(a, b) @ value)
            columns.append(self.model.posterior(roots, plan[0]) @ value)
        return np.stack(columns, axis=1)

    def search(self, roots, horizon=16, width=8):
        if (not isinstance(horizon, int) or horizon < 1
                or not isinstance(width, int) or width < 1):
            raise ValueError("horizonte e largura devem ser inteiros positivos")
        roots = np.asarray(roots, float)
        if roots.ndim != 2 or not len(roots) or not np.isfinite(roots).all():
            raise ValueError("estados de entrada inválidos")
        beams = [[] for _ in roots]
        expansions = np.zeros(len(roots), int)
        for a, outcome in enumerate(self.model.outcomes):
            probability = self.model.posterior(roots, a)
            value = probability @ outcome["reward"]
            after = probability * outcome["continuation"]
            for i in range(len(roots)):
                beams[i].append(Node((a,), float(value[i]), after[i].copy(), float(after[i].sum())))
                expansions[i] += 1
        beams = [sorted(b, key=lambda node: (-node.value, node.plan))[:width] for b in beams]
        for _ in range(1, horizon):
            grouped = {}
            for i, beam in enumerate(beams):
                for node in beam:
                    grouped.setdefault(node.plan[-1], []).append((i, node))
            candidates = [[] for _ in roots]
            for a, nodes in grouped.items():
                mass = np.stack([node.payload for _, node in nodes])
                previous = np.asarray([node.value for _, node in nodes])
                for b, outcome in enumerate(self.model.outcomes):
                    next_mass = mass @ self.transition(a, b)
                    value = previous + next_mass @ outcome["reward"]
                    after = next_mass * outcome["continuation"]
                    for j, (i, node) in enumerate(nodes):
                        candidates[i].append(Node(node.plan + (b,), float(value[j]), after[j].copy(), float(after[j].sum())))
                        expansions[i] += 1
            beams = [sorted(b, key=lambda node: (-node.value, node.plan))[:width] for b in candidates]
        return {"plans": [list(beam[0].plan) for beam in beams], "values": [beam[0].value for beam in beams],
                "expansions_per_root": expansions.tolist(),
                "traces": [[{"plan": list(node.plan), "predicted_return": node.value,
                             "predicted_survival": node.alive} for node in beam] for beam in beams],
                "scope": "predicted_joint_outcomes_not_observed_returns"}
