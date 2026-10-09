"""Variáveis de história geradas por resíduos observados fora do ajuste.

Projeção, normalização e dinâmica têm corpos completos exportáveis. Não é
uma arquitetura humana ou técnica científica inédita. Não executa Python
arbitrário nem consulta o futuro na função de leitura.
"""

from pathlib import Path
import json

import numpy as np

from avaliar import KernelRidge

MODES = ("state_only", "two_frames", "residual_history", "shuffled_residuals", "permuted_history", "pca_history")


def history_input(history, mode, ids=None):
    h = np.asarray(history, float)
    if h.ndim != 3 or not len(h) or not np.isfinite(h).all():
        raise ValueError("história deve ser amostras × tempo × observação finita")
    if mode == "two_frames":
        h = h[:, -2:]
    if mode == "permuted_history":
        if ids is None or len(ids) != len(h):
            raise ValueError("controle temporal exige identificadores de amostra")
        h = h.copy()
        for i, sample_id in enumerate(ids):
            rng = np.random.default_rng(np.random.SeedSequence([9793, int(sample_id)]))
            past = max(0, h.shape[1] - 2)
            h[i, :past] = h[i, rng.permutation(past)]
    return h.reshape(len(h), -1)


def distance_scale(x):
    ids = np.linspace(0, len(x) - 1, min(512, len(x))).astype(int)
    d = KernelRidge.distances(x[ids], x[ids])
    values = d[np.triu_indices(len(d), 1)]
    positive = values[values > 1e-14]
    return float(np.sqrt(np.median(positive))) if len(positive) else 1.0


class HistoryProjection:
    def fit(self, history, residual, actions, mode, ids=None, rank=32):
        if mode not in MODES or rank < 1:
            raise ValueError("modo ou rank inválido")
        self.mode, self.rank = mode, rank
        x = history_input(history, mode, ids)
        self.mean = x.mean(0)
        centered = x - self.mean
        if mode == "state_only":
            self.basis = np.zeros((x.shape[1], rank))
            return self
        if mode == "pca_history":
            rng = np.random.default_rng(9794)
            sketch = rng.normal(size=(x.shape[1], min(64, len(x), x.shape[1])))
            q, _ = np.linalg.qr(centered @ sketch)
            for _ in range(2):
                q, _ = np.linalg.qr(centered @ (centered.T @ q))
            _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
            basis = vt[:rank].T
        else:
            r = np.asarray(residual, float).copy()
            if r.ndim != 2 or len(r) != len(x) or not np.isfinite(r).all():
                raise ValueError("resíduos inválidos")
            if mode == "shuffled_residuals":
                rng = np.random.default_rng(9795)
                for action in sorted(set(actions)):
                    selected = np.flatnonzero(actions == action)
                    r[selected] = r[rng.permutation(selected)]
            r -= r.mean(0)
            r /= np.maximum(r.std(0), 1e-6)
            covariance = centered.T @ r / len(x)
            u, _, _ = np.linalg.svd(covariance, full_matrices=False)
            basis = u[:, :rank]
        self.basis = np.zeros((x.shape[1], rank))
        self.basis[:, :basis.shape[1]] = basis
        return self

    def transform(self, history, ids=None):
        return (history_input(history, self.mode, ids) - self.mean) @ self.basis


class MemoryDynamics:
    def fit(self, z, next_z, history, next_history, reward, continuation, actions,
            residual, mode, ids, next_ids):
        self.projection = HistoryProjection().fit(history, residual, actions, mode, ids)
        memory = self.projection.transform(history, ids)
        self.zmean = z.mean(0)
        self.scale_z, self.scale_memory = distance_scale(z), distance_scale(memory)
        source = self.state(z, history, ids)
        successor = self.state(next_z, next_history, next_ids)
        self.models, self.reward_ranges, self.counts = [], [], []
        self.action_count = int(actions.max()) + 1
        target = np.column_stack((successor - source, reward, continuation))
        for action in range(self.action_count):
            selected = actions == action
            if not selected.any():
                raise ValueError("ação sem experiências")
            self.models.append(KernelRidge().fit(source[selected], target[selected]))
            self.reward_ranges.append([float(reward[selected].min()), float(reward[selected].max())])
            self.counts.append(int(selected.sum()))
        return self

    def state(self, z, history, ids=None):
        memory = self.projection.transform(history, ids) / self.scale_memory
        return np.column_stack(((z - self.zmean) / self.scale_z, memory)) / np.sqrt(2)

    def predict(self, state, action):
        raw = self.models[action].predict(state)
        next_state = state + raw[:, :-2]
        reward = np.clip(raw[:, -2], *self.reward_ranges[action])
        continuation = np.clip(raw[:, -1], 0, 1)
        return next_state, reward, continuation

    def rollout(self, state, plans):
        state, plans = np.asarray(state, float).copy(), np.asarray(plans, int)
        if plans.ndim != 2 or len(state) != len(plans) or not plans.shape[1]:
            raise ValueError("estados e planos incompatíveis")
        if np.any(plans < 0) or np.any(plans >= self.action_count):
            raise ValueError("ação fora da linguagem")
        alive, total, returns, states = np.ones(len(state)), np.zeros(len(state)), [], []
        for t in range(plans.shape[1]):
            next_state = state.copy()
            for action in range(self.action_count):
                selected = plans[:, t] == action
                if selected.any():
                    successor, reward, continuation = self.predict(state[selected], action)
                    total[selected] += alive[selected] * reward
                    alive[selected] *= continuation
                    next_state[selected] = successor
            state = next_state
            states.append(state.copy())
            returns.append(total.copy())
        return {"returns": np.stack(returns, axis=1), "states": np.stack(states, axis=1), "survival": alive}

    def save(self, path):
        path = Path(path)
        data = {"projection_mean": self.projection.mean, "projection_basis": self.projection.basis,
                "mode": np.asarray(self.projection.mode), "rank": np.asarray(self.projection.rank),
                "zmean": self.zmean, "scale_z": np.asarray(self.scale_z),
                "scale_memory": np.asarray(self.scale_memory), "reward_ranges": np.asarray(self.reward_ranges),
                "counts": np.asarray(self.counts)}
        for action, model in enumerate(self.models):
            for name in ("mean", "x", "ym", "scale", "coeff"):
                data[f"{action}_{name}"] = np.asarray(getattr(model, name))
        np.savez_compressed(path, **data)
        definition = {"input": "state_vector_and_past_raw_history", "mode": self.projection.mode,
                      "memory_body": "(flatten(past_history)-projection_mean) @ projection_basis",
                      "state_body": "concat((z-zmean)/scale_z, memory/scale_memory)/sqrt(2)",
                      "transition_body": "state+RBF(state).delta; clipped_reward; clipped_continuation",
                      "parameters": path.name, "scope": "numeric_memory_from_experience_not_human_concepts"}
        path.with_suffix(".json").write_text(json.dumps(definition, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls()
        obj.projection = HistoryProjection()
        obj.projection.mean, obj.projection.basis = data["projection_mean"], data["projection_basis"]
        obj.projection.mode, obj.projection.rank = str(data["mode"]), int(data["rank"])
        obj.zmean = data["zmean"]
        obj.scale_z, obj.scale_memory = float(data["scale_z"]), float(data["scale_memory"])
        obj.reward_ranges, obj.counts = data["reward_ranges"].tolist(), data["counts"].tolist()
        obj.action_count = len(obj.counts)
        obj.models = []
        for action in range(obj.action_count):
            model = KernelRidge()
            for name in ("mean", "x", "ym", "scale", "coeff"):
                value = data[f"{action}_{name}"]
                setattr(model, name, float(value) if name == "scale" else value.copy())
            obj.models.append(model)
        return obj


def crossfit_residual(z, next_z, reward, continuation, actions, episodes):
    target = np.column_stack((next_z, reward, continuation))
    residual = np.empty_like(target)
    for action in sorted(set(actions)):
        for half in (0, 1):
            train = (actions == action) & (episodes % 2 != half)
            held = (actions == action) & (episodes % 2 == half)
            if not train.any() or not held.any():
                raise ValueError("ação sem episódios para ajuste separado")
            prediction = KernelRidge().fit(z[train], target[train]).predict(z[held])
            residual[held] = target[held] - prediction
    return residual
