"""Ações como operadores sobre funções observáveis, com corpo exportável.

Família finita de funções RBF gerada das observações; não inventa conceitos
universais. Composição algébrica é um contrato, acurácia é uma hipótese.
"""

import json
from pathlib import Path
import numpy as np


def distances(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0)


class FunctionTransport:
    def __init__(self, actions=4, supports=32, alpha=.05):
        if actions < 1 or supports < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("linguagem/regularização inválida")
        self.actions, self.supports_limit, self.alpha = actions, supports, float(alpha)

    def features(self, history):
        history = np.asarray(history, float)
        if history.ndim < 2 or not np.isfinite(history).all():
            raise ValueError("observações inválidas")
        x = history.reshape(len(history), -1)
        if x.shape[1] != self.support.shape[1]:
            raise ValueError("dimensão sensorial incompatível")
        return np.column_stack((np.ones(len(x)), np.exp(-distances(x, self.support) / self.scale)))

    def fit(self, source, successor, actions, reward, continuation, shuffled=False):
        source, successor = np.asarray(source, float), np.asarray(successor, float)
        actions, reward, continuation = np.asarray(actions), np.asarray(reward, float), np.asarray(continuation, float)
        n = len(source)
        if (source.shape != successor.shape or source.ndim < 2 or n == 0 or
            actions.shape != reward.shape or reward.shape != continuation.shape or reward.shape != (n,) or
            not all(np.isfinite(v).all() for v in (source, successor, reward, continuation)) or
            not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= self.actions) or
            not np.isin(continuation, [0, 1]).all()):
            raise ValueError("transições inválidas")
        # Dictionary proposal uses actual training observations only.
        raw = np.concatenate((source.reshape(n, -1), successor.reshape(n, -1)))
        unique = np.unique(raw, axis=0)
        middle = unique.mean(0, keepdims=True)
        ids = [int(distances(unique, middle).argmin())]
        nearest = distances(unique, unique[ids]).ravel()
        for _ in range(1, min(self.supports_limit, len(unique))):
            candidate = int(nearest.argmax())
            if nearest[candidate] <= 1e-14:
                break
            ids.append(candidate)
            nearest = np.minimum(nearest, distances(unique, unique[candidate:candidate + 1]).ravel())
        self.support = unique[ids].copy()
        dd = distances(self.support, self.support)
        positive = dd[np.triu_indices(len(dd), 1)]
        positive = positive[positive > 1e-14]
        self.scale = float(np.median(positive)) if len(positive) else 1.
        phi, next_phi = self.features(source), self.features(successor)
        self.initial_prior = phi.mean(0)
        self.reward, self.transport = [], []
        self.action_counts, self.local_residual_mse = [], []
        rng = np.random.default_rng(9861)
        for action in range(self.actions):
            selected = np.flatnonzero(actions == action)
            if len(selected) == 0:
                raise ValueError("ação sem experiência: não inventar um operador")
            x = phi[selected]
            # Terminal transitions deliver reward but no continuation function.
            target = np.column_stack((reward[selected], continuation[selected, None] * next_phi[selected]))
            if shuffled:
                target = target[rng.permutation(len(selected))]
            coefficients = np.linalg.solve(x.T @ x + self.alpha * np.eye(x.shape[1]), x.T @ target)
            self.reward.append(coefficients[:, 0])
            self.transport.append(coefficients[:, 1:])
            self.action_counts.append(len(selected))
            self.local_residual_mse.append(float(np.mean((x @ coefficients - target) ** 2)))
        self.reward, self.transport = np.asarray(self.reward), np.asarray(self.transport)
        self.mode = "transport"
        return self

    def word_value(self, word):
        if len(word) > 16 or any(not isinstance(a, (int, np.integer)) or not 0 <= a < self.actions for a in word):
            raise ValueError("programa fora da linguagem")
        value = np.zeros(len(self.initial_prior))
        for action in reversed(word):
            value = self.reward[action] + (value if self.mode == "no_transport" else self.transport[action] @ value)
        if not np.isfinite(value).all():
            raise ValueError("composição não finita")
        return value

    def predict(self, history, words):
        phi = self.features(history)
        if len(phi) != len(words):
            raise ValueError("palavras/observações incompatíveis")
        if self.mode == "no_context":
            phi = np.broadcast_to(self.initial_prior, phi.shape)
        values = np.asarray([self.word_value(word) for word in words])
        return np.einsum("ij,ij->i", phi, values)

    def comparison(self, history, left, right):
        return self.predict(history, left) - self.predict(history, right)

    def augmented(self, word):
        """Row [accumulated_reward,phi] executes a word via matrix products."""
        self.word_value(word)  # Validate language before composition.
        width = len(self.initial_prior)
        product = np.eye(width + 1)
        for action in word:
            operator = np.zeros_like(product)
            operator[0, 0] = 1
            operator[1:, 0] = self.reward[action]
            operator[1:, 1:] = np.eye(width) if self.mode == "no_transport" else self.transport[action]
            product = product @ operator
        return product

    def save(self, path):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo já existe; preservar")
        np.savez_compressed(path, support=self.support, scale=self.scale, reward=self.reward,
                            transport=self.transport, initial_prior=self.initial_prior, mode=self.mode,
                            actions=self.actions, alpha=self.alpha, supports_limit=self.supports_limit,
                            action_counts=np.asarray(self.action_counts), local_residual_mse=self.local_residual_mse)
        definition = {"operator": "learned_observable_function_pullback", "mode": self.mode,
            "features": "[1,exp(-squared_distance(history,observed_training_support)/scale)]",
            "composition": "v_empty=0;v_aQ=r_a+M_a@v_Q;value=phi(history)@v_word",
            "parameters": path.name, "language": {"anonymous_actions": self.actions, "maximum_blocks": 16},
            "training_target": "[observed_reward,observed_continuation*phi(observed_successor)]",
            "does_not_guarantee": ["physical_execution_consistency", "correct_decision", "unrestricted_variable_invention", "human_intelligence"]}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(definition, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(int(data["actions"]), int(data["supports_limit"]), float(data["alpha"]))
            for name in ("support", "reward", "transport", "initial_prior", "action_counts", "local_residual_mse"):
                setattr(obj, name, data[name].copy())
            obj.scale, obj.mode = float(data["scale"]), str(data["mode"])
        return obj
