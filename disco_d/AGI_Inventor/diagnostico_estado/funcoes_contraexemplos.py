"""Funções não lineares de pares de experiências; linguagem pequena, executável.

Tanh de uma comparação afim e produto de duas comparações. A família foi
escolhida por nós; os parâmetros e pares vêm dos dados. Não é invenção
irrestrita de representações nem técnica inédita demonstrada.
"""

import json
from pathlib import Path
import numpy as np
from transporte_funcoes import FunctionTransport, distances

ATOMS, PRODUCTS, SLOTS = 24, 8, 4


class PairFunctions:
    def __init__(self, directions, offsets, expressions, origins):
        self.directions, self.offsets = np.asarray(directions, float), np.asarray(offsets, float)
        self.expressions, self.origins = expressions, origins
        if self.directions.ndim != 2 or self.offsets.shape != (len(self.directions),) or not np.isfinite(self.directions).all() or not np.isfinite(self.offsets).all():
            raise ValueError("corpo numérico inválido")
        for expression in expressions:
            refs = expression["units"]
            if expression["op"] not in ("unit", "product") or len(refs) != (1 if expression["op"] == "unit" else 2) or any(type(i) is not int or not 0 <= i < len(self.directions) for i in refs):
                raise ValueError("árvore fora da linguagem")

    def apply(self, history, selected=None):
        h = np.asarray(history, float)
        if h.ndim < 2 or not np.isfinite(h).all():
            raise ValueError("história inválida")
        x = h.reshape(len(h), -1)
        if x.shape[1] != self.directions.shape[1]:
            raise ValueError("forma sensorial incompatível")
        units = np.tanh(x @ self.directions.T - self.offsets)
        indices = list(range(len(self.expressions))) if selected is None else selected
        output = []
        for index in indices:
            if index is None:
                output.append(np.zeros(len(h)))
                continue
            if not isinstance(index, int) or not 0 <= index < len(self.expressions):
                raise ValueError("função inexistente")
            refs = self.expressions[index]["units"]
            output.append(np.prod(units[:, refs], axis=1))
        return np.column_stack(output) if output else np.empty((len(h), 0))

    def save(self, path):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("funções já existem; preservar")
        np.savez_compressed(path, directions=self.directions, offsets=self.offsets)
        definition = {"body": "unit=tanh(flatten(history)@direction-offset);product=unit_i*unit_j",
                      "expressions": self.expressions, "origins": self.origins, "parameters": path.name,
                      "scope": "data_parameterized_small_function_language_not_unrestricted_invention"}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(definition, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        definition = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            return cls(data["directions"], data["offsets"], definition["expressions"], definition["origins"])


def crossfit_reward(phi, actions, reward, episodes, alpha=.05):
    prediction = np.empty(len(phi))
    for episode in sorted(set(episodes)):
        for action in sorted(set(actions)):
            train = (episodes != episode) & (actions == action)
            held = (episodes == episode) & (actions == action)
            if held.any() and not train.any():
                raise ValueError("pergunta sem outro episódio de construção")
            x = phi[train]
            coefficient = np.linalg.solve(x.T @ x + alpha * np.eye(phi.shape[1]), x.T @ reward[train])
            prediction[held] = phi[held] @ coefficient
    return prediction


def propose(source, actions, reward, episodes, phi, arm="counterexample_pairs"):
    if arm not in ("counterexample_pairs", "shuffled_outcomes", "random_pairs", "random_directions"):
        raise ValueError("gerador inválido")
    source, actions, reward, episodes = map(np.asarray, (source, actions, reward, episodes))
    raw = source.reshape(len(source), -1).astype(float)
    y = reward.astype(float).copy()
    if not all(np.isfinite(v).all() for v in (raw, y, phi)) or len(raw) != len(actions) or len(y) != len(episodes):
        raise ValueError("experiência inválida")
    rng = np.random.default_rng(9881)
    if arm == "shuffled_outcomes":
        for action in sorted(set(actions)):
            ids = np.flatnonzero(actions == action)
            y[ids] = y[rng.permutation(ids)]
    forecast = crossfit_reward(phi, actions, y, episodes)
    pairs = []
    raw_conflicts = 0
    for action in sorted(set(actions)):
        ids = np.flatnonzero(actions == action)
        ds = distances(phi[ids], phi[ids])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = int(ids[a]), int(ids[b])
                if episodes[i] == episodes[j]:
                    continue
                if np.array_equal(raw[i], raw[j]):
                    raw_conflicts += int(abs(y[i] - y[j]) > 1e-12)
                    continue
                if arm in ("counterexample_pairs", "shuffled_outcomes"):
                    if abs(y[i] - y[j]) <= 1e-12:
                        continue
                    score = abs((y[i] - y[j]) - (forecast[i] - forecast[j])) / (1 + float(ds[a, b]))
                else:
                    score = float(rng.random())
                pairs.append((score, i, j))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    directions, offsets, origins = [], [], []
    if arm == "random_directions":
        center = raw.mean(0)
        for i in range(ATOMS):
            direction = rng.normal(size=raw.shape[1])
            scale = float(np.std((raw - center) @ direction))
            if scale < 1e-12:
                continue
            direction /= scale
            directions.append(direction)
            offsets.append(float(center @ direction))
            origins.append({"kind": "random_direction", "index": i})
    else:
        # The bounded language has no handcrafted palette, lag, object, or region.
        for score, i, j in pairs:
            if arm in ("counterexample_pairs", "shuffled_outcomes") and y[i] < y[j]:
                i, j = j, i
            difference = raw[i] - raw[j]
            direction = 4 * difference / float(difference @ difference)
            offset = float((raw[i] + raw[j]) @ direction / 2)
            if any(np.allclose(direction, old, rtol=1e-8, atol=1e-12) and np.isclose(offset, old_offset, rtol=1e-8, atol=1e-12)
                   for old, old_offset in zip(directions, offsets)):
                continue
            directions.append(direction)
            offsets.append(offset)
            origins.append({"kind": "observed_pair", "rows": [i, j], "episodes": [int(episodes[i]), int(episodes[j])],
                            "action": int(actions[i]), "pair_outcomes_recorded": [float(y[i]), float(y[j])],
                            "outcome_driven_pairing": arm in ("counterexample_pairs", "shuffled_outcomes"),
                            "selection_residual_score": float(score)})
            if len(directions) == ATOMS:
                break
    directions = np.asarray(directions, float).reshape(-1, raw.shape[1])
    expressions = [{"op": "unit", "units": [i]} for i in range(len(directions))]
    # Products are proposed before either parent must earn admission.
    combinations = [(i, j) for i in range(min(6, len(directions))) for j in range(i + 1, min(6, len(directions)))]
    for i, j in combinations[:PRODUCTS]:
        expressions.append({"op": "product", "units": [i, j]})
    return PairFunctions(directions, offsets, expressions, origins), {"candidate_pairs": len(pairs),
        "atoms": len(directions), "products": len(expressions) - len(directions), "candidate_functions": len(expressions),
        "generation_episodes": sorted(map(int, set(episodes))), "validation_targets_used": 0,
        "raw_collisions_with_different_outcomes": raw_conflicts}


class GeneratedTransport(FunctionTransport):
    def __init__(self, base, functions, selected):
        super().__init__(base.actions, base.supports_limit, base.alpha)
        self.support, self.scale = base.support.copy(), base.scale
        self.functions, self.selected = functions, list(selected) + [None] * (SLOTS - len(selected))
        if len(self.selected) != SLOTS:
            raise ValueError("orçamento de funções excedido")
        self.mode = "transport"

    def features(self, history):
        return np.column_stack((super().features(history), self.functions.apply(history, self.selected)))

    def save(self, path):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo já existe; preservar")
        np.savez_compressed(path, support=self.support, scale=self.scale, reward=self.reward, transport=self.transport,
                            initial_prior=self.initial_prior, directions=self.functions.directions, offsets=self.functions.offsets,
                            actions=self.actions, alpha=self.alpha, supports_limit=self.supports_limit)
        definition = {"operator": "transport_with_data_generated_nonlinear_functions", "selected": self.selected, "mode": self.mode,
                      "expressions": self.functions.expressions, "origins": self.functions.origins,
                      "feature_body": "concat(base_observed_RBF, selected_affine_tanh_or_products, zero_unused_slots)",
                      "composition": "v_empty=0;v_aQ=r_a+M_a@v_Q;value=phi(history)@v_word",
                      "parameters": path.name, "scope": "bounded_numeric_function_generation_not_human_mind"}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(definition, stream, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        definition = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            base = FunctionTransport(int(data["actions"]), int(data["supports_limit"]), float(data["alpha"]))
            base.support, base.scale = data["support"].copy(), float(data["scale"])
            functions = PairFunctions(data["directions"], data["offsets"], definition["expressions"], definition["origins"])
            selected = [i for i in definition["selected"] if i is not None]
            body = cls(base, functions, selected)
            body.mode = definition["mode"]
            for name in ("reward", "transport", "initial_prior"):
                setattr(body, name, data[name].copy())
        return body


def fit_features(base, functions, selected, phi, successor_phi, reward, continuation, actions):
    body = GeneratedTransport(base, functions, selected)
    body.initial_prior = phi.mean(0)
    body.reward, body.transport = [], []
    for action in range(base.actions):
        ids = actions == action
        if not ids.any():
            raise ValueError("ação não observada")
        x = phi[ids]
        target = np.column_stack((reward[ids], continuation[ids, None] * successor_phi[ids]))
        coefficient = np.linalg.solve(x.T @ x + base.alpha * np.eye(x.shape[1]), x.T @ target)
        body.reward.append(coefficient[:, 0])
        body.transport.append(coefficient[:, 1:])
    body.reward, body.transport = np.asarray(body.reward), np.asarray(body.transport)
    return body


def append_slots(base_phi, all_functions, selected):
    chosen = all_functions[:, selected] if selected else np.empty((len(base_phi), 0))
    return np.column_stack((base_phi, chosen, np.zeros((len(base_phi), SLOTS - len(selected)))))
