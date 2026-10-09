"""Reconhecer transições e aprender um campo por diferenças observadas.

Família numérica limitada. Reconhecimento após ação não é previsão antes
dela; diferença de potencial é hipótese, não propriedade universal de recompensa.
"""
import json
from pathlib import Path
import numpy as np

MODES = ("before", "after", "delta", "transition", "potential", "action_prior", "zero")


def design(before, after, mode):
    before = np.asarray(before, float)
    if before.ndim != 2 or before.shape[1] < 1 or not np.isfinite(before).all() or mode not in MODES or not np.all(before[:, 0] == 1):
        raise ValueError("contrato de leitura inválido")
    n, d = before.shape
    if mode in ("after", "delta", "transition", "potential"):
        after = np.asarray(after, float)
        if after.shape != before.shape or not np.isfinite(after).all() or not np.all(after[:, 0] == 1):
            raise ValueError("reconhecimento exige observação posterior alinhada")
    zeros = np.zeros((n, d))
    if mode == "before":
        return np.column_stack((before, zeros))
    if mode == "after":
        return np.column_stack((after, zeros))
    if mode == "delta":
        return np.column_stack((np.ones(n), after[:, 1:] - before[:, 1:], zeros))
    if mode == "potential":
        return np.column_stack((after - before, zeros))
    if mode == "transition":
        return np.column_stack((before, after))
    if mode == "action_prior":
        return np.column_stack((np.ones(n), np.zeros((n, 2 * d - 1))))
    return np.zeros((n, 2 * d))


class EventReader:
    def __init__(self, mode, actions=4, alpha=.05):
        if mode not in MODES or actions < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("modo/orçamento numérico inválido")
        self.mode, self.actions, self.alpha = mode, int(actions), float(alpha)

    def fit(self, before, after, actions, target):
        x = design(before, after, self.mode)
        actions, target = np.asarray(actions), np.asarray(target, float)
        if len(x) == 0 or actions.shape != target.shape or target.shape != (len(x),) or not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= self.actions) or not np.isfinite(target).all():
            raise ValueError("experiência/ação inválida")
        self.width = x.shape[1] // 2
        self.coefficient = np.zeros((self.actions, x.shape[1]))
        if self.mode == "potential":
            delta = x[:, :self.width]
            self.field = np.linalg.solve(delta.T @ delta + self.alpha * np.eye(self.width), delta.T @ target)
            self.coefficient[:, :self.width] = self.field
        elif self.mode != "zero":
            for action in range(self.actions):
                selected = actions == action
                if not selected.any():
                    raise ValueError("ação sem experiência")
                xx = x[selected]
                self.coefficient[action] = np.linalg.solve(xx.T @ xx + self.alpha * np.eye(x.shape[1]), xx.T @ target[selected])
        return self

    def predict(self, before, after, actions):
        x = design(before, after, self.mode)
        actions = np.asarray(actions)
        if x.shape[1] != self.coefficient.shape[1] or actions.shape != (len(x),) or not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= self.actions):
            raise ValueError("entrada incompatível")
        return np.einsum("ij,ij->i", x, self.coefficient[actions])

    def state_value(self, phi):
        if self.mode != "potential":
            raise ValueError("este leitor não define campo de estado")
        phi = np.asarray(phi, float)
        if phi.ndim != 2 or phi.shape[1] != self.width or not np.isfinite(phi).all():
            raise ValueError("estado incompatível")
        return phi @ self.field

    def save(self, path, feature_reference, metadata):
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise RuntimeError("corpo existe; preservar")
        np.savez_compressed(path, coefficient=self.coefficient, mode=self.mode, actions=self.actions, alpha=self.alpha,
            width=self.width, field=getattr(self, "field", np.empty(0)))
        definition = {"mode": self.mode, "feature_reference": feature_reference,
            "timing": "before_action" if self.mode in ("before", "action_prior", "zero") else "after_action_observation_required",
            "body": "F(next)-F(current),F(phi)=phi@field" if self.mode == "potential" else "design(phi_before,phi_after,mode)@coefficient[action]",
            "parameters": path.name, "metadata": metadata, "not_human_level_or_exact_proof": True}
        with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(definition, stream, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            reader = cls(str(data["mode"]), int(data["actions"]), float(data["alpha"]))
            reader.width, reader.coefficient = int(data["width"]), data["coefficient"].copy()
            if reader.mode == "potential":
                reader.field = data["field"].copy()
            return reader


class PotentialPlanner:
    """Transporte existente vivo + transporte aprendido para estado absorvente.

O estado previsto é [alive_phi,dead_phi]. A parte morta mantém sua última
observação. Valores de retorno são F(final_predicted)-F(initial_observed).
Não usa imagens futuras como entrada durante a previsão.
"""
    def __init__(self, body, field, terminal_transport):
        self.body, self.field = body, np.asarray(field, float)
        self.terminal_transport = np.asarray(terminal_transport, float)
        d = len(body.initial_prior)
        if self.field.shape != (d,) or self.terminal_transport.shape != (body.actions, d, d) or not np.isfinite(self.field).all() or not np.isfinite(self.terminal_transport).all():
            raise ValueError("campo/transporte incompatível")

    @classmethod
    def fit(cls, body, field, before_phi, after_phi, actions, continuation):
        terminal = []
        for action in range(body.actions):
            ids = actions == action
            if not ids.any():
                raise ValueError("ação sem experiência para transporte terminal")
            x = before_phi[ids]
            target = (1 - continuation[ids, None]) * after_phi[ids]
            terminal.append(np.linalg.solve(x.T @ x + body.alpha * np.eye(x.shape[1]), x.T @ target))
        return cls(body, field, np.asarray(terminal))

    def value(self, phi, word):
        self.body.word_value(word)  # Validate the same finite action language.
        alive, dead = np.asarray(phi, float).copy(), np.zeros_like(self.field)
        if alive.shape != self.field.shape or not np.isfinite(alive).all():
            raise ValueError("estado inicial incompatível")
        initial = float(alive @ self.field)
        for action in word:
            dead += alive @ self.terminal_transport[action]
            alive = alive @ self.body.transport[action]
        return float((alive + dead) @ self.field) - initial

    def comparison(self, history, left, right):
        phi = self.body.features(history)
        if len(phi) != len(left) or len(phi) != len(right):
            raise ValueError("planos incompatíveis")
        return np.asarray([self.value(z, a) - self.value(z, b) for z, a, b in zip(phi, left, right)])


def potential_constraints(node_before, node_after, reward):
    """Compatibilidade exata finita de F(v)-F(u)=reward, via união ponderada.

Offsets por componente são livres. Consistência não prova aprendizagem,
representação suficiente em outros dados ou conservatividade universal.
"""
    if len(node_before) != len(node_after) or len(node_before) != len(reward):
        raise ValueError("restrições desalinhadas")
    reward = np.asarray(reward, float)
    if not np.isfinite(reward).all() or not np.array_equal(reward, np.rint(reward)):
        raise ValueError("verificador exato desta bancada aceita apenas recompensas inteiras")
    parent, weight, rank = {}, {}, {}
    def find(node):
        if node not in parent:
            parent[node], weight[node], rank[node] = node, 0, 0
        if parent[node] != node:
            root, offset = find(parent[node])
            weight[node] += offset
            parent[node] = root
        return parent[node], weight[node]
    conflicts = []
    for i, (u, v, r) in enumerate(zip(node_before, node_after, reward)):
        r = int(r)
        ru, du = find(u)
        rv, dv = find(v)
        if ru == rv:
            residual = dv - du - r
            if residual != 0:
                conflicts.append({"edge": i, "constraint_residual": residual})
        else:
            if rank[ru] < rank[rv]:
                parent[ru], weight[ru] = rv, dv - du - r
            else:
                parent[rv], weight[rv] = ru, r + du - dv
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
    roots = {find(node)[0] for node in parent}
    return {"observed_nodes": len(parent), "connected_components": len(roots),
        "conflict_count": len(conflicts), "first_conflicts": conflicts[:8],
        "compatible_on_observed_graph": not conflicts, "verification_scope": "exact_integer_constraints_on_observed_history_graph",
        "universal_conservativity_proven": False}
