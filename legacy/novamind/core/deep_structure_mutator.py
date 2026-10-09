"""
DeepStructureMutator — Mutacoes estruturais PROFUNDAS que inventam novos caminhos causais.

Diferenca do StructureMutator antigo:
  - Nao apenas adiciona variaveis, mas REESCREVE caminhos causais
  - Pode inverter causalidade (A->B vira B->A)
  - Pode inserir mediadores (A->B vira A->X->B)
  - Pode fundir ou split variaveis
  - Pode gerar bifurcacoes

Cada mutacao e uma nova HIPOTESE CAUSAL, nao apenas um ajuste.
"""
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import copy
import random


@dataclass
class DeepMutation:
    """Mutacao estrutural profunda."""
    operation: str
    description: str
    variables_affected: List[str] = None
    new_variables: List[str] = None

    def to_dict(self) -> dict:
        return {
            "op": self.operation,
            "desc": self.description[:80],
            "vars": self.variables_affected or [],
            "new_vars": self.new_variables or [],
        }


class DeepStructureMutator:
    """Gera mutacoes estruturais profundas em modelos ODE.

    Nao apenas adiciona parametros. INVENTA novas estruturas causais.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.history: List[DeepMutation] = []

        # Banco de novas variaveis que podem ser inseridas
        self._var_pool = {
            "Alzheimer": [
                "glymphatic_clearance", "autophagy", "complement_activation",
                "calcium_dysregulation", "epigenetic_age", "neural_network_disruption",
                "circadian_disruption", "vascular_dysfunction", "cortical_spreading_depression",
                "olfactory_bulb_atrophy", "locus_coeruleus_degeneration",
                "noradrenaline_depletion", "sleep_fragmentation",
            ],
        }

    def mutate(self, model, n_mutations: int = 3) -> list:
        """Gera N mutacoes. Cada uma pode INVENTAR um novo mecanismo."""
        results = []

        for _ in range(n_mutations):
            op = self.rng.choice([
                "insert_mediator",      # A->B vira A->X->B
                "reverse_causality",    # A->B vira B->A
                "bifurcate_pathway",    # A->B vira A->B e A->C
                "add_positive_feedback", # A->B vira A->B e B->A+
                "merge_variables",      # A e B viram AB
                "add_nonlinear_term",   # Adiciona termo logistico/sigmoidal
                "add_delay",            # Adiciona atraso temporal
            ])

            mutation = self._apply(model, op)
            if mutation:
                results.append(mutation)

        return results

    def _apply(self, model, operation: str):
        """Aplica uma mutacao ao modelo."""
        try:
            new = copy.deepcopy(model)
            disease = model.disease
            new.name = f"{model.name}_deep{self.rng.randint(100,999)}"
            new.parent_model = model.name

            if operation == "insert_mediator":
                # A->B vira A->M->B (insere mediador)
                if len(model.variables) >= 2:
                    a_idx = 0
                    b_idx = len(model.variables) - 1
                    a = model.variables[a_idx]
                    b = model.variables[b_idx]

                    # Escolhe uma nova variavel do banco
                    mediator = self.rng.choice(
                        self._var_pool.get(disease, ["latent_factor_X"])
                    )

                    # Adiciona mediador ao modelo
                    new.variables = list(model.variables) + [mediator]
                    new.initial_state = np.append(model.initial_state, [0.5])

                    ode_orig = model.ode_fn
                    m_idx = len(model.variables)

                    def ode_mediated(t, y, orig=ode_orig, ai=a_idx, bi=b_idx, mi=m_idx):
                        dy = list(orig(t, y[:len(y)-1]))
                        # Mediator: influenciado por A, influencia B
                        d_m = 0.01*y[mi] + 0.02*y[ai] - 0.01*y[mi]
                        # Ainda influencia direta (enfraquecida)
                        if ai < len(dy):
                            dy[ai] = dy[ai] * 0.7  # enfraquece A
                        # B agora influenciado por M
                        if bi < len(dy):
                            dy[bi] += 0.02*y[mi]  # M influencia B
                        dy.append(d_m)
                        return np.array(dy)

                    new.ode_fn = ode_mediated
                    desc = f"Inserted mediator '{mediator}' between {a} and {b}"
                    self.history.append(DeepMutation("insert_mediator", desc,
                                                     [a, b], [mediator]))
                    new.mutation_description = desc
                    return new

            elif operation == "reverse_causality":
                # Inverte causalidade entre duas variaveis principais
                if len(model.variables) >= 2:
                    v1, v2 = model.variables[0], model.variables[-1]
                    ode_orig = model.ode_fn

                    def ode_reversed(t, y, orig=ode_orig):
                        dy = orig(t, y)
                        # Troca as influencias
                        if len(dy) >= 2:
                            temp = dy[0]
                            dy[0] = dy[-1] * 0.5
                            dy[-1] = temp * 0.5
                        return dy

                    new.ode_fn = ode_reversed
                    desc = f"Reversed causality: {v1} <-> {v2}"
                    self.history.append(DeepMutation("reverse_causality", desc, [v1, v2]))
                    new.mutation_description = desc
                    return new

            elif operation == "bifurcate_pathway":
                # Adiciona um ramo paralelo
                if len(model.variables) >= 3:
                    source = model.variables[0]
                    branch_var = f"{source}_branch"
                    new.variables = list(model.variables) + [branch_var]
                    new.initial_state = np.append(model.initial_state, [0.3])

                    ode_orig = model.ode_fn
                    b_idx = len(model.variables)

                    def ode_branch(t, y, orig=ode_orig, bi=b_idx):
                        dy = list(orig(t, y[:len(y)-1]))
                        # Ramo paralelo influenciado pela mesma fonte
                        d_b = 0.01*y[bi] + 0.02*y[0] - 0.01*y[bi]
                        dy.append(d_b)
                        return np.array(dy)

                    new.ode_fn = ode_branch
                    desc = f"Bifurcated: {source} splits into parallel branch"
                    self.history.append(DeepMutation("bifurcate_pathway", desc,
                                                     [source], [branch_var]))
                    new.mutation_description = desc
                    return new

            elif operation == "add_positive_feedback":
                # Adiciona feedback positivo: B->A (fecha o ciclo)
                if len(model.variables) >= 2:
                    a_idx, b_idx = 0, len(model.variables) - 1
                    a, b = model.variables[a_idx], model.variables[b_idx]
                    strength = self.rng.uniform(0.02, 0.08)

                    ode_orig = model.ode_fn
                    def ode_feedback(t, y, orig=ode_orig, ai=a_idx, bi=b_idx, st=strength):
                        dy = orig(t, y)
                        if ai < len(dy):
                            dy[ai] += st * y[bi]
                        return dy

                    new.ode_fn = ode_feedback
                    desc = f"Positive feedback: {b} -> {a} (strength={strength:.3f})"
                    self.history.append(DeepMutation("add_positive_feedback", desc, [a, b]))
                    new.mutation_description = desc
                    return new

            elif operation == "add_nonlinear_term":
                # Adiciona termo sigmoidal: torna a dinamica nao-linear
                ode_orig = model.ode_fn
                threshold = self.rng.uniform(0.3, 0.7)
                steepness = self.rng.uniform(3.0, 10.0)

                def ode_sigmoid(t, y, orig=ode_orig, th=threshold, st=steepness):
                    dy = orig(t, y)
                    if len(dy) > 0:
                        # Termo sigmoidal: ativacao quando ultrapassa threshold
                        sig = 1.0 / (1.0 + np.exp(-st * (y[0] - th)))
                        dy[0] += 0.02 * sig
                    return dy

                new.ode_fn = ode_sigmoid
                desc = f"Nonlinear sigmoid: threshold={threshold:.2f}, steepness={steepness:.1f}"
                self.history.append(DeepMutation("add_nonlinear_term", desc, [model.variables[0]]))
                new.mutation_description = desc
                return new

            elif operation == "merge_variables":
                # Funde duas variaveis em uma (simplificacao)
                if len(model.variables) >= 4:
                    v1, v2 = model.variables[1], model.variables[2]
                    merged = f"{v1[:10]}_{v2[:10]}"

                    new.variables = [v for i, v in enumerate(model.variables)
                                    if v not in [v1, v2]] + [merged]
                    # Estado inicial: media
                    i1, i2 = model.variables.index(v1), model.variables.index(v2)
                    merged_init = (model.initial_state[i1] + model.initial_state[i2]) / 2
                    new_states = [v for i, v in enumerate(model.initial_state)
                                 if i not in [i1, i2]]
                    new.initial_state = np.append(new_states, [merged_init])

                    # ODE simplificada (usa a original mas ignora variaveis fundidas)
                    ode_orig = model.ode_fn
                    def ode_merged(t, y, orig=ode_orig):
                        # Expande y para ter o tamanho original
                        y_full = np.zeros(len(model.variables))
                        idx = 0
                        for i in range(len(model.variables)):
                            if i == i1 or i == i2:
                                y_full[i] = y[-1]  # valor fundido
                            else:
                                y_full[i] = y[idx]
                                idx += 1
                        dy_full = orig(t, y_full)
                        # Comprime de volta
                        dy = []
                        for i in range(len(model.variables)):
                            if i != i1 and i != i2:
                                dy.append(dy_full[i])
                        dy.append((dy_full[i1] + dy_full[i2]) / 2)
                        return np.array(dy)

                    new.ode_fn = ode_merged
                    desc = f"Merged {v1} and {v2} into {merged}"
                    self.history.append(DeepMutation("merge_variables", desc, [v1, v2], [merged]))
                    new.mutation_description = desc
                    return new

        except Exception as e:
            return None

        return None

    def get_stats(self) -> dict:
        ops = {}
        for m in self.history:
            ops[m.operation] = ops.get(m.operation, 0) + 1
        return {
            "total_mutations": len(self.history),
            "by_operation": ops,
        }
