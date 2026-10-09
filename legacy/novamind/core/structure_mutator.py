"""
StructureMutator — Modifica a ESTRUTURA CAUSAL dos modelos.

Nao muda parametros. Muda a topologia: adiciona/remove nos, edges,
muda tipos de interacao, adiciona feedback loops.

Cada mutacao gera um NOVO modelo com estrutura causal diferente.
O novo modelo e entao testado contra os dados.
"""
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import copy
import random


@dataclass
class Mutation:
    """Descricao de uma mutacao estrutural."""
    operation: str  # "add_node", "remove_node", "add_edge", "remove_edge", "add_feedback", "bifurcate"
    description: str
    target_variable: str = ""
    new_variable: str = ""
    confidence_delta: float = 0.0

    def to_dict(self) -> dict:
        return {
            "op": self.operation,
            "desc": self.description[:60],
            "target": self.target_variable,
            "new_var": self.new_variable,
        }


class StructureMutator:
    """Propoe mutacoes estruturais em modelos ODE.

    Uso:
        mut = StructureMutator()
        new_models = mut.mutate(model, n_mutations=5)
        for m in new_models:
            print(m.name, m.mutation_description)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.mutation_history: List[Mutation] = []

        self._latent_pool = {
            "Alzheimer": ["autophagy", "glymphatic_clearance", "complement_activation",
                          "neural_network_disruption", "epigenetic_dysregulation",
                          "calcium_dysregulation", "vascular_dysfunction",
                          "circadian_disruption", "gut_microbiome", "olfactory_deficit"],
            "Parkinson": ["gut_brain_axis", "ceramide_accumulation", "lysosomal_dysfunction",
                         "dopamine_metabolites", "calcium_channel_dysfunction"],
            "cancer": ["metabolic_reprogramming", "immune_exhaustion", "stromal_remodeling",
                      "epithelial_mesenchymal_transition", "dormancy"],
            "depression": ["circadian_disruption", "inflammatory_refractory",
                          "HPA_axis_feedback", "social_defeat_encoding"],
        }

        self._feedback_types = ["positive", "negative", "bistable", "ultrasensitive"]

    def mutate(self, model, n_mutations: int = 3) -> list:
        """Gera ate N modelos mutados a partir de um modelo base.

        Cada mutacao:
          1. Copia o modelo base
          2. Aplica 1 transformacao estrutural
          3. Ajusta as ODEs correspondentemente
          4. Retorna o novo modelo

        Returns: lista de (CausalModel, Mutation)
        """
        results = []

        for _ in range(n_mutations):
            operation = self.rng.choice([
                "add_latent_variable", "add_feedback_loop",
                "add_cross_coupling", "split_variable",
                "calibrate_ode", "mutate_intervention",
                "add_feedback_core",
            ])

            mutated_model = self._apply_mutation(model, operation)
            if mutated_model:
                results.append(mutated_model)

        return results

    def _apply_mutation(self, model, operation: str):
        """Aplica uma mutacao ao modelo."""
        try:
            disease = model.disease
            new_name = f"{model.name}_mut_{self.rng.randint(100,999)}"

            if operation == "add_latent_variable":
                # Adiciona uma variavel latente como mediadora
                latents = self._latent_pool.get(disease, ["latent_factor"])
                latent = self.rng.choice(latents)
                mutation = Mutation("add_node", f"Added latent variable '{latent}' as mediator",
                                  new_variable=latent)

                # Cria novo estado com variavel extra
                base_vars = list(model.variables)
                new_vars = base_vars + [latent]
                y0_new = np.append(model.initial_state, [0.3])

                # Nova ODE com termo para a variavel latente
                ode_original = model.ode_fn
                n = len(base_vars)

                def ode_new(t, y, ode_orig=ode_original, n=n, latent_idx=n):
                    dy = ode_orig(t, y[:n])
                    # Variavel latente: lenta, influenciada por primeira variavel
                    d_latent = 0.01*y[latent_idx] + 0.02*y[0] - 0.01*y[latent_idx]
                    # Primeira variavel agora tambem influenciada pela latente
                    dy[0] += 0.01*y[latent_idx]
                    return np.append(dy, [d_latent])

                new_model = copy.deepcopy(model)
                new_model.name = new_name
                new_model.variables = new_vars
                new_model.initial_state = y0_new
                new_model.ode_fn = ode_new
                new_model.is_mutated = True
                new_model.parent_model = model.name
                new_model.mutation_description = mutation.description
                self.mutation_history.append(mutation)
                return new_model

            elif operation == "add_feedback_loop":
                # Adiciona feedback de uma variavel tardia para uma inicial
                if len(model.variables) >= 3:
                    target = self.rng.choice(model.variables[:3])
                    source = self.rng.choice(model.variables[-3:])
                    mutation = Mutation("add_feedback", f"Feedback: {source} -> {target}",
                                      target_variable=target)

                    ode_orig = model.ode_fn
                    t_idx = model.variables.index(target)
                    s_idx = model.variables.index(source)

                    def ode_fb(t, y, ode_orig=ode_orig, t_idx=t_idx, s_idx=s_idx):
                        dy = ode_orig(t, y)
                        dy[t_idx] += 0.01 * y[s_idx]  # feedback
                        return dy

                    new_model = copy.deepcopy(model)
                    new_model.name = new_name
                    new_model.ode_fn = ode_fb
                    new_model.is_mutated = True
                    new_model.parent_model = model.name
                    new_model.mutation_description = mutation.description
                    self.mutation_history.append(mutation)
                    return new_model

            elif operation == "add_cross_coupling":
                # Acopla duas variaveis que antes eram independentes
                if len(model.variables) >= 4:
                    v1, v2 = self.rng.sample(model.variables[:4], 2)
                    mutation = Mutation("add_edge", f"Coupled {v1} <-> {v2}",
                                      target_variable=v1)

                    ode_orig = model.ode_fn
                    i1 = model.variables.index(v1)
                    i2 = model.variables.index(v2)

                    def ode_couple(t, y, ode_orig=ode_orig, i1=i1, i2=i2):
                        dy = ode_orig(t, y)
                        dy[i1] += 0.015 * y[i2]
                        dy[i2] += 0.015 * y[i1]
                        return dy

                    new_model = copy.deepcopy(model)
                    new_model.name = new_name
                    new_model.ode_fn = ode_couple
                    new_model.is_mutated = True
                    new_model.parent_model = model.name
                    new_model.mutation_description = mutation.description
                    self.mutation_history.append(mutation)
                    return new_model

            elif operation == "calibrate_ode":
                # Calibra ODEs: escala termos das variaveis principais
                target_vars = model.variables[:3]
                scales = {v: self.rng.uniform(0.5, 2.0) for v in target_vars}
                indices = {v: model.variables.index(v) for v in target_vars if v in model.variables}

                ode_orig = model.ode_fn
                def ode_cal(t, y, ode_orig=ode_orig, indices=indices, scales=scales):
                    dy = ode_orig(t, y)
                    for var_name, idx in indices.items():
                        if idx < len(dy):
                            dy[idx] *= scales.get(var_name, 1.0)
                    return dy

                desc = f"Calibrated ODE: {', '.join(f'{v}x{scales[v]:.2f}' for v in target_vars)}"
                mutation = Mutation("calibrate_ode", desc)
                new_model = copy.deepcopy(model)
                new_model.name = new_name; new_model.ode_fn = ode_cal
                new_model.is_mutated = True; new_model.parent_model = model.name
                new_model.mutation_description = desc
                self.mutation_history.append(mutation)
                return new_model

            elif operation == "mutate_intervention":
                # Muda a funcao de intervencao para afetar variaveis centrais
                if model.intervention_fn:
                    interv_orig = model.intervention_fn
                    def interv_mut(y0, i, d, orig=interv_orig):
                        y = orig(y0, i, d)
                        y[0] *= (1.0 + d * self.rng.uniform(-0.3, 0.0))
                        return y
                    desc = f"Mutated intervention: additional effect on {model.variables[0]}"
                    mutation = Mutation("mutate_intervention", desc, target_variable=model.variables[0])
                    new_model = copy.deepcopy(model)
                    new_model.name = new_name; new_model.intervention_fn = interv_mut
                    new_model.is_mutated = True; new_model.parent_model = model.name
                    new_model.mutation_description = desc
                    self.mutation_history.append(mutation)
                    return new_model

            elif operation == "add_feedback_core":
                # Feedback da ultima variavel (ex: cognitive) para a primeira (ex: tau/amyloid)
                if len(model.variables) >= 3:
                    s_idx = len(model.variables) - 1
                    t_idx = 0
                    strength = self.rng.uniform(0.01, 0.05)
                    ode_orig = model.ode_fn
                    def ode_fb_core(t, y, ode_orig=ode_orig, ti=t_idx, si=s_idx, st=strength):
                        dy = ode_orig(t, y)
                        dy[ti] += st * y[si]
                        return dy
                    desc = f"Core feedback: {model.variables[-1]} -> {model.variables[0]} (x{strength:.3f})"
                    mutation = Mutation("add_feedback_core", desc, target_variable=model.variables[0])
                    new_model = copy.deepcopy(model)
                    new_model.name = new_name; new_model.ode_fn = ode_fb_core
                    new_model.is_mutated = True; new_model.parent_model = model.name
                    new_model.mutation_description = desc
                    self.mutation_history.append(mutation)
                    return new_model

        except Exception:
            return None

        return None

    def get_stats(self) -> dict:
        ops = {}
        for m in self.mutation_history:
            ops[m.operation] = ops.get(m.operation, 0) + 1
        return {
            "total_mutations": len(self.mutation_history),
            "by_operation": ops,
        }
