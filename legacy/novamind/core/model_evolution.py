"""
ModelEvolution — Evolucao de modelos por selecao natural + mutacao estrutural.

Ciclo:
  1. G10: modelos iniciais competem (intervention discrimination)
  2. Vencedor é selecionado
  3. StructureMutator gera filhos (mutacoes estruturais)
  4. Filhos competem entre si + contra o pai
  5. Se algum filho vence o pai → nova geracao
  6. Repete: G10, G11, G12... ate convergencia

Cada geracao = modelos melhores.
A DMN aqui gera criatividade: "qual premissa posso inverter?"
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.integrate import solve_ivp
import numpy as np
import time
import random

from core.model_library import ModelLibrary, CausalModel
from core.structure_mutator import StructureMutator


@dataclass
class ModelFitness:
    """Fitness de um modelo apos competicao."""
    model_name: str
    prediction_error: float  # quao longe da realidade (0 = perfeito)
    parent: Optional[str] = None
    generation: int = 0
    mutations_applied: List[str] = field(default_factory=list)
    is_alive: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.model_name,
            "error": round(self.prediction_error, 4),
            "parent": self.parent or "root",
            "generation": self.generation,
            "mutations": self.mutations_applied,
        }


class ModelEvolution:
    """Evolucao de modelos por selecao natural.

    Uso:
        evo = ModelEvolution()
        evo.initialize("Alzheimer")  # carrega modelos iniciais (G10)
        evo.run(generations=10)      # evolui por 10 geracoes
        best = evo.get_best_model()
    """

    def __init__(self):
        self.lib = ModelLibrary()
        self.mutator = StructureMutator()
        self.generations: List[List[ModelFitness]] = []
        self._models: Dict[str, CausalModel] = {}  # name -> CausalModel
        self._fitness_history: List[dict] = []
        self._disease = ""

    def initialize(self, disease: str):
        """Carrega modelos iniciais para uma doenca."""
        self._disease = disease
        self.generations = []
        self._models = {}
        self._fitness_history = []

        models = self.lib.get_models(disease)
        for m in models:
            self._models[m.name] = m

        print(f"ModelEvolution: {disease}")
        print(f"  Modelos iniciais (G10): {len(models)}")
        for m in models:
            print(f"    {m.name}: {m.hypothesis[:60]}")

    def evaluate(self, model_name: str,
                 intervention: str = "BACE1_inhibition",
                 dose: float = 0.5,
                 real_outcome: float = 0.30) -> ModelFitness:
        """Avalia um modelo: simula intervencao, compara com realidade.

        Args:
            model_name: nome do modelo
            intervention: intervencao a simular
            dose: dose da intervencao (0-1)
            real_outcome: resultado real observado (ex: 0.30 = 30%)

        Returns: ModelFitness com erro (0 = perfeito)
        """
        model = self._models.get(model_name)
        if not model:
            return ModelFitness(model_name, prediction_error=999.0)

        try:
            # Simula controle
            sol_ctrl = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                                method="RK45", max_step=1, rtol=1e-6, atol=1e-8)
            # Simula com intervencao
            if model.intervention_fn:
                y_interv = model.intervention_fn(model.initial_state.copy(), intervention, dose)
            else:
                y_interv = model.initial_state.copy()
            sol_interv = solve_ivp(model.ode_fn, (0, 365), y_interv,
                                  method="RK45", max_step=1, rtol=1e-6, atol=1e-8)

            if sol_ctrl.y.shape[1] == 0 or sol_interv.y.shape[1] == 0:
                return ModelFitness(model_name, prediction_error=1.0)

            # Extrai predicao cognitiva
            cog_idx = model.variables.index("cognitive_function") if "cognitive_function" in model.variables else -1
            if cog_idx >= 0 and cog_idx < sol_ctrl.y.shape[0] and cog_idx < sol_interv.y.shape[0]:
                ctrl_end = float(sol_ctrl.y[cog_idx, -1])
                interv_end = float(sol_interv.y[cog_idx, -1])
                if abs(ctrl_end) > 1e-6:
                    predicted_change = (interv_end - ctrl_end) / abs(ctrl_end)
                else:
                    predicted_change = 0.0
            else:
                predicted_change = 0.0

            # Erro = |predicao - realidade|
            error = abs(predicted_change - real_outcome)

            return ModelFitness(model_name, prediction_error=round(error, 4))

        except Exception as e:
            return ModelFitness(model_name, prediction_error=1.0)

    def run_generation(self, models_to_test: List[str],
                       intervention: str = "BACE1_inhibition",
                       dose: float = 0.5,
                       real_outcome: float = 0.30,
                       generation_num: int = 0) -> Tuple[ModelFitness, List[ModelFitness]]:
        """Roda uma geracao: testa todos os modelos, seleciona vencedor."""
        print(f"\n  --- Geracao {generation_num} ({len(models_to_test)} modelos) ---")

        results = []
        for name in models_to_test:
            fitness = self.evaluate(name, intervention, dose, real_outcome)
            results.append(fitness)
            print(f"    {name:35s} erro={fitness.prediction_error:.4f}")

        # Seleciona vencedor (menor erro)
        results.sort(key=lambda f: f.prediction_error)
        best = results[0]

        print(f"    >> Vencedor: {best.model_name} (erro={best.prediction_error:.4f})")

        self.generations.append(results)
        self._fitness_history.append({
            "generation": generation_num,
            "best": best.model_name,
            "best_error": best.prediction_error,
            "n_models": len(models_to_test),
        })

        return best, results

    def mutate_winner(self, winner_name: str, n_children: int = 3) -> List[str]:
        """Gera mutacoes do modelo vencedor.

        Cada mutacao: adiciona variavel, feedback, ou muda estrutura.
        """
        parent = self._models.get(winner_name)
        if not parent:
            return []

        children = self.mutator.mutate(parent, n_mutations=n_children)
        child_names = []

        for child in children:
            # Nome unico
            child_name = child.name
            self._models[child_name] = child
            child_names.append(child_name)

            print(f"    -> Novo: {child_name}")
            print(f"       Mutacao: {child.mutation_description[:60]}")

        return child_names

    def run(self, generations: int = 5,
            intervention: str = "BACE1_inhibition",
            dose: float = 0.5,
            real_outcome: float = 0.30) -> ModelFitness:
        """Loop evolutivo principal.

        Para cada geracao:
          1. Testa todos os modelos vivos
          2. Seleciona vencedor
          3. Gera mutacoes do vencedor
          4. Testa mutacoes
          5. Se mutacao vence o pai -> nova geracao com ela
          6. Senao -> mantem o pai e para
        """
        if not self._models:
            raise ValueError("Execute initialize() primeiro")

        # G10: modelos iniciais
        initial_names = list(self._models.keys())
        best, _ = self.run_generation(initial_names, intervention, dose,
                                       real_outcome, generation_num=0)

        for gen in range(1, generations + 1):
            # Gera mutacoes do vencedor
            child_names = self.mutate_winner(best.model_name, n_children=2)
            if not child_names:
                print(f"  Nenhuma mutacao viavel. Evolucao parou.")
                break

            # Testa filhos + pai (para ver se algum filho supera)
            contenders = [best.model_name] + child_names
            new_best, _ = self.run_generation(contenders, intervention, dose,
                                               real_outcome, generation_num=gen)

            # Se filho venceu o pai, ele e o novo vencedor
            if new_best.model_name != best.model_name:
                best = new_best
                print(f"  -> EVOLUCAO: {best.model_name} substitui o pai!")
            else:
                print(f"  -> ESTAGNACAO: nenhum filho superou o pai.")
                # Para se estagnou por 2 geracoes consecutivas
                if gen >= 2:
                    print(f"  Evolucao convergiu na geracao {gen}.")
                    break

        return best

    def get_best_model(self) -> Optional[CausalModel]:
        """Retorna o melhor modelo da ultima geracao."""
        if not self.generations:
            return None
        best_fitness = min(self.generations[-1], key=lambda f: f.prediction_error)
        return self._models.get(best_fitness.model_name)

    def get_fitness_history(self) -> List[dict]:
        return self._fitness_history

    def get_stats(self) -> dict:
        total_models = len(self._models)
        total_generations = len(self.generations)
        best = self.get_best_model()
        return {
            "disease": self._disease,
            "total_generations": total_generations,
            "total_models_ever": total_models,
            "best_model": best.name if best else "none",
            "fitness_history": self._fitness_history,
        }
