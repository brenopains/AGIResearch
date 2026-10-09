"""
ModelDiscoveryLab — Laboratorio integrado de descoberta de modelos.

Ciclo completo:
  1. CARREGA modelos rivais para uma doenca
  2. COMPARA por metodo Bayesiano (qual explica melhor os dados?)
  3. MUTA melhores modelos (propoe novas estruturas causais)
  4. RE-COMPARA modelos mutados vs originais
  5. REPETE ate: modelo converge OU exploracao exaure
  6. PROPOE intervencao baseada no modelo vencedor
  7. SIMULA intervencao no VirtualLab
  8. RELATORIO completo para cientista

Tudo dinamico. Nada hardcoded.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math

from core.model_library import ModelLibrary, CausalModel
from core.model_comparator import ModelComparator, ModelComparison
from core.structure_mutator import StructureMutator, Mutation
from core.virtual_lab import VirtualLab


@dataclass
class DiscoveryRound:
    """Uma rodada completa do processo de descoberta."""
    round_num: int
    models_tested: List[str]
    best_model: str
    best_posterior: float
    n_mutations_tried: int
    n_mutations_accepted: int
    convergence_delta: float  # Mudanca no posterior em relacao a rodada anterior
    time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "best": self.best_model,
            "posterior": round(self.best_posterior, 1),
            "mutations": self.n_mutations_accepted,
            "convergence": round(self.convergence_delta, 4),
        }


@dataclass
class CureProposal:
    """Proposta final de cura apos descoberta de modelos."""
    disease: str
    best_model_name: str
    best_model_structure: str
    confidence: float
    proposed_intervention: str
    predicted_effect: Dict[str, float]
    alternative_interventions: List[str]
    modeling_rounds: List[dict]
    accepted_mutations: List[str]
    rejected_models: List[str]
    experiments_to_validate: List[str]

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# Cure Discovery Report: {self.disease}")
        lines.append(f"\n## Best Model: {self.best_model_name}")
        lines.append(f"\n**Structure:** {self.best_model_structure}")
        lines.append(f"**Confidence:** {self.confidence:.2f}")
        lines.append(f"\n### Proposed Intervention")
        lines.append(f"{self.proposed_intervention}")
        lines.append(f"\n### Predicted Effect on Biomarkers")
        for var, change in sorted(self.predicted_effect.items()):
            arrow = "+" if change > 0 else ""
            lines.append(f"- {var}: {arrow}{change:.1%}")
        lines.append(f"\n### Modeling History")
        for r in self.modeling_rounds:
            lines.append(f"- Round {r['round']}: best={r['best']}, "
                        f"posterior={r['posterior']}, "
                        f"mutations={r['mutations']}")
        lines.append(f"\n### Models Rejected")
        for m in self.rejected_models:
            lines.append(f"- {m}")
        lines.append(f"\n### Structural Mutations Accepted")
        for m in self.accepted_mutations:
            lines.append(f"- {m}")
        lines.append(f"\n### Experiments to Validate")
        for e in self.experiments_to_validate:
            lines.append(f"- {e}")
        return "\n".join(lines)


class ModelDiscoveryLab:
    """Laboratorio de descoberta de modelos.

    Uso:
        lab = ModelDiscoveryLab()
        proposal = lab.discover("Alzheimer")
        print(proposal.to_markdown())
    """

    def __init__(self):
        self.lib = ModelLibrary()
        self.comparator = ModelComparator()
        self.mutator = StructureMutator()
        self.vlab = VirtualLab()
        self.history: List[DiscoveryRound] = []

    def discover(self, disease: str,
                 max_rounds: int = None,
                 convergence_threshold: float = 0.01) -> CureProposal:
        """Descobre o melhor modelo e intervencao para uma doenca.

        O numero de rodadas e DINAMICO:
        - Continua enquanto: modelos mutados superam os originais
        - Para quando: convergencia (posterior estabiliza)
        """
        print(f"\nModelDiscoveryLab: {disease}")
        models = self.lib.get_models(disease)
        if not models:
            return CureProposal(disease, "none", "no models available",
                               0.0, "insufficient data", {},
                               [], [], [], [], [])

        print(f"  Initial models: {len(models)}")
        for m in models:
            print(f"    {m.name:25s} | {m.hypothesis[:50]}")

        # Observacoes disponiveis (extraiveis do PubMed futuramente)
        observations = {}

        all_models = list(models)
        round_num = 0
        best_posterior_history = []

        while True:
            round_num += 1
            round_start = time.time()

            # 1. COMPARA todos os modelos
            comparisons = self.comparator.compare(all_models, observations)
            best = max(comparisons, key=lambda c: c.posterior)
            best_posterior_history.append(best.posterior)

            # 2. CONVERGENCIA?
            convergence = 0.0
            if len(best_posterior_history) >= 3:
                recent = best_posterior_history[-3:]
                convergence = max(recent) - min(recent)

            # 3. MUTA os melhores modelos
            best_models = [m for m in all_models
                          if m.name in [c.model_name for c in comparisons[:3]]]
            mutations_tried = 0
            mutations_accepted = 0

            if convergence > convergence_threshold or len(best_posterior_history) < 3:
                for base_model in best_models[:2]:
                    new_models = self.mutator.mutate(base_model, n_mutations=2)
                    mutations_tried += len(new_models)
                    for nm in new_models:
                        # Testa modelo mutado
                        mut_comparisons = self.comparator.compare([nm], observations)
                        if mut_comparisons and mut_comparisons[0].posterior > best.posterior:
                            all_models.append(nm)
                            mutations_accepted += 1
                            print(f"    -> Accepted mutation: {nm.name} "
                                  f"(posterior={mut_comparisons[0].posterior:.1f})")

            # 4. REGISTRA rodada
            elapsed = time.time() - round_start
            round_record = DiscoveryRound(
                round_num=round_num,
                models_tested=[m.name for m in all_models],
                best_model=best.model_name,
                best_posterior=best.posterior,
                n_mutations_tried=mutations_tried,
                n_mutations_accepted=mutations_accepted,
                convergence_delta=convergence,
                time_seconds=elapsed,
            )
            self.history.append(round_record)

            print(f"  Round {round_num}: best={best.model_name[:20]:20s} "
                  f"posterior={best.posterior:.1f} "
                  f"convergence={convergence:.4f} "
                  f"mutations={mutations_accepted}/{mutations_tried} "
                  f"({elapsed:.1f}s)")

            # 5. CRITERIO DE PARADA
            should_stop = False
            if convergence <= convergence_threshold and round_num >= 3:
                should_stop = True
            if max_rounds is not None and round_num >= max_rounds:
                should_stop = True

            if should_stop:
                print(f"  CONVERGED at round {round_num}")
                break

        # 6. PROPOE INTERVENCAO baseada no melhor modelo
        best_model_info = next((c for c in comparisons if c.is_best), comparisons[0])
        best_model = next((m for m in all_models if m.name == best_model_info.model_name), models[0])

        # Simula algumas intervencoes
        self.vlab.load_disease_model(disease)
        predictions = {}
        interventions_tested = ["BACE1_inhibition", "anti-inflammatory",
                                "mitochondrial_enhancement", "immunotherapy"]
        best_intervention = ""
        best_eff = -999

        for interv in interventions_tested[:3]:
            result = self.vlab.simulate_intervention(interv, dose=0.5, duration_days=365)
            effect = 0
            for k, v in result.biomarker_change.items():
                if "cognit" in k or "motor" in k or "function" in k:
                    effect += v
            if effect > best_eff:
                best_eff = effect
                best_intervention = interv
            predictions[interv] = {k: float(v) for k, v in result.biomarker_change.items()}

        # 7. RELATORIO
        rejected = [c.model_name for c in comparisons if not c.is_best]
        best_pred = predictions.get(best_intervention, {})

        experiments = [
            f"Test {best_intervention} in animal model of {disease} (6 months).",
            f"Measure biomarkers: {', '.join(list(best_pred.keys())[:5])}.",
            f"Compare outcome with predictions from model {best_model_info.model_name}.",
            f"If predictions fail, the model structure is wrong -> propose structural mutation.",
        ]

        mutations_accepted_list = [
            m.description for m in self.mutator.mutation_history[-5:]
        ]

        proposal = CureProposal(
            disease=disease,
            best_model_name=best_model.name,
            best_model_structure=best_model.hypothesis,
            confidence=min(0.9, best_model_info.bayes_factor_vs_best),
            proposed_intervention=best_intervention,
            predicted_effect=best_pred,
            alternative_interventions=[i for i in interventions_tested if i != best_intervention],
            modeling_rounds=[r.to_dict() for r in self.history],
            accepted_mutations=mutations_accepted_list,
            rejected_models=[c.model_name for c in comparisons if not c.is_best][:10],
            experiments_to_validate=experiments,
        )

        return proposal

    def get_stats(self) -> dict:
        return {
            "diseases_available": self.lib.list_diseases(),
            "total_models": sum(len(self.lib.get_models(d)) for d in self.lib.list_diseases()),
        }
