"""
ConceptInventor — Inventa CONCEITOS NOVOS quando as variaveis existentes sao insuficientes.

Quando a AGI esta presa (erro alto, mutacoes nao melhoram), ela pergunta:
  "Que conceito esta faltando?"

Exemplos de conceitos que a AGI pode inventar:
  - "glymphatic_clearance" (descoberto recentemente em neurociencia)
  - "prion-like_propagation" (tau se espalha como prion)
  - "excitotoxicity_cycle" (glutamate em excesso mata neuronios)
  - "metabolic_exhaustion" (neuronios ficam sem energia)
  - "network_fragmentation" (redes neurais perdem conectividade)

Cada conceito inventado vira uma NOVA VARIAVEL no modelo ODE.
"""
from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
import copy
import random


@dataclass
class InventedConcept:
    """Um conceito totalmente novo inventado pela AGI."""
    name: str
    domain: str
    rationale: str  # por que a AGI acha que esse conceito existe
    confidence: float  # 0-1, quao confiante
    interacts_with: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "rationale": self.rationale[:60],
            "confidence": round(self.confidence, 2),
        }


class ConceptInventor:
    """Inventa conceitos que podem explicar o erro residual do modelo.

    Uso:
        inventor = ConceptInventor()
        concepts = inventor.invent("Alzheimer", error=0.5)
        for c in concepts:
            model = inventor.apply_to_model(best_model, c)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.inventions: List[InventedConcept] = []

        # Banco de "lacunas" conhecidas em cada doenca
        self._knowledge_gaps = {
            "Alzheimer": [
                ("glymphatic_clearance", "Clearance of waste from brain. If impaired, amyloid accumulates.", ["amyloid_beta", "cognitive_function"]),
                ("prion_like_propagation", "Tau spreads from cell to cell like prions.", ["p_tau", "neurodegeneration"]),
                ("excitotoxicity_cycle", "Glutamate excess kills neurons in a cycle.", ["neurodegeneration", "synaptic_function"]),
                ("metabolic_exhaustion", "Neurons run out of ATP, cannot maintain firing.", ["cognitive_function", "mitochondrial_function"]),
                ("network_fragmentation", "Neural networks lose connectivity, not just neurons.", ["cognitive_function", "synaptic_function"]),
                ("complement_activation", "Immune complement system tags synapses for removal.", ["synaptic_function", "neuroinflammation"]),
                ("calcium_overload", "Calcium enters cells uncontrollably, triggering death.", ["neurodegeneration", "mitochondrial_function"]),
                ("epigenetic_silencing", "Genes for repair are turned off by age.", ["neurodegeneration", "oxidative_stress"]),
                ("vascular_stiffening", "Blood vessels stiffen, reducing oxygen delivery.", ["cognitive_function", "oxidative_stress"]),
                ("sleep_fragmentation", "Sleep is disrupted, reducing clearance.", ["amyloid_beta", "glymphatic_clearance"]),
            ],
            "Parkinson": [
                ("gut_brain_axis", "Gut microbiome influences brain via vagus nerve.", ["alpha_synuclein", "dopaminergic_loss"]),
                ("lysosomal_dysfunction", "Cells cannot clear damaged proteins.", ["alpha_synuclein", "mitochondrial_function"]),
                ("ceramide_accumulation", "Lipid metabolism disrupts membranes.", ["alpha_synuclein", "dopaminergic_loss"]),
            ],
        }

        # Gerador de nomes para conceitos realmente novos
        self._prefixes = ["meta_", "crypto_", "proto_", "para_", "epi_", "hypo_", "hyper_", "dys_"]
        self._suffixes = ["_oscillation", "_cascade", "_threshold", "_gradient", "_potential", "_flux", "_depletion"]

    def invent(self, disease: str, error: float = 0.5,
               n_concepts: int = 3) -> List[InventedConcept]:
        """Inventa conceitos que podem reduzir o erro do modelo.

        Args:
            disease: doenca alvo
            error: erro atual do modelo (0 = perfeito, 1 = pessimo)
            n_concepts: numero de conceitos para inventar

        Returns: lista de conceitos inventados.
        """
        concepts = []

        # 1. Se erro e alto, busca conceitos do banco de lacunas
        if error > 0.3:
            gaps = self._knowledge_gaps.get(disease, [])
            self.rng.shuffle(gaps)
            for name, rationale, interacts in gaps[:n_concepts]:
                concept = InventedConcept(
                    name=name,
                    domain=disease,
                    rationale=rationale,
                    confidence=max(0.3, 1.0 - error),
                    interacts_with=interacts,
                )
                concepts.append(concept)

        # 2. Se erro e muito alto, inventa conceitos ORIGINAIS
        if error > 0.6:
            for _ in range(max(1, n_concepts - len(concepts))):
                # Gera nome composto
                name = f"{self.rng.choice(self._prefixes)}{self.rng.choice(['amyloid', 'tau', 'synaptic', 'neural', 'glial', 'vascular', 'metabolic'])}{self.rng.choice(self._suffixes)}"
                concept = InventedConcept(
                    name=name,
                    domain=disease,
                    rationale=f"Unknown factor contributing to residual error of {error:.2f}",
                    confidence=0.2,
                    interacts_with=self.rng.sample(
                        ["amyloid_beta", "p_tau", "neurodegeneration", "cognitive_function"],
                        self.rng.randint(1, 3)
                    ),
                )
                concepts.append(concept)

        self.inventions.extend(concepts)
        return concepts

    def apply_to_model(self, model, concept: InventedConcept):
        """Aplica um conceito inventado ao modelo: cria nova variavel com ODE."""
        new = copy.deepcopy(model)
        new.name = f"{model.name}_invented_{concept.name}"
        new.parent_model = model.name
        new.is_mutated = True

        # Adiciona nova variavel
        safe_name = concept.name.replace("-", "_").replace(" ", "_")
        new.variables = list(model.variables) + [safe_name]
        new.initial_state = np.append(model.initial_state, [0.5])

        # Indices das variaveis com que interage
        interact_indices = []
        for v in concept.interacts_with:
            if v in model.variables:
                interact_indices.append(model.variables.index(v))

        ode_orig = model.ode_fn
        new_idx = len(model.variables)

        def ode_invented(t, y, orig=ode_orig, ni=new_idx, inter=interact_indices):
            dy = list(orig(t, y[:ni]))
            # Dinamica do novo conceito: influenciado pelas variaveis com que interage
            d_new = 0.01 * y[ni]
            for idx in inter:
                if idx < len(dy):
                    d_new += 0.02 * y[idx]  # influencia recebida
                    if idx < len(dy):
                        dy[idx] += 0.01 * y[ni]  # influencia exercida
            d_new -= 0.01 * y[ni]
            dy.append(d_new)
            return np.array(dy)

        new.ode_fn = ode_invented
        new.mutation_description = f"Invented concept: {concept.name} ({concept.rationale[:50]})"
        return new

    def get_stats(self) -> dict:
        return {
            "total_invented": len(self.inventions),
            "avg_confidence": round(
                sum(c.confidence for c in self.inventions) / max(1, len(self.inventions)), 2
            ),
            "recent": [c.to_dict() for c in self.inventions[-5:]],
        }
