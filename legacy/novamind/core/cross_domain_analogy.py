"""
CrossDomainAnalogyEngine — Quando modelos falham, busca analogias em dominios DISTANTES.

Nao apenas biologia. Busca principios universais em:
  - Ecologia (redes troficas, sucessao, extincao)
  - Engenharia (feedback control, redundancy, fail-safe)
  - Teoria de controle (PID, estabilidade, bifurcacao)
  - Imunologia (tolerancia, auto-imunidade, memoria)
  - Redes eletricas (cascata, curto-circuito, isolamento)
  - Colonias de formigas (emergencia, divisao de trabalho, swarm intelligence)
  - Economia (externalidades, equilibrio, ciclos)
  - Astrofisica (colapso gravitacional, equilibrio hidrostatico)
  - Computacao (garbage collection, deadlock, race conditions)

Cada analogia gera um NOVO MODELO com estrutura inspirada no outro dominio.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import copy
import random


@dataclass
class Analogy:
    """Uma analogia entre dominios."""
    source_domain: str
    target_domain: str  # sempre "Alzheimer" ou a doenca alvo
    source_concept: str
    mapped_concept: str
    principle: str
    confidence: float
    description: str

    def to_dict(self) -> dict:
        return {
            "from": self.source_domain,
            "mapping": f"{self.source_concept} -> {self.mapped_concept}",
            "principle": self.principle,
            "confidence": round(self.confidence, 2),
        }


class CrossDomainAnalogyEngine:
    """Busca analogias em dominios distantes quando a AGI esta presa.

    Uso:
        engine = CrossDomainAnalogyEngine()
        analogies = engine.search("Alzheimer")
        for a in analogies:
            print(f"{a.source_domain}: {a.source_concept} -> {a.mapped_concept}")
    """

    def __init__(self):
        self.history: List[Analogy] = []

        # Banco de analogias: cada entrada e um dominio com principios e mapeamentos
        self._domain_knowledge = {
            "ecology": {
                "principles": [
                    "keystone_species: uma especie mantem todo ecossistema",
                    "trophic_cascade: predador controla herbivoro que controla planta",
                    "resilience: ecossistema volta ao equilibrio apos disturbio",
                    "regime_shift: ecossistema muda abruptamente de estado",
                    "succession: especies pioneiras preparam para climax",
                    "competitive_exclusion: duas especies nao ocupam o mesmo nicho",
                    "mycorrhizal_network: rede subterranea conecta plantas",
                ],
                "maps": {
                    "microglia": "keystone_species",
                    "amyloid": "invasive_species",
                    "tau": "trophic_cascade",
                    "neuron_loss": "extinction",
                    "neuroinflammation": "wildfire",
                }
            },
            "control_theory": {
                "principles": [
                    "negative_feedback: sistema se auto-regula",
                    "positive_feedback: sistema amplifica",
                    "deadband: sistema ignora pequenas perturbacoes",
                    "hysteresis: sistema tem memoria de estado",
                    "PID: proporcional-integral-derivativo controla",
                    "gain_margin: quao longe da instabilidade",
                    "observer: estado interno e estimado, nao medido",
                ],
                "maps": {
                    "amyloid": "measurement_noise",
                    "tau": "plant_dynamics",
                    "microglia": "controller",
                    "cognitive_decline": "output_error",
                    "neuroinflammation": "accumulated_error",
                }
            },
            "immunology": {
                "principles": [
                    "self_vs_non_self: sistema distingue proprio de estranho",
                    "clonal_selection: celula certa e amplificada",
                    "immune_tolerance: sistema aprende a ignorar",
                    "autoimmunity: sistema ataca o proprio corpo",
                    "memory: exposicao previa acelera resposta",
                    "checkpoint: molecula freia ou ativa imunidade",
                    "cytokine_storm: resposta inflamatoria descontrolada",
                ],
                "maps": {
                    "microglia": "macrophage",
                    "astrocytes": "dendritic_cell",
                    "amyloid": "antigen",
                    "neuroinflammation": "inflammation",
                    "tau": "autoantibody",
                }
            },
            "electrical_engineering": {
                "principles": [
                    "short_circuit: corrente ignora resistencia",
                    "ground_fault: corrente vai para caminho nao esperado",
                    "capacitor: acumula carga, suaviza variacao",
                    "diode: corrente so vai numa direcao",
                    "transistor: pequeno sinal controla grande corrente",
                    "oscillator: sistema entra em ciclo",
                    "brownout: queda gradual de tensao",
                    "cascading_failure: uma falha derruba o sistema",
                ],
                "maps": {
                    "neuron": "transistor",
                    "synapse": "diode",
                    "amyloid": "leakage_current",
                    "tau": "short_circuit",
                    "brain_network": "circuit_board",
                    "cognitive_decline": "brownout",
                }
            },
            "ant_colonies": {
                "principles": [
                    "stigmergy: ambiente guia comportamento",
                    "division_of_labor: especializacao emerge",
                    "swarm_intelligence: grupo resolve o que individuo nao consegue",
                    "pheromone_trail: comunicacao indireta",
                    "task_allocation: formigas se redistribuem conforme necessidade",
                    "colony_collapse: perda da rainha derruba tudo",
                ],
                "maps": {
                    "neuron": "ant",
                    "microglia": "worker_ant",
                    "astrocytes": "nurse_ant",
                    "amyloid": "garbage_pile",
                    "tau": "pheromone_path",
                    "brain": "colony",
                }
            },
            "economics": {
                "principles": [
                    "market_equilibrium: oferta encontra demanda",
                    "externalities: acao de um afeta outros sem compensacao",
                    "tragedy_of_commons: recurso compartilhado e esgotado",
                    "invisible_hand: interesse individual beneficia grupo",
                    "creative_destruction: novo substitui velho",
                    "network_effect: quanto mais usuarios, mais valioso",
                    "information_asymmetry: uma parte sabe mais que outra",
                ],
                "maps": {
                    "amyloid": "market_externality",
                    "tau": "information_asymmetry",
                    "microglia": "regulator",
                    "neuron": "firm",
                    "cognitive_decline": "recession",
                }
            },
            "astrophysics": {
                "principles": [
                    "gravitational_collapse: massa atrai massa, ciclo positivo",
                    "hydrostatic_equilibrium: pressao interna equilibra gravidade",
                    "degeneracy_pressure: particulas nao podem ocupar mesmo estado",
                    "supernova: colapso seguido de explosao",
                    "black_hole: ponto sem retorno",
                    "dark_matter: materia que nao emite luz mas exerce forca",
                    "entropy: desordem sempre aumenta",
                ],
                "maps": {
                    "amyloid": "dark_matter",
                    "tau": "gravitational_collapse",
                    "atrophy": "gravitational_collapse",
                    "neuron_loss": "stellar_death",
                    "cognitive_decline": "entropy",
                }
            },
        }

    def search(self, target_disease: str, n_analogies: int = 5) -> List[Analogy]:
        """Busca analogias de outros dominios para a doenca alvo.

        Returns: lista de analogias com mapeamentos.
        """
        results = []

        for domain, knowledge in self._domain_knowledge.items():
            for principle in knowledge["principles"]:
                parts = principle.split(": ", 1)
                concept = parts[0] if len(parts) > 0 else principle
                description = parts[1] if len(parts) > 1 else principle

                # Mapeia conceito do dominio para a doenca
                # Se houver mapping explicito, usa
                mapped = None
                for bio_concept, domain_concept in knowledge.get("maps", {}).items():
                    if domain_concept == concept:
                        mapped = bio_concept
                        break

                if mapped is None:
                    mapped = concept  # usa o conceito original

                analogy = Analogy(
                    source_domain=domain,
                    target_domain=target_disease,
                    source_concept=concept,
                    mapped_concept=mapped,
                    principle=description,
                    confidence=0.3,
                    description=f"In {domain}, {principle}. In {target_disease}, this maps to {mapped}.",
                )
                results.append(analogy)

        # Ordena por relevancia e retorna
        self.history.extend(results)
        return results[:n_analogies]

    def apply_analogy(self, model, analogy: Analogy):
        """Aplica uma analogia para GERAR UM NOVO MODELO.

        Pega um principio de outro dominio e o traduz em
        uma modificacao concreta no modelo ODE.
        """
        new = copy.deepcopy(model)
        new.name = f"{model.name}_{analogy.source_domain}_{len(self.history)}"
        new.is_mutated = True
        new.parent_model = model.name

        principle = analogy.principle.lower()
        source = analogy.source_domain

        if "feedback" in principle or "equilibrium" in principle:
            # Adiciona termo de feedback negativo (auto-regulacao)
            ode_orig = model.ode_fn
            def ode_feedback(t, y, orig=ode_orig):
                dy = orig(t, y)
                # Termo de estabilizacao
                if len(dy) > 0:
                    dy[0] -= 0.02 * (y[0] - 0.5)  # puxa para 0.5
                return dy
            new.ode_fn = ode_feedback
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        elif "collapse" in principle or "extinction" in principle or "death" in principle:
            # Adiciona termo de colapso (nao-linearidade forte)
            ode_orig = model.ode_fn
            def ode_collapse(t, y, orig=ode_orig):
                dy = orig(t, y)
                if len(dy) > 1:
                    dy[1] -= 0.05 * y[1] ** 2  # colapso quadratico
                return dy
            new.ode_fn = ode_collapse
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        elif "swarm" in principle or "colony" in principle or "emergence" in principle:
            # Adiciona termo de emergencia (interacao coletiva)
            ode_orig = model.ode_fn
            def ode_swarm(t, y, orig=ode_orig):
                dy = orig(t, y)
                if len(dy) >= 3:
                    # Interacao entre microglia, astrocytes, neurons
                    for i in range(min(3, len(dy))):
                        for j in range(min(3, len(dy))):
                            if i != j:
                                dy[i] += 0.01 * y[i] * y[j]  # interacao coletiva
                return dy
            new.ode_fn = ode_swarm
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        elif "cascade" in principle or "cascading" in principle:
            # Adiciona efeito cascata (uma variavel derruba as seguintes)
            ode_orig = model.ode_fn
            def ode_cascade(t, y, orig=ode_orig):
                dy = orig(t, y)
                for i in range(1, len(dy)):
                    dy[i] += 0.03 * y[i-1]  # cascata progressiva
                return dy
            new.ode_fn = ode_cascade
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        elif "memory" in principle or "hysteresis" in principle:
            # Adiciona memoria (integrador)
            ode_orig = model.ode_fn
            integral = [0.0]
            def ode_memory(t, y, orig=ode_orig, acc=integral):
                dy = orig(t, y)
                if len(dy) > 0:
                    acc[0] += dy[0] * 0.01  # acumula
                    dy[0] += acc[0] * 0.001  # memoria afeta dinamica
                return dy
            new.ode_fn = ode_memory
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        else:
            # Analogia generica: adiciona pequena perturbacao
            ode_orig = model.ode_fn
            def ode_generic(t, y, orig=ode_orig):
                dy = orig(t, y)
                if len(dy) > 0:
                    dy[0] += 0.01 * np.sin(t * 0.01)  # oscilacao lenta
                return dy
            new.ode_fn = ode_generic
            new.mutation_description = f"Analogy from {source}: {analogy.principle[:40]}"

        return new

    def get_stats(self) -> dict:
        domains = {}
        for a in self.history:
            domains[a.source_domain] = domains.get(a.source_domain, 0) + 1
        return {
            "total_analogies": len(self.history),
            "domains_explored": len(domains),
            "by_domain": domains,
        }
