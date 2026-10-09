"""
MetaModeler v2 — Primeiro extrai propriedades SEMANTICAS do problema,
depois ESCOLHE o formalismo baseado na adequacao conceitual,
e SO ENTAO calibra.

Muda a ordem:
  Antes: testar todos -> calibrar -> escolher menor erro
  Agora: analisar -> extrair fingerprint -> escolher formalismo -> construir -> calibrar -> testar
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
import time
import json
import re


@dataclass
class SemanticProblemProfile:
    """Perfil semantico do problema extraido do nome/descricao."""
    has_diffusion: bool = False
    has_spatial: bool = False
    has_network: bool = False
    has_agents: bool = False
    has_multiscale: bool = False
    has_chemical: bool = False
    has_waves: bool = False
    has_population: bool = False
    has_signaling: bool = False
    has_mechanical: bool = False
    
    # Fontes de evidencia para cada propriedade
    evidence: Dict[str, str] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.has_diffusion, self.has_spatial, self.has_network,
            self.has_agents, self.has_multiscale, self.has_chemical,
            self.has_waves, self.has_population, self.has_signaling,
            self.has_mechanical,
        ], dtype=float)

    @classmethod
    def from_text(cls, name: str, domain: str, variables: list) -> 'SemanticProblemProfile':
        """Extrai perfil semantico do texto do problema."""
        p = cls()
        text = (name + " " + domain + " " + " ".join(variables)).lower()
        
        # Difusao / propagacao
        if any(w in text for w in ["diffus", "propagat", "spread", "wave", "travel", "migrat"]):
            p.has_diffusion = True
            p.evidence["diffusion"] = "text mentions propagation/diffusion"
        
        # Espacial
        if any(w in text for w in ["spatial", "space", "tissue", "cortical", "gradient", "position"]):
            p.has_spatial = True
            p.evidence["spatial"] = "text mentions spatial/tissue"
        
        # Rede / grafo
        if any(w in text for w in ["network", "connect", "synaptic", "pathway", "circuit", "graph"]):
            p.has_network = True
            p.evidence["network"] = "text mentions network/connectivity"
        
        # Agentes / celulas
        if any(w in text for w in ["cell", "agent", "microglia", "neuron", "bacteri", "individ"]):
            p.has_agents = True
            p.evidence["agents"] = "text mentions cellular/agent-based entities"
        
        # Multiescala
        if any(w in text for w in ["multiscale", "multi-scale", "hybrid", "tissue", "organ"]):
            p.has_multiscale = True
            p.evidence["multiscale"] = "text involves multiple scales"
        
        # Quimico
        if any(w in text for w in ["chemical", "kinetic", "reaction", "concentration", "bind", "molecular"]):
            p.has_chemical = True
            p.evidence["chemical"] = "text mentions chemical/molecular processes"
        
        # Ondas
        if any(w in text for w in ["wave", "oscillat", "rhythm", "cycle", "circadian", "burst"]):
            p.has_waves = True
            p.evidence["waves"] = "text mentions waves/oscillations"
        
        # Populacional
        if any(w in text for w in ["population", "species", "ecology", "growth", "decline", "dynamic"]):
            p.has_population = True
            p.evidence["population"] = "text mentions population dynamics"
        
        # Sinalizacao
        if any(w in text for w in ["signaling", "signal", "cascade", "pathway", "receptor", "ligand"]):
            p.has_signaling = True
            p.evidence["signaling"] = "text mentions signaling pathways"
        
        # Mecanico
        if any(w in text for w in ["mechanical", "force", "stress", "strain", "deformation", "pressure"]):
            p.has_mechanical = True
            p.evidence["mechanical"] = "text mentions mechanical forces"
        
        return p


# Mapeamento: perfil semantico -> formalismos recomendados
FORMALISM_RULES = [
    # (condition_function, formalism, priority_score)
    # Cada regra diz: "se o perfil tem X, recomende formalismo Y com peso Z"
]

def _recommend_ode(profile: SemanticProblemProfile) -> float:
    """ODE e bom para: quimico, populacional, sinalizacao. Ruim para: espacial, rede."""
    score = 0.3
    if profile.has_chemical: score += 0.3
    if profile.has_population: score += 0.3
    if profile.has_signaling: score += 0.3
    if profile.has_spatial: score -= 0.2
    if profile.has_network: score -= 0.1
    if profile.has_agents: score -= 0.1
    if profile.has_waves: score -= 0.1
    return max(0, min(1, score))


def _recommend_agent(profile: SemanticProblemProfile) -> float:
    """Agentes para: heterogeneidade, celulas, comportamento local."""
    score = 0.1
    if profile.has_agents: score += 0.4
    if profile.has_spatial and not profile.has_diffusion: score += 0.2
    if profile.has_network: score += 0.1
    if profile.has_multiscale: score += 0.1
    if profile.has_chemical and not profile.has_agents: score -= 0.2
    if profile.has_waves: score -= 0.1
    if profile.has_population and not profile.has_agents: score -= 0.1
    return max(0, min(1, score))


def _recommend_graph(profile: SemanticProblemProfile) -> float:
    """Grafo para: redes, conectividade, hubs, propagacao em rede."""
    score = 0.1
    if profile.has_network: score += 0.5
    if profile.has_diffusion: score += 0.2
    if profile.has_agents: score += 0.1
    if profile.has_multiscale: score += 0.1
    if profile.has_chemical and not profile.has_network: score -= 0.2
    if profile.has_population and not profile.has_network: score -= 0.1
    return max(0, min(1, score))


def _recommend_ca(profile: SemanticProblemProfile) -> float:
    """Automato celular para: grade regular, vizinhanca local, ondas."""
    score = 0.0
    if profile.has_waves: score += 0.35
    if profile.has_spatial: score += 0.25
    if profile.has_diffusion and not profile.has_chemical: score += 0.2
    if profile.has_agents: score += 0.1
    if profile.has_network: score -= 0.1
    if profile.has_chemical and not profile.has_spatial: score -= 0.2
    return max(0, min(1, score))


def _recommend_pde(profile: SemanticProblemProfile) -> float:
    """PDE para: difusao espacial, ondas, reacao-difusao, frentes."""
    score = 0.0
    if profile.has_diffusion: score += 0.4
    if profile.has_waves: score += 0.3
    if profile.has_spatial: score += 0.2
    if profile.has_chemical and profile.has_spatial: score += 0.2
    if profile.has_agents: score -= 0.1
    if profile.has_network: score -= 0.1
    if profile.has_population: score -= 0.1
    return max(0, min(1, score))


def _recommend_hybrid(profile: SemanticProblemProfile) -> float:
    """Hibrido para: multiescala, combina multiplos fenomenos."""
    score = 0.0
    if profile.has_multiscale: score += 0.4
    if profile.has_agents and profile.has_diffusion: score += 0.2
    if profile.has_network and profile.has_spatial: score += 0.2
    # Quanto mais propriedades, maior a chance de precisar de hibrido
    props = sum([profile.has_diffusion, profile.has_spatial, profile.has_network,
                 profile.has_agents, profile.has_chemical, profile.has_waves])
    if props >= 3: score += 0.2
    return max(0, min(1, score))


class MetaModelerV2:
    """MetaModeler que extrai propriedades semanticas ANTES de escolher formalismo."""

    FORMALISM_RECOMMENDERS = {
        "ode": _recommend_ode,
        "agent_based": _recommend_agent,
        "graph_network": _recommend_graph,
        "cellular_automaton": _recommend_ca,
        "pde": _recommend_pde,
        "hybrid": _recommend_hybrid,
    }

    def __init__(self):
        self.reasoning_chain: List[str] = []

    def analyze(self, name: str, domain: str, variables: list) -> SemanticProblemProfile:
        """Extrai perfil semantico do problema."""
        profile = SemanticProblemProfile.from_text(name, domain, variables)
        self.reasoning_chain.append(f"=== Problem: {name} ===")
        self.reasoning_chain.append(f"Properties extracted:")
        for prop, val in profile.__dict__.items():
            if isinstance(val, bool) and val:
                evidence = profile.evidence.get(prop, "")
                self.reasoning_chain.append(f"  - {prop}: {evidence}")
        return profile

    def recommend(self, profile: SemanticProblemProfile) -> List[Tuple[str, float, str]]:
        """Recomenda formalismos baseado no perfil semantico."""
        scored = []
        for name, recommender in self.FORMALISM_RECOMMENDERS.items():
            score = recommender(profile)
            reason = self._explain(name, profile)
            scored.append((name, score, reason))
        
        scored.sort(key=lambda x: -x[1])
        return scored

    def _explain(self, formalism: str, profile: SemanticProblemProfile) -> str:
        """Gera explicacao legivel de por que este formalismo foi recomendado."""
        explanations = {
            "ode": "ODE chosen: well-mixed continuous dynamics suitable for chemical/population/signaling processes",
            "agent_based": "Agent-Based chosen: captures heterogeneity of individual cells/agents with local rules",
            "graph_network": "Graph/Network chosen: captures connectivity and propagation through network structure",
            "cellular_automaton": "Cellular Automaton chosen: captures spatial wave dynamics on regular grid",
            "pde": "PDE chosen: captures spatiotemporal diffusion and wave propagation",
            "hybrid": "Hybrid chosen: multiscale problem requires multiple formalisms simultaneously",
        }
        return explanations.get(formalism, f"Recommended based on semantic profile")

    def formulate(self, name: str, domain: str, variables: list) -> dict:
        """Pipeline completa: extrair -> recomendar -> relatorio."""
        self.reasoning_chain = []
        
        # 1. Extrai perfil semantico
        profile = self.analyze(name, domain, variables)
        
        # 2. Recomenda formalismos
        recommendations = self.recommend(profile)
        
        # 3. Gera relatorio
        decision = {
            "name": name,
            "domain": domain,
            "profile": {
                "has_diffusion": profile.has_diffusion,
                "has_spatial": profile.has_spatial,
                "has_network": profile.has_network,
                "has_agents": profile.has_agents,
                "has_multiscale": profile.has_multiscale,
                "has_chemical": profile.has_chemical,
                "has_waves": profile.has_waves,
                "has_population": profile.has_population,
                "has_signaling": profile.has_signaling,
            },
            "recommendations": [(r[0], round(r[1], 2), r[2]) for r in recommendations],
            "top_formalism": recommendations[0][0],
            "top_score": round(recommendations[0][1], 2),
            "reasoning": self.reasoning_chain,
        }
        
        return decision


# ═══════════════════════════════════════════════════════════════
# TEST: compare with experts
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 20 problems from expert comparison
    PROBLEMS = [
        ("Amyloid plaque formation in Alzheimer", "Alzheimer", ["amyloid_beta", "cognitive_function"], "ode"),
        ("Tau propagation along neural connectome", "tau_propagation", ["tau_spread", "neuron_loss"], "graph_network"),
        ("Microglial immune response to amyloid", "immunology", ["microglial_activation", "cytokine_levels"], "agent_based"),
        ("Calcium wave propagation in astrocyte networks", "calcium_signaling", ["calcium_concentration", "wave_speed"], "pde"),
        ("Dopamine neuron degeneration in Parkinson", "Parkinson", ["dopamine_levels", "motor_function"], "ode"),
        ("Cortical spreading depression in migraine", "neurology", ["depolarization_wave", "recovery_time"], "cellular_automaton"),
        ("Tumor growth with angiogenesis", "oncology", ["tumor_volume", "vascular_density"], "hybrid"),
        ("Prion protein misfolding cascade", "prion", ["misfolded_protein", "aggregation_rate"], "ode"),
        ("EEG seizure propagation in epilepsy", "epilepsy", ["seizure_spread", "inhibition_loss"], "graph_network"),
        ("Circadian rhythm disruption in aging", "circadian", ["clock_gene_expression", "sleep_quality"], "ode"),
        ("Metastatic cell migration through tissue", "metastasis", ["migration_speed", "invasion_depth"], "agent_based"),
        ("Neurotransmitter diffusion at synapse", "synaptic", ["neurotransmitter_concentration", "receptor_binding"], "pde"),
        ("Insulin-glucose homeostasis", "metabolism", ["blood_glucose", "insulin_levels"], "ode"),
        ("Programmed cell death signaling", "apoptosis", ["caspase_activity", "cell_survival"], "ode"),
        ("Gene regulatory network in development", "gene_regulation", ["gene_expression", "transcription_rate"], "graph_network"),
        ("Population dynamics of gut microbiome", "microbiome", ["species_diversity", "total_bacteria"], "ode"),
        ("HIV viral dynamics with immune response", "HIV", ["viral_load", "t_cell_count"], "ode"),
        ("Cardiac action potential propagation", "cardiac", ["membrane_potential", "conduction_velocity"], "pde"),
        ("Synaptic plasticity and long-term potentiation", "plasticity", ["synaptic_strength", "ltp_magnitude"], "ode"),
        ("Immune checkpoint blockade in melanoma", "immunotherapy", ["tumor_size", "t_cell_infiltration"], "hybrid"),
    ]

    print("="*70)
    print("METAMODELER V2 — CONCEPTUAL FORMALISM SELECTION")
    print("="*70)
    
    mm = MetaModelerV2()
    matches = 0
    total = len(PROBLEMS)
    
    for name, domain, vars, expert in PROBLEMS:
        decision = mm.formulate(name, domain, vars)
        top = decision["top_formalism"]
        match = top == expert
        
        if match:
            matches += 1
        
        print(f"\n  {name[:50]:50s}")
        print(f"    Expert:        {expert:15s}  MetaModeler: {top:15s} {'[MATCH]' if match else '[DIFF]'}")
        print(f"    Profile:       ", end="")
        props = [k for k, v in decision["profile"].items() if v]
        print(", ".join(props) if props else "(none detected)")
        
        # Top 3 recommendations
        for form, score, reason in decision["recommendations"][:3]:
            print(f"    {form:20s} score={score:.2f}  {reason[:50]}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Total: {total}")
    print(f"  Matches: {matches} ({matches/total*100:.0f}%)")
    
    # Compare with v1 (15%)
    print(f"\n  MetaModeler v1: 15%")
    print(f"  MetaModeler v2: {matches/total*100:.0f}%")
    print(f"  Improvement: {matches/total*100 - 15:.0f} percentage points")
