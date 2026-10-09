"""
DynamicTypeLibrary — Biblioteca de TIPOS DE DINAMICA, nao apenas ODEs.

Expande a AGI para usar:
  - ODE (sistemas de equacoes diferenciais ordinarias) — ja temos
  - PDE (equacoes diferenciais parciais) — reacao-difusao, ondas
  - Agent-based (celulas como agentes individuais)
  - Grafo/Network (propagacao em redes)
  - Regras locais (automato celular, Life-like)
  - Histerese (memoria de estado)
  - Delay (atraso temporal)
  - Sistemas hibridos (combina multiplos formalismos)

Cada tipo resolve uma CLASSE de problemas que ODE nao captura bem.
A AGI aprende qual tipo usar baseado no comportamento observado.
"""
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import random
import math


@dataclass
class DynamicType:
    """Um tipo de dinamica matematica."""
    name: str
    formalism: str  # "ode", "pde", "agent", "graph", "cellular_automaton", "delay", "hybrid"
    description: str
    
    # Para que problemas este tipo e adequado?
    best_for: List[str] = field(default_factory=list)
    
    # Exemplos de uso em biologia
    biological_examples: List[str] = field(default_factory=list)
    
    # Se este tipo foi usado com sucesso pela AGI
    times_used: int = 0
    success_rate: float = 0.0


class DynamicTypeLibrary:
    """Catalogo de tipos de dinamica que a AGI pode usar.

    Uso:
        lib = DynamicTypeLibrary()
        best = lib.recommend(problem_description="tau propagation")
        model = lib.create_model("reaction_diffusion", params)
    """

    def __init__(self):
        self.types: Dict[str, DynamicType] = {}
        self._build()
    
    def _build(self):
        # --- PDE: Reacao-Difusao ---
        self.types["reaction_diffusion"] = DynamicType(
            "Reaction-Diffusion (PDE)", "pde",
            "Sistema de EDPs: uma substancia difunde e reage. Gera padroes espaciais (Turing).",
            best_for=["protein propagation", "spatial spreading", "pattern formation",
                      "prion-like spread", "wave propagation"],
            biological_examples=["Tau propagacao prion-like", "Ondas de despolarizacao",
                                 "Frente de inflamacao", "Difusao de calcium"],
        )
        
        # --- AGENT-BASED ---
        self.types["agent_based"] = DynamicType(
            "Agent-Based Model", "agent",
            "Cada celula/neuronio e um agente individual com regras locais. Comportamento emerge.",
            best_for=["microglial behavior", "cell-cell interaction", "immune response",
                      "synaptic pruning", "network formation"],
            biological_examples=["Microglia varrendo sinaptos", "Astrocitos regulando neurotransmissores",
                                 "Celulas imunes infiltram cerebro", "Neuronios competem por BDNF"],
        )
        
        # --- GRAFO ---
        self.types["graph_network"] = DynamicType(
            "Graph/Network Model", "graph",
            "Sistema modelado como grafo: nos = neuronios/regioes, edges = conexoes. Propagacao em rede.",
            best_for=["network dysfunction", "connectivity loss", "information flow",
                      "hub vulnerability", "cascading failure"],
            biological_examples=["Rede de modo default (DMN) fragmentada", "Hub de conectividade (precuneus) atrofiado",
                                 "Propagacao de tau segue conectoma", "Perda de eficiencia de rede global"],
        )
        
        # --- AUTOMATO CELULAR ---
        self.types["cellular_automaton"] = DynamicType(
            "Cellular Automaton", "cellular_automaton",
            "Grid de celulas com regras locais deterministas. Complexidade emerge de regras simples.",
            best_for=["excitability waves", "spreading depression", "cortical spreading",
                      "seizure propagation", "pattern formation"],
            biological_examples=["Depressao alastrante cortical", "Aura de enxaqueca",
                                 "Propagacao de crise epileptica", "Ondas de atividade em desenvolvimento"],
        )
        
        # --- DELAY ---
        self.types["delay_differential"] = DynamicType(
            "Delay Differential Equation (DDE)", "delay",
            "ODE com atraso temporal: a taxa de mudanca depende do estado passado.",
            best_for=["feedback loops with delay", "circadian rhythms", "memory effects",
                      "slow-fast dynamics", "oscillation with lag"],
            biological_examples=["Ciclo sono-vigilia com atraso", "Clearance glinfatica com delay",
                                 "Feedback transcriptional com atraso", "Sinalizacao lenta entre neuronios"],
        )
        
        # --- HISTERESE ---
        self.types["hysteresis_system"] = DynamicType(
            "Hysteresis System", "hybrid",
            "Sistema com memoria de estado: o caminho de ida e diferente do de volta.",
            best_for=["irreversible transitions", "memory of damage", "threshold crossing",
                      "bistable switches", "snap-back dynamics"],
            biological_examples=["Acumulo de amyloid: clearance muito mais lento que deposicao",
                                 "Perda de cognicao: declinio repentino seguido de estabilizacao",
                                 "Cascata inflamatoria que nao desliga", "Transicao para estado doente"],
        )
        
        # --- MULTI-ESCALA (HIBRIDO) ---
        self.types["multiscale_hybrid"] = DynamicType(
            "Multiscale Hybrid", "hybrid",
            "Combina ODE para moleculas + ABM para celulas + PDE para difusao em uma simulacao integrada.",
            best_for=["complex diseases with multiple scales", "drug delivery and response",
                      "whole-organ simulation", "brain-wide dynamics"],
            biological_examples=["Alzheimer: amyloid molecular, microglial cellular, brain network scale",
                                 "Cancer: oncogenes (molecular), tumor growth (cellular), metastasis (tissue)"],
        )
    
    def recommend(self, problem_description: str) -> List[Tuple[str, float]]:
        """Recomenda tipos de dinamica para um problema."""
        desc = problem_description.lower()
        scores = []
        
        for name, dt in self.types.items():
            score = 0.0
            for keyword in dt.best_for:
                if keyword.lower() in desc:
                    score += 0.3
            for example in dt.biological_examples:
                for word in example.lower().split():
                    if word in desc and len(word) > 4:
                        score += 0.1
            if dt.success_rate > 0:
                score += dt.success_rate * 0.2
            scores.append((name, min(1.0, score)))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:3]
    
    def create_model(self, dynamic_type: str, params: dict = None) -> Optional[Callable]:
        """Cria uma funcao de dinamica do tipo especificado.

        Retorna: funcao f(t, y) -> dy/dt (interface compativel com ODE solver)
        """
        params = params or {}
        
        if dynamic_type == "reaction_diffusion":
            return self._create_reaction_diffusion(params)
        elif dynamic_type == "delay_differential":
            return self._create_delay(params)
        elif dynamic_type == "hysteresis_system":
            return self._create_hysteresis(params)
        elif dynamic_type == "graph_network":
            return self._create_graph(params)
        elif dynamic_type == "cellular_automaton":
            return self._create_cellular_automaton(params)
        elif dynamic_type == "agent_based":
            return self._create_agent_based(params)
        
        return None
    
    def _create_reaction_diffusion(self, params: dict) -> Callable:
        """Modelo de reacao-difusao (Turing-like).
        
        Para simulacao de propagacao de tau/amyloid como frente de onda.
        """
        D = params.get("diffusion_rate", 0.1)
        rho = params.get("growth_rate", 0.05)
        K = params.get("carrying_capacity", 1.0)
        n_vars = params.get("n_vars", 10)
        
        def ode_rd(t, y):
            dy = np.zeros_like(y)
            # Termo de reacao logistica
            for i in range(n_vars):
                dy[i] = rho * y[i] * (1 - y[i]/K)
                # Termo de difusao (acoplamento com vizinhos)
                if i > 0:
                    dy[i] += D * (y[i-1] - y[i])
                if i < n_vars - 1:
                    dy[i] += D * (y[i+1] - y[i])
            return dy
        
        return ode_rd
    
    def _create_delay(self, params: dict) -> Callable:
        """Modelo com atraso temporal aproximado por ODE.
        
        Usa uma variavel extra para simular o atraso (aproximacao de primeira ordem).
        """
        tau = params.get("delay", 5.0)
        gain = params.get("feedback_gain", 0.5)
        
        def ode_delay(t, y):
            dy = np.zeros_like(y)
            if len(y) >= 2:
                # Variavel 0: dinâmica principal
                dy[0] = -y[0]/tau + gain * y[1]
                # Variavel 1: versao atrasada (filtro passa-baixa)
                dy[1] = (y[0] - y[1]) / tau
            return dy
        
        return ode_delay
    
    def _create_hysteresis(self, params: dict) -> Callable:
        """Modelo com histerese (memoria de estado).
        
        Usa funcao tangente hiperbolica com threshold variavel.
        """
        beta = params.get("steepness", 5.0)
        threshold = params.get("threshold", 0.3)
        
        def ode_hysteresis(t, y):
            dy = np.zeros_like(y)
            if len(y) >= 2:
                # Variavel 0: estado lento (ex: amyloid acumulado)
                input_slow = y[1] - threshold
                dy[0] = 0.01 * (np.tanh(beta * input_slow) - y[0])
                # Variavel 1: entrada rapida (ex: producao de amyloid)
                dy[1] = -0.05 * y[1] + 0.02 * (1 - y[0])
            return dy
        
        return ode_hysteresis
    
    def _create_graph(self, params: dict) -> Callable:
        """Modelo em rede (grafo de regioes cerebrais)."""
        n_nodes = params.get("n_nodes", 10)
        connectivity = params.get("connectivity", 0.3)
        coupling = params.get("coupling", 0.1)
        
        if "adjacency_matrix" not in params:
            # Matriz de adjacencia aleatoria (small-world-like)
            W = np.random.RandomState(42).rand(n_nodes, n_nodes) * (np.random.RandomState(42).rand(n_nodes, n_nodes) < connectivity)
            np.fill_diagonal(W, 0)
            W = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        else:
            W = params["adjacency_matrix"]
            n_nodes = W.shape[0]
        
        def ode_graph(t, y):
            dy = np.zeros_like(y)
            for i in range(min(n_nodes, len(y))):
                # Dinamica local (logistica)
                dy[i] = 0.1 * y[i] * (1 - y[i])
                # Acoplamento com vizinhos
                for j in range(min(n_nodes, len(y))):
                    if i != j:
                        dy[i] += coupling * W[i, j] * y[j]
            return dy
        
        return ode_graph
    
    def _create_cellular_automaton(self, params: dict) -> Callable:
        """Automato celular continuo (aproximacao ODE).
        
        Cada celula interage com vizinhos. Regras locais geram comportamento global.
        """
        n_cells = params.get("n_cells", 10)
        rule_type = params.get("rule", "excitable")
        
        def ode_ca(t, y):
            dy = np.zeros_like(y)
            for i in range(min(n_cells, len(y))):
                left = y[i-1] if i > 0 else 0
                right = y[i+1] if i < n_cells - 1 else 0
                neighbors = (left + right) / 2
                
                if rule_type == "excitable":
                    # Celula excita se vizinhos estao acima de threshold
                    dy[i] = -0.1 * y[i] + 0.5 * (neighbors > 0.3) * (1 - y[i])
                elif rule_type == "inhibitory":
                    # Vizinhos inibem esta celula
                    dy[i] = 0.05 * (1 - y[i]) - 0.2 * neighbors * y[i]
                else:
                    dy[i] = -0.1 * y[i] + 0.3 * neighbors
            return dy
        
        return ode_ca
    
    def _create_agent_based(self, params: dict) -> Callable:
        """Modelo baseado em agentes (aproximacao ODE populacional).
        
        Cada grupo de agentes e uma variavel ODE (abordagem de campo medio).
        """
        n_groups = params.get("n_groups", 5)
        
        def ode_agents(t, y):
            dy = np.zeros_like(y)
            for i in range(min(n_groups, len(y))):
                # Cada grupo tem dinamica propria
                # Interacao entre grupos 
                interaction = 0
                for j in range(min(n_groups, len(y))):
                    if i != j:
                        interaction += 0.01 * y[i] * y[j]  # encontro entre agentes
                dy[i] = 0.05 * y[i] * (1 - y[i]) + interaction - 0.02 * y[i]
            return dy
        
        return ode_agents
    
    def get_stats(self) -> dict:
        return {
            "available_types": list(self.types.keys()),
            "formalisms": list(set(t.formalism for t in self.types.values())),
            "total_types": len(self.types),
        }
