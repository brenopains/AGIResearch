"""
StructuralPlasticity — Auto-Reconfiguracao Arquitetural
==========================================================
Permite que o sistema crie, remova e reconecte modulos em runtime.

Inspirado na neurogenese: o cerebro nao apenas ajusta synapses,
mas cria novos neuronios, poda os inuteis, e reorganiza circuitos.

Aqui, "modulos" sao componentes cognitivos (CGE, WorldModel,
SimulationEngine, SelfModel, etc.) e "conexoes" sao canais
de comunicacao (leitura/escrita no MentalState).

Operacoes:
  spawn: cria nova instancia de um modulo a partir de uma spec
  prune: remove modulo inativo
  rewire: redireciona fluxo entre modulos
  merge: funde dois modulos redundantes
  split: divide um modulo em especialistas
"""
import time
import uuid
import copy
import random
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ModuleSpec:
    """
    Especificacao de um modulo cognitivo.
    
    Pode ser instanciado para criar um modulo real.
    """
    name: str = ""
    module_type: str = ""  # "cge", "world_model", "simulation", etc.
    version: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    input_contracts: List[str] = field(default_factory=list)
    output_contracts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.module_type,
            "version": self.version,
            "config_keys": list(self.config.keys()),
            "inputs": self.input_contracts,
            "outputs": self.output_contracts,
        }


@dataclass
class ArchitectureNode:
    """Um no no grafo da arquitetura — representa um modulo ativo."""
    id: str = ""
    name: str = ""
    module_type: str = ""
    instance: Any = None
    spec: Optional[ModuleSpec] = None
    created_at: float = 0.0
    last_active: float = 0.0
    activation_count: int = 0
    input_weight: float = 1.0
    output_weight: float = 1.0
    health: float = 1.0  # 0 = morto, 1 = saudavel
    error_rate: float = 0.0
    tags: Set[str] = field(default_factory=set)


@dataclass
class ArchitectureEdge:
    """Uma conexao entre dois modulos — fluxo de dados."""
    from_id: str = ""
    to_id: str = ""
    bandwidth: float = 1.0
    n_transfers: int = 0
    last_used: float = 0.0


class ArchitectureGraph:
    """
    Grafo da arquitetura cognitiva.
    
    Nos = modulos. Arestas = fluxo de dados.
    Isto e uma representacao EXPLICITA de algo que hoje
    esta IMPLICITO no codigo (mind.py conecta modulos
    manualmente no __init__).
    """

    def __init__(self):
        self.nodes: Dict[str, ArchitectureNode] = {}
        self.edges: List[ArchitectureEdge] = []
        self._node_counter = 0

    def add_node(self, name: str, module_type: str, instance: Any = None,
                 spec: Optional[ModuleSpec] = None) -> str:
        """Adiciona um no ao grafo."""
        node_id = f"mod_{self._node_counter:04d}"
        self._node_counter += 1
        node = ArchitectureNode(
            id=node_id, name=name, module_type=module_type,
            instance=instance, spec=spec,
            created_at=time.time(), last_active=time.time(),
        )
        self.nodes[node_id] = node
        return node_id

    def remove_node(self, node_id: str) -> bool:
        """Remove um no e suas arestas."""
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self.edges = [e for e in self.edges
                     if e.from_id != node_id and e.to_id != node_id]
        return True

    def add_edge(self, from_id: str, to_id: str,
                 bandwidth: float = 1.0) -> bool:
        """Adiciona aresta entre dois nos."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        self.edges.append(ArchitectureEdge(
            from_id=from_id, to_id=to_id, bandwidth=bandwidth,
            last_used=time.time()
        ))
        return True

    def get_connections(self, node_id: str) -> Tuple[List[str], List[str]]:
        """Retorna (inputs, outputs) de um no."""
        inputs = [e.from_id for e in self.edges if e.to_id == node_id]
        outputs = [e.to_id for e in self.edges if e.from_id == node_id]
        return inputs, outputs

    def get_flow_path(self, start_type: str, end_type: str) -> Optional[List[str]]:
        """Encontra caminho entre tipos de modulo."""
        start_nodes = [nid for nid, n in self.nodes.items()
                      if n.module_type == start_type]
        end_nodes = [nid for nid, n in self.nodes.items()
                    if n.module_type == end_type]
        if not start_nodes or not end_nodes:
            return None
        
        from collections import deque
        for start in start_nodes:
            visited = {start: None}
            q = deque([start])
            while q:
                v = q.popleft()
                if v in end_nodes:
                    path = []
                    while v is not None:
                        path.append(v)
                        v = visited[v]
                    return list(reversed(path))
                for e in self.edges:
                    if e.from_id == v and e.to_id not in visited:
                        visited[e.to_id] = v
                        q.append(e.to_id)
        return None

    def get_state(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "nodes": {nid: {"name": n.name, "type": n.module_type,
                            "health": n.health, "error_rate": n.error_rate,
                            "age": time.time() - n.created_at}
                     for nid, n in self.nodes.items()},
        }


class StructuralPlasticity:
    """
    Motor de plasticidade estrutural.
    
    Monitora a arquitetura, detecta oportunidades de:
    - Spawn: modulo novo quando surge nova demanda
    - Prune: modulo inativo ou com alta taxa de erro
    - Rewire: fluxo subotimo entre modulos
    - Merge: dois modulos com funcao similar e baixa divergencia
    - Split: modulo com carga alta demais
    
    Integra com MetaEvolution (parametros) e ArchitecturalAudit (geracao).
    """

    def __init__(self, architecture: Optional[ArchitectureGraph] = None,
                 mental_state=None, verbose: bool = False):
        self.arch = architecture or ArchitectureGraph()
        self.mental_state = mental_state
        self.verbose = verbose
        
        self._spawn_fn: Dict[str, Callable] = {}
        self._adaptation_history: List[dict] = []
        self._max_history = 200
        self._cycle = 0
        
        self._n_spawned = 0
        self._n_pruned = 0
        self._n_merged = 0

    def register_spawn_fn(self, module_type: str,
                           factory_fn: Callable) -> None:
        """Registra funcao que cria novos modulos de um tipo."""
        self._spawn_fn[module_type] = factory_fn

    def tick(self) -> List[dict]:
        """Um ciclo de monitoramento e possivel adaptacao."""
        self._cycle += 1
        actions = []
        
        for node_id, node in list(self.arch.nodes.items()):
            node.last_active = time.time()
            
            # Prune: modulo com saude baixa ou erro alto
            if node.health < 0.2 or node.error_rate > 0.8:
                if self._prune_node(node_id):
                    actions.append({"action": "prune", "node": node_id,
                                    "reason": "low_health" if node.health < 0.2 else "high_error"})
                    if self.verbose:
                        print(f"  [Plasticity] Pruned {node.name} ({node.module_type})")
            
            # Split: modulo sobrecarregado
            if node.activation_count > 100 and node.error_rate > 0.5:
                self._split_node(node_id)
                actions.append({"action": "split", "node": node_id})
                if self.verbose:
                    print(f"  [Plasticity] Split {node.name} into specialists")
        
        # Spawn: modulo novo para gaps detectados
        if self._cycle % 10 == 0:
            spawn = self._detect_spawn_opportunity()
            if spawn:
                actions.append(spawn)
        
        # Merge: redundancia
        if self._cycle % 20 == 0:
            merge = self._detect_merge_opportunity()
            if merge:
                actions.append(merge)
        
        if actions:
            self._adaptation_history.extend(actions)
            if len(self._adaptation_history) > self._max_history:
                self._adaptation_history = self._adaptation_history[-self._max_history:]
        
        return actions

    def register_module(self, name: str, module_type: str,
                        instance: Any = None, spec: Optional[ModuleSpec] = None) -> str:
        """Registra um modulo existente no grafo."""
        return self.arch.add_node(name, module_type, instance, spec)

    def record_error(self, node_id: str) -> None:
        """Registra um erro em um modulo."""
        if node_id in self.arch.nodes:
            node = self.arch.nodes[node_id]
            node.error_rate = min(1.0, node.error_rate + 0.05)
            node.activation_count += 1

    def record_success(self, node_id: str) -> None:
        """Registra sucesso de um modulo."""
        if node_id in self.arch.nodes:
            node = self.arch.nodes[node_id]
            node.error_rate = max(0.0, node.error_rate - 0.03)
            node.activation_count += 1
            node.health = min(1.0, node.health + 0.01)

    def connect(self, from_name: str, to_name: str,
                bandwidth: float = 1.0) -> bool:
        """Conecta dois modulos registrados."""
        from_node = next((nid for nid, n in self.arch.nodes.items()
                         if n.name == from_name), None)
        to_node = next((nid for nid, n in self.arch.nodes.items()
                       if n.name == to_name), None)
        if from_node and to_node:
            return self.arch.add_edge(from_node, to_node, bandwidth)
        return False

    def get_module(self, name: str) -> Optional[Any]:
        """Retorna instancia de modulo por nome."""
        for node in self.arch.nodes.values():
            if node.name == name:
                return node.instance
        return None

    # ─── Operacoes Estruturais Privadas ───

    def _prune_node(self, node_id: str) -> bool:
        """Remove modulo inativo."""
        if node_id not in self.arch.nodes:
            return False
        self.arch.remove_node(node_id)
        self._n_pruned += 1
        return True

    def _split_node(self, node_id: str) -> Optional[str]:
        """Divide modulo em dois especialistas."""
        if node_id not in self.arch.nodes:
            return None
        original = self.arch.nodes[node_id]
        child_id = self.arch.add_node(
            f"{original.name}_child",
            original.module_type,
            instance=copy.deepcopy(original.instance) if original.instance else None,
        )
        self._n_spawned += 1
        return child_id

    def _detect_spawn_opportunity(self) -> Optional[dict]:
        """
        Detecta se falta um modulo para alguma demanda.
        
        Verifica no MentalState se ha requisicoes nao atendidas.
        """
        if self.mental_state is None:
            return None
        
        try:
            if hasattr(self.mental_state, 'blackboard'):
                bb = self.mental_state.blackboard
                if "causal_gap" in bb:
                    if "CausalCore" not in [n.name for n in self.arch.nodes.values()]:
                        return {"action": "spawn", "type": "causal_core",
                                "reason": "causal_gap_detected"}
            return None
        except Exception:
            return None

    def _detect_merge_opportunity(self) -> Optional[dict]:
        """Detecta modulos redundantes."""
        types = defaultdict(list)
        for node in self.arch.nodes.values():
            types[node.module_type].append(node.id)
        
        for mod_type, nodes in types.items():
            if len(nodes) > 2:
                return {"action": "merge", "type": mod_type,
                        "nodes": nodes[:2], "reason": "redundant"}
        return None

    def get_stats(self) -> dict:
        return {
            "n_nodes": len(self.arch.nodes),
            "n_edges": len(self.arch.edges),
            "n_spawned": self._n_spawned,
            "n_pruned": self._n_pruned,
            "n_merged": self._n_merged,
            "n_adaptations": len(self._adaptation_history),
            "cycle": self._cycle,
        }
