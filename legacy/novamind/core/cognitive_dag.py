"""
CognitiveDAG — Pipeline de Cognicao como Grafo DIRECIONADO ACICLICO

O pipeline inteiro vira um DAG manipulavel.
Nao mais:
  A → B → C → D (fixo)

Mas sim:
  Observe → [Attention, Compare] → Predict → [Simulate, Constraint] → Theory

Diferentes DAGs para diferentes dominios.
A ForjaEvolution descobre e otimiza DAGs.
"""
import time, uuid, hashlib, random
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CognitiveStep:
    """Um passo cognitivo (no do DAG)."""
    id: str = ""
    name: str = ""
    description: str = ""
    module: str = ""         # "perception", "cge", "workspace", "simulation", "forge", etc.
    function: str = ""       # "learn", "compute_errors", "post_to_workspace", etc.
    params: Dict = field(default_factory=dict)

    # Metadados
    cost: float = 0.3       # custo computacional relativo
    priority: float = 0.5   # prioridade (maior = executa primeiro em paralelo)
    timeout: float = 5.0    # segundos maximos
    required: bool = True   # se falhar, para o DAG?

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name, "module": self.module,
                "function": self.function, "cost": self.cost}


@dataclass
class CognitiveDAG:
    """Um pipeline de cognicao completo.

    E um DAG onde:
      - nodes: CognitiveStep
      - edges: (from_id, to_id) — fluxo de dados
      - entry_points: nodes que comecam
      - exit_points: nodes que terminam

    A ForjaEvolution pode:
      - Adicionar/remover nos
      - Reordenar edges
      - Criar variantes de DAG para cada dominio
    """
    id: str = ""
    name: str = ""
    domain: str = "general"
    nodes: Dict[str, CognitiveStep] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    confidence: float = 0.5
    n_executions: int = 0
    avg_reward: float = 0.0

    # Gating Field: pesos continuos para cada aresta
    # Neutro = 0.5, aberto = 1.0, fechado = 0.0
    gate_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def add_node(self, step: CognitiveStep):
        self.nodes[step.id] = step

    def add_edge(self, from_id: str, to_id: str):
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id))

    def get_topological_order(self) -> List[CognitiveStep]:
        """Ordenacao topologica simples do DAG."""
        in_degree = defaultdict(int)
        for f, t in self.edges:
            in_degree[t] += 1

        ready = [nid for nid in self.entry_points if nid in self.nodes]
        result = []
        while ready:
            # Ordenar por prioridade
            ready.sort(key=lambda nid: -self.nodes[nid].priority)
            nid = ready.pop(0)
            result.append(self.nodes[nid])
            for f, t in self.edges:
                if f == nid:
                    in_degree[t] -= 1
                    if in_degree[t] <= 0 and t not in [r.id for r in result]:
                        ready.append(t)
        return result

    def apply_gating(self, neurofield_state: Dict = None,
                     temperature: float = 0.5) -> List[CognitiveStep]:
        """Aplica Gating Field: seleciona passos baseado em neuromodulacao.

        Em vez de executar o DAG inteiro, calcula quais portas (edges)
        estao abertas baseado na temperatura do campo.

        Args:
            neurofield_state: estado do NeuroField com gating_temperature,
                              exploration_drive, focus_intensity
            temperature: fallback se neurofield_state nao fornecido

        Returns:
            Lista de CognitiveSteps ativos (gated)
        """
        if neurofield_state:
            temperature = neurofield_state.get("gating_temperature", temperature)

        # Ordenacao topologica base
        all_steps = self.get_topological_order()

        # Se temperatura alta, mantem mais passos (exploracao)
        # Se temperatura baixa, filtra passos de baixa prioridade (foco)
        if temperature > 0.7:
            # Modo exploratorio: mantem tudo
            return all_steps
        elif temperature < 0.3:
            # Modo focado: so os mais prioritarios
            threshold = 0.7
            return [s for s in all_steps if s.priority >= threshold]
        else:
            # Modo intermediario: gate baseado em prioridade + temperatura
            threshold = 0.5 - (temperature - 0.5) * 0.5  # 0.5 quando temp=0.5
            return [s for s in all_steps if s.priority >= threshold or
                    any(s.id in e for e in self.edges)]

    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name, "domain": self.domain,
                "n_nodes": len(self.nodes), "n_edges": len(self.edges),
                "topology": [s.name for s in self.get_topological_order()]}


# ─── DAG PRIMORDIAL (o ciclo cognitivo padrao) ───────────

def _build_primordial_dag() -> CognitiveDAG:
    """Constroi o DAG primordial — o ciclo cognitivo original."""
    dag = CognitiveDAG(id="primordial", name="Ciclo Cognitivo Padrao")

    steps = [
        CognitiveStep("select_topic", "Selecionar Topico", "Escolhe o que estudar",
                     "curiosity", "select_topic", {"mode": "current"}, priority=1.0),
        CognitiveStep("start_trace", "Iniciar Trace", "Inicia registro cognitivo",
                     "cognitive_trace", "start_trace", priority=0.9),
        CognitiveStep("learn", "Aprender", "Busca e processa conhecimento",
                     "perception", "learn", {"source": "wikipedia"}, cost=0.8),
        CognitiveStep("compute_errors", "Computar Erros", "Calcula sinais de erro",
                     "error_signal", "compute", cost=0.2),
        CognitiveStep("post_workspace", "Postar no Workspace", "Publica no espaco de trabalho",
                     "workspace", "post", priority=0.7),
        CognitiveStep("tick_workspace", "Decair Workspace", "Decaimento natural",
                     "workspace", "tick", cost=0.1),
        CognitiveStep("meta_reward", "Computar Recompensa", "Recompensa para MetaController",
                     "meta_controller", "compute_reward", cost=0.1),
        CognitiveStep("meta_tick", "Decidir Proximo Modo", "MetaController escolhe modo",
                     "meta_controller", "tick", priority=0.8),
        CognitiveStep("stance", "Aplicar Stance", "Seleciona stance para contexto",
                     "stances", "apply", cost=0.2),
        CognitiveStep("record_trace", "Registrar Trace", "Salva passo no CognitiveTrace",
                     "cognitive_trace", "step", priority=0.6),
        CognitiveStep("build_scene", "Construir Cena Mental", "MentalSceneGraph do estado",
                     "mental_scene_graph", "from_cge", cost=0.3),
        CognitiveStep("notebook", "Escrever no Caderno", "Registra aprendizado",
                     "notebook", "write", cost=0.2),
        CognitiveStep("forja_mine", "Forja: Minerar Traces", "ForjaEvolution analisa traces",
                     "forja_evolution", "mine", cost=0.7),
        CognitiveStep("archive", "Arquivar", "Armazena no CognitiveArchive",
                     "archive", "store", cost=0.3),
        CognitiveStep("bridge_update", "Atualizar Frontend", "Envia estado para UI",
                     "bridge", "update", priority=0.5),
        CognitiveStep("auto_save", "Auto-Salvar", "Persiste o cerebro",
                     "persistence", "save", cost=0.6),
        CognitiveStep("idle", "Modo Inativo", "Pausa curta entre ciclos",
                     "system", "sleep", cost=0.05, priority=0.0),
    ]
    for s in steps:
        dag.add_node(s)

    # Edges: fluxo principal
    edges = [
        ("select_topic", "start_trace"),
        ("start_trace", "learn"),
        ("learn", "compute_errors"),
        ("compute_errors", "post_workspace"),
        ("post_workspace", "tick_workspace"),
        ("tick_workspace", "meta_reward"),
        ("meta_reward", "meta_tick"),
        ("meta_tick", "stance"),
        ("stance", "record_trace"),
        ("record_trace", "build_scene"),
        ("build_scene", "notebook"),
        ("notebook", "forja_mine"),
        ("forja_mine", "archive"),
        ("archive", "bridge_update"),
        ("bridge_update", "auto_save"),
        ("auto_save", "select_topic"),
    ]
    for f, t in edges:
        dag.add_edge(f, t)

    dag.entry_points = ["select_topic"]
    dag.exit_points = ["auto_save", "idle"]
    return dag


PRIMORDIAL_DAGS = {
    "general": _build_primordial_dag(),
}


class DAGRegistry:
    """Registro DINAMICO de DAGs cognitivos.

    Diferentes dominios podem ter DAGs diferentes.
    A ForjaEvolution descobre novos DAGs e os registra aqui.
    """

    def __init__(self):
        self.dags: Dict[str, CognitiveDAG] = {}
        self._active_dag_id: str = "general"
        self._execution_history: List[Dict] = []

        for name, dag in PRIMORDIAL_DAGS.items():
            self.dags[name] = dag

    def register(self, dag: CognitiveDAG) -> str:
        """Registra um novo DAG (descoberto pela Forja)."""
        self.dags[dag.id] = dag
        return dag.id

    def set_active(self, domain: str = "general") -> bool:
        """Ativa o DAG mais adequado para um dominio."""
        # Procurar DAG especifico do dominio
        for dag in self.dags.values():
            if dag.domain == domain:
                self._active_dag_id = dag.id
                return True
        # Fallback para o primordial
        self._active_dag_id = "general"
        return True

    def get_active(self) -> CognitiveDAG:
        return self.dags.get(self._active_dag_id, self.dags["general"])

    def create_variant(self, base_id: str = "general",
                       domain: str = "") -> CognitiveDAG:
        """Cria uma variante de um DAG existente (mutacao)."""
        base = self.dags.get(base_id, self.dags["general"])
        new_id = f"dag_{uuid.uuid4().hex[:8]}"
        new_dag = CognitiveDAG(
            id=new_id,
            name=f"Variante de {base.name} para {domain or 'novo dominio'}",
            domain=domain or base.domain,
            nodes=dict(base.nodes),
            edges=list(base.edges),
            entry_points=list(base.entry_points),
            exit_points=list(base.exit_points),
            confidence=base.confidence * 0.8,
        )
        # Mutacao: embaralhar 1-2 edges
        if len(new_dag.edges) > 2:
            idx = random.randint(0, len(new_dag.edges) - 1)
            f, t = new_dag.edges[idx]
            # Trocar destino para outro no aleatorio
            targets = [nid for nid in new_dag.nodes if nid != f]
            new_t = random.choice(targets)
            new_dag.edges[idx] = (f, new_t)
        self.register(new_dag)
        return new_dag

    def get_state(self) -> Dict:
        active = self.get_active()
        return {
            "n_dags": len(self.dags),
            "active": active.name,
            "active_topology": [s.name for s in active.get_topological_order()],
            "domains": list(set(d.domain for d in self.dags.values())),
        }


DAG_REGISTRY = DAGRegistry()
