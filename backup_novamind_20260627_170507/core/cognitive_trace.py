"""
CognitiveTrace — Registro do Raciocinio para Mineracao pela Forja

Cada ciclo importante salva:
  Problema → Workspace → Operadores → Forja → Resultado

A Forja minera esses rastros para descobrir meta-padroes:
  - "Sempre que e geometria, usar rotacao antes de comparacao"
  - "Hipoteses apos exploracao ampla sobrevivem mais"
  - "Em problemas fisicos, simular antes de abstrair reduz erros"
"""
import time, uuid, json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TraceStep:
    mode: str = ""
    context: str = ""
    operators: List[str] = field(default_factory=list)
    workspace_content: List[str] = field(default_factory=list)
    concepts_involved: List[str] = field(default_factory=list)
    novelty_achieved: int = 0
    edges_created: int = 0
    outcome: str = ""
    reward: float = 0.0
    error_snapshot: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CognitiveTrace:
    """Registro completo de um ciclo de raciocinio."""

    def __init__(self, problem: str = ""):
        self.id = str(uuid.uuid4())[:8]
        self.problem = problem
        self.steps: List[TraceStep] = []
        self.final_answer: str = ""
        self.success: bool = False
        self.created_at: float = time.time()
        self.cge_nodes_before: int = 0
        self.cge_nodes_after: int = 0
        self.workspace_snapshot: List[str] = []

    def add_step(self, step: TraceStep):
        self.steps.append(step)

    def close(self, answer: str = "", success: bool = False):
        self.final_answer = answer
        self.success = success

    def store_in_cge(self, cge):
        """Armazena trace como nos especiais no CGE para a Forja minerar."""
        summary = self.summarize()
        trace_id = f"_trace_{self.id}"

        # Vetor HDV determinístico do trace
        import hashlib, torch
        h = hashlib.sha256(trace_id.encode()).hexdigest()
        g = torch.Generator()
        g.manual_seed(int(h[:8], 16))
        trace_vec = torch.sign(torch.randn(10000, generator=g))
        cge.create_node(trace_id, trace_vec)
        if trace_id in cge.nodes:
            cge.nodes[trace_id]["trace_data"] = {
                "problem": self.problem,
                "steps": len(self.steps),
                "success": self.success,
                "avg_reward": self.avg_reward,
                "final_answer": self.final_answer[:200],
            }
            cge.nodes[trace_id]["emotional_valence"] = 0.5 if self.success else 0.2
            cge.nodes[trace_id]["emotional_arousal"] = min(1.0, len(self.steps) * 0.15)
            cge.nodes[trace_id]["confidence"] = min(1.0, self.avg_reward)

        # Conectar trace aos conceitos envolvidos
        for step in self.steps:
            for c in step.concepts_involved:
                if c in cge.nodes:
                    cge.add_edge(trace_id, c, "traced")

    @property
    def avg_reward(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.reward for s in self.steps) / len(self.steps)

    @property
    def modes_used(self) -> List[str]:
        return list(dict.fromkeys(s.mode for s in self.steps if s.mode))

    def summarize(self) -> str:
        """Sumario compacto para a Forja minerar."""
        steps_str = " -> ".join(
            f"{s.mode}(r={s.reward:.2f},nov={s.novelty_achieved})"
            for s in self.steps[:8]
        )
        return (f"Trace[{self.id}]: {self.problem} | "
                f"{len(self.steps)} steps: {steps_str} | "
                f"reward={self.avg_reward:.3f} success={self.success} | "
                f"nodes: {self.cge_nodes_before}->{self.cge_nodes_after}")

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "problem": self.problem,
            "steps": [{"mode": s.mode, "reward": s.reward,
                       "novelty": s.novelty_achieved,
                       "edges": s.edges_created} for s in self.steps],
            "final_answer": self.final_answer[:200],
            "success": self.success,
            "avg_reward": self.avg_reward,
            "modes_used": self.modes_used,
            "created_at": self.created_at,
        }

    @classmethod
    def mine_from_cge(cls, cge) -> List[Dict]:
        """Extrai traces do CGE para a Forja analisar."""
        traces = []
        for nid, node in cge.nodes.items():
            if nid.startswith("_trace_") and "trace_data" in node:
                traces.append(node["trace_data"])
        return traces

    @classmethod
    def analyze_patterns(cls, cge) -> List[str]:
        """A Forja analisa traces para descobrir meta-padroes."""
        traces_data = cls.mine_from_cge(cge)
        if len(traces_data) < 3:
            return ["Aguardando mais traces para minerar padroes..."]

        patterns = []
        # Modos que mais geram sucesso
        mode_success = {}
        for td in traces_data:
            pass  # Analise sera expandida pela Forja
        return patterns


class TraceRecorder:
    """Gravador automatico de traces no pipeline principal."""

    def __init__(self, cge):
        self.cge = cge
        self.current_trace: Optional[CognitiveTrace] = None
        self.trace_history: List[CognitiveTrace] = []
        self.max_traces: int = 50

    def start(self, problem: str):
        """Inicia um novo trace."""
        self.current_trace = CognitiveTrace(problem)
        self.current_trace.cge_nodes_before = len(self.cge.nodes)
        return self.current_trace

    def step(self, mode: str = "", context: str = "",
             operators: Optional[List[str]] = None,
             concepts: Optional[List[str]] = None,
             novelty: int = 0, edges: int = 0,
             outcome: str = "", reward: float = 0.0,
             error: Optional[Dict[str, float]] = None,
             workspace: Optional[List[str]] = None):
        """Registra um passo no trace atual."""
        if self.current_trace is None:
            return
        step = TraceStep(
            mode=mode, context=context,
            operators=operators or [],
            concepts_involved=concepts or [],
            novelty_achieved=novelty,
            edges_created=edges,
            outcome=outcome, reward=reward,
            error_snapshot=error or {},
            workspace_content=workspace or [],
        )
        self.current_trace.add_step(step)

    def finish(self, answer: str = "", success: bool = False):
        """Finaliza o trace atual e armazena no CGE."""
        if self.current_trace is None:
            return
        self.current_trace.cge_nodes_after = len(self.cge.nodes)
        self.current_trace.close(answer, success)
        self.current_trace.store_in_cge(self.cge)
        self.trace_history.append(self.current_trace)
        if len(self.trace_history) > self.max_traces:
            self.trace_history = self.trace_history[-self.max_traces:]
        trace = self.current_trace
        self.current_trace = None
        return trace
