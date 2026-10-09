"""
FailedTrace — Registro de Fracassos e Mineracao de Anti-Heuristicas

Enquanto CognitiveTrace registra sucessos, FailedTrace registra:
  - Ciclos com prediction_error > threshold
  - Hipoteses refutadas
  - Caminhos que nao levaram a lugar nenhum

A Forja minera FailedTraces para descobrir ANTI-HEURISTICAS:
  "Nunca comecar por hipotese X no dominio Y"
  "Operador Z combinado com W gera beco sem saída"
"""
import time, uuid, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class FailedStep:
    mode: str = ""
    operator: str = ""
    prediction_error: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    concept: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailedTrace:
    id: str = ""
    problem: str = ""
    steps: List[FailedStep] = field(default_factory=list)
    avg_error: float = 0.0
    peak_error: float = 0.0
    duration: float = 0.0
    failure_type: str = "unknown"  # "impasse", "contradiction", "divergence", "cycle"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "problem": self.problem[:80],
            "avg_error": self.avg_error, "peak_error": self.peak_error,
            "failure_type": self.failure_type,
            "steps": [{"mode": s.mode, "operator": s.operator,
                      "error": s.prediction_error} for s in self.steps[-5:]],
        }


@dataclass
class AntiHeuristic:
    """Regra negativa: o que EVITAR fazer."""
    id: str = ""
    condition: str = ""
    action: str = ""
    confidence: float = 0.3
    n_observations: int = 0
    severity: float = 0.5  # 0=leve, 1=evitar sempre
    domain: str = "general"
    created_at: float = field(default_factory=time.time)

    def to_insight(self) -> str:
        return (f"EVITAR: quando {self.condition}, "
                f"{self.action} (conf={self.confidence:.2f}, "
                f"{self.n_observations} observacoes)")


class FailedTraceDB:
    """Banco de FailedTraces. Minera anti-heuristicas."""

    def __init__(self, cge=None, insight_library=None, max_traces: int = 200):
        self.cge = cge
        self.insight_library = insight_library
        self.traces: List[FailedTrace] = []
        self.anti_heuristics: Dict[str, AntiHeuristic] = {}
        self.max_traces = max_traces

    def record(self, problem: str, steps: List[FailedStep],
               failure_type: str = "unknown") -> FailedTrace:
        """Registra um trace de falha."""
        if not steps:
            return None
        ft = FailedTrace(
            id=f"fail_{uuid.uuid4().hex[:8]}",
            problem=problem,
            steps=steps,
            avg_error=sum(s.prediction_error for s in steps) / len(steps),
            peak_error=max(s.prediction_error for s in steps),
            duration=steps[-1].timestamp - steps[0].timestamp if len(steps) > 1 else 0,
            failure_type=failure_type,
        )
        self.traces.append(ft)
        if len(self.traces) > self.max_traces:
            self.traces.pop(0)
        self._sync_to_cge(ft)
        return ft

    def record_from_tick(self, problem: str, mode: str, concepts: list,
                         error_signals: dict):
        """Registra um passo de falha a partir de um tick do Maestro."""
        step = FailedStep(
            mode=mode,
            prediction_error=error_signals.get("prediction_error", 0),
            novelty=error_signals.get("novelty", 0),
            uncertainty=error_signals.get("uncertainty", 0),
            concept=concepts[0] if concepts else "",
        )
        # Verificar se ja temos um trace aberto para este problema
        for ft in reversed(self.traces):
            if ft.problem == problem and ft.failure_type == "open":
                ft.steps.append(step)
                ft.avg_error = sum(s.prediction_error for s in ft.steps) / len(ft.steps)
                ft.peak_error = max(s.prediction_error for s in ft.steps)
                return
        # Novo trace
        ft = FailedTrace(
            id=f"fail_{uuid.uuid4().hex[:8]}",
            problem=problem, steps=[step],
            avg_error=step.prediction_error,
            peak_error=step.prediction_error,
            failure_type="open",
        )
        self.traces.append(ft)

    def close_failed_trace(self, problem: str, failure_type: str = "impasse"):
        """Fecha um trace de falha aberto."""
        for ft in reversed(self.traces):
            if ft.problem == problem and ft.failure_type == "open":
                ft.failure_type = failure_type
                break

    def mine_anti_heuristics(self) -> List[AntiHeuristic]:
        """Minera todos os FailedTraces para descobrir anti-heuristicas."""
        new = []
        if len(self.traces) < 3:
            return new

        # 1. Modos que mais falham
        mode_fails = defaultdict(list)
        for ft in self.traces:
            for s in ft.steps:
                if s.prediction_error > 0.6:
                    mode_fails[s.mode].append(s.prediction_error)

        for mode, errors in mode_fails.items():
            if len(errors) < 3:
                continue
            avg_err = sum(errors) / len(errors)
            if avg_err < 0.5:
                continue
            ah = AntiHeuristic(
                id=f"anti_{hashlib.md5(('mode_'+mode).encode()).hexdigest()[:8]}",
                condition=f"modo {mode.upper()} tem alta taxa de falha "
                          f"(avg_err={avg_err:.2f})",
                action=f"evitar modo {mode.upper()} quando incerteza > 0.6",
                confidence=min(0.8, avg_err * 0.6 + len(errors) * 0.03),
                n_observations=len(errors),
                severity=min(1.0, avg_err),
            )
            if ah.id not in self.anti_heuristics:
                self.anti_heuristics[ah.id] = ah
                new.append(ah)

        # 2. Transicoes que levam a erro
        fail_transitions = defaultdict(list)
        for ft in self.traces:
            for i in range(len(ft.steps) - 1):
                key = f"{ft.steps[i].mode}->{ft.steps[i+1].mode}"
                fail_transitions[key].append(ft.steps[i+1].prediction_error)

        for trans, errors in fail_transitions.items():
            if len(errors) < 3:
                continue
            avg_err = sum(errors) / len(errors)
            if avg_err < 0.5:
                continue
            parts = trans.split("->")
            ah = AntiHeuristic(
                id=f"anti_{hashlib.md5(('trans_'+trans).encode()).hexdigest()[:8]}",
                condition=f"transicao {parts[0].upper()} -> {parts[1].upper()} "
                          f"gera erro alto ({avg_err:.2f})",
                action=f"evitar transicao direta de {parts[0].upper()} para {parts[1].upper()}",
                confidence=min(0.7, avg_err * 0.5),
                n_observations=len(errors),
                severity=0.6,
            )
            if ah.id not in self.anti_heuristics:
                self.anti_heuristics[ah.id] = ah
                new.append(ah)

        # 3. Sincronizar com InsightLibrary
        for ah in new:
            if self.insight_library:
                self.insight_library.add_insight(
                    principle=ah.to_insight(),
                    confidence=ah.confidence * 0.8,
                    tags=["anti_heuristic", ah.domain],
                )

        return new

    def get_anti_heuristics_for_context(self, mode: str = "",
                                         threshold: float = 0.3) -> List[AntiHeuristic]:
        """Retorna anti-heuristicas relevantes para o contexto."""
        results = []
        for ah in self.anti_heuristics.values():
            if ah.confidence < threshold:
                continue
            if mode and mode.upper() in ah.condition:
                results.append(ah)
        return sorted(results, key=lambda a: -a.confidence)[:5]

    def get_state(self) -> Dict:
        return {
            "n_traces": len(self.traces),
            "n_anti_heuristics": len(self.anti_heuristics),
            "recent_failures": [ft.to_dict() for ft in self.traces[-3:]],
            "worst_modes": self._worst_modes(),
        }

    def _worst_modes(self) -> Dict[str, float]:
        mode_errors = defaultdict(list)
        for ft in self.traces:
            for s in ft.steps:
                mode_errors[s.mode].append(s.prediction_error)
        return {m: round(sum(e)/len(e), 3) for m, e in mode_errors.items()
                if len(e) >= 2}

    def _sync_to_cge(self, ft: FailedTrace):
        if self.cge is None:
            return
        import torch
        nid = ft.id
        g = torch.Generator()
        g.manual_seed(int(hashlib.md5(nid.encode()).hexdigest()[:8], 16))
        vec = torch.sign(torch.randn(10000, generator=g))
        self.cge.create_node(nid, vec)
        if nid in self.cge.nodes:
            self.cge.nodes[nid]["type"] = "failed_trace"
            self.cge.nodes[nid]["fail_data"] = ft.to_dict()
