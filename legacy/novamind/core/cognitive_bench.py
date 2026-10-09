"""
CognitiveBench — Suite de Metricas Cognitivas para Avaliar a AGI

Nao mede accuracy em benchmark fixo.
Mede PROPRIEDADES EMERGENTES da arquitetura:

  1. TRANSFERENCIA: learning rate em dominio B apos A (CrossDomainTest)
  2. COMPRESSAO: quantos conceitos por no/aresta ao longo do tempo
  3. ABSTRACAO: tempo do primeiro contato ate uma teoria/paradigma
  4. NOVIDADE: quantos algoritmos/estrategias/teorias descobertos por N ciclos
  5. EFICIENCIA ADAPTATIVA: reward por custo computacional ao longo do tempo
  6. VELOCIDADE DE ABSTRACAO: ciclos necessarios para passar de operador->algoritmo->estrategia->teoria
  7. ROBUSTEZ: variacao do reward sob mudanca de dominio
"""
import time, math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class BenchSnapshot:
    """Uma foto das metricas em um ponto no tempo."""
    cycle: int = 0
    timestamp: float = field(default_factory=time.time)

    # Tamanho do conhecimento
    n_nodes: int = 0
    n_edges: int = 0
    n_archetypes: int = 0
    n_algorithms: int = 0
    n_heuristics: int = 0
    n_strategies: int = 0
    n_theories: int = 0
    n_paradigms: int = 0
    n_worldviews: int = 0
    n_stances: int = 0
    n_principles: int = 0
    n_constraints: int = 0
    n_policies: int = 0
    n_goals: int = 0

    # Qualidade
    avg_reward: float = 0.0
    avg_confidence: float = 0.0
    prediction_error: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0

    # Meta-evolucao
    n_mutations: int = 0
    mutation_acceptance: float = 0.0
    n_traces: int = 0
    n_failed_traces: int = 0


@dataclass
class BenchReport:
    """Relatorio completo de metrica."""
    duration_cycles: int = 0
    duration_seconds: float = 0.0

    # Crescimento
    knowledge_growth_rate: float = 0.0   # nos/cycle
    edge_density: float = 0.0            # edges / nos^2
    compression_rate: float = 0.0        # archetypes / nodes (maior = melhor)

    # Descoberta
    abstraction_speed: float = 0.0       # ciclos ate primeira teoria
    discovery_rate: float = 0.0          # (algorithms+strategies+theories) / cycle
    paradigm_rate: float = 0.0           # paradigms / cycle

    # Qualidade
    avg_reward: float = 0.0
    reward_stability: float = 0.0        # baixa variancia = estavel
    meta_diversity: int = 0              # quantos tipos de meta-objetos

    # Transferencia (preenchido pelo CrossDomainTest)
    transfer_positive: int = 0
    transfer_negative: int = 0
    transfer_avg_gain: float = 0.0

    # Meta-evolucao
    mutation_rate: float = 0.0
    mutation_acceptance_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


class CognitiveBench:
    """Coleta snapshots periodicos e gera relatorios."""

    def __init__(self, cross_test=None):
        self.cross_test = cross_test
        self.snapshots: List[BenchSnapshot] = []
        self._last_snapshot_cycle: int = 0
        self._interval: int = 10  # snapshots a cada N ciclos

    def snapshot(self, cycle: int, cge=None, forja_evol=None, meta=None,
                 error_tracker=None, meta_evol=None, worldviews=None,
                 stances=None, insights=None, constraints=None,
                 policies=None, goals=None, trace_rec=None, fail_db=None):
        """Tira um snapshot das metricas atuais."""
        if cycle - self._last_snapshot_cycle < self._interval:
            return None
        self._last_snapshot_cycle = cycle

        s = BenchSnapshot(cycle=cycle)

        # CGE
        if cge:
            s.n_nodes = len(cge.nodes)
            s.n_edges = len(cge.edges)

        # Forja
        if forja_evol:
            s.n_algorithms = forja_evol.get_state().get("algorithms", 0)
            s.n_heuristics = forja_evol.get_state().get("heuristics", 0)
            s.n_strategies = forja_evol.get_state().get("strategies", 0)
            s.n_theories = forja_evol.get_state().get("theories", 0)
            s.n_paradigms = forja_evol.get_state().get("paradigms", 0)

        # MetaController
        if meta:
            st = meta.get_state()
            s.avg_reward = sum(st.get("mode_performance", {}).values()) / max(1, len(st.get("mode_performance", {})))

        # Erro
        if error_tracker:
            est = error_tracker.get_state()
            s.prediction_error = est.get("avg_prediction_error", 0)
            s.novelty = est.get("avg_novelty", 0)
            s.uncertainty = est.get("avg_uncertainty", 0)

        # MetaEvolution
        if meta_evol:
            mest = meta_evol.get_state()
            s.n_mutations = mest.get("n_mutations", 0)
            s.mutation_acceptance = mest.get("acceptance_rate", 0)

        # Outros
        if worldviews:
            s.n_worldviews = worldviews.get_state().get("n_worldviews", 0)
        if stances:
            s.n_stances = stances.get_state().get("n_stances", 0)
        if insights:
            s.n_principles = insights.get_state().get("n_insights", 0)
        if constraints:
            s.n_constraints = constraints.get_state().get("total", 0)
        if policies:
            s.n_policies = policies.get_state().get("n_policies", 0)
        if goals:
            s.n_goals = goals.get_state().get("active", 0)
        if trace_rec:
            s.n_traces = len(trace_rec.trace_history)
        if fail_db:
            s.n_failed_traces = fail_db.get_state().get("n_traces", 0)

        self.snapshots.append(s)

        # Disparar relatorio a cada 50 snapshots
        if len(self.snapshots) >= 5 and len(self.snapshots) % 5 == 0:
            return self.generate_report()

        return None

    def generate_report(self) -> BenchReport:
        """Gera relatorio consolidado de todos os snapshots."""
        if len(self.snapshots) < 2:
            return BenchReport()

        first = self.snapshots[0]
        last = self.snapshots[-1]
        duration = last.cycle - first.cycle

        report = BenchReport()
        report.duration_cycles = duration
        report.duration_seconds = last.timestamp - first.timestamp

        # Crescimento
        report.knowledge_growth_rate = (last.n_nodes - first.n_nodes) / max(1, duration)
        report.edge_density = last.n_edges / max(1, last.n_nodes ** 2)
        report.compression_rate = last.n_archetypes / max(1, last.n_nodes)

        # Descoberta
        report.abstraction_speed = self._first_appearance("n_theories")
        report.discovery_rate = (
            (last.n_algorithms - first.n_algorithms) +
            (last.n_strategies - first.n_strategies) +
            (last.n_theories - first.n_theories)
        ) / max(1, duration)
        report.paradigm_rate = (last.n_paradigms - first.n_paradigms) / max(1, duration)

        # Qualidade
        rewards = [s.avg_reward for s in self.snapshots if s.avg_reward > 0]
        report.avg_reward = sum(rewards) / max(1, len(rewards))
        if len(rewards) >= 2:
            report.reward_stability = 1.0 - min(1.0, (max(rewards) - min(rewards)) / max(0.01, max(rewards)))

        # Diversidade meta
        meta_types = ["n_algorithms", "n_heuristics", "n_strategies", "n_theories",
                      "n_paradigms", "n_worldviews", "n_stances", "n_principles",
                      "n_constraints", "n_policies"]
        report.meta_diversity = sum(1 for t in meta_types if getattr(last, t, 0) > 0)

        # Transferencia
        if self.cross_test:
            cs = self.cross_test.get_state()
            report.transfer_positive = cs.get("positive_transfers", 0)
            report.transfer_negative = cs.get("negative_transfers", 0)
            report.transfer_avg_gain = cs.get("avg_gain", 0)

        # Meta-evolucao
        report.mutation_rate = last.n_mutations / max(1, duration)
        report.mutation_acceptance_rate = last.mutation_acceptance

        return report

    def _first_appearance(self, attr: str) -> float:
        """Ciclo em que um atributo apareceu pela primeira vez."""
        for s in self.snapshots:
            if getattr(s, attr, 0) > 0:
                return s.cycle
        return float('inf')

    def summary(self) -> str:
        """Resumo legivel do ultimo relatorio."""
        r = self.generate_report()
        lines = [
            "=" * 60,
            "COGNITIVE BENCH REPORT",
            "=" * 60,
            f"Duracao: {r.duration_cycles} ciclos ({r.duration_seconds:.0f}s)",
            f"",
            f"--- CRESCIMENTO ---",
            f"Knowledge growth: {r.knowledge_growth_rate:.2f} nodes/cycle",
            f"Edge density: {r.edge_density:.4f}",
            f"Compression rate: {r.compression_rate:.4f}",
            f"",
            f"--- DESCOBERTA ---",
            f"Abstraction speed: {r.abstraction_speed:.0f} cycles",
            f"Discovery rate: {r.discovery_rate:.4f} /cycle",
            f"Paradigm rate: {r.paradigm_rate:.4f} /cycle",
            f"Meta diversity: {r.meta_diversity} types",
            f"",
            f"--- QUALIDADE ---",
            f"Avg reward: {r.avg_reward:.3f}",
            f"Reward stability: {r.reward_stability:.3f}",
            f"",
            f"--- TRANSFERENCIA ---",
            f"Positive transfers: {r.transfer_positive}",
            f"Negative transfers: {r.transfer_negative}",
            f"Avg gain: {r.transfer_avg_gain:.4f}",
            f"",
            f"--- META-EVOLUCAO ---",
            f"Mutation rate: {r.mutation_rate:.4f} /cycle",
            f"Mutation acceptance: {r.mutation_acceptance_rate:.2%}",
            f"=" * 60,
        ]
        return "\n".join(lines)
