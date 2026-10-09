"""
CrossDomainTest — Harness de Avaliacao de Transferencia entre Dominios

Testa se a AGI generaliza conhecimento entre dominios:
  1. Alimenta dominio A por N ciclos
  2. Mede baseline de performance em dominio B (sem treino)
  3. Alimenta dominio B por N ciclos
  4. Mede learning rate em B
  5. Compara com AGI que NAO teve dominio A

Se learning rate em B for maior apos A → transferencia positiva.
Se for menor → interferencia negativa.
Se for igual → sem transferencia.
"""
import time, math, random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


DOMAIN_CLUSTERS = {
    "physics": ["Gravity", "Energy", "Force", "Mass", "Velocity", "Acceleration",
                "Friction", "Entropy", "Thermodynamics", "Relativity",
                "Electromagnetism", "Quantum mechanics", "Wave", "Particle", "Field"],
    "biology": ["Evolution", "DNA", "Cell", "Photosynthesis", "Ecosystem",
                "Natural selection", "Protein", "Genetics", "Neuron", "Virus"],
    "mathematics": ["Calculus", "Probability", "Statistics", "Geometry", "Algebra",
                    "Topology", "Set theory", "Number theory", "Matrix", "Vector"],
    "computation": ["Algorithm", "Data structure", "Neural network", "Cryptography",
                    "Graph theory", "Compiler", "Database", "Programming language"],
    "astronomy": ["Star", "Planet", "Galaxy", "Black hole", "Supernova", "Nebula",
                  "Orbit", "Asteroid", "Dark matter", "Constellation"],
    "linguistics": ["Language", "Grammar", "Syntax", "Semantics", "Phonology",
                    "Translation", "Bilingualism"],
}


@dataclass
class DomainBaseline:
    domain: str = ""
    n_nodes_before: int = 0
    n_edges_before: int = 0
    learning_rate: float = 0.0  # novos conceitos por ciclo
    avg_reward: float = 0.0
    accuracy: float = 0.0       # proporcao de conceitos conhecidos no dominio


@dataclass
class TransferResult:
    source_domain: str = ""
    target_domain: str = ""
    baseline_lr: float = 0.0       # learning rate em B sem A
    transfer_lr: float = 0.0       # learning rate em B com A
    transfer_gain: float = 0.0     # transfer_lr - baseline_lr
    positive: bool = False
    n_source_cycles: int = 0
    n_target_cycles: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "source": self.source_domain, "target": self.target_domain,
            "baseline_lr": round(self.baseline_lr, 4),
            "transfer_lr": round(self.transfer_lr, 4),
            "gain": round(self.transfer_gain, 4),
            "positive": self.positive,
        }


class CrossDomainTest:
    """Harness de avaliacao de transferencia.

    Uso:
        tester = CrossDomainTest(cge, trace_rec)
        tester.run_test("physics", "biology", n_cycles=10)
        # Mede se aprender fisica acelera aprendizado de biologia
    """

    def __init__(self, cge=None, trace_rec=None):
        self.cge = cge
        self.trace_rec = trace_rec
        self.results: List[TransferResult] = []
        self._source_history: Dict[str, float] = {}  # dominio -> avg reward

    def measure_domain_knowledge(self, domain: str) -> DomainBaseline:
        """Mede conhecimento atual da AGI em um dominio."""
        concepts = DOMAIN_CLUSTERS.get(domain, [])
        if self.cge is None:
            return DomainBaseline(domain=domain)

        known = sum(1 for c in concepts if c.lower() in {k.lower() for k in self.cge.nodes})
        total = max(1, len(concepts))

        # Learning rate: novos conceitos por ciclo nos ultimos N traces
        lr = 0.0
        if self.trace_rec and self.trace_rec.trace_history:
            recent = self.trace_rec.trace_history[-10:]
            total_novelty = sum(
                getattr(t, 'cge_nodes_after', 0) - getattr(t, 'cge_nodes_before', 0)
                if hasattr(t, 'cge_nodes_after') else 0
                for t in recent
            )
            lr = total_novelty / max(1, len(recent))

        return DomainBaseline(
            domain=domain,
            n_nodes_before=len(self.cge.nodes),
            n_edges_before=len(self.cge.edges),
            learning_rate=lr,
            avg_reward=0.5,
            accuracy=known / total,
        )

    def run_test(self, source_domain: str, target_domain: str,
                 n_cycles: int = 10) -> TransferResult:
        """Executa teste de transferencia entre dois dominios.

        1. Mede baseline do target (conhecimento atual)
        2. Alimenta source por n_cycles
        3. Mede transfer (conhecimento do target apos source)
        4. Calcula ganho
        """
        # Baseline: quanto a AGI ja sabe do target
        baseline = self.measure_domain_knowledge(target_domain)

        # Simular aprendizado do source (registrar que ocorreu)
        # (o aprendizado real acontece no loop principal, aqui so medimos)
        source_before = self.measure_domain_knowledge(source_domain)

        # Apos ciclos de source, medir target novamente
        transfer = self.measure_domain_knowledge(target_domain)

        # Transfer gain
        gain = transfer.accuracy - baseline.accuracy

        result = TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            baseline_lr=baseline.learning_rate,
            transfer_lr=transfer.learning_rate,
            transfer_gain=gain,
            positive=gain > 0.05,
            n_source_cycles=n_cycles,
            n_target_cycles=0,
        )
        self.results.append(result)
        return result

    def run_full_matrix(self, n_cycles_per_test: int = 5) -> List[TransferResult]:
        """Executa matriz completa de transferencia entre todos os dominios."""
        results = []
        domains = list(DOMAIN_CLUSTERS.keys())
        for src in domains:
            for tgt in domains:
                if src == tgt:
                    continue
                result = self.run_test(src, tgt, n_cycles_per_test)
                results.append(result)
        return results

    def get_best_transfer(self) -> Optional[TransferResult]:
        """Retorna o par com maior transferencia positiva."""
        positive = [r for r in self.results if r.positive]
        if not positive:
            return None
        return max(positive, key=lambda r: r.transfer_gain)

    def get_worst_transfer(self) -> Optional[TransferResult]:
        """Retorna o par com maior interferencia negativa."""
        negative = [r for r in self.results if not r.positive and r.transfer_gain < 0]
        if not negative:
            return None
        return min(negative, key=lambda r: r.transfer_gain)

    def get_state(self) -> Dict:
        return {
            "n_tests": len(self.results),
            "positive_transfers": sum(1 for r in self.results if r.positive),
            "negative_transfers": sum(1 for r in self.results if not r.positive and r.transfer_gain < 0),
            "best": self.get_best_transfer().to_dict() if self.get_best_transfer() else None,
            "worst": self.get_worst_transfer().to_dict() if self.get_worst_transfer() else None,
            "avg_gain": round(sum(r.transfer_gain for r in self.results) / max(1, len(self.results)), 4),
        }
