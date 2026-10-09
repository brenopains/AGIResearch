"""
Contracts — Tipos e contratos compartilhados entre todos os modulos cognitivos.

Nenhum componente importa a implementacao interna de outro.
Todos conversam apenas atraves destes contratos.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


# ─── Recall ─────────────────────────────────────────────────

class RecallStatus(Enum):
    EMPTY = "empty"
    ACCEPT = "accept"


@dataclass
class RecallDecision:
    """Decisao do RecallPolicy sobre se o recall e utilizavel."""
    status: RecallStatus
    best_similarity: float = 0.0
    threshold: float = 0.25
    reason: str = ""
    n_results: int = 0
    coverage: float = 0.0
    episodes: List[Dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.status == RecallStatus.EMPTY

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "best_similarity": round(self.best_similarity, 3),
            "threshold": self.threshold,
            "reason": self.reason,
            "n_results": self.n_results,
            "coverage": round(self.coverage, 3) if self.coverage else 0,
        }


# ─── Evidencia ─────────────────────────────────────────────

class EvidenceSource(Enum):
    TAXONOMY = "taxonomy"
    EPISODIC = "episodic"
    PERCEPTUAL = "perceptual"
    CAUSAL = "causal"
    LOGICAL = "logical"
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"


class EvidenceRelation(Enum):
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    RELATED = "related"
    SIMILAR = "similar"


@dataclass
class EvidenceItem:
    """Uma evidencia unica. Qualquer builder gera exatamente este formato."""
    source: EvidenceSource
    relation: EvidenceRelation
    confidence: float  # 0-1, ja calibrado
    raw_score: float = 0.0  # bruto, sem escala
    trace: str = ""  # "WordNet: whale -> cetacean -> mammal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 3),
            "raw_score": round(self.raw_score, 3),
            "trace": self.trace[:80],
        }


# ─── Hipotese ──────────────────────────────────────────────

# ─── Contribution Breakdown ─────────────────────────────────

@dataclass
class ContributionBreakdown:
    """Decomposicao detalhada do score de uma hipotese por fonte.

    Permite responder: "quanto cada fonte contribuiu para o score final?"
    """
    source: EvidenceSource
    score: float = 0.0         # 0-1, contribuicao absoluta desta fonte
    weight: float = 1.0        # peso relativo no calculo final
    detail: str = ""           # explicacao textual
    n_items: int = 0           # quantos EvidenceItem desta fonte

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "detail": self.detail[:80],
            "n_items": self.n_items,
        }


# ─── Convergence Step ──────────────────────────────────────

@dataclass
class ConvergenceStep:
    """Estado completo de uma iteracao da competicao.

    Permite reconstruir a trajetoria de cada hipotese ao longo do tempo.
    """
    iteration: int
    scores: Dict[str, float]       # hypothesis_name -> score atual
    winner: str = ""
    margin: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "winner": self.winner,
            "margin": round(self.margin, 4),
            "entropy": round(self.entropy, 4),
        }


# ─── Hipotese ──────────────────────────────────────────────

@dataclass
class Hypothesis:
    """Uma hipotese concorrente, com evidencias a favor e contra."""
    name: str
    category: str = "related"  # is_a, has_property, related, similar
    description: str = ""
    evidence_for: List[EvidenceItem] = field(default_factory=list)
    evidence_against: List[EvidenceItem] = field(default_factory=list)
    confidence: float = 0.0  # calibrado 0-1
    evidence_total: float = 0.0  # soma bruta sem escala
    evidence_by_type: Dict[str, dict] = field(default_factory=dict)
    contribution_breakdown: List[ContributionBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "evidence_total": round(self.evidence_total, 3),
            "evidence_by_type": self.evidence_by_type,
            "contribution_breakdown": [c.to_dict() for c in self.contribution_breakdown],
            "evidence_for": [e.to_dict() for e in self.evidence_for[-3:]],
            "evidence_against": [e.to_dict() for e in self.evidence_against[-2:]],
        }


# ─── Competicao ────────────────────────────────────────────

@dataclass
class CompetitionResult:
    """Resultado completo da competicao de hipoteses."""
    hypotheses: List[Hypothesis]
    winner: Optional[str] = None
    winner_confidence: float = 0.0
    runner_up: Optional[str] = None
    runner_up_confidence: float = 0.0
    margin: float = 0.0
    entropy: float = 0.0
    stability: List[float] = field(default_factory=list)
    by_category: Dict[str, List[dict]] = field(default_factory=dict)
    counterfactuals: List[dict] = field(default_factory=list)

    # ─── Novos campos de rastreabilidade profunda ──────────
    convergence_steps: List[ConvergenceStep] = field(default_factory=list)
    diversity_score: float = 0.0         # 0-1, quao diversas sao as hipoteses
    num_clusters: int = 1                # quantos tipos de solucao
    winner_reversals: int = 0            # quantas vezes o vencedor mudou entre iteracoes
    convergence_velocity: float = 0.0    # (score_final - score_inicial) / n_iters

    def to_dict(self) -> dict:
        return {
            "n_hypotheses": len(self.hypotheses),
            "winner": self.winner,
            "winner_confidence": round(self.winner_confidence, 3),
            "margin": round(self.margin, 3),
            "entropy": round(self.entropy, 3),
            "stability": [round(s, 3) for s in self.stability[-5:]],
            "by_category": self.by_category,
            "counterfactuals": self.counterfactuals[:3],
            "diversity_score": round(self.diversity_score, 4),
            "num_clusters": self.num_clusters,
            "winner_reversals": self.winner_reversals,
            "convergence_velocity": round(self.convergence_velocity, 4),
            "convergence_steps": [s.to_dict() for s in self.convergence_steps[-10:]],
        }


# ─── Benchmark ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Resultado de um benchmark de regressao."""
    name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    details: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in self.metrics.items()},
        }
