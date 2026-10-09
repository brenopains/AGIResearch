"""
ConflictResolver — Transforma contradicao em novo conhecimento.

O ciclo e analogo ao metodo cientifico:
  1. WorldModel detecta anomalia: lei X preve A, mas observamos B
  2. ConflictResolver registra o conflito
  3. HypothesisEngine gera explicacoes concorrentes
  4. Cada explicacao e testada contra ConstraintNetwork
  5. Se aprovada → nova WorldLaw ou refinamento de conceito
  6. Se rejeitada → o conflito permanece como "gap" (curiosidade futura)

Exemplo:
  Lei: "todos metais afundam"
  Observacao: "navio flutua"
  → CONFLITO
  → Hipotese: "densidade media explica flutuacao"
  → Nova lei: "densidade media determina flutuacao"
"""
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import uuid


@dataclass
class Conflict:
    """Um conflito entre o modelo do mundo e a realidade."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    law_name: str = ""           # lei que foi violada
    domain: str = ""             # dominio onde ocorreu
    observation: Any = None      # o que foi observado
    prediction: Any = None       # o que a lei previa
    description: str = ""        # descricao textual
    severity: float = 0.5       # 0-1, quao grave
    resolved: bool = False
    resolution: str = ""         # como foi resolvido
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "law": self.law_name,
            "domain": self.domain,
            "description": self.description[:80],
            "severity": round(self.severity, 2),
            "resolved": self.resolved,
            "resolution": self.resolution[:60] if self.resolution else "",
        }


@dataclass
class Hypothesis:
    """Hipotese gerada para resolver um conflito."""
    conflict_id: str
    description: str
    proposed_law_name: str = ""
    proposed_explanation: str = ""
    confidence: float = 0.3
    tested: bool = False
    passed: bool = False
    test_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "conflict": self.conflict_id,
            "hypothesis": self.description[:80],
            "proposed_law": self.proposed_law_name,
            "confidence": round(self.confidence, 2),
            "tested": self.tested,
            "passed": self.passed,
        }


class ConflictResolver:
    """Resolver de conflitos: anomalia → hipotese → teste → nova lei.

    Uso:
        resolver = ConflictResolver(world_law_registry, constraint_network)
        resolver.detect_conflict("conservation", "quantum", obs, pred)
        hypotheses = resolver.resolve(conflict)
        if hypotheses:
            for h in hypotheses:
                if h.passed:
                    print(f"Nova lei: {h.proposed_law_name}")
    """

    def __init__(self, law_registry=None, constraint_network=None,
                 hypothesis_engine=None):
        self.law_registry = law_registry
        self.constraint_network = constraint_network
        self.hypothesis_engine = hypothesis_engine
        self.conflicts: List[Conflict] = []
        self.hypotheses: List[Hypothesis] = []
        self.resolved_count: int = 0

    def detect_conflict(self, law_name: str, domain: str,
                        observation: Any, prediction: Any,
                        description: str = "",
                        severity: Optional[float] = None) -> Conflict:
        """Registra um conflito entre lei e observacao.

        Args:
            law_name: nome da lei violada
            domain: dominio onde ocorreu
            observation: o que foi observado
            prediction: o que a lei previa
            description: descricao textual

        Returns: Conflict registrado.
        """
        if severity is None:
            severity = self._compute_severity(observation, prediction)

        conflict = Conflict(
            law_name=law_name,
            domain=domain,
            observation=observation,
            prediction=prediction,
            description=description or f"conflict in {domain}: {law_name} violated",
            severity=severity,
        )

        self.conflicts.append(conflict)
        return conflict

    def _compute_severity(self, observation: Any, prediction: Any) -> float:
        """Computa gravidade do conflito."""
        if observation is None or prediction is None:
            return 0.3

        # Se sao arrays, usa diferenca relativa
        import numpy as np
        if hasattr(observation, '__len__') and hasattr(prediction, '__len__'):
            try:
                obs_arr = np.array(observation)
                pred_arr = np.array(prediction)
                if obs_arr.size > 0 and pred_arr.size > 0:
                    diff = np.abs(obs_arr - pred_arr).mean()
                    return min(1.0, diff / max(1e-6, np.abs(pred_arr).mean()))
            except:
                pass

        return 0.5

    def resolve(self, conflict: Conflict) -> List[Hypothesis]:
        """Tenta resolver um conflito gerando hipoteses.

        Returns: lista de hipoteses testadas.
        """
        candidates = []

        # 1. Gerar hipoteses baseadas no tipo de conflito
        hypotheses = self._generate_hypotheses(conflict)

        for hyp in hypotheses:
            # 2. Testar contra constraint network
            if self.constraint_network:
                verdicts = self.constraint_network.validate(
                    hyp.proposed_law_name,
                    concept_type="world_law",
                    domains={conflict.domain},
                )
                approved = sum(1 for v in verdicts if v.approved)
                required = max(1, len(verdicts) // 2 + 1)
                hyp.tested = True
                hyp.passed = approved >= required
                hyp.test_evidence = [
                    f"{v.source}: {'approved' if v.approved else 'rejected'} ({v.reason})"
                    for v in verdicts
                ]
            else:
                hyp.tested = True
                hyp.passed = True

            candidates.append(hyp)
            self.hypotheses.append(hyp)

            # Se passou, registra nova lei
            if hyp.passed and self.law_registry and hyp.proposed_law_name:
                from .world_laws import WorldLaw
                new_law = WorldLaw(
                    name=hyp.proposed_law_name,
                    description=hyp.proposed_explanation,
                    explains=set(),
                    domains={conflict.domain},
                    confidence=hyp.confidence,
                )
                self.law_registry.register(new_law)
                conflict.resolved = True
                conflict.resolution = hyp.proposed_explanation
                conflict.resolved_at = time.time()
                self.resolved_count += 1

        return candidates

    def _generate_hypotheses(self, conflict: Conflict) -> List[Hypothesis]:
        """Gera hipoteses para explicar um conflito.

        Estrategias:
          1. Lei muito geral: talvez a lei nao se aplique a este dominio
          2. Falta conceito: talvez exista um conceito intermediario
          3. Excecao: talvez seja uma excecao que refina a lei
          4. Nova lei: talvez uma lei diferente explique melhor
        """
        hypotheses = []

        # Estrategia 1: Lei muito geral
        hyp1 = Hypothesis(
            conflict_id=conflict.id,
            description=f"law '{conflict.law_name}' is too general for domain '{conflict.domain}'",
            proposed_law_name=f"{conflict.law_name}_refined",
            proposed_explanation=f"{conflict.law_name} does not apply to {conflict.domain}, "
                                 f"a refined version is needed",
            confidence=0.3,
        )
        hypotheses.append(hyp1)

        # Estrategia 2: Excecao que refina a lei
        hyp2 = Hypothesis(
            conflict_id=conflict.id,
            description=f"exception to '{conflict.law_name}' in domain '{conflict.domain}'",
            proposed_law_name=f"{conflict.law_name}_exception",
            proposed_explanation=f"under specific conditions in {conflict.domain}, "
                                 f"{conflict.law_name} behaves differently",
            confidence=0.4,
        )
        hypotheses.append(hyp2)

        # Estrategia 3: Nova lei mais geral
        hyp3 = Hypothesis(
            conflict_id=conflict.id,
            description=f"new law explaining both prediction and observation",
            proposed_law_name=f"generalized_{conflict.law_name}",
            proposed_explanation=f"a more general principle that explains when "
                                 f"{conflict.law_name} applies and when it doesn't",
            confidence=0.2,
        )
        hypotheses.append(hyp3)

        return hypotheses

    def get_unresolved_conflicts(self) -> List[Conflict]:
        return [c for c in self.conflicts if not c.resolved]

    def get_stats(self) -> dict:
        return {
            "total_conflicts": len(self.conflicts),
            "resolved": self.resolved_count,
            "unresolved": len(self.get_unresolved_conflicts()),
            "total_hypotheses": len(self.hypotheses),
            "passed_hypotheses": sum(1 for h in self.hypotheses if h.passed),
            "recent_conflicts": [c.to_dict() for c in self.conflicts[-5:]],
        }
