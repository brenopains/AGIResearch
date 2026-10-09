"""
ConstraintEngine — Sistema de Restricoes APRENDIDAS

Nao sao regras hardcoded.
Sao restricoes que o sistema DESCOBRE por observacao.

Exemplos de restricoes que podem surgir:
  "Teorias nao podem violar a hierarquia causal"
  "Nenhum objeto esta em dois lugares ao mesmo tempo"
  "Se A causa B, B nao pode preceder A"
  "Um modelo nao pode exigir memoria infinita"

Cada restricao tem:
  - descricao
  - dominio de aplicacao
  - forca (0..1, aprendida)
  - violacoes observadas
  - origem (que observacao gerou)
"""
import time, uuid, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Constraint:
    """Uma restricao aprendida sobre como o mundo funciona."""
    id: str = ""
    description: str = ""
    domain: str = "general"       # "physical", "logical", "temporal", "social", "mathematical"
    strength: float = 0.5         # 0=fraca, 1=forte (aprendida)
    n_violations: int = 0         # quantas vezes foi violada (observada)
    n_confirmations: int = 0      # quantas vezes foi confirmada
    origin: str = ""              # que observacao/cenario gerou
    source_concept: str = ""      # conceito que originou
    related_concepts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        """Confianca baseada em confirmacoes vs violacoes."""
        total = self.n_confirmations + self.n_violations
        if total == 0:
            return 0.3
        return self.n_confirmations / max(1, total)

    def to_dict(self) -> Dict:
        return {"id": self.id, "description": self.description[:80],
                "domain": self.domain, "strength": round(self.strength, 2),
                "confidence": round(self.confidence, 2),
                "n_confirmations": self.n_confirmations,
                "n_violations": self.n_violations}


PRIMORDIAL_CONSTRAINTS = [
    Constraint(
        description="Todo objeto ocupa uma posicao no espaco",
        domain="physical", strength=0.95, n_confirmations=1000,
        source_concept="space"),
    Constraint(
        description="Nenhum objeto pode estar em dois lugares simultaneamente",
        domain="physical", strength=0.98, n_confirmations=1000,
        source_concept="exclusion"),
    Constraint(
        description="Se A causa B, B nao pode ocorrer antes de A",
        domain="causal", strength=0.95, n_confirmations=800,
        source_concept="causality"),
    Constraint(
        description="Efeitos nao precedem causas",
        domain="temporal", strength=0.97, n_confirmations=900,
        source_concept="time"),
    Constraint(
        description="Um modelo nao pode contradizer a si mesmo",
        domain="logical", strength=0.99, n_confirmations=1000,
        source_concept="consistency"),
    Constraint(
        description="A complexidade de um modelo deve ser proporcional ao que ele explica",
        domain="epistemic", strength=0.7, n_confirmations=300,
        source_concept="parsimony"),
]


class ConstraintEngine:
    """Motor de restricoes.

    Aprende restricoes observando:
      - Causalidade no CGE (arestas CAUSES)
      - Simulacoes que falham (branches que divergem)
      - Teorias que se contradizem
      - Padroes temporais

    O Maestro consulta o ConstraintEngine antes de:
      - Aceitar uma nova teoria
      - Iniciar uma simulacao cara
      - Promover um operador na Forja
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.constraints: Dict[str, Constraint] = {}
        self._violation_log: List[Dict] = []
        self._discovery_log: List[Dict] = []

        for c in PRIMORDIAL_CONSTRAINTS:
            cid = f"constraint_{hashlib.md5(c.description.encode()).hexdigest()[:8]}"
            c.id = cid
            self.constraints[cid] = c

    def register(self, constraint: Constraint) -> str:
        """Registra uma nova restricao."""
        cid = f"constraint_{hashlib.md5(constraint.description.encode()).hexdigest()[:8]}"
        if cid in self.constraints:
            existing = self.constraints[cid]
            existing.n_confirmations += 1
            existing.strength = min(1.0, existing.strength + 0.02)
            return cid
        constraint.id = cid
        self.constraints[cid] = constraint
        self._discovery_log.append({
            "time": time.time(), "constraint": constraint.description[:60],
            "domain": constraint.domain, "strength": constraint.strength,
        })
        return cid

    def check(self, proposal: Dict) -> Dict:
        """Verifica se uma proposta (teoria/estrategia/algoritmo) viola restricoes.

        Args:
            proposal: {"type": "theory", "description": "...", ...}

        Returns:
            {"pass": bool, "violations": [...], "strongest_violation": str}
        """
        violations = []
        for c in self.constraints.values():
            if c.strength < 0.3:
                continue
            # Verificar violacao baseada no dominio
            prop_domain = proposal.get("domain", "general")
            if c.domain != "general" and c.domain != prop_domain:
                continue

            # Heuristica simples: se descricao contem negacao da restricao
            # (em uma implementacao real, isso seria mais sofisticado)
            prop_text = str(proposal.get("description", "")).lower()
            constraint_keywords = c.description.lower().split()

            violation_score = self._compute_violation(proposal, c)
            if violation_score > 0.5:
                violations.append({
                    "constraint": c.description[:60],
                    "strength": c.strength,
                    "score": violation_score,
                })

        if violations:
            self._violation_log.append({
                "time": time.time(),
                "proposal": str(proposal.get("description", ""))[:60],
                "n_violations": len(violations),
            })

        strongest = max(violations, key=lambda v: v["strength"]) if violations else None
        return {
            "pass": len(violations) == 0,
            "violations": violations,
            "strongest_violation": strongest["constraint"] if strongest else None,
        }

    def _compute_violation(self, proposal: Dict, constraint: Constraint) -> float:
        """Computa o quanto uma proposta viola uma restricao.

        Em uma implementacao real, isso usaria simulacao causal.
        Aqui usa correspondencia textual como proxy.
        """
        text = str(proposal.get("description", "")).lower()
        # Palavras que sugerem violacao
        violation_indicators = {
            "physical": ["dois lugares", "mesmo tempo", "impossivel fisicamente"],
            "causal": ["antes da causa", "efeito sem causa", "causa invertida"],
            "logical": ["contradicao", "paradoxo", "inconsistente"],
            "temporal": ["passado mudou", "tempo reverso", "antes de existir"],
            "epistemic": ["infinito", "complexidade infinita", "sem limites"],
        }
        indicators = violation_indicators.get(constraint.domain, [])
        matches = sum(1 for ind in indicators if ind in text)
        return min(1.0, matches * 0.3)

    def discover_from_traces(self, traces: List[Any]) -> List[Constraint]:
        """Minera traces para descobrir novas restricoes."""
        new_constraints = []
        # Agrupar por dominio e verificar padroes de inviolabilidade
        domain_patterns = defaultdict(list)
        for t in traces:
            td = t.to_dict() if hasattr(t, 'to_dict') else (t if isinstance(t, dict) else {})
            steps = td.get("steps", []) if isinstance(td, dict) else []
            for s in steps:
                if isinstance(s, dict):
                    domain_patterns[s.get("mode", "general")].append(s.get("reward", 0))

        for domain, rewards in domain_patterns.items():
            if len(rewards) < 5:
                continue
            avg_r = sum(rewards) / len(rewards)
            if avg_r > 0.8:
                # Dominio com alta recompensa consistente: pode ter restricoes implicitas
                new = Constraint(
                    description=f"No dominio {domain}, abordagens testadas consistentemente funcionam",
                    domain=domain[:15],
                    strength=min(0.6, avg_r * 0.6),
                    n_confirmations=len(rewards),
                    source_concept=domain,
                )
                cid = self.register(new)
                if cid:
                    new_constraints.append(new)

        return new_constraints

    def get_active(self, threshold: float = 0.4) -> List[Constraint]:
        """Retorna restricoes ativas (acima do threshold)."""
        return [c for c in self.constraints.values() if c.strength >= threshold]

    def get_state(self) -> Dict:
        return {
            "total": len(self.constraints),
            "active": len(self.get_active()),
            "by_domain": {d: sum(1 for c in self.constraints.values() if c.domain == d)
                         for d in set(c.domain for c in self.constraints.values())},
            "recent_violations": self._violation_log[-3:] if self._violation_log else [],
        }
