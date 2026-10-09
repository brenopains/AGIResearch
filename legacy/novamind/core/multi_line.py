"""
MultiLineInvestigator — Mantem N hipoteses concorrentes por doenca.

Cada linha e uma hipotese mecanistica que compete com as outras.
Linhas nascem, morrem, se combinam, se refutam.

Dinamicamente:
  - Linhas começam com score baseado em evidencias iniciais
  - Cada rodada: linhas acumulam ou perdem forca
  - Linhas com score baixo sao abandonadas
  - Linhas que encontram paradoxos geram filhas
  - Linhas que convergem sao fundidas
  - O numero de linhas varia organicamente

Nao ha numero fixo de rodadas. O processo para quando:
  - Linha sobrevivente estabilizou (score variou <1% por 5 rodadas)
  - OU exploracao exauriu (nenhuma linha nova gerada nas ultimas 10 rodadas)
"""
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


@dataclass
class HypothesisLine:
    """Uma linha de investigacao (hipotese concorrente)."""
    id: int
    mechanism: str
    intervention: str
    target: str
    evidence_ids: List[int] = field(default_factory=list)
    score: float = 0.3
    age: int = 0
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    paradoxes_found: List[str] = field(default_factory=list)
    refutations_survived: int = 0
    domains_explored: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def stagnated(self) -> bool:
        """Score nao mudou significativamente?"""
        return self.age > 10 and self.score < 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mechanism": self.mechanism,
            "intervention": self.intervention[:40],
            "score": round(self.score, 3),
            "age": self.age,
            "active": self.is_active,
            "refutations": self.refutations_survived,
            "domains": len(self.domains_explored),
        }


class MultiLineInvestigator:
    """Investiga uma doenca mantendo N hipoteses concorrentes.

    Uso:
        inv = MultiLineInvestigator()
        inv.initialize("Alzheimer", filtered_hypotheses, disease_mechanisms)
        inv.run_round(filtered_hypotheses, explorer_findings)
        best = inv.get_best_line()
    """

    def __init__(self):
        self.lines: List[HypothesisLine] = []
        self._next_id: int = 0
        self._round: int = 0
        self._history: List[dict] = []

    def initialize(self, disease: str, filtered_hypotheses: list,
                   disease_mechanisms: List[str],
                   min_lines: int = None) -> int:
        """Inicializa N linhas de investigacao.

        N e DINAMICO: quantos mecanismos tem evidencia suficiente.
        Nao ha um numero fixo. Se so 2 mecanismos tiverem dados,
        so 2 linhas comecam.
        """
        self.lines.clear()
        self._next_id = 0
        self._round = 0

        # Mecanismo → intervencao padrao
        from core.cure_reasoner import CureReasoner
        mech_to_interv = CureReasoner.MECHANISM_TO_INTERVENTION

        for mech in disease_mechanisms:
            # Broad matching: mecanismo como substring OU palavra-chave no texto
            mech_lower = mech.lower()
            supporting = []
            for h in filtered_hypotheses:
                h_text = (h.cause + " " + h.effect).lower()
                if mech_lower in h_text:
                    supporting.append(h)
                # Tbm busca por palavras do mecanismo separadamente
                for word in mech_lower.replace("_", " ").split():
                    if len(word) > 3 and word in h_text:
                        supporting.append(h)
                        break

            if not supporting and min_lines is None:
                continue  # Sem evidencia → nao cria linha

            intervention = mech_to_interv.get(mech, f"target {mech}")
            avg_score = sum(h.quality_score for h in supporting) / max(1, len(supporting))

            line = HypothesisLine(
                id=self._next_id,
                mechanism=mech,
                intervention=intervention,
                target=mech,
                score=avg_score,
                evidence_ids=[id(h) for h in supporting[:10]],
            )
            self._next_id += 1
            self.lines.append(line)

        # Garante minimo de linhas se pedido
        while min_lines is not None and len(self.lines) < min_lines:
            fallback_mech = f"unknown_mechanism_{len(self.lines)}"
            line = HypothesisLine(
                id=self._next_id,
                mechanism=fallback_mech,
                intervention="exploration needed",
                target=fallback_mech,
                score=0.1,
            )
            self._next_id += 1
            self.lines.append(line)

        self._log_round("initialized")
        return len(self.lines)

    def run_round(self, filtered_hypotheses: list,
                  explorer_findings: list = None) -> List[dict]:
        """Uma rodada de atualizacao de todas as linhas.

        Returns: eventos desta rodada (nascimentos, mortes, fusoes)
        """
        self._round += 1
        events = []

        for line in self.lines:
            if not line.is_active:
                continue

            line.age += 1

            # 1. Atualiza score com base em novas evidencias
            mech_lower = line.mechanism.lower()
            new_evidence = []
            for h in filtered_hypotheses:
                h_text = (h.cause + " " + h.effect).lower()
                if mech_lower in h_text and id(h) not in line.evidence_ids:
                    new_evidence.append(h)

            # Adiciona ate 5 novas evidencias por rodada
            for h in new_evidence[:5]:
                line.evidence_ids.append(id(h))
                # Score sobe com qualidade da evidencia
                line.score = min(1.0, line.score + h.quality_score * 0.05)

            # 2. Se score caiu muito → abandona
            if line.score < 0.1 and line.age > 3:
                line.is_active = False
                events.append({"event": "death", "line": line.id,
                              "reason": "score_below_threshold", "age": line.age})

            # 3. Se estagnou → gera filha em dominio diferente (exploracao forcada)
            if line.stagnated and line.age > 5:
                # Soh gera filha se tiver achados do explorer
                if explorer_findings:
                    for finding in explorer_findings[:2]:
                        child = HypothesisLine(
                            id=self._next_id,
                            mechanism=f"{line.mechanism}+{finding.concept[:20]}",
                            intervention=f"{line.intervention} via {finding.domain}",
                            target=finding.concept[:30],
                            score=line.score * finding.relevance_to_disease,
                            parent_id=line.id,
                        )
                        self._next_id += 1
                        self.lines.append(child)
                        line.children_ids.append(child.id)
                        events.append({"event": "birth", "line": child.id,
                                      "parent": line.id,
                                      "from": finding.domain,
                                      "concept": finding.concept[:30]})

        self._log_round()

        # Criterio de parada DINAMICO:
        # Para se a linha lider estabilizou por 5+ rodadas
        return events

    def should_stop(self) -> bool:
        """Decisao DINAMICA de parada."""
        best = self.get_best_line()
        if best is None:
            return True

        # Nao para antes de 3 rodadas
        if self._round < 3:
            return False

        # Para se a melhor linha estiver ativa ha 10+ rodadas e nao gerou filhas
        if best.age >= 10 and not best.children_ids:
            return True

        # Para se todas as linhas estao mortas ou estagnadas
        active = [l for l in self.lines if l.is_active]
        if not active:
            return True

        # Para se o score da melhor linha estabilizou
        recent = [h for h in self._history[-5:] if "best_score" in h]
        if len(recent) >= 5:
            scores = [r["best_score"] for r in recent]
            if max(scores) - min(scores) < 0.01:
                return True

        return False

    def get_best_line(self) -> Optional[HypothesisLine]:
        """Linha com maior score."""
        active = [l for l in self.lines if l.is_active]
        if not active:
            return None
        return max(active, key=lambda l: l.score)

    def get_score_distribution(self) -> dict:
        scores = [l.score for l in self.lines if l.is_active]
        if not scores:
            return {"min": 0, "max": 0, "avg": 0, "n": 0}
        return {
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "avg": round(sum(scores) / len(scores), 3),
            "n": len(scores),
        }

    def _log_round(self, event_type: str = "round"):
        best = self.get_best_line()
        self._history.append({
            "round": self._round,
            "event": event_type,
            "active_lines": sum(1 for l in self.lines if l.is_active),
            "total_lines": len(self.lines),
            "best_score": best.score if best else 0,
            "best_mechanism": best.mechanism if best else "none",
        })

    def get_full_trace(self) -> List[dict]:
        return self._history

    def get_stats(self) -> dict:
        return {
            "total_lines_ever": len(self.lines),
            "active_lines": sum(1 for l in self.lines if l.is_active),
            "rounds": self._round,
            "best_line": self.get_best_line().to_dict() if self.get_best_line() else None,
            "score_distribution": self.get_score_distribution(),
        }
