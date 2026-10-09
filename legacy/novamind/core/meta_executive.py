"""
MetaExecutive — Orquestrador de exploracao vs convergencia.

Monitora o progresso do raciocinio e decide dinamicamente:
  - CONTINUAR refinando a linha atual
  - EXPLORAR dominios distantes (se convergiu rapido demais)
  - ABANDONAR linha (se refutada ou estagnada)
  - CONVERGIR para resposta final (se hipotese robusta)

Nao ha numeros fixos. Cada decisao e baseada em:
  - Velocidade de convergencia
  - Densidade de contraexemplos
  - Diversidade de dominios explorados
  - Forca de refutacao sobrevivida
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class ExecutiveDecision:
    """Decisao do MetaExecutive."""
    action: str  # "refine", "explore", "abandon", "converge"
    reason: str
    target_line_id: Optional[int] = None
    target_domain: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason[:60],
            "target_domain": self.target_domain,
            "confidence": round(self.confidence, 2),
        }


class MetaExecutive:
    """Orquestrador do ciclo gerar → testar → refinar → convergir.

    Uso:
        exec = MetaExecutive()
        decision = exec.tick(multi_line_stats, adversary_stats, exploration_stats)
        if decision.action == "converge":
            print("Hipotese final:", decision.confidence)
    """

    def __init__(self):
        self.decisions: List[ExecutiveDecision] = []
        self._round: int = 0
        self._convergence_counter: int = 0
        self._exploration_counter: int = 0
        self._abandon_counter: int = 0

        # Rastreamento historico de scores
        self._score_history: List[float] = []

    def tick(self, multi_line_stats: dict,
             adversary_stats: dict,
             exploration_stats: dict) -> ExecutiveDecision:
        """Um tick do meta-executivo. Decide o que fazer.

        Logica de decisao DINAMICA:
          1. Se melhor linha foi refutada → explora
          2. Se melhor linha convergiu rapido → explora (suspeito)
          3. Se melhor linha sobreviveu a refutacao → refine
          4. Se nenhuma linha ativa → converge com fallback
          5. Se score estagnou por multiplas rodadas → converge
        """
        self._round += 1

        best_score = (multi_line_stats.get("best_line") or {}).get("score", 0)
        active_lines = multi_line_stats.get("active_lines", 0)

        if best_score is not None:
            self._score_history.append(best_score)

        # Decisao 1: REFUTACAO → abandonar linha?
        adv_survived = adversary_stats.get("survived", 0)
        adv_refuted = adversary_stats.get("refuted", 0)

        if adv_refuted > adv_survived and active_lines > 1:
            self._abandon_counter += 1
            decision = ExecutiveDecision(
                action="abandon",
                reason=f"Most hypotheses refuted ({adv_refuted}/{adv_survived+adv_refuted})",
                confidence=max(0, best_score - 0.2) if best_score else 0.1,
            )
            self.decisions.append(decision)
            return decision

        # Decisao 2: CONVERGENCIA RAPIDA → suspeito, explorar
        if len(self._score_history) >= 4:
            recent = self._score_history[-4:]
            # Se convergiu em < 4 rodadas (score variou < 1%)
            variance = max(recent) - min(recent)
            if variance < 0.01:
                self._convergence_counter += 1
                if self._convergence_counter <= 1:
                    # So suspect na primeira vez
                    decision = ExecutiveDecision(
                        action="explore",
                        reason=f"Converged too quickly ({self._round} rounds, variance={variance:.3f}). "
                               f"Known answer detected, forcing exploration.",
                        confidence=best_score * 0.7 if best_score else 0.2,
                    )
                    self.decisions.append(decision)
                    return decision

        # Decisao 3: SOBREVIVEU A REFUTACAO → refine
        if adv_survived > 0 and adv_refuted == 0:
            decision = ExecutiveDecision(
                action="refine",
                reason=f"Hypothesis survived refutation. Strengthening.",
                confidence=min(1.0, (best_score or 0) + 0.1),
            )
            self.decisions.append(decision)
            return decision

        # Decisao 4: ESTAGNOU → converge
        if len(self._score_history) >= 8:
            recent = self._score_history[-8:]
            variance = max(recent) - min(recent)
            if variance < 0.01:
                decision = ExecutiveDecision(
                    action="converge",
                    reason=f"Score stabilized over {len(recent)} rounds (variance={variance:.4f}). "
                           f"Converging to final hypothesis.",
                    confidence=best_score or 0.3,
                )
                self.decisions.append(decision)
                return decision

        # Decisao 5: PADRAO → explore se ativo, refine se inativo
        if active_lines > 0:
            self._exploration_counter += 1
            if self._exploration_counter % 3 == 0:
                decision = ExecutiveDecision(
                    action="explore",
                    reason=f"Periodic exploration (round {self._round})",
                    confidence=best_score or 0.2,
                )
            else:
                decision = ExecutiveDecision(
                    action="refine",
                    reason=f"Continuing refinement (round {self._round})",
                    confidence=best_score or 0.2,
                )
        else:
            decision = ExecutiveDecision(
                action="abandon",
                reason="No active hypotheses remaining",
                confidence=0.1,
            )

        self.decisions.append(decision)
        return decision

    def get_trace(self) -> List[dict]:
        return [d.to_dict() for d in self.decisions]

    def get_stats(self) -> dict:
        return {
            "total_decisions": len(self.decisions),
            "final_action": self.decisions[-1].action if self.decisions else "none",
            "refine_count": sum(1 for d in self.decisions if d.action == "refine"),
            "explore_count": sum(1 for d in self.decisions if d.action == "explore"),
            "abandon_count": sum(1 for d in self.decisions if d.action == "abandon"),
            "converge_count": sum(1 for d in self.decisions if d.action == "converge"),
        }
