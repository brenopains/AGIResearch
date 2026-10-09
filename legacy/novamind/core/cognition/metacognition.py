"""Meta-cognicao: auto-monitoramento do processo de pensamento.

Detecta:
  - Reversoes: quando e por que o sistema mudou de ideia
  - Impasse: quando o sistema esta preso (entropia alta, margem baixa)
  - Curiosidade: lacunas de conhecimento identificadas
  - Estilo de raciocinio predominante
"""
from typing import List, Dict, Optional
from datetime import datetime
from .contracts import CompetitionResult, ConvergenceStep


class MetaCognition:
    """Monitora o proprio processo de raciocinio do sistema.

    Nao influencia decisoes — apenas observa e registra.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.reversals: List[Dict] = []
        self.impasse_events: List[Dict] = []
        self.curiosity_questions: List[str] = []
        self._previous_winners: List[str] = []

    def observe(self, focus: str, result: CompetitionResult, step: int = 0) -> Dict:
        """Observa o resultado de uma competicao e extrai meta-informacao.

        Returns:
            dict com meta_observations do ciclo
        """
        meta = {
            "step": step,
            "focus": focus,
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Deteccao de reversao
        reversal = self._check_reversal(result)
        if reversal:
            self.reversals.append(reversal)
            meta["reversal"] = reversal
        else:
            meta["reversal"] = {"reversed": False}

        # 2. Deteccao de impasse
        impasse = self._check_impasse(result)
        if impasse["is_impasse"]:
            self.impasse_events.append(impasse)
        meta["impasse"] = impasse

        # 3. Classificacao do estilo de raciocinio
        style = self._classify_reasoning_style(result)
        meta["reasoning_style"] = style

        # 4. Curiosidade (lacunas)
        gaps = self._detect_knowledge_gaps(result)
        meta["knowledge_gaps"] = gaps

        return meta

    def _check_reversal(self, result: CompetitionResult) -> Optional[Dict]:
        """Detecta se o vencedor atual e diferente do anterior."""
        current = result.winner
        if not current:
            return None

        if self._previous_winners and self._previous_winners[-1] != current:
            prev = self._previous_winners[-1]
            # Tenta encontrar a causa
            reason = self._find_reversal_cause(result, prev, current)
            reversal = {
                "reversed": True,
                "from": prev,
                "to": current,
                "margin_before": result.margin,
                "entropy": result.entropy,
                "reason": reason,
                "diversity": result.diversity_score,
            }
            self._previous_winners.append(current)
            return reversal

        self._previous_winners.append(current)
        return None

    def _find_reversal_cause(self, result: CompetitionResult, prev: str, curr: str) -> str:
        """Tenta identificar qual evidencia causou a reversao."""
        # Procura a hipotese vencedora atual e verifica se ganhou devido a alguma fonte nova
        curr_h = next((h for h in result.hypotheses if h.name == curr), None)
        prev_h = next((h for h in result.hypotheses if h.name == prev), None)

        if curr_h and prev_h:
            # Compara contribution_breakdown
            curr_sources = {cb.source.value: cb.score for cb in curr_h.contribution_breakdown}
            prev_sources = {cb.source.value: cb.score for cb in prev_h.contribution_breakdown}

            gaps = []
            for src, score in curr_sources.items():
                prev_score = prev_sources.get(src, 0)
                if score > prev_score + 0.1:
                    gaps.append(f"{src} ({score:.2f} vs {prev_score:.2f})")

            if gaps:
                return f"new evidence from: {', '.join(gaps)}"

        return "evidence re-evaluation"

    def _check_impasse(self, result: CompetitionResult) -> Dict:
        """Detecta se o sistema esta em impasse (nao consegue decidir).

        Criterios:
          - Entropia > 0.9 (dispersao maxima)
          - Margem < 0.02 (quase empate)
          - Diversidade < 0.1 (hipoteses quase identicas)
          - winner_reversals > 2 (instabilidade)
        """
        is_impasse = False
        reasons = []

        if result.entropy > 0.9:
            reasons.append(f"entropy={result.entropy:.3f}>0.9")
        if result.margin < 0.02:
            reasons.append(f"margin={result.margin:.4f}<0.02")
        if result.diversity_score < 0.1 and len(result.hypotheses) > 1:
            reasons.append(f"diversity={result.diversity_score:.3f}<0.1")
        if result.winner_reversals > 2:
            reasons.append(f"reversals={result.winner_reversals}>2")
        if result.stability and len(result.stability) > 1 and result.stability[-1] < result.stability[0] * 0.8:
            reasons.append("winner losing confidence")

        if len(reasons) >= 2:
            is_impasse = True

        return {
            "is_impasse": is_impasse,
            "reasons": reasons,
            "entropy": float(result.entropy),
            "margin": float(result.margin),
            "diversity": float(result.diversity_score),
        }

    def _classify_reasoning_style(self, result: CompetitionResult) -> Dict[str, float]:
        """Classifica o estilo de raciocinio baseado nas fontes de evidencia.

        Estilos:
          - deductive: dominado por taxonomia e logica
          - analogical: dominado por episodios e similaridade
          - visual: dominado por percepcao
          - causal: dominado por relacoes causais
          - probabilistic: multiplas fontes com confianca similar
        """
        if not result.hypotheses:
            return {"unknown": 1.0}

        # Agrega contribution_breakdown de todas as hipoteses
        total_scores: Dict[str, float] = {}
        for h in result.hypotheses[:3]:
            for cb in h.contribution_breakdown:
                key = cb.source.value
                total_scores[key] = total_scores.get(key, 0) + cb.score

        total = sum(total_scores.values()) or 1
        style = {k: round(v / total, 4) for k, v in total_scores.items()}

        # Nome do estilo dominante
        if style:
            dominant = max(style, key=style.get)
            style["dominant"] = dominant

        return style

    def _detect_knowledge_gaps(self, result: CompetitionResult) -> List[str]:
        """Identifica lacunas de conhecimento baseado em hipoteses com baixa confianca."""
        gaps = []
        for h in result.hypotheses:
            if h.confidence < 0.3 and h.evidence_total < 0.5:
                gaps.append(f"baixa confianca em '{h.name}': poucas evidencias ({h.evidence_total:.2f})")
        return gaps[:5]

    def log_curiosity(self, question: str):
        """Registra uma pergunta de curiosidade (lacuna identificada)."""
        if question not in self.curiosity_questions:
            self.curiosity_questions.append(question)
            if len(self.curiosity_questions) > self.max_history:
                self.curiosity_questions = self.curiosity_questions[-self.max_history:]

    def get_curiosity_questions(self) -> List[str]:
        return self.curiosity_questions[-10:]

    def get_reversal_history(self) -> List[Dict]:
        return self.reversals[-20:]

    def get_impasse_rate(self) -> float:
        """Proporcao de ciclos que resultaram em impasse."""
        if not self.impasse_events:
            return 0.0
        total = len(self.reversals) + len(self.impasse_events) + 1
        return len(self.impasse_events) / total

    def summarize_session(self) -> Dict:
        """Sumario completo da sessa o de metacognicao."""
        return {
            "total_reversals": int(len(self.reversals)),
            "total_impasse_events": int(len(self.impasse_events)),
            "impasse_rate": float(round(self.get_impasse_rate(), 4)),
            "curiosity_questions_count": int(len(self.curiosity_questions)),
            "recent_reversals": self.reversals[-3:],
            "recent_curiosity": self.curiosity_questions[-3:],
        }
