from typing import Optional
from .contracts import CompetitionResult, Hypothesis, EvidenceSource


class Translator:
    """Le CompetitionResult e gera resposta textual.

    Nao le nos CGE diretamente. Resposta baseada na vencedora da competicao.
    """

    def __init__(self, include_audit: bool = True):
        self.include_audit = include_audit

    def translate(self, result: CompetitionResult, question: str = "") -> str:
        """Converte CompetitionResult em texto de resposta.

        Args:
            result: resultado da competicao de hipoteses
            question: pergunta original (opcional, para contexto)

        Returns:
            string formatada com resposta e auditoria
        """
        if not result or not result.winner:
            return self._no_answer(question)

        winner = self._find_hypothesis(result, result.winner)
        if not winner:
            return self._no_answer(question)

        response = self._build_answer(winner, result)
        if self.include_audit:
            audit = self._build_audit(result, winner)
            response += "\n\n" + audit

        return response.strip()

    def translate_short(self, result: CompetitionResult) -> str:
        """Resposta curta, so o essencial."""
        if not result or not result.winner:
            return "Nao foi possivel determinar."
        return f"{result.winner} (confianca: {result.winner_confidence:.2%})"

    def _build_answer(self, winner: Hypothesis, result: CompetitionResult) -> str:
        lines = [winner.name]

        if winner.description:
            lines.append(winner.description)

        # Evidencias mais fortes (top 3)
        evidence_lines = []
        for ev in sorted(winner.evidence_for, key=lambda e: e.confidence, reverse=True)[:3]:
            label = ev.source.value.capitalize()
            evidence_lines.append(f"  [{label}] {ev.trace[:100]} (conf: {ev.confidence:.2f})")

        if evidence_lines:
            lines.append("\nEvidencias:")
            lines.extend(evidence_lines)

        # Mencoes a contra-evidencias
        if winner.evidence_against:
            top_against = max(winner.evidence_against, key=lambda e: e.confidence)
            lines.append(f"\nContra-evidencia: {top_against.trace[:80]} (conf: {top_against.confidence:.2f})")

        return "\n".join(lines)

    def _build_audit(self, result: CompetitionResult, winner: Hypothesis) -> str:
        lines = ["--- Auditoria ---"]

        # Ranking
        lines.append(f"Vencedora: {result.winner} ({result.winner_confidence:.2%})")
        if result.runner_up:
            lines.append(f"Runner-up: {result.runner_up} ({result.runner_up_confidence:.2%})")
        lines.append(f"Margem: {result.margin:.3f} | Entropia: {result.entropy:.3f}")

        # Estabilidade
        if result.stability:
            trend = "subindo" if len(result.stability) > 1 and result.stability[-1] > result.stability[0] else "estavel"
            lines.append(f"Estabilidade: {result.stability} ({trend})")

        # Por categoria
        for cat, hyp_list in result.by_category.items():
            names = [h.get("name", "?") for h in hyp_list[:4]]
            lines.append(f"  {cat}: {', '.join(names)}")

        # Contrafactuais
        if result.counterfactuals:
            lines.append("Contrafactuais:")
            for cf in result.counterfactuals[:2]:
                reasons = "; ".join(cf.get("reasons_discarded", []))
                lines.append(f"  - {cf['hypothesis']}: {reasons[:100]}")

        return "\n".join(lines)

    def _no_answer(self, question: str) -> str:
        if question:
            return f"Nao encontrei resposta para: {question}"
        return "Nao foi possivel determinar uma resposta."

    def _find_hypothesis(self, result: CompetitionResult, name: str) -> Optional[Hypothesis]:
        for h in result.hypotheses:
            if h.name == name:
                return h
        return None
