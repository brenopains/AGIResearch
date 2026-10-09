"""
RecallPolicy — Decide se o recall e utilizavel ou deve ser ignorado.

Nao sabe como o EpisodeStore funciona. So recebe resultados e decide.
Retorna um RecallDecision com status EMPTY ou ACCEPT e a justificativa.
"""
from typing import List, Dict, Optional
from .contracts import RecallDecision, RecallStatus


class RecallPolicy:
    """Decide se um conjunto de resultados de recall e utilizavel."""

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def evaluate(self, results: List[Dict], method: str = "text") -> RecallDecision:
        """Avalia se os resultados do recall sao uteis.

        Args:
            results: lista bruta de episodios do EpisodeStore
            method: metodo usado ("text", "hnsw", etc.)

        Returns:
            RecallDecision com status EMPTY ou ACCEPT
        """
        if not results:
            return RecallDecision(
                status=RecallStatus.EMPTY,
                reason="NONE_FOUND",
                threshold=self.threshold,
            )

        # Encontra a melhor similaridade
        best_sim = max(r.get("score", 0) for r in results)

        if best_sim < self.threshold:
            reason = "LOW_SIMILARITY"
            if best_sim < 0.01:
                reason = "RANDOM_NOISE"

            return RecallDecision(
                status=RecallStatus.EMPTY,
                best_similarity=best_sim,
                threshold=self.threshold,
                reason=reason,
                n_results=len(results),
                episodes=[],
            )

        # Calcula coverage: quantos resultados tem similaridade util
        useful = [r for r in results if r.get("score", 0) > self.threshold * 0.5]
        coverage = len(useful) / max(1, len(results))

        return RecallDecision(
            status=RecallStatus.ACCEPT,
            best_similarity=best_sim,
            threshold=self.threshold,
            reason="OK",
            n_results=len(results),
            coverage=coverage,
            episodes=results[:min(5, len(results))],
        )
