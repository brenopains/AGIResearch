import math
import copy
import numpy as np
from typing import List, Optional, Dict, Tuple
from .contracts import (
    Hypothesis, EvidenceItem, CompetitionResult, EvidenceSource,
    ContributionBreakdown, ConvergenceStep,
)


class Competition:
    """Competicao de Hipoteses com rastreabilidade profunda.

    A cada iteracao, registra:
      - score de TODAS as hipoteses (nao so da vencedora)
      - decomposition: quanto cada fonte contribuiu
      - diversidade: quantos clusters de solucao existem
      - reversoes: se o vencedor mudou entre iteracoes

    Nao sabe a origem das evidencias. So recebe listas de Hypothesis.
    """

    def __init__(self, iterations: int = 3, diversity_threshold: float = 0.15):
        self.iterations = iterations
        self.diversity_threshold = diversity_threshold
        self._history: List[Dict[str, float]] = []
        self._convergence_steps: List[ConvergenceStep] = []

    def run(self, hypotheses: List[Hypothesis]) -> CompetitionResult:
        """Executa a competicao completa por N iteracoes."""
        if not hypotheses:
            return CompetitionResult(hypotheses=[])

        hypotheses = copy.deepcopy(hypotheses)
        self._history = []
        self._convergence_steps = []

        # Inicializa contribution_breakdown para cada hipotese
        self._init_breakdowns(hypotheses)

        for i in range(self.iterations):
            self._update_confidences(hypotheses)
            self._update_breakdowns(hypotheses)
            snapshot = {h.name: h.confidence for h in hypotheses}
            self._history.append(snapshot)

            # Salva ConvergenceStep completo
            sorted_h = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
            winner = sorted_h[0] if sorted_h else None
            runner_up = sorted_h[1] if len(sorted_h) > 1 else None
            margin = (winner.confidence - runner_up.confidence) if winner and runner_up else 0.0
            entropy = self._compute_entropy(sorted_h)

            self._convergence_steps.append(ConvergenceStep(
                iteration=i,
                scores={h.name: h.confidence for h in sorted_h},
                winner=winner.name if winner else "",
                margin=margin,
                entropy=entropy,
            ))

        result = self._build_result(hypotheses)

        # Metricas de convergencia
        result.convergence_steps = self._convergence_steps
        result.diversity_score = self._compute_diversity(hypotheses)
        result.num_clusters = self._compute_num_clusters(hypotheses)
        result.winner_reversals = self._count_winner_reversals()
        result.convergence_velocity = self._compute_convergence_velocity()

        return result

    def _init_breakdowns(self, hypotheses: List[Hypothesis]):
        """Inicializa contribution_breakdown de cada hipotese com base nas evidencias."""
        all_sources = list(EvidenceSource)

        for h in hypotheses:
            breakdown: List[ContributionBreakdown] = []
            for src in all_sources:
                items_for = [e for e in h.evidence_for if e.source == src]
                if not items_for:
                    continue
                total_score = sum(e.confidence for e in items_for)
                avg_score = total_score / len(items_for)
                top_items = sorted(items_for, key=lambda e: e.confidence, reverse=True)
                detail = "; ".join(t.trace[:60] for t in top_items[:2]) if top_items else src.value
                breakdown.append(ContributionBreakdown(
                    source=src,
                    score=avg_score,
                    weight=1.0,
                    detail=detail,
                    n_items=len(items_for),
                ))
            h.contribution_breakdown = breakdown

    def _update_confidences(self, hypotheses: List[Hypothesis]):
        """Atualiza confianca de cada hipotese baseado nas evidencias."""
        for h in hypotheses:
            for_val = sum(e.confidence for e in h.evidence_for)
            against_val = sum(e.confidence for e in h.evidence_against)

            h.evidence_total = for_val

            raw = for_val / max(1, for_val + against_val + 0.1)
            sources = set(e.source.value for e in h.evidence_for)
            diversity_bonus = len(sources) * 0.03

            h.confidence = min(1.0, raw + diversity_bonus)

            by_type: Dict[str, dict] = {}
            for e in h.evidence_for:
                key = e.source.value
                if key not in by_type:
                    by_type[key] = {"count": 0, "total_confidence": 0.0, "best": 0.0}
                by_type[key]["count"] += 1
                by_type[key]["total_confidence"] += e.confidence
                by_type[key]["best"] = max(by_type[key]["best"], e.confidence)
            h.evidence_by_type = by_type

    def _update_breakdowns(self, hypotheses: List[Hypothesis]):
        """Atualiza contribution_breakdown com scores calibrados apos cada iteracao."""
        for h in hypotheses:
            for cb in h.contribution_breakdown:
                # Atualiza weight baseado no numero de itens e importancia relativa
                total_items = sum(b.n_items for b in h.contribution_breakdown) or 1
                cb.weight = cb.n_items / total_items
                # Score reflete a contribuicao relativa desta fonte
                cb.score = cb.score * cb.weight

    def _compute_entropy(self, hypotheses: List[Hypothesis]) -> float:
        """Entropia de Shannon da distribuicao de confianca."""
        if not hypotheses:
            return 0.0
        total = sum(h.confidence for h in hypotheses) or 1
        entropy = 0.0
        for h in hypotheses:
            p = h.confidence / total
            if p > 0:
                entropy -= p * math.log2(p)
        n = len(hypotheses)
        max_entropy = math.log2(n) if n > 1 else 1.0
        return round(entropy / max_entropy, 4)

    def _name_hash(self, name: str) -> np.ndarray:
        """Gera um vetor de features a partir do nome da hipotese.

        Usa hash code da string para produzir um vetor deterministico
        que diferencia hipoteses com nomes diferentes.
        """
        h = abs(hash(name))
        vec = np.zeros(16, dtype=np.float32)
        for i in range(16):
            vec[i] = (h >> (i * 4)) & 0xF
        return vec / 16.0

    def _build_feature(self, h: Hypothesis) -> np.ndarray:
        """Feature vector combinando contribution_breakdown + nome da hipotese."""
        all_sources = [s.value for s in EvidenceSource]
        # Contribution breakdown (score por fonte)
        contrib = [0.0] * len(all_sources)
        for cb in h.contribution_breakdown:
            idx = all_sources.index(cb.source.value)
            contrib[idx] = cb.score
        # Nome da hipotese (hash)
        name_feat = self._name_hash(h.name)
        return np.concatenate([np.array(contrib, dtype=np.float32), name_feat])

    def _compute_diversity(self, hypotheses: List[Hypothesis]) -> float:
        """Diversidade entre hipoteses via similaridade de seus vetores de features.

        Combina contribution_breakdown + nome da hipotese.
        0 = todas identicas, 1 = todas ortogonais.
        """
        if len(hypotheses) < 2:
            return 0.0

        features = [self._build_feature(h) for h in hypotheses]

        n = len(features)
        sim_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                norm_i = np.linalg.norm(features[i]) or 1
                norm_j = np.linalg.norm(features[j]) or 1
                cos_sim = np.dot(features[i], features[j]) / (norm_i * norm_j)
                sim_sum += cos_sim
                pair_count += 1

        avg_sim = sim_sum / max(1, pair_count)
        diversity = 1.0 - avg_sim
        return round(diversity, 4)

    def _compute_num_clusters(self, hypotheses: List[Hypothesis]) -> int:
        """Numero de clusters de solucao via threshold de similaridade.

        Hipoteses com mesmo tipo de solucao (mesmo nome base) sao agrupadas.
        """
        if len(hypotheses) < 2:
            return 1

        features = [self._build_feature(h) for h in hypotheses]
        n = len(features)
        assignments = [-1] * n
        next_cluster = 0

        for i in range(n):
            if assignments[i] >= 0:
                continue
            assignments[i] = next_cluster
            for j in range(i + 1, n):
                if assignments[j] >= 0:
                    continue
                norm_i = np.linalg.norm(features[i]) or 1
                norm_j = np.linalg.norm(features[j]) or 1
                cos_sim = np.dot(features[i], features[j]) / (norm_i * norm_j)
                if cos_sim > (1.0 - self.diversity_threshold):
                    assignments[j] = assignments[i]
            next_cluster += 1

        return next_cluster

    def _count_winner_reversals(self) -> int:
        """Quantas vezes o vencedor mudou entre iteracoes consecutivas."""
        if len(self._history) < 2:
            return 0
        reversals = 0
        prev_winner = max(self._history[0], key=self._history[0].get)
        for snap in self._history[1:]:
            curr_winner = max(snap, key=snap.get)
            if curr_winner != prev_winner:
                reversals += 1
                prev_winner = curr_winner
        return reversals

    def _compute_convergence_velocity(self) -> float:
        """Velocidade de convergencia: (score_final - score_inicial) / n_iters.

        Positivo = hipotese vencedora ganhou confianca.
        Negativo = perdeu confianca (divergencia).
        """
        if len(self._history) < 2:
            return 0.0
        first = self._history[0]
        last = self._history[-1]
        # Identifica a vencedora final
        final_winner = max(last, key=last.get)
        if final_winner not in first:
            return 0.0
        delta = last[final_winner] - first[final_winner]
        return round(delta / len(self._history), 4)

    def _build_result(self, hypotheses: List[Hypothesis]) -> CompetitionResult:
        """Monta CompetitionResult com todas as metricas."""
        sorted_h = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

        by_category: Dict[str, List[dict]] = {}
        for h in sorted_h:
            cat = h.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(h.to_dict())

        winner = sorted_h[0] if sorted_h else None
        runner_up = sorted_h[1] if len(sorted_h) > 1 else None

        margin = 0.0
        if winner and runner_up:
            margin = winner.confidence - runner_up.confidence

        entropy = self._compute_entropy(sorted_h)
        stability = self._compute_stability(winner.name if winner else "")
        counterfactuals = self._build_counterfactuals(sorted_h)

        return CompetitionResult(
            hypotheses=sorted_h,
            winner=winner.name if winner else None,
            winner_confidence=winner.confidence if winner else 0.0,
            runner_up=runner_up.name if runner_up else None,
            runner_up_confidence=runner_up.confidence if runner_up else 0.0,
            margin=margin,
            entropy=entropy,
            stability=stability,
            by_category=by_category,
            counterfactuals=counterfactuals,
        )

    def _compute_stability(self, winner_name: str) -> List[float]:
        if not self._history:
            return []
        return [
            round(snap.get(winner_name, 0), 4)
            for snap in self._history
        ]

    def _build_counterfactuals(self, sorted_h: List[Hypothesis]) -> List[dict]:
        if len(sorted_h) < 2:
            return []

        winner = sorted_h[0]
        counterfactuals = []

        for runner in sorted_h[1:4]:
            reasons = []

            winner_sources = set(e.source.value for e in winner.evidence_for)
            runner_sources = set(e.source.value for e in runner.evidence_for)
            missing = winner_sources - runner_sources
            if missing:
                reasons.append(f"runner-up lacks evidence from: {', '.join(missing)}")

            winner_against = sum(e.confidence for e in winner.evidence_against)
            runner_against = sum(e.confidence for e in runner.evidence_against)
            if runner_against > winner_against + 0.1:
                reasons.append(f"more counter-evidence ({runner_against:.2f} vs {winner_against:.2f})")

            diff = winner.confidence - runner.confidence
            if diff > 0.1:
                reasons.append(f"confidence gap: {diff:.3f}")

            if not reasons:
                reasons.append("insufficient evidence total")

            counterfactuals.append({
                "hypothesis": runner.name,
                "category": runner.category,
                "confidence": runner.confidence,
                "reasons_discarded": reasons,
                "what_would_help": f"more {', '.join(list(winner_sources - runner_sources)[:2]) or 'diverse'} evidence",
            })

        return counterfactuals
