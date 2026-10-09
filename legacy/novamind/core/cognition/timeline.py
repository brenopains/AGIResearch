import time
import math
from typing import List, Dict, Optional, Any
from .contracts import CompetitionResult, ConvergenceStep


class Timeline:
    """Registra e analisa a evolucao do pensamento por iteracao.

    Cada entrada contem:
      - focus, confidence, prediction_error, sources, timestamp
      - contribution_breakdown de cada hipotese
      - convergence_steps (score de cada hipotese por iteracao)
      - attention_timeline (conceitos ativos)
      - recall_quality (redundancia/novidade)
      - prediction_error_map (erro por modalidade)
      - meta_reversal (mudanca de vencedor)
      - reasoning_style (padroes de raciocinio)
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: List[Dict] = []
        self._iteration = 0

    def record(
        self,
        focus: str,
        result: CompetitionResult,
        prediction_error: float = 0.0,
        attention_timeline: Optional[List[Dict]] = None,
        recall_quality: Optional[Dict] = None,
        prediction_error_map: Optional[Dict[str, float]] = None,
        reasoning_style: Optional[Dict[str, float]] = None,
    ):
        """Registra um frame completo do pensamento."""
        entry: Dict[str, Any] = {
            "iteration": self._iteration,
            "timestamp": time.time(),
            "focus": focus,
            "winner": result.winner or "",
            "winner_confidence": result.winner_confidence,
            "margin": result.margin,
            "entropy": result.entropy,
            "n_hypotheses": len(result.hypotheses),
            "prediction_error": round(prediction_error, 4),
            "stability": result.stability[-1] if result.stability else 0.0,
            "sources": self._extract_sources(result),
            "diversity_score": float(result.diversity_score),
            "num_clusters": int(result.num_clusters),
            "winner_reversals": int(result.winner_reversals),
            "convergence_velocity": float(result.convergence_velocity),
        }

        # Contribution breakdown de cada hipotese (top 3)
        entry["hypotheses_detail"] = self._extract_hypotheses_detail(result)

        # Convergence steps (score por hipotese ao longo das iteracoes)
        entry["convergence_steps"] = [
            {
                "iteration": cs.iteration,
                "winner": cs.winner,
                "margin": round(cs.margin, 4),
                "entropy": round(cs.entropy, 4),
                "scores": {k: round(v, 4) for k, v in cs.scores.items()},
                "score_delta": self._compute_delta(cs, result.convergence_steps),
            }
            for cs in result.convergence_steps
        ]

        # Attention timeline (conceitos ativos neste ciclo)
        if attention_timeline:
            entry["attention_timeline"] = attention_timeline
        else:
            entry["attention_timeline"] = []

        # Recall quality metrics
        if recall_quality:
            entry["recall_quality"] = recall_quality
        else:
            entry["recall_quality"] = {}

        # Prediction error map
        if prediction_error_map:
            entry["prediction_error_map"] = {
                k: round(v, 4) for k, v in prediction_error_map.items()
            }
        else:
            entry["prediction_error_map"] = {}

        # Reasoning style
        if reasoning_style:
            entry["reasoning_style"] = {
                k: round(v, 4) for k, v in reasoning_style.items()
            }
        else:
            entry["reasoning_style"] = {}

        # Meta-reversal detection
        entry["meta_reversal"] = self._detect_reversal(entry)

        self.entries.append(entry)
        self._iteration += 1

        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_timeline(self, n_last: int = 20) -> List[Dict]:
        return self.entries[-n_last:]

    def get_confidence_trace(self, hypothesis_name: str) -> List[float]:
        trace = []
        for e in self.entries:
            if e.get("winner") == hypothesis_name:
                trace.append(e.get("winner_confidence", 0))
        return trace

    def get_prediction_error_trace(self) -> List[float]:
        return [e.get("prediction_error", 0) for e in self.entries]

    def get_entropy_trace(self) -> List[float]:
        return [e.get("entropy", 0) for e in self.entries]

    def get_diversity_trace(self) -> List[float]:
        return [e.get("diversity_score", 0) for e in self.entries]

    def get_convergence_detail(self, n_last: int = 5) -> List[Dict]:
        """Retorna detalhes de convergencia dos ultimos N registros."""
        detail = []
        for e in self.entries[-n_last:]:
            steps = e.get("convergence_steps", [])
            if steps:
                detail.append({
                    "iteration": e["iteration"],
                    "focus": e["focus"],
                    "steps": steps,
                })
        return detail

    def get_winner_evolution(self) -> List[Dict]:
        """Trajetoria do vencedor: como evoluiu entre iteracoes."""
        evolution = []
        for e in self.entries:
            steps = e.get("convergence_steps", [])
            if len(steps) >= 2:
                first = steps[0]
                last = steps[-1]
                winner_trace = {}
                for h_name in first["scores"]:
                    scores = [s["scores"].get(h_name, 0) for s in steps]
                    winner_trace[h_name] = scores
                evolution.append({
                    "iteration": e["iteration"],
                    "focus": e["focus"],
                    "winner": e["winner"],
                    "winner_trace": winner_trace,
                })
        return evolution

    def get_attention_summary(self) -> Dict:
        """Sumario dos conceitos mais ativos."""
        concept_time: Dict[str, float] = {}
        for e in self.entries:
            for attn in e.get("attention_timeline", []):
                concept = attn.get("concept", "")
                activation = attn.get("activation", 0)
                if concept:
                    concept_time[concept] = concept_time.get(concept, 0) + activation
        sorted_c = sorted(concept_time.items(), key=lambda x: -x[1])[:20]
        total = sum(c for _, c in sorted_c) or 1
        return {
            "concepts": [c for c, _ in sorted_c],
            "total_activation": [round(c / total, 4) for _, c in sorted_c],
        }

    def get_recall_summary(self) -> Dict:
        """Sumario da qualidade do recall."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for e in self.entries:
            rq = e.get("recall_quality", {})
            for k, v in rq.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + v
                    counts[k] = counts.get(k, 1) + 1
        return {k: round(v / max(1, counts.get(k, 1)), 4) for k, v in totals.items()}

    def get_stability_summary(self) -> Dict:
        if len(self.entries) < 2:
            return {"status": "insufficient_data"}

        n = len(self.entries)
        confs = [float(e.get("winner_confidence", 0)) for e in self.entries]
        entropies = [float(e.get("entropy", 0)) for e in self.entries]
        diversities = [float(e.get("diversity_score", 0)) for e in self.entries]

        return {
            "n_iterations": n,
            "confidence_mean": round(sum(confs) / n, 4),
            "confidence_std": round(self._std(confs), 4),
            "entropy_mean": round(sum(entropies) / n, 4),
            "diversity_mean": round(sum(diversities) / n, 4),
            "convergence_trend": "converging" if entropies[-1] < entropies[0] * 0.8 else "stable",
            "winner_consistency": self._winner_consistency(),
            "total_reversals": int(sum(e.get("winner_reversals", 0) for e in self.entries)),
        }

    def get_heatmap_data(self, top_n_concepts: int = 20) -> Dict:
        winners = [e.get("winner", "") for e in self.entries if e.get("winner")]
        concept_counts: Dict[str, float] = {}
        for w in winners:
            concept_counts[w] = concept_counts.get(w, 0) + 1

        sorted_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:top_n_concepts]
        total = sum(c for _, c in sorted_concepts) or 1

        return {
            "concepts": [c for c, _ in sorted_concepts],
            "frequencies": [round(c / total, 4) for _, c in sorted_concepts],
            "time_span": len(self.entries),
            "unique_winners": len(set(winners)),
        }

    def get_decomposition_history(self) -> List[Dict]:
        source_timeline: List[Dict] = []
        for e in self.entries[-50:]:
            source_timeline.append({
                "iteration": e["iteration"],
                "sources": e.get("sources", {}),
                "diversity": e.get("diversity_score", 0),
                "clusters": e.get("num_clusters", 1),
            })
        return source_timeline

    def clear(self):
        self.entries.clear()
        self._iteration = 0

    def _extract_sources(self, result: CompetitionResult) -> Dict[str, float]:
        sources: Dict[str, float] = {}
        for h in result.hypotheses[:3]:
            for ev in h.evidence_for:
                key = ev.source.value
                sources[key] = sources.get(key, 0) + ev.confidence
        total = sum(sources.values()) or 1
        return {k: round(v / total, 4) for k, v in sources.items()}

    def _extract_hypotheses_detail(self, result: CompetitionResult) -> List[Dict]:
        detail = []
        for h in result.hypotheses[:5]:
            breakdown = [
                {"source": cb.source.value, "score": round(cb.score, 4), "weight": round(cb.weight, 4)}
                for cb in h.contribution_breakdown
            ]
            detail.append({
                "name": h.name,
                "confidence": round(h.confidence, 4),
                "category": h.category,
                "contribution_breakdown": breakdown,
            })
        return detail

    def _compute_delta(self, step: ConvergenceStep, all_steps: List[ConvergenceStep]) -> Dict:
        """Diferenca de score em relacao ao passo anterior."""
        if not all_steps:
            return {}
        idx = all_steps.index(step)
        if idx == 0:
            return {k: 0.0 for k in step.scores}
        prev = all_steps[idx - 1]
        delta = {}
        for k in step.scores:
            prev_v = prev.scores.get(k, 0)
            delta[k] = round(step.scores[k] - prev_v, 4)
        return delta

    def _detect_reversal(self, entry: Dict) -> Dict:
        """Detecta se houve reversao em relacao a entrada anterior."""
        if len(self.entries) < 1:
            return {"reversed": False, "from": "", "to": "", "reason": "first_iteration"}
        prev = self.entries[-1]
        prev_winner = prev.get("winner", "")
        curr_winner = entry.get("winner", "")
        if prev_winner and curr_winner and prev_winner != curr_winner:
            return {
                "reversed": True,
                "from": prev_winner,
                "to": curr_winner,
                "reason": self._guess_reversal_reason(entry),
            }
        return {"reversed": False, "from": prev_winner, "to": curr_winner, "reason": "stable"}

    def _guess_reversal_reason(self, entry: Dict) -> str:
        """Tenta inferir porque houve reversao."""
        steps = entry.get("convergence_steps", [])
        if len(steps) >= 2:
            for cs in steps:
                if cs.get("score_delta"):
                    for h_name, delta in cs["score_delta"].items():
                        if abs(delta) > 0.1:
                            return f"sharp score change: {h_name} changed by {delta:.2f}"
        return "evidence re-evaluation"

    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _winner_consistency(self) -> float:
        winners = [e.get("winner", "") for e in self.entries if e.get("winner")]
        if not winners:
            return 0.0
        most_common = max(set(winners), key=winners.count)
        return round(winners.count(most_common) / len(winners), 4)
