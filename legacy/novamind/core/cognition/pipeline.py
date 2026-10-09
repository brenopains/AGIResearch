"""Pipeline cognitivo completo: INPUT -> Evidence Builders -> Competition -> Calibration -> Consistency -> Translator -> Timeline.

Nenhum modulo conhece a implementacao interna de outro.
Todos se comunicam apenas por contratos.
"""
from typing import List, Optional, Dict, Callable
from .contracts import Hypothesis, EvidenceItem, CompetitionResult
from .competition import Competition
from .confidence_calibrator import ConfidenceCalibrator
from .consistency_checker import ConsistencyChecker
from .translator import Translator
from .timeline import Timeline


class CognitionPipeline:
    """Pipeline integrado do modulo de cognicao.

    Uso:
        pipeline = CognitionPipeline(competition=Competition(iterations=3))
        result = pipeline.run("whale is mammal")
        print(pipeline.translate(result))
    """

    def __init__(
        self,
        builders: Optional[List[Callable]] = None,
        competition: Optional[Competition] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        consistency: Optional[ConsistencyChecker] = None,
        translator: Optional[Translator] = None,
        timeline: Optional[Timeline] = None,
    ):
        self.builders = builders or []
        self.competition = competition or Competition(iterations=3)
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.consistency = consistency or ConsistencyChecker()
        self.translator = translator or Translator(include_audit=True)
        self.timeline = timeline or Timeline()

    def add_builder(self, builder_fn: Callable):
        self.builders.append(builder_fn)

    def run(
        self,
        focus: str,
        top_k_hypotheses: int = 5,
        attention_timeline: Optional[List[Dict]] = None,
        recall_quality: Optional[Dict] = None,
        prediction_error_map: Optional[Dict[str, float]] = None,
        reasoning_style: Optional[Dict[str, float]] = None,
    ) -> CompetitionResult:
        """Executa o pipeline completo para um foco.

        Args:
            focus: conceito/pergunta central
            top_k_hypotheses: maximo de hipoteses
            attention_timeline: conceitos ativos neste ciclo
            recall_quality: metricas de qualidade do recall
            prediction_error_map: erro de predicao por modalidade
            reasoning_style: classificacao do estilo de raciocinio

        Returns:
            CompetitionResult com todas as metricas e rastreabilidade
        """
        all_evidence: List[EvidenceItem] = []

        for builder_fn in self.builders:
            try:
                items = builder_fn(focus)
                all_evidence.extend(items)
            except Exception:
                pass

        hypotheses = self._build_hypotheses(all_evidence, focus, top_k_hypotheses)

        result = self.competition.run(hypotheses)

        # Calibra confianca
        for h in result.hypotheses:
            h.confidence = self.calibrator.calibrate(h.confidence)

        # Atualiza vencedor com confianca calibrada
        if result.hypotheses:
            result.winner_confidence = result.hypotheses[0].confidence
            if len(result.hypotheses) > 1:
                result.runner_up_confidence = result.hypotheses[1].confidence
            result.margin = result.hypotheses[0].confidence - (result.hypotheses[1].confidence if len(result.hypotheses) > 1 else 0)

        # Verifica consistencia
        conflicts = self.consistency.check_hypotheses(result.hypotheses)
        evidence_conflicts = self.consistency.check_evidence_consistency(all_evidence)

        # Prediction error = 1 - media das confiancas
        prediction_error = 1.0 - (result.winner_confidence if result.hypotheses else 0)

        # Registra na timeline com todos os metadados
        self.timeline.record(
            focus=focus,
            result=result,
            prediction_error=prediction_error,
            attention_timeline=attention_timeline,
            recall_quality=recall_quality,
            prediction_error_map=prediction_error_map,
            reasoning_style=reasoning_style,
        )

        return result

    def translate(self, result: CompetitionResult, question: str = "") -> str:
        return self.translator.translate(result, question)

    def translate_short(self, result: CompetitionResult) -> str:
        return self.translator.translate_short(result)

    def get_timeline(self, n_last: int = 20) -> List[Dict]:
        return self.timeline.get_timeline(n_last)

    def get_audit_summary(self, result: CompetitionResult) -> str:
        parts = [
            f"Vencedora: {result.winner} (conf={result.winner_confidence:.3f})",
            f"Runner-up: {result.runner_up} (conf={result.runner_up_confidence:.3f})" if result.runner_up else "",
            f"Margem: {result.margin:.3f}",
            f"Entropia: {result.entropy:.3f}",
            f"Diversidade: {result.diversity_score:.3f}",
            f"Clusters: {result.num_clusters}",
            f"Reversoes: {result.winner_reversals}",
            f"Velocidade Conv.: {result.convergence_velocity:.3f}",
            f"Hipoteses: {len(result.hypotheses)}",
        ]
        return "\n".join(p for p in parts if p)

    def get_deep_audit(self, result: CompetitionResult, n_last: int = 5) -> str:
        """Auditoria profunda com breakdown e convergencia."""
        lines = [self.get_audit_summary(result)]

        # Contribution breakdown da vencedora
        if result.hypotheses:
            winner = result.hypotheses[0]
            lines.append("\nBreakdown Vencedora:")
            for cb in winner.contribution_breakdown:
                lines.append(f"  {cb.source.value:15s}: score={cb.score:.3f}  weight={cb.weight:.3f}  ({cb.n_items} itens)")

            # Runner-up breakdown
            if len(result.hypotheses) > 1:
                runner = result.hypotheses[1]
                lines.append(f"\nBreakdown Runner-up ({runner.name}):")
                for cb in runner.contribution_breakdown:
                    lines.append(f"  {cb.source.value:15s}: score={cb.score:.3f}  weight={cb.weight:.3f}  ({cb.n_items} itens)")

        # Convergence steps
        if result.convergence_steps:
            lines.append("\nConvergencia:")
            for cs in result.convergence_steps:
                scores_str = ", ".join(f"{k}={v:.3f}" for k, v in cs.scores.items())
                lines.append(f"  Iter {cs.iteration}: {scores_str} | winner={cs.winner} margin={cs.margin:.4f}")

        return "\n".join(lines)

    def _build_hypotheses(self, evidence: List[EvidenceItem], focus: str, top_k: int) -> List[Hypothesis]:
        if not evidence:
            return []

        groups: Dict[str, List[EvidenceItem]] = {}
        for item in evidence:
            entity = item.metadata.get("to", "") or item.metadata.get("node_id", "") or focus
            if entity not in groups:
                groups[entity] = []
            groups[entity].append(item)

        hypotheses: List[Hypothesis] = []
        for entity, ev_list in groups.items():
            evidence_for = [e for e in ev_list if e.confidence >= 0.1]
            evidence_against = [e for e in ev_list if e.confidence < 0.1]

            total_conf = sum(e.confidence for e in evidence_for)
            categories = set(e.relation.value for e in evidence_for)
            main_cat = next(iter(categories)) if categories else "related"

            h = Hypothesis(
                name=entity,
                category=main_cat,
                description=f"relacao entre {focus} e {entity}",
                evidence_for=evidence_for,
                evidence_against=evidence_against,
                confidence=total_conf / max(1, len(evidence_for)) if evidence_for else 0.0,
                evidence_total=total_conf,
            )
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.evidence_total, reverse=True)
        return hypotheses[:top_k]
