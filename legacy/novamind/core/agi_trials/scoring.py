"""
scoring.py — Sistema de pontuacao dos AGI Trials
===================================================
Cada trial produz um TrialResult (0-1). O AGI Score e a media
geometrica de todos os trials — se qualquer trial for zero, o
score total e zero. Isto implementa o principio de que AGI
requer TODAS as capacidades fundamentais, nao apenas algumas.
"""
import math
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TrialResult:
    """Resultado de um unico trial AGI."""
    name: str = ""
    trial_id: int = 0
    seed: int = 0
    score: float = 0.0          # 0.0 a 1.0
    threshold: float = 0.3       # minimo para passar
    passed: bool = False
    details: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    n_trials: int = 1            # quantas repeticoes dentro do trial
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "details": self.details,
            "metrics": {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                        for k, v in self.metrics.items()},
            "error": self.error,
            "duration": round(self.duration, 2),
        }


def compute_agi_score(results: List[TrialResult],
                      chance_level: float = 0.2) -> Dict[str, Any]:
    """
    Calcula o AGI Score composto.
    
    Regras:
      1. Score = media geometrica de todos os trials
      2. Se qualquer trial <= chance_level, score = 0 (gate rigido)
      3. Passa se score > chance_level E todos acima do threshold
    
    Retorna dict com score, pass, breakdown.
    """
    if not results:
        return {"score": 0.0, "passed": False, "breakdown": [], "gate": "empty"}

    scores = [r.score for r in results]
    passed_check = [r.passed for r in results]

    geo_mean = math.exp(sum(math.log(max(s, 1e-10)) for s in scores) / len(scores))
    all_above_chance = all(s > chance_level for s in scores)

    overall_pass = all_above_chance and all(passed_check)

    return {
        "score": round(geo_mean, 4),
        "passed": overall_pass,
        "passed_count": sum(passed_check),
        "total_count": len(results),
        "arithmetic_mean": round(sum(scores) / len(scores), 4),
        "gate_all_above_chance": all_above_chance,
        "chance_level": chance_level,
        "breakdown": [r.to_dict() for r in results],
    }


@dataclass
class TrialSuiteResult:
    """Resultado consolidado de todos os trials."""
    results: List[TrialResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    overall: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        self.end_time = time.time()
        self.overall = compute_agi_score(self.results)
        self.overall["total_duration"] = round(self.end_time - self.start_time, 2)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("AGI TRIALS — RESULTS")
        lines.append("=" * 60)
        lines.append("")

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            score_str = f"{r.score:.2%}"
            lines.append(f"  [{status}] Trial {r.trial_id}: {r.name}")
            lines.append(f"         Score: {score_str}  (threshold: {r.threshold:.0%})")
            lines.append(f"         Seed: {r.seed}")
            if r.details:
                lines.append(f"         {r.details}")
            if r.metrics:
                for k, v in r.metrics.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"         {k}: {v:.4f}")
                    else:
                        lines.append(f"         {k}: {v}")
            lines.append("")

        o = self.overall
        lines.append("-" * 60)
        lines.append(f"  AGI SCORE (geometric mean): {o['score']:.2%}")
        lines.append(f"  Arithmetic mean: {o.get('arithmetic_mean', 0):.2%}")
        lines.append(f"  Passed: {o['passed_count']}/{o['total_count']}")
        lines.append(f"  Gate (all > chance={o.get('chance_level', 0.2):.0%}): {o.get('gate_all_above_chance', False)}")
        verdict = "GRANTED" if o['passed'] else "DENIED"
        lines.append(f"")
        lines.append(f"  >>> AGI STATUS: {verdict} <<<")
        if not o['passed']:
            for r in self.results:
                if not r.passed:
                    lines.append(f"    -> Failing: Trial {r.trial_id} ({r.name}) score={r.score:.2%}")
        lines.append("")
        lines.append(f"  Total duration: {o.get('total_duration', 0):.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "overall": {k: v for k, v in self.overall.items()
                       if k in ("score", "passed", "passed_count", "total_count",
                                "arithmetic_mean", "gate_all_above_chance", "total_duration")},
            "duration": round(self.end_time - self.start_time, 2),
        }
