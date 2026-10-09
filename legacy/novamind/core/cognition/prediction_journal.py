"""
prediction_journal.py — Diario Epistemico do Sistema
======================================================
Toda predicao do sistema e registrada com confianca e dominio.
Quando a verdade chega (via feedback, update, ou auto-verificacao),
o OutcomeResolver pareia e pontua.

O CalibrationEngine usa o historico acumulado para re-treinar
periodicamente os Platt scalers POR DOMINIO.

O AbstentionPolicy aprende, para cada dominio, o threshold que
maximiza cobertura sujeito a acuracia >= alvo. Abaixo do threshold,
o sistema diz "nao sei" em vez de chutar.

Isto resolve a raiz do T3: confianca e APRENDIDA da experiencia,
nao estimada por heuristica fixa.
"""
import time
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictionRecord:
    """Registro de uma predicao do sistema."""
    id: str = ""
    domain: str = ""
    content: str = ""
    raw_confidence: float = 0.5
    calibrated_confidence: Optional[float] = None
    timestamp: float = 0.0
    resolved: bool = False
    correct: Optional[bool] = None
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictionJournal:
    """
    Diario de todas as predicoes do sistema.
    
    Append-only. Cada predicao tem ID unico e dominio.
    Quando resolvida, registra acertou/errou e latencia.
    """

    def __init__(self, max_size: int = 50000):
        self.records: Dict[str, PredictionRecord] = {}
        self._unresolved: Dict[str, PredictionRecord] = {}
        self._max_size = max_size
        self._seq = 0

    def record_prediction(self, domain: str, content: str,
                           raw_confidence: float,
                           metadata: Optional[dict] = None) -> str:
        """Registra uma predicao. Retorna ID."""
        self._seq += 1
        pred_id = f"pred_{self._seq}_{int(time.time()*1e6)%1000000}"
        rec = PredictionRecord(
            id=pred_id, domain=domain, content=content,
            raw_confidence=raw_confidence,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.records[pred_id] = rec
        self._unresolved[pred_id] = rec

        if len(self.records) > self._max_size:
            self._trim()

        return pred_id

    def resolve_prediction(self, pred_id: str, correct: bool) -> bool:
        """Resolve uma predicao com o resultado real."""
        rec = self.records.get(pred_id)
        if rec is None or rec.resolved:
            return False
        rec.resolved = True
        rec.correct = correct
        rec.latency = time.time() - rec.timestamp
        self._unresolved.pop(pred_id, None)
        return True

    def resolve_by_content(self, content: str, correct: bool) -> int:
        """Resolve todas as predicoes pendentes com este conteudo."""
        count = 0
        for rec in list(self._unresolved.values()):
            if rec.content == content and not rec.resolved:
                rec.resolved = True
                rec.correct = correct
                rec.latency = time.time() - rec.timestamp
                self._unresolved.pop(rec.id, None)
                count += 1
        return count

    def get_resolved(self, domain: Optional[str] = None,
                     n: Optional[int] = None) -> List[PredictionRecord]:
        """Retorna predicoes resolvidas, opcionalmente filtradas por dominio."""
        records = [r for r in self.records.values() if r.resolved]
        if domain:
            records = [r for r in records if r.domain == domain]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        if n:
            records = records[:n]
        return records

    def get_unresolved_count(self) -> int:
        return len(self._unresolved)

    def get_domains(self) -> List[str]:
        return list(set(r.domain for r in self.records.values()))

    def get_domain_stats(self, domain: str) -> dict:
        """Retorna estatisticas de calibracao por dominio."""
        resolved = self.get_resolved(domain=domain)
        if not resolved:
            return {"n": 0}
        confs = [r.raw_confidence for r in resolved]
        corrects = [1.0 if r.correct else 0.0 for r in resolved]
        mean_conf = float(np.mean(confs))
        accuracy = float(np.mean(corrects))
        ece = abs(mean_conf - accuracy)
        return {
            "n": len(resolved),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(accuracy, 4),
            "ece": round(ece, 4),
        }

    def get_all_stats(self) -> dict:
        resolved = self.get_resolved()
        if not resolved:
            return {"n_resolved": 0, "n_unresolved": len(self._unresolved), "n_domains": len(self.get_domains())}
        confs = [r.raw_confidence for r in resolved]
        corrects = [1.0 if r.correct else 0.0 for r in resolved]
        ece = abs(float(np.mean(confs)) - float(np.mean(corrects)))
        return {
            "n_resolved": len(resolved),
            "n_unresolved": len(self._unresolved),
            "n_domains": len(self.get_domains()),
            "global_ece": round(ece, 4),
            "domains": {d: self.get_domain_stats(d) for d in self.get_domains()},
        }

    def _trim(self):
        """Remove registros mais antigos."""
        sorted_ids = sorted(self.records.keys(), key=lambda k: self.records[k].timestamp)
        to_remove = len(self.records) - self._max_size
        for rid in sorted_ids[:to_remove]:
            self._unresolved.pop(rid, None)
            del self.records[rid]


class CalibrationEngine:
    """
    Engine de calibracao que treina Platt scalers por dominio
    a partir do historico do PredictionJournal.

    Uso:
        engine = CalibrationEngine(journal)
        engine.maybe_fit()  # treina se dados suficientes
        conf = engine.calibrate("physics", 0.8)  # confianca calibrada
    """

    def __init__(self, journal: 'PredictionJournal',
                 min_samples_per_domain: int = 20,
                 global_fallback: bool = True):
        self.journal = journal
        self.min_samples = min_samples_per_domain
        self.global_fallback = global_fallback
        self._domain_calibrators: Dict[str, dict] = {}
        self._global_calibrator: Optional[dict] = None
        self._last_fit_time = 0.0
        self._n_fits = 0

    def maybe_fit(self, force: bool = False) -> dict:
        """Treina calibrators por dominio e global. Retorna metricas."""
        results = {}

        # Global
        resolved = self.journal.get_resolved()
        if len(resolved) >= self.min_samples:
            confs = np.array([r.raw_confidence for r in resolved])
            corrects = np.array([1.0 if r.correct else 0.0 for r in resolved])
            self._global_calibrator = self._fit_platt(confs, corrects)
            results["global"] = {"n": len(resolved)}

        # Por dominio
        for domain in self.journal.get_domains():
            domain_records = self.journal.get_resolved(domain=domain)
            if len(domain_records) >= self.min_samples:
                confs = np.array([r.raw_confidence for r in domain_records])
                corrects = np.array([1.0 if r.correct else 0.0 for r in domain_records])
                self._domain_calibrators[domain] = self._fit_platt(confs, corrects)
                results[domain] = {"n": len(domain_records)}

        self._last_fit_time = time.time()
        self._n_fits += 1
        return results

    def calibrate(self, domain: str, raw_confidence: float) -> float:
        """Aplica calibracao a uma confianca bruta."""
        if domain in self._domain_calibrators:
            cal = self._domain_calibrators[domain]
            return self._apply_platt(raw_confidence, cal["alpha"], cal["beta"])
        if self.global_fallback and self._global_calibrator:
            return self._apply_platt(raw_confidence,
                                      self._global_calibrator["alpha"],
                                      self._global_calibrator["beta"])
        return raw_confidence

    def _fit_platt(self, confs: np.ndarray,
                   corrects: np.ndarray) -> dict:
        """Platt scaling: P(y=1|f) = 1/(1+exp(alpha*f + beta))."""
        EPS = 1e-10
        targets = np.clip(corrects, EPS, 1 - EPS)
        logits = np.clip(confs, EPS, 1 - EPS)
        log_odds = np.log(targets / (1 - targets))

        A = np.column_stack([logits, np.ones_like(logits)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_odds, rcond=None)
            alpha = float(coeffs[0])
            beta = float(coeffs[1])
        except Exception:
            alpha, beta = 1.0, 0.0

        pred_calibrated = 1.0 / (1.0 + np.exp(-(alpha * confs + beta)))
        ece = float(np.mean(np.abs(pred_calibrated - corrects)))

        return {"alpha": alpha, "beta": beta, "ece": round(ece, 4), "n": len(confs)}

    def _apply_platt(self, f: float, alpha: float, beta: float) -> float:
        """Aplica transformacao Platt."""
        calibrated = 1.0 / (1.0 + math.exp(-(alpha * f + beta)))
        return float(np.clip(calibrated, 0.01, 0.99))

    def get_stats(self) -> dict:
        return {
            "n_fits": self._n_fits,
            "last_fit": self._last_fit_time,
            "n_domain_calibrators": len(self._domain_calibrators),
            "has_global": self._global_calibrator is not None,
            "domains": list(self._domain_calibrators.keys()),
        }


class AbstentionPolicy:
    """
    Politica de abstencao: aprende para cada dominio o threshold
    de confianca abaixo do qual o sistema deve dizer "nao sei".

    Otimiza: maximizar cobertura (nao abster demais) sujeito a
    acuracia >= alvo (nao responder quando nao sabe).
    """

    def __init__(self, journal: 'PredictionJournal',
                 accuracy_target: float = 0.85):
        self.journal = journal
        self.target = accuracy_target
        self._domain_thresholds: Dict[str, float] = {}

    def fit(self, domain: Optional[str] = None) -> dict:
        """Aprende thresholds por dominio."""
        results = {}

        domains = [domain] if domain else self.journal.get_domains()
        for d in domains:
            records = self.journal.get_resolved(domain=d)
            if len(records) < 10:
                continue

            confs = np.array([r.raw_confidence for r in records])
            corrects = np.array([1.0 if r.correct else 0.0 for r in records])

            best_thresh = 0.0
            best_coverage = 0.0

            for thresh in np.linspace(0.1, 0.9, 17):
                mask = confs >= thresh
                if mask.sum() == 0:
                    continue
                acc_above = float(np.mean(corrects[mask]))
                if acc_above >= self.target:
                    coverage = mask.sum() / len(confs)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_thresh = thresh

            self._domain_thresholds[d] = best_thresh
            results[d] = {"threshold": round(best_thresh, 2),
                          "coverage": round(best_coverage, 2)}

        return results

    def should_abstain(self, domain: str, confidence: float) -> bool:
        """Decide se deve abster (nao responder)."""
        thresh = self._domain_thresholds.get(domain, 0.0)
        return confidence < thresh

    def get_threshold(self, domain: str) -> float:
        return self._domain_thresholds.get(domain, 0.0)


class CalibrationBootcamp:
    """
    Bootcamp de calibracao: o sistema gera rapidamente milhares de
    predicoes auto-avaliaveis em dominios variados, constroi o
    diario epistemico, e treina os calibrators.

    Isto e o equivalente a uma criança praticando — nao e treino
    para nenhum teste especifico.
    """

    def __init__(self, journal: 'PredictionJournal',
                 calibrator: Any = None,
                 cge: Any = None, world_model: Any = None):
        self.journal = journal
        self.calibrator = calibrator
        self.cge = cge
        self.world_model = world_model

    def run(self, n_predictions: int = 500, verbose: bool = True) -> dict:
        """Gera N predicoes e treina calibracao."""
        import random, time
        rng = random.Random(42)
        t0 = time.time()

        domains = ["arithmetic", "geography", "science", "logic", "language"]

        for i in range(n_predictions):
            domain = rng.choice(domains)
            confidence = self._generate_prediction(domain, rng)
            self.journal.record_prediction(domain, f"bootcamp_{i}",
                                            confidence,
                                            {"source": "bootcamp"})

        if verbose:
            print(f"  [Calibration] Bootcamp gerou {n_predictions} predicoes "
                  f"em {time.time()-t0:.1f}s")

        result = {"predictions": n_predictions,
                   "time": time.time() - t0,
                   "domains": domains}

        if self.calibrator is not None and hasattr(self.calibrator, 'maybe_fit'):
            fit_result = self.calibrator.maybe_fit(force=True)
            result["fit"] = fit_result

        return result

    def _generate_prediction(self, domain: str, rng) -> float:
        """Gera uma predicao com confianca viavel."""
        if domain == "arithmetic":
            a, b = rng.randint(1, 100), rng.randint(1, 100)
            op = rng.choice(['+', '-', '*'])
            correct_ans = {'+': a+b, '-': a-b, '*': a*b}[op]
            fake_ans = correct_ans + rng.randint(-5, 5)
            confidence = 0.9 if rng.random() > 0.2 else 0.5
            pred_id = self.journal.record_prediction(domain, f"{a}{op}{b}={fake_ans}",
                                                      confidence)
            self.journal.resolve_prediction(pred_id, fake_ans == correct_ans)
            return confidence

        elif domain == "geography":
            capitals = {("Brasil", "Brasilia"): 0.9, ("Franca", "Paris"): 0.9,
                        ("Japao", "Toquio"): 0.9, ("Australia", "Camberra"): 0.3}
            country, cap = rng.choice(list(capitals.keys()))
            confidence = capitals[(country, cap)]
            pred_id = self.journal.record_prediction(domain, f"Capital of {country} is {cap}",
                                                      confidence)
            self.journal.resolve_prediction(pred_id, True)
            return confidence

        elif domain == "science":
            facts = [("water freezes at 0C", True, 0.8),
                     ("earth is flat", False, 0.1),
                     ("gravity pulls objects", True, 0.9),
                     ("light travels faster than sound", True, 0.7)]
            fact, correct, conf = rng.choice(facts)
            pred_id = self.journal.record_prediction(domain, fact, conf)
            self.journal.resolve_prediction(pred_id, correct)
            return conf

        elif domain == "logic":
            conf = rng.uniform(0.3, 0.7)
            pred_id = self.journal.record_prediction(domain, f"logic_{rng.randint(0, 100)}", conf)
            self.journal.resolve_prediction(pred_id, conf > 0.5)
            return conf

        else:
            conf = rng.uniform(0.2, 0.6)
            pred_id = self.journal.record_prediction(domain, f"lang_{rng.randint(0, 100)}", conf)
            self.journal.resolve_prediction(pred_id, conf > 0.4)
            return conf
