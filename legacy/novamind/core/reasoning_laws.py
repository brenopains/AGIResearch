"""
ReasoningLaws — Leis do Proprio Raciocinio

A AGI nao apenas aprende sobre o mundo.
Ela descobre LEIS sobre como ela mesma pensa.

Exemplos de leis que o sistema descobre:
  "Quando uncertainty > 0.7, combinar 3+ modelos melhora precisao em 23%"
  "Problemas espaciais se beneficiam de modelos com bias symmetry > 0.7"
  "Modelos que dormem por mais de 100 ciclos perdem precisao em 15%"
  "Coalizao de modelos com biases opostos (>0.4 distancia) gera solucoes criativas"

Auto-descoberta: o sistema monitora seu proprio comportamento,
identifica padroes, e os inscreve como leis.
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ReasoningLaw:
    """Uma lei sobre o proprio raciocinio."""
    id: str = ""
    statement: str = ""        # descricao textual da lei
    pattern: str = ""          # padrao formal que a gera
    evidence: int = 0          # quantas vezes foi observada
    confidence: float = 0.3
    domain: str = "general"
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "law": self.statement[:80],
                "evidence": self.evidence, "confidence": round(self.confidence, 2),
                "domain": self.domain}


PRIMORDIAL_LAWS = [
    ReasoningLaw(
        statement="Modelos com vieses opostos em coalizao geram solucoes criativas",
        pattern="coalition_distance > 0.3 AND coalition_success",
        evidence=0, confidence=0.5, domain="creativity"),
    ReasoningLaw(
        statement="Quando uncertainty e alta, mais modelos acordados melhoram precisao",
        pattern="uncertainty > 0.7 AND n_active_models > 3",
        evidence=0, confidence=0.5, domain="reasoning"),
    ReasoningLaw(
        statement="Modelos que dormem muito perdem precisao quando acordam",
        pattern="sleep_duration > 100 AND post_wake_accuracy",
        evidence=0, confidence=0.4, domain="ecology"),
]


class ReasoningLaws:
    """Descobre leis sobre o proprio raciocinio."""

    def __init__(self, insight_library=None, cge=None):
        self.cge = cge
        self.insight_library = insight_library
        self.laws: Dict[str, ReasoningLaw] = {}
        self._history: List[Dict] = []

        for law in PRIMORDIAL_LAWS:
            lid = f"rlaw_{hashlib.md5(law.statement.encode()).hexdigest()[:8]}"
            law.id = lid
            self.laws[lid] = law

    def observe(self, event_type: str, data: Dict):
        """Observa um evento do sistema e procura por padroes."""
        self._history.append({"type": event_type, "data": data,
                              "time": time.time()})
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def extract_laws(self) -> List[ReasoningLaw]:
        """Analisa o historico e descobre leis observaveis."""
        discovered = []
        history = self._history

        if len(history) < 5:
            return discovered

        # Lei 1: coalizao de modelos com vieses opostos gera criatividade
        coalition_events = [h for h in history
                           if h["type"] == "coalition_outcome" and h["data"].get("success")]
        for h in coalition_events:
            d = h["data"]
            if d.get("bias_distance", 0) > 0.3:
                self._strengthen_law(
                    "Modelos com vieses opostos em coalizao geram solucoes criativas",
                    domain="creativity", evidence=1)
                discovered.append(self.laws.get("creativity_law"))
                break

        # Lei 2: alta uncertainty + mais modelos = melhor precisao
        high_unc_events = [h for h in history
                          if h["type"] == "prediction"
                          and h["data"].get("uncertainty", 0) > 0.7]
        if len(high_unc_events) >= 3:
            successful = [h for h in high_unc_events
                          if h["data"].get("correct", False)]
            if successful and len(successful) > len(high_unc_events) * 0.4:
                self._strengthen_law(
                    "Quando uncertainty e alta, mais modelos acordados melhoram precisao",
                    domain="reasoning", evidence=1)
                discovered.append(self.laws.get("reasoning_law"))

        # Lei 3: modelos que dormem muito perdem precisao
        wake_events = [h for h in history
                      if h["type"] == "wake_up"]
        if len(wake_events) >= 3:
            long_sleep = [h for h in wake_events
                          if h["data"].get("sleep_duration", 0) > 100]
            low_accuracy = [h for h in long_sleep
                            if h["data"].get("post_wake_accuracy", 1) < 0.4]
            if len(low_accuracy) > 0 and len(low_accuracy) >= len(long_sleep) * 0.3:
                self._strengthen_law(
                    "Modelos que dormem muito perdem precisao quando acordam",
                    domain="ecology", evidence=1)
                discovered.append(self.laws.get("ecology_law"))

        return discovered

    def _strengthen_law(self, statement: str, domain: str = "general",
                         evidence: int = 1):
        """Reforca uma lei existente ou cria nova."""
        existing = [l for l in self.laws.values() if l.statement == statement]
        if existing:
            law = existing[0]
            law.evidence += evidence
            law.confidence = min(1.0, law.confidence + 0.05 * evidence)
            law.last_observed = time.time()
        else:
            lid = f"rlaw_{hashlib.md5(statement.encode()).hexdigest()[:8]}"
            self.laws[lid] = ReasoningLaw(
                id=lid, statement=statement, pattern=domain,
                evidence=evidence, confidence=min(0.5, evidence * 0.1),
                domain=domain)
            # Contribui para InsightLibrary tambem
            if self.insight_library is not None:
                self.insight_library.add_insight(
                    principle=f"LEI DO RACIOCINIO: {statement}",
                    confidence=0.3, tags=["reasoning_law", domain])

    def get_applicable_laws(self, context: Dict) -> List[ReasoningLaw]:
        """Retorna leis aplicaveis a um contexto."""
        results = []
        for law in self.laws.values():
            if law.confidence < 0.2:
                continue
            if law.domain == "general" or law.domain in context.get("domains", []):
                results.append(law)
        results.sort(key=lambda l: -l.confidence)
        return results[:3]

    def get_state(self) -> Dict:
        return {
            "n_laws": len(self.laws),
            "by_confidence": [{"law": l.statement[:60], "conf": round(l.confidence, 2)}
                              for l in sorted(self.laws.values(),
                                            key=lambda l: -l.confidence)[:5]],
            "n_observations": len(self._history),
        }
