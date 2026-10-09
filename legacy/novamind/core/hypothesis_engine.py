"""
HypothesisEngine — Camada entre conceitos e programas.

Antes de sintetizar um programa, o HypothesisEngine:
  1. Decompoe a observacao em conceitos
  2. Gera hipoteses concorrentes (explicacoes conceituais)
  3. Testa cada hipotese via predicoes e restricoes
  4. Seleciona a melhor hipotese
  5. Converte hipotese → guia para o ProgramSynthesizer

A diferenca fundamental:
  ANTES: "qual programa transforma isso?"
  AGORA: "que conceito explica isso? como teste essa explicacao?"

A primitive (programa) e o ULTIMO passo, nao o primeiro.
"""
from typing import List, Dict, Set, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time
import json


@dataclass
class Hypothesis:
    """Explicacao conceitual estruturada para uma observacao.

    Uma hipotese NAO e um programa. E uma explicacao:
    "esta observacao e uma instancia de CONCEITO com estas propriedades"
    """
    concept: str
    invariant_type: str
    confidence: float
    domain: str = "general"
    modality: str = "grid"

    # O que a hipotese prediz sobre a transformacao
    predictions: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    expected_primitives: List[str] = field(default_factory=list)

    # Evidencia
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    test_results: Dict[str, bool] = field(default_factory=dict)

    # Metadados
    compatible_hypotheses: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    complexity: float = 1.0       # quantos conceitos envolvidos
    falsifiability: float = 0.0   # quao testavel
    compression_gain: float = 0.0 # quanto reduz a descricao

    created_at: float = field(default_factory=time.time)
    tested_at: Optional[float] = None

    @property
    def testable_predictions(self) -> int:
        return len(self.predictions)

    @property
    def passed_tests(self) -> int:
        return sum(1 for v in self.test_results.values() if v)

    @property
    def failed_tests(self) -> int:
        return sum(1 for v in self.test_results.values() if not v)

    @property
    def is_confirmed(self) -> bool:
        return self.failed_tests == 0 and self.passed_tests > 0

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "type": self.invariant_type,
            "confidence": round(self.confidence, 3),
            "domain": self.domain,
            "expected_primitives": self.expected_primitives,
            "testable_predictions": self.testable_predictions,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "confirmed": self.is_confirmed,
        }


@dataclass
class HypothesisTest:
    """Resultado de testar uma hipotese contra dados reais."""
    hypothesis_concept: str
    passed: bool
    predictions_tested: int
    predictions_passed: int
    predictions_failed: int
    constraints_satisfied: int
    constraints_violated: int
    execution_time: float = 0.0
    error_message: str = ""


class HypothesisEngine:
    """Motor de hipoteses: observacao → conceitos → hipoteses → teste → programa.

    Uso:
        engine = HypothesisEngine(concept_registry)
        engine.register_agent(reflection_agent)
        engine.register_agent(branching_agent)

        obs = {"grid": inp, "target": out}
        hyp = engine.generate_hypotheses(obs)
        best = engine.select_best(hyp)

        guidance = engine.hypothesis_to_guidance(best)
        programs = synthesizer.synthesize(inp, out, hypothesis_guidance=guidance)
    """

    def __init__(self, concept_registry=None,
                 temperature: float = 0.5):
        from core.conceptual_discovery.concept_agent import ConceptAgentRegistry
        self.agent_registry = concept_registry or ConceptAgentRegistry()
        self.hypotheses: List[Hypothesis] = []
        self.test_history: List[HypothesisTest] = []
        self.temperature = temperature

        # Funcoes de teste por tipo de invariante
        self._test_fns: Dict[str, Callable] = {}
        self._guidance_fns: Dict[str, Callable] = {}

    def set_temperature(self, temperature: float):
        """Ajusta temperatura para geracao de hipoteses.

        Temperatura alta (0.8+): mais hipoteses, mais ousadas, lower confidence threshold
        Temperatura baixa (0.0-0.3): poucas hipoteses, conservadoras, high confidence
        """
        self.temperature = max(0.0, min(1.0, temperature))

    def register_agent(self, agent):
        """Registra um ConceptAgent no registry interno."""
        self.agent_registry.register(agent)

    def register_test_fn(self, invariant_type: str, fn: Callable):
        """Registra funcao que testa uma hipotese contra dados."""
        self._test_fns[invariant_type] = fn

    def register_guidance_fn(self, invariant_type: str, fn: Callable):
        """Registra funcao que converte hipotese em guia para o synthesizer."""
        self._guidance_fns[invariant_type] = fn

    # ---- Geracao de Hipoteses ----

    def generate_hypotheses(self, observation: Any,
                            domain: str = "general",
                            modality: str = "grid",
                            top_k: Optional[int] = None) -> List[Hypothesis]:
        """Gera hipoteses concorrentes para uma observacao.

        top_k: se None, usa valor baseado na temperatura.
               Temperatura alta → mais hipoteses.
        """
        if top_k is None:
            top_k = max(2, int(5 + self.temperature * 10))

        # Reduz threshold de confianca com temperatura alta
        min_confidence = max(0.05, 0.2 - self.temperature * 0.2)
        candidates = []

        for agent in self.agent_registry.get_all(min_confidence=min_confidence):
            hyp_data = agent.generate_hypothesis(observation, domain)
            if hyp_data is None:
                continue

            # Predicoes testaveis
            predictions = dict(agent.predictions)
            expected_prims = list(agent.compatible_with.keys())

            hyp = Hypothesis(
                concept=agent.name,
                invariant_type=agent.invariant_type,
                confidence=agent.confidence,
                domain=domain,
                modality=modality,
                predictions=predictions,
                expected_primitives=expected_prims[:5],
                constraints=[],
                complexity=1.0 / max(0.1, len(agent.predictions) + 1),
                falsifiability=len(predictions) / max(1, len(predictions) + 1),
                compression_gain=1.0 - (1.0 / max(1, agent.n_observations + 1)),
            )

            # Evidencia da observacao
            if isinstance(hyp_data, dict):
                pc = hyp_data.get("predictions_confirmed", 0)
                if pc > 0:
                    hyp.supporting_evidence.append(
                        f"predictions confirmed: {pc:.0%}"
                    )

            candidates.append(hyp)

        # Ordena por confianca
        candidates.sort(key=lambda h: (-h.confidence, h.falsifiability))

        # Marca alternativas
        for i, hyp in enumerate(candidates):
            hyp.alternative_hypotheses = [
                c.concept for c in candidates if c.concept != hyp.concept
            ]

        # Compatibilidade entre hipoteses
        for h1 in candidates:
            for h2 in candidates:
                if h1.concept != h2.concept:
                    agent = self.agent_registry.get(h1.concept)
                    if agent and h2.concept in agent.compatible_with:
                        h1.compatible_hypotheses.append(h2.concept)

        self.hypotheses.extend(candidates)
        return candidates[:top_k]

    # ---- Teste de Hipoteses ----

    def test_hypothesis(self, hyp: Hypothesis,
                        input_data: Any, output_data: Any) -> HypothesisTest:
        """Testa uma hipotese contra dados reais de entrada/saida."""
        start = time.time()
        passed = True
        predictions_tested = 0
        predictions_passed = 0
        predictions_failed = 0
        constraints_satisfied = 0
        constraints_violated = 0
        error = ""

        # Funcao de teste especifica do tipo de invariante
        if hyp.invariant_type in self._test_fns:
            try:
                result = self._test_fns[hyp.invariant_type](
                    hyp, input_data, output_data
                )
                predictions_tested = result.get("tested", 0)
                predictions_passed = result.get("passed", 0)
                predictions_failed = result.get("failed", 0)
                constraints_satisfied = result.get("constraints_ok", 0)
                constraints_violated = result.get("constraints_broken", 0)
                passed = predictions_failed == 0
            except Exception as e:
                passed = False
                error = str(e)
        else:
            # Teste generico: verifica predicoes do conceito
            for key, expected in hyp.predictions.items():
                predictions_tested += 1
                if isinstance(output_data, dict) and key in output_data:
                    if output_data[key] == expected:
                        predictions_passed += 1
                    else:
                        predictions_failed += 1
                elif isinstance(output_data, np.ndarray):
                    predictions_passed += 1
            passed = predictions_failed == 0

        elapsed = time.time() - start
        hyp.tested_at = time.time()
        hyp.test_results["passed"] = predictions_passed > 0
        hyp.test_results["failed"] = predictions_failed > 0

        test = HypothesisTest(
            hypothesis_concept=hyp.concept,
            passed=passed,
            predictions_tested=predictions_tested,
            predictions_passed=predictions_passed,
            predictions_failed=predictions_failed,
            constraints_satisfied=constraints_satisfied,
            constraints_violated=constraints_violated,
            execution_time=elapsed,
            error_message=error,
        )
        self.test_history.append(test)
        return test

    def test_all(self, hypotheses: List[Hypothesis],
                 input_data: Any, output_data: Any) -> List[HypothesisTest]:
        """Testa todas as hipoteses contra dados."""
        results = []
        for hyp in hypotheses:
            result = self.test_hypothesis(hyp, input_data, output_data)
            results.append(result)
        return results

    # ---- Selecao ----

    def select_best(self, hypotheses: List[Hypothesis],
                    results: Optional[List[HypothesisTest]] = None) -> Optional[Hypothesis]:
        """Seleciona a melhor hipotese.

        Criterios dependem da temperatura:
          Baixa temperatura: prefere hipoteses conservadoras (alta confianca, baixa complexidade)
          Alta temperatura: tolera hipoteses ousadas (menor confianca, maior complexidade)
        """
        if not hypotheses:
            return None

        results_map = {}
        if results:
            results_map = {r.hypothesis_concept: r for r in results}

        # Peso da complexidade diminui com temperatura alta
        complexity_penalty = 1.0 - self.temperature * 0.8

        def score(h: Hypothesis) -> float:
            r = results_map.get(h.concept)
            if r and r.predictions_failed > 0:
                # Temperatura alta: perdoa falhas parciais
                if self.temperature < 0.6 and r.predictions_failed > 0:
                    return -999
                if r.predictions_failed > r.predictions_passed:
                    return -999

            # Score composto: confianca + falsificabilidade - complexidade * penalidade
            s = h.confidence * 2.0
            s += h.falsifiability * 1.5
            s -= h.complexity * complexity_penalty

            if r and r.passed:
                s += 1.0

            # Temperatura alta: bonus por diversidade
            if self.temperature > 0.6:
                s += 0.2 * len(h.expected_primitives)

            return s

        best = max(hypotheses, key=score)
        return best

    # ---- Conversao para Guidance ----

    def hypothesis_to_guidance(self, hyp: Hypothesis) -> Dict:
        """Converte uma hipotese em guia para o ProgramSynthesizer.

        Returns:
            dict com:
              - primitive_weights: {primitive_type: boost}
              - expected_primitives: lista de primitivas esperadas
              - concept: nome do conceito
              - constraints: restricoes esperadas
        """
        guidance = {
            "concept": hyp.concept,
            "invariant_type": hyp.invariant_type,
            "confidence": hyp.confidence,
            "primitive_weights": {},
            "expected_primitives": hyp.expected_primitives,
            "constraints": hyp.constraints,
        }

        # Usa funcao de guidance especifica do invariante
        if hyp.invariant_type in self._guidance_fns:
            specific = self._guidance_fns[hyp.invariant_type](hyp)
            guidance["primitive_weights"] = specific.get("weights", {})
        else:
            # Generico: boost para primitivas esperadas
            for prim in hyp.expected_primitives:
                guidance["primitive_weights"][prim] = 2.0

        return guidance

    # ---- Estatisticas ----

    def get_hypothesis_summary(self) -> dict:
        return {
            "total_hypotheses": len(self.hypotheses),
            "total_tests": len(self.test_history),
            "confirmed_hypotheses": sum(1 for h in self.hypotheses if h.is_confirmed),
            "avg_confidence": round(
                sum(h.confidence for h in self.hypotheses) / max(1, len(self.hypotheses)), 3
            ) if self.hypotheses else 0,
            "recent_tests": [
                {"concept": t.hypothesis_concept, "passed": t.passed}
                for t in self.test_history[-5:]
            ],
        }
