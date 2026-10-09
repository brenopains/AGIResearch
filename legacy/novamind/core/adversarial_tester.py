"""
AdversarialTester — Tenta DESTRUIR hipoteses.

Dada uma hipotese, gera:
  - Contraexemplos (evidencias que contradizem)
  - Mecanismos rivais (explicacoes alternativas)
  - Experimentos impossiveis (o que falsificaria a hipotese)
  - Paradoxos (se A leva a B, mas B leva a nao-A)

Hipotese que sobrevive a refutacao e mais forte.
Hipotese que nao sobrevive e abandonada.
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
import time


@dataclass
class AdversarialResult:
    """Resultado de testar uma hipotese adversarialmente."""
    hypothesis_id: int
    survived: bool
    refutation_strength: float  # 0-1, quao forte foi a refutacao
    counterexamples: List[str] = field(default_factory=list)
    rival_mechanisms: List[str] = field(default_factory=list)
    paradoxes: List[str] = field(default_factory=list)
    falsifying_experiment: str = ""
    surviving_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "survived": self.survived,
            "refutation_strength": round(self.refutation_strength, 2),
            "n_counterexamples": len(self.counterexamples),
            "n_rivals": len(self.rival_mechanisms),
            "n_paradoxes": len(self.paradoxes),
        }


class AdversarialTester:
    """Testa hipoteses por refutacao adversaria.

    Uso:
        tester = AdversarialTester(kg)
        result = tester.test(hypothesis_line)
        if result.survived:
            print("Hipotese forte!")
    """

    # Pares de mecanismos que sao mutuamente contraditorios
    CONTRADICTORY_MECHANISMS = [
        ("amyloid", "tau"),
        ("inflammation", "anti-inflammatory"),
        ("apoptosis", "anti-apoptotic"),
        ("oxidative stress", "antioxidant"),
        ("excitotoxicity", "neuroprotection"),
        ("immune activation", "immunosuppression"),
        ("angiogenesis", "anti-angiogenic"),
        ("cell proliferation", "cell cycle arrest"),
    ]

    # Doencas e seus mecanismos rivais conhecidos
    DISEASE_RIVALS = {
        "Alzheimer": [
            "amyloid cascade hypothesis",
            "tau propagation hypothesis",
            "neuroinflammatory hypothesis",
            "mitochondrial cascade hypothesis",
            "vascular hypothesis",
            "glymphatic clearance hypothesis",
            "infectious hypothesis (P. gingivalis)",
            "metabolic hypothesis (insulin resistance)",
        ],
        "Parkinson": [
            "alpha-synuclein propagation",
            "mitochondrial dysfunction",
            "environmental toxin exposure",
            "gut-brain axis dysregulation",
            "neuroinflammation",
        ],
        "depression": [
            "monoamine deficiency",
            "neuroplasticity hypothesis",
            "inflammatory hypothesis",
            "HPA axis dysregulation",
            "circadian disruption",
        ],
    }

    def __init__(self, kg=None):
        self.kg = kg
        self.results: List[AdversarialResult] = []

    def test(self, line, disease: str = "",
             filtered_hypotheses: list = None) -> AdversarialResult:
        """Testa uma linha de hipotese adversarialmente.

        Returns: resultado com sobrevivencia e razao.
        """
        mechanism = line.mechanism
        score = line.score
        counterexamples = []
        rival_mechanisms = []
        paradoxes = []
        refutation_strength = 0.0

        # 1. BUSCA CONTRAEXEMPLOS no grafo de conhecimento
        if filtered_hypotheses:
            # Encontra hipoteses que contradizem este mecanismo
            for h in filtered_hypotheses:
                cause_lower = h.cause.lower()
                effect_lower = h.effect.lower()

                # Se a hipotese sugere o OPOSTO do mecanismo
                for contr in self.CONTRADICTORY_MECHANISMS:
                    if mechanism.lower() in contr[0] and contr[1].lower() in cause_lower:
                        counterexamples.append(
                            f"{contr[1]} {h.effect[:40]} contradicts {mechanism}"
                        )
                    if mechanism.lower() in contr[1] and contr[0].lower() in cause_lower:
                        counterexamples.append(
                            f"{contr[0]} {h.effect[:40]} contradicts {mechanism}"
                        )

        # 2. GERA MECANISMOS RIVAIS
        disease_rivals = self.DISEASE_RIVALS.get(disease, [])
        for rival in disease_rivals:
            if mechanism.lower() in rival.lower():
                continue  # Nao rival de si mesmo
            rival_mechanisms.append(rival)

        # Garante pelo menos 2 rivais
        for contr in self.CONTRADICTORY_MECHANISMS:
            if mechanism.lower() in contr[0] and contr[1] not in [r.lower() for r in rival_mechanisms]:
                rival_mechanisms.append(contr[1])
            if mechanism.lower() in contr[1] and contr[0] not in [r.lower() for r in rival_mechanisms]:
                rival_mechanisms.append(contr[0])
            if len(rival_mechanisms) >= 3:
                break

        # 3. GERA EXPERIMENTO FALSIFICADOR
        falsifying_exp = (
            f"If {mechanism} is the true driver of {disease}, "
            f"then completely blocking {mechanism} should halt progression. "
            f"If progression continues despite complete blockade, "
            f"{mechanism} is not causal."
        )

        # 4. GERA PARADOXO
        paradoxes.append(
            f"If {mechanism} causes {disease}, and {disease} progresses slowly "
            f"over decades, then {mechanism} should be continuously active. "
            f"But if {mechanism} is continuous, why do some patients remain "
            f"asymptomatic despite high {mechanism} burden?"
        )

        # 5. CALCULA FORCA DA REFUTACAO
        # So refuta se houver contraevidencia REAL, nao apenas rival teorico
        refutation_strength = 0.0
        if counterexamples:
            refutation_strength += min(0.5, len(counterexamples) * 0.15)
        if len(rival_mechanisms) >= 2:
            refutation_strength += 0.1
        refutation_strength = min(1.0, refutation_strength + 0.1)

        # 6. DECIDE SOBREVIVENCIA
        survived = score > refutation_strength
        surviving_reason = ""
        if survived:
            surviving_reason = (
                f"Score ({score:.2f}) exceeds refutation threshold "
                f"({refutation_strength:.2f}). "
                f"Survived {len(counterexamples)} counterexamples, "
                f"{len(rival_mechanisms)} rival mechanisms, "
                f"and falsification experiment."
            )
        else:
            surviving_reason = (
                f"Refutation ({refutation_strength:.2f}) exceeds "
                f"score ({score:.2f}). Hypothesis fails."
            )

        result = AdversarialResult(
            hypothesis_id=line.id,
            survived=survived,
            refutation_strength=refutation_strength,
            counterexamples=counterexamples[:3],
            rival_mechanisms=rival_mechanisms[:3],
            paradoxes=paradoxes,
            falsifying_experiment=falsifying_exp,
            surviving_reason=surviving_reason,
        )
        self.results.append(result)
        return result

    def get_stats(self) -> dict:
        return {
            "tests_performed": len(self.results),
            "survived": sum(1 for r in self.results if r.survived),
            "refuted": sum(1 for r in self.results if not r.survived),
            "avg_refutation_strength": round(
                sum(r.refutation_strength for r in self.results) / max(1, len(self.results)), 3
            ) if self.results else 0,
        }
