"""
AdversarialRealityGenerator — Gera pressao cognitiva.

Produz continuamente:
  - Contraexemplos: observacoes que violam leis existentes
  - Paradoxos: A implica B, B implica nao-A
  - Dominios misturados: conceito de um dominio aplicado em outro
  - Ruido: observacoes com variaveis ocultas ou distorcidas
  - Causalidade alterada: causa e efeito invertidos

Isso forc,a o sistema a:
  - Refinar leis (contraexemplos mostram limites)
  - Descobrir novas leis (paradoxos exigem explicacao)
  - Testar transferencia (dominios misturados testam abstracao)
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import random
import time


@dataclass
class AdversarialExample:
    """Um exemplo adversarial gerado para pressionar o sistema."""
    type: str  # "counterexample", "paradox", "domain_mix", "noise", "causal_swap"
    domain: str
    target_concept: str = ""
    target_law: str = ""

    # O exemplo em si
    observation: dict = field(default_factory=dict)
    expected_prediction: dict = field(default_factory=dict)

    # Metadados
    challenge: str = ""    # descricao do desafio
    severity: float = 0.5  # 0-1, quao dificil
    resolved: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "domain": self.domain,
            "concept": self.target_concept,
            "law": self.target_law,
            "challenge": self.challenge[:80],
            "severity": round(self.severity, 2),
            "resolved": self.resolved,
        }


class AdversarialRealityGenerator:
    """Gera pressao cognitiva sobre o sistema.

    Uso:
        gen = AdversarialRealityGenerator(law_registry, concept_registry)
        examples = gen.generate_pressure_batch(n=10)
        for ex in examples:
            # Apresenta ao sistema e mede reacao
            surprise = world_model.update(ex.observation, ex.expected_prediction)
            if surprise > 0.6:
                conflict = conflict_resolver.detect_conflict(...)
    """

    def __init__(self, law_registry=None, concept_registry=None,
                 seed: int = 42):
        self.law_registry = law_registry
        self.concept_registry = concept_registry
        self.rng = random.Random(seed)
        self.generated: List[AdversarialExample] = []
        self.resolved_count: int = 0

        # Dominios conhecidos
        self._domains = {
            "geometry", "physics", "biology", "audio", "programming",
            "geography", "economics", "linguistics", "psychology",
        }

    def generate_pressure_batch(self, n: int = 10) -> List[AdversarialExample]:
        """Gera um lote de exemplos adversariais."""
        examples = []
        generators = [
            self._gen_counterexample,
            self._gen_paradox,
            self._gen_domain_mix,
            self._gen_noise,
            self._gen_causal_swap,
        ]

        for _ in range(n):
            gen = self.rng.choice(generators)
            ex = gen()
            if ex:
                examples.append(ex)
                self.generated.append(ex)

        return examples

    def _gen_counterexample(self) -> Optional[AdversarialExample]:
        """Gera um contraexemplo: observacao que viola uma lei existente.

        Se a lei 'conservation' diz que quantidade se preserva,
        o contraexemplo mostra uma transformacao onde algo desaparece.
        """
        if not self.law_registry:
            return None

        laws = self.law_registry.get_all(min_confidence=0.1)
        if not laws:
            return None

        law = self.rng.choice(laws)
        domain = self.rng.choice(list(law.domains)) if law.domains else "general"

        ex = AdversarialExample(
            type="counterexample",
            domain=domain,
            target_law=law.name,
            observation={
                "domain": domain,
                "violates": law.name,
                "expected": "law_prediction",
                "actual": "contradiction",
            },
            expected_prediction={"contradiction": True, "law_violated": law.name},
            challenge=f"'{law.name}' predicts one thing, but observation shows opposite in {domain}",
            severity=0.7,
        )
        return ex

    def _gen_paradox(self) -> Optional[AdversarialExample]:
        """Gera um paradoxo: A implica B, B implica nao-A.

        Exemplo: "este objeto e simetrico e assimetrico ao mesmo tempo"
        """
        concepts = list(self._get_concept_names())
        if len(concepts) < 2:
            return None

        a, b = self.rng.sample(concepts, 2)
        domain = self.rng.choice(list(self._domains))

        ex = AdversarialExample(
            type="paradox",
            domain=domain,
            target_concept=a,
            observation={
                "domain": domain,
                "concept_a": a,
                "concept_b": b,
                "relation": "contradiction",
                "description": f"object is both {a} and {b} simultaneously",
            },
            expected_prediction={"paradox": True, "requires_new_law": True},
            challenge=f"'{a}' and '{b}' both apply to the same observation — they should contradict",
            severity=0.9,
        )
        return ex

    def _gen_domain_mix(self) -> Optional[AdversarialExample]:
        """Gera um exemplo de dominio misturado.

        Pega um conceito que so foi observado em dominios X e Y,
        e apresenta uma observacao que parece ser desse conceito
        num dominio Z completamente diferente.
        """
        if not self.concept_registry:
            return None

        agents = self.concept_registry.get_all(min_confidence=0.1)
        if not agents:
            return None

        agent = self.rng.choice(agents)
        known_domains = agent.domains
        if not known_domains:
            return None

        # Escolhe um dominio NOVO (nao conhecido para este conceito)
        possible_new = self._domains - known_domains
        if not possible_new:
            return None

        new_domain = self.rng.choice(list(possible_new))

        ex = AdversarialExample(
            type="domain_mix",
            domain=new_domain,
            target_concept=agent.name,
            observation={
                "domain": new_domain,
                "concept": agent.name,
                "known_domains": list(known_domains),
                "new_domain": new_domain,
                "description": f"pattern resembling '{agent.name}' appears in {new_domain}",
            },
            expected_prediction={"transfer_possible": True, "new_domain": new_domain},
            challenge=f"'{agent.name}' known in {known_domains}, now appears in {new_domain}",
            severity=0.6,
        )
        return ex

    def _gen_noise(self) -> Optional[AdversarialExample]:
        """Gera um exemplo com ruido ou variavel oculta.

        Apresenta uma observacao onde uma variavel causal crucial
        esta oculta ou distorcida.
        """
        domain = self.rng.choice(list(self._domains))
        hidden_var = self.rng.choice(["causal_factor", "context", "modality", "scale"])

        ex = AdversarialExample(
            type="noise",
            domain=domain,
            observation={
                "domain": domain,
                "hidden_variable": hidden_var,
                "surface_pattern": "misleading",
                "noise_level": 0.7,
                "description": f"observation in {domain} with hidden '{hidden_var}'",
            },
            expected_prediction={
                "incomplete": True,
                "hidden_variable": hidden_var,
                "needs_inference": True,
            },
            challenge=f"key variable '{hidden_var}' is hidden — model must infer it",
            severity=0.8,
        )
        return ex

    def _gen_causal_swap(self) -> Optional[AdversarialExample]:
        """Gera um exemplo com causalidade invertida.

        Apresenta efeito antes da causa, ou causa e efeito trocados.
        """
        domain = self.rng.choice(list(self._domains))

        ex = AdversarialExample(
            type="causal_swap",
            domain=domain,
            observation={
                "domain": domain,
                "temporal_order": "reversed",
                "effect": "observed_first",
                "cause": "inferred",
                "description": f"effect observed before cause in {domain}",
            },
            expected_prediction={"causal_reversal": True, "needs_temporal_model": True},
            challenge=f"cause and effect are swapped in {domain}",
            severity=0.85,
        )
        return ex

    def _get_concept_names(self) -> Set[str]:
        if self.concept_registry:
            return {a.name for a in self.concept_registry.get_all()}
        return set()

    def mark_resolved(self, example: AdversarialExample):
        """Marca um exemplo adversarial como resolvido."""
        example.resolved = True
        self.resolved_count += 1

    def get_stats(self) -> dict:
        return {
            "total_generated": len(self.generated),
            "total_resolved": self.resolved_count,
            "by_type": dict(
                (t, sum(1 for e in self.generated if e.type == t))
                for t in set(e.type for e in self.generated)
            ),
            "unresolved": len([e for e in self.generated if not e.resolved]),
        }
