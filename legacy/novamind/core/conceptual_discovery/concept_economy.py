"""
ConceptEconomy — Economia de conceitos: merge, split, prune.

Funciona como um processo autonomo que continuamente:
  - Procura conceitos que podem ser fundidos (sobreposicao > threshold)
  - Procura conceitos que deveriam ser divididos (explica dominios muito diferentes)
  - Poda conceitos com baixa confianca e baixo reuso
  - Mede compressao: quantos conceitos foram reduzidos

A metrica central e compression_ratio:
  conceitos_antes / conceitos_depois

Quando maior a compressao sem perder poder explicativo,
melhor o modelo do mundo.
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


@dataclass
class EconomyAction:
    """Uma acao de economia sobre conceitos."""
    action_type: str  # "merge", "split", "prune", "keep"
    concept_a: str
    concept_b: str = ""
    reason: str = ""
    compression_gain: float = 0.0
    confidence_delta: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action": self.action_type,
            "a": self.concept_a,
            "b": self.concept_b,
            "reason": self.reason,
            "compression": round(self.compression_gain, 3),
        }


class ConceptEconomy:
    """Economia de conceitos.

    Uso:
        economy = ConceptEconomy()
        economy.observe_concept("reflection", {"geometry", "physics", "audio"})
        economy.observe_concept("symmetry", {"geometry", "biology"})

        actions = economy.run()  # merge/split/prune
        print(f"Compression: {economy.compression_ratio}")
    """

    def __init__(self, merge_threshold: float = 0.7,
                 prune_threshold: float = 0.1,
                 split_threshold: float = 0.4):
        self._concept_domains: Dict[str, Set[str]] = {}
        self._concept_confidence: Dict[str, float] = {}
        self._concept_observations: Dict[str, int] = {}
        self._concept_reuse: Dict[str, int] = {}  # quantas vezes usado

        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.split_threshold = split_threshold

        self.actions: List[EconomyAction] = []
        self.total_original: int = 0
        self.total_current: int = 0

    def observe_concept(self, name: str, domains: Set[str],
                        confidence: float = 0.3, reuse: int = 1):
        """Registra observacao de um conceito."""
        if name in self._concept_domains:
            self._concept_domains[name] |= domains
            self._concept_confidence[name] = max(
                self._concept_confidence.get(name, 0), confidence
            )
            self._concept_observations[name] = self._concept_observations.get(name, 0) + 1
            self._concept_reuse[name] = self._concept_reuse.get(name, 0) + reuse
        else:
            self._concept_domains[name] = set(domains)
            self._concept_confidence[name] = confidence
            self._concept_observations[name] = 1
            self._concept_reuse[name] = reuse

    @property
    def compression_ratio(self) -> float:
        """Conceitos antes / conceitos depois das acoes."""
        if self.total_current == 0:
            return 1.0
        return self.total_original / max(1, self.total_current)

    def run(self) -> List[EconomyAction]:
        """Executa um ciclo de economia.

        1. Merge: funde conceitos com alta sobreposicao de dominios
        2. Split: divide conceitos que explicam dominios muito diferentes
        3. Prune: remove conceitos com baixa confianca e baixo reuso

        Returns: lista de acoes executadas.
        """
        self.total_original = len(self._concept_domains)
        actions = []

        # 1. MERGE: busca pares com alta sobreposicao
        merge_actions = self._find_merges()
        actions.extend(merge_actions)

        # 2. SPLIT: busca conceitos que deveriam ser divididos
        split_actions = self._find_splits()
        actions.extend(split_actions)

        # 3. PRUNE: remove conceitos inviaveis
        prune_actions = self._find_prunes()
        actions.extend(prune_actions)

        self.total_current = len(self._concept_domains)
        self.actions.extend(actions)
        return actions

    def _find_merges(self) -> List[EconomyAction]:
        """Encontra pares de conceitos que podem ser fundidos."""
        actions = []
        names = list(self._concept_domains.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                domains_a = self._concept_domains[a]
                domains_b = self._concept_domains[b]

                if not domains_a or not domains_b:
                    continue

                # Jaccard similarity entre dominios
                intersection = domains_a & domains_b
                union = domains_a | domains_b
                overlap = len(intersection) / max(1, len(union))

                if overlap >= self.merge_threshold:
                    # Merge: B é absorvido por A
                    self._concept_domains[a] |= domains_b
                    self._concept_confidence[a] = max(
                        self._concept_confidence.get(a, 0),
                        self._concept_confidence.get(b, 0),
                    )
                    self._concept_reuse[a] = (self._concept_reuse.get(a, 0) +
                                               self._concept_reuse.get(b, 0))

                    # Remove B
                    del self._concept_domains[b]
                    if b in self._concept_confidence:
                        del self._concept_confidence[b]
                    if b in self._concept_observations:
                        del self._concept_observations[b]
                    if b in self._concept_reuse:
                        del self._concept_reuse[b]

                    compression = 1.0 - (1.0 / max(1, len(union) - len(intersection) + 1))
                    actions.append(EconomyAction(
                        action_type="merge",
                        concept_a=a,
                        concept_b=b,
                        reason=f"domain overlap={overlap:.2f}",
                        compression_gain=compression,
                        confidence_delta=overlap * 0.1,
                    ))

                    # Recomeca scan (lista mudou)
                    return self._find_merges()

        return actions

    def _find_splits(self) -> List[EconomyAction]:
        """Encontra conceitos que deveriam ser divididos.

        Um conceito merece ser dividido se explica dominios
        que compartilham menos que split_threshold de similaridade.
        """
        actions = []
        for name, domains in list(self._concept_domains.items()):
            domain_list = list(domains)
            if len(domain_list) < 3:
                continue

            # Se os dominios sao muito diferentes → dividir
            # Mede dissimilaridade media entre pares de dominios
            total_sim = 0
            pairs = 0
            for i in range(len(domain_list)):
                for j in range(i + 1, len(domain_list)):
                    d1, d2 = domain_list[i], domain_list[j]
                    # Simulacao simples: dominios com mesmo prefixo sao similares
                    sim = 0.5 if d1[:3] == d2[:3] else 0.1
                    total_sim += sim
                    pairs += 1

            avg_sim = total_sim / max(1, pairs)
            if avg_sim < self.split_threshold:
                action = EconomyAction(
                    action_type="split",
                    concept_a=name,
                    reason=f"low domain similarity={avg_sim:.2f} for {len(domains)} domains",
                    compression_gain=-0.2,  # dividir custa compressao
                )
                actions.append(action)

        return actions

    def _find_prunes(self) -> List[EconomyAction]:
        """Encontra conceitos que podem ser podados.

        Criterios:
          - Confianca baixa (< prune_threshold)
          - Baixo reuso (< 3)
          - Ou nunca foi observado novamente (n_observacoes == 1 ha muito tempo)
        """
        actions = []
        for name in list(self._concept_domains.keys()):
            confidence = self._concept_confidence.get(name, 0)
            reuse = self._concept_reuse.get(name, 0)
            observations = self._concept_observations.get(name, 0)

            if confidence < self.prune_threshold and reuse < 3 and observations <= 1:
                del self._concept_domains[name]
                if name in self._concept_confidence:
                    del self._concept_confidence[name]
                if name in self._concept_observations:
                    del self._concept_observations[name]
                if name in self._concept_reuse:
                    del self._concept_reuse[name]

                actions.append(EconomyAction(
                    action_type="prune",
                    concept_a=name,
                    reason=f"low confidence={confidence:.2f}, reuse={reuse}",
                    compression_gain=0.1,  # podar aumenta compressao
                ))

        return actions

    def get_stats(self) -> dict:
        return {
            "total_original": self.total_original,
            "total_current": self.total_current,
            "compression_ratio": round(self.compression_ratio, 3),
            "total_actions": len(self.actions),
            "merges": sum(1 for a in self.actions if a.action_type == "merge"),
            "splits": sum(1 for a in self.actions if a.action_type == "split"),
            "prunes": sum(1 for a in self.actions if a.action_type == "prune"),
            "recent_actions": [a.to_dict() for a in self.actions[-10:]],
        }
