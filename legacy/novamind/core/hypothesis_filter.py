"""
HypothesisFilter — Elimina ruído de 200k hipóteses brutas.

Filtros:
  1. REDUNDANCIA: mesma causa-efeito já vista
  2. TRIVIALIDADE: A→A, ou cadeias curtas demais sem valor
  3. PLAUSIBILIDADE: causa ou efeito não são entidades biológicas reais
  4. UTILIDADE: a descoberta sugere um ponto de intervenção (alvo terapêutico)?
"""
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class FilteredHypothesis:
    """Hipotese apos filtragem, com score de qualidade."""
    cause: str
    effect: str
    inference_type: str
    raw_confidence: float
    quality_score: float  # 0-1, composto de todos os filtros
    has_therapeutic_target: bool
    is_novel: bool
    is_biologically_plausible: bool
    evidence_chain: List[str] = field(default_factory=list)
    suggested_intervention: str = ""

    def to_summary(self) -> str:
        return (f"{'🧬' if self.has_therapeutic_target else '  '} "
                f"[Q={self.quality_score:.2f}] "
                f"{self.cause[:50]:50s} → {self.effect[:50]}")

    def to_dict(self) -> dict:
        return {
            "cause": self.cause[:60],
            "effect": self.effect[:60],
            "type": self.inference_type,
            "quality": round(self.quality_score, 3),
            "has_target": self.has_therapeutic_target,
            "intervention": self.suggested_intervention[:60],
            "evidence": self.evidence_chain[:3],
        }


class HypothesisFilter:
    """Filtra hipoteses brutas em conhecimento acionavel.

    Uso:
        filter = HypothesisFilter()
        filtered = filter.process(raw_hypotheses)
        top = filter.get_top_k(20)
    """

    # Entidades biologicas conhecidas (genes, proteinas, drogas, doencas)
    BIOLOGICAL_ENTITIES = {
        "amyloid", "tau", "amyloid-beta", "Aβ", "p-tau", "alpha-synuclein",
        "dopamine", "serotonin", "acetylcholine", "glutamate", "GABA",
        "BDNF", "NGF", "GDNF", "CREB", "mTOR", "AMPK", "SIRT1",
        "TNF-alpha", "IL-6", "IL-10", "IL-1-beta", "IFN-gamma",
        "EGFR", "KRAS", "BRAF", "TP53", "PTEN", "MYC", "HER2",
        "insulin", "leptin", "ghrelin", "cortisol", "estrogen",
        "apoptosis", "autophagy", "neurogenesis", "synaptic plasticity",
        "neuroinflammation", "oxidative stress", "mitochondrial",
        "Alzheimer", "Parkinson", "Huntington", "cancer", "diabetes",
        "microglia", "astrocytes", "neurons", "oligodendrocytes",
        "CRISPR", "Cas9", "gene therapy", "immunotherapy",
        "dopamine receptor", "NMDA receptor", "GABA receptor",
        "SSRI", "SNRI", "MAO inhibitor", "antipsychotic",
        "lithium", "valproate", "lamotrigine", "ketamine",
        "exercise", "diet", "sleep", "meditation",
    }

    # Stopwords para limpeza de nomes
    STOP = {"that", "this", "with", "from", "have", "been", "were",
            "their", "they", "them", "than", "into", "also", "very",
            "just", "about", "which", "when", "where", "what", "after",
            "before", "could", "would", "should", "these", "those"}

    def __init__(self):
        self.filtered: List[FilteredHypothesis] = []
        self._seen_pairs: Set[Tuple[str, str]] = set()

    def _extract_entities(self, text: str) -> Set[str]:
        """Extrai entidades biologicas conhecidas de um texto."""
        found = set()
        text_lower = text.lower()
        for entity in self.BIOLOGICAL_ENTITIES:
            if entity.lower() in text_lower:
                found.add(entity)
        return found

    def _is_junk(self, text: str) -> bool:
        """Detecta se um texto e lixo (fragmento sem sentido)."""
        text_lower = text.lower()
        junk_indicators = [
            text_lower.startswith("the "),
            text_lower.startswith("a "),
            text_lower.startswith("of "),
            text_lower.startswith("and "),
            text_lower.startswith("to "),
            text_lower.startswith("in "),
            text_lower.startswith("that "),
            len(text.split()) <= 1,
            len(text) < 10,
        ]
        return sum(1 for j in junk_indicators if j) >= 2

    def _has_therapeutic_relevance(self, cause: str, effect: str) -> Tuple[bool, str]:
        """Uma hipotese sugere um ponto de intervencao terapeutica?"""
        cause_entities = self._extract_entities(cause)
        effect_entities = self._extract_entities(effect)
        all_entities = cause_entities | effect_entities

        # Alvos terapeuticos conhecidos
        targets = self.BIOLOGICAL_ENTITIES
        intervention = ""

        # Se causa ou efeito e um alvo conhecido → relevante
        for entity in all_entities:
            if any(entity in t for t in ["receptor", "inhibitor", "agonist",
                                          "antagonist", "therapy", "drug",
                                          "antibody", "vaccine", "gene",
                                          "CRISPR", "BDNF", "NGF", "insulin"]):
                intervention = f"modulate {entity}"
                return True, intervention

        # Se a relacao conecta algo a uma doenca → relevante
        diseases = {"Alzheimer", "Parkinson", "cancer", "diabetes", "depression",
                    "schizophrenia", "stroke", "Huntington", "ALS", "autism"}
        if all_entities & diseases:
            return True, "disease-modifying pathway"

        return False, ""

    def process(self, raw_hypotheses: list,
                min_quality: float = 0.1) -> List[FilteredHypothesis]:
        """Processa hipoteses brutas: filtra, pontua, ordena."""
        self.filtered = []

        for h in raw_hypotheses:
            cause = h.cause
            effect = h.effect
            itype = h.inference_type
            confidence = h.confidence

            # 1. FILTRO: par ja visto?
            pair = (cause, effect)
            if pair in self._seen_pairs:
                continue
            self._seen_pairs.add(pair)

            # 2. FILTRO: junk detection
            if self._is_junk(cause) or self._is_junk(effect):
                continue

            # 3. FILTRO: trivialidade (causa == efeito)
            if cause == effect:
                continue

            # 4. FILTRO: cadeia muito longa mas sem entidade biologica
            cause_ent = self._extract_entities(cause)
            effect_ent = self._extract_entities(effect)
            total_ent = cause_ent | effect_ent
            if not total_ent and confidence < 0.3:
                continue

            # 5. SCORE DE QUALIDADE
            quality = confidence

            # Bonus por entidades biologicas
            bio_bonus = min(0.3, len(total_ent) * 0.1)
            quality += bio_bonus

            # Bonus por ser inferencia de transferencia (cross-dominio)
            if itype == "transfer":
                quality += 0.1

            # Bonus por ter alvo terapeutico
            has_target, intervention = self._has_therapeutic_relevance(cause, effect)

            # Verifica plausibilidade biologica
            plausible = len(total_ent) >= 1

            quality = min(1.0, quality)

            if quality >= min_quality:
                fh = FilteredHypothesis(
                    cause=cause[:60],
                    effect=effect[:60],
                    inference_type=itype,
                    raw_confidence=confidence,
                    quality_score=quality,
                    has_therapeutic_target=has_target,
                    is_novel=h.is_novel,
                    is_biologically_plausible=plausible,
                    evidence_chain=h.evidence_chain,
                    suggested_intervention=intervention,
                )
                self.filtered.append(fh)

        # Ordena por qualidade
        self.filtered.sort(key=lambda f: (-f.quality_score, f.has_therapeutic_target))
        return self.filtered

    def get_top_k(self, k: int = 20) -> List[FilteredHypothesis]:
        return self.filtered[:k]

    def get_therapeutic_candidates(self) -> List[FilteredHypothesis]:
        """Hipoteses com potencial terapeutico real."""
        return [h for h in self.filtered if h.has_therapeutic_target]

    def get_stats(self) -> dict:
        return {
            "total_input": self._seen_pairs_total if hasattr(self, '_seen_pairs_total') else len(self._seen_pairs),
            "total_filtered": len(self.filtered),
            "therapeutic_candidates": len(self.get_therapeutic_candidates()),
            "biologically_plausible": sum(1 for h in self.filtered if h.is_biologically_plausible),
            "avg_quality": round(
                sum(h.quality_score for h in self.filtered) / max(1, len(self.filtered)), 3
            ) if self.filtered else 0,
        }
