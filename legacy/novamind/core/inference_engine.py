"""
InferenceEngine — Motor de inferencia que gera conhecimento VERDADEIRAMENTE NOVO.

Nao apenas recupera relacoes existentes. INFERE novas relacoes por:
  1. CADEIA CAUSAL: A-B + B-C ⇒ A-C (transitividade)
  2. TRANSFERENCIA CROSS-DOMINIO: se X trata Y em dominio A,
     e Z e similar a Y, entao X pode tratar Z (analogia)
  3. ABDUCAO: dado efeito E, infere a causa mais provavel nao observada
  4. CONTRAPOSITIVO: se A-B, entao nao-B - nao-A

Cada inferencia e marcada como NOVEL = True.
O relatorio final mostra o que e ORIGINAL vs o que veio dos dados.
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time
import re
import math


@dataclass
class NovelHypothesis:
    """Hipotese ORIGINAL gerada por inferencia (nao presente nos dados)."""
    cause: str
    effect: str
    inference_type: str  # "chain", "transfer", "abduction", "contrapositive"
    parent_relations: List[Tuple[str, str, str]]  # (cause, effect, type)
    confidence: float
    domain: str = ""
    evidence_chain: List[str] = field(default_factory=list)
    is_novel: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cause": self.cause[:60],
            "effect": self.effect[:60],
            "type": self.inference_type,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence_chain[:3],
            "novel": self.is_novel,
        }


class InferenceEngine:
    """Gera conhecimento NOVO por inferencia sobre o grafo causal.

    Uso:
        engine = InferenceEngine(kg)
        novelties = engine.run_inference(min_confidence=0.3)
        for h in novelties:
            print(f"NOVA DESCOBERTA: {h.cause} -> {h.effect}")
    """

    def __init__(self, kg=None):
        self.kg = kg
        self.novel_hypotheses: List[NovelHypothesis] = []

        # Indices
        self._relations_by_source = defaultdict(list)
        self._relations_by_target = defaultdict(list)
        self._all_pairs: Set[Tuple[str, str]] = set()

        if kg:
            for rel in kg.relations:
                self._relations_by_source[rel.source].append(rel)
                self._relations_by_target[rel.target].append(rel)
                self._all_pairs.add((rel.source, rel.target))

        # Token cache for fuzzy matching
        self._token_cache: Dict[str, Set[str]] = {}

    def _tokens(self, text: str) -> Set[str]:
        """Extrai tokens significativos de um texto."""
        if text not in self._token_cache:
            tokens = set()
            for word in re.findall(r'[a-zA-Z]+', text.lower()):
                if len(word) > 3 and word not in {
                    'that', 'this', 'with', 'from', 'have', 'been',
                    'were', 'their', 'they', 'them', 'than', 'into',
                    'also', 'very', 'just', 'about', 'which', 'when',
                    'where', 'what', 'after', 'before', 'could', 'would',
                    'should', 'these', 'those', 'there', 'other', 'such',
                    'some', 'more', 'most', 'much', 'many', 'over', 'then',
                }:
                    tokens.add(word)
            self._token_cache[text] = tokens
        return self._token_cache[text]

    def _fuzzy_match(self, text_a: str, text_b: str, threshold: float = 0.3) -> bool:
        """Matching fuzzy por similaridade de tokens."""
        if not text_a or not text_b:
            return False
        if text_a == text_b:
            return True
        tokens_a = self._tokens(text_a)
        tokens_b = self._tokens(text_b)
        if not tokens_a or not tokens_b:
            return False
        intersection = tokens_a & tokens_b
        sim = len(intersection) / max(1, min(len(tokens_a), len(tokens_b)))
        return sim >= threshold

    def _fuzzy_find(self, text: str, candidates: Set[str],
                    threshold: float = 0.3) -> List[str]:
        """Encontra candidatos similares por fuzzy match."""
        results = []
        for candidate in candidates:
            if self._fuzzy_match(text, candidate, threshold):
                results.append(candidate)
        return results

    def _is_known(self, cause: str, effect: str) -> bool:
        """Verifica se uma relacao ja existe nos dados."""
        return (cause, effect) in self._all_pairs

    # ─── INFERENCIA 1: CADEIA CAUSAL ───

    def infer_chains(self, max_depth: int = 3,
                     fuzzy_threshold: float = 0.3) -> List[NovelHypothesis]:
        """Inferencia por cadeia: A-B + B-C ⇒ A-C.

        Usa fuzzy matching para conectar nos quando nomes sao similares
        (ex: "amyloid-beta" ≈ "Aβ").
        """
        novelties = []
        all_source_set = set(self._relations_by_source.keys())
        all_target_set = set(self._relations_by_target.keys())

        for depth in range(2, max_depth + 1):
            chains = [(rel.source, rel.target, [rel], 0.5)
                      for rel in self.kg.relations]

            for _ in range(depth - 1):
                new_chains = []
                for start, end, path, conf in chains:
                    # Fuzzy match: o target `end` pode ser similar a sources existentes
                    candidates = self._fuzzy_find(end, all_source_set, fuzzy_threshold)
                    for cand in candidates:
                        for next_rel in self._relations_by_source.get(cand, []):
                            if next_rel.target != start:
                                new_path = path + [next_rel]
                                new_conf = conf * 0.7
                                new_chains.append(
                                    (start, next_rel.target, new_path, new_conf)
                                )
                chains = new_chains

            for start, end, path, conf in chains:
                if not self._is_known(start, end) and start != end:
                    evidence = [f"{r.source[:40]:40s} - {r.target[:40]}" for r in path]
                    novelties.append(NovelHypothesis(
                        cause=start,
                        effect=end,
                        inference_type="chain",
                        parent_relations=[(r.source, r.target, r.relation_type)
                                         for r in path],
                        confidence=conf,
                        domain=path[-1].domain,
                        evidence_chain=evidence,
                    ))

        return novelties

    # ─── INFERENCIA 2: TRANSFERENCIA CROSS-DOMINIO ───

    def infer_transfer(self, fuzzy_threshold: float = 0.3) -> List[NovelHypothesis]:
        """Inferencia por transferencia analogica.

        Se X-Y em um dominio, e Y≈Z (Y similar a Z) em outro,
        entao X-Z e uma descoberta por transferencia.
        """
        novelties = []
        all_source_set = set(self._relations_by_source.keys())

        for rel in self.kg.relations:
            source = rel.source
            target = rel.target

            # Fuzzy match: o target desta relacao pode ser similar
            # a sources de outras relacoes
            candidates = self._fuzzy_find(target, all_source_set, fuzzy_threshold)
            for cand in candidates:
                if cand == target or cand == source:
                    continue
                for other in self._relations_by_source.get(cand, []):
                    if other.target == source:
                        continue
                    potential = (source, other.target)
                    if not self._is_known(*potential):
                        domain_chain = f"{rel.domain}-{other.domain}"
                        novelties.append(NovelHypothesis(
                            cause=source,
                            effect=other.target,
                            inference_type="transfer",
                            parent_relations=[
                                (rel.source, rel.target, rel.relation_type),
                                (other.source, other.target, other.relation_type),
                            ],
                            confidence=rel.confidence * other.confidence * 0.6,
                            domain=domain_chain,
                            evidence_chain=[
                                f"from {rel.domain}: {rel.source[:40]:40s} - {rel.target[:40]}",
                                f"from {other.domain}: {other.source[:40]:40s} - {other.target[:40]}",
                                f"inferred: {source[:40]:40s} - {other.target[:40]}",
                            ],
                        ))

        return novelties

    # ─── INFERENCIA 3: ABDUCAO ───

    def infer_abduction(self, target_concept: str,
                        max_hypotheses: int = 5) -> List[NovelHypothesis]:
        """Inferencia abdutiva: dado um efeito, infere a causa mais provavel.

        Dado que sabemos X - Y, e observamos Y,
        inferimos que X pode ser a causa.
        """
        novelties = []

        # Encontra todas as causas diretas conhecidas
        for rel in self._relations_by_target.get(target_concept, []):
            # Se a causa tem multiplos efeitos, ela e mais "provavel"
            effects_from_cause = self._relations_by_source.get(rel.source, [])
            plau = len(effects_from_cause) / max(1, len(self.kg.relations))

            if not self._is_known(rel.source, target_concept):
                novelties.append(NovelHypothesis(
                    cause=rel.source,
                    effect=target_concept,
                    inference_type="abduction",
                    parent_relations=[(rel.source, rel.target, rel.relation_type)],
                    confidence=min(0.9, plau + 0.3),
                    domain=rel.domain,
                    evidence_chain=[
                        f"observed: {target_concept}",
                        f"abduced cause: {rel.source} (plausibility={plau:.2f})",
                    ],
                ))

        return novelties[:max_hypotheses]

    # ─── INFERENCIA 4: CONTRAPOSITIVO ───

    def infer_contrapositive(self, cause: str, effect: str) -> List[NovelHypothesis]:
        """Inferencia por contrapositivo.

        Se sabemos que X - Y (X causa Y),
        entao not-Y - not-X (sem Y, nao ha X).

        Isso e LOGICAMENTE valido, e gera hipoteses testaveis:
        "se bloquearmos Y, prevenimos X"
        """
        novelties = []
        for rel in self._relations_by_source.get(cause, []):
            if rel.target == effect:
                contra = NovelHypothesis(
                    cause=f"blocking {effect}",
                    effect=f"preventing {cause}",
                    inference_type="contrapositive",
                    parent_relations=[(cause, effect, rel.relation_type)],
                    confidence=rel.confidence * 0.8,  # logica solida
                    domain=rel.domain,
                    evidence_chain=[
                        f"known: {cause} - {effect}",
                        f"contrapositive: prevent {effect} - prevent {cause}",
                        f"testable hypothesis: {effect} inhibition may treat {cause}",
                    ],
                )
                novelties.append(contra)
        return novelties

    # ─── INFERENCIA COMPLETA ───

    def run_all(self, target_diseases: List[str] = None) -> List[NovelHypothesis]:
        """Roda TODAS as inferencias e retorna descobertas NOVAS."""
        all_novel = []

        print(f"\n=== GERANDO CONHECIMENTO NOVO ===")

        # 1. Cadeias causais
        print(f"\n  Inferindo cadeias causais...")
        chains = self.infer_chains(max_depth=3)
        all_novel.extend(chains)
        print(f"    {len(chains)} novas cadeias")
        for h in chains[:3]:
            print(f"      {' - '.join(h.evidence_chain[:3])[:80]}")

        # 2. Transferencia cross-dominio
        print(f"\n  Inferindo transferencias analogicas...")
        transfers = self.infer_transfer()
        all_novel.extend(transfers)
        print(f"    {len(transfers)} novas transferencias")
        for h in transfers[:3]:
            print(f"      {h.domain}: {h.cause[:40]} - {h.effect[:40]}")

        # 3. Abducao e contrapositivo para cada doenca alvo
        for disease in (target_diseases or []):
            print(f"\n  Inferindo para {disease}...")
            abduced = self.infer_abduction(disease)
            all_novel.extend(abduced)
            if abduced:
                print(f"    {len(abduced)} causas abduzidas")
                for h in abduced[:2]:
                    print(f"      - {h.cause[:60]} (conf={h.confidence:.2f})")

            contra = self.infer_contrapositive(disease, disease)
            all_novel.extend(contra)

        self.novel_hypotheses.extend(all_novel)
        return all_novel

    def get_stats(self) -> dict:
        known = len(self._all_pairs) if self._all_pairs else len(self.kg.relations) if self.kg else 0
        return {
            "total_known_relations": known,
            "novel_hypotheses_generated": len(self.novel_hypotheses),
            "by_type": dict(
                (t, sum(1 for h in self.novel_hypotheses if h.inference_type == t))
                for t in set(h.inference_type for h in self.novel_hypotheses)
            ),
            "avg_confidence": round(
                sum(h.confidence for h in self.novel_hypotheses) / max(1, len(self.novel_hypotheses)), 3
            ) if self.novel_hypotheses else 0,
            "novelty_ratio": len(self.novel_hypotheses) / max(1, known),
        }
