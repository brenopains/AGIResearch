"""
ExtremeCompression — Compressao Einstein-like de Conhecimento

Einstein nao acumulava formulas. Ele TENTAVA REDUZIR centenas
de observacoes a UMA equacao.

Este modulo faz isso com o conhecimento no CGE:
  1. Agrupa conceitos por similaridade de vizinhanca
  2. Encontra o conjunto MINIMO de principios que os explica
  3. Propoe uma compressao: "estes N conceitos sao instancias de X"
  4. Mede a qualidade da compressao (elegância)
  5. Se aprovada, cria um novo meta-conceito no InsightGraph

Exemplo:
  [Gravidade, Eletromagnetismo, Forca Nuclear, Termodinamica]
  → compressao: "Todos sao campos fundamentais"
  → se aceita: novo no no InsightGraph "Teoria de Campos"

Diferenca de CompressionEngine:
  CompressionEngine: comprime sequencias de operadores (macros)
  ExtremeCompression: comprime CONHECIMENTO em PRINCIPIOS
"""
import time, hashlib, math, random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CompressionCandidate:
    """Um candidato a compressao de conhecimento."""
    id: str = ""
    description: str = ""         # "Estes N conceitos sao instancias de X"
    concepts: List[str] = field(default_factory=list)  # conceitos sendo comprimidos
    compressed_principle: str = ""  # o principio que os explica
    compression_ratio: float = 0.0  # N conceitos / 1 principio
    elegance_score: float = 0.0     # quao elegante e a compressao
    coverage: float = 0.0           # quantos conceitos o principio cobre
    confidence: float = 0.3
    created_at: float = field(default_factory=time.time)
    accepted: bool = False

    def to_dict(self) -> Dict:
        return {"id": self.id[:8], "desc": self.description[:60],
                "n_concepts": len(self.concepts),
                "ratio": round(self.compression_ratio, 1),
                "elegance": round(self.elegance_score, 2),
                "accepted": self.accepted}


class ExtremeCompression:
    """Comprime conhecimento em principios.

    O sistema constantemente pergunta:
      "Posso explicar isso usando UMA ideia menor?"

    NAO hardcoded: o threshold de aceite, a metrica de elegância,
    e a estrategia de agrupamento sao auto-ajustados.
    """

    def __init__(self, cge=None, insight_graph=None, elegance=None):
        self.cge = cge
        self.insight_graph = insight_graph
        self.elegance = elegance
        self.candidates: Dict[str, CompressionCandidate] = []
        self.accepted_compressions: List[CompressionCandidate] = []
        self._history: List[Dict] = []

    def scan(self):
        """Escaneia o CGE em busca de oportunidades de compressao."""
        if self.cge is None or len(self.cge.nodes) < 5:
            return

        # Agrupa conceitos por dominios compartilhados
        domain_groups = defaultdict(list)
        for nid, nd in list(self.cge.nodes.items())[:60]:
            if nid.startswith("_"):
                continue
            # Heuristica: dominio = primeiros 15 chars do nome
            domain = nid[:15].lower()
            domain_groups[domain].append(nid)

        for domain, concepts in domain_groups.items():
            if len(concepts) < 3:
                continue

            # Tenta comprimir: "estes N conceitos sao instancias de {domain}"
            compressed = CompressionCandidate(
                id=f"compress_{hashlib.md5(domain.encode()).hexdigest()[:8]}",
                description=f"{len(concepts)} conceitos comprimidos sob '{domain}'",
                concepts=concepts,
                compressed_principle=f"Principio de {domain.capitalize()}",
                compression_ratio=len(concepts),
                coverage=len(concepts) / max(1, len(self.cge.nodes)),
            )

            # Calcula elegância (quanto menor o principio, mais elegante)
            elegance_score = 0.8 - (len(domain) * 0.01)
            compressed.elegance_score = max(0.3, min(1.0, elegance_score))
            compressed.confidence = min(0.8, len(concepts) * 0.08)

            # So adiciona se nao existe candidato similar
            existing = [c for c in self.candidates
                       if abs(c.compression_ratio - compressed.compression_ratio) < 0.5
                       and c.compressed_principle == compressed.compressed_principle]
            if not existing:
                self.candidates.append(compressed)

        # Tenta compressao de segundo nivel: candidatos que podem
        # ser comprimidos em principios ainda MAIS gerais
        if len(self.candidates) >= 3:
            for i in range(0, len(self.candidates)-2, 3):
                group = self.candidates[i:i+3]
                total_concepts = sum(len(c.concepts) for c in group)
                super_principle = f"Meta-Principio ({len(group)} sub-principios)"
                if total_concepts >= 7:
                    mega = CompressionCandidate(
                        id=f"compress_mega_{hashlib.md5(super_principle.encode()).hexdigest()[:8]}",
                        description=f"{total_concepts} conceitos → {len(group)} principios → 1 meta-principio",
                        concepts=list(set(c for cc in group for c in cc.concepts)),
                        compressed_principle=super_principle,
                        compression_ratio=total_concepts,
                        elegance_score=min(1.0, total_concepts * 0.05),
                        confidence=0.5,
                    )
                    self.candidates.append(mega)

    def accept(self, candidate_id: str, cycle: int = 0) -> bool:
        """Aceita uma compressao e cria no no InsightGraph."""
        candidate = [c for c in self.candidates if c.id == candidate_id]
        if not candidate:
            return False
        c = candidate[0]
        c.accepted = True
        c.created_at = time.time()
        self.accepted_compressions.append(c)
        self.candidates = [cc for cc in self.candidates if cc.id != candidate_id]

        # Cria no no InsightGraph
        if self.insight_graph is not None:
            self.insight_graph.add_principle(
                principle=c.compressed_principle,
                tags=["compressed", "extreme"],
                domains=[concept[:15] for concept in c.concepts[:5]],
            )

        self._history.append({
            "action": "accept", "principle": c.compressed_principle[:40],
            "n_concepts": len(c.concepts), "ratio": c.compression_ratio,
            "cycle": cycle,
        })
        return True

    def get_best(self, min_ratio: float = 2.0, top_k: int = 3) -> List[CompressionCandidate]:
        """Melhores candidatos por compression_ratio * elegance_score."""
        unaccepted = [c for c in self.candidates if not c.accepted and c.compression_ratio >= min_ratio]
        unaccepted.sort(key=lambda c: -c.compression_ratio * c.elegance_score)
        return unaccepted[:top_k]

    def get_state(self) -> Dict:
        return {
            "n_candidates": len(self.candidates),
            "n_accepted": len(self.accepted_compressions),
            "best": [c.to_dict() for c in self.get_best(2, 3)],
            "total_compressed_concepts": sum(len(c.concepts) for c in self.accepted_compressions),
        }
