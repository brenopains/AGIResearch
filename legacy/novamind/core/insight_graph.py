"""
InsightGraph — Grafo de Principios Universais Acima de Paradigmas

Enquanto o InsightLibrary armazena heuristicas (regras do tipo "se X entao Y"),
o InsightGraph armazena PRINCIPIOS que atravessam dominios inteiros.

Exemplos de nos no grafo:
  "Complexidade emerge de regras simples"
  "Estruturas hierarquicas reutilizam padroes"
  "Sistemas estaveis equilibram exploracao e conservacao"
  "Representacoes diferentes podem preservar a mesma estrutura"

Cada principio:
  - Conecta-se a paradigmas, teorias e heuristicas que o exemplificam
  - Pode ser decomposto em sub-principios
  - Tem confianca baseada em quantos dominios diferentes o confirmam
"""
import time, hashlib, math
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class PrincipleNode:
    """Um principio universal no InsightGraph."""
    id: str = ""
    principle: str = ""
    tags: List[str] = field(default_factory=list)

    # Conexoes com outros niveis
    paradigm_ids: List[str] = field(default_factory=list)
    theory_ids: List[str] = field(default_factory=list)
    heuristic_ids: List[str] = field(default_factory=list)

    # Metricas
    confidence: float = 0.3
    n_domains: int = 0          # em quantos dominios diferentes foi confirmado
    domains: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"id": self.id, "principle": self.principle[:100],
                "confidence": self.confidence, "n_domains": self.n_domains,
                "domains": self.domains[:3]}


PRIMORDIAL_PRINCIPLES = [
    PrincipleNode(
        principle="Complexidade emerge de regras simples",
        tags=["emergence", "complexity", "simplicity"],
        confidence=0.9, n_domains=10,
        domains=["physics", "biology", "computation", "mathematics"]),
    PrincipleNode(
        principle="Estruturas hierarquicas reutilizam padroes",
        tags=["hierarchy", "patterns", "reuse"],
        confidence=0.85, n_domains=8,
        domains=["biology", "society", "language", "engineering"]),
    PrincipleNode(
        principle="Sistemas estaveis equilibram exploracao e conservacao",
        tags=["stability", "exploration", "balance"],
        confidence=0.85, n_domains=7,
        domains=["ecology", "economics", "cognition", "evolution"]),
    PrincipleNode(
        principle="Tudo que e grande e feito de coisas pequenas",
        tags=["decomposition", "complexity", "scale"],
        confidence=0.95, n_domains=12,
        domains=["physics", "biology", "mathematics", "engineering"]),
    PrincipleNode(
        principle="Tudo que e complexo e feito de coisas simples",
        tags=["complexity", "simplicity", "emergence"],
        confidence=0.9, n_domains=10,
        domains=["physics", "biology", "computation"]),
    PrincipleNode(
        principle="Representacoes diferentes podem preservar a mesma estrutura",
        tags=["representation", "isomorphism", "invariance"],
        confidence=0.8, n_domains=6,
        domains=["mathematics", "cognition", "language"]),
    PrincipleNode(
        principle="Restricoes reduzem o espaco de busca e aumentam a eficiencia",
        tags=["constraints", "search", "efficiency"],
        confidence=0.85, n_domains=5,
        domains=["cognition", "evolution", "engineering"]),
    PrincipleNode(
        principle="O conhecimento se organiza em camadas de abstracao",
        tags=["abstraction", "layers", "organization"],
        confidence=0.85, n_domains=7,
        domains=["cognition", "biology", "society", "engineering"]),
    PrincipleNode(
        principle="Analogias revelam estruturas invariantes entre dominios",
        tags=["analogy", "invariance", "structure"],
        confidence=0.8, n_domains=5,
        domains=["cognition", "mathematics", "science"]),
    PrincipleNode(
        principle="Sistemas que nao evoluem estagnam",
        tags=["evolution", "adaptation", "change"],
        confidence=0.85, n_domains=8,
        domains=["biology", "society", "technology", "cognition"]),
]


class InsightGraph:
    """Grafo de principios universais.

    Os principios sao descobertos pela Forja observando padroes
    que aparecem em MULTIPLOS dominios simultaneamente.

    Um principio e forte se aparece em dominios DISTANTES.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.nodes: Dict[str, PrincipleNode] = {}
        self._edges: List[Tuple[str, str, str]] = []  # (from_id, relation, to_id)

        for p in PRIMORDIAL_PRINCIPLES:
            pid = f"principle_{hashlib.md5(p.principle.encode()).hexdigest()[:8]}"
            p.id = pid
            self.nodes[pid] = p

    def add_principle(self, principle: str, tags=None,
                      domains=None) -> PrincipleNode:
        """Adiciona um novo principio (descoberto pela Forja)."""
        pid = f"principle_{hashlib.md5(principle.encode()).hexdigest()[:8]}"
        if pid in self.nodes:
            existing = self.nodes[pid]
            if domains:
                existing.domains = list(set(existing.domains + domains))
                existing.n_domains = len(existing.domains)
                existing.confidence = min(1.0, existing.confidence + 0.05)
            return existing

        node = PrincipleNode(
            id=pid, principle=principle, tags=tags or [],
            domains=domains or [], n_domains=len(domains or []),
            confidence=min(0.7, len(domains or []) * 0.1),
        )
        self.nodes[pid] = node
        self._sync_to_cge(node)
        return node

    def connect(self, principle_id: str, target_id: str,
                relation: str = "exemplified_by"):
        """Conecta um principio a um paradigma/teoria/heuristica."""
        self._edges.append((principle_id, relation, target_id))

    def discover_from_paradigms(self, paradigms: List[Any],
                                 theories: List[Any]) -> List[PrincipleNode]:
        """Descobre novos principios observando paradigmas e teorias."""
        new_principles = []
        if len(paradigms) < 2 or len(theories) < 3:
            return new_principles

        # Extrair dominios de paradigmas
        paradigm_domains = set()
        for p in paradigms:
            data = p.to_dict() if hasattr(p, 'to_dict') else (p if isinstance(p, dict) else {})
            approach = data.get("approach", "") if isinstance(data, dict) else ""
            if approach:
                paradigm_domains.add(approach)

        # Se multiplos paradigmas compartilham dominios, pode haver principio
        if len(paradigm_domains) >= 2:
            new = self.add_principle(
                principle=f"Multiplos paradigmas convergem para {', '.join(list(paradigm_domains)[:2])}",
                tags=["convergence", "paradigm"],
                domains=list(paradigm_domains),
            )
            if new:
                new_principles.append(new)

        return new_principles

    def get_cross_domain_principles(self, min_domains: int = 3) -> List[PrincipleNode]:
        """Retorna principios que aparecem em N dominios diferentes."""
        return [n for n in self.nodes.values() if n.n_domains >= min_domains]

    def get_state(self) -> Dict:
        return {
            "n_principles": len(self.nodes),
            "cross_domain": len(self.get_cross_domain_principles(3)),
            "top": [n.to_dict() for n in sorted(
                self.nodes.values(), key=lambda n: -n.confidence)[:5]],
        }

    def _sync_to_cge(self, node: PrincipleNode):
        if self.cge is None:
            return
        import torch
        nid = node.id
        g = torch.Generator()
        g.manual_seed(int(hashlib.md5(node.principle.encode()).hexdigest()[:8], 16))
        vec = torch.sign(torch.randn(10000, generator=g))
        self.cge.create_node(nid, vec)
        if nid in self.cge.nodes:
            self.cge.nodes[nid]["type"] = "principle"
            self.cge.nodes[nid]["principle_data"] = node.to_dict()
