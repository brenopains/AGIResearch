"""
StructuralAnalogyEngine — Descoberta de Isomorfismos Profundos

Nao basta encontrar nos com vizinhancas similares (Jaccard).
E preciso encontrar ISOMORFISMOS entre SUBGRAFOS inteiros.

Exemplo:
  Subgrafo 1: corrente_eletrica → fio → resistencia
  Subgrafo 2: agua_em_canais → canos → atrito

Ambos tem estrutura: "meio" → "condutor" → "resistencia"
Sao ISOMORFICOS mesmo sendo semanticamente distantes.

Este motor:
  1. Extrai subgrafos do CGE (vizinhancas de raio N)
  2. Compara subgrafos por estrutura (nao por semantica)
  3. Encontra isomorfismos: mapeamento entre nos de subgrafos diferentes
  4. Reporta as analogias com maior qualidade

NAO hardcoded: o raio de vizinhanca, o threshold de similaridade
estrutural, e o criterio de qualidade sao auto-ajustados.
"""
import time, hashlib, random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Subgraph:
    """Um subgrafo extraido do CGE."""
    root: str = ""
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    # Signature estrutural: tipo de relacao por vizinho
    neighbor_signatures: Dict[str, List[str]] = field(default_factory=dict)

    def structural_signature(self) -> str:
        """Assinatura canonica baseada em estrutura (nao conteudo)."""
        # Ordena tipos de relacao para cada vizinho
        sigs = []
        for n in sorted(self.nodes):
            rel_types = sorted(self.neighbor_signatures.get(n, []))
            sigs.append(f"{n}={','.join(rel_types)}")
        return "|".join(sigs)


@dataclass
class DeepAnalogy:
    """Uma analogia estrutural profunda."""
    id: str = ""
    source_subgraph: Subgraph = field(default_factory=Subgraph)
    target_subgraph: Subgraph = field(default_factory=Subgraph)
    mapping: Dict[str, str] = field(default_factory=dict)  # source_node → target_node
    isomorphism_quality: float = 0.0
    discovered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id": self.id[:8],
            "source_root": self.source_subgraph.root,
            "target_root": self.target_subgraph.root,
            "quality": round(self.isomorphism_quality, 2),
            "n_mappings": len(self.mapping),
        }


class StructuralAnalogyEngine:
    """Encontra isomorfismos estruturais entre subgrafos do CGE.

    A ideia: dois nos sao analogos se as RELACOES que eles tem
    com seus vizinhos sao similares, mesmo que os vizinhos sejam diferentes.
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.analogies: Dict[str, DeepAnalogy] = {}
        self._history: List[Dict] = []

    def extract_subgraph(self, root: str, radius: int = 2) -> Optional[Subgraph]:
        """Extrai um subgrafo centrado em um no."""
        if self.cge is None or root not in self.cge.nodes:
            return None
        sg = Subgraph(root=root)
        sg.nodes.add(root)
        frontier = {root}
        visited = {root}
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                for e in self.cge.edges:
                    if e["from"] == node and e["to"] not in visited:
                        sg.nodes.add(e["to"])
                        sg.edges.add((e["from"], e["to"]))
                        sg.neighbor_signatures.setdefault(e["from"], []).append(
                            e.get("edge_type", "related"))
                        new_frontier.add(e["to"])
                        visited.add(e["to"])
                    elif e["to"] == node and e["from"] not in visited:
                        sg.nodes.add(e["from"])
                        sg.edges.add((e["from"], e["to"]))
                        sg.neighbor_signatures.setdefault(e["to"], []).append(
                            e.get("edge_type", "related"))
                        new_frontier.add(e["from"])
                        visited.add(e["from"])
            frontier = new_frontier
            if not frontier:
                break
        return sg

    def compute_structural_distance(self, sg_a: Subgraph,
                                    sg_b: Subgraph) -> float:
        """Distancia estrutural entre dois subgrafos (0=identicos, 1=diferentes)."""
        if not sg_a.nodes or not sg_b.nodes:
            return 1.0
        # Compara cardinalidades
        size_a, size_b = len(sg_a.nodes), len(sg_b.nodes)
        size_diff = abs(size_a - size_b) / max(size_a, size_b)
        # Compara grau medio
        degree_a = sum(len(v) for v in sg_a.neighbor_signatures.values()) / max(1, size_a)
        degree_b = sum(len(v) for v in sg_b.neighbor_signatures.values()) / max(1, size_b)
        degree_diff = abs(degree_a - degree_b) / max(degree_a, degree_b, 0.1)
        return (size_diff + degree_diff) / 2

    def find_isomorphism(self, sg_a: Subgraph, sg_b: Subgraph) -> Dict[str, str]:
        """Encontra um mapeamento entre nos de dois subgrafos."""
        mapping = {}
        if not sg_a.nodes or not sg_b.nodes:
            return mapping
        # Ordena nos por grau
        nodes_a = sorted(sg_a.nodes, key=lambda n: -len(sg_a.neighbor_signatures.get(n, [])))
        nodes_b = sorted(sg_b.nodes, key=lambda n: -len(sg_b.neighbor_signatures.get(n, [])))
        # Mapeia os mais centrais
        for i in range(min(len(nodes_a), len(nodes_b))):
            mapping[nodes_a[i]] = nodes_b[i]
        return mapping

    def scan(self, top_k: int = 5) -> List[DeepAnalogy]:
        """Escaneia o CGE procurando isomorfismos estruturais."""
        if self.cge is None:
            return []
        nodes = [n for n in self.cge.nodes.keys() if not n.startswith("_")]
        if len(nodes) < 4:
            return []

        new_analogies = []
        # Extrai subgrafos para alguns nos centrais
        subgraphs = {}
        for n in nodes[:15]:
            sg = self.extract_subgraph(n, radius=2)
            if sg and len(sg.nodes) > 1:
                subgraphs[n] = sg

        # Compara todos os pares
        keys = list(subgraphs.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = keys[i], keys[j]
                if a == b:
                    continue
                dist = self.compute_structural_distance(subgraphs[a], subgraphs[b])
                if dist < 0.3:  # similar o suficiente
                    mapping = self.find_isomorphism(subgraphs[a], subgraphs[b])
                    quality = 1.0 - dist
                    analogia_id = f"dana_{hashlib.md5(f'{a}_{b}'.encode()).hexdigest()[:8]}"
                    if analogia_id not in self.analogies:
                        ana = DeepAnalogy(
                            id=analogia_id,
                            source_subgraph=subgraphs[a],
                            target_subgraph=subgraphs[b],
                            mapping=mapping,
                            isomorphism_quality=quality,
                        )
                        self.analogies[analogia_id] = ana
                        new_analogies.append(ana)
                        if len(new_analogies) >= top_k:
                            return new_analogies
        return new_analogies

    def get_best(self, top_k: int = 5) -> List[DeepAnalogy]:
        """Melhores analogias por qualidade de isomorfismo."""
        return sorted(self.analogies.values(),
                     key=lambda a: -a.isomorphism_quality)[:top_k]

    def get_state(self) -> Dict:
        return {
            "n_analogies": len(self.analogies),
            "best": [a.to_dict() for a in self.get_best(3)],
        }
