"""
CognitiveDeltaExtractor — Computa o delta entre dois estados cognitivos do CGE.

Onde o DeltaExtractor visual computa diferencas entre grids (objetos,
cores, posicoes), este modulo computa diferencas entre GRAFOS DE CONCEITOS
(nos, arestas, confiancas, tipos de relacao).

Isso permite que o mesmo mecanismo de Program Induction opere sobre
PENSAMENTO ABSTRATO — nao apenas sobre pixels.

Exemplo:
  Estado A: CGE tem {whale->mammal (conf=0.7), mammal->breathes (conf=0.8)}
  Estado B: CGE tem {whale->mammal (conf=0.9), mammal->breathes (conf=0.8),
                      whale->breathes (conf=0.7) [NOVA]}
  
  Delta: {added: whale->breathes, confidence_up: whale->mammal de 0.7->0.9}
  
  Operador inferido: transitive_inference(whale, mammal, breathes)
"""
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class CGEState:
    """Snapshot de um estado do CGE em um momento."""
    nodes: Dict[str, dict] = field(default_factory=dict)
    edges: List[dict] = field(default_factory=list)

    @classmethod
    def from_cge(cls, cge) -> 'CGEState':
        """Captura estado atual do CGE."""
        nodes = {}
        for nid, ndata in cge.nodes.items():
            nodes[nid] = {
                "confidence": float(ndata.get("confidence", 0.5)),
                "state": ndata.get("state", "visible"),
                "access_count": int(ndata.get("access_count", 0)),
            }
        edges = []
        for e in cge.edges:
            edges.append({
                "from": str(e.get("from", "")),
                "to": str(e.get("to", "")),
                "confidence": float(e.get("confidence", 0.5)),
                "edge_type": str(e.get("edge_type", "related")),
            })
        return cls(nodes=nodes, edges=edges)


@dataclass
class CognitiveDelta:
    """Delta entre dois estados CGE.

    Campos:
      nodes_added: nos que nao existiam no estado anterior
      nodes_removed: nos que sumiram
      nodes_confidence_changed: {node_id: (conf_antes, conf_depois)}
      edges_added: arestas novas
      edges_removed: arestas removidas
      edges_confidence_changed: {(from, to, type): (conf_antes, conf_depois)}
      relation_types_added: novos tipos de relacao observados
    """
    nodes_added: Dict[str, float] = field(default_factory=dict)
    nodes_removed: List[str] = field(default_factory=list)
    nodes_confidence_changed: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    edges_added: List[dict] = field(default_factory=list)
    edges_removed: List[dict] = field(default_factory=list)
    edges_confidence_changed: Dict[tuple, Tuple[float, float]] = field(default_factory=dict)
    relation_types_added: Set[str] = field(default_factory=set)

    @property
    def has_changes(self) -> bool:
        return bool(self.nodes_added or self.edges_added or self.nodes_removed or self.edges_removed)

    @property
    def transformation_type(self) -> str:
        """Classifica em alto nivel o tipo de transformacao cognitiva."""
        if self.edges_added and not self.nodes_added:
            # Novas relacoes entre conceitos existentes → inferencia
            return "inference"
        if self.nodes_added and not self.edges_added:
            # Novos conceitos → expansao
            return "expansion"
        if self.edges_removed:
            # Remocao de relacoes → revisao
            return "revision"
        if self.nodes_confidence_changed:
            # Confianca mudou sem estrutura → calibracao
            return "calibration"
        if self.relation_types_added:
            return "new_relation_type"
        return "unknown"

    def summarize(self) -> dict:
        return {
            "type": self.transformation_type,
            "nodes_added": len(self.nodes_added),
            "nodes_removed": len(self.nodes_removed),
            "nodes_confidence_changed": len(self.nodes_confidence_changed),
            "edges_added": len(self.edges_added),
            "edges_removed": len(self.edges_removed),
            "relation_types_added": list(self.relation_types_added),
        }

    def to_description(self) -> str:
        parts = []
        if self.edges_added:
            for e in self.edges_added[:5]:
                parts.append(f"new: {e['from']} --[{e.get('edge_type','?')}]--> {e['to']}")
        if self.nodes_added:
            for nid, conf in list(self.nodes_added.items())[:5]:
                parts.append(f"new concept: {nid} (conf={conf:.2f})")
        if self.nodes_confidence_changed:
            for nid, (before, after) in list(self.nodes_confidence_changed.items())[:5]:
                parts.append(f"confidence {nid}: {before:.2f}->{after:.2f}")
        if self.edges_removed:
            parts.append(f"removed {len(self.edges_removed)} relations")
        return " | ".join(parts) if parts else "no change"


class CognitiveDeltaExtractor:
    """Extrai delta entre dois estados do CGE (antes/depois de uma operacao).

    Uso:
        before = CGEState.from_cge(cge)
        # ... sistema opera ...
        after = CGEState.from_cge(cge)
        delta = extractor.extract(before, after)
        # delta.transformation_type → "inference", "revision", etc.
    """

    def extract(self, before: CGEState, after: CGEState) -> CognitiveDelta:
        """Computa o delta cognitivo entre dois estados."""
        delta = CognitiveDelta()

        # NOS
        before_ids = set(before.nodes.keys())
        after_ids = set(after.nodes.keys())

        for nid in after_ids - before_ids:
            delta.nodes_added[nid] = float(after.nodes[nid].get("confidence", 0.5))

        for nid in before_ids - after_ids:
            delta.nodes_removed.append(nid)

        for nid in before_ids & after_ids:
            b_conf = float(before.nodes[nid].get("confidence", 0.5))
            a_conf = float(after.nodes[nid].get("confidence", 0.5))
            if abs(b_conf - a_conf) > 0.05:
                delta.nodes_confidence_changed[nid] = (b_conf, a_conf)

        # ARESTAS
        # Normaliza arestas para tuplas compaveis
        def edge_key(e: dict) -> tuple:
            return (str(e.get("from", "")), str(e.get("to", "")),
                    str(e.get("edge_type", "related")))

        before_edges = {edge_key(e): e for e in before.edges}
        after_edges = {edge_key(e): e for e in after.edges}

        before_keys = set(before_edges.keys())
        after_keys = set(after_edges.keys())

        for k in after_keys - before_keys:
            delta.edges_added.append(after_edges[k])

        for k in before_keys - after_keys:
            delta.edges_removed.append(before_edges[k])

        for k in before_keys & after_keys:
            b_conf = float(before_edges[k].get("confidence", 0.5))
            a_conf = float(after_edges[k].get("confidence", 0.5))
            if abs(b_conf - a_conf) > 0.05:
                delta.edges_confidence_changed[k] = (b_conf, a_conf)

        # Novos tipos de relacao
        before_types = set(e.get("edge_type", "related") for e in before.edges)
        after_types = set(e.get("edge_type", "related") for e in after.edges)
        delta.relation_types_added = after_types - before_types

        return delta
