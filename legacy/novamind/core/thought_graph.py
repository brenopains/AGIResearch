"""
ThoughtGraph ? Rastreamento completo do fluxo de pensamento.

Cada pergunta gera um grafo de nos e arestas mostrando:
  1. Pergunta original -> codificacao HDV
  2. Ativacao de conceitos no CGE (com similaridade)
  3. Recuperacao de memorias (episodica, semantica)
  4. Geracao de hipoteses
  5. Competicao entre hipoteses (com entropia, margem)
  6. Reversoes e descartes
  7. Resposta final

Isso permite VISUALIZAR o raciocinio, nao apenas medir accuracy.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class NodeType(Enum):
    QUESTION = "question"
    CONCEPT = "concept"
    MEMORY_EPISODIC = "memory_episodic"
    MEMORY_SEMANTIC = "memory_semantic"
    SIMULATION = "simulation"
    HYPOTHESIS = "hypothesis"
    COMPETITION = "competition"
    REVERSAL = "reversal"
    RESPONSE = "response"


@dataclass
class ThoughtNode:
    """Um no no grafo do pensamento."""
    id: str
    type: NodeType
    label: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label[:80],
            "confidence": round(self.confidence, 4),
            "timestamp": round(self.timestamp, 2),
            "metadata": {k: str(v)[:50] for k, v in self.metadata.items()},
        }


@dataclass
class ThoughtEdge:
    """Uma aresta no grafo do pensamento (fluxo de ativacao)."""
    source_id: str
    target_id: str
    weight: float = 1.0
    relation: str = "activates"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "weight": round(self.weight, 4),
            "relation": self.relation,
        }


class ThoughtGraph:
    """Grafo completo do pensamento para uma pergunta.
    
    Uso:
        graph = ThoughtGraph("como humanos pensam?")
        graph.add_concept("consciencia", 0.85)
        graph.add_hypothesis("System1_System2", 0.73)
        graph.connect("pergunta", "conceito_consciencia", 0.85)
        graph.connect("conceito_consciencia", "hipotese_System1", 0.73)
        print(graph.render())
    """
    
    def __init__(self, question: str = ""):
        self.question = question
        self.nodes: List[ThoughtNode] = []
        self.edges: List[ThoughtEdge] = []
        self._counter = 0
        
        # No raiz: a pergunta
        self._add_node("question", NodeType.QUESTION, question, confidence=1.0)
    
    def _add_node(self, node_id: str, ntype: NodeType, label: str, 
                  confidence: float = 0.0, metadata: dict = None) -> str:
        """Adiciona um no ao grafo."""
        self.nodes.append(ThoughtNode(
            id=node_id, type=ntype, label=label, 
            confidence=confidence, metadata=metadata or {}
        ))
        return node_id
    
    def add_concept(self, name: str, similarity: float, metadata: dict = None) -> str:
        """Registra um conceito ativado no CGE."""
        self._counter += 1
        nid = f"concept_{self._counter}"
        self._add_node(nid, NodeType.CONCEPT, name, confidence=similarity, metadata=metadata)
        return nid
    
    def add_memory(self, mem_type: str, content: str, confidence: float) -> str:
        """Registra uma memoria recuperada."""
        self._counter += 1
        nid = f"memory_{self._counter}"
        nt = NodeType.MEMORY_EPISODIC if mem_type == "episodic" else NodeType.MEMORY_SEMANTIC
        self._add_node(nid, nt, content, confidence=confidence)
        return nid
    
    def add_hypothesis(self, name: str, confidence: float, metadata: dict = None) -> str:
        """Registra uma hipotese gerada."""
        self._counter += 1
        nid = f"hyp_{self._counter}"
        self._add_node(nid, NodeType.HYPOTHESIS, name, confidence=confidence, metadata=metadata)
        return nid
    
    def add_competition(self, winner: str, entropy: float, margin: float) -> str:
        """Registra um ciclo de competicao."""
        self._counter += 1
        nid = f"comp_{self._counter}"
        self._add_node(nid, NodeType.COMPETITION, 
                      f"competition: winner={winner}", 
                      confidence=1.0 - entropy,
                      metadata={"entropy": entropy, "margin": margin})
        return nid
    
    def add_response(self, text: str, confidence: float) -> str:
        """Registra a resposta final."""
        nid = "response"
        self._add_node(nid, NodeType.RESPONSE, text, confidence=confidence)
        return nid
    
    def connect(self, source_id: str, target_id: str, weight: float = 1.0, 
                relation: str = "activates"):
        """Conecta dois nos (fluxo de ativacao)."""
        self.edges.append(ThoughtEdge(
            source_id=source_id, target_id=target_id, 
            weight=weight, relation=relation
        ))
    
    def get_stats(self) -> dict:
        """Estatisticas do grafo."""
        types = {}
        for n in self.nodes:
            t = n.type.value
            types[t] = types.get(t, 0) + 1
        total_weight = sum(e.weight for e in self.edges)
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "node_types": types,
            "total_flow": round(total_weight, 2),
            "question": self.question[:60],
        }
    
    def render(self) -> str:
        """Renderiza o grafo como texto (flow visual)."""
        def _s(t): return ''.join(c if ord(c) < 128 else '?' for c in str(t))
        lines = [
            f"\n{'='*70}",
            f"THOUGHT GRAPH: {_s(self.question)}",
            f"{'='*70}",
        ]
        
        # Agrupa por tipo
        by_type = {}
        for n in self.nodes:
            t = n.type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(n)
        
        # Ordem de exibicao
        type_order = ["question", "concept", "memory_episodic", "memory_semantic", 
                     "simulation", "hypothesis", "competition", "reversal", "response"]
        
        for t in type_order:
            if t not in by_type:
                continue
            nodes = by_type[t]
            label = t.upper()
            lines.append(f"\n  [{label}]")
            
            for n in nodes:
                bar = "#" * int(n.confidence * 20) + "." * (20 - int(n.confidence * 20))
                lines.append(f"  {_s(n.label)[:50]:50s} |{bar}| {n.confidence:.0%}")
                
                # Mostra conexoes de entrada
                incoming = [e for e in self.edges if e.target_id == n.id]
                for e in incoming:
                    src = next((x for x in self.nodes if x.id == e.source_id), None)
                    if src:
                        arrow = "^" if e.relation == "activates" else "->"
                        lines.append(f"    {arrow} de: {_s(src.label)[:40]} (peso={e.weight:.2f})")
        
        return "\n".join(lines)
    
    def render_flow(self) -> str:
        """Renderiza o fluxo cronologico do pensamento."""
        def _s(t): return ''.join(c if ord(c) < 128 else '?' for c in str(t))
        lines = [f"\nLINHA DO TEMPO DO PENSAMENTO:", f"{'-'*60}"]
        
        # Ordena por timestamp
        sorted_nodes = sorted(self.nodes, key=lambda n: n.timestamp)
        
        for i, n in enumerate(sorted_nodes):
            icon = {
                "question": "?",
                "concept": "?",
                "memory_episodic": "?",
                "memory_semantic": "?",
                "simulation": "?",
                "hypothesis": "?",
                "competition": "[scale]",
                "reversal": "?",
                "response": "?",
            }.get(n.type.value, "*")
            
            confidence_str = f"({n.confidence:.0%})" if n.confidence > 0 else ""
            lines.append(f"  {i:2d}. {icon} {_s(n.label)[:60]} {confidence_str}")
            
            # Mostra conexoes
            outgoing = [e for e in self.edges if e.source_id == n.id]
            for e in outgoing:
                target = next((x for x in sorted_nodes if x.id == e.target_id), None)
                if target:
                    arrow = "->" if e.weight > 0.5 else "..."
                    lines.append(f"      {arrow} {_s(target.label)[:50]} (peso={e.weight:.2f})")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Exporta o grafo completo como dict."""
        return {
            "question": self.question,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.get_stats(),
        }
    
    def save(self, path: str):
        """Salva o grafo em JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))


class ThoughtTracer:
    """Tracer que instrumenta CGE + MetaCognition para gerar ThoughtGraph.
    
    Uso:
        tracer = ThoughtTracer(cge, enc, meta)
        graph = tracer.trace("como humanos pensam?")
        print(graph.render_flow())
    """
    
    def __init__(self, cge=None, encoder=None, meta=None):
        self.cge = cge
        self.encoder = encoder
        self.meta = meta
    
    def trace(self, question: str) -> ThoughtGraph:
        """Executa o pipeline completo de pensamento e retorna o grafo."""
        graph = ThoughtGraph(question)
        
        if self.cge is None or self.encoder is None:
            graph.add_response("Sistema nao configurado", 0.0)
            return graph
        
        # --- 1. CODIFICA A PERGUNTA ---
        hdv = self.encoder.encode_bipolar(question)
        graph.connect("question", "question", 1.0)
        
        # --- 2. ATIVA CONCEITOS NO CGE ---
        results = []
        query_vec = self.encoder.encode_tensor(question)
        if hasattr(self.cge, 'nodes'):
            for nid, ndata in list(self.cge.nodes.items())[:300]:
                node_label = str(nid)
                node_vec = self.encoder.encode_tensor(node_label)
                cos = float((query_vec * node_vec).sum() / (query_vec.norm() * node_vec.norm() + 1e-8))
                if cos > 0.15:
                    results.append((node_label, ndata, cos))
        results.sort(key=lambda x: -x[2])
        results = results[:12]
        
        episodic_count = 0
        semantic_count = 0
        
        for nid, ndata, sim in results:
            if sim < 0.005:
                continue
            
            # Classifica o tipo de memoria
            conf = float(ndata.get("confidence", 0.5))
            access = int(ndata.get("access_count", 0))
            
            if access > 2:
                mem_type = "semantic"
                semantic_count += 1
            else:
                mem_type = "episodic"
                episodic_count += 1
            
            mem_id = graph.add_memory(mem_type, str(nid), sim)
            graph.connect("question", mem_id, sim)
        
        total_mem = episodic_count + semantic_count
        
        # --- 3. GERA HIPOTESES ---
        comp_result = None
        if self.meta:
            from core.cognition import Hypothesis, EvidenceItem, EvidenceSource, EvidenceRelation
            
            hypotheses = []
            for nid, ndata, sim in results[:5]:
                if sim < 0.01:
                    continue
                ev = EvidenceItem(
                    source=EvidenceSource.PERCEPTUAL,
                    relation=EvidenceRelation.RELATED,
                    confidence=sim,
                    raw_score=sim,
                    trace=f"CGE concept: {nid}",
                )
                h = Hypothesis(name=str(nid)[:30], category="related",
                              evidence_for=[ev], confidence=sim,
                              evidence_total=sim)
                hypotheses.append(h)
            
            if hypotheses:
                from core.cognition import Competition
                comp = Competition(iterations=3)
                comp_result = comp.run(hypotheses)
                
                for h in comp_result.hypotheses[:5]:
                    hyp_id = graph.add_hypothesis(h.name, h.confidence, {
                        "evidence_total": round(h.evidence_total, 3),
                        "category": h.category,
                    })
                    # Conecta das memorias que ativaram esta hipotese
                    for e in h.evidence_for[:3]:
                        mem_id = f"memory_{semantic_count}"  # simplificado
                        graph.connect(mem_id, hyp_id, e.confidence)
                
                comp_id = graph.add_competition(
                    comp_result.winner or "unknown",
                    comp_result.entropy,
                    comp_result.margin,
                )
                # Conecta hipoteses a competicao
                for h in comp_result.hypotheses[:3]:
                    hyp_idx = comp_result.hypotheses.index(h)
                    hyp_id = f"hyp_{hyp_idx + 1}"
                    graph.connect(hyp_id, comp_id, h.confidence)
                
                # Reversoes
                if comp_result.winner_reversals > 0:
                    rev_id = graph._add_node(
                        f"reversal_{time.time()}",
                        NodeType.REVERSAL,
                        f"winner changed {comp_result.winner_reversals} times",
                        confidence=comp_result.entropy,
                    )
                    graph.connect(comp_id, rev_id, comp_result.winner_reversals / 5)
        
        # --- 4. GERA RESPOSTA ---
        has_hypotheses = comp_result is not None and hasattr(comp_result, 'hypotheses') and len(comp_result.hypotheses) > 0
        if has_hypotheses:
            winner = comp_result.winner if comp_result else "unknown"
            response_text = f"{winner} (conf={comp_result.winner_confidence:.2f})"
            if comp_result:
                response_text += f" | margin={comp_result.margin:.3f} entropy={comp_result.entropy:.3f}"
        else:
            response_text = "Nao foi possivel gerar resposta (sem dados)"
        
        resp_conf = comp_result.winner_confidence if comp_result else 0.0
        resp_id = graph.add_response(response_text, resp_conf)
        
        if comp_result and comp_result.winner:
            winner_hyp = next((h for h in comp_result.hypotheses if h.name == comp_result.winner), None)
            if winner_hyp:
                hyp_idx = comp_result.hypotheses.index(winner_hyp)
                hyp_id = f"hyp_{hyp_idx + 1}"
                graph.connect(hyp_id, resp_id, comp_result.winner_confidence)
        
        # --- 5. METRICAS AGREGADAS ---
        # Calcula distribuicao por tipo
        total = len(results) if results else 1
        episodic_pct = round(episodic_count / max(1, total) * 100) if results else 0
        semantic_pct = round(semantic_count / max(1, total) * 100) if results else 0
        
        # Adiciona nos de simulacao/planejamento se aplicavel
        if comp_result and comp_result.diversity_score > 0:
            sim_id = graph._add_node("sim_1", NodeType.SIMULATION,
                                     f"competition diversity={comp_result.diversity_score:.2f}",
                                     confidence=comp_result.diversity_score)
            
            plan_id = graph._add_node("plan_1", NodeType.RESPONSE,  # reusing RESPONSE as placeholder
                                     f"winner={comp_result.winner} clusters={comp_result.num_clusters}",
                                     confidence=comp_result.winner_confidence)
            
            graph.connect(comp_id or "comp_1", sim_id, comp_result.diversity_score)
        
        return graph
