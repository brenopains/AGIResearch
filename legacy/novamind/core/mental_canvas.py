"""
MentalCanvas — Espaco de Trabalho Externo Interativo

Diferente do CognitiveNotebook (que e sequencial, tipo um diario),
o Canvas e ESPACIAL: objetos podem ser posicionados, conectados,
agrupados, movidos.

A AGI usa o canvas para:
  - Desenhar diagramas de raciocinio
  - Conectar hipoteses visualmente
  - Montar fluxogramas
  - Rascunhar ideias antes de consolida-las no CGE

Nao e memoria. E superficie de trabalho.
"""
import time, uuid, math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class CanvasNode:
    """Um elemento no canvas."""
    id: str = ""
    type: str = "note"       # "note", "concept", "hypothesis", "image", "equation"
    content: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 60.0
    color: str = "#1a2638"
    tags: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class CanvasEdge:
    """Uma conexao entre dois nos no canvas."""
    from_id: str = ""
    to_id: str = ""
    label: str = ""
    style: str = "solid"   # "solid", "dashed", "dotted"
    color: str = "#576574"


@dataclass
class CanvasFrame:
    """Um estado completo do canvas (para salvar/restaurar)."""
    nodes: List[CanvasNode] = field(default_factory=list)
    edges: List[CanvasEdge] = field(default_factory=list)
    view_x: float = 0.0
    view_y: float = 0.0
    zoom: float = 1.0
    created_at: float = field(default_factory=time.time)


class MentalCanvas:
    """Quadro branco interativo.

    Diferenca do CognitiveNotebook:
      - Notebook: sequencial, tipo diario, persistente
      - Canvas: espacial, tipo quadro branco, sessao
    """

    def __init__(self, cge=None):
        self.cge = cge
        self.nodes: Dict[str, CanvasNode] = {}
        self.edges: List[CanvasEdge] = []
        self._history: List[CanvasFrame] = []

    def add_node(self, type: str = "note", content: str = "",
                 x: float = 0.0, y: float = 0.0, **kwargs) -> str:
        """Adiciona um no ao canvas."""
        nid = f"cn_{uuid.uuid4().hex[:8]}"
        node = CanvasNode(id=nid, type=type, content=content, x=x, y=y, **kwargs)
        self.nodes[nid] = node
        return nid

    def connect(self, from_id: str, to_id: str, label: str = "") -> bool:
        """Conecta dois nos."""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append(CanvasEdge(from_id=from_id, to_id=to_id, label=label))
            return True
        return False

    def remove(self, node_id: str):
        """Remove um no e suas conexoes."""
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                     if e.from_id != node_id and e.to_id != node_id]

    def layout_hierarchy(self, root_id: str):
        """Organiza nos em hierarquia a partir de um no raiz."""
        if root_id not in self.nodes:
            return
        children = [e.to_id for e in self.edges if e.from_id == root_id]
        root = self.nodes[root_id]
        root.x, root.y = 400, 50
        for i, cid in enumerate(children[:8]):
            if cid in self.nodes:
                self.nodes[cid].x = 100 + i * 200
                self.nodes[cid].y = 200

    def snapshot(self) -> CanvasFrame:
        """Salva estado atual como frame (undo point)."""
        frame = CanvasFrame(
            nodes=list(self.nodes.values()),
            edges=list(self.edges),
        )
        self._history.append(frame)
        if len(self._history) > 20:
            self._history = self._history[-20:]
        return frame

    def restore(self, frame: CanvasFrame):
        """Restaura estado de um frame."""
        self.nodes = {n.id: n for n in frame.nodes}
        self.edges = list(frame.edges)

    def to_cge(self):
        """Consolida o canvas no CGE (promove de rascunho para conhecimento)."""
        if self.cge is None:
            return
        import torch
        for node in self.nodes.values():
            if node.type == "hypothesis" and node.content:
                nid = f"_canvas_{hashlib.md5(node.content.encode()).hexdigest()[:8]}"
                g = torch.Generator(); g.manual_seed(int(hashlib.md5(node.content.encode()).hexdigest()[:8], 16))
                vec = torch.sign(torch.randn(10000, generator=g))
                self.cge.create_node(nid, vec)
                if nid in self.cge.nodes:
                    self.cge.nodes[nid]["canvas_content"] = node.content[:200]
        self.clear()

    def clear(self):
        """Limpa o canvas (nao salva)."""
        self.nodes.clear()
        self.edges.clear()

    def get_state(self) -> Dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "undo_history": len(self._history),
        }
