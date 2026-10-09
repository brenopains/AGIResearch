"""
StateBridge — Ponte de estado AGI em tempo real

Singleton global que armazena o estado cognitivo atual.
Qualquer agente (MTB sandbox, pipeline, etc.) escreve aqui.
O servidor HTTP le daqui. Sem mock. Sem simulacao.
"""
import time
import threading
import json
from typing import Dict, Optional, List, Any


class StateBridge:
    """Estado cognitivo compartilhado em tempo real."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.state: Dict[str, Any] = {
            "cycle": 0,
            "uptime": 0.0,
            "running": False,
            "mode": "idle",
            "operator": None,
            "attention": None,
            "error": {},
            "cge": {"nodes": {}, "edges": [], "n_nodes": 0, "n_edges": 0},
            "self_model": {},
            "operators": {},
            "simulation": {"branches": []},
            "lifestream": {"current_media_type": "text", "content_title": "",
                          "content_body": "", "source": "", "reliability": 0.0,
                          "curiosity_targets": [], "current_concept": ""},
            "hdv_vectors": [],  # lista de (name, vector_sample) para preview
            "archetypes": [],
            "concept_history": [],  # [(cycle, n_nodes, n_edges)]
            "thoughts": [],
            "last_update": time.time(),
        }
        self.subscribers: List = []

    def update(self, **kwargs):
        """Atualiza campos do estado (thread-safe)."""
        with self._lock:
            self.state.update(kwargs)
            self.state["last_update"] = time.time()

    def set_from_cge(self, cge):
        """Extrai estado real do CognitiveGraphEngine."""
        if cge is None:
            return
        with self._lock:
            nodes_out = []
            for nid, node in list(cge.nodes.items())[:50]:
                vec = node.get("vector")
                vec_sample = None
                if vec is not None:
                    try:
                        v = vec.flatten()[:20]
                        vec_sample = [float(x) for x in v.tolist()]
                    except:
                        pass
                nodes_out.append({
                    "id": nid,
                    "confidence": node.get("confidence", 0.5),
                    "valence": node.get("emotional_valence", 0.0),
                    "arousal": node.get("emotional_arousal", 0.0),
                    "vector_sample": vec_sample,
                })
            edges_out = []
            for e in cge.edges[-100:]:
                edges_out.append({
                    "from": e["from"], "to": e["to"],
                    "weight": e.get("weight", 0.5),
                    "type": e.get("edge_type", ""),
                })
            self.state["cge"] = {
                "nodes": nodes_out,
                "edges": edges_out,
                "n_nodes": len(cge.nodes),
                "n_edges": len(cge.edges),
            }

    def set_from_self_model(self, sm):
        """Extrai estado real do SelfModel."""
        if sm is None:
            return
        try:
            s = sm.get_state()
            self.state["self_model"] = s
        except:
            pass

    def set_from_operators(self, op_system):
        """Extrai estado real do OperatorSystem."""
        if op_system is None:
            return
        try:
            s = op_system.get_state()
            self.state["operators"] = s
        except:
            pass

    def set_from_mtb_sandbox(self, sandbox):
        """Extrai estado do MTB sandbox."""
        if sandbox is None or sandbox.world is None:
            return
        w = sandbox.world
        with self._lock:
            self.state["lifestream"]["current_media_type"] = "mtb_grid"
            self.state["lifestream"]["content_title"] = (
                f"MTB World {w.world_id}: {w.physics['energy_flow']} + {w.physics['gravity']}"
            )
            self.state["lifestream"]["content_body"] = (
                f"Source: {w.source} -> Target: {w.target}\n"
                f"Materials: {', '.join(list(w.materials.keys())[:10])}\n"
                f"Grid: {w.width}x{w.height}, Energy at target: {w.get_energy_at_target():.1f}"
            )
            self.state["lifestream"]["source"] = "MTB Procedural World"
            self.state["lifestream"]["reliability"] = 1.0
            self.state["lifestream"]["curiosity_targets"] = list(
                sandbox._explored_materials.keys() if hasattr(sandbox, '_explored_materials') else []
            )[:5]
            self.state["lifestream"]["current_concept"] = "energy_flow"

            # MTB grid data para visualizacao (copia das celulas)
            grid_data = []
            for y in range(min(w.height, 8)):
                row = []
                for x in range(min(w.width, 12)):
                    mat = w.grid[y][x]
                    energy = w.energy_grid[y][x]
                    is_source = (x, y) == w.source
                    is_target = (x, y) == w.target
                    mat_cond = 0.0
                    mat_blocking = False
                    if mat and mat in w.materials:
                        m = w.materials[mat]
                        mat_cond = m.get_conductance()
                        mat_blocking = m.is_blocking()
                    row.append({
                        "x": x, "y": y, "material": mat,
                        "energy": round(energy, 1),
                        "conductance": round(mat_cond, 2),
                        "blocking": mat_blocking,
                        "is_source": is_source,
                        "is_target": is_target,
                    })
                grid_data.append(row)
            self.state["mtb_grid"] = grid_data
            self.state["mtb_agi_pos"] = list(sandbox.agi_pos)

    def add_thought(self, text: str, mode: str = "info"):
        """Adiciona pensamento ao log."""
        with self._lock:
            logs = self.state.get("thoughts", [])
            logs.append({"time": time.time(), "text": text, "mode": mode})
            if len(logs) > 50:
                logs = logs[-50:]
            self.state["thoughts"] = logs

    def add_concept_history(self, cycle: int, n_nodes: int, n_edges: int):
        """Registra crescimento de conceitos."""
        with self._lock:
            hist = self.state.get("concept_history", [])
            hist.append({"cycle": cycle, "n": n_nodes, "n_edges": n_edges})
            if len(hist) > 120:
                hist = hist[-120:]
            self.state["concept_history"] = hist

    def get_state(self) -> Dict:
        """Retorna estado serializavel como JSON. Sempre."""
        with self._lock:
            # Simplificar estado: remover campos problemáticos
            s = {}
            for k, v in self.state.items():
                if k == "cge":
                    nodes = []
                    for n in v.get("nodes", []):
                        nodes.append({sk: sv for sk, sv in n.items()
                                     if sk not in ("vector", "vector_sample")})
                    s["cge"] = {"nodes": nodes[:30], "edges": v.get("edges", [])[-50:],
                                "n_nodes": v.get("n_nodes", 0), "n_edges": v.get("n_edges", 0)}
                elif k == "thoughts":
                    s[k] = [{"text": str(t.get("text", "")), "mode": str(t.get("mode", "info")),
                             "time": t.get("time", 0)} for t in v[-30:]]
                elif k == "concept_history":
                    s[k] = v[-60:]
                else:
                    try:
                        json.dumps(v)
                        s[k] = v
                    except:
                        s[k] = str(v)[:100]
            return s

    def clear(self):
        self._init()

    @staticmethod
    def instance() -> "StateBridge":
        return StateBridge()
