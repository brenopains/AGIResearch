"""AGI Persistência — Salva e carrega o CGE + SelfModel em disco"""
import os, pickle, time
from typing import Optional

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "save")
os.makedirs(SAVE_DIR, exist_ok=True)

def save_state(cge, self_model=None, tag="default"):
    """Salva CGE e SelfModel em disco."""
    path = os.path.join(SAVE_DIR, f"agi_state_{tag}.pkl")
    data = {
        "cge_nodes": dict(cge.nodes) if cge else {},
        "cge_edges": list(cge.edges) if cge else [],
        "self_model": None,
        "timestamp": time.time(),
    }
    if self_model:
        try:
            data["self_model"] = self_model.get_state()
        except:
            pass
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path

def load_state(cge, tag="default") -> bool:
    """Carrega CGE do disco. Retorna True se carregou."""
    path = os.path.join(SAVE_DIR, f"agi_state_{tag}.pkl")
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        data = pickle.load(f)
    if cge and "cge_nodes" in data:
        for nid, ndata in data["cge_nodes"].items():
            if nid not in cge.nodes:
                cge.nodes[nid] = ndata
        for e in data.get("cge_edges", []):
            if e not in cge.edges:
                cge.edges.append(e)
    return True

def list_saves() -> list:
    """Lista saves disponiveis."""
    saves = []
    for f in os.listdir(SAVE_DIR):
        if f.startswith("agi_state_"):
            path = os.path.join(SAVE_DIR, f)
            tag = f.replace("agi_state_", "").replace(".pkl", "")
            saves.append({"tag": tag, "path": path, "size": os.path.getsize(path),
                         "mtime": os.path.getmtime(path)})
    return sorted(saves, key=lambda s: -s["mtime"])
