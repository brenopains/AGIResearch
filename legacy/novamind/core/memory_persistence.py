"""
Persistence Layer — Salva/carrega o MemorySystem em disco.

HDVs salvos como array numpy unico (rapido, compativel com memmap).
Estado (dict) salvo como pickle separado (leve).
Tudo em D:\AGI2\memories\.
"""
import os, json, time, pickle, shutil
import numpy as np
from typing import Optional
from .memory import MemorySystem, Episode, SemanticRule, Procedure

MEMORY_DIR = r"D:\AGI2\memories"
FULL_BACKUP = os.path.join(MEMORY_DIR, "memory_system.pkl")
HDV_BACKUP = os.path.join(MEMORY_DIR, "hdvs.npy")
os.makedirs(MEMORY_DIR, exist_ok=True)


def save_full_backup(mem: MemorySystem):
    """Backup completo: estado em pickle, HDVs em numpy array."""
    # Fecha qualquer mmap anterior
    if HDV_BACKUP in globals():
        try:
            del globals()["_hdv_mmap"]
        except: pass

    state = {
        "timestamp": time.time(),
        "total_episodes_stored": mem.total_episodes_stored,
        "total_episodes_compacted": mem.total_episodes_compacted,
        "total_episodes_forgotten": mem.total_episodes_forgotten,
        "current_emotional_context": mem.current_emotional_context,
        "current_arousal": mem.current_arousal,
        "compaction_events": mem.compaction_events[-100:],
    }
    ep_dicts = []
    hdvs_list = []
    for ep in mem.episodic:
        d = ep.to_dict()
        hdv = getattr(ep, "hdc_vector", None)
        if hdv is not None:
            d["_has_hdv"] = True
            # Make sure we have a regular ndarray, not memmap
            hdvs_list.append(np.asarray(hdv))
        else:
            d["_has_hdv"] = False
        ep_dicts.append(d)
    state["episodic"] = ep_dicts
    state["semantic"] = [r.to_dict() for r in mem.semantic]
    state["procedural"] = [p.to_dict() for p in mem.procedural]

    # Estado em pickle (sempre sobrescreve)
    tmp = FULL_BACKUP + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)
    shutil.move(tmp, FULL_BACKUP)

    # HDVs em numpy unico
    hdv_path = os.path.join(MEMORY_DIR, "hdvs.npy")
    if hdvs_list:
        try:
            arr = np.stack(hdvs_list, axis=0)
            tmp_hdv = hdv_path + ".tmp"
            np.save(tmp_hdv, arr)
            if os.path.exists(tmp_hdv):
                shutil.move(tmp_hdv, hdv_path)
        except Exception as e:
            print(f"[PERSIST] AVISO: erro ao salvar HDVs: {e}")
    elif os.path.exists(hdv_path):
        os.remove(hdv_path)


def load_full_backup() -> Optional[tuple]:
    """Carrega backup. Retorna (state_dict, hdvs_array) ou None."""
    if not os.path.exists(FULL_BACKUP):
        return None
    with open(FULL_BACKUP, "rb") as f:
        state = pickle.load(f)
    if isinstance(state, tuple):
        state = state[0]  # formato antigo (state, hdvs_list)
    # Carrega HDVs (copia para RAM, nao mmap, para poder salvar depois)
    hdv_path = os.path.join(MEMORY_DIR, "hdvs.npy")
    if os.path.exists(hdv_path):
        hdvs = np.load(hdv_path)  # carrega inteiro pra RAM
    else:
        hdvs = None
    return (state, hdvs)


def restore_memory(mem: MemorySystem, state: dict, hdvs: np.ndarray = None) -> MemorySystem:
    """Restaura o MemorySystem."""
    mem.total_episodes_stored = state.get("total_episodes_stored", 0)
    mem.total_episodes_compacted = state.get("total_episodes_compacted", 0)
    mem.total_episodes_forgotten = state.get("total_episodes_forgotten", 0)
    mem.current_emotional_context = state.get("current_emotional_context", 0.0)
    mem.current_arousal = state.get("current_arousal", 0.0)
    mem.compaction_events = state.get("compaction_events", [])

    for i, ep_data in enumerate(state.get("episodic", [])):
        ep = Episode(ep_data.get("content", {}), ep_data.get("type", "interaction"))
        ep.id = ep_data["id"]
        ep.timestamp = ep_data["timestamp"]
        ep.activation = ep_data["activation"]
        ep.access_count = ep_data["access_count"]
        ep.compressed = ep_data["compressed"]
        ep.importance = ep_data["importance"]
        ep.emotional_valence = ep_data["emotional_valence"]
        ep.surprise = ep_data["surprise"]
        ep.consolidation_count = ep_data.get("consolidation_count", 0)
        if hdvs is not None and i < len(hdvs) and ep_data.get("_has_hdv"):
            ep.hdc_vector = hdvs[i]
        mem.episodic.append(ep)

    for r_data in state.get("semantic", []):
        rule = SemanticRule(r_data.get("type", "fact"), r_data.get("content", {}))
        rule.id = r_data["id"]
        rule.confidence = r_data["confidence"]
        rule.source_episode_count = r_data["source_episodes"]
        rule.activation = r_data["activation"]
        rule.emotional_valence = r_data.get("emotional_valence", 0.0)
        rule.generalization_power = r_data.get("generalization_power", 0.0)
        rule.compression_ratio = r_data.get("compression_ratio", 0.0)
        mem.semantic.append(rule)

    for p_data in state.get("procedural", []):
        proc = Procedure(p_data["name"], p_data["steps"], p_data.get("goal", ""))
        proc.id = p_data["id"]
        proc.success_count = p_data.get("success_rate", 0.5) * 10
        proc.activation = p_data.get("activation", 0.7)
        mem.procedural.append(proc)

    return mem
