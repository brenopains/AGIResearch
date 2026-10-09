"""
MultiScaleHDV — Memória com Múltiplas Escalas de Tempo de Plasticidade
=======================================================================
Implementa 3 camadas de memória com diferentes TTLs e transferência
controlada por saliência emocional — análogo ao sistema hipocampo-córtex.

Arquitetura:
  HDV-L1 (Fast / Working)   — TTL ~30s, decay exponencial, 500 itens
  HDV-L2 (Episódico)        — TTL ~1-4h, salvo só se emoção saliente
  HDV-L3 (Semântico)        — permanente, persiste em disco, poda automática

Referências:
  McClelland (1995) - Complementary learning systems
  O'Reilly & Norman (2002) - Hippocampal and neocortical contributions
  Kumaran (2016) - What learning systems do intelligent agents need?
"""

import os
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


# ─── Constantes ────────────────────────────────────────────────────────────────
L1_TTL_SECONDS   = 30.0      # Working memory: 30s
L2_TTL_SECONDS   = 3600.0    # Episódico: 1h (muda com saliência)
L1_CAPACITY      = 500
L2_CAPACITY      = 10_000
L3_PRUNE_SIM     = 0.95      # Similaridade acima desta → merges no L3
SALIENCE_DA_THR  = 0.6       # DA acima disso → sobe para L2
SALIENCE_CRH_THR = 0.7       # CRH acima disso → sobe para L2 (ameaça memorável)
HDV_DIM          = 10_000


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade coseno entre dois vetores bipolares."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class HDVEntry:
    """Uma entrada no HDV com metadados temporais e emocionais."""
    __slots__ = ["key", "vector", "ttl", "created_at", "nt_state", "metadata", "access_count"]

    def __init__(self, key: str, vector: np.ndarray, ttl: float,
                 nt_state: Optional[Dict] = None, metadata: Optional[Dict] = None):
        self.key         = key
        self.vector      = vector.astype(np.float32)
        self.ttl         = ttl
        self.created_at  = time.time()
        self.nt_state    = nt_state or {}
        self.metadata    = metadata or {}
        self.access_count = 0


class FastHDV:
    """
    L1 — Working Memory (30s TTL, decay exponencial).
    Usa OrderedDict para FIFO de capacidade.
    """

    def __init__(self, capacity: int = L1_CAPACITY, ttl: float = L1_TTL_SECONDS):
        self.capacity = capacity
        self.default_ttl = ttl
        self._store: OrderedDict[str, HDVEntry] = OrderedDict()

    def write(self, key: str, vector: np.ndarray,
              nt_state: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """Escreve ou atualiza um item. FIFO se capacidade excedida."""
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)  # remove o mais antigo
        entry = HDVEntry(key, vector, self.default_ttl, nt_state, metadata)
        self._store[key] = entry

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Busca por similaridade nos itens ainda vivos."""
        results = []
        for key, entry in self._store.items():
            if entry.ttl > 0:
                sim = cosine_sim(query, entry.vector)
                if sim > 0.1:
                    entry.access_count += 1
                    results.append({
                        "key": key, "vector": entry.vector,
                        "similarity": sim, "layer": "L1",
                        "metadata": entry.metadata, "nt_state": entry.nt_state,
                    })
        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    def tick(self, dt: float):
        """Aplica decay de TTL por passagem de tempo."""
        dead = []
        for key, entry in self._store.items():
            # Decay exponencial: TTL cai mais rápido no início
            entry.ttl *= (0.5 ** (dt / (self.default_ttl / 2)))
            if entry.ttl < 0.1:
                dead.append(key)
        for k in dead:
            del self._store[k]

    def items(self) -> List[HDVEntry]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)


class EpisodicHDV:
    """
    L2 — Memória Episódica (1-4h TTL, salvo por saliência emocional).
    Ordenado por saliência para replay durante consolidação.
    """

    def __init__(self, capacity: int = L2_CAPACITY, ttl: float = L2_TTL_SECONDS):
        self.capacity = capacity
        self.default_ttl = ttl
        self._store: Dict[str, HDVEntry] = {}

    def write(self, key: str, vector: np.ndarray,
              nt_state: Optional[Dict] = None, metadata: Optional[Dict] = None,
              ttl_multiplier: float = 1.0):
        """
        Escreve item. TTL pode ser aumentado por saliência emocional alta.
        CRH alto → TTL * 2 (memórias de perigo duram mais).
        """
        if len(self._store) >= self.capacity:
            # Remove o menos saliente
            weakest = min(self._store.values(), key=lambda e: e.nt_state.get("DA", 0.5))
            del self._store[weakest.key]

        crh = (nt_state or {}).get("CRH", 0.0)
        ttl = self.default_ttl * ttl_multiplier * (1.0 + crh)  # perigo → lembra mais
        entry = HDVEntry(key, vector, ttl, nt_state, metadata)
        self._store[key] = entry

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        results = []
        for key, entry in self._store.items():
            if entry.ttl > 0:
                sim = cosine_sim(query, entry.vector)
                if sim > 0.1:
                    entry.access_count += 1
                    results.append({
                        "key": key, "vector": entry.vector,
                        "similarity": sim, "layer": "L2",
                        "metadata": entry.metadata, "nt_state": entry.nt_state,
                    })
        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]

    def tick(self, dt: float):
        """Decay linear lento."""
        dead = []
        for key, entry in self._store.items():
            entry.ttl -= dt
            if entry.ttl <= 0:
                dead.append(key)
        for k in dead:
            del self._store[k]

    def replay_by_salience(self) -> List[HDVEntry]:
        """Retorna itens ordenados por saliência emocional (para replay no sono)."""
        def salience(e: HDVEntry) -> float:
            nt = e.nt_state
            return max(nt.get("DA", 0.5), nt.get("CRH", 0.0), nt.get("NA", 0.0))
        return sorted(self._store.values(), key=salience, reverse=True)

    def items(self) -> List[HDVEntry]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)


class SemanticHDV:
    """
    L3 — Memória Semântica (permanente, persiste em disco).
    Poda automática de itens redundantes (sim > 0.95).
    """

    def __init__(self):
        self._keys: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict] = []

    def write(self, key: str, vector: np.ndarray,
              metadata: Optional[Dict] = None, prune: bool = True):
        """Escreve item. Se prune=True, verifica duplicatas antes."""
        vec = vector.astype(np.float32)
        if prune and len(self._vectors) > 0:
            matrix = np.stack(self._vectors)
            sims = matrix @ vec / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec) + 1e-8)
            if np.max(sims) > L3_PRUNE_SIM:
                # Muito similar a algo existente: atualiza em vez de duplicar
                idx = int(np.argmax(sims))
                self._vectors[idx] = vec
                self._metadata[idx] = metadata or {}
                return

        self._keys.append(key)
        self._vectors.append(vec)
        self._metadata.append(metadata or {})

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self._vectors:
            return []
        matrix = np.stack(self._vectors)
        norm_q = np.linalg.norm(query)
        norms  = np.linalg.norm(matrix, axis=1)
        sims   = (matrix @ query) / (norms * norm_q + 1e-8)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "key": self._keys[i], "vector": self._vectors[i],
                "similarity": float(sims[i]), "layer": "L3",
                "metadata": self._metadata[i], "nt_state": {},
            }
            for i in top_idx if sims[i] > 0.1
        ]

    def save(self, path: str):
        """Persiste L3 em disco como .npz + .json de metadados."""
        if not self._vectors:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        matrix = np.stack(self._vectors)
        np.savez_compressed(path, vectors=matrix, keys=np.array(self._keys))
        meta_path = path.replace(".npz", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"keys": self._keys, "metadata": self._metadata}, f, ensure_ascii=False)

    def load(self, path: str):
        """Carrega L3 do disco."""
        if not os.path.exists(path):
            return
        data = np.load(path, allow_pickle=False)
        self._vectors = [data["vectors"][i] for i in range(len(data["vectors"]))]
        self._keys    = list(data["keys"].astype(str))
        meta_path = path.replace(".npz", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._metadata = meta.get("metadata", [{} for _ in self._keys])

    def __len__(self) -> int:
        return len(self._keys)


class MultiScaleHDV:
    """
    Sistema unificado de memória multi-escala.

    Hierarquia:
      L1 (30s) → L2 (1-4h, se saliente) → L3 (permanente, via sono)

    Busca em cascata: L1 primeiro (mais recente), depois L2, depois L3.
    """

    def __init__(self, l3_path: Optional[str] = None):
        self.L1 = FastHDV()
        self.L2 = EpisodicHDV()
        self.L3 = SemanticHDV()
        self.l3_path = l3_path
        self._last_tick = time.time()
        self._write_count = 0

        if l3_path:
            self.load_l3(l3_path)

    def write(self, key: str, vector: np.ndarray,
              nt_state: Optional[Dict] = None, metadata: Optional[Dict] = None):
        """
        Escreve em L1. Promove para L2 se emocionalmente saliente.

        Saliência: DA > 0.6 (recompensa) OU CRH > 0.7 (ameaça) → L2.
        """
        nt = nt_state or {}
        self.L1.write(key, vector, nt, metadata)
        self._write_count += 1

        # Gate de saliência para L2
        da  = nt.get("DA", 0.0)
        crh = nt.get("CRH", 0.0)
        na  = nt.get("NA", 0.0)
        if da > SALIENCE_DA_THR or crh > SALIENCE_CRH_THR or na > 0.7:
            ttl_mult = 1.0 + (crh * 2)  # ameaça = lembra mais
            self.L2.write(key, vector, nt, metadata, ttl_multiplier=ttl_mult)

    def search(self, query_vec: np.ndarray,
               layer: str = "all", top_k: int = 5) -> List[Dict]:
        """
        Busca em cascata: L1 → L2 → L3.
        Retorna lista de {key, vector, similarity, layer, metadata}.
        """
        self._maybe_tick()

        if layer == "L1":
            return self.L1.search(query_vec, top_k)
        elif layer == "L2":
            return self.L2.search(query_vec, top_k)
        elif layer == "L3":
            return self.L3.search(query_vec, top_k)

        # Busca em todas as camadas e consolida
        results = {}
        for r in self.L1.search(query_vec, top_k):
            results[r["key"]] = r
        for r in self.L2.search(query_vec, top_k):
            if r["key"] not in results or r["similarity"] > results[r["key"]]["similarity"]:
                results[r["key"]] = r
        for r in self.L3.search(query_vec, top_k):
            if r["key"] not in results or r["similarity"] > results[r["key"]]["similarity"]:
                results[r["key"]] = r

        sorted_results = sorted(results.values(), key=lambda x: -x["similarity"])
        return sorted_results[:top_k]

    def consolidate(self):
        """
        Consolidação L2 → L3 (chamado pelo sleep_mode).
        Replica itens mais salientes do L2 no L3 permanente.
        """
        consolidated = 0
        for entry in self.L2.replay_by_salience():
            self.L3.write(entry.key, entry.vector, entry.metadata, prune=True)
            consolidated += 1

        if self.l3_path and consolidated > 0:
            self.save_l3(self.l3_path)

        return consolidated

    def tick(self, dt: Optional[float] = None):
        """Aplica decay temporal em L1 e L2."""
        now = time.time()
        if dt is None:
            dt = now - self._last_tick
        self._last_tick = now
        self.L1.tick(dt)
        self.L2.tick(dt)

    def _maybe_tick(self):
        """Tick automático a cada chamada de search."""
        now = time.time()
        dt  = now - self._last_tick
        if dt > 1.0:  # só aplica se passou mais de 1s
            self.tick(dt)

    def save_l3(self, path: str):
        self.L3.save(path)

    def load_l3(self, path: str):
        self.L3.load(path)

    def get_state(self) -> Dict:
        return {
            "L1_items": len(self.L1),
            "L2_items": len(self.L2),
            "L3_items": len(self.L3),
            "total_writes": self._write_count,
            "l3_path": self.l3_path,
        }


# ─── Teste rápido ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    hdv = MultiScaleHDV()

    # Escreve com alta saliência (ameaça)
    v1 = np.sign(rng.standard_normal(HDV_DIM)).astype(np.float32)
    hdv.write("fogo_cozinha", v1, nt_state={"DA": 0.3, "CRH": 0.8, "NA": 0.7})

    # Escreve com baixa saliência (neutro)
    v2 = np.sign(rng.standard_normal(HDV_DIM)).astype(np.float32)
    hdv.write("rotina_normal", v2, nt_state={"DA": 0.5, "CRH": 0.1, "NA": 0.1})

    print("Estado inicial:", hdv.get_state())

    # Busca
    results = hdv.search(v1, top_k=3)
    for r in results:
        print(f"  {r['layer']} | {r['key']} | sim={r['similarity']:.3f}")

    # Tick de 35s (mata L1)
    hdv.tick(35.0)
    print("\nApós 35s (L1 expirou):", hdv.get_state())

    results2 = hdv.search(v1, layer="L2", top_k=3)
    print(f"Ainda no L2: {[r['key'] for r in results2]}")

    # Consolida no sono
    n = hdv.consolidate()
    print(f"\nConsolidou {n} itens no L3. Estado: {hdv.get_state()}")
