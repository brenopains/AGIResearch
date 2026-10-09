"""
EpisodeStore — Loader robusto com memmap + SQLite + HNSW.

HDVs em numpy.memmap (nunca carrega tudo na RAM).
Metadados em SQLite (consultas eficientes).
HNSW para busca vetorial ultra-rápida (<10ms).
"""
import os, json, time, pickle, sqlite3, shutil
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

MEMORY_DIR = r"D:\AGI2\memories"
STATE_PKL = os.path.join(MEMORY_DIR, "memory_system.pkl")
HDVS_NPY = os.path.join(MEMORY_DIR, "hdvs.npy")
EPISODE_DB = os.path.join(MEMORY_DIR, "episodes.db")
HNSW_INDEX = os.path.join(MEMORY_DIR, "hnsw.index")

os.makedirs(MEMORY_DIR, exist_ok=True)


class EpisodeStore:
    """Armazenamento persistente com busca vetorial."""

    def __init__(self):
        self.hdvs: Optional[np.memmap] = None
        self.n_episodes: int = 0
        self.hdv_dim: int = 10000
        self._semantic: List[dict] = []
        self._procedural: List[dict] = []
        self.n_concepts: int = 0
        self.db_path = EPISODE_DB
        self._hnsw = None  # hnswlib.Index

    # ─── Carregamento ──────────────────────────────────────

    def load_backup(self) -> bool:
        if not os.path.exists(STATE_PKL):
            print("[STORE] Nenhum backup")
            return False

        # Estado primeiro (para saber n_episodes)
        with open(STATE_PKL, "rb") as f:
            raw = pickle.load(f)
        state = raw[0] if isinstance(raw, tuple) else raw
        n_episodes_state = len(state.get("episodic", []))

        # HDVs como memmap
        if os.path.exists(HDVS_NPY):
            file_size = os.path.getsize(HDVS_NPY)
            file_n = file_size // (10000 * 4)
            n = min(file_n, n_episodes_state) if n_episodes_state > 0 else file_n
            arr = np.memmap(HDVS_NPY, dtype=np.float32, mode='r', shape=(n, 10000))
            self.hdvs = arr
            self.n_episodes = n
            self.hdv_dim = 10000
            print(f"[STORE] HDVs: {arr.shape} (memmap)")
        else:
            self.n_episodes = n_episodes_state

        # Popula SQLite
        self._populate_db(state)
        self._semantic = state.get("semantic", [])
        self.n_concepts = len(self._semantic)
        self._procedural = state.get("procedural", [])

        print(f"[STORE] {self.n_episodes} episodios, {self.n_concepts} conceitos")
        return True

    def _populate_db(self, state: dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY, type TEXT, timestamp REAL, activation REAL,
            importance REAL, valence REAL, surprise REAL,
            consolidated INTEGER, access_count INTEGER,
            content TEXT, has_hdv INTEGER DEFAULT 0
        )""")
        for col in ["importance", "surprise", "type"]:
            c.execute(f"CREATE INDEX IF NOT EXISTS idx_episodes_{col} ON episodes({col})")
        conn.commit()

        episodes = state.get("episodic", [])
        if not episodes:
            conn.close(); return

        existing = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        if existing >= len(episodes):
            conn.close(); return

        c.execute("DELETE FROM episodes")
        batch = []
        for i, ep in enumerate(episodes):
            batch.append((
                ep["id"], ep.get("type", "interaction"),
                ep.get("timestamp", 0.0), ep.get("activation", 0.0),
                ep.get("importance", 0.5), ep.get("emotional_valence", 0.0),
                ep.get("surprise", 0.0), 1 if ep.get("compressed") else 0,
                ep.get("access_count", 0),
                json.dumps(ep.get("content", {}), ensure_ascii=False),
                1 if ep.get("_has_hdv") and i < self.n_episodes else 0,
            ))
        c.executemany("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
        conn.commit(); conn.close()

    # ─── HNSW ──────────────────────────────────────────────

    def build_hnsw(self, force: bool = False):
        """Constroi ou carrega indice HNSW."""
        import hnswlib
        if not force and os.path.exists(HNSW_INDEX):
            try:
                self._hnsw = hnswlib.Index(space='cosine', dim=self.hdv_dim)
                self._hnsw.load_index(HNSW_INDEX)
                print(f"[STORE] HNSW carregado ({os.path.getsize(HNSW_INDEX)/1e6:.0f}MB)")
                return
            except Exception as e:
                print(f"[STORE] Recriando HNSW ({e})")

        if self.hdvs is None or self.n_episodes == 0:
            print("[STORE] Sem HDVs para indexar")
            return

        print(f"[STORE] Construindo HNSW ({self.n_episodes} vetores)...")
        t0 = time.time()
        self._hnsw = hnswlib.Index(space='cosine', dim=self.hdv_dim)
        self._hnsw.init_index(
            max_elements=self.n_episodes,
            ef_construction=200,
            M=32,
            random_seed=42,
        )
        # Adiciona em lotes (para nao usar muita RAM temporaria)
        batch_size = 10000
        for start in range(0, self.n_episodes, batch_size):
            end = min(start + batch_size, self.n_episodes)
            batch = np.array(self.hdvs[start:end], dtype=np.float32, copy=True)
            self._hnsw.add_items(batch, np.arange(start, end))
            if (start // batch_size) % 5 == 0:
                print(f"  {end}/{self.n_episodes}")

        self._hnsw.set_ef(100)  # search time / accuracy tradeoff
        self._hnsw.save_index(HNSW_INDEX)
        print(f"[STORE] HNSW pronto: {time.time()-t0:.0f}s, {os.path.getsize(HNSW_INDEX)/1e6:.0f}MB")

    def query_hdv(self, query_hdv: np.ndarray, top_k: int = 5, modality: str = None) -> List[dict]:
        """Busca por similaridade de HDV via HNSW. Filtra por modalidade."""
        if self._hnsw is None:
            return self.query_text("", top_k)
        q = np.array([query_hdv], dtype=np.float32)
        labels, distances = self._hnsw.knn_query(q, k=top_k * 3)
        results = []
        conn = sqlite3.connect(self.db_path)
        for idx, dist in zip(labels[0], distances[0]):
            row = conn.execute("SELECT * FROM episodes WHERE rowid=?", (int(idx) + 1,)).fetchone()
            if row:
                content = json.loads(row[9]) if row[9] else {}
                ep_type = row[1] or ""
                if modality and modality not in str(ep_type):
                    continue
                results.append({
                    "id": row[0], "type": ep_type,
                    "importance": row[4], "valence": row[5], "surprise": row[6],
                    "content": content, "score": float(1 - dist),
                    "distance": float(dist), "has_hdv": bool(row[10]),
                })
                if len(results) >= top_k:
                    break
        conn.close()
        return results

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        """Busca HDV com filtro de modalidade texto (fallback textual)."""
        if self._hnsw is not None:
            query_hdv = self._text_to_hdv(query)
            results = self.query_hdv(query_hdv, top_k=top_k, modality="text")
            if len(results) >= top_k:
                return results
            more = self.query_hdv(query_hdv, top_k=top_k)
            existing_ids = {r["id"] for r in results}
            for m in more:
                if m["id"] not in existing_ids:
                    results.append(m)
                if len(results) >= top_k:
                    break
            return results
        return self.query_text(query, top_k=top_k)

    def query_text(self, query: str, top_k: int = 5) -> List[dict]:
        """Busca por similaridade de texto (fallback). Prioriza episodios textuais."""
        # Filtra stopwords da query
        STOP = frozenset({"what","why","how","when","where","who","which","a","an","the",
                          "is","are","was","were","be","been","being","have","has","had",
                          "do","does","did","will","would","could","should","may","might",
                          "can","shall","to","of","in","on","at","by","for","with","without",
                          "and","but","or","nor","not","so","yet","if","because","as",
                          "this","that","these","those","it","its","they","them",
                          "he","she","we","you","i","me","him","her","us",
                          "no","yes","just","very","all","each","every","some","any"})
        query_words = {w.lower() for w in query.split()
                      if w.lower() not in STOP and len(w) > 2}
        conn = sqlite3.connect(self.db_path)
        # Busca episodios textuais primeiro, depois outros tipos
        rows = conn.execute(
            "SELECT id, type, importance, valence, surprise, content, has_hdv FROM episodes "
            "WHERE type IN ('perception_text','perception_caption','perception_digit') "
            "ORDER BY importance DESC LIMIT ?",
            (top_k * 30,)
        ).fetchall()
        # Se nao achou suficientes, complementa com outros tipos
        if len(rows) < top_k * 5:
            more = conn.execute(
                "SELECT id, type, importance, valence, surprise, content, has_hdv FROM episodes "
                "WHERE type NOT IN ('perception_text','perception_caption','perception_digit') "
                "ORDER BY importance DESC LIMIT ?",
                (top_k * 10,)
            ).fetchall()
            rows.extend(more)
        conn.close()

        query_words = set(query.lower().split())
        scored = []
        for row in rows:
            ep_id, etype, imp, val, surprise, content_json, has_hdv = row
            content = json.loads(content_json) if content_json else {}
            text = str(content.get("input", "") or content.get("caption", "") or "")
            ep_words = set(text.lower().split())
            jaccard = len(query_words & ep_words) / max(1, len(query_words | ep_words)) if query_words and ep_words else 0
            score = jaccard * 50.0 + imp * 0.1  # Jaccard domina o score
            scored.append((score, ep_id, etype, imp, val, surprise, content, has_hdv))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, ep_id, etype, imp, val, surprise, content, has_hdv in scored[:top_k]:
            results.append({
                "id": ep_id, "type": etype, "importance": imp, "valence": val,
                "surprise": surprise, "content": content, "score": round(score, 4), "has_hdv": has_hdv,
            })
        return results

    def get_hdv(self, idx: int) -> Optional[np.ndarray]:
        if self.hdvs is not None and 0 <= idx < self.n_episodes:
            return self.hdvs[idx]
        return None

    def get_hdv_by_id(self, ep_id: str) -> Optional[np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT rowid FROM episodes WHERE id=?", (ep_id,)).fetchone()
        conn.close()
        if row and row[0] <= self.n_episodes:
            return self.get_hdv(row[0] - 1)
        return None

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        avg_imp = conn.execute("SELECT AVG(importance) FROM episodes").fetchone()[0] or 0
        avg_sur = conn.execute("SELECT AVG(surprise) FROM episodes").fetchone()[0] or 0
        n_cons = conn.execute("SELECT COUNT(*) FROM episodes WHERE consolidated=1").fetchone()[0]
        conn.close()
        return {
            "episodes": n, "avg_importance": round(avg_imp, 3),
            "avg_surprise": round(avg_sur, 3), "consolidated": n_cons,
            "concepts": self.n_concepts,
            "hdv_shape": (self.n_episodes, self.hdv_dim) if self.hdvs is not None else (0, 0),
        }

    def __len__(self):
        return self.n_episodes

    # ─── Text → HDV ───────────────────────────────────────

    def _text_to_hdv(self, text: str) -> np.ndarray:
        """Converte texto em HDV ESTRUTURADO via parser + bind de papeis."""
        if not hasattr(self, "_struct_encoder"):
            from .structured_encoder import StructuredHDVEncoder
            self._struct_encoder = StructuredHDVEncoder()
        hdv, _ = self._struct_encoder.encode(text)
        return hdv
        return hdv.astype(np.float32)

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        """Busca HDV: converte texto pra HDV, busca no HNSW.

        Funciona porque episodios de texto e caption foram
        codificados com o mesmo algoritmo _text_to_hdv.
        """
        if self._hnsw is not None:
            query_hdv = self._text_to_hdv(query)
            return self.query_hdv(query_hdv, top_k=top_k)
        return self.query_text(query, top_k=top_k)
