"""
EpisodicStream — Memória Episódica HDC Pura em Disco
=====================================================
Armazena experiências como um stream binário append-only de vetores HDC.
SEM SQLite. SEM banco de dados. Apenas vetores + metadados em arquivo .stream.

Formato:
  HEADER:  magic(4) + dim(4) + entry_count(8) + reserved(4) = 20 bytes
  ENTRY:   timestamp(8) + metadata_len(4) + metadata(utf8) + hdv_vec(dim*f32)

Busca: carrega chunks na GPU, computa similaridade de cosseno, retorna top_k.
"""

import os
import io
import json
import struct
import time
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path

MAGIC = b"HDVS"
DEFAULT_DIM = 10000
CHUNK_SIZE = 64  # entries per chunk (~2.5MB cada chunk)

class EpisodicStream:
    def __init__(self, path: str, dim: int = DEFAULT_DIM, device: Optional[str] = None):
        self.path = Path(path)
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._entry_count: int = 0
        self._fd: Optional[io.BufferedWriter] = None
        self._open_for_write: bool = False

        if self.path.exists():
            self._entry_count = self._read_header()
        else:
            self._write_header()

    # ─── Gerenciamento de Arquivo ────────────────────────────────

    def _read_header(self) -> int:
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Arquivo invalido: {self.path}")
            dim = struct.unpack("<I", f.read(4))[0]
            if dim != self.dim:
                raise ValueError(f"Dim mismatch: esperado {self.dim}, encontrado {dim}")
            count = struct.unpack("<Q", f.read(8))[0]
            return count

    def _write_header(self):
        with open(self.path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", self.dim))
            f.write(struct.pack("<Q", 0))
            f.write(b"\x00" * 4)

    def _update_entry_count(self, new_count: int):
        with open(self.path, "r+b") as f:
            f.seek(8)
            f.write(struct.pack("<Q", new_count))

    # ─── Escrita ─────────────────────────────────────────────────

    def append(self, hdv_vec: torch.Tensor, metadata: dict = None, timestamp: float = None) -> int:
        if metadata is None:
            metadata = {}
        if timestamp is None:
            timestamp = time.time()

        meta_bytes = json.dumps(metadata).encode("utf-8")
        vec_np = hdv_vec.detach().cpu().numpy().astype("float32")
        if vec_np.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vec_np.shape[0]} != {self.dim}")

        with open(self.path, "ab") as f:
            f.write(struct.pack("<d", timestamp))
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(vec_np.tobytes())

        new_count = self._entry_count + 1
        self._update_entry_count(new_count)
        self._entry_count = new_count
        return new_count

    def append_hdv(self, hdv_vector: np.ndarray, metadata: dict = None, timestamp: float = None) -> int:
        return self.append(torch.from_numpy(hdv_vector), metadata, timestamp)

    # ─── Leitura ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._entry_count

    @property
    def entry_count(self) -> int:
        return self._entry_count

    def _iter_entries(self, offset: int = 0, limit: Optional[int] = None) -> Iterator[Tuple[float, dict, np.ndarray]]:
        """Itera sobre entradas do arquivo. Yield (timestamp, metadata, hdv_vector_f32)."""
        entry_size = 8 + 4 + self.dim * 4
        with open(self.path, "rb") as f:
            f.seek(20 + offset * (8 + 4 + self.dim * 4))
            count = 0
            while True:
                if limit is not None and count >= limit:
                    break
                header = f.read(8 + 4)
                if len(header) < 12:
                    break
                timestamp, meta_len = struct.unpack("<dI", header)
                meta_bytes = f.read(meta_len)
                if len(meta_bytes) < meta_len:
                    break
                vec_bytes = f.read(self.dim * 4)
                if len(vec_bytes) < self.dim * 4:
                    break
                metadata = json.loads(meta_bytes.decode("utf-8"))
                hdv = np.frombuffer(vec_bytes, dtype="float32").copy()
                yield (timestamp, metadata, hdv)
                count += 1
                if limit is not None and count >= limit:
                    break

    def _read_entries_range(self, start: int, end: int) -> Tuple[torch.Tensor, List[float], List[dict]]:
        """Lê entradas de start até end (exclusive). Retorna (matrix, timestamps, metadatas)."""
        entry_size = 8 + 4 + self.dim * 4
        vecs = []
        timestamps = []
        metadatas = []
        with open(self.path, "rb") as f:
            f.seek(20 + start * entry_size)
            for _ in range(start, end):
                header = f.read(8 + 4)
                if len(header) < 12:
                    break
                ts, meta_len = struct.unpack("<dI", header)
                meta_bytes = f.read(meta_len)
                if len(meta_bytes) < meta_len:
                    break
                vec_bytes = f.read(self.dim * 4)
                if len(vec_bytes) < self.dim * 4:
                    break
                timestamps.append(ts)
                metadatas.append(json.loads(meta_bytes.decode("utf-8")))
                vecs.append(np.frombuffer(vec_bytes, dtype="float32"))
        if not vecs:
            return torch.empty(0, self.dim, device=self.device), [], []
        return torch.from_numpy(np.stack(vecs)).float().to(self.device), timestamps, metadatas

    # ─── Busca por Similaridade ──────────────────────────────────

    def query(self, query_vec: torch.Tensor, top_k: int = 5, min_sim: float = 0.3) -> List[Tuple[float, float, dict]]:
        """
        Busca as top_k entradas mais similares.
        Carrega chunks no GPU sequencialmente para memória O(1).

        Returns: Lista de (similaridade, timestamp, metadata) ordenada por sim descendente.
        """
        if self._entry_count == 0:
            return []

        qv = query_vec.detach().cpu().float()
        n = self._entry_count
        heap: List[Tuple[float, float, dict]] = []

        chunk_start = 0
        while chunk_start < n:
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            matrix, timestamps, metadatas = self._read_entries_range(chunk_start, chunk_end)
            if matrix.shape[0] == 0:
                break

            sims = torch.matmul(matrix.cpu(), qv) / self.dim
            for i in range(matrix.shape[0]):
                sim = sims[i].item()
                if sim >= min_sim:
                    heap.append((sim, timestamps[i], metadatas[i]))

            chunk_start = chunk_end

        heap.sort(key=lambda x: x[0], reverse=True)
        return heap[:top_k]

    def query_text(self, engine, text: str, top_k: int = 5, min_sim: float = 0.3) -> List[Tuple[float, float, dict]]:
        """Codifica texto como HDC e busca no stream."""
        mf = engine.encode_text(text)
        return self.query(mf.tensor, top_k, min_sim)

    def query_emotion(self, engine, emotion: str = "surprise", top_k: int = 5) -> List[Tuple[float, float, dict]]:
        """
        Busca entradas com metadados emocionais específicos.
        Ex: emotion="surprise" retorna entradas com ACh > 0.7.
        """
        results = []
        n = self._entry_count
        chunk_start = 0
        while chunk_start < n:
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            _, timestamps, metadatas = self._read_entries_range(chunk_start, chunk_end)
            for ts, meta in zip(timestamps, metadatas):
                nt = meta.get("neuromodulators", {})
                val = nt.get(emotion, 0)
                if val > 0.7:
                    results.append((val, ts, meta))
            chunk_start = chunk_end
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def query_recency(self, top_k: int = 5) -> List[Tuple[float, dict]]:
        """Retorna as top_k entradas mais recentes (últimos registros)."""
        n = self._entry_count
        start = max(0, n - top_k)
        results = []
        for ts, meta, hdv in self._iter_entries(offset=start, limit=top_k):
            results.append((ts, meta))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ─── Estatísticas ────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "path": str(self.path),
            "dim": self.dim,
            "entries": self._entry_count,
            "file_size_mb": round(self.path.stat().st_size / (1024 * 1024), 2) if self.path.exists() else 0,
            "device": self.device,
        }

    def get_recent(self, n: int = 10) -> List[dict]:
        """Retorna os metadados dos últimos n eventos."""
        results = []
        start = max(0, self._entry_count - n)
        for ts, meta, _ in self._iter_entries(offset=start, limit=n):
            results.append({"timestamp": ts, **meta})
        return results


def create_concept_vector(engine, text: str) -> np.ndarray:
    """Codifica texto como vetor HDC 10000-d."""
    mf = engine.encode_text(text)
    return mf.tensor.detach().cpu().numpy()
