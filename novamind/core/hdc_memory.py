"""
HDC Memory — Memória Semântica Hiperdimensional para NovaMind
================================================================
Carrega os .hdv compilados na VRAM e permite consulta por
similaridade de cosseno durante o ciclo think().

Integra-se como camada de memória de longo prazo entre a
Memory Layer (L3) e a Reasoning Layer (L4) do NovaMind.
"""

import os
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .coglang import Manifold
from .grounding_engine import GroundingEngine


# Locais padrão dos .hdv compilados
HDV_DIRS = [
    "D:/novamind_data/hdv",
    "D:/disk_d_brain",
]


class HDCMemory:
    """
    Memória Hiperdimensional: carrega .hdv na VRAM e consulta por
    similaridade de cosseno. Integrada ao NovaMind.think().
    """

    def __init__(self, device: Optional[str] = None, hdv_paths: Optional[List[str]] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = GroundingEngine(device=torch.device(self.device))
        self.matrices: Dict[str, torch.Tensor] = {}
        self.indices: Dict[str, List[Dict]] = {}
        self.sources: Dict[str, str] = {}
        self._loaded_paths: List[str] = []

        if hdv_paths:
            for p in hdv_paths:
                self.load(p)
        else:
            self._auto_discover()

        if self.matrices:
            print(f"[HDCMemory] {len(self.matrices)} arquivos .hdv carregados na {self.device}")
        else:
            print("[HDCMemory] Nenhum .hdv encontrado. Use compile_dataset_to_hdv() primeiro.")

    def _auto_discover(self):
        """Procura .hdv nos diretórios padrão."""
        for d in HDV_DIRS:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith(".hdv"):
                        self.load(os.path.join(d, f))

    def load(self, hdv_path: str):
        """Carrega um arquivo .hdv na VRAM."""
        from .coglang_compiler import HDV_MAGIC
        hdv_path = Path(hdv_path)
        if not hdv_path.exists():
            return

        with open(hdv_path, "rb") as f:
            magic = f.read(4)
            if magic != HDV_MAGIC:
                return
            num_items = int.from_bytes(f.read(4), "little")
            offset_meta = int.from_bytes(f.read(8), "little")

            f.seek(256)
            manifolds = []
            index = []
            import struct, json
            for _ in range(num_items):
                tam_data = struct.unpack("<I", f.read(4))[0]
                data = f.read(tam_data)
                tam_meta = struct.unpack("<I", f.read(4))[0]
                meta_raw = f.read(tam_meta)
                arr = __import__("numpy").frombuffer(data, dtype="float32").copy()
                manifolds.append(arr)
                index.append(json.loads(meta_raw))

            if manifolds:
                key = hdv_path.stem
                self.matrices[key] = torch.from_numpy(
                    __import__("numpy").stack(manifolds)
                ).float().to(self.device)
                self.indices[key] = index
                self.sources[key] = str(hdv_path)
                self._loaded_paths.append(str(hdv_path))
                sz_kb = self.matrices[key].element_size() * self.matrices[key].numel() / 1024
                print(f"  [HDCMemory] {hdv_path.name}: {num_items} itens, {sz_kb:.0f} KB na {self.device}")

    def query(self, text: str, top_k: int = 5, min_sim: float = 0.3) -> List[Tuple[float, Dict, str]]:
        """
        Codifica o texto como Manifold HDC e busca nos .hdv carregados.

        Estratégia dupla:
          1. Similaridade direta contra vetores armazenados (funciona para
             text_library.hdv, que armazena texto puro)
          2. Fallback: re-codifica o texto do metadado e compara no espaço HDC
             (funciona para msvd_brain.hdv, que armazena video * text)

        Returns: lista de (similaridade, metadata, source_key)
        """
        if not self.matrices:
            return []

        query_mf = self.engine.encode_text(text)
        query_vec = query_mf.tensor.to(self.device)

        results: List[Tuple[float, Dict, str]] = []
        used_ids = set()

        for key, matrix in self.matrices.items():
            index = self.indices[key]

            # Estratégia 1: similaridade direta contra vetores armazenados
            sims = torch.matmul(matrix, query_vec) / Manifold.DIMENSIONALITY
            values, indices_t = torch.topk(sims, min(top_k * 2, len(index)))

            for i, idx in enumerate(indices_t):
                sim = values[i].item()
                entry_id = index[idx].get('id', str(idx))
                if sim >= min_sim and entry_id not in used_ids:
                    results.append((sim, index[idx], key))
                    used_ids.add(entry_id)

            # Estratégia 2: re-codifica o texto do metadado e compara
            # (necessário quando o vetor armazenado é video*text, não texto puro)
            if top_k > 0 and len([r for r in results if r[2] == key]) < top_k:
                for idx, meta in enumerate(index):
                    entry_id = meta.get('id', str(idx))
                    if entry_id in used_ids:
                        continue

                    # 2a: Compara contra o ID (nome curto, ex: "alice_en", "sherlock_en")
                    meta_id = meta.get('id', '')
                    id_sim = 0.0
                    if meta_id:
                        try:
                            id_mf = self.engine.encode_text(meta_id.replace('_', ' '))
                            id_vec = id_mf.tensor.to(self.device)
                            id_sim = torch.dot(query_vec, id_vec).item() / Manifold.DIMENSIONALITY
                        except Exception:
                            pass

                    # 2b: Compara contra o texto completo
                    meta_text = meta.get('text', '')
                    text_sim = 0.0
                    if meta_text:
                        try:
                            text_mf = self.engine.encode_text(meta_text[:5000])
                            text_vec = text_mf.tensor.to(self.device)
                            text_sim = torch.dot(query_vec, text_vec).item() / Manifold.DIMENSIONALITY
                        except Exception:
                            pass

                    sim = max(id_sim, text_sim)
                    if sim >= min_sim * 0.7:
                        if entry_id not in used_ids:
                            factor = 0.90 if text_sim >= id_sim else 0.95
                            results.append((sim * factor, meta, key))
                            used_ids.add(entry_id)

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    def query_manifold(self, query_mf: Manifold, top_k: int = 5, min_sim: float = 0.3
                       ) -> List[Tuple[float, Dict, str]]:
        """Busca por similaridade a partir de um Manifold já codificado."""
        if not self.matrices:
            return []

        query_vec = query_mf.tensor.to(self.device)
        results = []
        for key, matrix in self.matrices.items():
            sims = torch.matmul(matrix, query_vec) / Manifold.DIMENSIONALITY
            values, indices = torch.topk(sims, min(top_k, len(self.indices[key])))

            for i, idx in enumerate(indices):
                sim = values[i].item()
                if sim >= min_sim:
                    results.append((sim, self.indices[key][idx], key))

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    def get_context(self, text: str, top_k: int = 3) -> str:
        """
        Retorna um resumo textual dos conceitos mais similares na memória HDV.
        Usado pelo NovaMind para enriquecer o contexto do reasoning.
        """
        results = self.query(text, top_k=top_k)
        if not results:
            return ""

        lines = []
        for sim, meta, source in results:
            label = meta.get("text", meta.get("id", ""))
            lines.append(f"[HDV:{source}] sim={sim:.3f} {label[:120]}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Estatísticas da memória HDC carregada."""
        total_items = sum(len(idx) for idx in self.indices.values())
        total_vram_kb = sum(
            m.element_size() * m.numel() / 1024 for m in self.matrices.values()
        )
        return {
            "sources": list(self.sources.keys()),
            "total_items": total_items,
            "total_vram_kb": total_vram_kb,
            "loaded_files": self._loaded_paths,
        }
