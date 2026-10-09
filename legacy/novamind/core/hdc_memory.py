"""HDC Memory — query sobre HDVs bit-packed sem estourar VRAM."""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .coglang import Manifold
from .grounding_engine import GroundingEngine

HDV_DIRS = [r"D:\novamind_data\hdv", r"D:\disk_d_brain"]
QUERY_CHUNK = 10000


class HDCMemory:
    """Memória HDC: carrega HDVs bit-packed em CPU, consulta em chunks."""

    def __init__(self, device: Optional[str] = None, hdv_paths: Optional[List[str]] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = GroundingEngine(device=torch.device(self.device))
        self._packed: Dict[str, torch.Tensor] = {}
        self._indices: Dict[str, List[Dict]] = {}
        self._sources: Dict[str, str] = {}
        self._loaded_paths: List[str] = []

        if hdv_paths:
            for p in hdv_paths:
                self.load(p)
        else:
            self._auto_discover()

        if self._packed:
            total_items = sum(len(v) for v in self._indices.values())
            mem_mb = sum(v.element_size() * v.numel() for v in self._packed.values()) / 1e6
            print(f"[HDCMemory] {len(self._packed)} arquivos, {total_items} itens, {mem_mb:.0f} MB RAM")
        else:
            print("[HDCMemory] Nenhum .hdv encontrado.")

    def _auto_discover(self):
        for d in HDV_DIRS:
            if os.path.isdir(d):
                all_hdvs = sorted(f for f in os.listdir(d) if f.endswith(".hdv"))
                prioritized = [f for f in all_hdvs if "mega_brain" in f] + \
                              [f for f in all_hdvs if "mega_brain" not in f]
                prioritized = [f for f in prioritized if "_v2." not in f and ".bak" not in f]
                for f in prioritized:
                    self.load(os.path.join(d, f))

    def load(self, hdv_path: str):
        """Carrega HDV como uint8 bit-packed em CPU. Suporta HDV1 e HDV2."""
        hdv_path = Path(hdv_path)
        if not hdv_path.exists():
            return
        try:
            with open(hdv_path, "rb") as f:
                magic = f.read(4)
                if magic not in (b"HDV1", b"HDV2"):
                    return
                is_bitpacked = (magic == b"HDV2")
                num_items = int.from_bytes(f.read(4), "little")
                f.read(8)  # offset_meta
                import struct, json
                from .hdc_utils import unpack_bipolar

                f.seek(256)
                packed_list = []
                indices = []
                for _ in range(num_items):
                    tam_data = struct.unpack("<I", f.read(4))[0]
                    data = f.read(tam_data)
                    tam_meta = struct.unpack("<I", f.read(4))[0]
                    meta_raw = f.read(tam_meta)
                    if is_bitpacked:
                        packed_list.append(np.frombuffer(data, dtype=np.uint8))
                    else:
                        arr = np.frombuffer(data, dtype=np.float32).copy()
                        vec = torch.from_numpy(arr)
                        from .hdc_utils import pack_bipolar
                        packed_list.append(np.frombuffer(pack_bipolar(vec), dtype=np.uint8))
                    indices.append(json.loads(meta_raw.decode("utf-8", errors="replace")))
        except Exception as e:
            print(f"  [HDCMemory] Erro carregando {hdv_path.name}: {e}")
            return

        if packed_list:
            key = hdv_path.stem
            self._packed[key] = torch.from_numpy(np.stack(packed_list)).to("cpu")
            self._indices[key] = indices
            self._sources[key] = str(hdv_path)
            self._loaded_paths.append(str(hdv_path))
            mem_mb = self._packed[key].element_size() * self._packed[key].numel() / 1e6
            print(f"  [HDCMemory] {hdv_path.name}: {num_items} itens, {mem_mb:.0f} MB CPU")

    def query(self, text: str, top_k: int = 5, min_sim: float = 0.3) -> List[Tuple[float, Dict, str]]:
        if not self._packed:
            return []
        query_mf = self.engine.encode_text(text)
        return self.query_manifold(query_mf, top_k=top_k, min_sim=min_sim)

    def query_manifold(self, query_mf: Manifold, top_k: int = 5, min_sim: float = 0.3
                       ) -> List[Tuple[float, Dict, str]]:
        return self.query_vec(query_mf.tensor, top_k=top_k, min_sim=min_sim)

    # Tabela de popcount lookup (256 entradas, conta bits setados em cada byte)
    _POPCOUNT = torch.zeros(256, dtype=torch.uint8)
    for _i in range(256):
        _POPCOUNT[_i] = bin(_i).count("1")

    def query_vec(self, vec: torch.Tensor, top_k: int = 5, min_sim: float = 0.3,
                  include_vecs: bool = True) -> List[Tuple[float, Dict, str]]:
        if not self._packed:
            return []
        from .hdc_utils import pack_bipolar
        qflat = vec.flatten().float().cpu()
        q_packed = torch.frombuffer(bytearray(pack_bipolar(qflat)), dtype=torch.uint8)

        results = []
        for key, packed_mat in self._packed.items():
            index = self._indices[key]
            n_items = len(index)
            chunk_sims = []

            for start in range(0, n_items, QUERY_CHUNK):
                end = min(start + QUERY_CHUNK, n_items)
                chunk = packed_mat[start:end]  # (chunk, 1250) uint8
                xor_all = torch.bitwise_xor(chunk, q_packed.unsqueeze(0))  # (chunk, 1250)
                mismatches = self._POPCOUNT[xor_all.long()].sum(dim=1)
                sims = 1.0 - 2.0 * mismatches.float() / 10000.0
                chunk_sims.append(sims)

            all_sims = torch.cat(chunk_sims)
            top_k_local = min(top_k * 2, n_items)
            top_values, top_indices = torch.topk(all_sims, top_k_local)

            for i, idx in enumerate(top_indices):
                sim = top_values[i].item()
                if sim >= min_sim:
                    meta = dict(index[idx])
                    if include_vecs:
                        meta["_vec"] = qflat
                    results.append((sim, meta, key))

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    def get_context(self, text: str, top_k: int = 3) -> str:
        results = self.query(text, top_k=top_k)
        if not results:
            return ""
        lines = []
        for sim, meta, source in results:
            label = meta.get("text", meta.get("id", ""))
            lines.append(f"[HDV:{source}] sim={sim:.3f} {label[:120]}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        total_items = sum(len(idx) for idx in self._indices.values())
        total_mem = sum(
            m.element_size() * m.numel() / 1024 for m in self._packed.values()
        )
        return {
            "sources": list(self._sources.keys()),
            "total_items": total_items,
            "total_ram_kb": total_mem,
            "loaded_files": self._loaded_paths,
        }
