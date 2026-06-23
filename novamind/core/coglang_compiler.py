"""
CogLang Compiler — Compilador de Experiências em Fluxo Contínuo (.hdv)
======================================================================
Transforma datasets brutos (vídeos, áudios, textos, PDFs) em vetores
hiperdimensionais (.hdv) processando UM ITEM POR VEZ e descartando o
dado bruto imediatamente após a compressão.

Características:
- RAM constante (~100 MB) independente do tamanho do dataset
- GPU acelera cada item individualmente (não lote, evitando OOM)
- Saída .hdv: ~40 KB por vídeo (10.000 dimensões, float32)
- Carregamento na VRAM em segundos para consulta por similaridade

Fases:
  Fase 1 — Conhecimento: livros, artigos, documentação técnica
  Fase 2 — Entretenimento: filmes, música, jogos
  Fase 3 — Corpo: interação com o mundo real (Minecraft, robótica)
"""

import torch
import torchhd
import json
import os
import re
import time
import struct
import numpy as np
from typing import Dict, List, Optional, Generator, Tuple
from pathlib import Path

from .coglang import Manifold
from .grounding_engine import GroundingEngine


# Cabeçalho mágico para arquivos .hdv
HDV_MAGIC = b"HDV1"
HDV_DIM = Manifold.DIMENSIONALITY  # 10000


def _serialize_manifold(m: Manifold) -> bytes:
    """Serializa um Manifold em bytes binários (float32, 40KB)."""
    return m.tensor.cpu().numpy().astype(np.float32).tobytes()


def _deserialize_manifold(data: bytes, device: str = "cpu") -> Manifold:
    """Desserializa bytes em um Manifold."""
    arr = np.frombuffer(data, dtype=np.float32).copy()
    return Manifold(torch.from_numpy(arr), device=torch.device(device))


class StreamProcessor:
    """
    Processador de streaming: lê dataset, processa 1 item por vez,
    descarta o bruto, escreve .hdv incrementalmente.
    RAM nunca acumula.
    """

    def __init__(self, engine: GroundingEngine, output_dir: str):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._hdv_path: Optional[Path] = None
        self._hdv_file = None
        self._index: List[Dict] = []
        self._item_count = 0
        self._start_time = 0.0

    def open(self, name: str = "brain_memory"):
        """Abre um arquivo .hdv para escrita incremental."""
        self._hdv_path = self.output_dir / f"{name}.hdv"
        self._hdv_file = open(self._hdv_path, "wb")
        # Placeholder para cabeçalho (será atualizado no fechamento)
        self._hdv_file.write(b"\x00" * 256)
        self._index = []
        self._item_count = 0
        self._start_time = time.time()
        return self

    def write_item(self, manifold: Manifold, metadata: Dict):
        """
        Escreve UM manifold no arquivo .hdv e descarta da RAM.
        Cada item: 4 bytes (tamanho) + 40000 bytes (dados float32) + metadados.
        """
        data = _serialize_manifold(manifold)
        meta_str = json.dumps(metadata).encode("utf-8")
        # Formato: [4 bytes tam_data][data][4 bytes tam_meta][meta]
        self._hdv_file.write(struct.pack("<I", len(data)))
        self._hdv_file.write(data)
        self._hdv_file.write(struct.pack("<I", len(meta_str)))
        self._hdv_file.write(meta_str)
        self._index.append(metadata)
        self._item_count += 1

    def close(self):
        """Fecha o arquivo .hdv, escrevendo o índice final."""
        if self._hdv_file is None:
            return
        # Escreve índice no final
        index_bytes = json.dumps(self._index).encode("utf-8")
        offset_meta = self._hdv_file.tell()
        self._hdv_file.write(struct.pack("<Q", offset_meta))
        self._hdv_file.write(index_bytes)
        # Atualiza cabeçalho
        self._hdv_file.seek(0)
        header = HDV_MAGIC + struct.pack("<IQ", self._item_count, offset_meta)
        self._hdv_file.write(header + b"\x00" * (256 - len(header)))
        self._hdv_file.close()
        elapsed = time.time() - self._start_time
        rate = self._item_count / elapsed if elapsed > 0 else 0
        print(f"[CogLangCompiler] {self._hdv_path.name}: {self._item_count} itens em {elapsed:.1f}s ({rate:.1f} itens/s)")
        self._hdv_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._hdv_file is not None:
            self.close()


class CogLangCompiler:
    """
    Compilador de experiências: ingere datasets e produz .hdv.

    Uso:
        compiler = CogLangCompiler(device="cuda")
        compiler.compile_msvd("D:/novamind_data/MSVD/msvd_train.json", "D:/novamind_data/hdv/")
        compiler.compile_text_library("D:/novamind_data/text_datasets/", "D:/novamind_data/hdv/")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = GroundingEngine(device=torch.device(self.device))
        print(f"[CogLangCompiler] Device: {self.device}")

    # ─── Stream de Vídeo ─────────────────────────────────────

    def stream_video(self, video_path: str, num_frames: int = 8) -> Generator[Manifold, None, None]:
        """
        Lê vídeo frame a frame, processa e DESCARTE imediato.
        Gera UM manifold por vídeo. RAM: ~50 MB durante o processamento.
        """
        import cv2  # lazy import
        if not os.path.exists(video_path):
            return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        position_mf = [
            Manifold(torchhd.random(1, HDV_DIM, "MAP", device=torch.device(self.device)).squeeze(0),
                     device=torch.device(self.device))
            for _ in range(num_frames)
        ]

        bundled = None
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = self.engine.transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.engine.feature_extractor(img_tensor).view(-1)
                hdc_vec = torch.sign(torch.matmul(self.engine.proj_visual, features))

            mf = Manifold(hdc_vec, device=torch.device(self.device))
            pos_mf = position_mf[i % len(position_mf)]
            bound = mf * pos_mf
            bundled = bound if bundled is None else bundled + bound

            # Libera tensores intermediários
            del img_tensor, features, hdc_vec, mf

        cap.release()

        if bundled is not None:
            yield bundled

    # ─── Compilação MSVD (Fase 1: Conhecimento) ─────────────

    def compile_msvd(self, metadata_path: str, output_dir: str, max_items: int = 0):
        """
        Compila dataset MSVD (vídeos com legendas) em .hdv.
        Processa UM vídeo por vez, RAM constante.
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        items = metadata if isinstance(metadata, list) else metadata.get("videos", metadata.get("data", []))
        if max_items > 0:
            items = items[:max_items]

        base_dir = os.path.dirname(metadata_path)
        # Tenta achar diretório de vídeos
        video_dir = os.path.join(os.path.dirname(metadata_path), "YouTubeClips")
        if not os.path.isdir(video_dir):
            video_dir = os.path.join(os.path.dirname(os.path.dirname(metadata_path)), "MSVD", "YouTubeClips")
        if not os.path.isdir(video_dir):
            video_dir = base_dir

        with StreamProcessor(self.engine, output_dir).open("msvd_brain") as sp:
            for i, item in enumerate(items):
                video_id = item.get("video_id", item.get("id", f"video_{i:04d}"))
                caption_raw = item.get("caption", item.get("text", ""))
                if isinstance(caption_raw, list):
                    caption = caption_raw[0]
                else:
                    caption = caption_raw
                video_path = os.path.join(video_dir, f"{video_id}.avi")
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_dir, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    continue

                # 1. Processa vídeo (streaming, RAM constante)
                video_mf = None
                for mf in self.stream_video(video_path, num_frames=8):
                    video_mf = mf

                if video_mf is None:
                    continue

                # 2a. Texto flat (para busca semântica + reasoning)
                text_flat = self.engine.encode_text(caption)

                # 2b. Texto role-filler (para BrocaDecoder decodificar com sintaxe)
                if len(caption.split()) <= 30:
                    text_role = self.engine.encode_phrase(caption)
                else:
                    text_role = text_flat

                # 3a. Bound concept: video * text_flat (para raciocínio + busca)
                concept = video_mf * text_flat
                # Binariza para vector bipolar limpo
                binary = torch.sign(concept.tensor)
                binary[binary == 0] = 1
                concept = Manifold(binary, device=concept.device)
                sp.write_item(concept, {
                    "id": video_id,
                    "text": (caption if isinstance(caption, str) else str(caption))[:200],
                    "source": "msvd",
                    "type": "video_text",
                })

                # 3b. Texto role-filler puro (para BrocaDecoder decodificar com sintaxe)
                sp.write_item(text_role, {
                    "id": f"{video_id}_syn",
                    "text": (caption if isinstance(caption, str) else str(caption))[:200],
                    "source": "msvd",
                    "type": "syntax",
                })

                if (i + 1) % 100 == 0:
                    print(f"  [MSVD] {i+1}/{len(items)} processados... RAM livre")

    # ─── Compilação de Texto (Fase 1: Conhecimento) ─────────

    def compile_text_library(self, text_dir: str, output_dir: str):
        """
        Compila uma biblioteca de arquivos .txt inteiros em .hdv.
        Cada arquivo vira UM manifold (o conceito do livro).
        RAM: apenas o arquivo atual em memória.
        """
        text_dir = Path(text_dir)
        txt_files = sorted(text_dir.glob("*.txt")) + sorted(text_dir.glob("*.md"))

        if not txt_files:
            print(f"  Nenhum .txt/.md encontrado em {text_dir}")
            return

        with StreamProcessor(self.engine, output_dir).open("text_library") as sp:
            for txt_path in txt_files:
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(txt_path, "r", encoding="latin-1") as f:
                        text = f.read()

                if len(text.strip()) < 50:
                    continue

                # Versão flat (para busca semântica)
                mf_flat = self.engine.encode_text(text[:5000])

                # Versão role-filler (frases curtas para sintaxe)
                import re as _re2
                sentences = _re2.split(r'(?<=[.!?])\s+', text)
                role_phrases = []
                for sent in sentences[:50]:
                    if len(sent.split()) <= 30:
                        role_phrases.append(self.engine.encode_phrase(sent))
                    else:
                        role_phrases.append(self.engine.encode_text(sent[:500]))

                if not role_phrases:
                    continue

                bundled_role = role_phrases[0]
                for mf in role_phrases[1:]:
                    bundled_role = bundled_role + mf
                # Binariza para manter bipolaridade (essencial para unbinding)
                binary = torch.sign(bundled_role.tensor)
                binary[binary == 0] = 1
                bundled_role = Manifold(binary, device=bundled_role.device)

                # Item flat
                sp.write_item(mf_flat, {
                    "id": txt_path.stem,
                    "text": text[:1000].replace("\n", " "),
                    "source": "text_library",
                    "type": "book",
                    "size_bytes": len(text),
                })

                # Item syntax
                sp.write_item(bundled_role, {
                    "id": f"{txt_path.stem}_syn",
                    "text": text[:1000].replace("\n", " "),
                    "source": "text_library",
                    "type": "syntax",
                })

                print(f"  [TEXT] {txt_path.name} comprimido ({len(role_phrases)} frases)")

    # ─── Leitura e Consulta de .hdv ──────────────────────────

    def load_hdv(self, hdv_path: str, device: str = "cuda") -> Tuple[torch.Tensor, List[Dict]]:
        """
        Carrega um arquivo .hdv na VRAM para consulta por similaridade.

        Returns:
            matrix: Tensor [N, 10000] com todos os manifolds
            index: lista de metadados
        """
        hdv_path = Path(hdv_path)
        if not hdv_path.exists():
            raise FileNotFoundError(f".hdv não encontrado: {hdv_path}")

        with open(hdv_path, "rb") as f:
            # Lê cabeçalho
            magic = f.read(4)
            if magic != HDV_MAGIC:
                raise ValueError(f"Arquivo .hdv inválido: {magic}")
            num_items = struct.unpack("<I", f.read(4))[0]
            offset_meta = struct.unpack("<Q", f.read(8))[0]

            # Lê todos os manifolds
            f.seek(256)
            manifolds = []
            index = []
            for _ in range(num_items):
                tam_data = struct.unpack("<I", f.read(4))[0]
                data = f.read(tam_data)
                tam_meta = struct.unpack("<I", f.read(4))[0]
                meta_raw = f.read(tam_meta)
                manifolds.append(_deserialize_manifold(data, device="cpu"))
                index.append(json.loads(meta_raw))

            matrix = torch.stack([m.tensor for m in manifolds]).to(device)
            print(f"[HDV] {hdv_path.name}: {num_items} itens carregados na {device} ({matrix.element_size() * matrix.numel() / 1024:.0f} KB)")

        return matrix, index

    def query(self, query_manifold: Manifold, matrix: torch.Tensor, index: List[Dict],
              top_k: int = 5, min_sim: float = 0.5) -> List[Tuple[float, Dict]]:
        """
        Busca os top_k itens mais similares na memória .hdv.

        Returns:
            lista de (similaridade, metadata) ordenada decrescente
        """
        query_vec = query_manifold.tensor.to(matrix.device)
        sims = torch.matmul(matrix, query_vec) / HDV_DIM  # cosseno bipolar
        values, indices = torch.topk(sims, min(top_k, len(index)))

        results = []
        for i, idx in enumerate(indices):
            sim = values[i].item()
            if sim >= min_sim:
                results.append((sim, index[idx]))

        return sorted(results, key=lambda x: x[0], reverse=True)


def ingest_all(dataset_root: str = "D:/novamind_data", output_root: str = "D:/novamind_data/hdv"):
    """
    Pipeline completo de ingestão: compila todos os datasets disponíveis.
    """
    compiler = CogLangCompiler()

    # Fase 1: Conhecimento — Texto
    text_dir = os.path.join(dataset_root, "text_datasets")
    if os.path.isdir(text_dir):
        print("\n=== Compilando Biblioteca de Texto ===")
        compiler.compile_text_library(text_dir, output_root)

    # Fase 1: Conhecimento — MSVD (vídeo + texto)
    msvd_json = os.path.join(dataset_root, "MSVD", "msvd_train.json")
    if os.path.exists(msvd_json):
        print("\n=== Compilando MSVD (vídeo+texto) ===")
        compiler.compile_msvd(msvd_json, output_root, max_items=1000)

    print(f"\nIngestão concluída. Arquivos .hdv em: {output_root}")
    print("Use load_hdv() para carregar na GPU e query() para buscar.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest-all":
        ingest_all()
    else:
        print("Uso: python -m core.coglang_compiler --ingest-all")
