"""
Mega-Brain Compiler — Pipeline de Compilação em Streaming para HDV Único
===========================================================================
Lê volumes massivos de texto (Wikipedia, arXiv, GitHub, livros) em chunks
de 1GB sem estourar RAM, extrai conceitos semânticos, projeta no hiperespaço,
e salva TUDO num ÚNICO mega_brain.hdv com modulação por contexto.

Arquitetura:
  1. STREAM: lê N linhas por vez, processa, descarta (memória O(1))
  2. ENCODE: cada parágrafo/conceito → spaCy 300D → JL projeção → HDC 10000D
  3. CONTEXT: EmergentContextBinder categoriza o domínio AUTOMATICAMENTE
  4. SAVE: no HDV1 com metadados completos (texto original + domínio + contexto)
  5. UNIFIED: TUDO no mesmo arquivo — sem separação por domínio

Uso:
    from core.mega_brain_compiler import MegaBrainCompiler
    compiler = MegaBrainCompiler()
    compiler.compile_text_file("D:/novamind_data/simplewiki_clean.txt",
                                label="wikipedia", max_chunks=500)
    compiler.compile_directory("D:/novamind_data/books/",
                                label="books")
    compiler.finalize()  # escreve o HDV único

Formato de saída: D:/novamind_data/hdv/mega_brain.hdv (HDV1 padrão)
"""

import os
import sys
import json
import time
import math
import struct
import hashlib
import threading
from typing import Optional, List, Dict, Generator, Tuple
from pathlib import Path

import torch
import numpy as np

_NOVAMIND = str(Path(__file__).parent.parent)
if _NOVAMIND not in sys.path:
    sys.path.insert(0, _NOVAMIND)

from core.grounding_engine import GroundingEngine
from core.emergent_context import EmergentContextBinder
from core.coglang import Manifold

# ─── Constantes ──────────────────────────────────────────────

HDV_MAGIC = b"HDV2"  # HDV2 = bit-packed bipolar (±1 em 1 bit/dim), 32x menor que HDV1 float32
HDC_DIM = 10000
CHUNK_LINES = 500          # linhas por chunk de processamento
MAX_PARAGRAPH_CHARS = 2000 # tamanho máximo de parágrafo
PARAGRAPH_MIN_CHARS = 80   # parágrafo mínimo para ser conceito
OUTPUT_PATH = "D:/novamind_data/hdv/mega_brain.hdv"


class MegaBrainCompiler:
    """
    Compilador que constrói um ÚNICO mega_brain.hdv a partir de
    múltiplas fontes de texto, usando streaming para não estourar RAM.

    Fluxo:
      1. Para cada chunk de texto:
         a. Extrai parágrafos como conceitos
         b. Codifica cada um como HDC (spaCy → JL → bipolar)
         c. EmergentContextBinder categoriza o domínio automaticamente
         d. Acumula em buffer
      2. Quando buffer atinge N itens, despeja no HDV via append
      3. No final, fecha o arquivo e salva índice de metadados
    """

    def __init__(self, output_path: str = OUTPUT_PATH,
                 device: str = "cuda",
                 context_binder_path: str = "D:/temp/contexts.json"):
        self.output_path = output_path
        self.device = torch.device(device)
        self.context_path = context_binder_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Motores
        self.engine = GroundingEngine()
        self.context_binder = EmergentContextBinder(
            engine=self.engine, path=context_binder_path,
            device=device)

        # Buffer de escritas
        self._item_buffer: List[bytes] = []  # serializados
        self._meta_buffer: List[Dict] = []
        self._total_items = 0
        self._file_handle = None
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Estatísticas
        self.stats = {
            "total_lines_read": 0,
            "total_chunks_processed": 0,
            "total_concepts_saved": 0,
            "sources": {},
            "domains_created": 0,
            "bytes_written": 0,
        }

        print(f"[MEGA-BRAIN] Compilador iniciado")
        print(f"[MEGA-BRAIN] Saída: {output_path}")
        print(f"[MEGA-BRAIN] Device: {device}")

    def compile_text_file(self, file_path: str, label: str = "text",
                          max_chunks: int = 0, skip_existing: bool = True):
        """
        Compila um arquivo de texto (qualquer tamanho) em chunks.

        Lê em streaming: só mantém CHUNK_LINES linhas na RAM por vez.
        Processa 1GB sem usar mais que ~200MB de RAM.

        Args:
            file_path: caminho do .txt
            label: identificador da fonte (ex: "wikipedia", "arxiv")
            max_chunks: máximo de chunks (0 = ilimitado)
            skip_existing: se True, verifica se arquivo já foi compilado
        """
        if not os.path.exists(file_path):
            print(f"[MEGA-BRAIN] Arquivo não encontrado: {file_path}")
            return

        if skip_existing and self._is_already_compiled(file_path):
            print(f"[MEGA-BRAIN] Pulando (já compilado): {file_path}")
            return

        print(f"\n[MEGA-BRAIN] Compilando: {file_path} (label={label})")
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"[MEGA-BRAIN] Tamanho: {file_size_gb:.2f} GB")
        t0 = time.time()

        # Abre para append
        self._open_for_append()

        chunk_count = 0
        line_buffer = []
        lines_read = 0

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_buffer.append(line)
                lines_read += 1

                if len(line_buffer) >= CHUNK_LINES:
                    chunk_count += 1
                    self._process_chunk(line_buffer, label)
                    line_buffer = []

                    if max_chunks > 0 and chunk_count >= max_chunks:
                        break

        # Processa linhas restantes
        if line_buffer:
            chunk_count += 1
            self._process_chunk(line_buffer, label)

        # Marca como compilado
        self._mark_compiled(file_path)
        self._flush_buffer()

        elapsed = time.time() - t0
        rate = self.stats["total_concepts_saved"] / max(elapsed, 0.1)
        print(f"[MEGA-BRAIN] Concluído: {lines_read} linhas, "
              f"{self.stats['total_concepts_saved']} conceitos "
              f"em {elapsed:.0f}s ({rate:.1f} conc/s)")

    def compile_directory(self, dir_path: str, label: str = "books",
                          pattern: str = "*.txt", max_chunks: int = 0):
        """Compila todos os .txt de um diretório."""
        if not os.path.isdir(dir_path):
            print(f"[MEGA-BRAIN] Diretório não encontrado: {dir_path}")
            return

        files = sorted(Path(dir_path).glob(pattern))
        print(f"[MEGA-BRAIN] Compilando diretório: {dir_path} ({len(files)} arquivos)")

        for fpath in files:
            self.compile_text_file(str(fpath), label=f"{label}_{fpath.stem}",
                                   max_chunks=max_chunks)

    def finalize(self):
        """Finaliza o HDV: escreve índice de metadados e fecha."""
        self._flush_buffer()
        self._write_footer()
        self._close_file()

        elapsed = time.time() - self._start_time
        print(f"\n[MEGA-BRAIN] ===== COMPILAÇÃO CONCLUÍDA =====")
        print(f"[MEGA-BRAIN] Total: {self._total_items} conceitos")
        print(f"[MEGA-BRAIN] Tempo: {elapsed:.0f}s")
        print(f"[MEGA-BRAIN] Domínios criados: {len(self.context_binder.contexts)}")
        print(f"[MEGA-BRAIN] Tamanho: {self.stats['bytes_written'] / (1024**3):.2f} GB")
        print(f"[MEGA-BRAIN] Arquivo: {self.output_path}")
        self._print_domains()

    # ─── Pipeline Interno ───────────────────────────────────

    def _process_chunk(self, lines: List[str], source_label: str):
        """Processa um chunk de linhas: extrai parágrafos, codifica em lote, salva."""
        paragraphs = self._extract_paragraphs(lines)

        # Filtra e trunca
        valid = []
        for p in paragraphs:
            if len(p) < PARAGRAPH_MIN_CHARS:
                continue
            if len(p) > MAX_PARAGRAPH_CHARS:
                p = p[:MAX_PARAGRAPH_CHARS]
            if self._is_junk_paragraph(p):
                continue
            valid.append(p)

        if not valid:
            self.stats["total_chunks_processed"] += 1
            self.stats["total_lines_read"] += len(lines)
            return

        # Batch encode (muito mais rápido que individual)
        manifolds = self.engine.encode_texts_batch(valid)

        for para, mf in zip(valid, manifolds):
            self._encode_and_save(para, mf.tensor, source_label)

        self.stats["total_chunks_processed"] += 1
        self.stats["total_lines_read"] += len(lines)

    def _encode_and_save(self, text: str, pure_vec: torch.Tensor, source_label: str):
        """
        Salva um parágrafo já codificado no buffer de escrita.
        """

        # Categoriza com contexto emergente
        ctx_name, ctx_vec = self.context_binder.find_or_create(pure_vec, label=source_label)
        bound_vec = torch.sign(pure_vec * ctx_vec)

        # Prepara metadados
        meta = {
            "id": f"{source_label}_{self._total_items}",
            "text": text[:500],  # trunca para o índice (texto completo pesa)
            "source": source_label,
            "domain": ctx_name,
            "length": len(text),
            "compiled_at": time.time(),
        }

        # Serializa
        item_data = self._serialize_item(pure_vec, meta)
        self._item_buffer.append(item_data)
        self._meta_buffer.append(meta)
        self._total_items += 1
        self.stats["total_concepts_saved"] += 1

        # Atualiza estatísticas da fonte
        if source_label not in self.stats["sources"]:
            self.stats["sources"][source_label] = 0
        self.stats["sources"][source_label] += 1

        # Buffer flush a cada 100 itens
        if len(self._item_buffer) >= 100:
            self._flush_buffer()

        # Progresso a cada 1000 itens
        if self._total_items % 1000 == 0:
            elapsed = time.time() - self._start_time
            rate = self._total_items / max(elapsed, 0.1)
            print(f"[MEGA-BRAIN] {self._total_items} conceitos "
                  f"(ctx={len(self.context_binder.contexts)}, "
                  f"{rate:.0f}/s)")

    def _serialize_item(self, vec: torch.Tensor,
                        metadata: dict) -> bytes:
        """Serializa um item no formato HDV2 (bit-packed).

        Formato: [4B data_len][1250B bit-packed vec][4B meta_len][meta_bytes]
        Cada vetor bipolar 10000D ocupa 1250 bytes (1 bit/dim) em vez de 40000.
        """
        from .hdc_utils import pack_bipolar
        vec_bytes = pack_bipolar(vec)
        meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

        data_len = len(vec_bytes)
        meta_len = len(meta_bytes)

        header = struct.pack("<I", data_len)
        trailer = struct.pack("<I", meta_len)
        return header + vec_bytes + trailer + meta_bytes

    def _flush_buffer(self):
        """Escreve o buffer no arquivo."""
        if not self._item_buffer:
            return

        with self._lock:
            if self._file_handle is None:
                return

            for item_data in self._item_buffer:
                self._file_handle.write(item_data)
                self.stats["bytes_written"] += len(item_data)

        n_flushed = len(self._item_buffer)
        self._item_buffer = []
        # Mantém no máximo 1000 metadados no buffer do footer
        if len(self._meta_buffer) > 1000:
            self._meta_buffer = self._meta_buffer[-1000:]

    # ─── Gerenciamento de Arquivo ───────────────────────────

    def _open_for_append(self):
        """Abre o HDV para append. Se arquivo novo, escreve header."""
        if self._file_handle is not None:
            return

        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0

        if file_exists:
            self._file_handle = open(self.output_path, "ab")
            print(f"[MEGA-BRAIN] Abrindo existente para append: {self.output_path}")
        else:
            self._file_handle = open(self.output_path, "wb")
            # Header placeholder (256 bytes)
            header = HDV_MAGIC + b"\x00" * 252
            self._file_handle.write(header)
            print(f"[MEGA-BRAIN] Criando novo HDV: {self.output_path}")

    def _write_footer(self):
        """Escreve o footer com o índice completo de metadados."""
        if self._file_handle is None:
            return

        with self._lock:
            # Posição atual (início do footer)
            footer_pos = self._file_handle.tell()

            # Volta e atualiza header
            self._file_handle.seek(0)
            header = (HDV_MAGIC
                      + np.uint32(self._total_items).tobytes()
                      + np.uint64(footer_pos).tobytes()
                      + b"\x00" * 240)
            self._file_handle.write(header)

            # Volta para o footer e escreve índice
            self._file_handle.seek(footer_pos)
            footer_data = (np.uint64(footer_pos).tobytes()
                           + json.dumps(self._meta_buffer, ensure_ascii=False).encode("utf-8"))
            self._file_handle.write(footer_data)
            self.stats["bytes_written"] += len(footer_data)

    def _close_file(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    # ─── Utilitários ────────────────────────────────────────

    def _extract_paragraphs(self, lines: List[str]) -> List[str]:
        """
        Extrai parágrafos de um chunk de linhas.
        
        Suporta dois formatos:
        1. Um parágrafo por linha (ex: simplewiki_clean.txt)
        2. Parágrafos separados por linhas em branco (ex: livros .txt)
        """
        joined = "".join(lines)
        # Tenta split duplo primeiro (formato com linhas em branco)
        double_split = [p.strip() for p in joined.split("\n\n") if p.strip()]
        if len(double_split) >= 2:
            return double_split
        # Fallback: cada linha não-vazia é um parágrafo
        return [l.strip() for l in lines if l.strip()
                and len(l.strip()) > 20]

    def _is_junk_paragraph(self, text: str) -> bool:
        """Detecta se um parágrafo é lixo (não tem conteúdo semântico)."""
        if len(text) < 20:
            return True
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.3:
            return True
        return False

    def _is_already_compiled(self, file_path: str) -> bool:
        """Verifica se arquivo já foi compilado (por hash)."""
        marker_path = os.path.join(os.path.dirname(self.output_path),
                                   ".compiled_markers.json")
        if not os.path.exists(marker_path):
            return False
        try:
            with open(marker_path) as f:
                markers = json.load(f)
            file_hash = self._file_hash(file_path)
            return markers.get(file_path) == file_hash
        except Exception:
            return False

    def _mark_compiled(self, file_path: str):
        """Marca arquivo como compilado."""
        marker_path = os.path.join(os.path.dirname(self.output_path),
                                   ".compiled_markers.json")
        try:
            markers = {}
            if os.path.exists(marker_path):
                with open(marker_path) as f:
                    markers = json.load(f)
            markers[file_path] = self._file_hash(file_path)
            with open(marker_path, "w") as f:
                json.dump(markers, f, indent=2)
        except Exception:
            pass

    def _file_hash(self, file_path: str) -> str:
        """Hash rápido do arquivo (primeiros 1MB + tamanho)."""
        try:
            size = os.path.getsize(file_path)
            h = hashlib.md5()
            with open(file_path, "rb") as f:
                h.update(f.read(1024 * 1024))  # primeiro 1MB
            h.update(str(size).encode())
            return h.hexdigest()[:16]
        except Exception:
            return ""

    def _print_domains(self):
        """Mostra domínios que emergiram durante a compilação."""
        print(f"\n[MEGA-BRAIN] Domínios emergentes:")
        for name, meta in sorted(self.context_binder.context_metadata.items(),
                                 key=lambda x: x[1].get("n_concepts", 0), reverse=True):
            print(f"  {name:30s} {meta.get('n_concepts', 0):6d} conceitos "
                  f"(seed: {meta.get('seed_label', '')[:20]})")

    def get_summary(self) -> dict:
        return {
            "output": self.output_path,
            "total_concepts": self._total_items,
            "domains": len(self.context_binder.contexts),
            "sources": self.stats["sources"],
            "bytes_written": self.stats["bytes_written"],
            "elapsed": time.time() - self._start_time,
        }


def quick_compile():
    """
    Compilação rápida de arquivos existentes para testar o pipeline.

    Uso:
        python -c "from core.mega_brain_compiler import quick_compile; quick_compile()"
    """
    compiler = MegaBrainCompiler()

    # Arquivos para compilar (existentes no sistema)
    candidates = [
        ("D:/novamind_data/temp/simplewiki_clean.txt", "wikipedia"),
        ("D:/novamind_data/temp/simplewiki.txt", "wikipedia_raw"),
    ]

    for fpath, label in candidates:
        if os.path.exists(fpath):
            compiler.compile_text_file(fpath, label=label, max_chunks=0)
        else:
            print(f"[MEGA-BRAIN] Pulando (não encontrado): {fpath}")

    compiler.finalize()
    return compiler


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Modo CLI: python mega_brain_compiler.py <arquivo.txt> [label]
        compiler = MegaBrainCompiler()
        compiler.compile_text_file(sys.argv[1],
                                   label=sys.argv[2] if len(sys.argv) > 2 else "cli")
        compiler.finalize()
    else:
        quick_compile()
