"""
Compilador de Wikipedia Simple → mega_brain.hdv
-----------------------------------------------
Baixa o dump XML da Wikipedia Simple em inglês,
extrai frases, codifica cada uma como vetor HDC de 10.000 dim,
e salva num unico arquivo .hdv para carregamento na VRAM.

Uso:
    python scripts/compile_wikipedia_hdv.py
    # ou em background:
    python scripts/compile_wikipedia_hdv.py --batch 5000
"""

import sys, os, re, json, time, struct, math, urllib.request, bz2, xml.parsers.expat
from pathlib import Path
from typing import Optional, List, Generator

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.grounding_engine import GroundingEngine
from core.coglang import Manifold

HDV_MAGIC = b"HDV1"
HDV_DIM = 10000
OUTPUT_DIR = Path("D:/novamind_data/hdv")
DUMP_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DUMP_PATH = Path("D:/novamind_data/temp/simplewiki.xml.bz2")
EXTRACTED_PATH = Path("D:/novamind_data/temp/simplewiki.txt")

SENTENCE_SPLITTER = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def download_dump():
    """Baixa o dump da Wikipedia Simple se não existir."""
    if DUMP_PATH.exists():
        print(f"[Download] Já existe: {DUMP_PATH} ({DUMP_PATH.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"[Download] Baixando {DUMP_URL}...")
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DUMP_URL, DUMP_PATH)
    mb = DUMP_PATH.stat().st_size / 1e6
    print(f"[Download] OK: {mb:.0f} MB")


def extract_text_from_dump():
    """Extrai texto puro do XML dump da Wikipedia."""
    if EXTRACTED_PATH.exists():
        print(f"[Extract] Já existe: {EXTRACTED_PATH}")
        return

    print("[Extract] Extraindo texto do XML...")
    EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Regex para limpar wiki markup
    wiki_cleaners = [
        (re.compile(r"'''|'''"), ''),
        (re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"), r'\1'),
        (re.compile(r"<[^>]+>"), ''),
        (re.compile(r"\[https?://[^\]]+\]"), ''),
        (re.compile(r"'{2,}"), ''),
        (re.compile(r"={2,}"), ''),
        (re.compile(r"^\*"), ''),
        (re.compile(r"^\#"), ''),
        (re.compile(r"\n\n+"), '\n'),
    ]

    in_text = False
    in_page = False
    text_content = []
    ns_value = None
    title = ""

    def start_element(name, attrs):
        nonlocal in_text, in_page, ns_value
        if name == 'page':
            in_page = True
            ns_value = None
        elif name == 'text' and in_page:
            in_text = True

    def end_element(name):
        nonlocal in_text, in_page
        if name == 'text':
            in_text = False
        elif name == 'page':
            in_page = False

    def char_data(data):
        nonlocal ns_value, title
        # Track ns and title
        pass

    # Streaming parse - não carrega XML inteiro na RAM
    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    parser.CharacterDataHandler = char_data

    current_tag = ''
    current_data = []
    in_revision_text = False
    page_count = 0
    sentence_count = 0

    with bz2.open(DUMP_PATH, 'rb') as f_in, open(EXTRACTED_PATH, 'w', encoding='utf-8') as f_out:
        # Processa em blocos de 1MB
        buffer = ""
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk.decode('utf-8', errors='replace')

            # Extrai tags <text>...</text> com regex simples
            texts = re.findall(r'<text[^>]*>(.*?)</text>', buffer, re.DOTALL)
            buffer = ""  # Não precisa manter buffer pois buscamos tags completas

            for text_xml in texts:
                # Remove comentários e templates
                text = re.sub(r'<!--.*?-->', '', text_xml, flags=re.DOTALL)
                text = re.sub(r'\{\{[^}]*\}\}', '', text)

                # Aplica limpadores
                for pattern, repl in wiki_cleaners:
                    text = pattern.sub(repl, text)

                # Remove linhas vazias e não-texto
                lines = []
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('|') or line.startswith('!'):
                        continue
                    if len(line) < 20:
                        continue
                    lines.append(line)

                text_clean = ' '.join(lines)
                if len(text_clean) < 20:
                    continue

                f_out.write(text_clean + '\n')
                page_count += 1

                if page_count % 1000 == 0:
                    print(f"  {page_count} páginas extraídas...", end='\r')

        print(f"\n[Extract] {page_count} páginas extraídas")


def split_sentences() -> List[str]:
    """Divide o texto extraído em frases individuais."""
    if not EXTRACTED_PATH.exists():
        return []

    print("[Split] Dividindo em frases...")
    sentences = []
    with open(EXTRACTED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Divide a linha em frases
            parts = SENTENCE_SPLITTER.split(line)
            for part in parts:
                part = part.strip()
                if 10 < len(part) < 500:
                    sentences.append(part)

    print(f"[Split] {len(sentences)} frases extraídas")
    return sentences


def encode_sentences(engine: GroundingEngine, sentences: List[str],
                     batch_size: int = 500) -> Generator:
    """Codifica frases em lote e gera (vetor, texto)."""
    total = len(sentences)
    print(f"[Encode] Codificando {total} frases em lotes de {batch_size}...")

    for start in range(0, total, batch_size):
        batch = sentences[start:start + batch_size]
        batch_vectors = []
        batch_texts = []

        for text in batch:
            try:
                mf = engine.encode_text(text)
                vec = mf.tensor.cpu().numpy().astype(np.float32)
                batch_vectors.append(vec)
                batch_texts.append(text)
            except Exception as e:
                continue

        if batch_vectors:
            yield batch_texts, np.stack(batch_vectors)

        progress = min(start + batch_size, total)
        print(f"  {progress}/{total} ({100*progress/total:.0f}%)", end='\r')


def save_hdv(sentences: List[str], output_name: str = "mega_brain",
             batch_size: int = 500):
    """Codifica e salva todas as frases em .hdv."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[HDV] Device: {device}")
    engine = GroundingEngine(device=torch.device(device))

    output_path = OUTPUT_DIR / f"{output_name}.hdv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Buffer para escrita
    item_count = 0
    total_sentences = len(sentences)

    with open(output_path, 'wb') as f:
        # Placeholder para header
        f.write(b'\x00' * 256)

        for batch_texts, batch_vectors in encode_sentences(engine, sentences, batch_size):
            for i, vec in enumerate(batch_vectors):
                text = batch_texts[i]
                meta = json.dumps({"id": f"wiki_{item_count}", "text": text}).encode('utf-8')
                data = vec.tobytes()

                f.write(struct.pack("<I", len(data)))
                f.write(data)
                f.write(struct.pack("<I", len(meta)))
                f.write(meta)
                item_count += 1

        # Escreve índice no final
        offset_meta = f.tell()
        f.write(struct.pack("<Q", offset_meta))

        # Atualiza header
        header = HDV_MAGIC + struct.pack("<IQ", item_count, offset_meta)
        f.seek(0)
        f.write(header + b'\x00' * (256 - len(header)))

    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[HDV] {output_name}.hdv: {item_count} itens, {mb:.1f} MB")
    return item_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compila Wikipedia Simple para HDV")
    parser.add_argument('--batch', type=int, default=500, help='Tamanho do lote de codificação')
    parser.add_argument('--max', type=int, default=0, help='Máximo de frases (0 = todas)')
    parser.add_argument('--skip-download', action='store_true', help='Pular download')
    parser.add_argument('--skip-extract', action='store_true', help='Pular extração')
    args = parser.parse_args()

    t0 = time.time()

    if not args.skip_download:
        download_dump()

    if not args.skip_extract:
        extract_text_from_dump()

    sentences = split_sentences()
    if args.max > 0:
        sentences = sentences[:args.max]
        print(f"[Main] Limitado a {args.max} frases")

    if not sentences:
        print("[Main] Nenhuma frase para codificar!")
        return

    n = save_hdv(sentences, batch_size=args.batch)
    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"[Main] Concluído: {n} conceitos em {elapsed:.0f}s ({rate:.1f} conceitos/s)")


if __name__ == '__main__':
    main()
