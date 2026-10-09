"""Compilador Wikipedia v2 - usa simplewiki.txt já extraído.
Processa em lotes GPU e salva HDV com checkpointing."""

import sys, os, json, struct, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.grounding_engine import GroundingEngine
import core.grounding_engine as ge_module

HDV_MAGIC = b"HDV1"
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
INPUT_PATH = Path("D:/novamind_data/temp/simplewiki.txt")
OUTPUT_PATH = Path("D:/novamind_data/hdv/mega_brain.hdv")
MAX_CONCEPTS = 50000
BATCH_SIZE = 2000
CHECKPOINT_EVERY = 10000


def extract_sentences(text: str) -> list:
    text = re.sub(r'\[\[[^\]]+\]\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if 10 < len(s.strip()) < 300]


def write_hdv_header(f, count, file_end):
    f.seek(0)
    f.write(HDV_MAGIC)
    f.write(struct.pack("<I", count))
    f.write(struct.pack("<Q", file_end))
    f.write(b"\x00" * (256 - 16))
    f.seek(file_end)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Compiler] Device: {device}")
    engine = GroundingEngine(device=device)

    print(f"[Compiler] Lendo {INPUT_PATH}...")
    all_sentences = []
    with open(INPUT_PATH, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sents = extract_sentences(line)
            all_sentences.extend(sents)
            if len(all_sentences) >= MAX_CONCEPTS:
                all_sentences = all_sentences[:MAX_CONCEPTS]
                break
            if i > 0 and i % 10000 == 0:
                print(f"  {i} lines, {len(all_sentences)} sentencas...")

    print(f"[Compiler] {len(all_sentences)} sentencas extraidas")
    if not all_sentences:
        print("[Compiler] Nenhuma sentenca!")
        return

    # Pre-encode ALL sentences via spaCy batch pipeline (10x faster than sequential)
    print(f"[Compiler] Pre-encoding {len(all_sentences)} sentences via spaCy batch...")
    t0 = time.time()
    semantic_vecs = []

    from core.grounding_engine import _nlp as spacy_nlp
    if spacy_nlp is None:
        print("[Compiler] ERRO: spaCy nao carregado!")
        return

    spacy_batch = 2000
    for bstart in range(0, len(all_sentences), spacy_batch):
        batch = all_sentences[bstart:bstart + spacy_batch]
        docs = list(spacy_nlp.pipe(batch, n_process=1))
        for doc in docs:
            semantic_vecs.append(doc.vector.astype(np.float32))
        t1 = time.time()
        print(f"  spaCy: {len(semantic_vecs)}/{len(all_sentences)} ({t1-t0:.0f}s)")

    print(f"[Compiler] spaCy OK: {len(semantic_vecs)} vetores em {time.time()-t0:.0f}s")

    # GPU batch projection: (N,300) @ (300,10000) -> (N,10000)
    if ge_module._PROJ_SEMANTIC is None:
        ge_module._PROJ_SEMANTIC = torch.randn(
            ge_module.SEMANTIC_DIM, ge_module.HDC_DIM, device=device)
        ge_module._PROJ_SEMANTIC /= np.sqrt(ge_module.SEMANTIC_DIM)
    proj = ge_module._PROJ_SEMANTIC.to(device)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    item_count = 0
    t_start = time.time()

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    with open(OUTPUT_PATH, "wb") as f:
        f.write(b"\x00" * 256)

        for batch_start in range(0, len(semantic_vecs), BATCH_SIZE):
            batch_vecs = semantic_vecs[batch_start:batch_start + BATCH_SIZE]
            batch_texts = all_sentences[batch_start:batch_start + BATCH_SIZE]

            sem_batch = torch.from_numpy(np.stack(batch_vecs)).to(device)
            hdc_batch = torch.sign(torch.matmul(sem_batch, proj))
            hdc_np = hdc_batch.cpu().numpy().astype(np.float32)

            for i in range(len(batch_texts)):
                data = hdc_np[i].tobytes()
                meta = json.dumps({
                    "id": f"wiki_{item_count}",
                    "text": batch_texts[i][:500],
                    "source": "wikipedia"
                }, ensure_ascii=False).encode("utf-8")
                f.write(struct.pack("<I", len(data)))
                f.write(data)
                f.write(struct.pack("<I", len(meta)))
                f.write(meta)
                item_count += 1

            elapsed = time.time() - t_start
            rate = item_count / elapsed if elapsed > 0 else 0
            print(f"  HDV: {item_count}/{len(semantic_vecs)} ({100*item_count/len(semantic_vecs):.0f}%) "
                  f"| {rate:.0f} it/s | {elapsed:.0f}s")

            if item_count % CHECKPOINT_EVERY < BATCH_SIZE and item_count > 0:
                file_end = f.tell()
                write_hdv_header(f, item_count, file_end)
                f.seek(file_end)
                print(f"  CHECKPOINT: {item_count} itens")

        file_end = f.tell()
        write_hdv_header(f, item_count, file_end)

    elapsed = time.time() - t_start
    mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n[Compiler] CONCLUIDO: {item_count} conceitos, {mb:.1f} MB em {elapsed:.0f}s")
    print(f"[Compiler] Arquivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
