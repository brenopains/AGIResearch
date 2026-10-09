"""
train_W_from_wikipedia.py — Baixa Wikipedia (streaming) e treina Matriz W.
Estratégia:
  1. Stream artigos da Simple Wikipedia via datasets
  2. Para cada artigo: chunking 4 palavras → pares (chunk_n, chunk_{n+1})
  3. Codifica como HDV posicional → W += Σ outer(next, current)
  4. Salva periodicamente + coleta vocabulário para decoder
"""

import sys, os, warnings, time, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch
from datasets import load_dataset
from core.mind import NovaMind
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

DIM = Manifold.DIMENSIONALITY
OUT_DIR = r'D:\novamind_data'
OUT_PATH = os.path.join(OUT_DIR, 'transition_matrix_wiki.pt')
VOCAB_PATH = os.path.join(OUT_DIR, 'wiki_vocab.txt')
BATCH_SIZE = 500   # checkpoint + log a cada N artigos
MAX_ARTICLES = 5000  # ~1h de processamento
CHECKPOINT_EVERY = 500
CHUNK_SIZE = 4

os.makedirs(OUT_DIR, exist_ok=True)

# --- Init ---
print('[BOOT] NovaMind...')
mind = NovaMind()
engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=engine.device)
device = engine.device

# W já existente?
W_path = r'D:\novamind_data\transition_matrix.pt'
if os.path.exists(W_path):
    data = torch.load(W_path, map_location=device, weights_only=False)
    raw = data['W'] if isinstance(data, dict) else data
    W = raw.to(device=device, dtype=torch.float32)
    print(f'[W] Carregada: {list(W.shape)}, norm={W.norm().item():.1f}')
else:
    W = torch.zeros(DIM, DIM, device=device, dtype=torch.float32)
    print(f'[W] Nova: {list(W.shape)}')

word_cache = {}  # word → tensor
vocab_set = set()

def get_vec(word):
    w = word.lower().strip(".,!?;:'\"()[]{}")
    if not w or len(w) < 2:
        return None
    if w not in word_cache:
        word_cache[w] = engine.encode_text(w)
    return word_cache[w]

def encode_chunk(words):
    """Retorna tensor 1D [DIM] para um chunk de palavras."""
    vecs = [get_vec(w) for w in words]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    result = None
    for i, v in enumerate(vecs):
        pos = binder.position_vectors[i].to(device=device, dtype=torch.float32)
        bound = pos * v.tensor  # v é Manifold
        result = bound if result is None else result + bound
    if result is None:
        return None
    return torch.sign(result).float()

def process_article(text):
    """Extrai transições W += outer(next, current) de um artigo."""
    words = [w.lower().strip(".,!?;:'\"()[]{}")
             for w in text.split() if len(w.strip(".,!?;:'\"()[]{}")) >= 2]
    if len(words) < CHUNK_SIZE * 2:
        return 0
    chunks = [words[i:i+CHUNK_SIZE]
              for i in range(0, len(words), CHUNK_SIZE)]
    pairs = 0
    for i in range(len(chunks) - 1):
        current = encode_chunk(chunks[i])
        nxt = encode_chunk(chunks[i+1])
        if current is None or nxt is None:
            continue
        # W += outer(nxt, current) — equivalente a W += nxt[:, None] @ current[None, :]
        W.addmm_(nxt[:, None], current[None, :])
        pairs += 1
        for w in chunks[i] + chunks[i+1]:
            vocab_set.add(w.lower().strip(".,!?;:'\"()[]{}"))
    return pairs

# --- Dataset ---
print('[DATA] Carregando Simple Wikipedia (streaming)...')
ds = load_dataset("wikipedia", "20220301.simple",
                  split="train", streaming=True)
print(f'[DATA] Dataset pronto. Iterando...')

total_pairs = 0
total_articles = 0
t0 = time.time()

for idx, example in enumerate(ds):
    if MAX_ARTICLES is not None and idx >= MAX_ARTICLES:
        break
    text = example.get('text', '')
    if not text or len(text) < 50:
        continue
    pairs = process_article(text)
    if pairs > 0:
        total_pairs += pairs
        total_articles += 1

    if (idx + 1) % BATCH_SIZE == 0:
        dt = time.time() - t0
        rate = total_pairs / dt if dt > 0 else 0
        print(f'  [{idx+1:6d}] artigos={total_articles}, pares={total_pairs}, '
              f'vocab={len(vocab_set)}, W_norm={W.norm().item():.1f}, '
              f'{rate:.0f} pares/s, {dt:.0f}s')
        gc.collect()
        torch.cuda.empty_cache()

    # Save checkpoint
    if (idx + 1) % CHECKPOINT_EVERY == 0:
        W_bin = torch.sign(W).float()
        W_bin[W_bin == 0] = 1
        ckpt_path = OUT_PATH.replace('.pt', f'_ckpt{idx+1}.pt')
        torch.save({'W': W_bin.cpu(),
                     'vocab': sorted(vocab_set),
                     'articles': total_articles,
                     'pairs': total_pairs}, ckpt_path)
        print(f'  [CKPT] Salvo: {ckpt_path}')
        # Limpa cache de palavras para economizar RAM
        word_cache.clear()

print(f'\n[FIM] {time.time()-t0:.0f}s total')
print(f'  Artigos: {total_articles}')
print(f'  Pares: {total_pairs}')
print(f'  Vocab: {len(vocab_set)}')
print(f'  W norm: {W.norm().item():.1f}')

# --- Salvar W ---
W_bin = torch.sign(W).float()
W_bin[W_bin == 0] = 1
torch.save({'W': W_bin.cpu()}, OUT_PATH)
print(f'[SAVE] W salva: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)')

# Salvar vocabulário
with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
    for w in sorted(vocab_set):
        f.write(w + '\n')
print(f'[SAVE] Vocab salvo: {VOCAB_PATH} ({len(vocab_set)} palavras)')
