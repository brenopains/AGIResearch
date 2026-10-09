"""
train_transition_matrix.py — Mini teste do pipeline completo.
MAX_WIKI=100, MAX_CODE=100, verifica se o fluxo funciona.
"""
import sys, os, time, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch
from core.grounding_engine import GroundingEngine
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

CHUNK_SIZE = 3
DIM = Manifold.DIMENSIONALITY
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_PATH = r'D:\novamind_data\transition_matrix.pt'
MAX_WIKI = 100
MAX_CODE = 100
BATCH_SIZE = 2000

print(f'[TRAIN] Device: {DEVICE}, Dim: {DIM}, Chunk: {CHUNK_SIZE}')
engine = GroundingEngine()
binder = PositionalBinder(max_length=CHUNK_SIZE, device=DEVICE)

def encode_chunk(words):
    words = [w for w in words if len(w) >= 1]
    words = words[:CHUNK_SIZE]
    if len(words) < 2:
        return None
    try:
        return binder.encode_sequence(words, engine)
    except Exception:
        return None

def text_to_pairs(text):
    sents = re.split(r'[.!?\n;:()]+', text)
    pairs = []
    for sent in sents:
        words = sent.strip().split()
        if len(words) < 4:
            continue
        tensors = []
        for i in range(0, len(words), CHUNK_SIZE):
            mf = encode_chunk(words[i:i + CHUNK_SIZE])
            if mf is not None:
                tensors.append(mf.tensor)
        for i in range(len(tensors) - 1):
            pairs.append((tensors[i], tensors[i + 1]))
    return pairs

W = torch.zeros((DIM, DIM), device=DEVICE, dtype=torch.float32)
total_pairs = 0
t_start = time.time()
BUF_N, BUF_N1 = [], []

def flush_batch():
    global BUF_N, BUF_N1, W
    if not BUF_N:
        return
    A = torch.stack(BUF_N)
    B = torch.stack(BUF_N1)
    W += B.T @ A
    print(f'  Flushed batch: {len(BUF_N)} pares, '
          f'W.norm={W.norm().item():.1f}, '
          f'total={total_pairs}', flush=True)
    BUF_N, BUF_N1 = [], []

def add_pairs(pairs):
    global total_pairs, BUF_N, BUF_N1
    for n_vec, n1_vec in pairs:
        BUF_N.append(n_vec)
        BUF_N1.append(n1_vec)
        total_pairs += 1
        if len(BUF_N) >= BATCH_SIZE:
            flush_batch()

# Wikipedia
print('\n=== 1. WIKIPEDIA ===', flush=True)
from datasets import load_dataset
ds = load_dataset('wikipedia', '20220301.simple',
                  split='train', streaming=True,
                  trust_remote_code=True)
for i, article in enumerate(ds):
    text = article.get('text', '')
    if text and len(text) >= 100:
        pairs = text_to_pairs(text)
        add_pairs(pairs)
    if i % 20 == 0:
        print(f'  Artigo {i}: pairs={total_pairs}', flush=True)
    if i >= MAX_WIKI:
        break
print(f'  Wikipedia OK: {total_pairs} pares', flush=True)

# CodeSearchNet  
print('\n=== 2. CODESEARCHNET ===', flush=True)
ds_code = load_dataset('code_search_net', 'python',
                       split='train', streaming=True,
                       trust_remote_code=True)
for i, example in enumerate(ds_code):
    code = example.get('func_code_string', '')
    doc = example.get('func_documentation_string', '')
    combined = f'{doc}\n{code}'
    if len(combined) >= 30:
        pairs = text_to_pairs(combined)
        add_pairs(pairs)
    if i % 20 == 0:
        print(f'  Code {i}: pairs={total_pairs}', flush=True)
    if i >= MAX_CODE:
        break
print(f'  CodeSearchNet OK: {total_pairs} pares', flush=True)

# Finalizar
print('\n=== 3. FINALIZAR ===', flush=True)
flush_batch()
print(f'  Total: {total_pairs} pares', flush=True)
print(f'  Tempo: {time.time()-t_start:.0f}s', flush=True)

if total_pairs > 0:
    W_bin = torch.sign(W)
    W_bin[W_bin == 0] = 1
    torch.save(W_bin, OUT_PATH)
    sz = os.path.getsize(OUT_PATH) / (1024**2)
    print(f'  Salvo: {OUT_PATH} ({sz:.0f} MB)', flush=True)
    print(f'  OK!', flush=True)
else:
    print(f'  [ERRO] Nenhum par!', flush=True)
