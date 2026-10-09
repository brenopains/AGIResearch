"""train_W_from_hdv.py — Treina W usando os metadados dos .hdv ja compilados."""
import sys, os, time, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch
from core.mind import NovaMind
from core.grounding_engine import GroundingEngine
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

CHUNK_SIZE = 3
DIM = Manifold.DIMENSIONALITY
DEVICE = 'cuda'
OUT_PATH = r'D:\novamind_data\transition_matrix.pt'
BATCH_SIZE = 2000

print(f'[TRAIN] Device: {DEVICE}')
mind = NovaMind()
engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=DEVICE)

def encode_chunk(words):
    words = [w for w in words if len(w) >= 1][:CHUNK_SIZE]
    if len(words) < 2:
        return None
    try:
        return binder.encode_sequence(words, engine)
    except:
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
            mf = encode_chunk(words[i:i+CHUNK_SIZE])
            if mf is not None:
                tensors.append(mf.tensor)
        for i in range(len(tensors)-1):
            pairs.append((tensors[i], tensors[i+1]))
    return pairs

W = torch.zeros((DIM, DIM), device=DEVICE, dtype=torch.float32)
total_pairs = 0
t0 = time.time()
BUF_N, BUF_N1 = [], []

def flush():
    global BUF_N, BUF_N1, W
    if not BUF_N:
        return
    A = torch.stack(BUF_N)
    B = torch.stack(BUF_N1)
    W += B.T @ A
    BUF_N, BUF_N1 = [], []

def add(pairs):
    global total_pairs, BUF_N, BUF_N1
    for n, n1 in pairs:
        BUF_N.append(n); BUF_N1.append(n1)
        total_pairs += 1
        if len(BUF_N) >= BATCH_SIZE:
            flush()

# Processa TODOS os metadados dos .hdv
all_texts = []
for src, idx_list in mind.hdc_memory.indices.items():
    for meta in idx_list:
        txt = meta.get('text', meta.get('id', ''))
        if txt and len(txt) > 5:
            all_texts.append(txt)

print(f'Total textos: {len(all_texts)}')
for i, txt in enumerate(all_texts):
    pairs = text_to_pairs(txt)
    add(pairs)
    if i % 500 == 0:
        print(f'  {i}/{len(all_texts)}: {total_pairs} pares, t={time.time()-t0:.0f}s', flush=True)

print(f'\nFinalizando... {total_pairs} pares totais')
flush()
print(f'Tempo: {time.time()-t0:.0f}s')

if total_pairs > 100:
    W_bin = torch.sign(W).float()  # .float() garante tensor regular
    W_bin[W_bin == 0] = 1
    torch.save({'W': W_bin.cpu()}, OUT_PATH)
    sz = os.path.getsize(OUT_PATH)/1024**2
    print(f'Salvo: {OUT_PATH} ({sz:.0f} MB)')
    print(f'OK!')
else:
    print(f'Poucos pares: {total_pairs}')
