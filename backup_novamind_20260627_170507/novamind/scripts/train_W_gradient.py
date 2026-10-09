"""
train_W_gradient.py — Treino com Gradiente (otimizado RAM 4GB)
===============================================================
Cache limpo a cada artigo (evita swap). MAX_ARTICLES=2000 (~300K pares).
"""

import sys, os, warnings, time, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from core.mind import NovaMind
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

DIM = Manifold.DIMENSIONALITY
OUT_PATH = r'D:\novamind_data\transition_matrix_W.pt'
CHUNK_SIZE = 4
MAX_ARTICLES = 2000
BATCH_SIZE = 256
LOCAL_EPOCHS = 5
LR = 3e-4

device = torch.device('cuda')

print('[BOOT] NovaMind...')
mind = NovaMind()
engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=device)

class HDCTransitionModel(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        return self.linear(x)

model = HDCTransitionModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CosineEmbeddingLoss()
scaler = torch.cuda.amp.GradScaler()

def encode_chunk(words, word_cache):
    vecs = []
    for w in words:
        w = w.lower().strip(".,!?;:'\"()[]{}")
        if not w or len(w) < 2:
            continue
        if w not in word_cache:
            word_cache[w] = engine.encode_text(w)
        vecs.append(word_cache[w])
    if not vecs:
        return None
    result = None
    for i, v in enumerate(vecs):
        pos = binder.position_vectors[i].to(device=device, dtype=torch.float32)
        result = (pos * v.tensor) if result is None else result + (pos * v.tensor)
    return torch.sign(result).float() if result is not None else None

def extract_pairs(text, word_cache):
    raw = text.lower().split()
    words = [w.strip(".,!?;:'\"()[]{}") for w in raw if len(w.strip(".,!?;:'\"()[]{}")) >= 2]
    if len(words) < CHUNK_SIZE * 2:
        return []
    chunks = [words[i:i+CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    pairs = []
    for i in range(len(chunks) - 1):
        cur = encode_chunk(chunks[i], word_cache)
        nxt = encode_chunk(chunks[i+1], word_cache)
        if cur is not None and nxt is not None:
            pairs.append((cur, nxt))
    return pairs

# --- Data + Train ---
print('[DATA] Simple Wikipedia...')
ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)

total_pairs = 0
total_trained = 0
pair_buffer = []
t0 = time.time()
t_log = t0

for idx, example in enumerate(ds):
    if MAX_ARTICLES and idx >= MAX_ARTICLES:
        break
    text = example.get('text', '')
    if not text or len(text) < 50:
        continue

    word_cache = {}  # limpa a cada artigo
    pairs = extract_pairs(text, word_cache)
    pair_buffer.extend(pairs)
    total_pairs += len(pairs)

    while len(pair_buffer) >= BATCH_SIZE:
        batch = pair_buffer[:BATCH_SIZE]
        pair_buffer = pair_buffer[BATCH_SIZE:]
        current = torch.stack([p[0] for p in batch])
        nxt = torch.stack([p[1] for p in batch])

        for _ in range(LOCAL_EPOCHS):
            with torch.amp.autocast('cuda'):
                pred = model(current)
                loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_trained += len(batch)

        if time.time() - t_log > 15:
            print(f'  pares={total_pairs} trein={total_trained} loss={loss.item():.4f} '
                  f'norm={model.linear.weight.norm().item():.1f}')
            t_log = time.time()

# Final buffer
if pair_buffer:
    current = torch.stack([p[0] for p in pair_buffer])
    nxt = torch.stack([p[1] for p in pair_buffer])
    for _ in range(LOCAL_EPOCHS):
        with torch.amp.autocast('cuda'):
            pred = model(current)
            loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    total_trained += len(pair_buffer)

dt = time.time() - t0
print(f'\n[FIM] {dt:.0f}s  pares={total_pairs} trein={total_trained}')
print(f'W norm={model.linear.weight.norm().item():.1f}  loss={loss.item():.4f}')

W_trained = torch.sign(model.linear.weight.data).float()
W_trained[W_trained == 0] = 1
torch.save({'W': W_trained.cpu()}, OUT_PATH)
print(f'[SAVE] {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)')
