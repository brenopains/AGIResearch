"""
train_mlp_wiki_only.py — MLP: Wikipedia 2000 artigos
======================================================
Retreina MLP apenas com Wikipedia para restaurar baseline.
11 min, 286K pares, 2000 artigos.
"""
import sys, os, warnings, time, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch, torch.nn as nn, torch.optim as optim
from datasets import load_dataset
from core.mind import NovaMind
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

DIM = Manifold.DIMENSIONALITY
OUT_PATH = r'D:\novamind_data\transition_mlp.pt'
CHUNK_SIZE = 4; BATCH_SIZE = 512; LOCAL_EPOCHS = 3; LR = 3e-4
device = torch.device('cuda')

class MLP(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4000), nn.ReLU(), nn.Linear(4000, dim))
    def forward(self, x): return self.net(x)

print('[BOOT] NovaMind...', flush=True)
model = MLP().to(device)
opt = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CosineEmbeddingLoss()
scaler = torch.cuda.amp.GradScaler()
mind = NovaMind(); engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=device)

def enc(words, cache):
    vecs = []
    for w in words:
        w = w.strip(".,!?;:'\"()[]{}")
        if not w or len(w) < 2: continue
        if w not in cache: cache[w] = engine.encode_text(w)
        vecs.append(cache[w])
    if not vecs: return None
    r = None
    for i, v in enumerate(vecs):
        p = binder.position_vectors[i].to(device=device, dtype=torch.float32)
        r = (p * v.tensor) if r is None else r + (p * v.tensor)
    return torch.sign(r).float() if r is not None else None

def pairs(text, cache):
    raw = text.lower().split()
    ws = [w.strip(".,!?;:'\"()[]{}") for w in raw if len(w.strip(".,!?;:'\"()[]{}")) >= 2]
    if len(ws) < CHUNK_SIZE * 2: return []
    ch = [ws[i:i+CHUNK_SIZE] for i in range(0, len(ws), CHUNK_SIZE)]
    pts = []
    for i in range(len(ch)-1):
        c = enc(ch[i], cache); n = enc(ch[i+1], cache)
        if c is not None and n is not None: pts.append((c, n))
    return pts

def drain(buf):
    batches = 0
    last_loss = torch.tensor(0.0)
    while len(buf) >= BATCH_SIZE:
        batch = buf[:BATCH_SIZE]; del buf[:BATCH_SIZE]
        cur = torch.stack([p[0] for p in batch]); nxt = torch.stack([p[1] for p in batch])
        for _ in range(LOCAL_EPOCHS):
            with torch.amp.autocast('cuda'):
                pred = model(cur)
                last_loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
            opt.zero_grad(); scaler.scale(last_loss).backward(); scaler.step(opt); scaler.update()
        batches += 1
    return batches, last_loss

tp = 0; tt = 0; buf = []; loss = torch.tensor(0.0); t0 = time.time(); t_last_log = t0
MAX_ARTICLES = 2000
print('[WIKI]...', flush=True)
for idx, ex in enumerate(load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)):
    if idx >= MAX_ARTICLES: break
    text = ex.get('text', '')
    if not text or len(text) < 50: continue
    cache = {}; pts = pairs(text, cache); buf.extend(pts); tp += len(pts)
    b, loss_val = drain(buf); tt += b * BATCH_SIZE; loss = loss_val
    if time.time() - t_last_log > 30:
        dt = time.time() - t0
        print(f'  [{idx+1}/{MAX_ARTICLES}] pares={tp} trein={tt} loss={loss.item():.4f} time={dt:.0f}s', flush=True)
        t_last_log = time.time()
    if (idx+1) % 500 == 0: gc.collect(); torch.cuda.empty_cache()

if buf:
    batch = buf[:]; buf.clear()
    cur = torch.stack([p[0] for p in batch]); nxt = torch.stack([p[1] for p in batch])
    for _ in range(LOCAL_EPOCHS):
        with torch.amp.autocast('cuda'):
            pred = model(cur); loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    tt += len(batch)

dt = time.time() - t0
print(f'[FIM] {dt:.0f}s pares={tp} trein={tt} loss={loss.item():.4f}', flush=True)
torch.save({'model': model.state_dict()}, OUT_PATH)
print(f'[SAVE] {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)', flush=True)
