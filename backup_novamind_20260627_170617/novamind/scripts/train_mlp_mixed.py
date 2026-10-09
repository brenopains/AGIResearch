"""
train_mlp_mixed.py — MLP: Wikipedia (286K) + CodeSearchNet Python (200K)
========================================================================
Remove bias do "the" adicionando transições de código (Python).
~486K pares, treino ~12 min.
"""

import sys, os, warnings, time, gc
warnings.filterwarnings('ignore')
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))

import torch, torch.nn as nn, torch.optim as optim
from datasets import load_dataset
from core.mind import NovaMind
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

DIM = Manifold.DIMENSIONALITY; OUT_PATH = r'D:\novamind_data\transition_mlp_mixed.pt'
CHUNK_SIZE = 4; BATCH_SIZE = 512; LOCAL_EPOCHS = 3; LR = 3e-4
device = torch.device('cuda')

class MLP(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4000), nn.ReLU(), nn.Linear(4000, dim))
    def forward(self, x): return self.net(x)

model = MLP().to(device); opt = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CosineEmbeddingLoss(); scaler = torch.cuda.amp.GradScaler()

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
    while len(buf) >= BATCH_SIZE:
        batch = buf[:BATCH_SIZE]; del buf[:BATCH_SIZE]
        cur = torch.stack([p[0] for p in batch]); nxt = torch.stack([p[1] for p in batch])
        for _ in range(LOCAL_EPOCHS):
            with torch.amp.autocast('cuda'):
                pred = model(cur)
                loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

print('[BOOT] NovaMind...')
mind = NovaMind(); engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=device)
tp = 0; tt = 0; buf = []; t0 = time.time()

# 1. Wikipedia
print('[WIKI]...')
for idx, ex in enumerate(load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)):
    if idx >= 2000: break
    text = ex.get('text', '')
    if not text or len(text) < 50: continue
    cache = {}; pts = pairs(text, cache); buf.extend(pts); tp += len(pts)
    drain(buf); tt = tp
    if (idx+1) % 500 == 0: gc.collect(); torch.cuda.empty_cache(); print(f'  Wiki [{idx+1}/2000] pares={tp} time={time.time()-t0:.0f}s')

# 2. CodeSearchNet Python
print('[CODE] CodeSearchNet Python...')
try:
    for idx, ex in enumerate(load_dataset("code_search_net", "python", split="train", streaming=True)):
        if idx >= 50000: break
        code = ex.get('whole_func_string', '') or ''
        if not code or len(code) < 20: continue
        code_clean = code.replace('\n', ' ').replace('\t', ' ')
        cache = {}; pts = pairs(code_clean, cache); buf.extend(pts); tp += len(pts)
        drain(buf); tt = tp
        if (idx+1) % 10000 == 0: gc.collect(); torch.cuda.empty_cache(); print(f'  Code [{idx+1}/50000] pares={tp} time={time.time()-t0:.0f}s')
except Exception as e: print(f'  [CODE] Erro: {e}')

# Final
while buf:
    drain(buf)

dt = time.time() - t0
print(f'\n[FIM] {dt:.0f}s pares={tp} loss={loss.item():.4f}')
torch.save({'model': model.state_dict()}, OUT_PATH)
print(f'[SAVE] {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)')
