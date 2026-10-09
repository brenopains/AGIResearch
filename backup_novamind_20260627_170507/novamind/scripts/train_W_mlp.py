"""
train_W_mlp.py — MLP não-linear (ReLU) para predizer próximo chunk
=================================================================
Arquitetura: Linear(10000, 4000) → ReLU → Linear(4000, 10000)
A ReLU quebra a linearidade que causa o viés do "the".
Treino: CosineEmbeddingLoss + AdamW + AMP + streaming Wikipedia.
"""

import sys, os, warnings, time
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
OUT_PATH = r'D:\novamind_data\transition_mlp.pt'
CHUNK_SIZE = 4
MAX_ARTICLES = 2000
BATCH_SIZE = 512
LOCAL_EPOCHS = 3
LR = 3e-4

device = torch.device('cuda')

print('[BOOT] NovaMind...')
mind = NovaMind()
engine = mind.hdc_memory.engine
binder = PositionalBinder(max_length=CHUNK_SIZE, device=device)

class HDCSequencePredictor(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 4000),
            nn.ReLU(),
            nn.Linear(4000, dim)
        )
    def forward(self, x):
        return self.network(x)

model = HDCSequencePredictor().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CosineEmbeddingLoss()
scaler = torch.cuda.amp.GradScaler()

def enc(words, cache):
    vecs = []
    for w in words:
        w = w.lower().strip(".,!?;:'\"()[]{}")
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
        c = enc(ch[i], cache)
        n = enc(ch[i+1], cache)
        if c is not None and n is not None: pts.append((c, n))
    return pts

print('[DATA] Simple Wikipedia...')
ds = load_dataset("wikipedia", "20220301.simple", split="train", streaming=True)

total_pairs = 0
total_trained = 0
buf = []
t0 = time.time()
t_log = t0

for idx, example in enumerate(ds):
    if MAX_ARTICLES and idx >= MAX_ARTICLES: break
    text = example.get('text', '')
    if not text or len(text) < 50: continue
    cache = {}
    pts = pairs(text, cache)
    buf.extend(pts)
    total_pairs += len(pts)
    while len(buf) >= BATCH_SIZE:
        batch = buf[:BATCH_SIZE]
        buf = buf[BATCH_SIZE:]
        cur = torch.stack([p[0] for p in batch])
        nxt = torch.stack([p[1] for p in batch])
        for _ in range(LOCAL_EPOCHS):
            with torch.amp.autocast('cuda'):
                pred = model(cur)
                loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_trained += len(batch)
        if time.time() - t_log > 15:
            print(f'  pares={total_pairs} trein={total_trained} loss={loss.item():.4f}')
            t_log = time.time()

if buf:
    cur = torch.stack([p[0] for p in buf])
    nxt = torch.stack([p[1] for p in buf])
    for _ in range(LOCAL_EPOCHS):
        with torch.amp.autocast('cuda'):
            pred = model(cur)
            loss = criterion(pred, nxt, torch.ones(pred.size(0), device=device))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    total_trained += len(buf)

dt = time.time() - t0
print(f'\n[FIM] {dt:.0f}s pares={total_pairs} trein={total_trained} loss={loss.item():.4f}')

# Salva state_dict da MLP (não binariza — a MLP precisa dos pesos float)
torch.save({'model': model.state_dict()}, OUT_PATH)
print(f'[SAVE] {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.0f} MB)')
