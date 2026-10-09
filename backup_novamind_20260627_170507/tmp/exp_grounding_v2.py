"""
EXPERIMENT 3b — Actively hunting a compositional-grounding signal
=================================================================

The v1 grounding run hit 33% zero-shot (chance 11%) -> there IS signal. Now we
chase it scientifically by sweeping hypotheses, all on the same tiny hardware:

  H1 (coverage)  : more attributes seen (NxN grid) -> held-out combos have more
                   shared support -> zero-shot should rise.
  H2 (capacity)  : does a bigger shared embedding help or saturate?
  H3 (training)  : do more epochs close the gap?

Metric per config: zero-shot top-1 on combinations NEVER trained together,
vs chance = 1/(C*S). A run is "interesting" if it clearly beats chance; a
"strong signal" if it approaches the seen-combo ceiling.

Self-contained, no downloads. Honest scoreboard at the end.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB = {"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0),"cyan":(0,1,1)}
ALL_COLORS = list(COLOR_RGB)
ALL_SHAPES = ["circle","square","triangle","cross","diamond"]


def shape_mask(shape, G):
    m = np.zeros((G,G), np.float32); cy=cx=(G-1)/2; r=G*0.40
    for y in range(G):
        for x in range(G):
            dy, dx = y-cy, x-cx
            if shape=="circle"   and dy*dy+dx*dx <= r*r: m[y,x]=1
            elif shape=="square" and 2<=y<G-2 and 2<=x<G-2: m[y,x]=1
            elif shape=="triangle" and y>=2 and abs(dx)<=(y-2)*0.5: m[y,x]=1
            elif shape=="cross"  and (abs(dy)<=1.5 or abs(dx)<=1.5): m[y,x]=1
            elif shape=="diamond" and abs(dy)+abs(dx)<=r: m[y,x]=1
    return m


def render(ci, si, colors, shapes, G, noise=0.06):
    m = shape_mask(shapes[si], G)
    img = np.stack([m*c for c in COLOR_RGB[colors[ci]]], 0)
    return img + np.random.randn(*img.shape).astype(np.float32)*noise


def text_vec(ci, si, C, S):
    v = np.zeros(C+S, np.float32); v[ci]=1; v[C+si]=1; return v


class ImgEnc(nn.Module):
    def __init__(self, d, hid=16):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(3,hid,3,padding=1), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                               nn.Linear(hid*16, d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)


class TxtEnc(nn.Module):
    def __init__(self, d, n_in):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(n_in,64), nn.ReLU(), nn.Linear(64,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)


def run_config(n_attr, embed_dim, epochs, G=14):
    colors, shapes = ALL_COLORS[:n_attr], ALL_SHAPES[:n_attr]
    C, S = len(colors), len(shapes)
    ALL = list(itertools.product(range(C), range(S)))
    held = {(i,(i+1)%S) for i in range(C)}          # one per row, shifted diagonal
    TRAIN = [c for c in ALL if c not in held]
    TEST  = [c for c in ALL if c in held]

    def batch(combos, reps):
        I,T=[],[]
        for (ci,si) in combos:
            for _ in range(reps):
                I.append(render(ci,si,colors,shapes,G)); T.append(text_vec(ci,si,C,S))
        return torch.tensor(np.array(I)).to(DEVICE), torch.tensor(np.array(T)).to(DEVICE)

    ie, te = ImgEnc(embed_dim).to(DEVICE), TxtEnc(embed_dim, C+S).to(DEVICE)
    opt = torch.optim.Adam(list(ie.parameters())+list(te.parameters()), 3e-3)
    tau = 0.07
    for _ in range(epochs):
        imgs,txts = batch(TRAIN, 6)
        zi,zt = ie(imgs), te(txts)
        logits = zi@zt.t()/tau
        lbl = torch.arange(len(imgs),device=DEVICE)
        loss = (F.cross_entropy(logits,lbl)+F.cross_entropy(logits.t(),lbl))/2
        opt.zero_grad(); loss.backward(); opt.step()

    ie.eval(); te.eval()
    with torch.no_grad():
        cand = torch.tensor(np.array([text_vec(c,s,C,S) for (c,s) in ALL])).to(DEVICE)
        zt_all = te(cand)
        ti,_ = batch(TRAIN,15); s_tr = ie(ti)@zt_all.t()
        tr_lbl = torch.tensor([ALL.index(c) for c in TRAIN for _ in range(15)],device=DEVICE)
        seen = (s_tr.argmax(1)==tr_lbl).float().mean().item()
        hi,_ = batch(TEST,40); s_h = ie(hi)@zt_all.t()
        h_lbl = torch.tensor([ALL.index(c) for c in TEST for _ in range(40)],device=DEVICE)
        zs = (s_h.argmax(1)==h_lbl).float().mean().item()
    return C, S, len(TRAIN), len(TEST), seen, zs, 100.0/(C*S)


CONFIGS = [
    # (n_attr, embed_dim, epochs)  -- hypotheses H1/H2/H3
    (3, 32, 400),   # v1 baseline-ish
    (4, 32, 400),   # H1: more coverage
    (5, 32, 400),   # H1: even more
    (5, 64, 400),   # H2: bigger embedding
    (5, 64, 900),   # H3: more training
]

print(f"Device: {DEVICE}")
print(f"{'attrs':>6} {'emb':>4} {'ep':>5} | {'seen%':>6} {'zero-shot%':>11} {'chance%':>8}  signal")
print("-"*60)
for (na, ed, ep) in CONFIGS:
    C,S,ntr,nte,seen,zs,ch = run_config(na, ed, ep)
    ratio = zs/(ch/100)
    tag = "STRONG" if zs>=0.8 else ("signal" if zs>ch/100*1.8 else "weak")
    print(f"{C}x{S:>3} {ed:>4} {ep:>5} | {seen:>6.0%} {zs:>11.0%} {ch:>7.0f}%  {ratio:.1f}x {tag}")
print(f"\nVRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU run")
