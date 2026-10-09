"""
MENU DE ARQUETIPOS EXPANDIDO — 7 ARQUÉTIPOS
Inclui os 3 originais (LOCAL, GLOBAL, OBJETO) + 4 novos (RELACIONAL, CONTAGEM, SHAPE-CHANGE, AGÊNCIA).
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import label
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 4

# ---------------- leis (7 estruturas) ----------------
# 1. LOCAL
def L_local(g):
    H, W = g.shape; out = g.copy(); nz = (g != 0).astype(np.int32)
    cnt = sum(np.roll(np.roll(nz, dy, 0), dx, 1) for dy in (-1,0,1) for dx in (-1,0,1)) - nz
    for i in range(H):
        for j in range(W):
            c = cnt[i, j]
            if g[i, j] == 0:
                if c == 3:
                    vals = [g[(i+dy)%H,(j+dx)%W] for dy in (-1,0,1) for dx in (-1,0,1) if not (dy==0 and dx==0)]
                    vals = [v for v in vals if v != 0]; out[i, j] = np.bincount(vals).argmax() if vals else 0
            elif c < 2 or c > 3: out[i, j] = 0
    return out

# 2. GLOBAL
def L_global(g):
    out = np.zeros_like(g); cols = [c for c in range(1, C) if (g == c).any()]
    order = sorted(cols, key=lambda c: (-(g == c).sum(), c)); rank = {c: i+1 for i, c in enumerate(order)}
    for c in cols: out[g == c] = rank[c]
    return out

# 3. OBJETO
def L_object(g):
    out = np.zeros_like(g); lab, n = label(g != 0)
    if n == 0: return out
    sizes = np.bincount(lab.ravel()); order = sorted(range(1, n+1), key=lambda k: (-sizes[k], k))
    rank = {k: min(i+1, C-1) for i, k in enumerate(order)}
    for k in range(1, n+1): out[lab == k] = rank[k]
    return out

# 4. RELACIONAL
def L_relational(g):
    # Se dois pixels estao na mesma linha e tem a mesma cor (>0), preenche entre eles.
    out = g.copy()
    H, W = g.shape
    for i in range(H):
        cols = np.where(g[i] > 0)[0]
        if len(cols) >= 2:
            c_val = g[i, cols[0]]
            if g[i, cols[-1]] == c_val:
                out[i, cols[0]:cols[-1]+1] = c_val
    return out

# 5. CONTAGEM
def L_count(g):
    # Pinta tudo com a cor = numero de pixels nao-zero % C
    out = np.full_like(g, np.count_nonzero(g) % C)
    return out

# 6. SHAPE-CHANGE (reduz tamanho)
def L_shape_change(g):
    # Downsample para grade 2x2. Retorna a cor predominante ou 0 no quadrante.
    H, W = g.shape
    out = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            quad = g[i*(H//2):(i+1)*(H//2), j*(W//2):(j+1)*(W//2)]
            vals = quad[quad > 0]
            if len(vals) > 0:
                out[i, j] = np.bincount(vals).argmax()
    return out

# 7. AGÊNCIA
def L_agency(g):
    # Move cada pixel colorido N passos para a direita, onde N = cor
    H, W = g.shape
    out = np.zeros_like(g)
    for i in range(H):
        for j in range(W):
            c = g[i, j]
            if c > 0:
                nj = (j + c) % W
                out[i, nj] = c
    return out

def rand_grid(H, W, p=0.30):
    g = (np.random.rand(H, W) < p).astype(int); col = np.random.randint(1, C, size=(H, W)); return g * col
def make(n, H, W, law): X = np.stack([rand_grid(H, W) for _ in range(n)]); return X, np.stack([law(x) for x in X])

# ---------------- arquetipos ----------------
class CNNLocal(nn.Module):
    def __init__(s, ch=48):
        super().__init__(); s.net = nn.Sequential(nn.Conv2d(C, ch, 3, padding=1, padding_mode='circular'),
            nn.ReLU(), nn.Conv2d(ch, ch, 1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X): return s.net(s.feats(X))

class GlobalCtx(nn.Module):
    def __init__(s, ch=48):
        super().__init__(); s.loc = nn.Conv2d(C, ch, 1); s.glob = nn.Linear(C, ch)
        s.head = nn.Sequential(nn.Conv2d(ch*2, ch, 1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X):
        oh = s.feats(X); B, _, H, W = oh.shape
        g = s.glob(oh.mean((2,3)))[:, :, None, None].expand(B, -1, H, W)
        return s.head(torch.cat([s.loc(oh), g], 1))

NF = C + 4
class ObjectNet(nn.Module):
    def __init__(s, hid=64):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(NF, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def feats(s, X):
        B = X.shape[0]; H, W = X.shape[1], X.shape[2]; out = np.zeros((B, H, W, NF), np.float32)
        for b in range(B):
            g = X[b]; lab, n = label(g != 0); sizes = np.bincount(lab.ravel())
            order = sorted(range(1, n+1), key=lambda k: (-sizes[k], k)) if n else []
            rank = {k: i+1 for i, k in enumerate(order)}
            for i in range(H):
                for j in range(W):
                    l = lab[i, j]
                    if l == 0: out[b, i, j, C] = 1.0
                    else:
                        out[b, i, j, g[i, j]] = 1.0
                        out[b, i, j, C+1] = sizes[l]/(H*W); out[b, i, j, C+2] = rank[l]/max(n,1); out[b, i, j, C+3] = n/(H*W)
        return torch.tensor(out, device=DEV)
    def forward(s, X):
        f = s.feats(X); B, H, W, _ = f.shape
        return s.mlp(f.reshape(B*H*W, NF)).reshape(B, H, W, C).permute(0, 3, 1, 2)

class RelationalNet(nn.Module):
    def __init__(s, hid=32):
        super().__init__()
        s.proj_q = nn.Conv2d(C, hid, 1); s.proj_k = nn.Conv2d(C, hid, 1); s.proj_v = nn.Conv2d(C, hid, 1)
        s.head = nn.Conv2d(hid, C, 1)
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X):
        f = s.feats(X); B, _, H, W = f.shape
        q = s.proj_q(f).view(B, -1, H*W).permute(0, 2, 1) # B, N, hid
        k = s.proj_k(f).view(B, -1, H*W) # B, hid, N
        v = s.proj_v(f).view(B, -1, H*W).permute(0, 2, 1) # B, N, hid
        attn = torch.softmax(torch.bmm(q, k) / (q.shape[-1]**0.5), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, -1, H, W)
        return s.head(out)

class CountNet(nn.Module):
    def __init__(s, hid=32):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(C, hid), nn.ReLU(), nn.Linear(hid, C))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X):
        f = s.feats(X); B, _, H, W = f.shape
        # Count the number of non-zero pixels basically by summing over spatial dims
        counts = f.sum(dim=(2,3)) # B, C
        out_counts = s.mlp(counts) # B, C
        return out_counts.view(B, C, 1, 1).expand(B, C, H, W)

class ShapeChangeNet(nn.Module):
    def __init__(s, out_size=(2,2), hid=32):
        super().__init__()
        s.out_size = out_size
        s.pool = nn.AdaptiveAvgPool2d(out_size)
        s.net = nn.Sequential(nn.Conv2d(C, hid, 1), nn.ReLU(), nn.Conv2d(hid, C, 1))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X):
        f = s.feats(X) # B, C, H, W
        p = s.pool(f)  # B, C, out_size[0], out_size[1]
        return s.net(p)

class AgencyNet(nn.Module):
    def __init__(s, hid=32):
        super().__init__()
        # RNN-like behavior: applies a transformation repeatedly
        s.cell = nn.Sequential(nn.Conv2d(C, hid, 3, padding=1, padding_mode='circular'),
                               nn.ReLU(), nn.Conv2d(hid, C, 1))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X):
        state = s.feats(X)
        for _ in range(3): # fixed 3 steps of simulation
            state = F.relu(state + s.cell(state))
        return s.cell(state)

class Dense(nn.Module):
    def __init__(s, in_H, in_W, out_H, out_W, hid=512):
        super().__init__(); s.out_H, s.out_W = out_H, out_W
        s.net = nn.Sequential(nn.Linear(in_H*in_W*C, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out_H*out_W*C))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X): B = X.shape[0]; return s.net(s.feats(X).reshape(B, -1)).reshape(B, C, s.out_H, s.out_W)

def train(m, X, Y, iters=1200, bs=64):
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    Yt = torch.tensor(Y, device=DEV, dtype=torch.long); n = len(X)
    for _ in range(iters):
        idx = np.random.randint(0, n, bs); loss = F.cross_entropy(m(X[idx]), Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def ev(m, X, Y):
    p = m(X).argmax(1).cpu().numpy(); return np.mean([np.array_equal(a, b) for a, b in zip(p, Y)])

def run_law(name, law, want, out_size=None):
    H, W = 8, 8
    Hbg, Wbg = 12, 12
    out_H, out_W = out_size if out_size else (H, W)
    out_Hbg, out_Wbg = out_size if out_size else (Hbg, Wbg)

    Xtr, Ytr = make(1500, H, W, law); Xte, Yte = make(300, H, W, law); Xbg, Ybg = make(150, Hbg, Wbg, law)
    archs = {
        "LOCAL": CNNLocal().to(DEV), 
        "GLOBAL": GlobalCtx().to(DEV), 
        "OBJETO": ObjectNet().to(DEV), 
        "RELACIONAL": RelationalNet().to(DEV),
        "CONTAGEM": CountNet().to(DEV),
        "SHAPECHANGE": ShapeChangeNet(out_size=(2,2)).to(DEV),
        "AGENCIA": AgencyNet().to(DEV),
        "DENSO": Dense(H, W, out_H, out_W).to(DEV)
    }
    
    # We only train models whose output spatial size matches out_size or are dynamic (and happen to match)
    res = {}
    for nm, m in archs.items():
        try:
            # Check if output size matches
            dummy_out = m(Xtr[:2])
            if dummy_out.shape[2:] != (out_H, out_W):
                continue
            
            train(m, Xtr, Ytr); same = ev(m, Xte, Yte)
            try: big = ev(m, Xbg, Ybg)
            except Exception: big = float('nan')
            res[nm] = (same, big)
        except Exception as e:
            pass # Architecture failed to process

    pick = max(res, key=lambda k: (res[k][0], 0 if np.isnan(res[k][1]) else res[k][1]))
    print(f"  LEI [{name}] (casa com {want}):")
    for nm, (s, b) in res.items():
        print(f"      {nm:11s} held-out={s:.0%}  transfere={'N/A' if np.isnan(b) else f'{b:.0%}'}")
    print(f"      -> elege {pick}   {'OK' if pick == want else 'ERRO'}", flush=True)
    return pick == want

def main():
    t0 = time.time(); print(f"[{DEV}] MENU DE ARQUETIPOS EXPANDIDO | 7 leis contrastantes", flush=True)
    oks = [
        run_law("local (CA)", L_local, "LOCAL"),
        run_law("global (rank de cor)", L_global, "GLOBAL"),
        run_law("objeto (rank de tamanho)", L_object, "OBJETO"),
        run_law("relacional (conectar pixels da mesma cor)", L_relational, "RELACIONAL"),
        run_law("contagem (paridade da contagem modulo C)", L_count, "CONTAGEM"),
        run_law("shape-change (downsample 2x2)", L_shape_change, "SHAPECHANGE", out_size=(2,2)),
        run_law("agência (movimento N passos)", L_agency, "AGENCIA")
    ]
    print("=" * 70)
    print(f"  acertos de selecao: {sum(oks)}/7")
    print(f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
