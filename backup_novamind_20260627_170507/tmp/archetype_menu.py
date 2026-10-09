"""
MENU DE ARQUETIPOS (core-knowledge) — expansao de H186. Biblioteca de arquetipos estruturais que NASCEM frescos
e sao selecionados por generalizacao. Cada arquetipo encarna UM prior de Spelke:
  LOCAL  (objeto/vizinhanca)  : conv 3x3 equivariante      — ve vizinhanca, nao ve global.
  GLOBAL (numero/estatistica) : 1x1 + pooling do grid todo  — ve estatistica global, nao ve vizinhanca.
  OBJETO (topologia)          : componentes conexos -> feat por-objeto -> MLP — ve objetos/tamanho/rank, nao ve celula.
Tres leis de estruturas CONTRASTANTES; cada uma deve eleger o arquetipo cuja estrutura casa com ela, guiada SO por
generalizacao (held-out exato + transfere tamanho), sem dizer qual e a lei. Se o menu de 3 elege certo p/ as 3 ->
descoberta-de-vies automatica com menu maior. Sem LLM. Reutilizavel pelo solver de ARC real.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import label
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 4

# ---------------- leis (3 estruturas) ----------------
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
def L_global(g):
    out = np.zeros_like(g); cols = [c for c in range(1, C) if (g == c).any()]
    order = sorted(cols, key=lambda c: (-(g == c).sum(), c)); rank = {c: i+1 for i, c in enumerate(order)}
    for c in cols: out[g == c] = rank[c]
    return out
def L_object(g):                                   # recolore cada OBJETO pelo rank do seu TAMANHO (maior->1)
    out = np.zeros_like(g); lab, n = label(g != 0)
    if n == 0: return out
    sizes = np.bincount(lab.ravel()); order = sorted(range(1, n+1), key=lambda k: (-sizes[k], k))
    rank = {k: min(i+1, C-1) for i, k in enumerate(order)}
    for k in range(1, n+1): out[lab == k] = rank[k]
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
NF = C + 4                                          # feat por celula do arquetipo OBJETO
class ObjectNet(nn.Module):
    def __init__(s, hid=64):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(NF, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def feats(s, X):                                # componentes conexos -> feat por celula (tamanho, rank, ncomp, cor, bg)
        B = X.shape[0]; H, W = X.shape[1], X.shape[2]; out = np.zeros((B, H, W, NF), np.float32)
        for b in range(B):
            g = X[b]; lab, n = label(g != 0); sizes = np.bincount(lab.ravel())
            order = sorted(range(1, n+1), key=lambda k: (-sizes[k], k)) if n else []
            rank = {k: i+1 for i, k in enumerate(order)}
            for i in range(H):
                for j in range(W):
                    l = lab[i, j]
                    if l == 0: out[b, i, j, C] = 1.0     # bg flag
                    else:
                        out[b, i, j, g[i, j]] = 1.0      # cor onehot (0..C-1)
                        out[b, i, j, C+1] = sizes[l]/(H*W); out[b, i, j, C+2] = rank[l]/max(n,1); out[b, i, j, C+3] = n/(H*W)
        return torch.tensor(out, device=DEV)
    def forward(s, X):
        f = s.feats(X); B, H, W, _ = f.shape
        return s.mlp(f.reshape(B*H*W, NF)).reshape(B, H, W, C).permute(0, 3, 1, 2)

class Dense(nn.Module):
    def __init__(s, H, W, hid=512):
        super().__init__(); s.H, s.W = H, W
        s.net = nn.Sequential(nn.Linear(H*W*C, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, H*W*C))
    def feats(s, X): return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0,3,1,2).float()
    def forward(s, X): B = X.shape[0]; return s.net(s.feats(X).reshape(B, -1)).reshape(B, C, s.H, s.W)

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

def run_law(name, law, want):
    Xtr, Ytr = make(1500, 8, 8, law); Xte, Yte = make(300, 8, 8, law); Xbg, Ybg = make(150, 12, 12, law)
    archs = {"LOCAL": CNNLocal().to(DEV), "GLOBAL": GlobalCtx().to(DEV), "OBJETO": ObjectNet().to(DEV), "DENSO": Dense(8, 8).to(DEV)}
    res = {}
    for nm, m in archs.items():
        train(m, Xtr, Ytr); same = ev(m, Xte, Yte)
        try: big = ev(m, Xbg, Ybg)
        except Exception: big = float('nan')
        res[nm] = (same, big)
    pick = max(res, key=lambda k: (res[k][0], 0 if np.isnan(res[k][1]) else res[k][1]))
    print(f"  LEI [{name}] (casa com {want}):")
    for nm, (s, b) in res.items():
        print(f"      {nm:7s} held-out={s:.0%}  transfere={'N/A' if np.isnan(b) else f'{b:.0%}'}")
    print(f"      -> elege {pick}   {'OK' if pick == want else 'ERRO'}", flush=True)
    return pick == want

def main():
    t0 = time.time(); print(f"[{DEV}] MENU DE ARQUETIPOS core-knowledge | 3 leis contrastantes", flush=True)
    oks = [run_law("local (CA)", L_local, "LOCAL"),
           run_law("global (rank de cor)", L_global, "GLOBAL"),
           run_law("objeto (rank de tamanho)", L_object, "OBJETO")]
    print("=" * 70)
    print(f"  acertos de selecao: {sum(oks)}/3")
    print("  LEITURA: 3/3 -> menu maior tambem elege certo so por generalizacao (descoberta-de-vies automatica).")
    print(f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
