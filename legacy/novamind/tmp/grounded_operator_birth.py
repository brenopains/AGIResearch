"""
NASCIMENTO DE OPERADOR ATERRADO (grounded operator birth) — o teste que a noite inteira NAO fez.
Diagnostico ate aqui: compor NAO escapa do fecho (biblioteca cresce, cobertura trava). A pergunta aberta e
INVENTAR operador novo sem LLM. As 3 colunas do relatorio (active inference + embodiment/fisica + crescimento
estrutural) destilam-se a UMA afirmacao testavel:
  "nao componha (fica no fecho) nem hard-code; FACA NASCER um operador parametrico fresco ajustado ao RESIDUO
   que a biblioteca atual nao explica — e o VIES ANCORADO (localidade/equivariancia = fisica) e o que faz esse
   operador GENERALIZAR a lei em vez de decorar."

LEI ESCONDIDA L = automato celular colorido (local, nao-linear, depende de CONTAGEM):
  - conta vizinhos nao-fundo (8-viz); nascimento se centro=fundo e count==3 (vira cor mais comum dos vizinhos);
  - morte se centro!=fundo e (count<2 ou count>3); senao mantem.
  L muda a CONTAGEM de cores -> e PROVADAMENTE fora do fecho de {rot,flip,shift,perm-cor} (todas bijecoes globais
  que preservam o multiset de cores). Logo nenhuma composicao das seed-ops pode computar L.

Tres regimes:
  (A) FECHO: melhor composicao (1..2) das seed-ops ajustando L            -> espera FALHAR (prova: fora do fecho).
  (B) OPERADOR ATERRADO: nasce CNN local 3x3 equivariante (vies de fisica) treinada no residuo input->L(input).
  (C) OPERADOR NAO-ATERRADO: nasce MLP densa (mesmo budget) sobre a grade achatada (sem vies local).
Teste decisivo de INDUCAO-DA-LEI vs DECOREBA: avalia em grades HELD-OUT (8x8) e em TAMANHO NUNCA VISTO (16x16).
  - aterrado generaliza E transfere de tamanho -> operador NOVO, fora do fecho, que INDUZIU A LEI (sinal real).
  - nao-aterrado ajusta 8x8 mas nao transfere tamanho -> decorou, nao induziu.
Sem LLM. GPU. Honesto: isto testa UM mecanismo (nascimento aterrado), nao "a AGI".
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 4                      # 0=fundo, 1..3 cores

# ---------------- LEI ESCONDIDA: automato celular colorido (local, conta, nao-linear) ----------------
def ca_step(g):
    H, W = g.shape; out = g.copy()
    nz = (g != 0).astype(np.int32)
    cnt = np.zeros((H, W), np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            cnt += np.roll(np.roll(nz, dy, 0), dx, 1)
    for i in range(H):
        for j in range(W):
            c = cnt[i, j]
            if g[i, j] == 0:
                if c == 3:                                   # nascimento = cor mais comum dos 8-viz
                    vals = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0: continue
                            vals.append(g[(i + dy) % H, (j + dx) % W])
                    vals = [v for v in vals if v != 0]
                    out[i, j] = np.bincount(vals).argmax() if vals else 0
            else:
                if c < 2 or c > 3: out[i, j] = 0             # morte
    return out

def rand_grid(H, W, p=0.35):
    g = (np.random.rand(H, W) < p).astype(int)
    col = np.random.randint(1, C, size=(H, W))
    return g * col

def make(n, H, W):
    X = np.stack([rand_grid(H, W) for _ in range(n)])
    Y = np.stack([ca_step(x) for x in X])
    return X, Y

# ---------------- (A) FECHO: seed-ops globais (bijecoes que preservam multiset de cor) ----------------
def rot180(g): return np.rot90(g, 2)
def fliph(g): return g[:, ::-1]
def flipv(g): return g[::-1, :]
def shiftU(g): return np.roll(g, -1, 0)
def shiftD(g): return np.roll(g, 1, 0)
def shiftL(g): return np.roll(g, -1, 1)
def shiftR(g): return np.roll(g, 1, 1)
def cperm(g): o = g.copy(); o[g == 1] = 2; o[g == 2] = 3; o[g == 3] = 1; return o
SEED = [rot180, fliph, flipv, shiftU, shiftD, shiftL, shiftR, cperm, lambda g: g]

def closure_baseline(X, Y):
    progs = [[f] for f in SEED] + [[a, b] for a in SEED for b in SEED]
    best_cell, best_exact = 0.0, 0.0
    for prog in progs:
        cell = ex = 0
        for x, y in zip(X, Y):
            o = x.copy()
            for f in prog: o = np.ascontiguousarray(f(o))
            cell += (o == y).mean(); ex += int(np.array_equal(o, y))
        cell /= len(X); ex /= len(X)
        if cell > best_cell: best_cell, best_exact = cell, ex
    return best_cell, best_exact

# ---------------- (B) operador ATERRADO: CNN local 3x3 equivariante (vies de fisica) ----------------
class Grounded(nn.Module):
    def __init__(s, ch=48):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(ch, ch, 1), nn.ReLU(),
            nn.Conv2d(ch, C, 1))
    def forward(s, oh): return s.net(oh)                      # [B,C,H,W]

# ---------------- (C) operador NAO-ATERRADO: MLP densa (sem vies local, tamanho fixo) ----------------
class Dense(nn.Module):
    def __init__(s, H, W, hid=512):
        super().__init__(); s.H, s.W = H, W
        s.net = nn.Sequential(nn.Linear(H * W * C, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                              nn.Linear(hid, H * W * C))
    def forward(s, oh):
        B = oh.shape[0]; z = s.net(oh.reshape(B, -1)); return z.reshape(B, C, s.H, s.W)

def onehot(X):
    t = torch.tensor(X, device=DEV, dtype=torch.long)
    return F.one_hot(t, C).permute(0, 3, 1, 2).float()

def train(model, Xtr, Ytr, iters=1500, bs=64):
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    Xo = onehot(Xtr); Yt = torch.tensor(Ytr, device=DEV, dtype=torch.long); n = len(Xtr)
    for it in range(iters):
        idx = torch.randint(0, n, (bs,), device=DEV)
        logit = model(Xo[idx]); loss = F.cross_entropy(logit, Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def evalm(model, X, Y):
    pred = model(onehot(X)).argmax(1).cpu().numpy()
    cell = (pred == Y).mean()
    ex = np.mean([np.array_equal(p, y) for p, y in zip(pred, Y)])
    return cell, ex

def main():
    t0 = time.time()
    Xtr, Ytr = make(2000, 8, 8)            # treino: grades 8x8
    Xte, Yte = make(400, 8, 8)             # held-out mesmo tamanho
    Xbig, Ybig = make(200, 16, 16)         # TAMANHO NUNCA VISTO (so o aterrado consegue tentar)
    print(f"[{DEV}] NASCIMENTO DE OPERADOR ATERRADO | lei=CA colorido (local, conta) | C={C}", flush=True)

    bc, be = closure_baseline(Xte, Yte)
    print(f"  (A) FECHO  (melhor composicao das seed-ops): cell={bc:.1%}  exato={be:.1%}   <- prova: fora do fecho", flush=True)

    g = train(Grounded().to(DEV), Xtr, Ytr)
    gc, ge = evalm(g, Xte, Yte); gcb, geb = evalm(g, Xbig, Ybig)
    npar_g = sum(p.numel() for p in g.parameters())
    print(f"  (B) ATERRADO   (CNN 3x3 equivar, {npar_g} par): 8x8 cell={gc:.1%} exato={ge:.1%} | 16x16(transfere) cell={gcb:.1%} exato={geb:.1%}", flush=True)

    d = train(Dense(8, 8).to(DEV), Xtr, Ytr)
    dc, de = evalm(d, Xte, Yte)
    npar_d = sum(p.numel() for p in d.parameters())
    print(f"  (C) NAO-ATERRADO (MLP densa, {npar_d} par): 8x8 cell={dc:.1%} exato={de:.1%} | 16x16 = impossivel por construcao (dim fixa)", flush=True)

    print("=" * 70)
    print("  LEITURA HONESTA:")
    print("   - (A) baixo confirma: a lei esta FORA do fecho das seed-ops (composicao nao alcanca).")
    print("   - (B) alto E transfere p/ 16x16 -> o operador que NASCEU induziu A LEI (saiu do fecho, generalizou).")
    print("   - (C) ajusta 8x8 mas nao transfere -> sem ancoragem, decora; nao inventa o operador-lei.")
    print("   => se B>>A e B transfere e C nao: 'fisica como professora' = a ancoragem e o que torna o")
    print("      nascimento de operador GENERALIZAVEL. Mecanismo real p/ escapar do fecho sem LLM.")
    print(f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
