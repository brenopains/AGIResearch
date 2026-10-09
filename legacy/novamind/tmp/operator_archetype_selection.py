"""
SELECAO AUTONOMA DE ARQUETIPO DE OPERADOR — fecha o loop honesto do experimento anterior.
O 'grounded_operator_birth' mostrou: operador fresco com VIES ANCORADO induz lei fora-do-fecho e generaliza
(transfere tamanho), enquanto denso decora. MAS eu ESCOLHI o vies (local) porque sabia que a lei era local.
A pergunta honesta restante (3a coluna do relatorio = evolucao estrutural): o sistema escolhe SOZINHO o vies
certo para uma lei DESCONHECIDA — sem hard-code de qual e a lei?

Mecanismo testado (sem LLM): dada uma lei desconhecida (so input->output), NASCEM 3 arquetipos de operador,
cada um treina no residuo, e o sistema SELECIONA o que GENERALIZA (held-out + transfere tamanho). Se o sinal de
generalizacao seleciona o arquetipo cuja ESTRUTURA casa com a lei verdadeira, para leis de tipos DIFERENTES,
entao a 'escolha do vies' nao e hard-code: e descoberta estrutural guiada por generalizacao.

Leis de ESTRUTURAS contrastantes (cada uma casa com UM arquetipo):
  L_local : automato celular colorido (vizinhanca 3x3, contagem)      -> casa com CNN LOCAL (3x3).
  L_global: recolore cada celula pelo RANK GLOBAL da frequencia da cor -> casa com GLOBAL (pooling do grid todo).
Arquetipos (nascem frescos):
  CNN-LOCAL : conv 3x3 -> 1x1 (campo receptivo 3x3; ve vizinhanca, NAO ve estatistica global).
  GLOBAL    : 1x1 + mean-pool global broadcast (ve estatistica do grid todo, NAO ve vizinhanca local).
  DENSO     : MLP achatada (sem estrutura; tamanho fixo).
Selecao = arquetipo de maior EXATO held-out (desempate: transfere tamanho). Honesto: se cada lei elege o
arquetipo certo, a descoberta-de-vies e automatica. Se elege errado/empata, o muro continua aberto.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 4

# ---------------- leis ----------------
def L_local(g):                                   # CA colorido local (conta vizinhos)
    H, W = g.shape; out = g.copy(); nz = (g != 0).astype(np.int32)
    cnt = sum(np.roll(np.roll(nz, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - nz
    for i in range(H):
        for j in range(W):
            c = cnt[i, j]
            if g[i, j] == 0:
                if c == 3:
                    vals = [g[(i+dy) % H, (j+dx) % W] for dy in (-1,0,1) for dx in (-1,0,1) if not (dy==0 and dx==0)]
                    vals = [v for v in vals if v != 0]
                    out[i, j] = np.bincount(vals).argmax() if vals else 0
            else:
                if c < 2 or c > 3: out[i, j] = 0
    return out

def L_global(g):                                  # recolore pelo RANK global da frequencia da cor (precisa do grid todo)
    out = np.zeros_like(g)
    cols = [c for c in range(1, C) if (g == c).any()]
    order = sorted(cols, key=lambda c: (-(g == c).sum(), c))   # mais frequente -> rank 1
    rank = {c: i + 1 for i, c in enumerate(order)}
    for c in cols: out[g == c] = rank[c]
    return out

def rand_grid(H, W, p=0.35):
    g = (np.random.rand(H, W) < p).astype(int); col = np.random.randint(1, C, size=(H, W)); return g * col
def make(n, H, W, law):
    X = np.stack([rand_grid(H, W) for _ in range(n)]); Y = np.stack([law(x) for x in X]); return X, Y

# ---------------- arquetipos ----------------
class CNNLocal(nn.Module):                         # campo receptivo 3x3 (local, equivariante)
    def __init__(s, ch=48):
        super().__init__(); s.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1, padding_mode='circular'), nn.ReLU(),
            nn.Conv2d(ch, ch, 1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def forward(s, oh): return s.net(oh)
class GlobalCtx(nn.Module):                        # 1x1 por celula + estatistica GLOBAL do grid (sem vizinhanca 3x3)
    def __init__(s, ch=48):
        super().__init__()
        s.loc = nn.Conv2d(C, ch, 1); s.glob = nn.Linear(C, ch)
        s.head = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def forward(s, oh):
        B, _, H, W = oh.shape
        loc = s.loc(oh); g = s.glob(oh.mean((2, 3)))[:, :, None, None].expand(B, -1, H, W)
        return s.head(torch.cat([loc, g], 1))
class Dense(nn.Module):                            # sem estrutura (tamanho fixo)
    def __init__(s, H, W, hid=512):
        super().__init__(); s.H, s.W = H, W
        s.net = nn.Sequential(nn.Linear(H*W*C, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, H*W*C))
    def forward(s, oh): B = oh.shape[0]; return s.net(oh.reshape(B, -1)).reshape(B, C, s.H, s.W)

def onehot(X):
    return F.one_hot(torch.tensor(X, device=DEV, dtype=torch.long), C).permute(0, 3, 1, 2).float()
def train(m, X, Y, iters=1500, bs=64):
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    Xo = onehot(X); Yt = torch.tensor(Y, device=DEV, dtype=torch.long); n = len(X)
    for _ in range(iters):
        idx = torch.randint(0, n, (bs,), device=DEV)
        loss = F.cross_entropy(m(Xo[idx]), Yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def ev(m, X, Y):
    p = m(onehot(X)).argmax(1).cpu().numpy()
    return np.mean([np.array_equal(a, b) for a, b in zip(p, Y)])

def run_law(name, law):
    Xtr, Ytr = make(2000, 8, 8, law); Xte, Yte = make(400, 8, 8, law); Xbg, Ybg = make(200, 16, 16, law)
    archs = {"CNN-LOCAL": CNNLocal().to(DEV), "GLOBAL": GlobalCtx().to(DEV), "DENSO": Dense(8, 8).to(DEV)}
    res = {}
    for nm, m in archs.items():
        train(m, Xtr, Ytr)
        same = ev(m, Xte, Yte)
        try: big = ev(m, Xbg, Ybg)
        except Exception: big = float('nan')      # denso: dim fixa, nao transfere
        res[nm] = (same, big)
    pick = max(res, key=lambda k: (res[k][0], (0 if np.isnan(res[k][1]) else res[k][1])))
    print(f"  LEI [{name}]:")
    for nm, (s, b) in res.items():
        bt = "  N/A " if np.isnan(b) else f"{b:.0%}"
        print(f"      {nm:9s} held-out(8x8)={s:.0%}  transfere(16x16)={bt}")
    print(f"      -> SELECIONADO por generalizacao: {pick}", flush=True)
    return name, pick

def main():
    t0 = time.time()
    print(f"[{DEV}] SELECAO AUTONOMA DE ARQUETIPO | leis de estruturas contrastantes", flush=True)
    picks = [run_law("local (CA, vizinhanca)", L_local), run_law("global (rank de frequencia)", L_global)]
    print("=" * 70)
    ok = dict(picks).get("local (CA, vizinhanca)") == "CNN-LOCAL" and dict(picks).get("global (rank de frequencia)") == "GLOBAL"
    print("  LEITURA HONESTA:")
    if ok:
        print("   CADA lei elegeu o arquetipo cuja ESTRUTURA casa com ela (local->CNN, global->GLOBAL),")
        print("   guiada SO por generalizacao — sem eu dizer qual era a lei. A 'escolha do vies' NAO e")
        print("   hard-code: e descoberta estrutural automatica. Escapar do fecho fica = ter o MENU certo")
        print("   de arquetipos + selecionar por generalizacao. Mecanismo real, sem LLM.")
    else:
        print("   a selecao por generalizacao NAO elegeu o arquetipo certo p/ cada lei -> a descoberta-de-vies")
        print("   ainda nao e automatica; o muro (qual estrutura nasce) continua aberto (honesto).")
    print(f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
