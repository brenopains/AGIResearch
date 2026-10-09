"""
MLC — META-LEARNING FOR COMPOSITIONALITY (a fundacao: compor o conhecido -> generalizar p/ composicoes NOVAS).
Dominio: grades. Primitivas geometricas base (rot/flip/transpose/shift). Tarefa = COMPOSICAO de 2 primitivas.
Meta-treino sobre um SUBCONJUNTO das composicoes; teste em composicoes HELD-OUT (cada primitiva foi vista em
OUTRAS combinacoes, mas ESTA combinacao nunca). Se generaliza -> generalizacao SISTEMATICA (o que LLM/interpolacao
faz dentro do fecho de composicoes). Sem LLM. GPU. (System-1: rapido, compoe o conhecido.)
"""
import sys, os, itertools, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
G, NC = 5, 4                          # grade 5x5, 4 cores
KDEMO = 4

# primitivas base (parameter-free) sobre [G,G] long
def rot90(x): return torch.rot90(x, 1, (-2, -1))
def flipH(x): return x.flip(-1)
def flipV(x): return x.flip(-2)
def transp(x): return x.transpose(-1, -2)
def shiftR(x): return x.roll(1, -1)
def shiftD(x): return x.roll(1, -2)
PRIMS = [rot90, flipH, flipV, transp, shiftR, shiftD]
PAIRS = list(itertools.product(range(len(PRIMS)), repeat=2))   # 36 composicoes (a,b) = aplica a depois b
import random; random.seed(0); random.shuffle(PAIRS)
TRAIN_PAIRS = PAIRS[:26]; HELD_PAIRS = PAIRS[26:]              # 26 vistas, 10 held-out (combinacao nova)

def apply_all(x):                     # x:[*,G,G] -> [6,*,G,G] (todas as primitivas, vetorizado)
    return torch.stack([f(x) for f in PRIMS])

def make_episode(pairs, B):           # VETORIZADO: 12 ops totais, independente de B
    pr = torch.tensor([random.choice(pairs) for _ in range(B)], device=DEV)   # [B,2]
    a_idx, b_idx = pr[:, 0], pr[:, 1]
    xs = torch.randint(0, NC, (B, KDEMO + 1, G, G), device=DEV)
    flat = xs.reshape(B, (KDEMO + 1), G, G)
    ys = torch.empty_like(xs)
    bi = torch.arange(B, device=DEV)
    for k in range(KDEMO + 1):        # so KDEMO+1=5 iteracoes (nao B)
        x = xs[:, k]                  # [B,G,G]
        s1 = apply_all(x)             # [6,B,G,G]
        x1 = s1[a_idx, bi]            # aplica primitiva A por elemento
        s2 = apply_all(x1)
        ys[:, k] = s2[b_idx, bi]      # aplica primitiva B
    return xs, ys

class MLC(nn.Module):
    def __init__(s, d=128, heads=8, layers=5):
        super().__init__()
        s.col = nn.Embedding(NC + 1, d)               # +1 = MASK p/ celulas de saida da query
        s.pos = nn.Embedding(G * G, d)
        s.seg = nn.Embedding(3, d)                    # 0=demo-in,1=demo-out,2=query-in
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, NC)
    def grid_tok(s, g, seg):                          # g:[B,G,G] -> [B,G*G,d]
        B = g.shape[0]; idx = torch.arange(G * G, device=DEV)
        return s.col(g.reshape(B, -1)) + s.pos(idx)[None] + s.seg(torch.full((G * G,), seg, device=DEV))[None]
    def forward(s, xs, ys):
        B = xs.shape[0]; toks = []
        for k in range(KDEMO):
            toks.append(s.grid_tok(xs[:, k], 0)); toks.append(s.grid_tok(ys[:, k], 1))
        toks.append(s.grid_tok(xs[:, KDEMO], 2))      # query input
        qmask = s.col(torch.full((B, G * G), NC, device=DEV)) + s.pos(torch.arange(G * G, device=DEV))[None] + s.seg(torch.full((G * G,), 1, device=DEV))[None]
        toks.append(qmask)                            # slots de saida a prever
        h = s.tr(torch.cat(toks, 1))
        return s.out(h[:, -G * G:]).reshape(B, G, G, NC)

def run(iters=6000):
    net = MLC().to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        xs, ys = make_episode(TRAIN_PAIRS, 64)
        logit = net(xs, ys); loss = F.cross_entropy(logit.reshape(-1, NC), ys[:, KDEMO].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 3000 == 0:
            ex = (logit.argmax(-1) == ys[:, KDEMO]).all(-1).all(-1).float().mean().item()
            print(f"  it{it} loss {loss.item():.3f} grade-exata {ex:.1%}", flush=True)
    net.eval()
    def ev(pairs, n=40):
        ac = []
        with torch.no_grad():
            for _ in range(n):
                xs, ys = make_episode(pairs, 64); pred = net(xs, ys).argmax(-1)
                ac.append((pred == ys[:, KDEMO]).all(-1).all(-1).float().mean().item())
        return sum(ac) / len(ac)
    return ev(TRAIN_PAIRS), ev(HELD_PAIRS)

if __name__ == "__main__":
    print(f"[{DEV}] MLC | {len(TRAIN_PAIRS)} composicoes vistas | {len(HELD_PAIRS)} HELD-OUT (combinacao nova) | chance(grade)={(1/NC)**(G*G):.0e}")
    tr, he = run()
    print("=" * 60)
    print(f"  composicoes VISTAS .......... {tr:.1%}")
    print(f"  composicoes NOVAS (held-out)  {he:.1%}   <- generalizacao SISTEMATICA")
    print("=" * 60)
    print("LEITURA: held-out alto -> meta-learning compoe o conhecido e generaliza p/ combinacao nova (MLC confirmado).")
