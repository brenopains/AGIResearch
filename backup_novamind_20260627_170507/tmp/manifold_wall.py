"""
ATAQUE AO MURO FORA-DA-FAMILIA — hipotese nova (OFC manifold): o muro e COBERTURA do manifold de estruturas,
nao extrapolacao impossivel. Estrutura = POLINOMIO mod N: s2 = (a*s^2 + b*s + c) mod N. (a,b,c) = ponto no manifold.
Dois regimes contrastantes:
  R1 — LINEAR-ONLY: treina so a=0 (afim), testa a>0 (quadratica) = TIPO fora do mapa treinado (= o nosso teste antigo).
  R2 — FAMILIA-CHEIA: treina (a,b,c) variados (inclui quadraticas), testa (a,b,c) HELD-OUT = ponto NOVO no mapa coberto.
Query em s NUNCA vistos (induzir o polinomio, nao lookup). Se R2 generaliza e R1 nao -> o muro e COBERTURA
(treine no manifold e generaliza), nao extrapolacao -> caminho = cobrir o mapa (dados/diversidade), achavel. Sem LLM.
"""
import sys, os, random, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, A, KSUP = 13, 6, 50

def all_pairs():
    s = torch.arange(N, device=DEV).repeat_interleave(A); a = torch.arange(A, device=DEV).repeat(N); return s, a
def split(B):
    s_all, a_all = all_pairs(); P = s_all.shape[0]
    perm = torch.argsort(torch.rand(B, P, device=DEV), 1); return s_all, a_all, perm[:, :KSUP], perm[:, KSUP:]

def trans(coef, s, a):                       # coef: (A_,B_,C_) cada [B,A]; s2=(A_*s^2+B_*s+C_)%N
    A_, B_, C_ = coef; sq = (s.dim() == 1)
    if sq: s = s.unsqueeze(1); a = a.unsqueeze(1)
    B = s.shape[0]; bi = torch.arange(B, device=DEV).unsqueeze(1).expand_as(s)
    r = (A_[bi, a] * (s * s) + B_[bi, a] * s + C_[bi, a]) % N
    return r.squeeze(1) if sq else r

def gen_coef(B, regime):
    if regime == 'linear':   A_ = torch.zeros(B, A, dtype=torch.long, device=DEV)        # a=0 (so afim)
    else:                    A_ = torch.randint(0, N, (B, A), device=DEV)                # familia cheia (inclui quadr.)
    B_ = torch.randint(0, N, (B, A), device=DEV); C_ = torch.randint(0, N, (B, A), device=DEV)
    return (A_, B_, C_)

class Ind(nn.Module):
    def __init__(s, d=128, heads=8, layers=5):
        super().__init__()
        s.es = nn.Embedding(N, d); s.ea = nn.Embedding(A, d); s.es2 = nn.Embedding(N + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True); s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, N)
    def forward(s, ss, sa, ss2, qs, qa):
        sup = s.es(ss) + s.ea(sa) + s.es2(ss2); q = s.es(qs) + s.ea(qa) + s.es2(torch.full_like(qs, N))
        h = s.tr(torch.cat([sup, q], 1)); return s.out(h[:, sup.shape[1]:])

def batch(regime, B):
    coef = gen_coef(B, regime); s_all, a_all, sup, qry = split(B)
    S = s_all[sup]; Aa = a_all[sup]; S2 = trans(coef, S, Aa)
    QS = s_all[qry]; QA = a_all[qry]; QS2 = trans(coef, QS, QA)
    return S, Aa, S2, QS, QA, QS2

def run(train_regime, test_regime, iters=14000):
    net = Ind().to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    for it in range(iters):
        S, Aa, S2, QS, QA, QS2 = batch(train_regime, 96)
        loss = F.cross_entropy(net(S, Aa, S2, QS, QA).reshape(-1, N), QS2.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 4000 == 0: print(f"  [{train_regime}->{test_regime}] it{it} loss {loss.item():.3f}", flush=True)
    net.eval(); acc = []
    with torch.no_grad():
        for _ in range(40):
            S, Aa, S2, QS, QA, QS2 = batch(test_regime, 64)
            acc.append((net(S, Aa, S2, QS, QA).argmax(-1) == QS2).float().mean().item())
    return sum(acc) / len(acc)

if __name__ == "__main__":
    print(f"[{DEV}] MURO FORA-DA-FAMILIA via MANIFOLD | chance={1/N:.1%}")
    r1 = run('linear', 'full')      # treina linear, testa quadratica (TIPO fora do mapa) = o teste antigo
    r2 = run('full', 'full')        # treina familia cheia, testa held-out (ponto novo no mapa coberto)
    print("=" * 64)
    print(f"  R1 — treina LINEAR, testa QUADRATICA (tipo fora do mapa) ... {r1:.1%}")
    print(f"  R2 — treina FAMILIA CHEIA, testa held-out (no mapa) ........ {r2:.1%}")
    print("=" * 64)
    print("  LEITURA: R1 baixo e R2 alto -> o muro e COBERTURA do manifold (treine no mapa e generaliza),")
    print("           NAO extrapolacao -> caminho viavel = cobrir o mapa de estruturas (dados/diversidade).")
    print("           R2 tambem baixo -> ha um muro real ate dentro do manifold (honesto).")
