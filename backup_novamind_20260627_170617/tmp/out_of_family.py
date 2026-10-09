"""
O TESTE DECISIVO — generalizacao FORA DA FAMILIA (o que separa meta-aprendizado conhecido de AGI).
O A1 acertou 99.8% DENTRO da familia (treinou afim, testou afim). Aqui treino o indutor numa MISTURA de familias
de estrutura (aditiva, multiplicativa, afim) e testo em familias HELD-OUT que ele NUNCA viu (quadratica, permutacao).
Se generaliza pro tipo NOVO -> sinal forte de inducao de estrutura aberta. Se cai p/ ~chance -> o muro do ARC/AGI:
in-context interpola DENTRO das familias treinadas, nao extrapola pra tipo novo. Sem LLM. GPU.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N, A, KSUP = 13, 6, 46

def all_pairs():
    s = torch.arange(N, device=DEV).repeat_interleave(A); a = torch.arange(A, device=DEV).repeat(N); return s, a
def split(B):
    s_all, a_all = all_pairs(); P = s_all.shape[0]
    perm = torch.argsort(torch.rand(B, P, device=DEV), 1); return s_all, a_all, perm[:, :KSUP], perm[:, KSUP:]

# ---- familias de estrutura (cada uma: params por dominio + transicao) ----
def fam_add(B):  return torch.randint(0, N, (B, A), device=DEV)
def fam_mul(B):  return torch.randint(1, N, (B, A), device=DEV)
def fam_aff(B):  return (torch.randint(1, N, (B, A), device=DEV), torch.randint(0, N, (B, A), device=DEV))
def fam_sq(B):   return torch.randint(0, N, (B, A), device=DEV)              # quadratica
def fam_perm(B): return torch.argsort(torch.rand(B, A, N, device=DEV), 2)     # permutacao arbitraria

def trans(fam, p, s, a):
    sq = (s.dim() == 1)
    if sq: s = s.unsqueeze(1); a = a.unsqueeze(1)
    B = s.shape[0]; bi = torch.arange(B, device=DEV).unsqueeze(1).expand_as(s)
    if fam == 'add':  r = (s + p[bi, a]) % N
    elif fam == 'mul': r = (p[bi, a] * s) % N
    elif fam == 'aff': k, b = p; r = (k[bi, a] * s + b[bi, a]) % N
    elif fam == 'sq':  r = (s * s + p[bi, a]) % N
    elif fam == 'perm': r = p[bi, a, s]
    return r.squeeze(1) if sq else r

GENS = {'add': fam_add, 'mul': fam_mul, 'aff': fam_aff, 'sq': fam_sq, 'perm': fam_perm}
TRAIN_FAMS = ['add', 'mul', 'aff']            # familias VISTAS no treino
HELDOUT = ['sq', 'perm']                       # tipos de estrutura NUNCA vistos

class Ind(nn.Module):
    def __init__(s, d=128, heads=8, layers=5):
        super().__init__()
        s.es = nn.Embedding(N, d); s.ea = nn.Embedding(A, d); s.es2 = nn.Embedding(N + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True); s.tr = nn.TransformerEncoder(enc, layers); s.out = nn.Linear(d, N)
    def forward(s, ss, sa, ss2, qs, qa):
        sup = s.es(ss) + s.ea(sa) + s.es2(ss2)
        q = s.es(qs) + s.ea(qa) + s.es2(torch.full_like(qs, N))
        h = s.tr(torch.cat([sup, q], 1)); return s.out(h[:, sup.shape[1]:])

def gen_batch(fam, B):
    p = GENS[fam](B); s_all, a_all, sup, qry = split(B)
    S = s_all[sup]; Aa = a_all[sup]; S2 = trans(fam, p, S, Aa)
    QS = s_all[qry]; QA = a_all[qry]; QS2 = trans(fam, p, QS, QA)
    return S, Aa, S2, QS, QA, QS2

def run(iters=14000):
    net = Ind().to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-2)
    import random
    for it in range(iters):
        fam = random.choice(TRAIN_FAMS)
        S, Aa, S2, QS, QA, QS2 = gen_batch(fam, 96)
        loss = F.cross_entropy(net(S, Aa, S2, QS, QA).reshape(-1, N), QS2.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 3000 == 0: print(f"  it{it} loss {loss.item():.3f}", flush=True)
    net.eval(); res = {}
    with torch.no_grad():
        for fam in TRAIN_FAMS + HELDOUT:
            acc = []
            for _ in range(30):
                S, Aa, S2, QS, QA, QS2 = gen_batch(fam, 64)
                acc.append((net(S, Aa, S2, QS, QA).argmax(-1) == QS2).float().mean().item())
            res[fam] = sum(acc) / len(acc)
    return res

if __name__ == "__main__":
    print(f"[{DEV}] OUT-OF-FAMILY | treino={TRAIN_FAMS} | held-out={HELDOUT} | chance={1/N:.1%}")
    r = run()
    print("=" * 64)
    print("DENTRO da familia (tipos vistos no treino):")
    for f in TRAIN_FAMS: print(f"  {f:5s} ... {r[f]:.1%}")
    print("FORA da familia (tipos NUNCA vistos) — o teste de AGI:")
    for f in HELDOUT: print(f"  {f:5s} ... {r[f]:.1%}")
    print("=" * 64)
    avg_in = sum(r[f] for f in TRAIN_FAMS) / len(TRAIN_FAMS); avg_out = sum(r[f] for f in HELDOUT) / len(HELDOUT)
    print(f"  MEDIA dentro={avg_in:.1%} | fora={avg_out:.1%} | chance={1/N:.1%}")
    print("  LEITURA: fora ~ chance -> in-context NAO extrapola p/ estrutura nova = o muro real (honesto).")
    print("           fora alto -> generalizacao de estrutura ABERTA (sinal raro e forte).")
