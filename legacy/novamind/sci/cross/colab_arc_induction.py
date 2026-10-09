"""
colab_arc_induction.py — PESQUISA: cracar a INDUCAO-DE-REGRA few-shot (o muro do ARC).
Self-contained (cola no Colab T4). HIPOTESE-CHAVE: o diagnostico (sci/cross/diag_grid.py) mostrou que a
inducao trava com grades de RUIDO DENSO (correspondencia in->out ambigua). ARC real usa OBJETOS de cores
DISTINTAS. Com objetos distintos, o modelo rastreia p/ onde cada cor foi -> a transformacao fica inferivel.
Este script testa exatamente isso: mesma arquitetura in-context, mas grades OBJECT-BASED.

Uso local rapido:  python colab_arc_induction.py
Uso Colab (T4):    cole tudo; aumente STEPS e B no topo (a T4 aguenta bem mais que a 3050).
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
H = W = 8; HW = 64; C = 10                      # 10 cores -> ate 9 objetos de cores DISTINTAS
B, K, Q, STEPS = 32, 3, 3, 1500                 # Colab: suba STEPS p/ 8000+ e B p/ 64

# ---------- gerador OBJECT-BASED: poucos objetos de cores distintas, transformacao geometrica ----------
def _base(): return torch.arange(HW).reshape(H, W)
def geom_maps():
    b = _base(); ms = [b, torch.rot90(b,1), torch.rot90(b,2), torch.rot90(b,3),
                       torch.flip(b,[0]), torch.flip(b,[1]), b.t().contiguous(), torch.flip(b.t().contiguous(),[0])]
    return torch.stack([m.reshape(-1).long() for m in ms]).to(DEV)
GMAPS = geom_maps()
def rand_objects(B, M, nmin=3, nmax=6):
    """cada amostra: nobj celulas de cores DISTINTAS (1..C-1) em posicoes aleatorias. fundo=0."""
    g = torch.zeros(B, M, HW, dtype=torch.long, device=DEV)
    for b in range(B):
        for m in range(M):
            nobj = torch.randint(nmin, nmax + 1, (1,)).item()
            pos = torch.randperm(HW, device=DEV)[:nobj]
            cols = torch.randperm(C - 1, device=DEV)[:nobj] + 1     # cores distintas
            g[b, m, pos] = cols
    return g
def episode():
    x = rand_objects(B, K + Q)
    rule = torch.randint(0, GMAPS.shape[0], (B,), device=DEV)       # 1 transformacao por episodio
    sel = GMAPS[rule]                                               # [B,HW]
    y = torch.gather(x, 2, sel[:, None, :].expand(-1, K + Q, -1))
    return (x[:, :K], y[:, :K]), (x[:, K:], y[:, K:])

# ---------- modelo IN-CONTEXT (mesma arquitetura corrigida do diagnostico) ----------
class GridIC(nn.Module):
    def __init__(s, d=64, heads=4, layers=4):
        super().__init__()
        s.col = nn.Embedding(C, d); s.pos = nn.Embedding(HW, d); s.role = nn.Embedding(3, d)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d*4, batch_first=True, dropout=0.0), layers)
        s.head = nn.Linear(d, C); s.d = d
    def tok(s, g, role):
        return s.col(g) + s.pos(torch.arange(HW, device=g.device))[None] + s.role.weight[role][None, None]
    def forward(s, xq, sup):
        xs, ys = sup; B_, Kk = xs.shape[0], xs.shape[1]; Q_ = xq.shape[1]
        sin = s.tok(xs.reshape(B_*Kk, HW), 1).reshape(B_, Kk*HW, s.d)
        sout = s.tok(ys.reshape(B_*Kk, HW), 2).reshape(B_, Kk*HW, s.d)
        sup_tok = torch.cat([sin, sout], 1)
        sup_exp = sup_tok[:, None].expand(-1, Q_, -1, -1).reshape(B_*Q_, Kk*HW*2, s.d)
        qtok = s.tok(xq.reshape(B_*Q_, HW), 0)
        h = s.tf(torch.cat([sup_exp, qtok], 1))[:, -HW:, :]
        return s.head(h).reshape(B_, Q_, H, W, C).permute(0, 1, 4, 2, 3)

# NOTA (achado local 3050): Wt[0]=0.1 -> objeto 87% mas EXATO 0 (super-prediz fundo->falsos-positivos);
# Wt[0]=0.5 -> objeto 9% (sub-prediz). O ponto otimo e no meio E precisa de ESCALA (steps/B) p/ EXATO subir.
# Colab: faca varredura de Wt[0] em {0.2,0.3} x STEPS {8000,16000} x B {64,128}. O sinal-chave (objeto destravou
# de 55% no ruido denso p/ 87% com objetos) ja valida que a inducao APRENDE a transformacao com dados object-based.
Wt = torch.ones(C, device=DEV); Wt[0] = 0.25
m = GridIC().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)

@torch.no_grad()
def ev(n=30):
    ex, fg = [], []
    for _ in range(n):
        (xs, ys), (xq, yq) = episode(); pred = m(xq, (xs, ys)).argmax(2); tgt = yq.reshape(B, Q, H, W)
        ex.append((pred == tgt).all(-1).all(-1).float().mean().item())
        msk = tgt > 0; fg.append(((pred == tgt) & msk).sum().item() / max(1, msk.sum().item()))
    return sum(ex)/n, sum(fg)/n

print(f"[{DEV}] INDUCAO few-shot com grades OBJECT-BASED (cores distintas):", flush=True)
for it in range(STEPS + 1):
    (xs, ys), (xq, yq) = episode()
    loss = F.cross_entropy(m(xq, (xs, ys)).reshape(B*Q, C, H, W), yq.reshape(B*Q, H, W), weight=Wt)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 300 == 0:
        e, f = ev(); print(f"  it{it:5d}  loss {loss.item():.3f}  EXATO {e:.1%}  objeto {f:.1%}", flush=True)
e, f = ev(60)
print(f"\nRESULTADO: EXATO {e:.1%} | objeto {f:.1%}  -> {'INDUCAO DESTRAVOU (hipotese object-based vale!)' if e>0.4 else ('progresso parcial' if e>0.1 else 'ainda travado')}", flush=True)
