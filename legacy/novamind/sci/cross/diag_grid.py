"""
diag_grid.py — DIAGNOSTICO METODICO (escada): antes de testar a tese da largura, PROVAR os pre-requisitos.
  STAGE 1: o modelo aprende UMA transformacao FIXA (sem variacao, sem precisar de contexto) a >90% EXATO?
           -> isola CAPACIDADE. grades DENSAS + peso na loss (mata o minimo 'preveja fundo').
  STAGE 2: aprende uma familia COM variacao de regra (precisa inferir do suporte) a alto EXATO?
           -> isola INDUCAO-DE-REGRA few-shot.
Se 1 passa e 2 nao, o gargalo e a inducao; se 1 falha, e capacidade/desbalanceamento.
"""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.grid_cognition import GridWorldModel, GridDomains, geom_maps, C, H, W, HW

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D = GridDomains(DEV, p=0.45)                       # grades DENSAS (~45% coloridas) -> sem minimo 'tudo fundo'
B, K, Q = 32, 3, 3
Wt = torch.ones(C, device=DEV); Wt[0] = 0.2        # peso: desvaloriza o fundo

def lossf(logits, tgt):
    return F.cross_entropy(logits.reshape(B * Q, C, H, W), tgt.reshape(B * Q, H, W), weight=Wt)

@torch.no_grad()
def acc(model, gen, n=40):
    cell, exact, fg = [], [], []
    for _ in range(n):
        (xs, ys), (xq, yq) = gen()
        pred = model(xq, (xs, ys)).argmax(2); tgt = yq.reshape(B, Q, H, W)
        cell.append((pred == tgt).float().mean().item())
        exact.append((pred == tgt).all(-1).all(-1).float().mean().item())
        m = tgt > 0
        fg.append(((pred == tgt) & m).sum().item() / max(1, m.sum().item()))
    return sum(cell) / n, sum(exact) / n, sum(fg) / n

def train(model, gen, steps, tag):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    print(f"[{DEV}] {tag} ...")
    for it in range(steps):
        (xs, ys), (xq, yq) = gen()
        loss = lossf(model(xq, (xs, ys)), yq)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 800 == 0: print(f"  it{it} loss {loss.item():.3f}")

# ---------- STAGE 1: transformacao FIXA (fliplr) ----------
FMAP = geom_maps()[5].to(DEV)                       # idx 5 = espelho horizontal (fixo)
def gen_fixed():
    x = D.rand_in(B, K + Q); y = torch.gather(x, 2, FMAP[None, None, :].expand(B, K + Q, -1))
    return (x[:, :K], y[:, :K]), (x[:, K:], y[:, K:])
m1 = GridWorldModel(layers=4).to(DEV)
train(m1, gen_fixed, 2500, "STAGE 1 (transformacao FIXA: espelho-h)")
c1, e1, f1 = acc(m1, gen_fixed)
print(f"  -> celula {c1:.1%} | EXATO {e1:.1%} | acerto no objeto {f1:.1%}")

# ---------- STAGE 2: familia COM variacao (geom, 8 regras) -> precisa inferir do suporte ----------
def gen_geom():
    return D.episode("geom", B, K, Q)
m2 = GridWorldModel(layers=4).to(DEV)
train(m2, gen_geom, 4000, "STAGE 2 (familia geom: 8 regras, infere do suporte)")
c2, e2, f2 = acc(m2, gen_geom)
print(f"  -> celula {c2:.1%} | EXATO {e2:.1%} | acerto no objeto {f2:.1%}")

print("\n" + "=" * 60 + "\nDIAGNOSTICO:")
print(f"  STAGE 1 (capacidade, regra fixa) ...... EXATO {e1:.1%}  {'OK (>90%)' if e1 > 0.9 else 'FALHOU'}")
print(f"  STAGE 2 (inducao-de-regra few-shot) ... EXATO {e2:.1%}  {'OK (>50%)' if e2 > 0.5 else 'FALHOU'}")
if e1 > 0.9 and e2 > 0.5:
    print("  => arquitetura aprende E induz regra. Pronto p/ o teste de largura.")
elif e1 > 0.9:
    print("  => CAPACIDADE ok; gargalo e a INDUCAO-DE-REGRA few-shot (proximo: + suporte / espaco menor / + passos).")
else:
    print("  => CAPACIDADE falha mesmo com regra fixa: problema mais fundo (arquitetura/representacao).")
