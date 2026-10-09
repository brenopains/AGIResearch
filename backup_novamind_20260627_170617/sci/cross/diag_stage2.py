"""diag_stage2.py — SO o STAGE 2 (inducao de regra in-context), enxuto + progresso AO VIVO (flush)."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.grid_cognition import GridWorldModel, GridDomains, C, H, W
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
D = GridDomains(DEV, p=0.45)
B, K, Q = 32, 3, 3
Wt = torch.ones(C, device=DEV); Wt[0] = 0.2
def gen(): return D.episode("geom", B, K, Q)
@torch.no_grad()
def acc(m, n=30):
    ex = []
    for _ in range(n):
        (xs, ys), (xq, yq) = gen(); pred = m(xq, (xs, ys)).argmax(2); tgt = yq.reshape(B, Q, H, W)
        ex.append((pred == tgt).all(-1).all(-1).float().mean().item())
    return sum(ex) / n
m = GridWorldModel(layers=4).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
print(f"[{DEV}] STAGE 2 in-context (geom, 8 regras) — progresso ao vivo:", flush=True)
for it in range(2501):
    (xs, ys), (xq, yq) = gen()
    loss = F.cross_entropy(m(xq, (xs, ys)).reshape(B * Q, C, H, W), yq.reshape(B * Q, H, W), weight=Wt)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        print(f"  it{it:4d}  loss {loss.item():.3f}  EXATO {acc(m):.1%}", flush=True)
print(f"\nRESULTADO STAGE 2 (in-context): EXATO {acc(m, 60):.1%}  {'-> INDUCAO FUNCIONA' if acc(m,40)>0.5 else '-> ainda nao'}", flush=True)
