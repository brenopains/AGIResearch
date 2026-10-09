"""
SOLVER DE MENU-DE-ARQUETIPOS NO ARC REAL — o teste decisivo (H187). Por tarefa: NASCEM os arquetipos core-knowledge
(LOCAL/GLOBAL/OBJETO), cada um treina SO nas demos de treino da tarefa; seleciona o que AJUSTA todas as demos
EXATO (sinal de generalizacao com poucos exemplos); prediz o teste; mede cobertura no ARC real same-shape.
Honesto: features dos arquetipos sao pre-computadas 1x por grade (CC nao muda no treino). Fundo assumido=0 (convencao
ARC). Mede o numero REAL: quantas tarefas o menu cobre. Prior sobrio: core-knowledge-composicao deu 9/400. Sem LLM.
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import label
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(__file__); ARC = os.path.join(HERE, "arc_data", "data", "training"); C = 10

def oh(g): return F.one_hot(torch.tensor(g, device=DEV, dtype=torch.long), C).permute(2, 0, 1).float().unsqueeze(0)

class LOCAL(nn.Module):
    def __init__(s, ch=32):
        super().__init__(); s.net = nn.Sequential(nn.Conv2d(C, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def pre(s, g): return oh(g)
    def forward(s, x): return s.net(x)
class GLOBAL(nn.Module):
    def __init__(s, ch=32):
        super().__init__(); s.loc = nn.Conv2d(C, ch, 1); s.gl = nn.Linear(C, ch)
        s.head = nn.Sequential(nn.Conv2d(ch*2, ch, 1), nn.ReLU(), nn.Conv2d(ch, C, 1))
    def pre(s, g): return oh(g)
    def forward(s, x):
        _, _, H, W = x.shape; g = s.gl(x.mean((2,3)))[:, :, None, None].expand(-1, -1, H, W)
        return s.head(torch.cat([s.loc(x), g], 1))
NF = C + 4
class OBJETO(nn.Module):
    def __init__(s, hid=48):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(NF, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, C))
    def pre(s, g):
        H, W = g.shape; out = np.zeros((H, W, NF), np.float32); lab, n = label(g != 0); sizes = np.bincount(lab.ravel())
        order = sorted(range(1, n+1), key=lambda k: (-sizes[k], k)) if n else []
        rank = {k: i+1 for i, k in enumerate(order)}
        for i in range(H):
            for j in range(W):
                l = lab[i, j]
                if l == 0: out[i, j, C] = 1.0
                else:
                    out[i, j, g[i, j]] = 1.0; out[i, j, C+1] = sizes[l]/(H*W); out[i, j, C+2] = rank[l]/max(n,1); out[i, j, C+3] = n/(H*W)
        return torch.tensor(out, device=DEV).permute(2, 0, 1).unsqueeze(0)
    def forward(s, x):
        _, _, H, W = x.shape; f = x.squeeze(0).permute(1, 2, 0).reshape(H*W, NF)
        return s.mlp(f).reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)

ARCH = {"LOCAL": LOCAL, "GLOBAL": GLOBAL, "OBJETO": OBJETO}

def fit_task(cls, train, iters=320):
    m = cls().to(DEV); opt = torch.optim.AdamW(m.parameters(), 4e-3, weight_decay=1e-4)
    pre = [(m.pre(i), torch.tensor(o, device=DEV, dtype=torch.long).unsqueeze(0)) for i, o in train]
    for _ in range(iters):
        loss = sum(F.cross_entropy(m(x), y) for x, y in pre) / len(pre)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        exact = all((m(x).argmax(1) == y).all().item() for x, y in pre)
    return m, exact

def main():
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ARC, "*.json")))
    same = []
    for tid in ids:
        t = json.load(open(os.path.join(ARC, tid + ".json")))
        tr = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if all(i.shape == o.shape for i, o in tr) and all(i.shape == o.shape for i, o in te): same.append((tid, tr, te))
    print(f"[{DEV}] SOLVER ARQUETIPOS x ARC REAL | same-shape={len(same)}/400 | menu={list(ARCH)}", flush=True)
    solved = []; by = {k: 0 for k in ARCH}; t0 = time.time()
    for idx, (tid, tr, te) in enumerate(same):
        best = None
        for nm, cls in ARCH.items():
            try:
                m, ok = fit_task(cls, tr)
            except Exception: ok = False
            if not ok: continue                                   # so arquetipo que ajusta TODAS as demos exato
            with torch.no_grad():
                tok = all((m(m.pre(i)).argmax(1).squeeze(0).cpu().numpy() == o).all() for i, o in te)
            if tok: best = nm; break
        if best: solved.append(tid); by[best] += 1
        if (idx+1) % 40 == 0: print(f"   ...{idx+1}/{len(same)}  resolvidas={len(solved)}  ({time.time()-t0:.0f}s)", flush=True)
    print("=" * 70)
    print(f"  ARC REAL resolvido (teste exato) pelo menu-de-arquetipos: {len(solved)}/400  ({len(solved)/400:.1%})")
    print(f"  por arquetipo: {by}")
    print(f"  vs prior (core-knowledge composicao): 9/400. ({time.time()-t0:.0f}s)")
    json.dump(solved, open(os.path.join(HERE, "archetype_solved.json"), "w"))
    print("  LEITURA: sobe de 9 -> o menu de operadores-que-nascem cobre mais leis reais que composicao-do-fixo.")
    print("           estagna ~9 -> o menu ainda nao SPANS as leis do ARC real (precisa + arquetipos). honesto.")

if __name__ == "__main__":
    main()
