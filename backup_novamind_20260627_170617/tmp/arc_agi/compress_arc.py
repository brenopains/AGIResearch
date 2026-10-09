"""
compress_arc.py — MINIMAL CompressARC (MDL/compressao): por-puzzle, SEM primitiva minha, SEM busca, SEM pretrain.
Princípio (Occam/Solomonoff): a regra mais SIMPLES que explica os demos generaliza. Um net minusculo aprende,
por gradiente no proprio puzzle, a mapear input->output com um LATENTE-REGRA pequeno (gargalo = compressao) +
equivariancias (cor permutavel + d8) como prior. Sem DSL -> escapa do fecho. Testa sinal numa fatia same-shape.
"""
import sys, json, time, pathlib, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
from solver import ARC_DIR
random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 10

def oh(g):  # [H,W] long -> [C,H,W]
    return F.one_hot(torch.tensor(g, dtype=torch.long), C).permute(2, 0, 1).float()

class RuleNet(nn.Module):
    """f(z_regra, input) -> output. z e o 'codigo' compartilhado (gargalo MDL). Conv equivariante a translacao.
    GPU + maior capacidade + REFINAMENTO RECORRENTE (introspeccao): re-aplica o nucleo K vezes refinando a regra
    (profundidade por recursao, estilo TRM/ponto-fixo da Ada) sem inflar parametros p/ overfitar nos 3-4 exemplos."""
    def __init__(s, zd=64, ch=96, iters=3):
        super().__init__()
        s.iters = iters
        s.z = nn.Parameter(torch.zeros(zd))                 # o LATENTE-REGRA (otimizado por puzzle)
        s.enc = nn.Conv2d(C, ch, 3, padding=1)
        s.film = nn.Linear(zd, 2 * ch)                       # condiciona nas features pela regra
        s.mid1 = nn.Conv2d(ch, ch, 3, padding=1)
        s.mid2 = nn.Conv2d(ch, ch, 3, padding=1)            # camada extra (mais abstracao)
        s.gru = nn.Conv2d(2 * ch, ch, 1)                    # integra o estado refinado (introspeccao/ponto-fixo)
        s.dec = nn.Conv2d(ch, C, 1)
    def forward(s, x):                                       # x:[B,C,H,W]
        h = F.relu(s.enc(x)); g, b = s.film(s.z).chunk(2)
        for _ in range(s.iters):                            # laco de refinamento: pensa de novo sobre a regra
            m = F.relu(s.mid1(h) * (1 + g[None, :, None, None]) + b[None, :, None, None])
            m = F.relu(s.mid2(m))
            h = F.relu(s.gru(torch.cat([h, m], 1)))         # integra o novo pensamento ao estado
        return s.dec(h)

def dih(g, k):
    a = np.array(g)
    return [a, np.rot90(a, 1), np.rot90(a, 2), np.rot90(a, 3), a[:, ::-1], a[::-1, :], a.T, np.rot90(a.T, 2)][k]

def solve(demos, test_in, iters=600):
    if not all(np.array(i).shape == np.array(o).shape for i, o in demos): return None
    net = RuleNet().to(DEV); opt = torch.optim.Adam(net.parameters(), 3e-3)
    net.train()
    for it in range(iters):
        k = random.randrange(8); cp = list(range(1, C)); random.shuffle(cp); lut = np.array([0] + cp)  # cor permutavel
        loss = 0.0
        for i, o in demos:
            xi = lut[dih(i, k)]; yo = lut[dih(o, k)]
            pred = net(oh(xi).unsqueeze(0).to(DEV))
            loss = loss + F.cross_entropy(pred, torch.tensor(yo, dtype=torch.long, device=DEV).unsqueeze(0))
        loss = loss / len(demos) + 1e-3 * (net.z ** 2).sum()                # + custo do codigo (MDL)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        # gate: reproduz os demos exato (na orientacao/cor original)?
        for i, o in demos:
            p = net(oh(i).unsqueeze(0).to(DEV))[0].argmax(0).cpu().numpy()
            if not np.array_equal(p, np.array(o)): return None
        return net(oh(test_in).unsqueeze(0).to(DEV))[0].argmax(0).cpu().numpy()

if __name__ == "__main__":
    allt = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json"))
    same = []
    for tid in allt:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        d = [(p["input"], p["output"]) for p in t["train"]]; te = [(p["input"], p["output"]) for p in t["test"]]
        if d and all(np.array(i).shape == np.array(o).shape for i, o in d + te): same.append(tid)
    sl = same[:40]
    base = set(json.load(open(HERE / "unified_solved.json"))) if (HERE / "unified_solved.json").exists() else set()
    print(f"CompressARC minimal (MDL, sem primitivas) em {len(sl)} same-shape", flush=True)
    sc = 0; got = []; novas = []; t0 = time.time()
    for i, tid in enumerate(sl):
        t = json.load(open(ARC_DIR / (tid + ".json")))
        demos = [(p["input"], p["output"]) for p in t["train"]]
        tin = t["test"][0]["input"]; tout = np.array(t["test"][0]["output"])
        try: pred = solve(demos, tin)
        except Exception: pred = None
        if pred is not None and pred.shape == tout.shape and np.array_equal(pred, tout):
            sc += 1; got.append(tid)
            if tid not in base: novas.append(tid)
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(sl)}  resolvidas {sc}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nCompressARC minimal: {sc}/{len(sl)} same-shape | NOVAS (simbolico nao pega): {len(novas)} {novas}", flush=True)
    print(f"  resolvidas: {got}", flush=True)
