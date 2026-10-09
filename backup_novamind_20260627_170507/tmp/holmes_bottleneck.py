"""
BOTTLENECK DISCRETO do Holmes: a hipotese e que a rede neural acerta ~82% por-celula mas 0% EXATO porque
faz a MEDIA de regras (latente continuo). Forcando a regra por um GARGALO DISCRETO (Gumbel-Softmax sobre um
codebook de K operadores) + commitment loss, a rede COMETE-SE a UMA regra discreta -> pode converter acuracia
celular em acerto EXATO. Teste honesto: roda DISCRETO vs CONTINUO (mesmo budget, mesma amostra same-shape) e
compara quantas ZERAM o erro nos demos e acertam EXATO no teste. Sem LLM. GPU (3050).
"""
import os, json, glob, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; C = 10
random.seed(0); torch.manual_seed(0)

def oh(g): return F.one_hot(torch.tensor(np.array(g), dtype=torch.long), C).permute(2, 0, 1).float()
def dih(g, k):
    a = np.array(g); return [a, np.rot90(a, 1), np.rot90(a, 2), np.rot90(a, 3), a[:, ::-1], a[::-1, :], a.T, np.rot90(a.T, 2)][k]

class Net(nn.Module):
    def __init__(s, discrete, K=32, zd=64, ch=96, iters=3):
        super().__init__(); s.discrete = discrete; s.iters = iters
        if discrete:
            s.codebook = nn.Parameter(torch.randn(K, zd) * 0.1)   # K operadores discretos
            s.sel = nn.Parameter(torch.zeros(K))                  # qual operador (committed por puzzle)
        else:
            s.z = nn.Parameter(torch.zeros(zd))                   # latente continuo (baseline)
        s.enc = nn.Conv2d(C, ch, 3, padding=1); s.film = nn.Linear(zd, 2 * ch)
        s.mid1 = nn.Conv2d(ch, ch, 3, padding=1); s.mid2 = nn.Conv2d(ch, ch, 3, padding=1)
        s.gru = nn.Conv2d(2 * ch, ch, 1); s.dec = nn.Conv2d(ch, C, 1)
    def code(s, tau, hard):
        if not s.discrete: return s.z, None
        w = F.gumbel_softmax(s.sel.unsqueeze(0), tau=tau, hard=hard)   # [1,K] ~ one-hot
        return (w @ s.codebook).squeeze(0), w
    def forward(s, x, tau=1.0, hard=False):
        z, w = s.code(tau, hard); h = F.relu(s.enc(x)); g, b = s.film(z).chunk(2)
        for _ in range(s.iters):
            m = F.relu(s.mid1(h) * (1 + g[None, :, None, None]) + b[None, :, None, None])
            m = F.relu(s.mid2(m)); h = F.relu(s.gru(torch.cat([h, m], 1)))
        return s.dec(h), w

def fit(demos, discrete, iters=400):
    net = Net(discrete).to(DEV); opt = torch.optim.Adam(net.parameters(), 3e-3); net.train()
    for it in range(iters):
        tau = max(0.4, 2.0 * (1 - it / iters))            # anela: forca commitment discreto no fim
        k = random.randrange(8); cp = list(range(1, C)); random.shuffle(cp); lut = np.array([0] + cp)
        loss = 0.0
        for i, o in demos:
            xi = lut[dih(i, k)]; yo = lut[dih(o, k)]
            pred, w = net(oh(xi).unsqueeze(0).to(DEV), tau=tau, hard=False)
            loss = loss + F.cross_entropy(pred, torch.tensor(yo, dtype=torch.long, device=DEV).unsqueeze(0))
        loss = loss / len(demos)
        if discrete and w is not None:                    # commitment: empurra selecao p/ one-hot
            p = F.softmax(net.sel, 0); loss = loss + 0.02 * (-(p * (p + 1e-9).log()).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def predict(net, g):
    with torch.no_grad():
        pr, _ = net(oh(g).unsqueeze(0).to(DEV), tau=0.3, hard=True if net.discrete else False)
        return pr[0].argmax(0).cpu().numpy()

def solves(net, demos, tests):
    for i, o in demos:
        if not np.array_equal(predict(net, i), np.array(o)): return None      # gate: demos exato
    return [predict(net, xi) for xi, _ in tests]

def cell_acc(net, demos):
    tot = cor = 0
    for i, o in demos:
        p = predict(net, i); o = np.array(o); tot += o.size; cor += int((p == o).sum())
    return cor / max(1, tot)

def main():
    t0 = time.time(); ceil = set()
    for f in ["arc_ceiling_v4.json", "arc_unified_final.json", "h111_solved.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: ceil |= set(json.load(open(p)))
            except Exception: pass
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    sample = []
    for fp in files:
        name = os.path.basename(fp)
        if name in ceil: continue                         # so RESIDUAL (nao-resolvido)
        t = json.load(open(fp))
        demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if demos and all(i.shape == o.shape for i, o in demos) and all(i.shape == o.shape for i, o in tests):
            sample.append((name, demos, tests))
    sample = sample[:45]
    print(f"BOTTLENECK DISCRETO (Holmes) vs CONTINUO | amostra={len(sample)} same-shape residuais | {DEV}")
    res = {}
    for tag, disc in [("CONTINUO", False), ("DISCRETO", True)]:
        sc = 0; got = []; accs = []
        for name, demos, tests in sample:
            net = fit(demos, disc); accs.append(cell_acc(net, demos))
            outs = solves(net, demos, tests)
            if outs and all(r.shape == yo.shape and np.array_equal(r, yo) for r, (_, yo) in zip(outs, tests)):
                sc += 1; got.append(name)
        res[tag] = (sc, got, float(np.mean(accs)))
        print(f"  {tag:9s}: EXATO {sc}/{len(sample)} | acuracia-celular media {np.mean(accs):.2f} | {got}")
    print("-" * 74)
    dnew = sorted(set(res['DISCRETO'][1]) - set(res['CONTINUO'][1]))
    print(f"VEREDITO: discreto={res['DISCRETO'][0]} vs continuo={res['CONTINUO'][0]} | discreto pegou que continuo nao: {dnew}")
    print(f"  {'discreto AJUDA (quebra a degeneracao exato=0)' if res['DISCRETO'][0] > res['CONTINUO'][0] else 'discreto NAO superou aqui (honesto)'}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
