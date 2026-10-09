"""
TTT + MODELO DE MUNDO DAS TRANSFORMACOES (a intuicao do usuario: APRENDER os demos, nao so buscar regra).
Holmes provou: treinar do zero em 3 demos = 78% por-celula, 0% EXATO (a rede faz media, nao entende a regra).
A correcao: PRIORS. Um modelo IN-CONTEXT pre-treinado em tarefas sinteticas APRENDE a INFERIR a transformacao
a partir dos pares demo (input->output). Depois, TTT: fine-tuna nos demos REAIS da tarefa antes de prever.
  rule_enc: olha cada par (demo_in, demo_out) -> vetor-regra (o que mudou). media sobre demos = "entendi o exemplo".
  apply: aplica o vetor-regra no test_in (FiLM + refinamento recorrente) -> saida.
Mede EXATO nos same-shape residuais vs o baseline 0/45 (sem priors). Sem LLM. GPU (3050).
"""
import os, json, glob, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "arc_data", "data", "training")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; C = 10
random.seed(0); torch.manual_seed(0); np.random.seed(0)

# ---------------- gerador SINTETICO (os priors: familias de transformacao conhecidas) ----------------
def rand_grid(H, W, ncol):
    g = np.zeros((H, W), int)
    for _ in range(random.randint(2, 5)):
        c = random.randint(1, ncol); y, x = random.randint(0, H-1), random.randint(0, W-1)
        h, w = random.randint(1, max(1, H//2)), random.randint(1, max(1, W//2))
        g[y:y+h, x:x+w] = c
    return g

def comps(g):
    from scipy import ndimage
    out = []
    for c in range(1, 10):
        lbl, n = ndimage.label(g == c)
        for k in range(1, n+1): out.append((c, np.argwhere(lbl == k)))
    return out

def apply_family(g, fam, p):
    o = g.copy()
    if fam == "colormap":
        for a, b in p.items(): o[g == a] = b
    elif fam == "rank_recolor":
        cs = sorted(comps(g), key=lambda z: len(z[1]), reverse=True)
        for i, (c, cells) in enumerate(cs):
            col = p[min(i, len(p)-1)]
            for y, x in cells: o[y, x] = col
    elif fam == "fill_enclosed":
        from scipy import ndimage
        lbl, n = ndimage.label(g == 0)
        border = set(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]).tolist())
        for k in range(1, n+1):
            if k not in border: o[lbl == k] = p
    elif fam == "keep_largest":
        cs = sorted(comps(g), key=lambda z: len(z[1]), reverse=True)
        o = np.zeros_like(g)
        if cs:
            c, cells = cs[0]
            for y, x in cells: o[y, x] = c
    elif fam == "mirror_h":
        o = np.where(g == 0, g[:, ::-1], g)
    elif fam == "gravity_down":
        for x in range(g.shape[1]):
            col = g[:, x]; nz = col[col != 0]; o[:, x] = 0
            if len(nz): o[g.shape[0]-len(nz):, x] = nz
    return o

FAMS = ["colormap", "rank_recolor", "fill_enclosed", "keep_largest", "mirror_h", "gravity_down"]
def gen_task(H=10, W=10):
    fam = random.choice(FAMS); ncol = random.randint(2, 5)
    if fam == "colormap":
        perm = list(range(1, 10)); random.shuffle(perm); p = {a: perm[a-1] for a in range(1, 10)}
    elif fam == "rank_recolor": p = [random.randint(1, 9) for _ in range(5)]
    elif fam == "fill_enclosed": p = random.randint(1, 9)
    else: p = None
    demos = []
    for _ in range(4):
        gi = rand_grid(H, W, ncol); go = apply_family(gi, fam, p)
        demos.append((gi, go))
    return demos[:3], demos[3]

# ---------------- modelo IN-CONTEXT ----------------
def oh(g): return F.one_hot(torch.tensor(np.array(g), dtype=torch.long), C).permute(2, 0, 1).float()

class TTTNet(nn.Module):
    def __init__(s, D=96, ch=96, iters=3):
        super().__init__(); s.iters = iters
        s.renc = nn.Sequential(nn.Conv2d(2*C, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, D, 1))
        s.enc = nn.Conv2d(C, ch, 3, padding=1); s.film = nn.Linear(D, 2*ch)
        s.mid1 = nn.Conv2d(ch, ch, 3, padding=1); s.mid2 = nn.Conv2d(ch, ch, 3, padding=1)
        s.gru = nn.Conv2d(2*ch, ch, 1); s.dec = nn.Conv2d(ch, C, 1)
    def rule(s, demos):                       # estuda os demos -> vetor-regra (o que muda de in p/ out)
        vs = []
        for i, o in demos:
            x = torch.cat([oh(i), oh(o)], 0).unsqueeze(0).to(DEV)
            vs.append(s.renc(x).mean(dim=(2, 3)))    # [1,D]
        return torch.stack(vs, 0).mean(0)            # media sobre demos = "entendi a transformacao"
    def forward(s, test_in, rvec):
        h = F.relu(s.enc(oh(test_in).unsqueeze(0).to(DEV))); g, b = s.film(rvec).chunk(2, dim=1)
        g = g[0]; b = b[0]
        for _ in range(s.iters):
            m = F.relu(s.mid1(h) * (1 + g[None, :, None, None]) + b[None, :, None, None]); m = F.relu(s.mid2(m))
            h = F.relu(s.gru(torch.cat([h, m], 1)))
        return s.dec(h)

def meta_pretrain(net, steps=4000, bs=8):
    opt = torch.optim.Adam(net.parameters(), 1e-3); net.train(); t0 = time.time()
    for it in range(steps):
        loss = 0.0
        for _ in range(bs):
            demos, (ti, to) = gen_task()
            rvec = net.rule(demos); pred = net(ti, rvec)
            loss = loss + F.cross_entropy(pred, torch.tensor(to, dtype=torch.long, device=DEV).unsqueeze(0))
        loss = loss / bs; opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 500 == 0: print(f"  pretrain {it+1}/{steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()

def ttt_predict(net, demos, test_in, steps=40):
    import copy
    local = copy.deepcopy(net); opt = torch.optim.Adam(local.parameters(), 5e-4); local.train()
    for _ in range(steps):                    # TTT: fine-tuna nos demos REAIS (leave-one-out) = presta atencao no treino
        loss = 0.0
        for j in range(len(demos)):
            ctx = [demos[k] for k in range(len(demos)) if k != j] or demos
            i, o = demos[j]; rvec = local.rule(ctx); pred = local(i, rvec)
            loss = loss + F.cross_entropy(pred, torch.tensor(o, dtype=torch.long, device=DEV).unsqueeze(0))
        loss = loss / len(demos); opt.zero_grad(); loss.backward(); opt.step()
    local.eval()
    with torch.no_grad():
        rvec = local.rule(demos)
        # gate: reproduz os demos?
        for i, o in demos:
            if not np.array_equal(local(i, rvec)[0].argmax(0).cpu().numpy(), np.array(o)): return None
        return local(test_in, rvec)[0].argmax(0).cpu().numpy()

def main():
    t0 = time.time(); ckpt = os.path.join(HERE, "ttt_world.pt")
    net = TTTNet().to(DEV)
    print(f"META-PRETRAIN (modelo de mundo das transformacoes) | {DEV}", flush=True)
    meta_pretrain(net, steps=4000)
    torch.save(net.state_dict(), ckpt)
    # sanity: acerta tarefas sinteticas novas?
    sc = 0
    for _ in range(50):
        demos, (ti, to) = gen_task()
        with torch.no_grad():
            r = net(ti, net.rule(demos))[0].argmax(0).cpu().numpy()
        if np.array_equal(r, to): sc += 1
    print(f"  sanity sintetico (in-context, sem TTT): {sc}/50", flush=True)
    # mede nos same-shape residuais REAIS com TTT
    ceil = set()
    for f in ["arc_ceiling_v4.json", "arc_unified_final.json", "h111_solved.json"]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            try: ceil |= set(json.load(open(p)))
            except Exception: pass
    files = sorted(glob.glob(os.path.join(DATA, "*.json"))); sample = []
    for fp in files:
        name = os.path.basename(fp)
        if name in ceil: continue
        t = json.load(open(fp))
        demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if demos and all(i.shape == o.shape for i, o in demos) and all(i.shape == o.shape for i, o in tests):
            sample.append((name, demos, tests))
    sample = sample[:45]
    sc = 0; got = []
    for name, demos, tests in sample:
        try: outs = [ttt_predict(net, demos, xi) for xi, _ in tests]
        except Exception: outs = None
        if outs and all(o is not None and o.shape == yo.shape and np.array_equal(o, yo) for o, (_, yo) in zip(outs, tests)):
            sc += 1; got.append(name)
    print("=" * 74)
    print(f"TTT (pre-treino + fine-tune nos demos) | amostra={len(sample)} same-shape residuais")
    print(f"  EXATO: {sc}/{len(sample)} (baseline sem priors do Holmes = 0/45) | {got}")
    print(f"  {'TTT QUEBROU a degeneracao (priors + atencao aos demos funcionam!)' if sc>0 else 'TTT ainda 0 aqui (honesto)'}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
