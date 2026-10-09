"""
dmt_nca.py — "DMT" / REORGANIZACAO DE PLASTICIDADE (ideia do usuario, base neuro real: psicodelicos aumentam a
plasticidade p/ escapar de padroes rigidos). Quando o NCA TRAVA (loss em plato = "receptores desregulados",
colapsado no fundo), dispara um SURTO DE PLASTICIDADE DIRIGIDO: perturba os pesos + boost de learning-rate,
reorganizando a rede p/ escapar do mode-collapse local. Testa: o DMT acende tarefas que o NCA normal nao pega?
"""
import sys, json, time, pathlib, numpy as np, torch, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified_nca3 as N
from solver import ARC_DIR
torch.manual_seed(0)

def train(demos, steps=420, T=16, dmt=True, seed=0):
    torch.manual_seed(seed)
    net = N.NCA3().to(N.DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3)
    X, Y = [], []
    for i, o in demos:
        for k in range(8):
            X.append(N.to_state(np.ascontiguousarray(N.dih(i, k)))); Y.append(N.to_target(np.ascontiguousarray(N.dih(o, k))))
    X = torch.stack(X).to(N.DEV); Y = torch.stack(Y).to(N.DEV); net.train(); hist = []; bursts = 0
    for it in range(steps):
        outs = net.rollout(X.clone(), T, collect_last=3)
        loss = sum(F.cross_entropy(s[:, :N.NCOL], Y, ignore_index=-100) for s in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
        if dmt and it >= 90 and it % 45 == 0:
            window = hist[-45:]
            if window[-1] > 0.93 * window[0]:                       # TRAVOU (plato) -> SURTO DE PLASTICIDADE
                with torch.no_grad():
                    for p in net.parameters():
                        p.add_(torch.randn_like(p) * 0.06 * (p.detach().std() + 1e-4))   # perturba/reorganiza
                for gpar in opt.param_groups: gpar["lr"] = 6e-3      # boost de plasticidade
                bursts += 1
            else:
                for gpar in opt.param_groups: gpar["lr"] = 2e-3      # estabiliza qd progride
    net.eval(); return net, bursts

def solve(demos, tests, dmt, seeds=(0, 1)):
    if not demos or not all(i.shape == o.shape for i, o in demos): return None
    for seed in seeds:
        net, _ = train(demos, dmt=dmt, seed=seed)
        if all(np.array_equal(N._pred_canon(net, i, 16), o) for i, o in demos):
            return [N._pred_vote(net, xi, 16) for xi, _ in tests]
    return None

if __name__ == "__main__":
    allt = sorted(p.name[:-5] for p in ARC_DIR.glob("*.json")); same = []
    for tid in allt:
        t = json.load(open(ARC_DIR / (tid + ".json")))
        d = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
        te = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
        if d and all(i.shape == o.shape for i, o in d + te): same.append(tid)
    sl = same[:40]; print(f"DMT-NCA vs NCA normal em {len(sl)} same-shape", flush=True); t0 = time.time()
    def run(dmt, tag):
        sc = 0; got = []
        for tid in sl:
            t = json.load(open(ARC_DIR / (tid + ".json")))
            demos = [(np.array(p["input"]), np.array(p["output"])) for p in t["train"]]
            tests = [(np.array(p["input"]), np.array(p["output"])) for p in t["test"]]
            try: outs = solve(demos, tests, dmt)
            except Exception: outs = None
            if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
                sc += 1; got.append(tid)
        print(f"  {tag}: {sc}/{len(sl)} -> {got}  ({time.time()-t0:.0f}s)", flush=True); return sc, set(got)
    n0, g0 = run(False, "NCA normal")
    n1, g1 = run(True, "DMT-NCA (plasticidade)")
    novas = sorted(g1 - g0)
    print(f"\n=== DMT (reorganizacao de plasticidade): normal {n0} | DMT {n1} | novas pelo DMT: {len(novas)} {novas} ===", flush=True)
    print(f"  DMT {'AJUDA a escapar do mode-collapse' if n1 > n0 else 'nao ajudou aqui'}", flush=True)
