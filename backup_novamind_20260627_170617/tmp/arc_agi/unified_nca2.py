"""
unified_nca2.py — base simbolica forte (unified) + NCA FORTALECIDO (Fase 1.5): mais capacidade (hidden 128,
HID 14), mais passos (260), update estocastico, e MULTI-SEED com early-stop no gate de treino (NCA e
seed-dependente -> varias seeds + aceitar a que reproduz os demos sobe o numero de forma confiavel).
Mede a uniao base+NCA2 vs a base e vs o NCA v1 (16).
"""
import sys, json, time, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U
from solver import ARC_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SZ, NCOL, HID = 30, 10, 14
CH = NCOL + HID

def dihedral(g, k):
    return [g, np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3), g[:, ::-1], g[::-1, :],
            np.swapaxes(g, 0, 1), np.swapaxes(g[:, ::-1], 0, 1)][k]
def to_state(g):
    H, W = g.shape; s = torch.zeros(CH, SZ, SZ)
    for c in range(NCOL): s[c, :H, :W] = torch.tensor(g == c, dtype=torch.float32)
    return s
def to_target(g):
    H, W = g.shape; y = torch.full((SZ, SZ), -100, dtype=torch.long); y[:H, :W] = torch.tensor(g, dtype=torch.long); return y

class NCA2(nn.Module):
    def __init__(s, hidden=128):
        super().__init__()
        s.perceive = nn.Conv2d(CH, hidden, 3, padding=1)
        s.h2 = nn.Conv2d(hidden, hidden, 1)
        s.rule = nn.Conv2d(hidden, CH, 1); nn.init.zeros_(s.rule.weight); nn.init.zeros_(s.rule.bias)
    def forward(s, state, T):
        for _ in range(T):
            dx = s.rule(F.relu(s.h2(F.relu(s.perceive(state)))))
            mask = (torch.rand_like(state[:, :1]) < 0.85).float()
            state = state + dx * mask
        return state

def _train_one(train, T, steps, seed):
    torch.manual_seed(seed)
    X, Y = [], []
    for i, o in train:
        for k in range(8):
            X.append(to_state(np.ascontiguousarray(dihedral(i, k)))); Y.append(to_target(np.ascontiguousarray(dihedral(o, k))))
    X = torch.stack(X).to(DEV); Y = torch.stack(Y).to(DEV)
    net = NCA2().to(DEV); opt = torch.optim.Adam(net.parameters(), 2e-3); net.train()
    for _ in range(steps):
        out = net(X.clone(), T); loss = F.cross_entropy(out[:, :NCOL], Y, ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def solve_nca2(train, tests, seeds=(0, 1, 2), T=16, steps=260):
    if not train or not all(i.shape == o.shape for i, o in train): return None
    for seed in seeds:
        net = _train_one(train, T, steps, seed)
        def predict(g):
            with torch.no_grad():
                H, W = g.shape; out = net(to_state(g).unsqueeze(0).to(DEV), T)
                return out[0, :NCOL, :H, :W].argmax(0).cpu().numpy().astype(np.int64)
        if all(np.array_equal(predict(i), o) for i, o in train):      # gate: reproduz TODOS os demos
            return [predict(xi) for xi, _ in tests]
    return None

if __name__ == "__main__":
    tasks = sorted(ARC_DIR.glob("*.json"))
    sc = by_sym = by_nca = n = 0; t0 = time.time(); nca_tasks = []
    print(f"UNIFIED+NCA2 (multi-seed, hidden128, 260 passos) | {len(tasks)} tasks", flush=True)
    for p in tasks:
        task = json.load(open(p)); task["_id"] = p.name; n += 1
        prog = None
        try: prog = U.solve_unified(task)
        except Exception: prog = None
        if prog:
            by_sym += 1; sc += 1
        else:
            train = [(np.array(x["input"]), np.array(x["output"])) for x in task["train"]]
            tests = [(np.array(x["input"]), np.array(x["output"])) for x in task["test"]]
            if train and all(i.shape == o.shape for i, o in train) and all(i.shape == o.shape for i, o in tests):
                try: outs = solve_nca2(train, tests)
                except Exception: outs = None
                if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
                    by_nca += 1; sc += 1; nca_tasks.append(p.name[:-5])
        if n % 25 == 0:
            print(f"  {n}/{len(tasks)}  total {sc} (sym {by_sym} + NCA2 {by_nca})  {sc/n:.1%}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump({"n": n, "solved": sc, "by_sym": by_sym, "by_nca": by_nca, "nca_tasks": nca_tasks},
                      open(HERE / "unified_nca2_progress.json", "w"))
    print(f"\nUNIFIED+NCA2: {sc}/{n} = {sc/n:.1%}  (simbolico {by_sym} + NCA2 {by_nca})", flush=True)
    print(f"NCA2 resolveu ({by_nca}): {nca_tasks}", flush=True)
