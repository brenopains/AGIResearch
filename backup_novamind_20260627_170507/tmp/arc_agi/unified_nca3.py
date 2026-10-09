"""
unified_nca3.py — base simbolica forte (unified) + TTT-NCA (Fase 1.5) com a receita SOTA da pesquisa:
  - perceive aprendivel + canais ocultos (Xu&Miikkulainen 2025)
  - FIRE-RATE amostrado por rollout em [0.1, 0.75]  (Xu: o achado-chave de estabilidade/generalizacao)
  - LOSS supervisionada nos ULTIMOS passos (nao so no final)
  - treino com augmentacao d8 dos demos
  - MULTI-SEED (3) com early-stop no gate de reproducao dos demos
  - INFERENCE com VOTO por augmentacao d8 (transform -> predict -> inverse-transform -> majority) (TTT/Guichard)
Mede a uniao base+TTT-NCA. Roda nas 400 reais, GPU, sem LLM.
"""
import sys, json, time, pathlib, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE))
import unified as U
from solver import ARC_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SZ, NCOL, HID = 30, 10, 16
CH = NCOL + HID
INV = [0, 3, 2, 1, 4, 5, 6, 7]                         # inverso de cada op d8

def dih(g, k):
    return [g, np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3), g[:, ::-1], g[::-1, :],
            np.swapaxes(g, 0, 1), np.swapaxes(g[:, ::-1], 0, 1)][k]
def to_state(g):
    H, W = g.shape; s = torch.zeros(CH, SZ, SZ)
    for c in range(NCOL): s[c, :H, :W] = torch.tensor(g == c, dtype=torch.float32)
    return s
def to_target(g):
    H, W = g.shape; y = torch.full((SZ, SZ), -100, dtype=torch.long); y[:H, :W] = torch.tensor(g, dtype=torch.long); return y

class NCA3(nn.Module):
    def __init__(s, hidden=128):
        super().__init__()
        s.perceive = nn.Conv2d(CH, hidden, 3, padding=1)
        s.h2 = nn.Conv2d(hidden, hidden, 1)
        s.rule = nn.Conv2d(hidden, CH, 1); nn.init.zeros_(s.rule.weight); nn.init.zeros_(s.rule.bias)
    def step(s, state):
        dx = s.rule(F.relu(s.h2(F.relu(s.perceive(state)))))
        p = random.uniform(0.1, 0.75)                  # fire-rate amostrado por rollout (Xu)
        mask = (torch.rand_like(state[:, :1]) < p).float()
        return state + dx * mask
    def rollout(s, state, T, collect_last=0):
        outs = []
        for t in range(T):
            state = s.step(state)
            if collect_last and t >= T - collect_last: outs.append(state)
        return (outs if collect_last else state)

def _train(train, T, steps, seed, hidden=128):
    torch.manual_seed(seed); random.seed(seed)
    X, Y = [], []
    for i, o in train:
        for k in range(8):
            X.append(to_state(np.ascontiguousarray(dih(i, k)))); Y.append(to_target(np.ascontiguousarray(dih(o, k))))
    X = torch.stack(X).to(DEV); Y = torch.stack(Y).to(DEV)
    net = NCA3(hidden).to(DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3); net.train()
    for it in range(steps):
        states = net.rollout(X.clone(), T, collect_last=3)        # loss nos ultimos 3 passos
        loss = sum(F.cross_entropy(st[:, :NCOL], Y, ignore_index=-100) for st in states) / len(states)
        opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups: g["lr"] = 2e-3 * (1 - 0.7 * it / steps)   # decaimento leve
    net.eval(); return net

@torch.no_grad()
def _pred_canon(net, g, T):
    H, W = g.shape; out = net.rollout(to_state(g).unsqueeze(0).to(DEV), T)
    return out[0, :NCOL, :H, :W].argmax(0).cpu().numpy().astype(np.int64)
@torch.no_grad()
def _pred_vote(net, g, T):
    H, W = g.shape; votes = np.zeros((H, W, NCOL), dtype=np.float32)
    for k in range(8):
        gk = np.ascontiguousarray(dih(g, k)); pk = _pred_canon(net, gk, T)
        pinv = np.ascontiguousarray(dih(pk, INV[k]))               # volta a orientacao original
        if pinv.shape != (H, W): continue
        for v in range(NCOL): votes[:, :, v] += (pinv == v)
    return votes.argmax(2)

def solve_nca3(train, tests, seeds=(0, 1, 2), T=16, steps=300):
    if not train or not all(i.shape == o.shape for i, o in train): return None
    for seed in seeds:
        net = _train(train, T, steps, seed)
        if all(np.array_equal(_pred_canon(net, i, T), o) for i, o in train):   # gate: reproduz os demos
            return [_pred_vote(net, xi, T) for xi, _ in tests]                 # teste com VOTO d8
    return None

if __name__ == "__main__":
    tasks = sorted(ARC_DIR.glob("*.json"))
    sc = by_sym = by_nca = n = 0; t0 = time.time(); nca_tasks = []
    print(f"UNIFIED+TTT-NCA (fire-rate sampled, per-step loss, d8-vote, multi-seed) | {len(tasks)} tasks", flush=True)
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
                try: outs = solve_nca3(train, tests)
                except Exception: outs = None
                if outs is not None and all(pr.shape == yo.shape and np.array_equal(pr, yo) for pr, (_, yo) in zip(outs, tests)):
                    by_nca += 1; sc += 1; nca_tasks.append(p.name[:-5])
        if n % 25 == 0:
            print(f"  {n}/{len(tasks)}  total {sc} (sym {by_sym} + NCA {by_nca})  {sc/n:.1%}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump({"n": n, "solved": sc, "by_sym": by_sym, "by_nca": by_nca, "nca_tasks": nca_tasks},
                      open(HERE / "unified_nca3_progress.json", "w"))
    print(f"\nUNIFIED+TTT-NCA: {sc}/{n} = {sc/n:.1%}  (simbolico {by_sym} + TTT-NCA {by_nca})", flush=True)
    print(f"TTT-NCA resolveu ({by_nca}): {nca_tasks}", flush=True)
