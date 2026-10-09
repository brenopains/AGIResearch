"""
EXPERIMENT 2 — Compositional Generalization (memorize vs. generalize)
=====================================================================

"Abstraction / generalization" the user asked about. The real test of
understanding: can the system handle COMBINATIONS it never saw, by reusing
parts it does know? A child who knows "jump" and "twice" understands "jump
twice" without being taught that exact phrase. This is the SCAN problem.

Task: input = (verb, modifier), e.g. ("walk", "twice").
      output = action + count, e.g. (WALK, 2).
We hold out specific (verb, modifier) COMBINATIONS from training entirely,
then test on those unseen combinations.

We compare TWO models on the SAME data, to show the honest lesson — that
generalization is not magic, it comes from STRUCTURE:

  ENTANGLED : one softmax over all (verb,modifier) classes.
              An unseen combo is a label it never saw -> cannot predict it.
  FACTORED  : two heads (action from verb, count from modifier).
              Decouples the parts -> can recombine them -> generalizes.

Fully self-contained, runs in <5s on CPU/GPU.
"""
import sys, itertools
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VERBS = ["jump", "walk", "run", "look", "turn"]
MODS  = ["once", "twice", "thrice", "fourtimes"]
COUNT = {"once": 1, "twice": 2, "thrice": 3, "fourtimes": 4}

ALL = list(itertools.product(range(len(VERBS)), range(len(MODS))))
# Hold out a diagonal of combinations so every verb & every modifier is still
# seen somewhere in training, just never in these exact pairings.
HELDOUT = {(0, 1), (1, 2), (2, 3), (3, 0), (4, 1)}
TRAIN = [c for c in ALL if c not in HELDOUT]
TEST  = [c for c in ALL if c in HELDOUT]


def encode(v, m):
    x = np.zeros(len(VERBS) + len(MODS), dtype=np.float32)
    x[v] = 1; x[len(VERBS) + m] = 1
    return x


def make(combos, reps=200):
    X, yv, ym, ycombo = [], [], [], []
    for (v, m) in combos:
        for _ in range(reps):
            X.append(encode(v, m)); yv.append(v); ym.append(m)
            ycombo.append(v * len(MODS) + m)
    t = lambda a, dt: torch.tensor(np.array(a), dtype=dt)
    return (t(X, torch.float32), t(yv, torch.long), t(ym, torch.long), t(ycombo, torch.long))


class Entangled(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(VERBS)+len(MODS), 64), nn.ReLU(),
                                 nn.Linear(64, len(VERBS)*len(MODS)))
    def forward(self, x): return self.net(x)


class Factored(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(len(VERBS)+len(MODS), 64), nn.ReLU())
        self.action_head = nn.Linear(64, len(VERBS))
        self.count_head  = nn.Linear(64, len(MODS))
    def forward(self, x):
        h = self.trunk(x)
        return self.action_head(h), self.count_head(h)


def train_entangled(Xtr, ytr):
    m = Entangled().to(DEVICE); opt = torch.optim.Adam(m.parameters(), 1e-2)
    lf = nn.CrossEntropyLoss(); X, y = Xtr.to(DEVICE), ytr.to(DEVICE)
    for _ in range(300):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m


def train_factored(Xtr, yv, ym):
    m = Factored().to(DEVICE); opt = torch.optim.Adam(m.parameters(), 1e-2)
    lf = nn.CrossEntropyLoss()
    X, a, c = Xtr.to(DEVICE), yv.to(DEVICE), ym.to(DEVICE)
    for _ in range(300):
        opt.zero_grad(); pa, pc = m(X)
        (lf(pa, a) + lf(pc, c)).backward(); opt.step()
    return m


def run():
    Xtr, yv, ym, yc = make(TRAIN)
    Xte, yv_te, ym_te, yc_te = make(TEST, reps=1)
    print(f"Device: {DEVICE}\nTrain combos: {len(TRAIN)}  | Held-out (never seen): {sorted(TEST)}\n{'='*64}")

    # ENTANGLED
    me = train_entangled(Xtr, yc)
    with torch.no_grad():
        pred = me(Xte.to(DEVICE)).argmax(1).cpu()
    ent_acc = (pred == yc_te).float().mean().item()

    # FACTORED
    mf = train_factored(Xtr, yv, ym)
    with torch.no_grad():
        pa, pc = mf(Xte.to(DEVICE))
        ok = (pa.argmax(1).cpu() == yv_te) & (pc.argmax(1).cpu() == ym_te)
    fac_acc = ok.float().mean().item()

    print(f"ENTANGLED (single label) | accuracy on UNSEEN combos: {ent_acc:.0%}  -> memorizes, cannot recombine")
    print(f"FACTORED  (2 heads)      | accuracy on UNSEEN combos: {fac_acc:.0%}  -> generalizes by reusing parts")
    print(f"{'='*64}\nLesson: generalization came from STRUCTURE, not scale or brain-naming.")


if __name__ == '__main__':
    run()
