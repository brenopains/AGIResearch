"""
EXPERIMENT 1 — Neuroplasticity / Catastrophic Forgetting
========================================================

The core of "plasticity" the user asked about: can a network learn something
new WITHOUT erasing what it already knew? This is one of the genuine open walls
between current AI and AGI. A human who learns to ride a bike doesn't forget how
to walk. A naive neural net does exactly that.

Setup (fully self-contained, no downloads, runs in seconds on CPU or the 3050):
  - Synthetic K-class classification, D features (Gaussian clusters).
  - Task A: original features.  Task B: a FIXED random permutation of features
    (a genuinely different task that competes for the same weights).
  - Phase 1: train on A, measure A accuracy.
  - Phase 2: train on B, then RE-measure A accuracy.
       NAIVE  -> A collapses (catastrophic forgetting): the wall.
       REPLAY -> mix a small buffer of old A samples into B training and re-test.
  - Metric: % of Task-A accuracy retained after learning Task B.

This does NOT claim AGI. It isolates ONE capability (plasticity) and measures
honestly whether a cheap technique recovers it.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D, K = 20, 10           # features, classes
N_TRAIN, N_TEST = 4000, 1000
EPOCHS = 15
REPLAY_BUFFER = 400     # how many Task-A samples we keep for rehearsal


def make_task(permutation=None, n=N_TRAIN):
    """Gaussian-cluster classification. Optional fixed feature permutation.
    Centers scaled down -> overlapping classes -> more weight interference,
    which is where catastrophic forgetting actually bites."""
    centers = np.random.RandomState(42).randn(K, D) * 1.0
    y = np.random.randint(0, K, size=n)
    x = centers[y] + np.random.randn(n, D)
    if permutation is not None:
        x = x[:, permutation]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, K),
        )
    def forward(self, x):
        return self.net(x)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(DEVICE)).argmax(1).cpu()
    return (pred == y).float().mean().item()


def train(model, x, y, epochs=EPOCHS, extra=None):
    """Train on (x,y). If `extra` (xb,yb) given, mix it in each step (replay)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x, y = x.to(DEVICE), y.to(DEVICE)
    if extra is not None:
        xb, yb = extra[0].to(DEVICE), extra[1].to(DEVICE)
    bs = 128
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            idx = perm[i:i+bs]
            bx, by = x[idx], y[idx]
            if extra is not None:
                ridx = torch.randint(0, len(xb), (min(bs//4, len(xb)),), device=DEVICE)
                bx = torch.cat([bx, xb[ridx]]); by = torch.cat([by, yb[ridx]])
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()


def run():
    perm = np.random.permutation(D)
    Ax, Ay = make_task(None);        Atx, Aty = make_task(None, N_TEST)
    Bx, By = make_task(perm);        Btx, Bty = make_task(perm, N_TEST)

    print(f"Device: {DEVICE}\n{'='*64}")

    # ---- NAIVE: learn A, then learn B with no protection ----
    m = MLP().to(DEVICE)
    train(m, Ax, Ay)
    a_before = accuracy(m, Atx, Aty)
    train(m, Bx, By)
    a_after_naive = accuracy(m, Atx, Aty)
    b_naive = accuracy(m, Btx, Bty)

    # ---- REPLAY: same, but rehearse a small buffer of A while learning B ----
    m2 = MLP().to(DEVICE)
    train(m2, Ax, Ay)
    a_before2 = accuracy(m2, Atx, Aty)
    buf = (Ax[:REPLAY_BUFFER], Ay[:REPLAY_BUFFER])
    train(m2, Bx, By, extra=buf)
    a_after_replay = accuracy(m2, Atx, Aty)
    b_replay = accuracy(m2, Btx, Bty)

    print(f"NAIVE  | A before B: {a_before:.0%}  -> A after B: {a_after_naive:.0%}"
          f"  (B: {b_naive:.0%})  retained A: {a_after_naive/a_before:.0%}")
    print(f"REPLAY | A before B: {a_before2:.0%}  -> A after B: {a_after_replay:.0%}"
          f"  (B: {b_replay:.0%})  retained A: {a_after_replay/a_before2:.0%}")
    print(f"{'='*64}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU run")


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f"time: {time.time()-t0:.1f}s")
