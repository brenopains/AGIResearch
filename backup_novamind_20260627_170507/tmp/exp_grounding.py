"""
EXPERIMENT 3 — Multimodal Grounding (image <-> text in a shared space)
======================================================================

The capability the user intuited as central to a "world model": associating an
IMAGE with a TEXT in a shared meaning space — and, crucially, generalizing that
association to combinations never seen together.

This is a tiny, self-contained CLIP-style contrastive setup (no downloads):
  - Synthetic images: a colored shape on a small RGB grid.
        colors = {red, green, blue}, shapes = {circle, square, triangle} -> 9 concepts
  - Text: factored tokens [color one-hot | shape one-hot].
  - Two encoders (image CNN-ish, text MLP) map both into a 32-d shared space.
  - Trained contrastively (InfoNCE) to pull matching image/text together.

THE TEST (zero-shot compositional grounding):
  We hold out 3 (color,shape) COMBINATIONS from training entirely. Every color
  and every shape is still seen — just never in those pairings. Then we ask:
  given a held-out image, does its embedding retrieve the CORRECT unseen text
  among all 9 candidates? Chance = 1/9 ~ 11%.

Honest by design: if it beats chance, the shared space captured compositional
structure (real grounding signal). If not, we report that too.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
SHAPES = ["circle", "square", "triangle"]
CN, SN = list(COLORS), SHAPES
G = 12  # grid size


def shape_mask(shape):
    m = np.zeros((G, G), np.float32)
    cy, cx = G / 2 - 0.5, G / 2 - 0.5
    for y in range(G):
        for x in range(G):
            if shape == "circle":
                if (y - cy) ** 2 + (x - cx) ** 2 <= (G * 0.38) ** 2:
                    m[y, x] = 1
            elif shape == "square":
                if 2 <= y < G - 2 and 2 <= x < G - 2:
                    m[y, x] = 1
            elif shape == "triangle":
                if y >= 2 and abs(x - cx) <= (y - 2) * 0.5:
                    m[y, x] = 1
    return m


def render(color, shape, noise=0.06):
    m = shape_mask(shape)
    img = np.stack([m * c for c in COLORS[color]], 0)            # (3,G,G)
    img += np.random.randn(*img.shape).astype(np.float32) * noise
    return img


def text_vec(ci, si):
    v = np.zeros(len(CN) + len(SN), np.float32)
    v[ci] = 1; v[len(CN) + si] = 1
    return v


import itertools
ALL = list(itertools.product(range(len(CN)), range(len(SN))))
HELDOUT = {(0, 2), (1, 1), (2, 0)}     # red-triangle, green-square, blue-circle
TRAIN = [c for c in ALL if c not in HELDOUT]
TEST  = [c for c in ALL if c in HELDOUT]


class ImgEnc(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(16 * 16, d))
    def forward(self, x): return F.normalize(self.c(x), dim=1)


class TxtEnc(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(len(CN)+len(SN), 64), nn.ReLU(), nn.Linear(64, d))
    def forward(self, x): return F.normalize(self.m(x), dim=1)


def batch(combos, reps):
    imgs, txts = [], []
    for (ci, si) in combos:
        for _ in range(reps):
            imgs.append(render(CN[ci], SHAPES[si]))
            txts.append(text_vec(ci, si))
    return (torch.tensor(np.array(imgs)).to(DEVICE),
            torch.tensor(np.array(txts)).to(DEVICE))


def run():
    ie, te = ImgEnc().to(DEVICE), TxtEnc().to(DEVICE)
    opt = torch.optim.Adam(list(ie.parameters()) + list(te.parameters()), 3e-3)
    tau = 0.07
    print(f"Device: {DEVICE}\nHeld-out combos (never trained): "
          f"{[(CN[c],SHAPES[s]) for c,s in sorted(TEST)]}\n{'='*64}")

    for ep in range(400):
        imgs, txts = batch(TRAIN, reps=8)
        zi, zt = ie(imgs), te(txts)
        logits = zi @ zt.t() / tau
        labels = torch.arange(len(imgs), device=DEVICE)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- Zero-shot test: held-out images vs ALL 9 candidate texts ----
    ie.eval(); te.eval()
    with torch.no_grad():
        cand = torch.tensor(np.array([text_vec(c, s) for (c, s) in ALL])).to(DEVICE)
        zt_all = te(cand)                               # (9, d)
        # train-combo accuracy (sanity)
        ti, _ = batch(TRAIN, reps=20); zti = ie(ti)
        sims = zti @ zt_all.t()
        train_lbl = torch.tensor([ALL.index(c) for c in TRAIN for _ in range(20)], device=DEVICE)
        train_acc = (sims.argmax(1) == train_lbl).float().mean().item()
        # held-out (zero-shot) accuracy
        hi, _ = batch(TEST, reps=50); zhi = ie(hi)
        sims_h = zhi @ zt_all.t()
        held_lbl = torch.tensor([ALL.index(c) for c in TEST for _ in range(50)], device=DEVICE)
        zs_acc = (sims_h.argmax(1) == held_lbl).float().mean().item()

    print(f"Seen combos      | image->text top-1: {train_acc:.0%}  (sanity, should be high)")
    print(f"UNSEEN combos    | zero-shot top-1:   {zs_acc:.0%}  (chance = {100/len(ALL):.0f}%)")
    print(f"{'='*64}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU run")


if __name__ == '__main__':
    run()
