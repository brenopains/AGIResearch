import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
D_IN = 16
K = 2

# Encoders
class CoordinateEncoder:
    def encode(self, x): return x

class FourierEncoder:
    def encode(self, x): return torch.abs(torch.fft.fft(x.to(torch.complex64)))

class GraphEncoder:
    def encode(self, x):
        shifted = torch.roll(x, shifts=1, dims=1)
        diff = torch.abs(x - shifted)
        sorted_diff, _ = torch.sort(diff, dim=1)
        return sorted_diff

# Load task data (Seed 1)
rng = np.random.RandomState(1)
n_samples = 800

# Task 1
x1 = torch.randn(n_samples, D_IN, device=DEVICE) * 0.1
y1 = []
for i in range(n_samples):
    p = rng.randint(0, D_IN)
    x1[i, p] += 4.0
    y1.append(1 if p >= D_IN // 2 else 0)
y1 = torch.tensor(y1, dtype=torch.long, device=DEVICE)

# Task 2
x2 = torch.randn(n_samples, D_IN, device=DEVICE) * 0.2
freqs = rng.choice([1, 4], size=n_samples)
phases = rng.uniform(0, 2 * np.pi, size=n_samples)
t = torch.linspace(0, 2 * np.pi, D_IN, device=DEVICE)
for i in range(n_samples):
    amp = 4.0 if freqs[i] == 1 else 1.0
    x2[i] += amp * torch.sin(t * freqs[i] + phases[i])
y2 = torch.tensor(freqs == 4, dtype=torch.long, device=DEVICE)

# Task 3
x3 = torch.randn(n_samples, D_IN, device=DEVICE)
diffs, _ = torch.max(torch.abs(x3[:, 1:] - x3[:, :-1]), dim=1)
threshold = diffs.median().item()
y3 = (diffs > threshold).long()

task_inputs = [x1, x2, x3]
task_targets = [y1, y2, y3]
encoders = [CoordinateEncoder(), FourierEncoder(), GraphEncoder()]

class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(D_IN, K)
    def forward(self, z): return self.net(z)

# Train all combinations
print("3x3 Accuracy Matrix:")
print(f"{'Task':<25} | {'Coord':^8} | {'Fourier':^8} | {'Graph':^8}")
print("-" * 59)

names = ["Spatial Translation", "Fourier Tiling", "Graph Connectivity"]
for t_idx in range(3):
    tx, ty = task_inputs[t_idx], task_targets[t_idx]
    accs = []
    for s_idx in range(3):
        s = Specialist().to(DEVICE)
        opt = torch.optim.Adam(s.parameters(), 1.5e-2)
        # Train
        for epoch in range(400):
            perm = torch.randperm(len(tx))[:64]
            bx, by = tx[perm], ty[perm]
            opt.zero_grad()
            z = encoders[s_idx].encode(bx)
            loss = F.cross_entropy(s(z), by)
            loss.backward()
            opt.step()
        s.eval()
        # Eval
        with torch.no_grad():
            z_eval = encoders[s_idx].encode(tx)
            pred = s(z_eval).argmax(1)
            acc = (pred == ty).float().mean().item()
            accs.append(acc)
    print(f"{names[t_idx]:<25} | {accs[0]:>7.1%} | {accs[1]:>7.1%} | {accs[2]:>7.1%}")
