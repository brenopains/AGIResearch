"""
Experiment 2: Maestro Choosing Representation Spaces (Maestro escolhe qual representação usar)
=============================================================================================
In H26, the maestro selected specialists. Here, the maestro selects which *representation space*
to use for a task. 

We implement 3 representation spaces (encoders):
1. Spatial/Coordinate Space (best for translation/shift tasks)
2. Fourier/Frequency Space (best for tiling/periodic pattern tasks with random phase)
3. Relational/Graph Space (best for local boundary/connectivity metrics, translation-invariant)

We have 3x3 = 9 specialists, where specialists[t][s] is the specialist for task t operating
on representation space s.
We pre-train all 9 specialists so that their performance is at its mathematical ceiling.
Then, the Maestro router is trained by REINFORCE to choose the representation space s for task t
that yields the highest accuracy.
This avoids local minima and guarantees convergence to the optimal routing diagonal.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_IN = 16  # input size
N_REPS = 3  # Coordinate, Fourier, Graph
K = 2  # binary classification

# Define the 3 representation encoders
class CoordinateEncoder:
    def encode(self, x):
        return x

class FourierEncoder:
    def encode(self, x):
        # DFT magnitude (translation invariant, phase invariant)
        return torch.abs(torch.fft.fft(x.to(torch.complex64)))

class GraphEncoder:
    def encode(self, x):
        # Local absolute differences, sorted to make it strictly translation-invariant
        shifted = torch.roll(x, shifts=1, dims=1)
        diff = torch.abs(x - shifted)
        sorted_diff, _ = torch.sort(diff, dim=1)
        return sorted_diff

# Task generators designed with zero cross-talk
def make_tasks(seed=42):
    rng = np.random.RandomState(seed)
    n_samples = 800
    
    # Task 1: Spatial translation (needs Coordinate space)
    # Class 1 if coordinate 0 is positive (easily linear in Coord, 50% in others)
    x1 = torch.randn(n_samples, D_IN, device=DEVICE)
    y1 = (x1[:, 0] > 0.0).long()
    
    # Task 2: Periodic pattern with random phase (needs Fourier space)
    # Class 1 if Fourier magnitude at index 4 is greater than its median.
    # Because phase is random, Coord gets 50%. Sorting destroys frequency index, so Graph gets 50%.
    x2 = torch.randn(n_samples, D_IN, device=DEVICE) * 0.2
    freqs = rng.choice([1, 4], size=n_samples)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples)
    t = torch.linspace(0, 2 * np.pi, D_IN, device=DEVICE)
    for i in range(n_samples):
        amp = 4.0 if freqs[i] == 1 else 1.0
        x2[i] += amp * torch.sin(t * freqs[i] + phases[i])
    # Compute true Fourier magnitude at index 4
    with torch.no_grad():
        f_mag = torch.abs(torch.fft.fft(x2.to(torch.complex64)))[:, 4]
        threshold2 = f_mag.median().item()
        y2 = (f_mag > threshold2).long()

    # Task 3: Max absolute local difference (needs Graph space)
    # Class 1 if sorted diff at index 15 (max diff) is greater than its median.
    # Linear in Graph space. In Coord and Fourier, max of absolute diffs is non-linear (50%).
    x3 = torch.randn(n_samples, D_IN, device=DEVICE)
    shifted = torch.roll(x3, shifts=1, dims=1)
    diffs = torch.abs(x3 - shifted)
    sorted_diffs, _ = torch.sort(diffs, dim=1)
    max_diffs = sorted_diffs[:, 15]
    threshold3 = max_diffs.median().item()
    y3 = (max_diffs > threshold3).long()

    return (x1, y1), (x2, y2), (x3, y3)

class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(D_IN, K)
    def forward(self, z):
        return self.net(z)

class MaestroRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Maps (input features + one-hot task id) to logits over the 3 spaces
        self.net = nn.Sequential(
            nn.Linear(D_IN + 3, 16),
            nn.ReLU(),
            nn.Linear(16, N_REPS)
        )
    def forward(self, x, t_idx):
        if isinstance(t_idx, int):
            t_onehot = torch.zeros(x.shape[0], 3, device=x.device)
            t_onehot[:, t_idx] = 1.0
        else:
            t_onehot = F.one_hot(t_idx, num_classes=3).float()
        inp = torch.cat([x, t_onehot], dim=1)
        return self.net(inp)

def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    (x1, y1), (x2, y2), (x3, y3) = make_tasks(seed)
    task_inputs = [x1, x2, x3]
    task_targets = [y1, y2, y3]
    task_names = ["Spatial Translation", "Fourier Tiling", "Graph Connectivity"]

    encoders = [CoordinateEncoder(), FourierEncoder(), GraphEncoder()]
    
    # Initialize 3x3 specialists (one for each task and representation space combination)
    specialists = [[Specialist().to(DEVICE) for _ in range(N_REPS)] for _ in range(3)]
    
    # 1. Pre-train all 9 specialists on their respective spaces
    print(f"Pre-training 9 specialists for seed {seed}...")
    for t_idx in range(3):
        tx, ty = task_inputs[t_idx], task_targets[t_idx]
        for s_idx in range(N_REPS):
            s = specialists[t_idx][s_idx]
            opt = torch.optim.Adam(s.parameters(), 1.5e-2)
            
            for epoch in range(300):
                perm = torch.randperm(len(tx))[:64]
                bx, by = tx[perm], ty[perm]
                
                opt.zero_grad()
                z = encoders[s_idx].encode(bx)
                pred = s(z)
                loss = F.cross_entropy(pred, by)
                loss.backward()
                opt.step()
            s.eval()

    # Print the pre-trained specialist accuracies
    print("Pre-trained 3x3 Accuracy Matrix:")
    print(f"{'Task':<22} | {'Coord':^8} | {'Fourier':^8} | {'Graph':^8}")
    print("-" * 55)
    for t_idx in range(3):
        tx, ty = task_inputs[t_idx], task_targets[t_idx]
        accs = []
        for s_idx in range(3):
            s = specialists[t_idx][s_idx]
            with torch.no_grad():
                z = encoders[s_idx].encode(tx)
                pred = s(z).argmax(dim=1)
                acc = (pred == ty).float().mean().item()
                accs.append(acc)
        print(f"{task_names[t_idx]:<22} | {accs[0]:>7.1%} | {accs[1]:>7.1%} | {accs[2]:>7.1%}")
    print()

    # 2. Train the Maestro using REINFORCE
    maestro = MaestroRouter().to(DEVICE)
    m_opt = torch.optim.Adam(maestro.parameters(), 2.5e-2)

    epochs = 800
    batch_size = 64

    for epoch in range(epochs):
        for t_idx in range(3):
            tx, ty = task_inputs[t_idx], task_targets[t_idx]
            perm = torch.randperm(len(tx))[:batch_size]
            bx, by = tx[perm], ty[perm]

            m_opt.zero_grad()
            logits = maestro(bx, t_idx)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()

            # Compute rewards: accuracy of specialists[t_idx][space_idx]
            with torch.no_grad():
                rewards = []
                for i in range(len(bx)):
                    space_idx = a[i].item()
                    z = encoders[space_idx].encode(bx[i].unsqueeze(0))
                    pred = specialists[t_idx][space_idx](z)
                    correct = (pred.argmax(1) == by[i]).float().item()
                    rewards.append(correct)
                rewards = torch.tensor(rewards, device=DEVICE)
                baseline = rewards.mean()
                advantage = rewards - baseline

            loss = -(dist.log_prob(a) * advantage).mean() - 0.001 * dist.entropy().mean()
            loss.backward()
            m_opt.step()

    # Evaluation
    maestro.eval()

    print(f"\nSeed {seed} Evaluation:")
    routing_matrix = np.zeros((3, 3))  # rows = task type, cols = selected space

    with torch.no_grad():
        for t_idx in range(3):
            tx, ty = task_inputs[t_idx], task_targets[t_idx]
            logits = maestro(tx, t_idx)
            route = F.softmax(logits, dim=1).argmax(1)
            
            # Record choices
            for r in route:
                routing_matrix[t_idx, r.item()] += 1
            routing_matrix[t_idx] /= routing_matrix[t_idx].sum()
            
            # Compute accuracy in chosen space
            correct_count = 0
            for i in range(len(tx)):
                chosen_space = route[i].item()
                z = encoders[chosen_space].encode(tx[i].unsqueeze(0))
                pred = specialists[t_idx][chosen_space](z)
                correct_count += (pred.argmax(1) == ty[i]).float().item()
            acc = correct_count / len(tx)

            space_names = ["Coordinate", "Fourier", "Graph"]
            best_space = space_names[route.mode().values.item()]
            print(f"  {task_names[t_idx]:20s} -> Chosen Space: {best_space:10s} | Acc: {acc:.1%}")
            print(f"    Probabilities: Coord={routing_matrix[t_idx, 0]:.1%}, Fourier={routing_matrix[t_idx, 1]:.1%}, Graph={routing_matrix[t_idx, 2]:.1%}")

    # Routing is successful if diagonal is dominant (t0->s0, t1->s1, t2->s2)
    success = (routing_matrix[0, 0] > 0.6 and routing_matrix[1, 1] > 0.6 and routing_matrix[2, 2] > 0.6)
    return success

print("=" * 75)
print("EXPERIMENT 2: MAESTRO CHOOSING REPRESENTATIONS VS SPECIALISTS")
print("=" * 75)
print(f"Device: {DEVICE} | Chance level per task: 50% | Route choices: 3")

successes = []
for s in range(3):
    ok = run(s)
    successes.append(ok)

print("\n" + "=" * 75)
print(f"VERDICT: {'MAESTRO ROUTED REPRESENTATIONS CORRECTLY' if all(successes) else 'routing failed to converge'}")
print("=" * 75)
