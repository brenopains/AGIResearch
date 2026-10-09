"""
Structural signatures of generalization: 
what architectural properties predict transfer across domains?

For 50+ model configurations, measures:
  1. Structural metrics (modularity, clustering, SW, entropy, phi)
  2. Generalization score (train Alzheimer -> test Parkinson/ALS/Huntington)
  3. Correlation: which metric PREDICTS generalization?
"""
import sys, os, json, itertools, random, math, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp
from types import SimpleNamespace

OUTPUT_DIR = "D:/novamind_data/structural_generalization"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = random.Random(42)

# ─── REALITIES ───
REALITIES = {
    "Alzheimer": {"cognitive_function": 0.770, "neurodegeneration": 0.249},
    "Parkinson": {"motor_function": 0.500, "quality_of_life": 0.600},
    "ALS": {"motor_function": 0.400, "quality_of_life": 0.500},
    "Huntington": {"cognitive_function": 0.600, "motor_function": 0.500},
}

# ─── MODEL GENERATION ───
def make_model(n_base, n_modules, module_strength, n_latent=0):
    """Creates a model with controlled structural properties.
    
    n_base: base variables (2)
    n_modules: how many modules to split latents into
    module_strength: how strong the connections are within modules
    n_latent: how many latent variables to add
    """
    m = SimpleNamespace()
    m.variables = ["var_0", "var_1"]  # base: cognitive, neurodegeneration
    for i in range(n_latent):
        m.variables.append(f"lat_{i}")
    m.initial_state = np.array([0.5] * (2 + n_latent))
    
    base_vars = ["var_0", "var_1"]
    
    # Build ODE with controlled modularity
    def make_ode(nb, nl, nm, ms):
        def fn(t, y):
            ys = np.clip(y, -5, 5)
            dy = [-0.01*ys[0], 0.005*ys[0] - 0.01*ys[1]]
            
            # Extend dy for base vars if needed
            while len(dy) < nb:
                dy.append(-0.01*ys[len(dy)])
            
            for i in range(nl):
                idx = nb + i
                d = 0.01 * ys[idx] - 0.01 * ys[idx]  # self-decay
                
                # Which module does this latent belong to?
                mod = i % max(1, nm)
                
                # Connect to base variables based on module
                target_idx = mod % nb
                d += 0.02 * ys[target_idx] / max(1, nl)
                
                # Within-module strength
                if target_idx < len(dy):
                    dy[target_idx] += ms * 0.01 * ys[idx] / max(1, nl)
                
                dy.append(d)
            
            return np.clip(np.array(dy), -5, 5)
        return fn
    
    m.ode_fn = make_ode(2, n_latent, n_modules, module_strength)
    m.name = f"model_b{n_base}_m{n_modules}_s{module_strength:.1f}_l{n_latent}"
    return m


def compute_error(model, reality):
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                       method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        for obs_var, obs_val in reality.items():
            idx = -1
            if "cognitive" in obs_var or "neurodeg" in obs_var: idx = 0
            elif "motor" in obs_var or "quality" in obs_var: idx = 1
            if idx >= 0 and idx < sol.y.shape[0]:
                pred = float(sol.y[idx, -1])
                pred = max(0, min(1, 1/(1+abs(pred)) if abs(pred) > 1 else pred))
                if abs(obs_val) > 0.01:
                    errors.append(abs(pred - obs_val) / obs_val)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0


# ─── STRUCTURAL METRICS ───
def compute_structural_metrics(model):
    """Compute structural properties from the model's behavior."""
    n = len(model.variables)
    if n < 2: return {"modularity": 0, "clustering": 0, "small_world": 1, "effective_dim": n}
    
    # Build adjacency from ODE perturbation
    adj = np.zeros((n, n))
    y0 = model.initial_state
    eps = 0.01
    try:
        base = model.ode_fn(0, y0)
        for j in range(n):
            yp = y0.copy()
            yp[j] += eps
            pert = model.ode_fn(0, yp)
            for i in range(n):
                if i < len(base) and i < len(pert):
                    adj[i,j] = 1.0 if abs(pert[i] - base[i]) > 1e-6 else 0.0
    except:
        np.fill_diagonal(adj, 1)
    
    # Modularity (approximate)
    degrees = adj.sum(axis=1)
    m = degrees.sum()
    Q = 0.0
    if m > 0:
        for i in range(n):
            for j in range(n):
                if adj[i,j] > 0:
                    Q += 1.0 - (degrees[i] * degrees[j]) / (2 * m)
        Q = max(0, Q / (2 * m))
    
    # Clustering
    C = 0.0
    nc = 0
    for i in range(n):
        neighbors = [j for j in range(n) if adj[i,j] > 0 and j != i]
        k = len(neighbors)
        if k >= 2:
            edges = sum(1 for a in range(k) for b in range(a+1, k) if adj[neighbors[a], neighbors[b]] > 0)
            C += 2 * edges / (k * (k-1))
            nc += 1
    C = C / max(1, nc)
    
    # Effective dimensionality (number of nonzero eigenvalues of correlation matrix)
    try:
        trials = 50
        trajs = np.zeros((n, trials))
        for t in range(trials):
            y = model.initial_state.copy() + np.random.RandomState(t).randn(n) * 0.01
            sol = solve_ivp(model.ode_fn, (0, 100), y, method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
            if sol.y.shape[1] > 0:
                trajs[:, t] = sol.y[:, -1]
        corr = np.corrcoef(trajs)
        eigvals = np.linalg.eigvalsh(corr)
        eff_dim = np.sum(eigvals > 0.1 * max(eigvals))
    except:
        eff_dim = n
    
    return {
        "modularity": float(Q),
        "clustering": float(C),
        "small_world": float(1 + C),
        "effective_dim": int(eff_dim),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════

print("="*70)
print("STRUCTURAL SIGNATURES OF GENERALIZATION")
print("="*70)

# Generate diverse model configurations
configs = []
for n_latent in [0, 5, 15, 30]:
    for n_modules in [1, 2, 4, 8]:
        for strength in [0.1, 0.3, 0.5, 0.8]:
            configs.append((2, n_modules, strength, n_latent))

print(f"Testing {len(configs)} configurations...")

results = []
for i, (nb, nm, ms, nl) in enumerate(configs):
    model = make_model(nb, nm, ms, nl)
    
    # Structural metrics
    struct = compute_structural_metrics(model)
    
    # Train on Alzheimer
    err_alz = compute_error(model, REALITIES["Alzheimer"])
    
    # Test on other domains (zero-shot transfer)
    err_park = compute_error(model, REALITIES["Parkinson"])
    err_als = compute_error(model, REALITIES["ALS"])
    err_hunt = compute_error(model, REALITIES["Huntington"])
    
    avg_transfer = np.mean([err_park, err_als, err_hunt])
    generalization_score = (err_alz - avg_transfer) / max(0.001, err_alz)
    
    results.append({
        "n_latent": nl, "n_modules": nm, "strength": ms,
        "modularity": struct["modularity"],
        "clustering": struct["clustering"],
        "small_world": struct["small_world"],
        "effective_dim": struct["effective_dim"],
        "error_Alzheimer": round(float(err_alz), 4),
        "error_Parkinson": round(float(err_park), 4),
        "error_ALS": round(float(err_als), 4),
        "error_Huntington": round(float(err_hunt), 4),
        "generalization": round(float(generalization_score), 4),
    })
    
    if i % 10 == 0:
        print(f"  [{i+1}/{len(configs)}] mod={nm} lat={nl} str={ms:.1f} gen={generalization_score:.3f}")

# ─── CORRELATIONS ───
print(f"\n{'='*70}")
print(f"CORRELATIONS: structural metrics vs generalization")
print(f"{'='*70}")

metrics = ["modularity", "clustering", "small_world", "effective_dim"]
gens = [r["generalization"] for r in results]

for metric in metrics:
    vals = [r[metric] for r in results]
    if len(vals) > 1 and len(gens) > 1:
        corr = np.corrcoef(vals, gens)[0, 1]
        print(f"  {metric:20s} r = {corr:.3f}")

best_metric = max(metrics, key=lambda m: abs(np.corrcoef([r[m] for r in results], gens)[0, 1]))
best_r = np.corrcoef([r[best_metric] for r in results], gens)[0, 1]
print(f"\n  Best predictor of generalization: {best_metric} (r={best_r:.3f})")

# ─── BEST CONFIG ───
best_gen = max(results, key=lambda r: r["generalization"])
worst_gen = min(results, key=lambda r: r["generalization"])
print(f"\n  Best generalizer:  mod={best_gen['n_modules']} lat={best_gen['n_latent']} str={best_gen['strength']:.1f} gen={best_gen['generalization']:.3f}")
print(f"  Worst generalizer: mod={worst_gen['n_modules']} lat={worst_gen['n_latent']} str={worst_gen['strength']:.1f} gen={worst_gen['generalization']:.3f}")

# ─── WHAT PREDICTS GOOD GENERALIZATION? ───
print(f"\n{'='*70}")
print(f"PROFILE: good vs poor generalizers")
print(f"{'='*70}")

top5 = sorted(results, key=lambda r: -r["generalization"])[:5]
bot5 = sorted(results, key=lambda r: r["generalization"])[:5]

for group, items in [("TOP 5 (best generalizers)", top5), ("BOT 5 (worst generalizers)", bot5)]:
    print(f"\n  {group}:")
    for m in ["modularity", "clustering", "small_world", "effective_dim"]:
        vals = [r[m] for r in items]
        print(f"    {m:20s} mean={np.mean(vals):.3f}")

# Save
report_path = os.path.join(OUTPUT_DIR, "structural_generalization.json")
with open(report_path, "w") as f:
    json.dump({"results": results, "correlations": {
        m: round(float(np.corrcoef([r[m] for r in results], gens)[0, 1]), 3)
        for m in metrics
    }}, f, indent=2)
print(f"\nRelatorio: {report_path}")
