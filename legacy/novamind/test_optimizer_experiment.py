"""
Optimizer Experiment: fix 50 vars, compare 4 optimizers.
Separates architecture from optimizer.
"""
import sys, os, json, random, math, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp
from types import SimpleNamespace

OUTPUT_DIR = "D:/novamind_data/optimizer_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REALITY = {"cognitive_function": 0.770, "neurodegeneration": 0.249}

# ─── STABLE BASE MODEL (same for all) ───
def make_base():
    m = SimpleNamespace()
    m.name = "base"
    m.variables = ["cognitive_function", "neurodegeneration"]
    m.initial_state = np.array([1.0, 0.0])
    ode = lambda t, y: np.array([-0.01*y[0], 0.005*y[0] - 0.01*y[1]])
    
    # Ensure stability by clipping
    def ode_safe(t, y):
        ys = np.clip(y, -5, 5)
        return np.clip(ode(t, ys), -5, 5)
    
    m.ode_fn = ode_safe
    m.intervention_fn = lambda y0, i, d: y0.copy()
    return m

# ─── BUILD 50-VAR MODEL ───
def build_50var():
    m = make_base()
    n_add = 48
    m.variables = m.variables + [f"lat_{i}" for i in range(n_add)]
    m.initial_state = np.append(m.initial_state, [0.5]*n_add)
    
    base_vars = ["cognitive_function", "neurodegeneration"]
    ode_orig = lambda t, y: np.array([-0.01*y[0], 0.005*y[0] - 0.01*y[1]])
    
    def make_ode(orig, nb, nl, bv):
        def fn(t, y):
            ys = np.clip(y, -5, 5)
            dy = list(orig(t, ys[:nb]))
            for i in range(nl):
                idx = nb + i
                d = 0.01 * ys[idx]
                ti = i % nb  # each latent connects to one base var
                if ti < len(dy):
                    d += 0.02 * ys[ti] / max(1, nl)
                    dy[ti] += 0.01 * ys[idx] / max(1, nl)
                d -= 0.01 * ys[idx]
                dy.append(d)
            return np.clip(np.array(dy), -5, 5)
        return fn
    
    m.ode_fn = make_ode(ode_orig, 2, n_add, base_vars)
    return m

def compute_error(model):
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), np.clip(model.initial_state, 0, 1),
                       method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        for var, obs in REALITY.items():
            if var in model.variables:
                idx = model.variables.index(var)
                pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                pred = max(0, min(1, pred))
                if abs(obs) > 0.01:
                    errors.append(abs(pred - obs) / obs)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0

model = build_50var()
base_err = compute_error(model)
print(f"Base error: {base_err:.4f}")

# ═══════════════════════════════════════════════════════════════
# 4 OPTIMIZERS
# ═══════════════════════════════════════════════════════════════

import copy

def optimizer_random_search(n_iterations=200):
    """Testa escalas aleatorias em 4 variaveis."""
    best = base_err
    best_model = model
    for _ in range(n_iterations):
        m = copy.deepcopy(model)
        m.ode_fn = model.ode_fn  # reuse (no deepcopy on lambdas)
        scales = {v: random.uniform(0.1, 5.0) for v in ["cognitive_function", "neurodegeneration"]}
        orig = model.ode_fn
        def wrap(orig_fn, sc):
            def fn(t, y):
                dy = orig_fn(t, y)
                for v, s in sc.items():
                    if v in m.variables:
                        i = m.variables.index(v)
                        if i < len(dy):
                            dy[i] *= s
                return dy
            return fn
        m.ode_fn = wrap(orig, scales)
        err = compute_error(m)
        if err < best:
            best = err
    return (base_err - best) / max(0.001, base_err)

def optimizer_one_var_at_a_time(n_trials_per_var=30):
    """Calibra UMA variavel por vez."""
    m = copy.deepcopy(model)
    m.ode_fn = model.ode_fn
    best = base_err
    
    for var_name in ["cognitive_function", "neurodegeneration"]:
        for _ in range(n_trials_per_var):
            scale = random.uniform(0.1, 5.0)
            orig = m.ode_fn
            idx = m.variables.index(var_name)
            def wrap(orig_fn, vi, sc):
                def fn(t, y):
                    dy = orig_fn(t, y)
                    if vi < len(dy):
                        dy[vi] *= sc
                    return dy
                return fn
            m.ode_fn = wrap(orig, idx, scale)
            err = compute_error(m)
            if err < best:
                best = err
                break  # keep this scale
            m.ode_fn = orig  # revert
    
    return (base_err - best) / max(0.001, base_err)

def optimizer_hierarchical(n_trials_per_module=30):
    """Calibracao hierarquica: modulo 1 (cognition), modulo 2 (neurodeg)."""
    m = copy.deepcopy(model)
    m.ode_fn = model.ode_fn
    best = base_err
    
    modules = [("cognitive_function", ["cognitive_function"]),
               ("neurodegeneration", ["neurodegeneration"])]
    
    for mod_name, mod_vars in modules:
        for _ in range(n_trials_per_module):
            scale = random.uniform(0.1, 5.0)
            orig = m.ode_fn
            def wrap(orig_fn, mv, sc):
                def fn(t, y):
                    dy = orig_fn(t, y)
                    for v in mv:
                        if v in model.variables:
                            i = model.variables.index(v)
                            if i < len(dy):
                                dy[i] *= sc
                    return dy
                return fn
            m.ode_fn = wrap(orig, mod_vars, scale)
            err = compute_error(m)
            if err < best:
                best = err
                break
            m.ode_fn = orig
    
    return (base_err - best) / max(0.001, base_err)

def optimizer_bruteforce_mesh(n_per_dim=5):
    """Grid search: 5 valores para cada uma das 2 variaveis = 25 tentativas."""
    best = base_err
    for s1 in np.linspace(0.2, 3.0, n_per_dim):
        for s2 in np.linspace(0.2, 3.0, n_per_dim):
            orig = model.ode_fn
            m = copy.deepcopy(model)
            m.ode_fn = model.ode_fn
            def wrap(orig_fn, s1, s2):
                def fn(t, y):
                    dy = orig_fn(t, y)
                    if len(dy) >= 2:
                        dy[0] *= s1
                        dy[1] *= s2
                    return dy
                return fn
            m.ode_fn = wrap(orig, s1, s2)
            err = compute_error(m)
            if err < best:
                best = err
    return (base_err - best) / max(0.001, base_err)

# ─── RUN ───
optimizers = [
    ("Random Search (200)", optimizer_random_search, 200),
    ("One-at-a-time (30/var)", optimizer_one_var_at_a_time, 30),
    ("Hierarchical (30/mod)", optimizer_hierarchical, 30),
    ("Grid Search 5x5", optimizer_bruteforce_mesh, 5),
]

print(f"\n{'='*65}")
print(f"OPTIMIZER COMPARISON (50-var model, fixed)")
print(f"{'='*65}")
print(f"\nBase error (no optimization): {base_err:.4f}")
print(f"\n{'Optimizer':30s} {'Gain':>8s} {'Effort':>8s} {'Eff/Gain':>10s}")
print(f"{'-'*60}")

results = []
for name, fn, effort in optimizers:
    random.seed(42)
    gain = fn(effort) if "effort" in str(fn.__code__.co_varnames) else (fn() if not name.startswith("Grid") else optimizer_bruteforce_mesh(5))
    
    # Actually call properly
    random.seed(42)
    if "Random" in name:
        gain = optimizer_random_search(200)
    elif "One" in name:
        gain = optimizer_one_var_at_a_time(30)
    elif "Hier" in name:
        gain = optimizer_hierarchical(30)
    elif "Grid" in name:
        gain = optimizer_bruteforce_mesh(5)
    
    eff_gain = gain / effort * 10000 if effort > 0 else 0
    results.append({"name": name, "gain": float(gain), "effort": effort})
    print(f"{name:30s} {gain:8.1%} {effort:8d} {eff_gain:10.1f}")

print(f"\n{'='*65}")
print(f"CONCLUSIONS")
print(f"{'='*65}")
best = max(results, key=lambda r: r["gain"])
most_efficient = max(results, key=lambda r: r["gain"] / max(1, r["effort"]))
print(f"  Best gain:          {best['name']} ({best['gain']:.1%})")
print(f"  Most efficient:     {most_efficient['name']}")

# Compare best vs worst
if len(results) >= 2:
    r_best = max(results, key=lambda r: r["gain"])
    r_worst = min(results, key=lambda r: r["gain"])
    print(f"  Worst gain:         {r_worst['name']} ({r_worst['gain']:.1%})")
    print(f"  Ratio (best/worst): {r_best['gain']/max(0.001, r_worst['gain']):.1f}x")
    print(f"\n  => The optimizer MATTERS by up to {r_best['gain']/max(0.001, r_worst['gain']):.1f}x")

report_path = os.path.join(OUTPUT_DIR, "optimizer_comparison.json")
with open(report_path, "w") as f:
    json.dump({"base_error": float(base_err), "results": results}, f, indent=2)
print(f"\nRelatorio: {report_path}")
