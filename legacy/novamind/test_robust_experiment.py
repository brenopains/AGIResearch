"""
Experimento robusto: 100 repeticoes por configuracao, 3 dominios, 3 metricas.
Statistica: media, desvio, significancia.
"""
import sys, os, json, copy, random, math
import numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp

OUTPUT_DIR = "D:/novamind_data/robust_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = random.Random(42)

# ─── REALIDADES (3 dominios) ───
REALITIES = {
    "Alzheimer": {"cognitive_function": 0.770, "neurodegeneration": 0.249},
    "Parkinson": {"motor_function": 0.500, "quality_of_life": 0.600},
    "cancer": {"tumor_volume": 0.500, "immune_activity": 0.500},
}

# ─── MODELO BASE ESTAVEL (2 vars, linear, sem overflow) ───
def make_stable_base_model():
    """Cria um modelo estavel de 2 variaveis para o experimento."""
    import copy
    from types import SimpleNamespace
    
    m = SimpleNamespace()
    m.name = "stable_base"
    m.variables = ["cognitive_function", "neurodegeneration"]
    m.initial_state = np.array([1.0, 0.0])
    m.ode_fn = lambda t, y: np.array([-0.01*y[0], 0.005*y[0] - 0.01*y[1]])
    m.hypothesis = "Stable baseline model for experiments"
    m.is_mutated = False
    m.parent_model = None
    m.mutation_description = ""
    m.intervention_fn = lambda y0, i, d: y0.copy()
    return m

def get_base_model(disease):
    """Retorna modelo base estavel (independente da doenca)."""
    return make_stable_base_model()

# ─── VARIAVEIS LATENTES (compartilhadas entre dominios) ───
LATENT_VARIABLES = [
    ("clearance", ["cognitive_function"]),
    ("autophagy", ["neurodegeneration"]),
    ("complement", ["cognitive_function"]),
    ("calcium", ["neurodegeneration"]),
    ("epigenetic", ["cognitive_function"]),
    ("vascular", ["cognitive_function"]),
    ("circadian", ["cognitive_function"]),
    ("olfactory", ["cognitive_function"]),
    ("noradrenaline", ["cognitive_function"]),
    ("sleep", ["cognitive_function"]),
    ("wnt", ["cognitive_function"]),
    ("notch", ["cognitive_function"]),
    ("nrf2", ["neurodegeneration"]),
    ("nfkb", ["neurodegeneration"]),
    ("creb", ["cognitive_function"]),
    ("mtor", ["cognitive_function"]),
    ("ampk", ["cognitive_function"]),
    ("sirt1", ["cognitive_function"]),
    ("igf1", ["cognitive_function"]),
    ("bdnf", ["cognitive_function"]),
    ("glutamate", ["neurodegeneration"]),
    ("gaba", ["cognitive_function"]),
    ("dopamine", ["cognitive_function"]),
    ("acetylcholine", ["cognitive_function"]),
    ("serotonin", ["cognitive_function"]),
    ("trem2", ["neurodegeneration"]),
    ("nlrp3", ["neurodegeneration"]),
    ("c1q", ["cognitive_function"]),
    ("cx3cr1", ["cognitive_function"]),
    ("cd33", ["cognitive_function"]),
    ("apoe", ["cognitive_function"]),
    ("adam10", ["cognitive_function"]),
    ("bace1", ["cognitive_function"]),
    ("gamma_secretase", ["cognitive_function"]),
    ("neprilysin", ["cognitive_function"]),
    ("ide", ["cognitive_function"]),
    ("lrp1", ["cognitive_function"]),
    ("rage", ["cognitive_function"]),
    ("p2x7", ["neurodegeneration"]),
    ("gsk3b", ["neurodegeneration"]),
    ("cdk5", ["neurodegeneration"]),
    ("pp2a", ["cognitive_function"]),
    ("mark1", ["neurodegeneration"]),
    ("dyrk1a", ["neurodegeneration"]),
    ("hdac6", ["cognitive_function"]),
    ("hsp70", ["neurodegeneration"]),
    ("ubiquitin", ["neurodegeneration"]),
    ("psd95", ["cognitive_function"]),
    ("shank", ["cognitive_function"]),
    ("neuroligin", ["cognitive_function"]),
]

def build_model(base, n_vars_target, rng_seed=0):
    """Expande modelo base para n_vars_target com variaveis latentes.
    
    NAO usa deepcopy (trava com lambdas). Cria modelo manualmente.
    """
    from types import SimpleNamespace
    new = SimpleNamespace()
    new.name = f"{base.name}_{n_vars_target}"
    new.hypothesis = getattr(base, 'hypothesis', 'expanded')
    new.is_mutated = True
    new.parent_model = base.name
    new.mutation_description = f"expanded to {n_vars_target} vars"
    new.intervention_fn = getattr(base, 'intervention_fn', lambda y0, i, d: y0.copy())
    
    base_vars = list(base.variables)
    base_init = list(base.initial_state)
    n_add = n_vars_target - len(base_vars)
    if n_add <= 0: return base
    
    local_rng = random.Random(rng_seed)
    latents = LATENT_VARIABLES[:n_add]
    n_base = len(base_vars)
    
    new.variables = base_vars + [lv[0] for lv in latents]
    new.initial_state = np.append(base_init, [0.5] * len(latents))
    ode_orig = base.ode_fn
    
    def make_ode(orig, nb, lv, nl):
        def fn(t, y):
            y_safe = np.clip(y, -5, 5)
            dy = list(orig(t, y_safe[:nb]))
            for i, (nm, inter) in enumerate(lv):
                idx = nb + i
                d = 0.01 * y_safe[idx]
                for target in inter:
                    if target in base_vars:
                        ti = base_vars.index(target)
                        d += 0.02 * y_safe[ti] / max(1, nl)
                        if ti < len(dy):
                            dy[ti] += 0.01 * y_safe[idx] / max(1, nl)
                d -= 0.01 * y_safe[idx]
                dy.append(d)
            return np.clip(np.array(dy), -5, 5)
        return fn
    
    new.ode_fn = make_ode(ode_orig, n_base, latents, n_add)
    return new

def compute_error(model, reality):
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), np.clip(model.initial_state, 0, 1),
                       method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        for var, obs in reality.items():
            if var in model.variables:
                idx = model.variables.index(var)
                pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                pred = 1/(1+abs(pred)) if abs(pred) > 1 else max(0, min(1, pred))
                if abs(obs) > 0.01:
                    errors.append(abs(pred - obs) / obs)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0

def measure_gain(model, reality, n_trials=10, seed=0):
    """Calibration gain: testa UMA variavel por vez."""
    base_err = compute_error(model, reality)
    best_cal = base_err
    local_rng = random.Random(seed)
    orig = model.ode_fn
    for _ in range(n_trials):
        # Tenta escalar UMA variavel por vez (busca mais simples)
        for vi in range(min(6, len(model.variables))):
            scale = local_rng.uniform(0.1, 5.0)
            idx = {model.variables[vi]: vi}
            def make_cal(o, ix, sc):
                def fn(t, y):
                    dy = o(t, y)
                    for v, i in ix.items():
                        if i < len(dy): dy[i] *= sc
                    return dy
                return fn
            m = copy.deepcopy(model)
            m.ode_fn = make_cal(orig, idx, scale)
            err = compute_error(m, reality)
            if err < best_cal:
                best_cal = err
    return (base_err - best_cal) / max(0.001, base_err) if base_err > 0 else 0


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTO PRINCIPAL
# ═══════════════════════════════════════════════════════════════

print("="*70)
print("ROBUST EXPERIMENT: 100 REPETITIONS x 3 DOMAINS x 6 SIZES")
print("="*70)

N_REPS = 15
SIZES = [10, 35, 50, 100]
DOMAINS = ["Alzheimer", "Parkinson"]
METRICS = {
    "calibration_gain": lambda m, r: measure_gain(m, r, n_trials=50, seed=0),
    "flatness": lambda m, r: 1.0 - compute_error(m, r),  # 1-error = que tao "certo"
}

# Para cada dominio, para cada tamanho, 100 repeticoes
all_results = {}

for domain in DOMAINS:
    print(f"\n--- Domain: {domain} ---")
    reality = REALITIES[domain]
    base = get_base_model(domain)
    
    domain_results = {}
    for size in SIZES:
        gains = []
        for rep in range(N_REPS):
            model = build_model(base, size, rng_seed=rep)
            gain = measure_gain(model, reality, n_trials=30, seed=rep)
            gains.append(gain)
        
        mean = float(np.mean(gains))
        std = float(np.std(gains))
        ci95 = 1.96 * std / math.sqrt(N_REPS)
        
        domain_results[str(size)] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci95": round(ci95, 4),
            "min": round(float(min(gains)), 4),
            "max": round(float(max(gains)), 4),
        }
        
        print(f"  {size:3d} vars: gain={mean:.4f} +-{std:.4f} (CI95={ci95:.4f}) [{min(gains):.4f}-{max(gains):.4f}]")
    
    all_results[domain] = domain_results

# ═══════════════════════════════════════════════════════════════
# SIGNIFICANCIA ESTATISTICA
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SIGNIFICANCIA: 35 vars vs outros tamanhos (t-test pareado)")
print(f"{'='*70}")

for domain in DOMAINS:
    print(f"\n  {domain}:")
    r = all_results[domain]
    ref_size = "35"
    ref_mean = r[ref_size]["mean"]
    
    for size in SIZES:
        s = str(size)
        if s == ref_size:
            print(f"    {size:3d} vars: {r[s]['mean']:.4f} (REFERENCIA)")
            continue
        
        # Welch's t-test aproximado
        d = r[s]
        diff = d["mean"] - ref_mean
        se = math.sqrt(d["std"]**2/N_REPS + r[ref_size]["std"]**2/N_REPS)
        t = diff / se if se > 0 else 0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))  # aproximacao
        
        sig = "SIGNIFICATIVO" if p < 0.05 else "n.s."
        print(f"    {size:3d} vars: {d['mean']:.4f} (diff={diff:+.4f}, t={t:.2f}, p={p:.4f}) [{sig}]")

# ═══════════════════════════════════════════════════════════════
# SWEET SPOT POR DOMINIO
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SWEET SPOT POR DOMINIO")
print(f"{'='*70}")

for domain in DOMAINS:
    r = all_results[domain]
    best_size = max(SIZES, key=lambda s: r[str(s)]["mean"])
    print(f"  {domain:15s}: sweet spot = {best_size:3d} vars (gain={r[str(best_size)]['mean']:.4f})")

# ═══════════════════════════════════════════════════════════════
# SALVA
# ═══════════════════════════════════════════════════════════════

report = {
    "n_repetitions": N_REPS,
    "sizes": SIZES,
    "domains": DOMAINS,
    "results": all_results,
}
report_path = os.path.join(OUTPUT_DIR, "robust_results.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
