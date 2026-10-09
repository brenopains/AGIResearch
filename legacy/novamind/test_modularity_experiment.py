"""
Modularity Experiment: 50 vars all-connected vs 5 modules of 10 vs 10 modules of 5.
Tests whether MODULARITY, not just variable count, drives creativity.
"""
import sys, json, os, copy, random, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp

OUTPUT_DIR = "D:/novamind_data/modularity_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REALITY = {"cognitive_function": 0.770, "neurodegeneration": 0.249}
rng = random.Random(42)

def compute_error(model):
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                       method="RK45", max_step=50, rtol=1e-3, atol=1e-6)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        for var, obs in REALITY.items():
            if var in model.variables:
                idx = model.variables.index(var)
                pred = float(sol.y[idx, -1]) if idx < sol.y.shape[0] else 0.5
                pred = 1/(1+abs(pred)) if abs(pred) > 1 else max(0, min(1, pred))
                if abs(obs) > 0.01:
                    errors.append(abs(pred - obs) / obs)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0

def measure_creativity(model):
    """Returns: (base_error, calibration_gain, mutation_success_rate)"""
    base_err = compute_error(model)
    best_cal = base_err
    for _ in range(30):
        m = copy.deepcopy(model)
        scales = {v: rng.uniform(0.1, 5.0) for v in model.variables[:4]}
        idx = {v: model.variables.index(v) for v in scales if v in model.variables}
        orig = model.ode_fn
        def make_cal(o, ix, sc):
            def fn(t, y):
                dy = o(t, y)
                for v, i in ix.items():
                    if i < len(dy): dy[i] *= sc.get(v, 1.0)
                return dy
            return fn
        m.ode_fn = make_cal(orig, idx, scales)
        err = compute_error(m)
        if err < best_cal: best_cal = err
    cal_gain = (base_err - best_cal) / max(0.001, base_err) if base_err > 0 else 0
    return base_err, best_cal, cal_gain

# MODULO definitions
MODULES = {
    "amyloid_tau": ["amyloid_beta", "p_tau", "neurodegeneration", "neuroinflammation", "synaptic_function"],
    "metabolism": ["mitochondrial_function", "oxidative_stress", "insulin_metabolism", "cognitive_function"],
    "microglia": ["microglial_activation", "neuroinflammation", "synaptic_function", "cognitive_function"],
    "vascular": ["vascular_dysfunction", "oxidative_stress", "cognitive_function", "neurodegeneration"],
    "circadian": ["circadian_disruption", "cognitive_function", "neuroinflammation", "sleep_fragmentation"],
}

MODULE_VARS = {}
for mod_name, varlist in MODULES.items():
    for v in varlist:
        if v not in MODULE_VARS:
            MODULE_VARS[v] = []
        MODULE_VARS[v].append(mod_name)

def build_model(structure_type, module_strength=0.5):
    """Builds a model with given modularity structure."""
    from core.model_library import ModelLibrary
    lib = ModelLibrary()
    base = [m for m in lib.get_models("Alzheimer") if m.name == "dual_state"][0]
    new = copy.deepcopy(base)
    new.name = f"mod_{structure_type}"
    new.initial_state = base.initial_state.copy()
    
    # Number of latent variables to add
    n_latent = 40  # total = 50 vars
    n_base = 10
    
    # Create latent variable names  
    latent_names = [f"latent_{i}" for i in range(n_latent)]
    new.variables = list(base.variables) + latent_names
    new.initial_state = np.append(new.initial_state, [0.5] * n_latent)
    
    # Assign latents to modules
    latent_modules = {}
    mod_names = list(MODULES.keys())
    for i in range(n_latent):
        if structure_type == "all_connected":
            latent_modules[i] = mod_names  # connect to ALL modules
        elif structure_type == "5_modules_of_10":
            mod_idx = i // 10
            latent_modules[i] = [mod_names[mod_idx % len(mod_names)]]
        elif structure_type == "10_modules_of_5":
            mod_idx = i // 5
            latent_modules[i] = [mod_names[mod_idx % len(mod_names)]]
    
    ode_orig = new.ode_fn
    def make_ode(orig, n_base, latent_modules, structure_type):
        def fn(t, y):
            dy = list(orig(t, y[:n_base]))
            for i in range(n_latent):
                idx = n_base + i
                d = 0.01 * y[idx]
                mods = latent_modules.get(i, [])
                for mod_name in mods:
                    for target in MODULES.get(mod_name, []):
                        if target in base.variables:
                            t_idx = base.variables.index(target)
                            strength = module_strength / max(1, len(mods))
                            d += 0.02 * y[t_idx] * strength / max(1, n_latent)
                            if t_idx < len(dy):
                                dy[t_idx] += 0.01 * y[idx] * strength / max(1, n_latent)
                d -= 0.01 * y[idx]
                dy.append(d)
            return np.array(dy)
        return fn
    
    new.ode_fn = make_ode(ode_orig, n_base, latent_modules, structure_type)
    return new

print("="*65)
print("MODULARITY EXPERIMENT")
print("="*65)

structures = [
    ("all_connected", "50 vars, all connected"),
    ("5_modules_of_10", "50 vars, 5 modules of 10"),
    ("10_modules_of_5", "50 vars, 10 modules of 5"),
]

results = []
for struct_name, struct_desc in structures:
    model = build_model(struct_name)
    base_err, best_cal, cal_gain = measure_creativity(model)
    results.append((struct_name, struct_desc, base_err, best_cal, cal_gain))
    print(f"\n--- {struct_desc} ---")
    print(f"  Base error:      {base_err:.4f}")
    print(f"  Best calibration: {best_cal:.4f}")
    print(f"  Calibration gain: {cal_gain:.1%}")

print(f"\n{'='*65}")
print(f"RESULTS")
print(f"{'='*65}")
print(f"{'Structure':25s} {'Base err':>10s} {'Best cal':>10s} {'Gain':>10s}")
print(f"{'-'*55}")
for s, d, be, bc, cg in results:
    print(f"{d:25s} {be:10.4f} {bc:10.4f} {cg:10.1%}")

# Test with different module strengths
print(f"\n--- Module strength sweep ---")
for strength in [0.1, 0.3, 0.5, 0.7, 0.9]:
    m = build_model("5_modules_of_10", module_strength=strength)
    be, bc, cg = measure_creativity(m)
    print(f"  Strength={strength:.1f}: base={be:.4f} cal={bc:.4f} gain={cg:.1%}")

report = {
    "results": [{"structure": s, "base_error": round(float(be),4), 
                 "calibrated_error": round(float(bc),4), "gain": round(float(cg),4)}
                for s, d, be, bc, cg in results],
}
with open(os.path.join(OUTPUT_DIR, "modularity_results.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {os.path.join(OUTPUT_DIR, 'modularity_results.json')}")
