"""
ExperimentalMatrix — Fatorial completo: arquitetura x otimizador x formalismo x dominio.

Roda N combinacoes e mede performance para cada uma.
Isolando fatores: qual realmente importa?

Fatores:
  - Dominio: Alzheimer, Parkinson, ALS, Huntington
  - Formalismo: ODE, PDE, Grafo, Agentes (via DynamicTypeLibrary)
  - Otimizador: Random, Grid, Hierarchical
  - Complexidade: 10, 35, 50 vars

Saida: matriz de resultados para analise estatistica.
"""
import sys, os, json, itertools, random, time, math, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp
from types import SimpleNamespace

OUTPUT_DIR = "D:/novamind_data/experimental_matrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DOMINIOS ───
DOMAINS = {
    "Alzheimer": {"cognitive_function": 0.770, "neurodegeneration": 0.249},
    "Parkinson": {"motor_function": 0.500, "quality_of_life": 0.600},
    "ALS": {"motor_function": 0.400, "quality_of_life": 0.500},
    "Huntington": {"cognitive_function": 0.600, "motor_function": 0.500},
}

# ─── FORMALISMOS (via DynamicTypeLibrary) ───
from core.dynamic_type_library import DynamicTypeLibrary

def make_model(formalism, domain, n_vars):
    """Cria modelo ESTAVEL com o formalismo especificado.
    
    Cada modelo e simples, linear, e NAO explode.
    """
    m = SimpleNamespace()
    m.name = f"{formalism}_{domain}_{n_vars}"
    m.variables = [f"var_{i}" for i in range(n_vars)]
    m.initial_state = np.array([0.5] * n_vars)
    m.domain = domain
    
    eps = 1e-6
    
    if formalism == "ode":
        # ODE linear: cada variavel decai independentemente
        def ode_linear(t, y):
            dy = np.zeros_like(y)
            for i in range(n_vars):
                dy[i] = -0.01 * y[i]
            return np.clip(dy, -1, 1)
        m.ode_fn = ode_linear
    
    elif formalism == "graph_network":
        # Rede com acoplamento fraco
        def ode_graph(t, y):
            dy = np.zeros_like(y)
            for i in range(n_vars):
                dy[i] = -0.01 * y[i]
                for j in range(n_vars):
                    if i != j:
                        dy[i] += 0.002 * (y[j] - y[i])
            return np.clip(dy, -1, 1)
        m.ode_fn = ode_graph
    
    elif formalism == "cellular_automaton":
        # Automato celular continuo: cada celula interage com vizinhos
        def ode_ca(t, y):
            dy = np.zeros_like(y)
            for i in range(n_vars):
                left = y[i-1] if i > 0 else 0
                right = y[i+1] if i < n_vars-1 else 0
                dy[i] = -0.01 * y[i] + 0.005 * (left + right - 2*y[i])
            return np.clip(dy, -1, 1)
        m.ode_fn = ode_ca
    
    elif formalism == "agent_based":
        # Agentes com interacao populacional (campo medio)
        def ode_agents(t, y):
            dy = np.zeros_like(y)
            for i in range(n_vars):
                mean_field = np.mean(y) - y[i]
                dy[i] = -0.01 * y[i] + 0.003 * mean_field
            return np.clip(dy, -1, 1)
        m.ode_fn = ode_agents
    
    else:
        return None
    
    return m

def compute_error(model, reality):
    if model is None: return 1.0
    try:
        sol = solve_ivp(model.ode_fn, (0, 365), model.initial_state,
                       method="LSODA", max_step=10, rtol=1e-1, atol=1e-2)
        if sol.y.shape[1] == 0: return 1.0
        errors = []
        # Map: first 2 variables of any model = {cognitive, motor/neurodeg}
        var_names = model.variables
        for obs_var, obs_val in reality.items():
            idx = -1
            if "cognitive" in obs_var and len(var_names) > 0:
                idx = 0
            elif "neurodeg" in obs_var and len(var_names) > 1:
                idx = 1
            elif "motor" in obs_var and len(var_names) > 1:
                idx = 1
            elif "quality" in obs_var and len(var_names) > 1:
                idx = 1
            
            if idx >= 0 and idx < sol.y.shape[0]:
                pred = float(sol.y[idx, -1])
                pred = max(0, min(1, 1/(1+abs(pred)) if abs(pred) > 1 else pred))
                if abs(obs_val) > 0.01:
                    errors.append(abs(pred - obs_val) / obs_val)
        return float(np.mean(errors)) if errors else 1.0
    except:
        return 1.0

def calibrate(model, reality, optimizer, n_trials=50):
    """Calibra modelo com o otimizador especificado."""
    base_err = compute_error(model, reality)
    best = base_err
    
    if optimizer == "none":
        return base_err, 0.0
    
    for _ in range(n_trials):
        m2 = SimpleNamespace()
        m2.variables = model.variables
        m2.initial_state = model.initial_state.copy()
        m2.ode_fn = model.ode_fn
        
        if optimizer == "random":
            scales = {v: random.uniform(0.1, 5.0) for v in model.variables[:3]}
            orig = model.ode_fn
            def wrap(o, sc):
                def fn(t, y):
                    dy = o(t, y)
                    for v, s in sc.items():
                        if v in m2.variables:
                            i = m2.variables.index(v)
                            if i < len(dy): dy[i] *= s
                    return dy
                return fn
            m2.ode_fn = wrap(orig, scales)
        elif optimizer == "grid":
            for mi in range(min(3, len(model.variables))):
                scale = [0.3, 0.5, 1.0, 2.0, 3.0][_ % 5]
                orig = m2.ode_fn
                def wrap2(o, vi, sc):
                    def fn(t, y):
                        dy = o(t, y)
                        if vi < len(dy): dy[vi] *= sc
                        return dy
                    return fn
                m2.ode_fn = wrap2(orig, mi, scale)
        
        err = compute_error(m2, reality)
        if err < best:
            best = err
    
    return base_err, (base_err - best) / max(0.001, base_err)


# ═══════════════════════════════════════════════════════════════
# EXECUCAO
# ═══════════════════════════════════════════════════════════════

FORMALISMS = ["ode", "graph_network", "cellular_automaton", "agent_based"]
OPTIMIZERS = ["none", "random", "grid"]
COMPLEXITIES = [10, 35, 50]
DOMAIN_NAMES = list(DOMAINS.keys())

print("="*70)
print("EXPERIMENTAL MATRIX")
print("="*70)
print(f"\nFatores:")
print(f"  Dominios: {DOMAIN_NAMES}")
print(f"  Formalisms: {FORMALISMS}")
print(f"  Optimizers: {OPTIMIZERS}")
print(f"  Complexities: {COMPLEXITIES}")
print(f"  Total combinacoes: {len(DOMAIN_NAMES) * len(FORMALISMS) * len(OPTIMIZERS) * len(COMPLEXITIES)}")

results = []
total = len(DOMAIN_NAMES) * len(FORMALISMS) * len(OPTIMIZERS) * len(COMPLEXITIES)
done = 0

for domain in DOMAIN_NAMES:
    for formalism in FORMALISMS:
        for optimizer in OPTIMIZERS:
            for n_vars in COMPLEXITIES:
                done += 1
                model = make_model(formalism, domain, n_vars)
                if model is None:
                    results.append({
                        "domain": domain, "formalism": formalism,
                        "optimizer": optimizer, "n_vars": n_vars,
                        "base_error": 1.0, "gain": 0.0, "error": "formalism_not_available"
                    })
                    continue
                base_err, gain = calibrate(model, DOMAINS[domain], optimizer, n_trials=30)
                
                results.append({
                    "domain": domain,
                    "formalism": formalism,
                    "optimizer": optimizer,
                    "n_vars": n_vars,
                    "base_error": round(float(base_err), 4),
                    "gain": round(float(gain), 4),
                })
                
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {domain:15s} {formalism:20s} {optimizer:10s} {n_vars:3d}v gain={gain:.2%}")

# ANALISE
print(f"\n{'='*70}")
print(f"ANALISE — Efeito principal de cada fator")
print(f"{'='*70}")

for factor in ["domain", "formalism", "optimizer", "n_vars"]:
    levels = set(r[factor] for r in results)
    print(f"\n  {factor.upper()}:")
    for level in sorted(levels, key=lambda l: -np.mean([r["gain"] for r in results if r[factor] == l])):
        gains = [r["gain"] for r in results if r[factor] == level]
        print(f"    {str(level):20s} gain medio={np.mean(gains):.2%} (std={np.std(gains):.2%})")

# MELHOR COMBINACAO
best = max(results, key=lambda r: r["gain"])
worst = min(results, key=lambda r: r["gain"])
print(f"\nMelhor combinacao: {best}")
print(f"Pior combinacao:   {worst}")

# Salva
report_path = os.path.join(OUTPUT_DIR, "experimental_matrix.json")
with open(report_path, "w") as f:
    json.dump({"results": results, "best": best, "worst": worst}, f, indent=2)
print(f"\nRelatorio: {report_path}")
