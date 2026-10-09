"""
test_structural_correlation.py — Correlaciona metricas ESTRUTURAIS com criatividade.

Pergunta: "O que realmente importa? Modularidade? Clustering? Entropia?"

Para cada modelo (10, 20, 35, 50, 70, 100 vars, modulos, etc.):
  1. Mede perfil estrutural (modularity, clustering, small-world, entropy, etc.)
  2. Mede criatividade (calibration gain, mutation success)
  3. Correlaciona: qual metrica estrutural PREDIZ criatividade?
"""
import sys, json, os, copy, random, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp

OUTPUT_DIR = "D:/novamind_data/structural_correlation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REALITY = {"cognitive_function": 0.770, "neurodegeneration": 0.249}

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
    base_err = compute_error(model)
    best_cal = base_err
    for _ in range(20):
        m = copy.deepcopy(model)
        scales = {v: random.uniform(0.1, 5.0) for v in model.variables[:4]}
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
    return float(base_err), float(best_cal), float(cal_gain)

from core.structural_metrics import StructuralAnalyzer
from core.model_library import ModelLibrary

analyzer = StructuralAnalyzer()
lib = ModelLibrary()
base = [m for m in lib.get_models("Alzheimer") if m.name == "dual_state"][0]

# Gera modelos com diferentes complexidades e modularidades
all_models = [base]
all_models.append(copy.deepcopy(base))  # baseline

# 10, 20, 35, 50, 70, 100 vars normalized
from test_complexity_experiment import create_expanded_model, LATENT_VARIABLES
for n in [20, 35, 50, 70, 100]:
    all_models.append(create_expanded_model(base, n))

# Modelos modulares (50 vars em 5 modulos)
from test_modularity_experiment import build_model
for struct in ["all_connected", "5_modules_of_10", "10_modules_of_5"]:
    all_models.append(build_model(struct))

print("="*65)
print("STRUCTURAL METRICS VS CREATIVITY")
print("="*65)
print(f"{'Modelo':30s} {'Mod':>6s} {'Clust':>6s} {'SW':>6s} {'Entrop':>6s} {'Phi':>6s} {'Comp':>6s} {'Gain':>8s}")
print(f"{'-'*75}")

results = []
for m in all_models:
    try:
        profile = analyzer.analyze(m)
        be, bc, gain = measure_creativity(m)
        results.append((profile, gain))
        print(f"{m.name[:28]:28s} {profile.modularity:6.2f} {profile.clustering:6.2f} "
              f"{profile.small_worldness:6.2f} {profile.entropy:6.2f} "
              f"{profile.integrated_info:6.2f} {profile.compressibility:6.2f} "
              f"{gain:8.1%}")
    except Exception as e:
        print(f"{m.name[:28]:28s} ERRO: {str(e)[:40]}")

# Correlacoes
print(f"\n{'='*65}")
print(f"CORRELACOES COM CRIATIVIDADE")
print(f"{'='*65}")

if results:
    metrics = ["modularity", "clustering", "small_worldness", "entropy", 
               "integrated_info", "compressibility"]
    gains = [r[1] for r in results]
    
    for metric in metrics:
        vals = [getattr(r[0], metric) for r in results]
        if len(vals) > 1 and len(gains) > 1:
            corr = np.corrcoef(vals, gains)[0, 1]
            print(f"  {metric:20s}: r = {corr:.3f}")
    
    # Qual estrutura prediz melhor?
    print(f"\n  Metrica que MAIS prediz criatividade:")
    best_metric = max(metrics, key=lambda m: abs(np.corrcoef(
        [getattr(r[0], m) for r in results], gains)[0, 1]) if len(gains) > 1 else 0)
    best_r = np.corrcoef([getattr(r[0], best_metric) for r in results], gains)[0, 1]
    print(f"    {best_metric}: r={best_r:.3f}")

report = {"results": [{"name": r[0].name, "gain": r[1], **r[0].to_dict()} for r in results]}
report_path = os.path.join(OUTPUT_DIR, "structural_correlation.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
