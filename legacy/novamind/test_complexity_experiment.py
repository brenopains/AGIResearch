"""
Experiment: complexity scaling — 10→20→35→60→100 vars.
Measures how creativity (calibration gain, mutation success, DMN efficacy)
scales with model richness.
"""
import sys, json, time, copy, random, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from scipy.integrate import solve_ivp

REALITY = {"cognitive_function": 0.770, "neurodegeneration": 0.249}
OUTPUT_DIR = "D:/novamind_data/complexity_experiment"
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Banco de variaveis latentes com significado biologico real
LATENT_VARIABLES = [
    # Grupo 1 (10 vars extras): vias metabolicas
    ("glymphatic_clearance", ["amyloid_beta", "cognitive_function"]),
    ("autophagy", ["p_tau", "neurodegeneration"]),
    ("complement_activation", ["synaptic_function", "neuroinflammation"]),
    ("calcium_dysregulation", ["mitochondrial_function", "neurodegeneration"]),
    ("epigenetic_age", ["oxidative_stress", "cognitive_function"]),
    ("vascular_dysfunction", ["cognitive_function", "oxidative_stress"]),
    ("circadian_disruption", ["cognitive_function", "neuroinflammation"]),
    ("olfactory_bulb_atrophy", ["cognitive_function", "neurodegeneration"]),
    ("noradrenaline_depletion", ["cognitive_function", "neuroinflammation"]),
    ("sleep_fragmentation", ["amyloid_beta", "glymphatic_clearance"]),
    # Grupo 2 (15 vars): sinalizacao celular
    ("wnt_signaling", ["synaptic_function", "neurodegeneration"]),
    ("notch_pathway", ["neurogenesis", "synaptic_function"]),
    ("nrf2_response", ["oxidative_stress", "neuroinflammation"]),
    ("nfkb_pathway", ["neuroinflammation", "apoptosis"]),
    ("creb_signaling", ["cognitive_function", "synaptic_function"]),
    ("mTOR_pathway", ["autophagy", "synaptic_function"]),
    ("ampk_energy", ["mitochondrial_function", "autophagy"]),
    ("sirt1_aging", ["epigenetic_age", "mitochondrial_function"]),
    ("igf1_signaling", ["neurogenesis", "cognitive_function"]),
    ("bdnf_trkb", ["synaptic_function", "cognitive_function"]),
    ("glutamate_excitotoxicity", ["neurodegeneration", "calcium_dysregulation"]),
    ("gaba_inhibition", ["neuroinflammation", "synaptic_function"]),
    ("dopamine_modulation", ["cognitive_function", "motivation"]),
    ("acetylcholine_release", ["cognitive_function", "attention"]),
    ("serotonin_regulation", ["mood", "cognitive_function"]),
    # Grupo 3 (15 vars): microglia e imunidade
    ("trem2_signaling", ["microglial_activation", "amyloid_beta"]),
    ("nlrp3_inflammasome", ["neuroinflammation", "pyroptosis"]),
    ("c1q_complement", ["synaptic_pruning", "synaptic_function"]),
    ("cx3cr1_signaling", ["microglial_activation", "synaptic_function"]),
    ("cd33_sialic_acid", ["microglial_activation", "amyloid_beta"]),
    ("apoe_lipidation", ["amyloid_beta", "cholesterol_metabolism"]),
    ("trem2_variants", ["microglial_activation", "phagocytosis"]),
    ("adam10_sheddase", ["amyloid_beta", "synaptic_function"]),
    ("bace1_activity", ["amyloid_beta", "app_processing"]),
    ("gamma_secretase", ["amyloid_beta", "notch_pathway"]),
    ("neprilysin_degradation", ["amyloid_beta", "glymphatic_clearance"]),
    ("ide_degradation", ["amyloid_beta", "insulin_metabolism"]),
    ("lrp1_transport", ["amyloid_beta", "vascular_dysfunction"]),
    ("rage_signaling", ["amyloid_beta", "neuroinflammation"]),
    ("p2x7_purinergic", ["neuroinflammation", "nlrp3_inflammasome"]),
    # Grupo 4 (10 vars): tau e citoesqueleto
    ("gsk3b_activity", ["p_tau", "wnt_signaling"]),
    ("cdk5_deregulation", ["p_tau", "neurodegeneration"]),
    ("pp2a_phosphatase", ["p_tau", "synaptic_function"]),
    ("mark1_microtubule", ["p_tau", "axonal_transport"]),
    ("dyrk1a_kinase", ["p_tau", "neurodegeneration"]),
    ("tau_acetylation", ["p_tau", "proteasome"]),
    ("tau_truncation", ["p_tau", "mitochondrial_function"]),
    ("hdac6_deacetylase", ["p_tau", "autophagy"]),
    ("hsp70_chaperone", ["p_tau", "protein_quality_control"]),
    ("ubiquitin_proteasome", ["p_tau", "autophagy"]),
    # Grupo 5 (10 vars): funcao sinaptica e rede
    ("psd95_scaffold", ["synaptic_function", "glutamate_excitotoxicity"]),
    ("shank_scaffold", ["synaptic_function", "autism_pathway"]),
    ("neuroligin_neurexin", ["synaptic_function", "autism_pathway"]),
    ("synaptophysin", ["synaptic_function", "cognitive_function"]),
    ("complexin_snare", ["neurotransmitter_release", "synaptic_function"]),
    ("synapsin_phosphorylation", ["neurotransmitter_release", "synaptic_function"]),
    ("ampareceptor_trafficking", ["synaptic_function", "cognitive_function"]),
    ("nmda_receptor_function", ["synaptic_function", "calcium_dysregulation"]),
    ("ltd_long_term_depression", ["synaptic_function", "cognitive_function"]),
    ("ltp_long_term_potentiation", ["synaptic_function", "cognitive_function"]),
]

def create_expanded_model(base_model, n_vars_target):
    """Expande modelo base para n_vars_target adicionando variaveis latentes."""
    new = copy.deepcopy(base_model)
    new.name = f"{base_model.name}_complex_{n_vars_target}"
    new.parent_model = base_model.name
    new.is_mutated = True
    
    # Variaveis base
    base_vars = list(base_model.variables)
    base_init = list(base_model.initial_state)
    
    # Quantas variaveis adicionar
    n_add = n_vars_target - len(base_vars)
    if n_add <= 0:
        return base_model
    
    # Seleciona as primeiras N variaveis latentes
    latents = LATENT_VARIABLES[:n_add]
    n_base = len(base_vars)
    
    new.variables = base_vars + [lv[0] for lv in latents]
    new.initial_state = np.append(base_init, [0.5] * len(latents))
    
    ode_orig = base_model.ode_fn
    
    def make_ode(orig, n_base, latents):
        n_lat = len(latents)
        def ode_complex(t, y):
            dy = list(orig(t, y[:n_base]))
            for i, (name, interacts_with) in enumerate(latents):
                idx = n_base + i
                d_latent = 0.01 * y[idx]
                for target in interacts_with:
                    if target in base_vars:
                        t_idx = base_vars.index(target)
                        # Influencia NORMALIZADA: independe de n_lat
                        d_latent += 0.02 * y[t_idx] / max(1, n_lat)
                        if t_idx < len(dy):
                            dy[t_idx] += 0.01 * y[idx] / max(1, n_lat)
                d_latent -= 0.01 * y[idx]
                dy.append(d_latent)
            return np.array(dy)
        return ode_complex
    
    new.ode_fn = make_ode(ode_orig, n_base, latents)
    new.mutation_description = f"Expanded to {n_vars_target} variables (+{n_add} biological mechanisms)"
    return new


def measure_creativity(model, n_trials=20):
    """Mede metricas de criatividade para um modelo:
    - Erro base
    - Ganho de calibracao (melhor de N tentativas)
    - Sensibilidade a mutacao
    """
    base_err = compute_error(model)
    
    # Calibration gain: best of N random parameter scalings
    best_cal_err = base_err
    cal_attempts = 0
    for _ in range(n_trials):
        scales = {v: random.uniform(0.1, 5.0) for v in model.variables[:4]}
        cal_model = copy.deepcopy(model)
        cal_model.name = f"{model.name}_cal"
        indices = {v: model.variables.index(v) for v in scales if v in model.variables}
        ode_orig = model.ode_fn
        def make_cal(o, idx, sc):
            def fn(t, y):
                dy = o(t, y)
                for v, i in idx.items():
                    if i < len(dy):
                        dy[i] *= sc.get(v, 1.0)
                return dy
            return fn
        cal_model.ode_fn = make_cal(ode_orig, indices, scales)
        err = compute_error(cal_model)
        cal_attempts += 1
        if err < best_cal_err:
            best_cal_err = err
    
    cal_gain = (base_err - best_cal_err) / max(0.001, base_err) if base_err > 0 else 0
    
    # Mutation sensitivity: how many random mutations improve the model?
    mutation_successes = 0
    mutation_total = 0
    from core.deep_structure_mutator import DeepStructureMutator
    mutator = DeepStructureMutator()
    
    for _ in range(min(10, n_trials)):
        children = mutator.mutate(model, n_mutations=2)
        mutation_total += len(children)
        for child in children:
            cerr = compute_error(child)
            if cerr < base_err:
                mutation_successes += 1
    
    mutation_rate = mutation_successes / max(1, mutation_total) if mutation_total > 0 else 0
    
    return {
        "n_vars": len(model.variables),
        "base_error": round(float(base_err), 4),
        "best_calibrated_error": round(float(best_cal_err), 4),
        "calibration_gain": round(float(cal_gain), 4),
        "mutation_success_rate": round(float(mutation_rate), 4),
        "mutation_total": mutation_total,
    }


print("="*65)
print("COMPLEXITY SCALING EXPERIMENT")
print("="*65)

from core.model_library import ModelLibrary
lib = ModelLibrary()
base = [m for m in lib.get_models("Alzheimer") if m.name == "dual_state"][0]

complexities = [10, 20, 35, 50, 70, 100]
results = []

print(f"\nBase model: {base.name} ({len(base.variables)} vars)")
print(f"Base error: {compute_error(base):.4f}")
print()

for n_vars in complexities:
    model = create_expanded_model(base, n_vars)
    print(f"\n--- Complexity: {n_vars} variables ---")
    metrics = measure_creativity(model, n_trials=15)
    results.append(metrics)
    print(f"  Base error:           {metrics['base_error']:.4f}")
    print(f"  Best calibration:     {metrics['best_calibrated_error']:.4f}")
    print(f"  Calibration gain:     {metrics['calibration_gain']:.1%}")
    print(f"  Mutation success:     {metrics['mutation_success_rate']:.1%}")
    print(f"  Vars added:           {n_vars - 10} ({n_vars} total)")

print(f"\n{'='*65}")
print(f"CURVA DE COMPLEXIDADE VS CRIATIVIDADE")
print(f"{'='*65}")
print(f"{'Variaveis':>10s} {'Erro base':>10s} {'Erro calib':>10s} {'Gain cal':>10s} {'Sucesso mut':>10s}")
print(f"{'-'*50}")
for r in results:
    print(f"{r['n_vars']:10d} {r['base_error']:10.4f} {r['best_calibrated_error']:10.4f} "
          f"{r['calibration_gain']:10.1%} {r['mutation_success_rate']:10.1%}")

# Report
report_path = os.path.join(OUTPUT_DIR, "complexity_curve.json")
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRelatorio: {report_path}")
