"""
Generalization Test: train on Alzheimer, test on Parkinson, ALS, Huntington.
No hyperparameter changes between diseases.

Hypothesis: if the architecture finds general principles, 
the best formalism/optimizer/complexity for Alzheimer should
also work well on other diseases.
"""
import sys, json, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

from test_experimental_matrix import make_model, compute_error, calibrate, DOMAINS, DOMAIN_NAMES

print("="*70)
print("GENERALIZATION TEST")
print("="*70)

# Step 1: Train on Alzheimer — find best config
print(f"\n--- FASE 1: TREINO em Alzheimer ---")
alz_results = []
for formalism in ["ode", "graph_network", "cellular_automaton"]:
    for optimizer in ["random", "grid"]:
        for n_vars in [10, 35]:
            model = make_model(formalism, "Alzheimer", n_vars)
            base_err, gain = calibrate(model, DOMAINS["Alzheimer"], optimizer, n_trials=30)
            alz_results.append({
                "formalism": formalism, "optimizer": optimizer, "n_vars": n_vars,
                "gain": gain, "base_error": base_err
            })
            print(f"  {formalism:20s} {optimizer:10s} {n_vars:3d}v gain={gain:.2%}")

best_config = max(alz_results, key=lambda r: r["gain"])
print(f"\n  Melhor config (Alzheimer): {best_config['formalism']} + {best_config['optimizer']} + {best_config['n_vars']}v (gain={best_config['gain']:.2%})")
worst_config = min(alz_results, key=lambda r: r["gain"])
print(f"  Pior config (Alzheimer):   {worst_config['formalism']} + {worst_config['optimizer']} + {worst_config['n_vars']}v (gain={worst_config['gain']:.2%})")

# Step 2: Test best config on other diseases
print(f"\n--- FASE 2: TESTE em outras doencas (config FIXA) ---")

generalization_results = []
for domain in ["Parkinson", "ALS", "Huntington"]:
    # Test best config
    model_best = make_model(best_config["formalism"], domain, best_config["n_vars"])
    _, gain_best = calibrate(model_best, DOMAINS[domain], best_config["optimizer"], n_trials=30)
    
    # Test worst config
    model_worst = make_model(worst_config["formalism"], domain, worst_config["n_vars"])
    _, gain_worst = calibrate(model_worst, DOMAINS[domain], worst_config["optimizer"], n_trials=30)
    
    # Test average of all configs on this domain
    all_gains = []
    for r in alz_results:
        m = make_model(r["formalism"], domain, r["n_vars"])
        _, g = calibrate(m, DOMAINS[domain], r["optimizer"], n_trials=30)
        all_gains.append(g)
    
    generalization_results.append({
        "domain": domain,
        "best_config_gain": round(float(gain_best), 4),
        "worst_config_gain": round(float(gain_worst), 4),
        "avg_gain": round(float(np.mean(all_gains)), 4),
        "best_transfer": gain_best > np.mean(all_gains),
    })
    
    print(f"\n  {domain:15s}:")
    print(f"    Melhor config (Alzheimer): gain={gain_best:.2%}")
    print(f"    Pior config (Alzheimer):   gain={gain_worst:.2%}")
    print(f"    Media todas configs:       {np.mean(all_gains):.2%}")
    print(f"    Transferencia:             {'SIM' if gain_best > np.mean(all_gains) else 'NAO'}")

# Summary
print(f"\n{'='*70}")
print(f"RESUMO — GENERALIZACAO")
print(f"{'='*70}")
transfers = sum(1 for r in generalization_results if r["best_transfer"])
print(f"  Transferencia positiva: {transfers}/{len(generalization_results)} dominios")
if transfers == len(generalization_results):
    print(f"  A configuracao otima para Alzheimer GENERALIZA para outras doencas!")
elif transfers > 0:
    print(f"  Generalizacao PARCIAL: funciona em {transfers} de {len(generalization_results)} dominios")
else:
    print(f"  NENHUMA generalizacao detectada. A configuracao otima e especifica do dominio.")

report = {
    "alzheimer_best": best_config,
    "alzheimer_worst": worst_config,
    "generalization": generalization_results,
}
report_path = "D:/novamind_data/generalization_test.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
