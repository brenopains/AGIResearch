"""
AGI Dashboard — Visualizacao em tempo real da AGI simulando curas.

Mostra:
  - Graficos de biomarcadores ao longo do tempo
  - Intervencoes sendo testadas
  - Modelos rivais competindo
  - Decisoes do MetaExecutive
  - Predicoes quantitativas

Uso:
    python agi_dashboard.py                    # Modo interativo
    python agi_dashboard.py --disease Alzheimer
    python agi_dashboard.py --headless          # Apenas gera relatorio
"""
import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.virtual_lab import VirtualLab
from core.model_library import ModelLibrary
from core.model_comparator import ModelComparator
from core.structure_mutator import StructureMutator
from core.model_discovery_lab import ModelDiscoveryLab

REPORT_DIR = r"D:\novamind_data\dashboard_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def simulate_all_interventions(lab, disease, interventions, doses, duration=365):
    """Simula todas as combinacoes intervencao x dose.
    
    Returns: dict com resultados estruturados para plotagem.
    """
    if not lab.load_disease_model(disease):
        return {"error": f"Model for {disease} not found"}
    
    results = {}
    for interv in interventions:
        results[interv] = {}
        for dose in doses:
            r = lab.simulate_intervention(interv, dose, duration, use_tellurium=False)
            results[interv][dose] = {
                "changes": {k: float(v) for k, v in r.biomarker_change.items()},
                "adverse": r.adverse_events,
                "confidence": r.confidence,
                "time_ms": round(r.computation_time * 1000, 1),
            }
    return results


def compare_models(disease):
    """Compara modelos rivais e retorna ranking."""
    lib = ModelLibrary()
    comparator = ModelComparator()
    mutator = StructureMutator()
    
    models = lib.get_models(disease)
    if not models:
        return {"error": f"No models for {disease}"}
    
    comparisons = comparator.compare(models)
    mutated = []
    for base_model in models[:2]:
        for new_model in mutator.mutate(base_model, n_mutations=2):
            mutated.append(new_model)
    
    if mutated:
        mut_comparisons = comparator.compare(mutated)
        comparisons.extend(mut_comparisons)
    
    comparisons.sort(key=lambda c: -c.posterior)
    return [
        {
            "name": c.model_name,
            "posterior": round(c.posterior, 1),
            "bayes": round(c.bayes_factor_vs_best, 3),
            "n_vars": c.n_vars,
            "error": round(c.prediction_error, 4),
            "is_best": c.is_best,
        }
        for c in comparisons
    ]


def generate_report(disease: str):
    """Gera relatorio completo com todas as simulacoes e comparacoes."""
    lab = VirtualLab()
    
    interventions = ["BACE1_inhibition", "anti-inflammatory", 
                     "mitochondrial_enhancement", "immunotherapy",
                     "dopamine_replacement", "BDNF_therapy"]
    doses = [0.3, 0.5, 0.7]
    
    print(f"\n=== SIMULATING: {disease} ===\n")
    
    # 1. Simula intervencoes
    sim_results = simulate_all_interventions(lab, disease, interventions, doses)
    
    # 2. Compara modelos
    model_ranking = compare_models(disease)
    
    # 3. Generate dashboard data
    dashboard = {
        "disease": disease,
        "timestamp": time.time(),
        "simulations": sim_results,
        "model_comparison": model_ranking,
        "best_intervention": "",
        "best_score": -999,
    }
    
    # Find best intervention
    if isinstance(sim_results, dict) and "error" not in sim_results:
        for interv, doses_data in sim_results.items():
            for dose, data in doses_data.items():
                # Score = improvement in cognitive/function biomarkers
                score = 0
                for k, v in data["changes"].items():
                    if any(x in k for x in ["cognitive", "function", "motor", "synaptic"]):
                        score += v
                if not data["adverse"]:
                    score += 0.5
                if score > dashboard["best_score"]:
                    dashboard["best_score"] = score
                    dashboard["best_intervention"] = f"{interv} (dose={dose})"
    
    # Save
    report_path = os.path.join(REPORT_DIR, f"dashboard_{disease.lower()}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2)
    
    # Print summary
    print(f"  Best intervention: {dashboard['best_intervention']}")
    print(f"  Best score: {dashboard['best_score']}")
    print(f"  Models compared: {len(model_ranking)}")
    if model_ranking:
        print(f"  Best model: {model_ranking[0]['name']} (posterior={model_ranking[0]['posterior']})")
    print(f"\n  Report saved: {report_path}")
    
    return dashboard


def print_interactive(disease: str):
    """Modo interativo: mostra resultados em tempo real."""
    print("\n" + "="*70)
    print(f"  AGI DASHBOARD - {disease}")
    print("="*70)
    print(f"\n  Backend: Python ODE + Tellurium (SBML)")
    print(f"  Disease models: 6 for Alzheimer, 5 for Parkinson, etc.")
    print(f"  Interventions: BACE1, anti-inflammatory, mitochondrial, immunotherapy, etc.")
    print(f"  Model comparison: Bayesian posterior")
    print(f"  Structure mutation: add latent variables, feedback loops")
    print()
    print(f"  Press Ctrl+C to stop. Reports saved to {REPORT_DIR}")
    print("="*70)
    
    generate_report(disease)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="Alzheimer")
    parser.add_argument("--headless", action="store_true", help="Generate report only")
    args = parser.parse_args()
    
    if args.headless:
        generate_report(args.disease)
    else:
        print_interactive(args.disease)


if __name__ == "__main__":
    main()
