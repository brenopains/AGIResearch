"""
Expert Comparison Experiment: MetaModeler vs known best practices.

20 biomedical problems with known best formalisms from literature.
MetaModeler chooses autonomously. Results compared against expert consensus.

This tests whether the MetaModeler's decisions are:
  (A) Genuine inference from problem properties
  (B) Heuristic artifacts
  (C) Benchmark-specific
"""
import sys, json, os
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from core.meta_modeler import MetaModeler

OUTPUT_DIR = "D:/novamind_data/expert_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 20 PROBLEMS WITH KNOWN BEST FORMALISMS ───
PROBLEMS = [
    # (name, disease/domain, reality_dict, expert_choice, reasoning)
    {
        "id": "P01",
        "name": "Amyloid plaque formation in Alzheimer",
        "domain": "Alzheimer",
        "reality": {"amyloid_beta": 0.300, "cognitive_function": 0.770},
        "expert_formalism": "ode",
        "expert_reasoning": "Chemical kinetics of aggregation. Well-mixed, continuous. ODE is standard.",
    },
    {
        "id": "P02",
        "name": "Tau propagation along neural connectome",
        "domain": "tau_propagation",
        "reality": {"tau_spread": 0.400, "neuron_loss": 0.500},
        "expert_formalism": "graph_network",
        "expert_reasoning": "Tau spreads via synaptic connections. Network/graph model captures connectivity.",
    },
    {
        "id": "P03",
        "name": "Microglial immune response to amyloid",
        "domain": "immunology",
        "reality": {"microglial_activation": 0.600, "cytokine_levels": 0.500},
        "expert_formalism": "agent_based",
        "expert_reasoning": "Microglia act as individual agents with local rules. ABM captures heterogeneity.",
    },
    {
        "id": "P04",
        "name": "Calcium wave propagation in astrocyte networks",
        "domain": "calcium_signaling",
        "reality": {"calcium_concentration": 0.700, "wave_speed": 0.500},
        "expert_formalism": "pde",
        "expert_reasoning": "Calcium waves are spatiotemporal. Reaction-diffusion PDE captures wave dynamics.",
    },
    {
        "id": "P05",
        "name": "Dopamine neuron degeneration in Parkinson",
        "domain": "Parkinson",
        "reality": {"dopamine_levels": 0.400, "motor_function": 0.600},
        "expert_formalism": "ode",
        "expert_reasoning": "Gradual loss of neurons. Continuous, well-mixed. ODE sufficient.",
    },
    {
        "id": "P06",
        "name": "Cortical spreading depression in migraine",
        "domain": "neurology",
        "reality": {"depolarization_wave": 0.500, "recovery_time": 0.500},
        "expert_formalism": "cellular_automaton",
        "expert_reasoning": "Wave of depolarization across cortical surface. CA captures excitable media.",
    },
    {
        "id": "P07",
        "name": "Tumor growth with angiogenesis",
        "domain": "oncology",
        "reality": {"tumor_volume": 0.300, "vascular_density": 0.500},
        "expert_formalism": "hybrid",
        "expert_reasoning": "Multiscale: molecular VEGF, cellular proliferation, tissue angiogenesis. Hybrid needed.",
    },
    {
        "id": "P08",
        "name": "Prion protein misfolding cascade",
        "domain": "prion",
        "reality": {"misfolded_protein": 0.200, "aggregation_rate": 0.500},
        "expert_formalism": "ode",
        "expert_reasoning": "Nucleation-polymerization kinetics. Well-mixed, continuous. ODE is standard.",
    },
    {
        "id": "P09",
        "name": "EEG seizure propagation in epilepsy",
        "domain": "epilepsy",
        "reality": {"seizure_spread": 0.500, "inhibition_loss": 0.500},
        "expert_formalism": "graph_network",
        "expert_reasoning": "Seizure travels through brain network. Graph captures connectivity and hubs.",
    },
    {
        "id": "P10",
        "name": "Circadian rhythm disruption in aging",
        "domain": "circadian",
        "reality": {"clock_gene_expression": 0.600, "sleep_quality": 0.500},
        "expert_formalism": "ode",
        "expert_reasoning": "Genetic oscillator with delay. ODE with Goodwin oscillator is standard.",
    },
    {
        "id": "P11",
        "name": "Metastatic cell migration through tissue",
        "domain": "metastasis",
        "reality": {"migration_speed": 0.500, "invasion_depth": 0.300},
        "expert_formalism": "agent_based",
        "expert_reasoning": "Individual cells moving through ECM. ABM captures heterogeneity and local decisions.",
    },
    {
        "id": "P12",
        "name": "Neurotransmitter diffusion at synapse",
        "domain": "synaptic",
        "reality": {"neurotransmitter_concentration": 0.700, "receptor_binding": 0.600},
        "expert_formalism": "pde",
        "expert_reasoning": "Diffusion in synaptic cleft. Reaction-diffusion PDE captures spatiotemporal dynamics.",
    },
    {
        "id": "P13",
        "name": "Insulin-glucose homeostasis",
        "domain": "metabolism",
        "reality": {"blood_glucose": 0.500, "insulin_levels": 0.500},
        "expert_formalism": "ode",
        "expert_reasoning": "Well-mixed compartment model. ODE is standard for metabolic pathways.",
    },
    {
        "id": "P14",
        "name": "Programmed cell death (apoptosis) signaling",
        "domain": "apoptosis",
        "reality": {"caspase_activity": 0.500, "cell_survival": 0.700},
        "expert_formalism": "ode",
        "expert_reasoning": "Biochemical signaling cascade. ODE is standard for signaling pathways.",
    },
    {
        "id": "P15",
        "name": "Gene regulatory network in development",
        "domain": "gene_regulation",
        "reality": {"gene_expression": 0.600, "transcription_rate": 0.500},
        "expert_formalism": "graph_network",
        "expert_reasoning": "Genes regulate each other via network. Boolean/ODE network captures regulation.",
    },
    {
        "id": "P16",
        "name": "Population dynamics of gut microbiome",
        "domain": "microbiome",
        "reality": {"species_diversity": 0.600, "total_bacteria": 0.800},
        "expert_formalism": "ode",
        "expert_reasoning": "Species competition in well-mixed environment. Lotka-Volterra ODE is standard.",
    },
    {
        "id": "P17",
        "name": "HIV viral dynamics with immune response",
        "domain": "HIV",
        "reality": {"viral_load": 0.200, "t_cell_count": 0.400},
        "expert_formalism": "ode",
        "expert_reasoning": "Well-mixed virus-immune dynamics. ODE (Nowak-May) is standard.",
    },
    {
        "id": "P18",
        "name": "Cardiac action potential propagation",
        "domain": "cardiac",
        "reality": {"membrane_potential": 0.700, "conduction_velocity": 0.600},
        "expert_formalism": "pde",
        "expert_reasoning": "Electrical wave through heart tissue. Reaction-diffusion PDE (FitzHugh-Nagumo) is standard.",
    },
    {
        "id": "P19",
        "name": "Synaptic plasticity and long-term potentiation",
        "domain": "plasticity",
        "reality": {"synaptic_strength": 0.500, "ltp_magnitude": 0.500},
        "expert_formalism": "ode",
        "expert_reasoning": "Biochemical cascades in single synapse. ODE is standard for signaling.",
    },
    {
        "id": "P20",
        "name": "Immune checkpoint blockade in melanoma",
        "domain": "immunotherapy",
        "reality": {"tumor_size": 0.300, "t_cell_infiltration": 0.400},
        "expert_formalism": "hybrid",
        "expert_reasoning": "Molecular (checkpoint), cellular (T cells), tissue (tumor). Hybrid captures multiscale.",
    },
]

# ─── RUN ───
print("="*70)
print("EXPERT COMPARISON: MetaModeler vs Human Experts")
print("="*70)

mm = MetaModeler()
results = []
matches = 0
total = len(PROBLEMS)

for p in PROBLEMS:
    print(f"\n  {p['id']}: {p['name']}")
    print(f"    Expert: {p['expert_formalism']}")

    # MetaModeler decides
    mm_result = mm.solve(p["domain"], p["reality"])
    mm_choice = mm_result["best_formalism"]

    match = mm_choice == p["expert_formalism"]
    if match:
        matches += 1

    print(f"    MetaModeler: {mm_choice} {'[MATCH]' if match else '[DIFFERS]'}")
    print(f"    Reasoning: {mm_result['reasoning'][-1][:80]}")

    results.append({
        "id": p["id"],
        "name": p["name"],
        "expert": p["expert_formalism"],
        "metamodeler": mm_choice,
        "match": match,
        "expert_reasoning": p["expert_reasoning"],
        "metamodeler_reasoning": mm_result["reasoning"][-1] if mm_result["reasoning"] else "",
    })

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Problems: {total}")
print(f"  Matches:  {matches} ({matches/total*100:.0f}%)")
print(f"  Mismatches: {total - matches}")

# Analyze by formalism
from collections import Counter
expert_counts = Counter(p["expert_formalism"] for p in PROBLEMS)
mm_counts = Counter(r["metamodeler"] for r in results)

print(f"\n  Expert formalism distribution:")
for f, c in expert_counts.most_common():
    mm_c = mm_counts.get(f, 0)
    print(f"    {f:20s}: expert={c:2d} metamodeler={mm_c:2d}")

# Mismatches detail
print(f"\n  Mismatches detail:")
for r in results:
    if not r["match"]:
        print(f"    {r['id']}: expert={r['expert']:15s} mm={r['metamodeler']:15s}")
        print(f"      Expert: {r['expert_reasoning'][:70]}")
        print(f"      MM:     {r['metamodeler_reasoning'][:70]}")

# Conclusion
if matches / total > 0.7:
    print(f"\n  >>> MetaModeler concorda com especialistas em >70% dos casos.")
elif matches / total > 0.5:
    print(f"\n  >>> MetaModeler concorda com especialistas em >50% dos casos (moderado).")
else:
    print(f"\n  >>> MetaModeler concorda pouco com especialistas. Heuristicas podem precisar de ajuste.")

# Save
report_path = os.path.join(OUTPUT_DIR, "expert_comparison.json")
with open(report_path, "w") as f:
    json.dump({"results": results, "summary": {"total": total, "matches": matches,
               "accuracy": round(matches/total, 3)}}, f, indent=2)
print(f"\nRelatorio: {report_path}")
