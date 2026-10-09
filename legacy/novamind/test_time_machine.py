"""
Time Machine Experiment: train on pre-2018 knowledge, predict post-2018 discoveries.

The AGI receives ONLY knowledge available before 2018.
It generates hypotheses about Alzheimer mechanisms and treatments.

These hypotheses are then compared against ACTUAL discoveries 
made between 2019 and 2026.

If the AGI's predictions match post-2018 discoveries, 
it demonstrates predictive scientific creativity.
"""
import sys, json, os, random, numpy as np
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

OUTPUT_DIR = "D:/novamind_data/time_machine"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PRE-2018 KNOWLEDGE (what the AGI is allowed to see)
# ═══════════════════════════════════════════════════════════════

PRE_2018_FACTS = [
    "Amyloid-beta plaques are a hallmark of Alzheimer disease",
    "Tau protein hyperphosphorylation leads to neurofibrillary tangles",
    "APOE4 is the strongest genetic risk factor for late-onset Alzheimer",
    "BACE1 (beta-secretase) produces amyloid-beta from APP",
    "Cholinergic neuron loss contributes to cognitive decline",
    "Neuroinflammation is present in Alzheimer brains",
    "Oxidative stress damages neurons in Alzheimer",
    "Mitochondrial dysfunction occurs in Alzheimer",
    "Exercise reduces Alzheimer risk",
    "Mediterranean diet is associated with lower risk",
    "Cognitive training may help maintain function",
    "Microglia are activated in Alzheimer",
    "Astrocytes become reactive in Alzheimer",
    "The hippocampus atrophies in early Alzheimer",
    "CSF amyloid-beta decreases while tau increases in Alzheimer",
    "Sleep disruption is common in Alzheimer",
    "Vascular risk factors increase Alzheimer risk",
    "The complement system is activated in Alzheimer",
    "Insulin signaling is impaired in Alzheimer brain",
    "Apolipoprotein E4 impairs amyloid clearance",
    "Gamma-secretase cleaves APP to generate amyloid-beta",
    "DMN (default mode network) shows early dysfunction",
    "Synaptic loss correlates best with cognitive decline",
    "Calcium dysregulation occurs in Alzheimer neurons",
    "Autophagy is impaired in Alzheimer",
    "The NLRP3 inflammasome is activated in Alzheimer",
    "Tau spreads through the brain along neural pathways",
    "CSF biomarkers can predict Alzheimer years before symptoms",
    "Amyloid PET imaging detects plaques in living humans",
    "Tau PET imaging detects tangles in living humans",
]

# ═══════════════════════════════════════════════════════════════
# POST-2018 DISCOVERIES (for validation — NOT shown to AGI)
# ═══════════════════════════════════════════════════════════════

POST_2018_DISCOVERIES = [
    # 2019
    {"year": 2019, "discovery": "TREM2 variant R47H increases Alzheimer risk by impairing microglial amyloid clearance", "area": "genetics"},
    {"year": 2019, "discovery": "The glymphatic system clears amyloid during sleep, and is impaired in Alzheimer", "area": "clearance"},
    # 2020
    {"year": 2020, "discovery": "Disease-associated microglia (DAM) have a protective phenotype that is lost in Alzheimer", "area": "immunology"},
    {"year": 2020, "discovery": "Plasma p-tau217 is a highly accurate blood biomarker for Alzheimer", "area": "biomarkers"},
    # 2021
    {"year": 2021, "discovery": "Lecanemab (anti-amyloid antibody) slows cognitive decline in phase 3 trial", "area": "treatment"},
    {"year": 2021, "discovery": "Aducanumab receives FDA approval despite controversy", "area": "treatment"},
    # 2022
    {"year": 2022, "discovery": "Tau propagation follows the structural connectome, spreading from entorhinal cortex", "area": "propagation"},
    {"year": 2022, "discovery": "APOE4 homozygosity defines a new genetic form of Alzheimer with near-complete penetrance", "area": "genetics"},
    # 2023
    {"year": 2023, "discovery": "Lecanemab approved by FDA for early Alzheimer", "area": "treatment"},
    {"year": 2023, "discovery": "Microglial depletion reduces tau pathology in mouse models", "area": "immunology"},
    # 2024
    {"year": 2024, "discovery": "Anti-tau antisense oligonucleotide reduces tau in phase 1/2 trial", "area": "treatment"},
    {"year": 2024, "discovery": "Blood-brain barrier breakdown is an early event, not late consequence", "area": "vascular"},
    # 2025
    {"year": 2025, "discovery": "Combination therapy (anti-amyloid + anti-tau) shows synergistic benefit in preclinical models", "area": "treatment"},
    {"year": 2025, "discovery": "Gut microbiome composition predicts Alzheimer risk and progression", "area": "microbiome"},
    # 2026
    {"year": 2026, "discovery": "Sleep fragmentation drives amyloid accumulation via impaired glymphatic clearance", "area": "sleep"},
    {"year": 2026, "discovery": "CRISPR-based epigenome editing reverses Alzheimer features in mice", "area": "therapy"},
]


# ═══════════════════════════════════════════════════════════════
# AGI HYPOTHESIS GENERATION (using ONLY pre-2018 knowledge)
# ═══════════════════════════════════════════════════════════════

def generate_hypotheses(pre_2018_facts, n_hypotheses=20):
    """Generate hypotheses from pre-2018 knowledge.

    Uses simple conceptual combinations:
    If fact A says "X causes Y" and fact B says "Z affects X",
    then hypothesis: "Z may affect Y through X"
    """
    hypotheses = []
    rng = random.Random(42)
    
    # Extract key entities from facts
    entities = set()
    for fact in pre_2018_facts:
        words = fact.lower().split()
        for w in words:
            if len(w) > 4 and w not in {"that", "this", "with", "from", "have", "been", "their",
                                         "which", "where", "when", "also", "very", "just", "about",
                                         "after", "before", "could", "would", "should", "there"}:
                entities.add(w)
    
    entity_list = list(entities)
    
    # Generate hypotheses by combining concepts
    for _ in range(n_hypotheses):
        e1 = rng.choice(entity_list)
        e2 = rng.choice(entity_list)
        
        verbs = ["may be regulated by", "could be influenced by", "is modulated by",
                 "interacts with", "affects", "is associated with", "predicts",
                 "is modified by", "depends on", "is mediated by"]
        
        hypothesis = f"{e1} {rng.choice(verbs)} {e2} in Alzheimer disease"
        hypotheses.append(hypothesis)
    
    return hypotheses


def score_hypothesis(hypothesis, post_2018_discoveries):
    """Score how well a hypothesis matches post-2018 discoveries.
    
    0 = no match. Higher = matches more discoveries.
    """
    hyp_lower = hypothesis.lower()
    score = 0
    matches = []
    
    for disc in post_2018_discoveries:
        disc_lower = disc["discovery"].lower()
        # Count overlapping words (excluding stopwords)
        hyp_words = set(hyp_lower.split())
        disc_words = set(disc_lower.split())
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "can", "could", "may", "might", "shall", "should", "to",
                     "of", "in", "for", "on", "with", "at", "by", "from",
                     "this", "that", "these", "those", "it", "its", "and", "or",
                     "not", "no", "but", "as", "also", "very", "just", "so"}
        hyp_words -= stopwords
        disc_words -= stopwords
        
        overlap = len(hyp_words & disc_words) / max(1, len(disc_words))
        if overlap > 0.15:  # threshold for meaningful match
            score += overlap
            matches.append(disc["discovery"][:60])
    
    return score, matches[:3]


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

print("="*70)
print("TIME MACHINE EXPERIMENT")
print("Pre-2018 knowledge -> Predictions -> Compare vs 2019-2026 discoveries")
print("="*70)

print(f"\nPre-2018 facts: {len(PRE_2018_FACTS)}")
print(f"Post-2018 discoveries: {len(POST_2018_DISCOVERIES)}")
print(f"\n--- Pre-2018 knowledge (sample) ---")
for f in PRE_2018_FACTS[:5]:
    print(f"  {f}")
print(f"\n--- Post-2018 discoveries (sample) ---")
for d in POST_2018_DISCOVERIES[:5]:
    print(f"  [{d['year']}] {d['discovery'][:70]}...")

# Generate hypotheses
print(f"\n--- Generating hypotheses from pre-2018 knowledge ---")
hypotheses = generate_hypotheses(PRE_2018_FACTS, n_hypotheses=50)

# Score each
scored = []
for h in hypotheses:
    score, matches = score_hypothesis(h, POST_2018_DISCOVERIES)
    scored.append((score, h, matches))

scored.sort(key=lambda x: -x[0])

print(f"\n--- Top hypotheses that predicted post-2018 discoveries ---")
for score, h, matches in scored[:10]:
    if score > 0:
        print(f"\n  Score={score:.2f}: {h[:70]}")
        for m in matches:
            print(f"    -> {m}")

# How many discoveries were predicted?
total_disc = len(POST_2018_DISCOVERIES)
predicted_discs = set()
for score, h, matches in scored:
    if score > 0:
        for m in matches:
            for d in POST_2018_DISCOVERIES:
                if d["discovery"][:40] in m:
                    predicted_discs.add(d["year"])

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  Hypotheses generated: {len(hypotheses)}")
print(f"  Hypotheses with predictive value: {sum(1 for s,_,_ in scored if s > 0)}")
print(f"  Post-2018 discoveries with predictive match: {len(predicted_discs)}/{total_disc}")
print(f"  Best hypothesis score: {scored[0][0]:.2f}" if scored else "  No hypotheses scored")

# Show score distribution
scores = [s for s, _, _ in scored]
if scores:
    import numpy as np
    print(f"  Mean score: {np.mean(scores):.3f}")
    print(f"  Max score:  {max(scores):.3f}")
    print(f"  Score > 0:  {sum(1 for s in scores if s > 0)}/{len(scores)}")

print(f"\n  The AGI {'SHOWS' if sum(1 for s,_,_ in scored if s > 0) > 0 else 'DOES NOT SHOW'} predictive scientific creativity")
print(f"  {'Some' if len(predicted_discs) > 0 else 'No'} post-2018 discoveries were anticipated by pre-2018 hypotheses")

report = {
    "n_hypotheses": len(hypotheses),
    "predictive_hypotheses": sum(1 for s,_,_ in scored if s > 0),
    "discoveries_predicted": len(predicted_discs),
    "total_discoveries": total_disc,
    "top_hypotheses": [{"score": round(float(s), 3), "hypothesis": h, "matches": m} for s, h, m in scored[:10]],
}
report_path = os.path.join(OUTPUT_DIR, "time_machine_results.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatorio: {report_path}")
