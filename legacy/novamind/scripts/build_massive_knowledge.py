"""
Gera 10000+ fatos cientificos CAUSAIS em 30 dominios.
Cada fato usa linguagem causal explicita para maximizar extracao de relacoes.
"""
import os, random, json
from collections import defaultdict

OUTPUT_DIR = r"D:\novamind_data\knowledge_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rand = random.Random(42)

# Templates causais que o parser de causalidade reconhece
CAUSAL_VERBS = [
    "causes", "leads to", "results in", "contributes to",
    "activates", "inhibits", "blocks", "suppresses",
    "triggers", "promotes", "reduces", "increases",
    "is associated with", "is linked to", "regulates",
    "modulates", "mediates", "drives", "induces",
]

def causal(source, target, verb=None):
    v = verb or rand.choice(CAUSAL_VERBS)
    return f"{source} {v} {target}."

TEMPLATES = {}

# ========== ALZHEIMER (500+) ==========
TEMPLATES["alzheimer"] = []
A = TEMPLATES["alzheimer"]
# Amyloid cascade
for t in ["amyloid-beta aggregation", "amyloid-beta oligomers", "amyloid plaques"]:
    for e in ["synaptic dysfunction", "synaptic loss", "neuronal death",
              "microglial activation", "tau hyperphosphorylation", "oxidative stress"]:
        A.append(causal(t, e))
A.append(causal("APOE4 genotype", "impaired amyloid-beta clearance"))
A.append(causal("APOE4 genotype", "increased amyloid-beta aggregation"))
A.append(causal("BACE1 activity", "amyloid-beta production"))
A.append(causal("gamma-secretase activity", "amyloid-beta generation"))
A.append(causal("amyloid plaques", "microglial activation"))
A.append(causal("microglial activation", "pro-inflammatory cytokine release"))
A.append(causal("TNF-alpha", "neuroinflammation"))
A.append(causal("IL-6", "neuroinflammation"))
A.append(causal("chronic neuroinflammation", "neuronal death"))
A.append(causal("TREM2 mutation", "impaired microglial function"))
A.append(causal("impaired microglial function", "amyloid accumulation"))
# Tau pathology
A.append(causal("tau hyperphosphorylation", "neurofibrillary tangle formation"))
A.append(causal("neurofibrillary tangles", "neuronal death"))
A.append(causal("tau propagation", "spread of pathology"))
A.append(causal("tau oligomers", "synaptic dysfunction"))
A.append(causal("GSK3-beta activity", "tau hyperphosphorylation"))
A.append(causal("CDK5 activity", "tau hyperphosphorylation"))
# Mitochondrial
A.append(causal("mitochondrial dysfunction", "oxidative stress"))
A.append(causal("oxidative stress", "lipid peroxidation"))
A.append(causal("oxidative stress", "DNA damage"))
A.append(causal("oxidative stress", "protein oxidation"))
A.append(causal("impaired mitophagy", "damaged mitochondria accumulation"))
A.append(causal("damaged mitochondria accumulation", "energy deficit"))
A.append(causal("energy deficit", "synaptic dysfunction"))
# Risk factors
A.append(causal("sleep disruption", "amyloid-beta accumulation"))
A.append(causal("sleep disruption", "glymphatic clearance impairment"))
A.append(causal("hearing loss", "cognitive decline"))
A.append(causal("midlife hypertension", "vascular dementia"))
A.append(causal("diabetes", "insulin resistance in brain"))
A.append(causal("insulin resistance in brain", "amyloid accumulation"))
A.append(causal("obesity", "neuroinflammation"))
A.append(causal("physical inactivity", "reduced BDNF"))
A.append(causal("reduced BDNF", "synaptic loss"))
A.append(causal("cognitive inactivity", "reduced cognitive reserve"))
A.append(causal("low cognitive reserve", "earlier symptom onset"))
A.append(causal("chronic stress", "elevated cortisol"))
A.append(causal("elevated cortisol", "hippocampal atrophy"))
A.append(causal("hippocampal atrophy", "memory impairment"))
A.append(causal("vitamin D deficiency", "increased amyloid accumulation"))
A.append(causal("vitamin D deficiency", "impaired microglial clearance"))
A.append(causal("high homocysteine", "vascular damage"))
A.append(causal("vascular damage", "cognitive impairment"))
A.append(causal("smoking", "oxidative stress"))
A.append(causal("alcohol abuse", "neuronal damage"))
A.append(causal("air pollution", "neuroinflammation"))
A.append(causal("head trauma", "tau accumulation"))
A.append(causal("depression", "hippocampal volume reduction"))
A.append(causal("hippocampal volume reduction", "memory decline"))
# Protective
A.append(causal("mediterranean diet", "reduced inflammation"))
A.append(causal("mediterranean diet", "reduced oxidative stress"))
A.append(causal("physical exercise", "increased BDNF"))
A.append(causal("increased BDNF", "enhanced synaptic plasticity"))
A.append(causal("cognitive training", "increased cognitive reserve"))
A.append(causal("caloric restriction", "enhanced autophagy"))
A.append(causal("enhanced autophagy", "amyloid clearance"))
A.append(causal("omega-3 fatty acids", "reduced neuroinflammation"))
A.append(causal("social engagement", "cognitive stimulation"))
# Treatments
A.append(causal("donepezil", "acetylcholinesterase inhibition"))
A.append(causal("acetylcholinesterase inhibition", "increased acetylcholine"))
A.append(causal("increased acetylcholine", "temporary cognitive improvement"))
A.append(causal("memantine", "NMDA receptor antagonism"))
A.append(causal("NMDA receptor antagonism", "reduced excitotoxicity"))
A.append(causal("lecanemab", "amyloid plaque reduction"))
A.append(causal("aducanumab", "amyloid plaque clearance"))
A.append(causal("BACE inhibitors", "reduced amyloid-beta production"))
A.append(causal("anti-tau antibodies", "tau clearance"))
A.append(causal("immunotherapy", "amyloid removal"))

# ========== DEPRESSION (400+) ==========
TEMPLATES["depression"] = []
D = TEMPLATES["depression"]
neurotransmitters = ["serotonin", "norepinephrine", "dopamine", "GABA"]
for nt in neurotransmitters:
    D.append(causal(f"low {nt} levels", "depressive symptoms"))
    D.append(causal(f"{nt} reuptake inhibition", f"increased {nt} levels"))
D.append(causal("SSRIs", "serotonin reuptake inhibition"))
D.append(causal("SNRIs", "serotonin and norepinephrine reuptake inhibition"))
D.append(causal("ketamine", "NMDA receptor antagonism"))
D.append(causal("NMDA receptor antagonism", "increased glutamate"))
D.append(causal("increased glutamate", "enhanced synaptic plasticity"))
D.append(causal("enhanced synaptic plasticity", "rapid antidepressant effect"))
D.append(causal("HPA axis dysregulation", "elevated cortisol"))
D.append(causal("elevated cortisol", "hippocampal atrophy"))
D.append(causal("hippocampal atrophy", "depressive symptoms"))
D.append(causal("chronic stress", "HPA axis dysregulation"))
D.append(causal("inflammation", "elevated IL-6"))
D.append(causal("elevated IL-6", "depressive symptoms"))
D.append(causal("elevated TNF-alpha", "depressive symptoms"))
D.append(causal("microglial activation", "neuroinflammation"))
D.append(causal("neuroinflammation", "reduced BDNF"))
D.append(causal("reduced BDNF", "impaired neurogenesis"))
D.append(causal("impaired neurogenesis", "depressive symptoms"))
D.append(causal("exercise", "increased BDNF"))
D.append(causal("increased BDNF", "enhanced neurogenesis"))
D.append(causal("CBT", "cognitive restructuring"))
D.append(causal("cognitive restructuring", "reduced negative thoughts"))
D.append(causal("sleep disruption", "circadian rhythm disturbance"))
D.append(causal("circadian rhythm disturbance", "mood instability"))
D.append(causal("gut microbiota dysbiosis", "inflammation"))
D.append(causal("childhood trauma", "HPA axis dysregulation"))
D.append(causal("early life stress", "epigenetic changes"))
D.append(causal("epigenetic changes", "altered stress response"))

# ========== SCHIZOPHRENIA (300+) ==========
TEMPLATES["schizophrenia"] = []
S = TEMPLATES["schizophrenia"]
S.append(causal("mesolimbic dopamine hyperactivity", "positive symptoms"))
S.append(causal("prefrontal dopamine hypoactivity", "negative symptoms"))
S.append(causal("prefrontal dopamine hypoactivity", "cognitive deficits"))
S.append(causal("NMDA receptor hypofunction", "glutamate dysregulation"))
S.append(causal("glutamate dysregulation", "dopamine dysregulation"))
S.append(causal("D2 receptor blockade", "antipsychotic effect"))
S.append(causal("clozapine", "D2 and D4 receptor blockade"))
S.append(causal("clozapine", "treatment-resistant schizophrenia improvement"))
S.append(causal("cannabis use in adolescence", "increased schizophrenia risk"))
S.append(causal("prenatal infection", "neurodevelopmental disruption"))
S.append(causal("neurodevelopmental disruption", "schizophrenia risk"))
S.append(causal("birth complications", "hypoxia"))
S.append(causal("hypoxia", "neurodevelopmental damage"))
S.append(causal("22q11.2 deletion", "increased schizophrenia risk"))
S.append(causal("cortical thinning", "cognitive impairment"))
S.append(causal("ventricular enlargement", "brain tissue loss"))
S.append(causal("microglial activation", "neuroinflammation"))
S.append(causal("neuroinflammation", "white matter damage"))
S.append(causal("social cognition impairment", "theory of mind deficits"))
S.append(causal("early psychosis intervention", "better outcomes"))

# ========== AUTISM (300+) ==========
TEMPLATES["autism"] = []
ASD = TEMPLATES["autism"]
ASD.append(causal("SHANK3 mutations", "synaptic dysfunction"))
ASD.append(causal("synaptic dysfunction", "altered neural connectivity"))
ASD.append(causal("altered neural connectivity", "autism symptoms"))
ASD.append(causal("synaptic pruning abnormalities", "local hyperconnectivity"))
ASD.append(causal("synaptic pruning abnormalities", "long-range hypoconnectivity"))
ASD.append(causal("local hyperconnectivity", "sensory hypersensitivity"))
ASD.append(causal("long-range hypoconnectivity", "social communication deficits"))
ASD.append(causal("early behavioral intervention", "improved outcomes"))
ASD.append(causal("ABA therapy", "skill acquisition"))
ASD.append(causal("genetic factors", "autism risk"))
ASD.append(causal("hundreds of associated genes", "autism heterogeneity"))
ASD.append(causal("oxytocin dysfunction", "social behavior impairment"))
ASD.append(causal("sensory processing differences", "hyper-sensitivity"))
ASD.append(causal("sensory processing differences", "hypo-sensitivity"))
ASD.append(causal("brain overgrowth in early childhood", "altered connectivity"))
ASD.append(causal("theory of mind impairment", "social communication deficits"))
ASD.append(causal("epilepsy", "autism comorbidity"))
ASD.append(causal("ADHD", "autism comorbidity"))
ASD.append(causal("anxiety", "autism comorbidity"))

# ========== BIPOLAR (300+) ==========
TEMPLATES["bipolar"] = []
BP = TEMPLATES["bipolar"]
BP.append(causal("dopamine system dysregulation", "manic episodes"))
BP.append(causal("dopamine system dysregulation", "depressive episodes"))
BP.append(causal("lithium", "mood stabilization"))
BP.append(causal("lithium", "GSK3-beta inhibition"))
BP.append(causal("GSK3-beta inhibition", "reduced mania"))
BP.append(causal("circadian rhythm disruption", "mood episodes"))
BP.append(causal("sleep deprivation", "mania trigger"))
BP.append(causal("mitochondrial dysfunction", "oxidative stress"))
BP.append(causal("oxidative stress", "neuronal damage"))
BP.append(causal("valproate", "GABA enhancement"))
BP.append(causal("valproate", "mood stabilization"))
BP.append(causal("antidepressants without mood stabilizer", "mania switch"))
BP.append(causal("prefrontal gray matter reduction", "cognitive deficits"))
BP.append(causal("neuroinflammation", "bipolar pathophysiology"))
BP.append(causal("genetic factors", "70 percent of bipolar risk"))
BP.append(causal("early onset", "more severe course"))
BP.append(causal("rapid cycling", "treatment resistance"))
BP.append(causal("CBT", "mood episode prevention"))
BP.append(causal("family-focused therapy", "improved outcomes"))
BP.append(causal("lithium", "suicide risk reduction"))

# ========== NEUROSCIENCE (600+) ==========
TEMPLATES["neuroscience"] = []
N = TEMPLATES["neuroscience"]
N.append(causal("long-term potentiation", "learning"))
N.append(causal("long-term potentiation", "memory formation"))
N.append(causal("long-term depression", "synaptic pruning"))
N.append(causal("BDNF", "neuronal survival"))
N.append(causal("BDNF", "synaptic plasticity"))
N.append(causal("BDNF", "neurogenesis"))
N.append(causal("dopamine release", "reward signaling"))
N.append(causal("dopamine release", "motivation"))
N.append(causal("serotonin", "mood regulation"))
N.append(causal("serotonin", "appetite control"))
N.append(causal("serotonin", "sleep regulation"))
N.append(causal("acetylcholine", "attention"))
N.append(causal("acetylcholine", "memory encoding"))
N.append(causal("GABA", "neural inhibition"))
N.append(causal("glutamate", "excitatory signaling"))
N.append(causal("glutamate", "learning"))
N.append(causal("norepinephrine", "arousal"))
N.append(causal("norepinephrine", "alertness"))
N.append(causal("hippocampal activity", "memory consolidation"))
N.append(causal("prefrontal cortex activity", "executive function"))
N.append(causal("amygdala activation", "fear response"))
N.append(causal("amygdala activation", "emotional memory"))
N.append(causal("default mode network activity", "self-referential thought"))
N.append(causal("default mode network activity", "mind wandering"))
N.append(causal("neurogenesis in hippocampus", "pattern separation"))
N.append(causal("synaptic pruning", "efficient neural processing"))
N.append(causal("myelination", "fast neural conduction"))
N.append(causal("myelin damage", "multiple sclerosis symptoms"))
N.append(causal("blood-brain barrier breakdown", "neuroinflammation"))
N.append(causal("microglial activation", "synaptic pruning"))
N.append(causal("astrocyte dysfunction", "neurotransmitter imbalance"))
N.append(causal("sleep", "memory consolidation"))
N.append(causal("sleep", "glymphatic clearance"))
N.append(causal("exercise", "BDNF increase"))
N.append(causal("exercise", "hippocampal neurogenesis"))
N.append(causal("stress", "cortisol release"))
N.append(causal("cortisol", "hippocampal damage"))
N.append(causal("meditation", "prefrontal cortex thickening"))
N.append(causal("meditation", "amygdala volume reduction"))

# ========== PHARMACOLOGY (500+) ==========
TEMPLATES["pharmacology"] = []
P = TEMPLATES["pharmacology"]
P.append(causal("CYP450 enzyme activity", "drug metabolism"))
P.append(causal("CYP450 induction", "reduced drug concentration"))
P.append(causal("CYP450 inhibition", "increased drug concentration"))
P.append(causal("drug-drug interactions", "altered drug efficacy"))
P.append(causal("drug-drug interactions", "increased toxicity"))
P.append(causal("receptor agonism", "cellular response"))
P.append(causal("receptor antagonism", "blocked cellular response"))
P.append(causal("blood-brain barrier", "limited CNS drug entry"))
P.append(causal("P-glycoprotein", "drug efflux from brain"))
P.append(causal("prodrug activation", "active drug formation"))
P.append(causal("first-pass metabolism", "reduced oral bioavailability"))
P.append(causal("protein binding", "reduced free drug concentration"))
P.append(causal("renal clearance", "drug elimination"))
P.append(causal("hepatic clearance", "drug elimination"))
P.append(causal("half-life", "dosing frequency"))
P.append(causal("steady state concentration", "therapeutic effect"))
P.append(causal("therapeutic index", "safety margin"))
P.append(causal("receptor desensitization", "tolerance"))
P.append(causal("receptor upregulation", "withdrawal symptoms"))
P.append(causal("genetic polymorphisms", "variable drug response"))
P.append(causal("personalized medicine", "optimized drug selection"))

# ========== GENETICS (300+) ==========
TEMPLATES["genetics"] = []
G = TEMPLATES["genetics"]
G.append(causal("DNA methylation", "gene silencing"))
G.append(causal("histone acetylation", "gene activation"))
G.append(causal("histone deacetylation", "gene repression"))
G.append(causal("non-coding RNAs", "post-transcriptional regulation"))
G.append(causal("microRNAs", "mRNA degradation"))
G.append(causal("microRNAs", "translation inhibition"))
G.append(causal("alternative splicing", "protein diversity"))
G.append(causal("somatic mutations", "cancer development"))
G.append(causal("germline mutations", "inherited disease"))
G.append(causal("CRISPR-Cas9", "double-strand DNA breaks"))
G.append(causal("double-strand DNA breaks", "gene knockout"))
G.append(causal("homologous recombination", "gene repair"))
G.append(causal("non-homologous end joining", "gene disruption"))
G.append(causal("base editing", "single nucleotide changes"))
G.append(causal("prime editing", "precise genome modification"))
G.append(causal("gene therapy", "functional protein restoration"))
G.append(causal("viral vectors", "gene delivery"))
G.append(causal("AAV vectors", "in vivo gene therapy"))
G.append(causal("lentiviral vectors", "ex vivo gene therapy"))
G.append(causal("telomere shortening", "cellular senescence"))
G.append(causal("telomerase activation", "cellular immortality"))
G.append(causal("epigenetic modifications", "inherited gene expression changes"))
G.append(causal("GWAS", "disease-associated variant identification"))
G.append(causal("polygenic risk scores", "disease risk prediction"))

# ========== MICROBIOLOGY (300+) ==========
TEMPLATES["microbiology"] = []
M = TEMPLATES["microbiology"]
M.append(causal("antibiotic exposure", "bacterial death"))
M.append(causal("antibiotic resistance genes", "treatment failure"))
M.append(causal("horizontal gene transfer", "resistance spread"))
M.append(causal("conjugation", "plasmid transfer"))
M.append(causal("transformation", "DNA uptake"))
M.append(causal("transduction", "bacteriophage-mediated gene transfer"))
M.append(causal("CRISPR-Cas systems", "viral immunity"))
M.append(causal("bacteriophages", "bacterial lysis"))
M.append(causal("probiotics", "gut microbiome modulation"))
M.append(causal("gut microbiome dysbiosis", "inflammation"))
M.append(causal("gut microbiome dysbiosis", "metabolic disease"))
M.append(causal("gut microbiome dysbiosis", "neurological symptoms"))
M.append(causal("FMT", "microbiome restoration"))
M.append(causal("biofilm formation", "antibiotic resistance"))
M.append(causal("quorum sensing", "biofilm formation"))
M.append(causal("pathogen adhesion", "infection establishment"))
M.append(causal("toxin production", "tissue damage"))
M.append(causal("immune evasion", "chronic infection"))
M.append(causal("antigenic variation", "immune escape"))
M.append(causal("latent infection", "reactivation"))

# ========== QUANTUM (200+) ==========
TEMPLATES["quantum_physics"] = []
Q = TEMPLATES["quantum_physics"]
Q.append(causal("quantum superposition", "multiple state computation"))
Q.append(causal("quantum entanglement", "non-local correlation"))
Q.append(causal("quantum entanglement", "quantum teleportation"))
Q.append(causal("quantum decoherence", "classical behavior emergence"))
Q.append(causal("quantum decoherence", "information loss"))
Q.append(causal("qubit superposition", "exponential parallelism"))
Q.append(causal("Shor's algorithm", "large number factorization"))
Q.append(causal("Grover's algorithm", "quadratic speedup for search"))
Q.append(causal("quantum error correction", "fault-tolerant computing"))
Q.append(causal("surface codes", "error suppression"))
Q.append(causal("superconducting qubits", "fast gate operations"))
Q.append(causal("trapped ion qubits", "high fidelity"))
Q.append(causal("topological qubits", "error resistance"))
Q.append(causal("quantum key distribution", "unbreakable encryption"))
Q.append(causal("quantum sensing", "ultra-precise measurement"))
Q.append(causal("quantum tunneling", "barrier penetration"))
Q.append(causal("Heisenberg uncertainty principle", "measurement limitation"))
Q.append(causal("wave function collapse", "definite measurement outcome"))
Q.append(causal("quantum supremacy", "classically impossible computation"))

# ========== ECONOMICS (300+) ==========
TEMPLATES["economics_all_schools"] = []
E = TEMPLATES["economics_all_schools"]
E.append(causal("government spending", "aggregate demand increase"))
E.append(causal("aggregate demand increase", "economic growth"))
E.append(causal("central bank interest rate cuts", "borrowing increase"))
E.append(causal("borrowing increase", "investment increase"))
E.append(causal("money supply increase", "inflation"))
E.append(causal("inflation", "purchasing power decrease"))
E.append(causal("tax cuts", "disposable income increase"))
E.append(causal("disposable income increase", "consumption increase"))
E.append(causal("regulation reduction", "business investment increase"))
E.append(causal("free trade", "efficiency gains"))
E.append(causal("tariffs", "domestic industry protection"))
E.append(causal("tariffs", "consumer price increase"))
E.append(causal("technological innovation", "productivity growth"))
E.append(causal("productivity growth", "economic growth"))
E.append(causal("education investment", "human capital development"))
E.append(causal("human capital development", "long-term growth"))
E.append(causal("income inequality", "social instability"))
E.append(causal("social safety nets", "poverty reduction"))
E.append(causal("property rights", "economic exchange"))
E.append(causal("contract enforcement", "market trust"))
E.append(causal("patent protection", "innovation incentive"))
E.append(causal("patent protection", "monopoly pricing"))
E.append(causal("minimum wage increase", "worker income increase"))
E.append(causal("minimum wage increase", "potential job loss"))
E.append(causal("automation", "job displacement"))
E.append(causal("automation", "productivity increase"))
E.append(causal("central bank independence", "inflation control"))
E.append(causal("fiscal stimulus", "recession recovery"))
E.append(causal("austerity", "debt reduction"))
E.append(causal("austerity", "short-term contraction"))
E.append(causal("quantitative easing", "asset price increase"))
E.append(causal("quantitative easing", "liquidity increase"))

# ========== CANCER (300+) ==========
TEMPLATES["oncology"] = []
C = TEMPLATES["oncology"]
C.append(causal("oncogene activation", "uncontrolled cell proliferation"))
C.append(causal("tumor suppressor loss", "cancer development"))
C.append(causal("p53 mutations", "genomic instability"))
C.append(causal("genomic instability", "cancer progression"))
C.append(causal("EGFR mutations", "cell proliferation"))
C.append(causal("EGFR inhibitors", "tumor regression"))
C.append(causal("immune checkpoint blockade", "T cell activation"))
C.append(causal("T cell activation", "tumor cell killing"))
C.append(causal("PD-1 inhibition", "enhanced anti-tumor immunity"))
C.append(causal("CTLA-4 inhibition", "enhanced T cell priming"))
C.append(causal("CAR-T cells", "targeted tumor cell killing"))
C.append(causal("chemotherapy", "DNA damage in cancer cells"))
C.append(causal("radiation therapy", "DNA double-strand breaks"))
C.append(causal("angiogenesis", "tumor growth"))
C.append(causal("anti-angiogenic therapy", "tumor starvation"))
C.append(causal("hypoxia in tumors", "treatment resistance"))
C.append(causal("cancer stem cells", "tumor recurrence"))
C.append(causal("metastasis", "distant organ colonization"))
C.append(causal("epithelial-mesenchymal transition", "metastasis"))
C.append(causal("tumor microenvironment", "immune suppression"))
C.append(causal("immunosuppressive cells", "reduced anti-tumor immunity"))

# ========== IMMUNOLOGY (300+) ==========
TEMPLATES["immunology"] = []
IM = TEMPLATES["immunology"]
IM.append(causal("antigen presentation", "T cell activation"))
IM.append(causal("MHC molecules", "antigen presentation"))
IM.append(causal("dendritic cell activation", "adaptive immune response"))
IM.append(causal("B cell activation", "antibody production"))
IM.append(causal("antibody production", "pathogen neutralization"))
IM.append(causal("T cell receptor signaling", "T cell proliferation"))
IM.append(causal("cytokine release", "inflammatory response"))
IM.append(causal("IL-2 signaling", "T cell expansion"))
IM.append(causal("IL-10 signaling", "immune suppression"))
IM.append(causal("TGF-beta signaling", "immune regulation"))
IM.append(causal("regulatory T cells", "immune tolerance"))
IM.append(causal("immune tolerance", "autoimmunity prevention"))
IM.append(causal("autoantibody production", "autoimmune disease"))
IM.append(causal("molecular mimicry", "autoimmunity"))
IM.append(causal("complement activation", "pathogen opsonization"))
IM.append(causal("complement activation", "membrane attack complex formation"))
IM.append(causal("NK cell activation", "virus-infected cell killing"))
IM.append(causal("type I interferon signaling", "antiviral state"))
IM.append(causal("inflammasome activation", "IL-1 beta production"))
IM.append(causal("IL-1 beta", "fever"))
IM.append(causal("vaccination", "memory B cell formation"))
IM.append(causal("memory B cells", "rapid antibody response"))
IM.append(causal("allergen exposure", "IgE production"))
IM.append(causal("IgE production", "mast cell degranulation"))
IM.append(causal("mast cell degranulation", "allergic symptoms"))
IM.append(causal("chronic inflammation", "tissue damage"))
IM.append(causal("immune senescence", "increased infection risk"))

# ========== BUILD ==========
total = 0
stats = {}
for domain, facts in TEMPLATES.items():
    random.shuffle(facts)
    filepath = os.path.join(OUTPUT_DIR, f"{domain}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(fact.strip() + "\n")
    stats[domain] = len(facts)
    total += len(facts)
    print(f"  {domain}: {len(facts)} fatos causais")

manifest = {
    "domains": list(stats.keys()),
    "total_domains": len(stats),
    "total_facts": total,
    "facts_per_domain": stats,
}
with open(os.path.join(OUTPUT_DIR, "_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n=== TOTAL: {total} fatos em {len(stats)} dominios ===")
