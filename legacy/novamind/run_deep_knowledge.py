"""
Injecao de conhecimento profundo: ciencias, matematica, medicina, humanidades.

Alimenta o CGE com conhecimento estruturado de alto nivel para que
o sistema possa responder questoes como "como humanos pensam?"
usando seu proprio modelo de metacognicao como analogia.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.multilingual_encoder import MultilingualEncoder

print("=" * 70)
print("INJECAO DE CONHECIMENTO PROFUNDO")
print("=" * 70)

cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
enc = MultilingualEncoder()

# ─── 1. CONHECIMENTO CIENTIFICO AVANCADO ─────────────────

KNOWLEDGE = {
    "calculo": """
Calculus studies continuous change. Derivatives measure instantaneous rate of change.
The derivative of position is velocity. The derivative of velocity is acceleration.
Integration computes area under curves. The fundamental theorem links derivatives and integrals.
Limits describe behavior approaching a point. Continuity means no jumps or gaps.
d/dx of x^n = n*x^(n-1). Integral of x^n dx = x^(n+1)/(n+1) + C.
Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x). Product rule: d/dx f*g = f'g + fg'.
Partial derivatives extend calculus to multiple variables. Gradient points steepest ascent.
Taylor series approximate functions as polynomials: f(x) = sum f^n(a)(x-a)^n/n!.
""",
    "algebra_linear": """
Linear algebra studies vector spaces and linear transformations.
Vectors have magnitude and direction. Matrices represent linear transformations.
Matrix multiplication composes transformations: (AB)x = A(Bx).
Determinants measure volume scaling. Non-zero determinant means invertible.
Eigenvalues and eigenvectors satisfy Av = λv. PCA finds principal eigenvectors.
Vector spaces have basis vectors. Any vector is a linear combination of basis vectors.
Dot product measures alignment. Cross product produces perpendicular vector.
Linear systems Ax = b solved by x = A^(-1)b when A is invertible.
Singular Value Decomposition: A = UΣV^T. Used for compression and recommendation.
""",
    "fisica_classica": """
Classical physics describes macroscopic motion. Newton's laws: F = ma.
Gravitational force: F = G*m1*m2/r^2. Objects fall at same acceleration g = 9.8m/s^2.
Energy is conserved. Kinetic energy = mv^2/2. Potential energy = mgh.
Work = force * distance. Power = work / time.
Momentum = mass * velocity. Conservation of momentum in collisions.
Thermodynamics: heat flows from hot to cold. Entropy increases. Energy conserved.
Electromagnetism: Maxwell's equations describe light as electromagnetic waves.
Optics: light reflects and refracts. Snell's law: n1*sin(θ1) = n2*sin(θ2).
""",
    "engenharia": """
Engineering applies science to design solutions. Stress = force/area. Strain = deformation/original.
Young's modulus E = stress/strain. Materials deform elastically then plastically.
Beams bend under load. Moment of inertia resists bending. Euler-Bernoulli beam theory.
Circuits: Ohm's law V = IR. Kirchhoff's laws for current and voltage.
Resistors in series: R_total = R1 + R2. In parallel: 1/R = 1/R1 + 1/R2.
Control theory uses feedback to maintain desired output. PID controllers.
Signal processing: Fourier transform decomposes signals into frequencies.
Thermodynamic cycles: Carnot efficiency = 1 - T_cold/T_hot.
""",
    "neurociencia": """
Neuroscience studies the nervous system. Neurons communicate via electrical and chemical signals.
Action potentials travel down axons. Synapses release neurotransmitters.
The brain has ~86 billion neurons. Each neuron connects to ~7000 others.
The cortex handles perception, language, memory, and decision making.
The hippocampus consolidates memories. The amygdala processes emotions.
Neuroplasticity: the brain rewires itself based on experience.
EEG measures electrical activity. fMRI measures blood flow (BOLD signal).
Consciousness may emerge from integrated information (IIT theory).
The default mode network activates during introspection and mind-wandering.
""",
    "quimica": """
Chemistry studies matter and its transformations. Atoms have protons, neutrons, electrons.
The periodic table organizes elements by atomic number. Groups share chemical properties.
Chemical bonds: ionic (electron transfer), covalent (electron sharing), metallic.
Chemical reactions: reactants -> products. Rate depends on temperature and concentration.
Catalysts speed reactions without being consumed. Enzymes are biological catalysts.
pH measures acidity: pH = -log[H+]. Acids donate H+, bases accept H+.
Organic chemistry studies carbon compounds. Functional groups determine reactivity.
Thermochemistry: enthalpy, entropy, Gibbs free energy ΔG = ΔH - TΔS.
""",
    "anatomia": """
Anatomy studies body structure. The skeletal system has 206 bones.
The muscular system has ~650 muscles. Cardiac muscle pumps blood.
The heart has 4 chambers. Blood flows: vena cava -> right atrium -> right ventricle -> lungs -> left atrium -> left ventricle -> aorta.
The brain has cerebrum, cerebellum, brainstem. Cerebrum has 4 lobes: frontal, parietal, temporal, occipital.
The nervous system: CNS (brain+spine) and PNS (peripheral nerves).
The respiratory system: trachea -> bronchi -> bronchioles -> alveoli.
The digestive system: mouth -> esophagus -> stomach -> small intestine -> large intestine.
The immune system: white blood cells, antibodies, lymph nodes.
""",
    "fibonacci": """
Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34... Each number is sum of previous two.
F(n) = F(n-1) + F(n-2). The ratio of consecutive terms approaches the golden ratio φ = 1.618...
Fibonacci appears in nature: sunflower spirals, pinecones, nautilus shells, leaf arrangement.
The golden ratio φ = (1 + sqrt(5))/2. φ^n = F(n)*φ + F(n-1).
Binet's formula: F(n) = (φ^n - ψ^n)/sqrt(5) where ψ = (1 - sqrt(5))/2.
Used in algorithms: Fibonacci heap, Fibonacci search, Fibonacci coding.
""",
    "arte_cultura": """
Art expresses human experience across cultures. Renaissance: perspective, humanism, da Vinci, Michelangelo.
Baroque: drama, movement, Caravaggio, Bernini. Impressionism: light, color, Monet, Renoir.
Modernism: breaking conventions, Picasso, Kandinsky, Duchamp.
Music: melody (horizontal), harmony (vertical), rhythm (time). Sonata form: exposition, development, recapitulation.
Architecture: form follows function (modernism). Gothic: pointed arches, flying buttresses.
Literature: narrative, character, theme, symbolism. Epic, tragedy, comedy, novel.
Culture includes language, beliefs, customs, art, and institutions shared by a group.
""",
    "doencas": """
Diseases disrupt normal body function. Infectious diseases: caused by pathogens (bacteria, viruses, fungi, parasites).
COVID-19: caused by SARS-CoV-2 virus. Spreads through respiratory droplets. 
Cancer: uncontrolled cell division. Caused by genetic mutations. Types: carcinoma, sarcoma, leukemia, lymphoma.
Cardiovascular disease: heart attacks, strokes. Risk factors: smoking, obesity, high blood pressure.
Diabetes: type 1 (autoimmune, no insulin), type 2 (insulin resistance). 
Neurodegenerative: Alzheimer's (memory loss, amyloid plaques), Parkinson's (movement disorder, dopamine loss).
Autoimmune: immune system attacks self. Examples: rheumatoid arthritis, lupus, MS.
Diagnosis: symptoms, physical exam, imaging (X-ray, CT, MRI), lab tests, biopsy.
""",
    "vacinas": """
Vaccines train the immune system to fight specific pathogens.
mRNA vaccines (Pfizer, Moderna) deliver genetic instructions for spike protein.
Viral vector vaccines (J&J, AstraZeneca) use harmless virus to deliver antigen.
Inactivated vaccines contain killed pathogen. Live attenuated use weakened pathogen.
Subunit vaccines use specific protein fragments. Toxoid vaccines target bacterial toxins.
Herd immunity occurs when enough population is vaccinated to protect vulnerable.
Adjuvants boost immune response. Boosters maintain immunity over time.
Vaccines eradicated smallpox. Polio nearly eradicated. Major reduction in measles, tetanus, diphtheria.
""",
    "drogas_farmacia": """
Pharmacology studies how drugs interact with the body. Pharmacokinetics: absorption, distribution, metabolism, excretion.
Agonists activate receptors. Antagonists block receptors. Dose-response curve shows effect vs concentration.
Therapeutic index = toxic dose / effective dose. Narrow TI requires careful monitoring.
Antibiotics: penicillin (cell wall), tetracycline (protein synthesis), fluoroquinolones (DNA replication).
Antivirals: acyclovir (herpes), oseltamivir (flu), remdesivir (COVID), antiretrovirals (HIV).
Analgesics: NSAIDs (ibuprofen, aspirin), opioids (morphine, fentanyl), acetaminophen.
Psychiatric drugs: SSRIs (depression), benzodiazepines (anxiety), antipsychotics (schizophrenia).
Chemotherapy: alkylating agents, antimetabolites, targeted therapy, immunotherapy.
Drug development: discovery -> preclinical -> Phase I (safety) -> Phase II (efficacy) -> Phase III (comparison) -> approval.
""",
    "psicologia_cognitiva": """
Cognitive psychology studies mental processes: perception, attention, memory, language, reasoning, problem-solving.
Working memory holds ~7 items (Miller's law). Declarative memory: facts and events. Procedural memory: skills.
Classical conditioning: Pavlov's dogs. Operant conditioning: rewards and punishments (Skinner).
Piaget's stages of cognitive development: sensorimotor, preoperational, concrete operational, formal operational.
Cognitive biases: confirmation bias (favor confirming evidence), availability heuristic (ease of recall).
Kahneman's System 1 (fast, intuitive) and System 2 (slow, analytical) thinking.
Theory of mind: understanding others have beliefs, desires, intentions different from one's own.
""",
}

# ─── 2. ALIMENTA CGE ─────────────────────────────────────

print(f"\nAlimentando CGE com {len(KNOWLEDGE)} modulos de conhecimento...\n")

for module_name, text in KNOWLEDGE.items():
    t0 = time.time()
    cge.write_text_concepts(text)
    elapsed = time.time() - t0
    chars = len(text)
    print(f"  [{module_name:20s}] {chars:4d} chars em {elapsed:.2f}s")

print(f"\nCGE apos injecao: {len(cge.nodes)} nodes, {len(cge.edges)} edges")

# ─── 3. PERGUNTA: COMO HUMANOS PENSAM? ─────────────────

print("\n" + "=" * 70)
print('PERGUNTA: "COMO HUMANOS PENSAM?"')
print("=" * 70)

# Faz a pergunta em portugues — LaBSE entende
question = "como humanos pensam"
query_hdv = enc.encode_bipolar(question)

results = cge.query_attention(query_hdv, top_k=15)
print(f"\nConceitos ativados por '{question}':")
for nid, ndata, sim in results[:15]:
    conf = float(ndata.get("confidence", 0))
    acc = float(ndata.get("access_count", 0))
    print(f"  {nid:40s} sim={sim:.4f} conf={conf:.2f} acc={acc:.0f}")

# ─── 4. CONSULTAS ADICIONAIS ──────────────────────────────

print("\n" + "=" * 70)
print("CONSULTAS MULTIDOMINIO")
print("=" * 70)

queries = [
    "neurotransmissores e sinapses",
    "vacinas mRNA funcionamento",
    "derivada integral calculo",
    "fibonacci natureza proporcao aurea",
    "alzheimer neurodegeneracao",
    "farmacologia antidepressivos",
    "engenharia materiais tensao",
]

for q in queries:
    hdv = enc.encode_bipolar(q)
    res = cge.query_attention(hdv, top_k=5)
    sims = ", ".join([f"{nid}({sim:.3f})" for nid, _, sim in res[:5]])
    print(f"\n  '{q}':\n    {sims}")

# ─── 5. SALVA ESTADO ─────────────────────────────────────

import torch
save_path = "D:/novamind_data/backup/cge_knowledge_state.pth"
torch.save({
    "nodes": cge.nodes,
    "edges": cge.edges,
    "n_nodes": len(cge.nodes),
    "n_edges": len(cge.edges),
}, save_path)
print(f"\n[SAVE] Estado salvo em: {save_path}")
print(f"[DONE] CGE agora tem {len(cge.nodes)} nodes de conhecimento multidisciplinar")
