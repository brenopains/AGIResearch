# Dados de conhecimento para injecao no CGE
# Separado do script de injecao para importacao segura

KNOWLEDGE = {
    "calculo": """Calculus studies continuous change. Derivatives measure instantaneous rate of change.
The derivative of position is velocity. The derivative of velocity is acceleration.
Integration computes area under curves. The fundamental theorem links derivatives and integrals.
Limits describe behavior approaching a point. Continuity means no jumps or gaps.
d/dx of x^n = n*x^(n-1). Integral of x^n dx = x^(n+1)/(n+1) + C.
Chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x). Product rule: d/dx f*g = f'g + fg'.
Partial derivatives extend calculus to multiple variables. Gradient points steepest ascent.
Taylor series approximate functions as polynomials.""",
    "algebra_linear": """Linear algebra studies vector spaces and linear transformations.
Vectors have magnitude and direction. Matrices represent linear transformations.
Matrix multiplication composes transformations: (AB)x = A(Bx).
Determinants measure volume scaling. Non-zero determinant means invertible.
Eigenvalues and eigenvectors satisfy Av = lv. PCA finds principal eigenvectors.
Vector spaces have basis vectors. Any vector is a linear combination of basis vectors.
Dot product measures alignment. Cross product produces perpendicular vector.
Singular Value Decomposition: A = USV^T. Used for compression and recommendation.""",
    "fisica_classica": """Classical physics describes macroscopic motion. Newton's laws: F = ma.
Gravitational force: F = G*m1*m2/r^2. Objects fall at same acceleration g = 9.8m/s^2.
Energy is conserved. Kinetic energy = mv^2/2. Potential energy = mgh.
Work = force * distance. Power = work / time.
Momentum = mass * velocity. Conservation of momentum in collisions.
Thermodynamics: heat flows from hot to cold. Entropy increases. Energy conserved.
Electromagnetism: Maxwell's equations describe light as electromagnetic waves.""",
    "engenharia": """Engineering applies science to design solutions. Stress = force/area. Strain = deformation/original.
Young's modulus E = stress/strain. Materials deform elastically then plastically.
Beams bend under load. Moment of inertia resists bending.
Circuits: Ohm's law V = IR. Kirchhoff's laws for current and voltage.
Resistors in series: R_total = R1 + R2. In parallel: 1/R = 1/R1 + 1/R2.
Control theory uses feedback to maintain desired output. PID controllers.
Signal processing: Fourier transform decomposes signals into frequencies.""",
    "neurociencia": """Neuroscience studies the nervous system. Neurons communicate via electrical and chemical signals.
Action potentials travel down axons. Synapses release neurotransmitters.
The brain has ~86 billion neurons. Each neuron connects to ~7000 others.
The cortex handles perception, language, memory, and decision making.
The hippocampus consolidates memories. The amygdala processes emotions.
Neuroplasticity: the brain rewires itself based on experience.
EEG measures electrical activity. fMRI measures blood flow (BOLD signal).
Consciousness may emerge from integrated information (IIT theory).""",
    "quimica": """Chemistry studies matter and its transformations. Atoms have protons, neutrons, electrons.
The periodic table organizes elements by atomic number. Groups share chemical properties.
Chemical bonds: ionic (electron transfer), covalent (electron sharing), metallic.
Chemical reactions: reactants -> products. Rate depends on temperature and concentration.
Catalysts speed reactions without being consumed. Enzymes are biological catalysts.
pH measures acidity: pH = -log(H+). Acids donate H+, bases accept H+.
Thermochemistry: enthalpy, entropy, Gibbs free energy dG = dH - TdS.""",
    "anatomia": """Anatomy studies body structure. The skeletal system has 206 bones.
The muscular system has ~650 muscles. Cardiac muscle pumps blood.
The heart has 4 chambers. Blood flows: vena cava, right atrium, right ventricle, lungs, left atrium, left ventricle, aorta.
The brain has cerebrum, cerebellum, brainstem. Cerebrum has 4 lobes: frontal, parietal, temporal, occipital.
The nervous system: CNS (brain+spine) and PNS (peripheral nerves).
The respiratory system: trachea, bronchi, bronchioles, alveoli.
The immune system: white blood cells, antibodies, lymph nodes.""",
    "fibonacci": """Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34. Each number is sum of previous two.
F(n) = F(n-1) + F(n-2). The ratio of consecutive terms approaches the golden ratio phi = 1.618.
Fibonacci appears in nature: sunflower spirals, pinecones, nautilus shells, leaf arrangement.
The golden ratio phi = (1 + sqrt(5))/2. Binet's formula: F(n) = (phi^n - psi^n)/sqrt(5).
Used in algorithms: Fibonacci heap, Fibonacci search, Fibonacci coding.""",
    "arte_cultura": """Art expresses human experience across cultures. Renaissance: perspective, humanism, da Vinci, Michelangelo.
Baroque: drama, movement, Caravaggio, Bernini. Impressionism: light, color, Monet, Renoir.
Modernism: breaking conventions, Picasso, Kandinsky, Duchamp.
Music: melody (horizontal), harmony (vertical), rhythm (time). Sonata form: exposition, development, recapitulation.
Architecture: form follows function (modernism). Gothic: pointed arches, flying buttresses.
Culture includes language, beliefs, customs, art, and institutions shared by a group.""",
    "doencas": """Diseases disrupt normal body function. Infectious diseases: caused by pathogens.
COVID-19: caused by SARS-CoV-2 virus. Spreads through respiratory droplets.
Cancer: uncontrolled cell division. Caused by genetic mutations. Types: carcinoma, sarcoma, leukemia, lymphoma.
Cardiovascular disease: heart attacks, strokes. Risk factors: smoking, obesity, high blood pressure.
Diabetes: type 1 (autoimmune, no insulin), type 2 (insulin resistance).
Neurodegenerative: Alzheimer's (memory loss, amyloid plaques), Parkinson's (movement disorder, dopamine loss).""",
    "vacinas": """Vaccines train the immune system to fight specific pathogens.
mRNA vaccines (Pfizer, Moderna) deliver genetic instructions for spike protein.
Viral vector vaccines use harmless virus to deliver antigen.
Inactivated vaccines contain killed pathogen. Live attenuated use weakened pathogen.
Herd immunity occurs when enough population is vaccinated to protect vulnerable.
Adjuvants boost immune response. Boosters maintain immunity over time.
Vaccines eradicated smallpox. Polio nearly eradicated.""",
    "drogas_farmacia": """Pharmacology studies how drugs interact with the body.
Agonists activate receptors. Antagonists block receptors.
Antibiotics: penicillin (cell wall), tetracycline (protein synthesis).
Antivirals: acyclovir (herpes), oseltamivir (flu), remdesivir (COVID).
Analgesics: NSAIDs (ibuprofen, aspirin), opioids (morphine, fentanyl).
Psychiatric drugs: SSRIs (depression), benzodiazepines (anxiety).
Chemotherapy: alkylating agents, antimetabolites, targeted therapy.
Drug development: discovery, preclinical, Phase I, Phase II, Phase III, approval.""",
    "psicologia_cognitiva": """Cognitive psychology studies mental processes: perception, attention, memory, reasoning.
Working memory holds ~7 items (Miller's law).
Classical conditioning: Pavlov's dogs. Operant conditioning: rewards and punishments.
Piaget's stages: sensorimotor, preoperational, concrete operational, formal operational.
Cognitive biases: confirmation bias, availability heuristic.
Kahneman's System 1 (fast, intuitive) and System 2 (slow, analytical) thinking.
Theory of mind: understanding others have beliefs, desires, intentions different from one's own.""",
}

CODE_CURRICULUM = {
    "python_basics": """Python is a programming language. Variables store data: x = 5, name = hello.
Lists hold multiple items: items = [1, 2, 3]. Dictionaries map keys to values.
Functions are defined with def: def add(a, b): return a + b.
Classes create objects: class Dog: def bark(self): print(woof).
If statements branch: if x > 0: print(positive). Loops iterate: for i in range(5).
List comprehensions: squares = [x*x for x in range(10)].
Error handling: try: risky_code() except Exception as e: print(e).""",
    "javascript": """JavaScript runs in browsers and servers. Variables: let x = 5.
Functions: function add(a, b) { return a + b; }. Arrow: const add = (a,b) => a+b.
Objects: const person = { name: Alice, age: 30 }.
Arrays: const items = [1,2,3]; items.map(x => x*2).
Promises: fetch(url).then(res => res.json()).catch(err => console.log(err)).
Async/await: async function load() { const res = await fetch(url); }.
DOM: document.querySelector('.class').addEventListener('click', handler).""",
    "algorithms": """Sorting: bubble sort O(n^2), quick sort O(n log n), merge sort O(n log n).
Searching: binary search O(log n) on sorted arrays.
Graph algorithms: BFS shortest path, DFS explores all paths.
Dijkstra finds shortest weighted path. A* uses heuristics for pathfinding.
Dynamic programming: Fibonacci with memoization, knapsack problem.
Hash tables: O(1) average lookup. Big O: O(1), O(n), O(n^2), O(log n).""",
    "game_dev": """Game development uses game loops: while running: process_input(); update(); render().
Game objects have position, velocity, state: x, y, vx, vy, hp.
Collision detection: AABB collision between rectangles.
Sprite rendering: load images, draw at positions each frame.
Physics: gravity pulls objects down. Velocity changes position each frame.
State machines: menu, playing, paused, game_over.
Entity Component System: entities are IDs, components hold data, systems process.""",
}
