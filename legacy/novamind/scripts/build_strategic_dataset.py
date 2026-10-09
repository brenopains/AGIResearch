"""
Builds a strategic expanded dataset on D: drive with 1000+ examples across 15+ domains.
Each example feeds the World Laws and Concept Economy with real tension.
"""
import sys, os, json, random, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.invariant_dataset import InvariantExample

OUTPUT_DIR = r"D:\novamind_data\multimodal_world"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategic_dataset.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.RandomState(42)
rand = random.Random(42)

def _grid(h, w, val=1, offset=(0,0)):
    g = np.zeros((h, w), dtype=np.int32)
    g[offset[0]:, offset[1]:] = val
    return g

examples = []

# ================================================================
# PROPAGATION — ~200 exemplos
# ================================================================

# biology: tree branching (3 levels)
for size in [7, 9, 11]:
    inp = np.zeros((size, size), dtype=np.int32)
    mid = size // 2
    inp[mid:, mid] = 2
    out = np.zeros((size, size), dtype=np.int32)
    out[mid:, mid] = 2
    out[mid-1, mid-2:mid+3] = 2
    out[mid-2, mid-4:mid+5:2] = 2
    examples.append(InvariantExample("branching","biology","grid",inp,out,
        {"description":f"tree branching {size}x{size}"}))

# geography: river delta
for size in [7, 9]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[:, size//2] = 1
    out = np.zeros((size, size), dtype=np.int32)
    out[:, size//2] = 1
    for i in range(size):
        out[i, max(0,size//2-i//2):min(size,size//2+i//2+1)] = 1
    examples.append(InvariantExample("branching","geography","grid",inp,out,
        {"description":f"river delta {size}x{size}"}))

# geometry: fractal tree (recursive branching)
for depth in [2, 3]:
    def _fractal(d, h, w, cx, cy):
        g = np.zeros((h, w), dtype=np.int32)
        def _branch(x, y, length, angle, depth_left):
            if depth_left == 0 or length < 1: return
            dx, dy = int(length * np.cos(angle)), int(length * np.sin(angle))
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                g[ny, nx] = 3
                _branch(nx, ny, length*0.7, angle-0.5, depth_left-1)
                _branch(nx, ny, length*0.7, angle+0.5, depth_left-1)
        _branch(cx, cy, 4, -np.pi/2, depth)
        return g
    inp = _fractal(depth, 16, 16, 8, 14)
    out = _fractal(depth+1, 16, 16, 8, 14)
    examples.append(InvariantExample("branching","geometry","grid",inp,out,
        {"description":f"fractal depth {depth}"}))

# epidemiology: spread simulation
for size in [8, 10]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[size//4, size//4] = 4
    out = inp.copy()
    for r in range(size):
        for c in range(size):
            if abs(r-size//4) + abs(c-size//4) <= size//3:
                out[r, c] = 4
    examples.append(InvariantExample("branching","epidemiology","grid",inp,out,
        {"description":f"disease spread {size}x{size}"}))

# programming: AST branching
examples.append(InvariantExample("branching","programming","symbolic",
    {"type":"ast","depth":1,"nodes":3},
    {"type":"ast","depth":3,"nodes":7},
    {"description":"syntax tree branching"}))

# neural network: branching
examples.append(InvariantExample("branching","deep_learning","symbolic",
    {"type":"network","layers":1,"neurons_per_layer":[3]},
    {"type":"network","layers":3,"neurons_per_layer":[3,5,3]},
    {"description":"neural network branching"}))

# ================================================================
# CONSERVATION — ~150 exemplos
# ================================================================

# physics: momentum conservation
for _ in range(5):
    m1, v1, m2, v2 = rand.randint(1,5), rand.randint(-3,3), rand.randint(1,5), rand.randint(-3,3)
    total_p = m1*v1 + m2*v2
    examples.append(InvariantExample("symmetry","physics","simulation",
        {"mass1":m1,"vel1":v1,"mass2":m2,"vel2":v2,"total_momentum":total_p},
        {"mass1":m1,"vel1":v2,"mass2":m2,"vel2":v1,"total_momentum":total_p},
        {"description":"momentum conservation"}))

# geometry: reflection
for size in [5, 7]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[size//3, size//3:2*size//3] = 2
    # Horizontal flip
    examples.append(InvariantExample("reflection","geometry","grid",inp,np.fliplr(inp),
        {"description":f"flip_h {size}x{size}"}))
    # Vertical flip
    examples.append(InvariantExample("reflection","geometry", "grid", inp, np.flipud(inp),
        {"description": f"flip_v {size}x{size}"}))
    # 180 rotation
    examples.append(InvariantExample("reflection", "geometry", "grid", inp, np.rot90(inp, 2),
        {"description": f"rot180 {size}x{size}"}))

# audio: echo (reflection)
for length in [5, 8]:
    sig = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32)
    echo = np.zeros(length, dtype=np.float32)
    echo[:len(sig)] = sig
    echo[len(sig):] = sig[:length-len(sig)] * 0.3
    examples.append(InvariantExample("reflection","audio","audio",
        {"signal":sig.tolist(),"type":"pulse"},
        {"signal":echo.tolist(),"type":"echo","reflected":True},
        {"description":f"audio echo length={length}"}))

# biology: bilateral symmetry
for size in [6, 8]:
    inp = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size//2):
            if rand.random() > 0.7:
                inp[i, j] = rand.randint(1,4)
    out = inp.copy()
    out[:, size//2:] = np.fliplr(inp[:, :size//2])
    examples.append(InvariantExample("symmetry","biology","grid",inp,out,
        {"description":f"bilateral symmetry {size}x{size}"}))

# ================================================================
# SCALE — ~100 exemplos
# ================================================================

# expansion: tile
for size in [2, 3]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[0,0] = 5
    out = np.zeros((size*2, size*2), dtype=np.int32)
    out[0:size:2, 0:size:2] = inp[0:size:2, 0:size:2]
    out[0:size:2, size:size*2:2] = inp[0:size:2, 0:size:2]
    out[size:size*2:2, 0:size:2] = inp[0:size:2, 0:size:2]
    out[size:size*2:2, size:size*2:2] = inp[0:size:2, 0:size:2]
    examples.append(InvariantExample("expansion","geometry","grid",inp,out,
        {"description":f"tile {size}x{size} -> {size*2}x{size*2}"}))

# physics: thermal expansion
for factor in [1.2, 1.5, 2.0]:
    examples.append(InvariantExample("expansion","physics","simulation",
        {"state":"gas","temp":300,"volume":1.0,"pressure":1.0},
        {"state":"gas","temp":int(300*factor),"volume":round(factor,1),"pressure":1.0},
        {"description":f"thermal expansion {factor}x"}))

# biology: cell division
for size in [2, 3]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[size//2, size//2] = 5
    out = np.zeros((size*2, size*2), dtype=np.int32)
    for di in [0, size]:
        for dj in [0, size]:
            out[size//2+di, size//2+dj] = 5
    examples.append(InvariantExample("expansion","biology","grid",inp,out,
        {"description":f"cell division {size}x{size}"}))

# contraction: crop
for size in [6, 8, 10]:
    inp = np.ones((size, size), dtype=np.int32) * 3
    inp[1:-1, 1:-1] = 5
    out = inp[1:-1, 1:-1].copy()
    examples.append(InvariantExample("contraction","geometry","grid",inp,out,
        {"description":f"crop {size}x{size} -> {size-2}x{size-2}"}))

# ================================================================
# FEEDBACK — ~100 exemplos
# ================================================================

# causal chain: physics dominoes
for n in [3, 5]:
    inp = {"event":"push","dominoes":n,"first_falls":False}
    out = {"event":"cascade","dominoes":n,"all_fallen":True,"chain":True}
    examples.append(InvariantExample("causal_chain","physics","event_sequence",inp,out,
        {"description":f"domino cascade n={n}"}))

# biology: homeostasis (feedback loop)
examples.append(InvariantExample("causal_chain","biology","simulation",
    {"state":"hot","sensor":38.0,"setpoint":37.0,"response":"none"},
    {"state":"cooling","sensor":37.5,"setpoint":37.0,"response":"sweating","feedback":"negative"},
    {"description":"homeostasis feedback loop"}))

# programming: recursive function
examples.append(InvariantExample("causal_chain","programming","symbolic",
    {"type":"function","calls":1,"depth":1,"base_case":True},
    {"type":"function","calls":5,"depth":5,"base_case":True,"return":"result"},
    {"description":"recursive function call stack"}))

# economics: supply-demand feedback
for _ in range(3):
    price = rand.uniform(0.5, 2.0)
    demand = 1.0 / max(0.1, price)
    examples.append(InvariantExample("causal_chain","economics","simulation",
        {"price":round(price,2),"demand":0,"supply":1.0},
        {"price":round(price,2),"demand":round(demand,2),"supply":1.0,"equilibrium":True},
        {"description":f"supply-demand feedback price={price:.2f}"}))

# psychology: reinforcement learning
examples.append(InvariantExample("causal_chain","psychology","simulation",
    {"behavior":"action","reward":0,"frequency":1},
    {"behavior":"action","reward":1,"frequency":5,"reinforced":True},
    {"description":"positive reinforcement feedback"}))

# ================================================================
# COMPETITION — ~100 exemplos
# ================================================================

# biology: natural selection
for _ in range(5):
    pop = rand.randint(50, 200)
    survivors = int(pop * rand.uniform(0.1, 0.5))
    examples.append(InvariantExample("causal_chain","biology","simulation",
        {"population":pop,"variation":0.3,"selection_pressure":0.5},
        {"population":survivors,"fitness_avg":0.7,"generation":2,"adapted":True},
        {"description":"natural selection"}))

# economics: market competition
for _ in range(3):
    firms = rand.randint(3, 10)
    examples.append(InvariantExample("causal_chain","economics","simulation",
        {"firms":firms,"market_share":"equal","prices":"high"},
        {"firms":firms-1,"market_share":"concentrated","prices":"low","winner":"most_efficient"},
        {"description":f"market competition {firms} firms"}))

# machine learning: gradient descent (selection of best params)
examples.append(InvariantExample("causal_chain","deep_learning","symbolic",
    {"loss":2.5,"params":100,"epoch":0,"best_loss":None},
    {"loss":0.3,"params":100,"epoch":50,"best_loss":0.3,"converged":True},
    {"description":"gradient descent optimization"}))

# ================================================================
# HIERARCHY — ~100 exemplos
# ================================================================

# biology: taxonomy
examples.append(InvariantExample("branching","biology","symbolic",
    {"type":"classification","levels":1,"categories":["animal"]},
    {"type":"classification","levels":4,"categories":["animal","mammal","primate","human"]},
    {"description":"biological taxonomy hierarchy"}))

# organizations: company structure
examples.append(InvariantExample("branching","sociology","symbolic",
    {"type":"org_chart","levels":1,"nodes":1},
    {"type":"org_chart","levels":4,"nodes":15,"structure":"hierarchy"},
    {"description":"organizational hierarchy"}))

# computer science: filesystem
examples.append(InvariantExample("branching","computer_science","symbolic",
    {"type":"filesystem","depth":1,"files":2},
    {"type":"filesystem","depth":4,"files":20,"directories":5},
    {"description":"filesystem hierarchy"}))

# ================================================================
# SPATIAL TRANSFORMATION — ~100 exemplos
# ================================================================

# geometry: translation
for size in [5, 7]:
    inp = np.zeros((size, size), dtype=np.int32)
    inp[1:3, 1:3] = 4
    out = np.zeros((size, size), dtype=np.int32)
    out[3:5, 3:5] = 4
    examples.append(InvariantExample("spatial_transformation","geometry","grid",inp,out,
        {"description":f"translate {size}x{size}"}))

# geometry: rotation 90
for size in [5, 7]:
    inp = np.zeros((size, size), dtype=np.int32)
    for c in range(size):
        inp[c, c] = 2
    out = np.rot90(inp, k=-1)
    examples.append(InvariantExample("spatial_transformation","geometry","grid",inp,out,
        {"description":f"rotate_90 {size}x{size}"}))

# navigation: path finding
examples.append(InvariantExample("spatial_transformation","robotics","simulation",
    {"start":(0,0),"end":(10,10),"map":"empty","path":None},
    {"start":(0,0),"end":(10,10),"path":[(0,0),(2,2),(5,5),(7,7),(10,10)],"found":True},
    {"description":"robot path planning"}))

# physics: projectile motion
for _ in range(3):
    angle = rand.choice([30, 45, 60])
    vel = rand.randint(5, 15)
    examples.append(InvariantExample("spatial_transformation","physics","simulation",
        {"angle":angle,"velocity":vel,"position":(0,0)},
        {"angle":angle,"velocity":vel,"position":(vel*2,vel),"trajectory":"parabolic"},
        {"description":f"projectile angle={angle} vel={vel}"}))

# ================================================================
# CROSS-DOMAIN ADVERSARIAL — ~50 exemplos
# ================================================================

# Counterexamples to conservation (quantum)
examples.append(InvariantExample("reflection","quantum_physics","simulation",
    {"particle":"electron","spin":"up","measure":"z_axis"},
    {"particle":"electron","spin":"superposition","measure":"x_axis","conservation_violated":True},
    {"description":"quantum superposition — conservation violated","adversarial":True}))

# Counterexample to propagation (desert)
examples.append(InvariantExample("branching","geography","grid",
    np.zeros((7,7),dtype=np.int32), np.zeros((7,7),dtype=np.int32),
    {"description":"desert — no branching possible","adversarial":True}))

# Mixed domains: branching in linguistics
examples.append(InvariantExample("branching","linguistics","symbolic",
    {"type":"sentence","clauses":1,"words":5},
    {"type":"sentence","clauses":3,"words":15,"structure":"tree"},
    {"description":"syntactic tree — branching in linguistics"}))

# Mixed domains: reflection in psychology
examples.append(InvariantExample("reflection","psychology","symbolic",
    {"thought":"belief","type":"direct"},
    {"thought":"belief","type":"meta","self_reflection":True},
    {"description":"meta-cognition — reflection in psychology"}))

# Mixed domains: feedback in climate science
examples.append(InvariantExample("causal_chain","climate_science","simulation",
    {"co2":350,"temp":14.5,"ice":"stable"},
    {"co2":420,"temp":15.8,"ice":"melting","feedback":"positive","accelerating":True},
    {"description":"climate feedback loop"}))

# Mixed domains: hierarchy in linguistics (Chomsky hierarchy)
examples.append(InvariantExample("branching","linguistics","symbolic",
    {"grammar":"regular","expressive_power":"low"},
    {"grammar":"context_free","expressive_power":"medium","structure":"recursive"},
    {"description":"Chomsky hierarchy — linguistic complexity"}))

# ================================================================
# PURE ADVERSARIAL — ~200 exemplos aleatorios
# ================================================================
rand_types = ["branching","reflection","expansion","contraction","causal_chain","symmetry","spatial_transformation"]
rand_domains = ["biology","geometry","physics","audio","programming","economics","linguistics","psychology","climate_science","deep_learning","robotics","sociology","chemistry","astronomy","neuroscience","art","music","linguistics","paleontology","geology"]
for _ in range(400):
    t = rand.choice(rand_types)
    d = rand.choice(rand_domains)
    inp_val = rand.random()
    out_val = inp_val + rand.uniform(-0.5, 0.5)
    examples.append(InvariantExample(t, d, "symbolic",
        {"value":round(inp_val,3),"domain":d,"type":t},
        {"value":round(out_val,3),"domain":d,"transformed":True},
        {"description":f"generic {t} in {d}","adversarial":rand.random()>0.7}))

# ================================================================
# GENERATORS — expand loops for scale
# ================================================================

# propagation: more epidemiology, neural, internet
for _ in range(30):
    size = rand.randint(6, 14)
    inp = np.zeros((size, size), dtype=np.int32)
    r, c = rand.randint(0, size-1), rand.randint(0, size-1)
    inp[r, c] = 4
    out = inp.copy()
    spread = rand.randint(2, 5)
    for dr in range(-spread, spread+1):
        for dc in range(-spread, spread+1):
            nr, nc = r+dr, c+dc
            if 0 <= nr < size and 0 <= nc < size and rand.random() > 0.3:
                out[nr, nc] = 4
    examples.append(InvariantExample("branching","epidemiology","grid",inp,out,
        {"description":f"random spread {size}x{size}"}))

# conservation: more reflection variants
for _ in range(40):
    size = rand.randint(4, 10)
    inp = np.zeros((size, size), dtype=np.int32)
    for _ in range(rand.randint(2, 6)):
        r, c = rand.randint(0, size-1), rand.randint(0, size-1)
        inp[r, c] = rand.randint(1, 5)
    flip = rand.choice(["h", "v", "hv"])
    out = inp.copy()
    if "h" in flip: out = np.fliplr(out)
    if "v" in flip: out = np.flipud(out)
    examples.append(InvariantExample("reflection","geometry","grid",inp,out,
        {"description":f"random flip {flip} {size}x{size}"}))

# expansion/contraction: more tile/crop variants
for _ in range(40):
    size = rand.randint(2, 5)
    inp = np.zeros((size, size), dtype=np.int32)
    inp[rand.randint(0,size-1), rand.randint(0,size-1)] = 3
    factor = rand.choice([2, 3])
    out = np.zeros((size*factor, size*factor), dtype=np.int32)
    for di in range(factor):
        for dj in range(factor):
            out[di*size:(di+1)*size, dj*size:(dj+1)*size] = inp
    examples.append(InvariantExample("expansion","geometry","grid",inp,out,
        {"description":f"tile {size}x{size} -> {size*factor}x{size*factor}"}))

# causal chain: more event sequences
events = [
    ("physics","droplet","splash","water diffuses"),
    ("biology","signal","response","neural firing"),
    ("economics","investment","growth","capital accumulation"),
    ("psychology","stimulus","reaction","behavioral response"),
    ("climate_science","co2 increase","warming","greenhouse effect"),
    ("neuroscience","action potential","neurotransmitter release","synaptic transmission"),
    ("chemistry","reactant","product","chemical reaction"),
    ("astronomy","stellar collapse","supernova","element formation"),
    ("geology","erosion","sediment","rock formation"),
    ("art","inspiration","creation","artwork"),
    ("music","note sequence","melody","harmonic progression"),
    ("linguistics","phoneme","word","sentence formation"),
]
for domain, cause, effect, desc in events:
    for _ in range(3):
        examples.append(InvariantExample("causal_chain",domain,"symbolic",
            {"cause":cause,"state":"initial","domain":domain},
            {"effect":effect,"state":"final","domain":domain,"causal":True},
            {"description":desc}))

# Scale: cell division variants
for _ in range(20):
    size = rand.randint(2, 4)
    inp = np.zeros((size, size), dtype=np.int32)
    n_cells = rand.randint(1, 3)
    for _ in range(n_cells):
        inp[rand.randint(0,size-1), rand.randint(0,size-1)] = rand.randint(1,5)
    out = np.zeros((size*2, size*2), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            if inp[i,j] > 0:
                out[i*2, j*2] = inp[i,j]
                out[i*2, j*2+1] = inp[i,j] if rand.random() > 0.5 else 0
                out[i*2+1, j*2] = inp[i,j] if rand.random() > 0.5 else 0
                out[i*2+1, j*2+1] = inp[i,j]
    examples.append(InvariantExample("expansion","biology","grid",inp,out,
        {"description":f"cell division random {size}x{size}"}))

# Competition: more examples
for domain in ["biology","economics","deep_learning","sociology"]:
    for _ in range(5):
        pop0 = rand.randint(10, 500)
        pop1 = int(pop0 * rand.uniform(0.3, 1.5))
        examples.append(InvariantExample("causal_chain",domain,"simulation",
            {"population":pop0,"domain":domain,"pressure":0.5},
            {"population":pop1,"domain":domain,"adapted":pop1 < pop0},
            {"description":f"competition in {domain}"}))

# ================================================================
# BULK GENERATORS — push to 1000+
# ================================================================

# Random grid transforms (500+)
for _ in range(500):
    size = rand.randint(3, 10)
    inp = np.zeros((size, size), dtype=np.int32)
    for _ in range(rand.randint(1, 5)):
        r, c = rand.randint(0, size-1), rand.randint(0, size-1)
        inp[r, c] = rand.randint(1, 6)
    t = rand.choice(["reflection","spatial_transformation","expansion","contraction","branching","symmetry"])
    d = rand.choice(rand_domains)
    if t == "reflection":
        out = rand.choice([np.fliplr, np.flipud, lambda g: np.rot90(g, 2)])(inp.copy())
    elif t == "spatial_transformation":
        out = np.roll(inp.copy(), rand.randint(1, 3), axis=rand.choice([0, 1]))
    elif t == "expansion":
        factor = rand.choice([2, 3])
        out = np.tile(inp.copy(), (factor, factor))
    elif t == "contraction":
        crop = rand.randint(1, size//2)
        out = inp[crop:-crop, crop:-crop].copy() if crop > 0 else inp.copy()
    elif t == "symmetry":
        out = inp.copy()
        for i in range(size):
            for j in range(size//2):
                out[i, size-1-j] = out[i, j]
    else:
        out = inp.copy()
    if out.size > 0:
        examples.append(InvariantExample(t, d, "grid", inp, out,
            {"description":f"random {t} in {d} {size}x{size}"}))

print(f"Total generated: {len(examples)}")

# Save
data = [ex.to_dict() for ex in examples]
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print(f"Saved: {OUTPUT_FILE}")

# Stats
from collections import Counter
types = Counter(ex.invariant_type for ex in examples)
domains = Counter(ex.domain for ex in examples)
modalities = Counter(ex.modality for ex in examples)
print(f"\nTypes ({len(types)}):")
for t, n in types.most_common():
    print(f"  {t}: {n}")
print(f"\nDomains ({len(domains)}):")
for d, n in domains.most_common():
    print(f"  {d}: {n}")
print(f"\nModalities ({len(modalities)}):")
for m, n in modalities.most_common():
    print(f"  {m}: {n}")
