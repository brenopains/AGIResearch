"""Curriculo Cientifico — Ensina causalidade, fisica, biologia para a AGI.

Adiciona episodios com relacoes causais explicitas:
  - Causa-efeito (fogo → calor, agua → ferver)
  - Taxonomia cientifica (mamifero → respira, peixe → guelras)
  - Relacoes fisicas (gravidade → queda, atrito → calor)

Gera episodios com HDVs semanticos + arestas CAUSES no CGE.
"""
import sys, os, time, sqlite3, hashlib
sys.path.insert(0, ('C:/Users/usuario/Desktop/AGI 2/novamind').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
import numpy as np
import spacy
import json as _json
# ─── Config ──────────────────────────────────────────────────
MEMORY_DIR = r"D:\AGI2\memories"
EPISODE_DB = os.path.join(MEMORY_DIR, "episodes.db")
CURROOT = os.path.join(MEMORY_DIR, "curriculum")

os.makedirs(CURROOT, exist_ok=True)

# ─── Encoder semantico ─────────────────────────────────────
nlp = spacy.load("en_core_web_lg")
np.random.seed(46)
PROJ = np.random.randn(10000, 300).astype(np.float32) / np.sqrt(300)

def text_to_hdv(text):
    doc = nlp(str(text)[:1000])
    vecs = [t.vector for t in doc if t.has_vector and t.vector_norm > 0]
    if not vecs:
        return np.zeros(10000, dtype=np.float32)
    avg = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    hdv = np.sign(PROJ @ avg)
    hdv[hdv == 0] = 1
    return hdv.astype(np.float32)

# ─── Conhecimento cientifico ─────────────────────────────────

SCIENCE_FACTS = [
    # Causa e efeito (fisica)
    ("fire causes heat", "fire", "heat"),
    ("heat boils water", "heat", "boil"),
    ("water boils at 100 degrees", "water", "boil"),
    ("cold freezes water", "cold", "freeze"),
    ("gravity pulls objects down", "gravity", "fall"),
    ("friction produces heat", "friction", "heat"),
    ("electricity powers machines", "electricity", "power"),
    ("light travels faster than sound", "light", "travel"),
    ("sound travels through air", "sound", "travel"),
    ("magnetism attracts iron", "magnetism", "attract"),
    
    # Biologia
    ("mammals breathe air", "mammal", "breathe"),
    ("fish have gills", "fish", "gill"),
    ("birds have wings", "bird", "wing"),
    ("plants need sunlight", "plant", "sunlight"),
    ("cells are the unit of life", "cell", "life"),
    ("dna contains genetic code", "dna", "genetic"),
    ("the heart pumps blood", "heart", "pump"),
    ("the lungs exchange oxygen", "lung", "oxygen"),
    ("the brain controls the body", "brain", "control"),
    ("muscles pull bones", "muscle", "pull"),
    
    # Quimica
    ("water is H2O", "water", "h2o"),
    ("salt dissolves in water", "salt", "dissolve"),
    ("metal conducts electricity", "metal", "conduct"),
    ("acid corrodes metal", "acid", "corrode"),
    ("oxygen supports burning", "oxygen", "burn"),
    ("carbon dioxide is a gas", "carbon_dioxide", "gas"),
    ("hydrogen is flammable", "hydrogen", "flammable"),
    
    # Astronomia
    ("earth orbits the sun", "earth", "orbit"),
    ("the moon orbits earth", "moon", "orbit"),
    ("gravity keeps planets in orbit", "gravity", "orbit"),
    ("stars produce light", "star", "light"),
    ("the sun is a star", "sun", "star"),
    
    # Matematica (relacional)
    ("two plus two equals four", "addition", "equals"),
    ("a square has four sides", "square", "side"),
    ("a triangle has three sides", "triangle", "side"),
    ("a circle has no corners", "circle", "corner"),
    
    # Engenharia
    ("a lever multiplies force", "lever", "force"),
    ("a wheel reduces friction", "wheel", "friction"),
    ("a pulley lifts heavy objects", "pulley", "lift"),
    ("a screw converts rotation to linear motion", "screw", "convert"),
    
    # Relacoes de classe (taxonomia cientifica)
    ("whale is a mammal", "whale", "mammal"),
    ("dolphin is a mammal", "dolphin", "mammal"),
    ("bat is a mammal", "bat", "mammal"),
    ("eagle is a bird", "eagle", "bird"),
    ("penguin is a bird", "penguin", "bird"),
    ("sparrow is a bird", "sparrow", "bird"),
    ("shark is a fish", "shark", "fish"),
    ("salmon is a fish", "salmon", "fish"),
    ("frog is an amphibian", "frog", "amphibian"),
    ("snake is a reptile", "snake", "reptile"),
    ("turtle is a reptile", "turtle", "reptile"),
    ("ant is an insect", "ant", "insect"),
    ("bee is an insect", "bee", "insect"),
    ("spider is an arachnid", "spider", "arachnid"),
    ("rose is a flower", "rose", "flower"),
    ("oak is a tree", "oak", "tree"),
    ("pine is a tree", "pine", "tree"),
    ("gold is a metal", "gold", "metal"),
    ("iron is a metal", "iron", "metal"),
    ("copper is a metal", "copper", "metal"),
    ("diamond is a gemstone", "diamond", "gemstone"),
    ("ruby is a gemstone", "ruby", "gemstone"),
    ("quartz is a mineral", "quartz", "mineral"),
    ("granite is a rock", "granite", "rock"),
    ("marble is a rock", "marble", "rock"),
    
    # Propriedades
    ("ice is solid water", "ice", "solid"),
    ("steam is gaseous water", "steam", "gas"),
    ("wood is flammable", "wood", "flammable"),
    ("glass is transparent", "glass", "transparent"),
    ("rubber is elastic", "rubber", "elastic"),
    ("iron is magnetic", "iron", "magnetic"),
    ("copper conducts electricity", "copper", "conduct"),
    
    # Cadeias causais complexas
    ("sunlight warms the ground", "sunlight", "warm"),
    ("warm air rises", "warm_air", "rise"),
    ("rising air creates wind", "wind", "create"),
    ("wind moves clouds", "wind", "move"),
    ("clouds produce rain", "cloud", "rain"),
    ("rain waters plants", "rain", "water"),
    ("plants produce oxygen", "plant", "oxygen"),
    ("animals breathe oxygen", "animal", "oxygen"),
    
    # Comportamento animal
    ("cats purr when happy", "cat", "purr"),
    ("dogs bark", "dog", "bark"),
    ("birds sing", "bird", "sing"),
    ("fish swim", "fish", "swim"),
    ("snakes slither", "snake", "slither"),
    ("birds fly", "bird", "fly"),
    ("horses run", "horse", "run"),
    
    # Tempo e clima
    ("rain falls from clouds", "rain", "fall"),
    ("snow is frozen rain", "snow", "freeze"),
    ("wind blows", "wind", "blow"),
    ("thunder follows lightning", "thunder", "follow"),
    ("seasons change yearly", "season", "change"),
]


FACTS_TOTAL = len(SCIENCE_FACTS)
print(f"[SCIENCE] {FACTS_TOTAL} fatos cientificos para adicionar")


# ─── Gerar HDVs e salvar ───────────────────────────────────

def generate_science_curriculum():
    """Gera episodios de ciencia e salva em arquivo JSON."""
    episodes = []
    for i, (text, concept_a, concept_b) in enumerate(SCIENCE_FACTS):
        hdv = text_to_hdv(text).tolist()
        hdv_a = text_to_hdv(concept_a).tolist()
        hdv_b = text_to_hdv(concept_b).tolist()
        ep = {
            "id": f"science_{i}",
            "input": text,
            "concept_a": concept_a,
            "concept_b": concept_b,
            "text_hdv": hdv,
            "concept_a_hdv": hdv_a,
            "concept_b_hdv": hdv_b,
        }
        episodes.append(ep)
    
    out_file = os.path.join(CURROOT, "science_curriculum.json")
    with open(out_file, "w") as f:
        _json.dump(episodes, f, indent=1)
    print(f"[SCIENCE] Salvo: {len(episodes)} episodios em {out_file}")
    return out_file


# ─── Integrar na EpisodeStore ──────────────────────────────

def integrate_into_store(storage_file):
    import json as _j
    with open(storage_file) as f:
        episodes = _j.load(f)
    conn = sqlite3.connect(EPISODE_DB)
    c = conn.cursor()
    existing = c.execute("SELECT COUNT(*) FROM episodes WHERE type='science_fact'").fetchone()[0]
    if existing >= len(episodes):
        print(f"[SCIENCE] Ja existem {existing} episodios de ciencia")
        conn.close(); return
    c.execute("DELETE FROM episodes WHERE type='science_fact'")
    for ep in episodes:
        c.execute("INSERT INTO episodes (id, type, content, importance, surprise) VALUES (?,?,?,?,?)",
                  (ep["id"], "science_fact", _j.dumps({"input": ep["input"]}), 0.8, 0.05))
    conn.commit(); conn.close()
    hdvs = __import__("numpy").array([ep["text_hdv"] for ep in episodes], dtype=__import__("numpy").float32)
    hdv_file = os.path.join(CURROOT, "science_hdvs.npy")
    __import__("numpy").save(hdv_file, hdvs)
    print(f"[SCIENCE] Integrados: {len(episodes)} episodios na EpisodeStore")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrate", action="store_true", help="Integrar na EpisodeStore")
    args = parser.parse_args()
    
    json_file = generate_science_curriculum()
    
    if args.integrate:
        integrate_into_store(json_file)
    else:
        print(f"\n[SCIENCE] Use --integrate para adicionar ao banco")
        print(f"[SCIENCE] Depois execute _merge_hdvs.py para fundir HDVs")
