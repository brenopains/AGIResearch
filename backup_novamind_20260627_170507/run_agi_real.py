"""AGI Cerebro Completo — Loop Emergente com MetaController + SemanticWorkspace"""
import sys, os, time, threading, json, random, hashlib, math, io
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
os.chdir(_BASE)
import torch
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, quote, parse_qs

from core.state_bridge import StateBridge
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.self_model import SelfModel

# ─── NOVOS MODULOS ───
from core.meta_controller import MetaController
from core.mode_registry import MODE_REGISTRY
from core.error_signal import MultiErrorTracker
from core.semantic_workspace import SemanticWorkspace
from core.mental_scene_graph import MentalSceneGraph
from core.cognitive_trace import TraceRecorder, CognitiveTrace
from core.cognitive_notebook import CognitiveNotebook
from core.insight_library import InsightLibrary
from core.model_competition import ModelEcosystem, Model
from core.forja_evolution import ForjaEvolution, MetaForja
from core.cognitive_program import LoopEvaluator
from core.cognitive_stances import StanceLibrary
from core.failed_trace import FailedTraceDB
from core.cognitive_archive import CognitiveArchive, ArchiveItem, MemoryLayer
from core.meta_evolution import MetaEvolution, ValueSystem
from core.constraint_engine import ConstraintEngine
from core.insight_graph import InsightGraph

# Renderizadores
from core.mental_renderer import (MentalRenderer, VisualRenderer,
                                  AudioRenderer, TextRenderer, CodeRenderer)

# ─── GPU ───
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE.upper()}")

# ─── PIPELINES OPCIONAIS (se disponiveis) ───
GROUNDING = None
VISUAL_PIPELINE = None
AUDIO_VOCODER = None

try:
    from core.grounding_engine import GroundingEngine
    GROUNDING = GroundingEngine(device=DEVICE)
    from core.visual_pipeline import VisualPipeline
    VISUAL_PIPELINE = VisualPipeline(GROUNDING.proj_visual, device=DEVICE)
    print("[OK] GroundingEngine + VisualPipeline")
except Exception as e:
    print(f"[!] VisualPipeline nao disponivel: {e}")

try:
    from core.audio_vocoder import AudioVocoder
    AUDIO_VOCODER = AudioVocoder(device=DEVICE)
    vocoder_path = os.path.join(os.path.dirname(__file__), "save", "vocoder.pt")
    if os.path.exists(vocoder_path):
        AUDIO_VOCODER.load_pretrained(vocoder_path)
        print("[OK] AudioVocoder carregado")
    else:
        print("[OK] AudioVocoder (nao treinado)")
except Exception as e:
    print(f"[!] AudioVocoder nao disponivel: {e}")

# ─── BACKGROUND INCUBATOR ───
INCUBATOR = None
try:
    from core.background_incubator import BackgroundIncubator
    INCUBATOR = BackgroundIncubator(
        tick_interval=3.0, fe_threshold=0.25, fe_drop_ratio=0.5,
        device=DEVICE)
    print("[OK] BackgroundIncubator pronto")
except Exception as e:
    print(f"[!] BackgroundIncubator nao disponivel: {e}")

# ─── PERSISTENCIA ───
_SAVE_DIR = os.path.join(_BASE, "save")
os.makedirs(_SAVE_DIR, exist_ok=True)

def save(nodes, edges, tag="live"):
    path = os.path.join(_SAVE_DIR, f"brain_{tag}.pth")
    simple_edges = []
    for e in edges:
        ed = {k: v for k, v in e.items()}
        et = ed.get("edge_type")
        if isinstance(et, torch.Tensor):
            if et.numel() == 1:
                ed["edge_type"] = int(et.item())
            else:
                ed["edge_type"] = int(et[0].item()) if et[0].numel() == 1 else str(et.tolist())
        simple_edges.append(ed)
    data = {"nodes": dict(nodes), "edges": simple_edges, "time": time.time()}
    torch.save(data, path)
    return path

def load(cge, tag="live"):
    path = os.path.join(_SAVE_DIR, f"brain_{tag}.pth")
    if not os.path.exists(path):
        return False
    data = torch.load(path, weights_only=False)
    if "nodes" in data:
        for nid, nd in data["nodes"].items():
            cge.nodes[nid] = nd
        for e in data.get("edges", []):
            cge.edges.append(e)
    return len(data["nodes"]) > 0

# ─── INICIALIZACAO ───
bridge = StateBridge.instance()
bridge.clear()
cge = CognitiveGraphEngine(device=DEVICE)
loaded = load(cge)
sm = SelfModel()

# Sistemas cognitivos emergentes
error_tracker = MultiErrorTracker(max_errors=16)
workspace = SemanticWorkspace(capacity=50)
meta = MetaController(error_tracker=error_tracker, cge=cge)
trace_rec = TraceRecorder(cge)
notebook = CognitiveNotebook(cge=cge)
insights = InsightLibrary(cge=cge)
model_eco = ModelEcosystem(max_models=24)
loop_eval = LoopEvaluator(max_programs=32)
forja_evol = ForjaEvolution(cge=cge, insight_library=insights,
                            loop_evaluator=loop_eval)
stances = StanceLibrary(meta_controller=meta, cge=cge)
meta_forja = MetaForja(operator_forge=None, loop_evaluator=loop_eval,
                       stance_library=stances, cge=cge,
                       insight_library=insights)
fail_db = FailedTraceDB(cge=cge, insight_library=insights)
archive = CognitiveArchive(cge=cge)
values = ValueSystem(cge=cge)
meta_evol = MetaEvolution(meta_controller=meta, workspace=workspace,
                          stance_library=stances, cge=cge,
                          value_system=values)
constraints = ConstraintEngine(cge=cge)
insight_graph = InsightGraph(cge=cge)

# Renderer plugavel
renderer = MentalRenderer(
    visual=VisualRenderer(visual_pipeline=VISUAL_PIPELINE),
    audio=AudioRenderer(vocoder=AUDIO_VOCODER),
    text=TextRenderer(),
    code=CodeRenderer(),
)

# Callback Eureka: quando incubator encontra insight, posta no workspace + notebook
def _on_eureka(event):
    workspace.post(
        content=f"EUREKA: {event.description[:80]}",
        source="background", weight=0.9, novelty=0.8, urgency=0.6,
        tags=["eureka", "insight"])
    notebook.add_insight(
        f"EUREKA: {event.description[:200]}",
        confidence=0.7)
    bridge.add_thought(f"[EUREKA] {event.description[:120]}", "insight")

if INCUBATOR:
    INCUBATOR.on_eureka = _on_eureka

bridge.set_from_cge(cge)
bridge.set_from_self_model(sm)
bridge.update(running=True)

if loaded and len(cge.nodes) > 0:
    bridge.add_thought(f"CARREGOU {len(cge.nodes)} nos, {len(cge.edges)} arestas (GPU={DEVICE})", "system")
else:
    bridge.add_thought(f"Cerebro novo — GPU={DEVICE}", "system")

# ─── MULTILINGUA ───
TRANSLATIONS = {
    "gravity": {"pt":"gravidade","es":"gravedad","fr":"gravite","de":"schwerkraft"},
    "energy": {"pt":"energia","es":"energia","fr":"energie","de":"energie"},
    "evolution": {"pt":"evolucao","es":"evolucion","fr":"evolution","de":"evolution"},
    "dna": {"pt":"dna","es":"adn","fr":"adn","de":"dns"},
    "galaxy": {"pt":"galaxia","es":"galaxia","fr":"galaxie","de":"galaxie"},
    "star": {"pt":"estrela","es":"estrella","fr":"etoile","de":"stern"},
    "planet": {"pt":"planeta","es":"planeta","fr":"planete","de":"planet"},
    "language": {"pt":"linguagem","es":"lenguaje","fr":"langage","de":"sprache"},
    "consciousness": {"pt":"consciencia","es":"conciencia","fr":"conscience","de":"bewusstsein"},
    "algorithm": {"pt":"algoritmo","es":"algoritmo","fr":"algorithme","de":"algorithmus"},
    "entropy": {"pt":"entropia","es":"entropia","fr":"entropie","de":"entropie"},
    "photosynthesis": {"pt":"fotossintese","es":"fotosintesis","fr":"photosynthese","de":"photosynthese"},
    "black hole": {"pt":"buraco negro","es":"agujero negro","fr":"trou noir","de":"schwarzes loch"},
    "supernova": {"pt":"supernova","es":"supernova","fr":"supernova","de":"supernova"},
    "neural network": {"pt":"rede neural","es":"red neuronal","fr":"reseau neuronal","de":"neuronales netz"},
    "quantum": {"pt":"quantico","es":"cuantico","fr":"quantique","de":"quanten"},
    "probability": {"pt":"probabilidade","es":"probabilidad","fr":"probabilite","de":"wahrscheinlichkeit"},
    "cell": {"pt":"celula","es":"celula","fr":"cellule","de":"zelle"},
    "force": {"pt":"forca","es":"fuerza","fr":"force","de":"kraft"},
    "mass": {"pt":"massa","es":"masa","fr":"masse","de":"masse"},
}

def all_names(concept: str) -> list:
    names = [concept]
    if concept.lower() in TRANSLATIONS:
        for t in TRANSLATIONS[concept.lower()].values():
            names.append(t)
    return names

def make_hdv(text: str) -> torch.Tensor:
    h = hashlib.sha256(text.encode()).hexdigest()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(int(h[:8], 16))
    return torch.sign(torch.randn(10000, generator=g, device=DEVICE))

def encode_real_image(image_url: str) -> torch.Tensor:
    """Codifica imagem real da Wikipedia para HDV 10000D."""
    if GROUNDING is None or VISUAL_PIPELINE is None:
        return None
    try:
        req = urllib.request.Request(image_url, headers={"User-Agent": "AGI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = r.read()
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(io.BytesIO(data)).convert("RGB")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = GROUNDING.feature_extractor(tensor).flatten()
        hdv_mf, presign, f = VISUAL_PIPELINE.encode_frame(features)
        return hdv_mf.data.flatten()
    except Exception as e:
        return None

# ─── FONTES ───
def wikipedia_full(topic: str, lang: str = "en") -> dict:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ','_'))}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AGI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read())
            thumb = (d.get("thumbnail") or {}).get("source", "")
            return {"title": d.get("title", topic), "extract": d.get("extract", "")[:3000],
                    "image_url": thumb, "source": f"Wikipedia ({lang})"}
    except:
        return {"title": topic, "extract": f"{topic} is a concept.", "image_url": "", "source": "Wikipedia"}

CONCEPTS = [
    "Gravity","Energy","Force","Mass","Velocity","Acceleration","Inertia","Momentum",
    "Friction","Entropy","Quantum mechanics","Electromagnetism","Thermodynamics","Relativity",
    "Wave","Particle","Field","Potential","Kinetic energy","Nuclear fusion",
    "Evolution","DNA","Cell","Photosynthesis","Ecosystem","Natural selection","Protein",
    "Genetics","Neuron","Mitosis","Virus","Bacteria","Genome",
    "Chemical bond","Molecule","Atom","Periodic table","Reaction","Acid","Base",
    "Catalysis","Oxidation","Crystal","Polymer","Isotope",
    "Calculus","Probability","Statistics","Geometry","Algebra","Topology",
    "Set theory","Number theory","Matrix","Vector","Differential equation",
    "Algorithm","Data structure","Neural network","Cryptography","Information theory",
    "Graph theory","Compiler","Database","Operating system","Programming language",
    "Star","Planet","Galaxy","Black hole","Supernova","Nebula","Solar system",
    "Orbit","Asteroid","Dark matter","Constellation","Telescope",
    "Language","Grammar","Syntax","Semantics","Phonology","Translation","Bilingualism",
    "Logic","Epistemology","Metaphysics","Ethics","Consciousness","Free will",
    "Determinism","Aesthetics","Paradox","Infinity",
    "Gravidade","Energia","Evolucao","Celula","Probabilidade","Algoritmo",
    "Red neural","Consciencia","Fotossintese","Entropia","Galaxia","Buraco negro",
]

def curiosity(cge, n=5, workspace=None):
    """Seleciona topicos baseado em curiosidade + workspace."""
    known = {k.lower() for k in cge.nodes}
    # Preferir topicos no workspace (estao "conscientes")
    if workspace:
        conscious = workspace.get_conscious(3)
        for entry in conscious:
            tag_lower = entry.content.lower()
            if tag_lower not in known and tag_lower in [c.lower() for c in CONCEPTS]:
                return [entry.content]
    candidates = [t for t in CONCEPTS if t.lower() not in known]
    if not candidates:
        candidates = random.sample(CONCEPTS, min(n, len(CONCEPTS)))
    random.shuffle(candidates)
    return candidates[:n]

def compute_errors(cge, novos: int, total_extracted: int, concepts: list,
                   arch_created: int) -> dict:
    """Computa sinais de erro reais baseados no aprendizado."""
    n_nodes = len(cge.nodes)
    novelty = 1.0 - (total_extracted - novos) / max(1, total_extracted)
    # Incerteza: quao conectado e o conhecimento
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    uncertainty = 1.0 - min(1.0, len(cge.edges) / max(1, max_edges))
    # Erro de predicao: quantos conceitos ja conhecidos vs esperados
    overlap = total_extracted - novos
    prediction_error = 1.0 - overlap / max(1, total_extracted) if total_extracted > 0 else 0.5
    # Violacao: se temos conceitos com baixa consistencia (heuristica simples)
    violation = random.uniform(0, 0.1)  # baixo por padrao, pode crescer
    # Saliência: quantas conexoes potenciais este topico tem
    salience = min(1.0, novos / max(1, 5))
    return {
        "prediction_error": prediction_error,
        "novelty": novelty,
        "salience": salience,
        "uncertainty": uncertainty,
        "violation": violation,
    }

def compute_reward(mode: str, novos: int, errors: int, arch: int,
                   total_extracted: int, error_signals: dict) -> float:
    """Recompensa para o MetaController baseada no desempenho do modo."""
    if mode == "explore":
        return novos / max(1, total_extracted) if total_extracted > 0 else 0.0
    elif mode == "hypothesize":
        return errors / max(1, total_extracted)
    elif mode == "consolidate":
        return arch / max(1, 3)
    elif mode == "reflect":
        return 1.0 - error_signals.get("uncertainty", 0.5)
    elif mode == "exploit":
        return novos / max(1, total_extracted) * 0.5 + 0.3
    else:
        return novos / max(1, total_extracted + 1)

def learn(topic: str, mode: str) -> dict:
    """Aprende um topico. Retorna metricas do aprendizado."""
    src = wikipedia_full(topic, "en")
    extract = src.get("extract", "")
    img_url = src.get("image_url", "")
    reliability = 0.95

    names = all_names(topic)
    title_display = f"{topic} ({' / '.join(names[:3])})"

    words = extract.lower().split()[:400]
    concepts_set = set()
    for w in words:
        w = w.strip(".,;:!?()[]{}'\"")
        if len(w) > 4 and w.isalpha() and w not in ("from","with","that","this","have","been","which","their","there","where"):
            concepts_set.add(w)
    concepts = list(concepts_set)[:20]
    if len(concepts) < 3:
        concepts = [topic.lower(), "concept", "learning"] + concepts

    # Atualizar bridge com fonte
    bridge._lock.acquire()
    bridge.state["lifestream"] = {
        "current_media_type": "text" if not img_url else "image",
        "content_title": title_display,
        "content_body": f"Source: {src.get('source','?')}\n\n{extract[:1500]}",
        "source": src.get("source", "?"),
        "reliability": reliability,
        "curiosity_targets": curiosity(cge, 5),
        "current_concept": topic,
        "image_url": img_url,
    }
    bridge._lock.release()

    # Criar NOS no CGE com HDV perceptual real (se imagem disponivel) ou hash
    real_hdv = encode_real_image(img_url) if img_url else None
    novos = 0
    for c in concepts:
        if c not in cge.nodes:
            vec = real_hdv if (real_hdv is not None and c == topic.lower()) else make_hdv(c)
            cge.create_node(c, vec)
            if c in cge.nodes:
                cge.nodes[c]["emotional_valence"] = 0.3
                cge.nodes[c]["emotional_arousal"] = 0.5
                cge.nodes[c]["image_url"] = img_url
                if real_hdv is not None and c == topic.lower():
                    cge.nodes[c]["hdv_pictorico"] = real_hdv.cpu().tolist()
                novos += 1

    # Arestas
    edges_created = 0
    for i in range(min(len(concepts)-1, 10)):
        if concepts[i] in cge.nodes and concepts[i+1] in cge.nodes:
            cge.add_edge(concepts[i], concepts[i+1], cge.EDGE_CAUSES)
            edges_created += 1

    # HDV para frontend
    hdv_full = []
    for c in concepts[:3]:
        if c in cge.nodes:
            vec = cge.nodes[c].get("vector")
            if vec is not None:
                try:
                    flat = vec.flatten().tolist()
                    bits = [0 if x < 0 else 1 for x in flat]
                    hdv_full.append({"concept": c, "vector": bits, "n_dims": len(bits),
                                     "image_url": cge.nodes[c].get("image_url", "")})
                except:
                    pass

    # Archetypes
    arch = []
    for i in range(0, len(concepts), 4):
        chunk = concepts[i:i+4]
        if len(chunk) >= 2:
            arch.append({"name": f"{topic[:10]}_{i//4}",
                         "confidence": round(0.3+len(chunk)*0.12,2),
                         "strength": round(0.2+novos*0.05,2), "n_instances": novos})

    bridge._lock.acquire()
    bridge.state["hdv_vectors"] = hdv_full
    bridge.state["archetypes"] = arch[:6]
    bridge.state["simulation"] = {"branches": [
        {"id": "expect", "prob": round(0.3+novos*0.03,2), "desc": f"Futuro: {topic}"},
        {"id": f"cf_{topic[:4]}", "prob": round(0.2+random.random()*0.15,2), "desc": f"E se {topic}?"},
        {"id": "next", "prob": round(0.2+random.random()*0.1,2), "desc": f"Proximo: {curiosity(cge,3)[0] if curiosity(cge,3) else '?'}"}]}
    bridge._lock.release()

    bridge.set_from_cge(cge)
    bridge.add_concept_history(len(cge.nodes), len(cge.nodes), len(cge.edges))
    sm.learn_from_experience(topic, extract[:300], reliability=reliability,
                             domain="general", modality="text", success=novos > 0)
    bridge.set_from_self_model(sm)

    # Consolidacao se o MetaController escolheu CONSOLIDATE
    if mode == "consolidate" and len(cge.nodes) > 0:
        try:
            cge.consolidation_cycle()
            bridge.add_thought("[CONSOLIDACAO] Forja + CGE", "sleep")
        except:
            pass

    time.sleep(0.6)

    return {
        "topic": topic,
        "concepts": concepts,
        "novos": novos,
        "edges_created": edges_created,
        "arch_created": len(arch),
        "total_extracted": len(concepts),
        "img_url": img_url,
    }

def select_topic(cge, workspace, mode: str) -> str:
    """Seleciona topico baseado no modo cognitivo e estado interno."""
    conscious = workspace.get_conscious(3)
    # Se workspace tem algo saliente, usar isso
    if conscious and mode == "hypothesize":
        for entry in conscious:
            if entry.content in CONCEPTS:
                return entry.content
    # Caso contrario, curiosidade
    targets = curiosity(cge, 5, workspace)
    if targets:
        choice = targets[0]
        if mode == "explore":
            choice = targets[0]
        elif mode == "exploit":
            known = [t for t in CONCEPTS if t.lower() in {k.lower() for k in cge.nodes}]
            if known:
                choice = random.choice(known)
        elif mode == "reflect":
            choice = targets[-1] if len(targets) > 1 else targets[0]
        return choice
    return random.choice(CONCEPTS)

# ─── SERVER ───
FRONTEND_FILE = os.path.join(_BASE, "agiview_live.html")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            try:
                with open(FRONTEND_FILE, "r", encoding="utf-8") as f:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(f.read().encode())
            except:
                self.send_error(404)
        elif p == "/data":
            try:
                s = bridge.get_state()
                # Incluir novos sinais
                s["meta"] = meta.get_state() if meta else {}
                s["workspace"] = workspace.get_state() if workspace else {}
                s["error_signals"] = error_tracker.get_state() if error_tracker else {}
                # Ultimo trace
                if trace_rec and trace_rec.trace_history:
                    s["last_trace"] = trace_rec.trace_history[-1].to_dict()
                payload = json.dumps(s, default=str)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload.encode())
            except Exception as e:
                self.send_error(500, str(e))
        elif p == "/save":
            save(cge.nodes, cge.edges)
            self.send_response(200);
            self.end_headers();
            self.wfile.write(b"ok")
        elif p.startswith("/render"):
            self._serve_render(p)
        elif p.startswith("/scene"):
            self._serve_scene(p)
        elif p == "/traces":
            traces = CognitiveTrace.mine_from_cge(cge)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(traces, default=str).encode())
        else:
            self.send_error(404)

    def _serve_render(self, path: str):
        qs = parse_qs(urlparse(path).query)
        concept = (qs.get("concept") or ["unknown"])[0]
        fmt = (qs.get("format") or ["png"])[0]
        mode_param = (qs.get("mode") or ["concept"])[0]

        # Construir estado cognitivo para o renderer
        hdvs_list = []
        if concept in cge.nodes:
            vec = cge.nodes[concept].get("vector")
            if vec is not None:
                hdvs_list.append(vec)

        cognitive_state = {
            "focus": concept,
            "hdvs": hdvs_list,
            "mode": meta.mode if meta else "unknown",
            "attention": workspace.get_conscious(3) if workspace else [],
            "memories": [],
        }

        if fmt == "wav":
            audio = renderer.render(cognitive_state, "audio")
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        elif fmt == "json":
            # Retorna scene graph como JSON
            scene = MentalSceneGraph.from_cge(cge, concept)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(scene.to_dict(), default=str).encode())
        else:
            # PNG: render do estado cognitivo
            png = renderer.render(cognitive_state, "image")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

    def _serve_scene(self, path: str):
        """Retorna MentalSceneGraph como JSON."""
        qs = parse_qs(urlparse(path).query)
        concept = (qs.get("concept") or ["unknown"])[0]
        temporal = qs.get("temporal", ["0"])[0] == "1"
        n_frames = int(qs.get("frames", ["8"])[0])

        scene = MentalSceneGraph.from_cge(cge, concept)
        if temporal:
            frames = scene.to_temporal(n_frames)
            data = [f.to_dict() for f in frames]
        else:
            data = scene.to_dict()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, *a):
        pass

# ─── MAIN ───
if __name__ == "__main__":
    PORT = 8765
    print(f"\nAGI CEREBRO COMPLETO — {DEVICE} | Loop Emergente | Multimodal")
    print(f"  CGE: {len(cge.nodes)} nos, {len(cge.edges)} arestas")
    print(f"  MetaController: Q-learning em {len(meta._q_table)} contextos")
    print(f"  SemanticWorkspace: {workspace.capacity} slots")
    print(f"  http://127.0.0.1:{PORT}\n")

    t = threading.Thread(target=lambda: HTTPServer(("0.0.0.0", PORT), Handler).serve_forever(), daemon=False)
    t.start()
    time.sleep(1)

    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{PORT}")

    bridge.update(mode="explore", operator="curiosity")
    bridge.add_thought(f"Cerebro ativo GPU={DEVICE} — aprendizado emergente", "system")

    # Iniciar BackgroundIncubator (processamento inconsciente)
    if INCUBATOR:
        INCUBATOR.start()
        bridge.add_thought("[INCUBATOR] Processamento de fundo ativo", "system")

    # Notebook: pagina inicial
    notebook.new_page("Aprendizado Continuo",
                      "Explorar conceitos da Wikipedia e construir modelo de mundo")

    start_time = time.time()
    ciclo = 0

    while True:
        ciclo += 1

        # 1. Selecionar topico baseado no modo + workspace
        topic = select_topic(cge, workspace, meta.mode)

        # 2. Iniciar trace cognitivo
        trace_rec.start(topic)

        # 3. Aprender
        result = learn(topic, meta.mode)

        # 4. Computar sinais de erro REAIS
        error_signals = compute_errors(
            cge, result["novos"], result["total_extracted"],
            result["concepts"], result["arch_created"])
        error_tracker.create_error(topic, **error_signals)

        # 5. Atualizar bridge com sinais de erro (frontend espera d.error)
        bridge._lock.acquire()
        bridge.state["error"] = {
            "prediction_error": round(error_signals["prediction_error"], 2),
            "novelty": round(error_signals["novelty"], 2),
            "uncertainty": round(error_signals["uncertainty"], 2),
            "violation": round(error_signals["violation"], 2),
            "salience": round(error_signals["salience"], 2),
        }
        bridge._lock.release()

        # 7. Postar no SemanticWorkspace
        workspace.post(
            content=topic,
            source="perception",
            weight=0.8,
            confidence=0.5 + result["novos"] / max(1, result["total_extracted"]) * 0.3,
            novelty=error_signals["novelty"],
            urgency=0.3 + error_signals["salience"] * 0.5,
            tags=["wikipedia", result.get("img_url", "") and "has_image" or "text"]
        )
        workspace.tick(dt=0.3)

        # 6. Recompensa para o MetaController
        reward = compute_reward(
            meta.mode, result["novos"], result["edges_created"],
            result["arch_created"], result["total_extracted"], error_signals)

        # 8. MetaController decide PROXIMO modo
        next_mode = meta.tick(reward_signal=reward)

        # 8b. Stances: selecionar e aplicar stance para o contexto/modo atual
        context_key = meta._get_context_key() if hasattr(meta, '_get_context_key') else ""
        best_stance = stances.select_for_context(context_key, next_mode)
        if best_stance and best_stance.id != stances.active_stance_id:
            changes = stances.apply_stance(best_stance.id, meta)
            if changes:
                bridge.add_thought(
                    f"[STANCE] {best_stance.name}: {'; '.join(changes[:2])}",
                    "stance")

        # 9. Registrar no trace
        trace_rec.step(
            mode=meta.mode,
            context=error_tracker.get_state().get("most_pressing", ""),
            concepts=result["concepts"][:5],
            novelty=result["novos"],
            edges=result["edges_created"],
            outcome=f"aprendeu {result['novos']} conceitos",
            reward=reward,
            error=error_signals,
            workspace=[e.content for e in workspace.get_conscious(5)],
        )

        # 10. Construir MentalSceneGraph do estado cognitivo
        scene = MentalSceneGraph.from_cge(cge, topic)
        scene_merged = MentalSceneGraph.merge([
            scene,
            MentalSceneGraph.from_error(error_signals, "errors"),
            MentalSceneGraph.from_simulation(
                bridge.state.get("simulation", {}).get("branches", []), "sim"),
        ])

        # 11. Finalizar trace e salvar no CGE
        trace_rec.finish(answer=topic, success=result["novos"] > 0)

        # 11ab. FailedTrace: registrar se erro alto
        if error_signals["prediction_error"] > 0.6 or error_signals["novelty"] > 0.7:
            fail_db.record_from_tick(topic, next_mode, result["concepts"][:3], error_signals)
        else:
            fail_db.close_failed_trace(topic, "resolved")

        # 11ac. Archive: armazenar descobertas
        for alg in forja_evol.get_best_algorithms(3):
            archive.store(ArchiveItem(
                id=alg["id"], content_type="algorithm",
                name=alg["name"], data=alg,
                utility=alg.get("avg_reward", 0.5),
            ))
        # Arqueologia: combinar itens frios
        if ciclo % 15 == 0:
            cold_pairs = archive.archaeology(2)
            for a, b in cold_pairs:
                bridge.add_thought(
                    f"[ARQUEOLOGIA] Combinando {a.content_type} '{a.name[:20]}' "
                    f"+ {b.content_type} '{b.name[:20]}'", "archive")

        # 11b. Notebook: registrar aprendizado
        if result["novos"] > 0:
            notebook.write("experiment", f"Wikipedia: {topic}",
                          tags=[next_mode, "wikipedia"])
            notebook.add_hypothesis(f"{topic} se conecta com {', '.join(result['concepts'][:3])}")
            notebook.add_result(f"+{result['novos']} conceitos, {result['edges_created']} arestas",
                              confidence=reward)
        if error_signals["novelty"] > 0.6:
            notebook.add_insight(f"{topic} e um dominio novo com alta novidade")

        # 11c. ForjaEvolution: minera traces para algoritmos/estrategias/teorias
        if ciclo % 5 == 0:
            traces_data = CognitiveTrace.mine_from_cge(cge)
            # Tambem usar traces do recorder
            trace_dicts = [t.to_dict() for t in trace_rec.trace_history[-20:]]
            # InsightLibrary: minera padroes basicos
            discovered = insights.discover_from_traces(traces_data)
            for ins in discovered:
                bridge.add_thought(f"[NOVO INSIGHT] {ins.principle[:100]}", "insight")
            # ForjaEvolution: minera algoritmos, heuristicas, estrategias, teorias
            fr = forja_evol.mine_traces(trace_dicts + traces_data)
            if fr.get("algorithms", 0) > 0:
                best = forja_evol.get_best_algorithms(3)
                for a in best:
                    bridge.add_thought(f"[ALGORITMO] {a['name']} (r={a['avg_reward']:.2f})", "forja")
            if fr.get("heuristics", 0) > 0:
                best_h = forja_evol.get_best_heuristics(3)
                for h in best_h:
                    bridge.add_thought(f"[HEURISTICA] {h[:80]}", "forja")
            if fr.get("strategies", 0) > 0:
                bridge.add_thought(f"[ESTRATEGIA] +{fr['strategies']} novas", "forja")
            if fr.get("theories", 0) > 0:
                bridge.add_thought(f"[TEORIA] +{fr['theories']} novas", "forja")
            # ConstraintEngine: descobrir restricoes de traces
            new_constraints = constraints.discover_from_traces(trace_dicts + traces_data)
            for c in new_constraints:
                bridge.add_thought(f"[RESTRICAO] {c.description[:60]} (dom:{c.domain})", "constraint")
            # InsightGraph: descobrir principios de paradigmas
            new_principles = insight_graph.discover_from_paradigms(
                list(forja_evol.paradigms.values()),
                list(forja_evol.theories.values()))
            for p in new_principles:
                bridge.add_thought(f"[PRINCIPIO] {p.principle[:80]} ({p.n_domains} dominios)", "principle")
            # Cross-domain forge: fertilizacao cruzada entre paradigmas
            if len(forja_evol.paradigms) >= 2:
                p_list = list(forja_evol.paradigms.values())
                for i in range(0, len(p_list)-1, 2):
                    a, b = p_list[i], p_list[i+1]
                    bridge.add_thought(
                        f"[CRUZAMENTO] Paradigma '{a.name[:20] if hasattr(a, 'name') else a.id[:10]}' "
                        f"+ '{b.name[:20] if hasattr(b, 'name') else b.id[:10]}'",
                        "cross_domain")
            # Stances: descobrir novas de traces
            new_stances = stances.discover_from_traces(trace_dicts + traces_data)
            for s in new_stances:
                bridge.add_thought(f"[STANCE] Nova: {s.name} (conf={s.confidence:.2f})", "stance")
            # MetaForja: minerar regras de criacao do OperatorForge
            forge_log = getattr(forja_evol, 'discovery_log', [])
            new_rules = meta_forja.mine(trace_dicts + traces_data, forge_log)
            for r in new_rules:
                bridge.add_thought(f"[METAFORJA] Regra: {r.condition[:80]}", "metaforja")

        # 11d. ModelEcosystem: criar modelo do topico aprendido
        model = Model(
            id=f"model_{topic[:8]}_{ciclo}",
            description=f"Modelo para {topic}",
            confidence=0.3 + result["novos"] / max(1, result["total_extracted"]) * 0.4,
            explanatory_power=result["novos"] / max(1, result["total_extracted"]),
            parsimony=0.8,
            domain=topic[:20],
        )
        model_eco.add_model(model)
        # Tentar fusao com modelo similar existente
        for existing_id in list(model_eco.models.keys())[-10:]:
            if existing_id != model.id:
                child = model_eco.merge(existing_id, model.id,
                                       merge_name=f"merge_{topic[:8]}")
                if child:
                    bridge.add_thought(f"[MERCADO] Fusao: {existing_id[:8]}+{model.id[:8]}={child.id[:12]}", "model")

        # 11e. Atualizar bridge com novos estados
        bridge._lock.acquire()
        bridge.state["notebook"] = notebook.get_state()
        bridge.state["insights"] = insights.get_state()
        bridge.state["model_eco"] = model_eco.get_state()
        bridge.state["forja_evol"] = forja_evol.get_state()
        bridge.state["stances"] = stances.get_state()
        bridge.state["meta_forja"] = meta_forja.get_state()
        bridge.state["failed_traces"] = fail_db.get_state()
        bridge.state["archive"] = archive.get_state()
        bridge.state["meta_evol"] = meta_evol.get_state()
        bridge.state["value_system"] = {"priorities": values.get_priorities(5)}
        bridge.state["constraints"] = constraints.get_state()
        bridge.state["insight_graph"] = insight_graph.get_state()
        bridge._lock.release()

        # 11f. ValueSystem: registrar valor do topico aprendido
        values.record_outcome(topic, "wikipedia", reward, cost=0.3 + result.get("total_extracted", 5) * 0.02)

        # 11g. MetaEvolution: a cada ~60s propoe mutacao na arquitetura
        recent_rewards = [m.reward for m in trace_rec.trace_history[-5:]] if trace_rec.trace_history else [0]
        evol_msg = meta_evol.tick(ciclo, recent_rewards)
        if evol_msg:
            bridge.add_thought(f"[EVOLUTION] {evol_msg}", "evolution")

        # 12. Atualizar ponte para frontend
        bridge.update(
            cycle=ciclo,
            mode=next_mode,
            operator="curiosity",
            uptime=int(time.time() - start_time),
        )
        bridge.set_from_cge(cge)
        bridge.set_from_self_model(sm)

        conscious_entries = workspace.get_conscious(3)
        bridge.add_thought(
            f"[{next_mode.upper()}] {topic}: +{result['novos']} nos, "
            f"{result['edges_created']} edges | "
            f"reward={reward:.2f} err_PE={error_signals['prediction_error']:.2f} "
            f"err_N={error_signals['novelty']:.2f}",
            next_mode)

        # 13. Auto-save
        if ciclo % 5 == 0:
            save(cge.nodes, cge.edges)
            bridge.add_thought(f"[AUTO-SAVE] {len(cge.nodes)} nos", "system")
