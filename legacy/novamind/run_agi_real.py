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
from core.neuro_field import NeuroField
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
from core.cognitive_dag import DAG_REGISTRY, CognitiveStep
from core.discovery_policies import PolicyLibrary, UnknownSpace
from core.worldview import WorldviewLibrary
from core.goal_emergence import GoalEmergence
from core.cross_domain_test import CrossDomainTest, DOMAIN_CLUSTERS
from core.cognitive_bench import CognitiveBench
from core.scene_decoder import SceneRepresentation, MentalDecoder as CoreDecoder
from core.cognitive_symbols import SymbolLibrary
from core.mental_canvas import MentalCanvas
from core.explainability_space import ExplainabilitySpace
from core.latent_workspace import LatentWorkspace
from core.cognitive_tension import CognitiveTension
from core.analogy_engine import AnalogyEngine
from core.compression_engine import CompressionEngine
from core.quote_memory import QuoteMemory
from core.audience_model import AudienceModel
from core.teaching_engine import TeachingEngine
from core.communication_memory import CommunicationMemory
from core.discourse_planner import DiscoursePlanner
from core.cognitive_ecology import CognitiveEcology, CognitiveBiasVector
from core.reasoning_laws import ReasoningLaws
from core.cognitive_budget import CognitiveBudget
from core.multi_confidence import MultiConfidence, ConfidenceTracker
from core.active_experiment import ActiveExperimentDesigner
from core.inventor import Inventor
from core.structural_analogy import StructuralAnalogyEngine, Subgraph
from core.analogy_engine import AnalogyEngine
from core.elegance import EleganceEvaluator
from core.gap_graph import GapGraph
from core.skill_acquisition import SkillAcquisition
from core.forge_validator import ForgeValidator
from core.extreme_compression import ExtremeCompression

# Renderizadores → Decodificadores
from core.mental_renderer import MentalDecoder

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
meta = MetaController(error_tracker=error_tracker, cge=cge, neurofield=NeuroField())
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
policies = PolicyLibrary(cge=cge)
unknown = UnknownSpace(cge=cge)
worldviews = WorldviewLibrary(cge=cge)
goal_emergence = GoalEmergence(unknown_space=unknown, cge=cge)
cross_test = CrossDomainTest(cge=cge, trace_rec=trace_rec)
bench = CognitiveBench(cross_test=cross_test)
symbols = SymbolLibrary(cge=cge)
canvas = MentalCanvas(cge=cge)
explain = ExplainabilitySpace(cge=cge, insight_library=insights)
latent = LatentWorkspace(capacity=30)
tension = CognitiveTension(cge=cge)
analogy = AnalogyEngine(cge=cge, insight_library=insights)
compression = CompressionEngine(cge=cge, loop_evaluator=loop_eval)
quotes = QuoteMemory(cge=cge)
audience = AudienceModel()
teacher = TeachingEngine()
comm_mem = CommunicationMemory()
discourse = DiscoursePlanner(cge=cge)
ecology = CognitiveEcology(cge=cge, worldviews=worldviews, stances=stances)
reasoning_laws = ReasoningLaws(insight_library=insights, cge=cge)
budget = CognitiveBudget()
confidence_tracker = ConfidenceTracker()
experimenter = ActiveExperimentDesigner()
inventor = Inventor(cge=cge, unknown=unknown, explainability=explain, tension=tension)
struct_analogy = StructuralAnalogyEngine(cge=cge)
elegance = EleganceEvaluator()
gaps = GapGraph(cge=cge)
skills = SkillAcquisition(cge=cge)
extreme = ExtremeCompression(cge=cge, insight_graph=insight_graph, elegance=elegance)
forge_val = ForgeValidator(cge=cge, neurofield=meta.neurofield)

# Decodificador mental (3 decoders em um)
renderer = MentalDecoder(visual_pipeline=VISUAL_PIPELINE, vocoder=AUDIO_VOCODER)

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
        return hdv_mf.tensor.flatten()
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
        mode_param = (qs.get("mode") or ["reconstruction"])[0]
        decoder_mode = "reconstruction"
        if mode_param in ("imagination", "communication"):
            decoder_mode = mode_param

        cognitive_state = {
            "focus": concept,
            "mode": meta.mode if meta else "unknown",
            "attention": workspace.get_conscious(3) if workspace else [],
            "_cge": cge,
            "_simulation": bridge.state.get("simulation", {}).get("branches", []),
            "_goal": qs.get("goal", [""])[0],
        }

        if fmt == "wav":
            audio = renderer.reconstruct(cognitive_state, "communication")
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        elif fmt == "json":
            scene = MentalSceneGraph.from_cge(cge, concept)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(scene.to_dict(), default=str).encode())
        else:
            png = renderer.reconstruct(cognitive_state, decoder_mode)
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

    # Auto-seed: CognitiveEcology se inicializa a partir de Worldviews
    if not ecology.models:
        seeded = ecology.seed_from_worldviews()
        bridge.add_thought(f"[ECOLOGY] Auto-seed: {len(seeded)} modelos de {len(worldviews.worldviews)} worldviews", "ecology")

    # Iniciar BackgroundIncubator (processamento inconsciente)
    if INCUBATOR:
        INCUBATOR.start()
        bridge.add_thought("[INCUBATOR] Processamento de fundo ativo", "system")

    # Notebook: pagina inicial
    notebook.new_page("Aprendizado Continuo",
                      "Explorar conceitos da Wikipedia e construir modelo de mundo")

    start_time = time.time()
    ciclo = 0


    # ─── DAG EXECUTOR ───────────────────────────────────────
    # O ciclo cognitivo e um DAG. A ordem nao e fixa.
    # A ForjaEvolution pode criar novos DAGs com ordens diferentes.

    _CTX = {}

    def _step_select_topic():
        _CTX["topic"] = select_topic(cge, workspace, meta.mode)
        trace_rec.start(_CTX["topic"])
        unknown.scan()

    def _step_learn():
        _CTX["result"] = learn(_CTX["topic"], meta.mode)

    def _step_compute_errors():
        r = _CTX["result"]
        _CTX["error_signals"] = compute_errors(cge, r["novos"], r["total_extracted"], r["concepts"], r["arch_created"])
        error_tracker.create_error(_CTX["topic"], **_CTX["error_signals"])
        bridge._lock.acquire()
        bridge.state["error"] = {k: round(v, 2) for k, v in _CTX["error_signals"].items()}
        bridge._lock.release()

    def _step_post_workspace():
        es = _CTX["error_signals"]; r = _CTX["result"]
        workspace.post(content=_CTX["topic"], source="perception", weight=0.8,
            confidence=0.5 + r["novos"] / max(1, r["total_extracted"]) * 0.3,
            novelty=es["novelty"], urgency=0.3 + es["salience"] * 0.5,
            tags=["wikipedia", r.get("img_url", "") and "has_image" or "text"])
        workspace.tick(dt=0.3)

    def _step_meta_reward():
        r = _CTX["result"]; es = _CTX["error_signals"]
        _CTX["reward"] = compute_reward(meta.mode, r["novos"], r["edges_created"], r["arch_created"], r["total_extracted"], es)
        # NeuroField: atualiza com sinais de erro continuos (sem if/else de modo)
        if meta.neurofield and es:
            meta.neurofield.update(
                prediction_error=es.get("prediction_error", 0),
                novelty=es.get("novelty", 0),
                uncertainty=es.get("uncertainty", 0.5),
                violation=es.get("violation", 0),
                fatigue=0.3 + len(_CTX.get("result", {}).get("concepts", [])) / 30,
                reward=_CTX["reward"] if _CTX.get("reward") else 0,
            )

    def _step_meta_tick():
        _CTX["next_mode"] = meta.tick(reward_signal=_CTX["reward"])
        # Flutuacao: perturba neurofield para evitar cristalizacao
        if meta.neurofield:
            meta.neurofield.perturb_state()
        # Energy Landscape: se neurofield ativo, computa paisagem
        if meta.neurofield and cge:
            temp = meta.neurofield.gating_temperature
            # Pega error_field do error_tracker se disponivel
            error_field = {}
            try:
                for sid, sig in error_tracker.errors.items():
                    if hasattr(sig, 'prediction_error'):
                        error_field[sid] = sig.prediction_error
            except:
                pass
            landscape = cge.compute_energy_landscape(error_field)
            _CTX["energy_landscape"] = landscape
            # Propaga ativacao pela paisagem
            if _CTX.get("topic") and landscape:
                cge.activate_with_landscape([_CTX["topic"]], landscape, temperature=temp)
            # Micro-mutacao: perturba CGE para evitar cristalizacao
            noise = meta.neurofield.fluctuation_noise if meta.neurofield else 0.0
            if noise > 0.01:
                cge.apply_micro_mutations(noise)

    def _step_stance():
        nxt = _CTX["next_mode"]
        best = stances.select_for_context("", nxt)
        if best and best.id != stances.active_stance_id:
            ch = stances.apply_stance(best.id, meta)
            if ch: bridge.add_thought("[STANCE] " + best.name + ": " + "; ".join(ch[:2]), "stance")
        # Worldview: selecionar para o contexto
        wv = worldviews.select_for_context(nxt)
        if wv:
            wv_ctx = _CTX.get("topic", "general")
            bridge.add_thought(f"[WORLDVIEW] {wv.name} ativa para {wv_ctx}", "worldview")

    def _step_record_trace():
        r = _CTX["result"]; es = _CTX["error_signals"]; reward = _CTX["reward"]
        trace_rec.step(mode=meta.mode, context=error_tracker.get_state().get("most_pressing", ""),
            concepts=r["concepts"][:5], novelty=r["novos"], edges=r["edges_created"],
            outcome=f"aprendeu {r['novos']} conceitos", reward=reward, error=es,
            workspace=[e.content for e in workspace.get_conscious(5)])

    def _step_build_scene():
        es = _CTX["error_signals"]
        scene = MentalSceneGraph.from_cge(cge, _CTX["topic"])
        _CTX["scene_merged"] = MentalSceneGraph.merge([scene,
            MentalSceneGraph.from_error(es, "errors"),
            MentalSceneGraph.from_simulation(bridge.state.get("simulation", {}).get("branches", []), "sim")])

    def _step_notebook():
        r = _CTX["result"]; es = _CTX["error_signals"]; reward = _CTX["reward"]
        nxt = _CTX["next_mode"]; topic = _CTX["topic"]
        trace_rec.finish(answer=topic, success=r["novos"] > 0)
        if es["prediction_error"] > 0.6 or es["novelty"] > 0.7:
            fail_db.record_from_tick(topic, nxt, r["concepts"][:3], es)
        else: fail_db.close_failed_trace(topic, "resolved")
        for alg in forja_evol.get_best_algorithms(3):
            archive.store(ArchiveItem(id=alg["id"], content_type="algorithm", name=alg["name"], data=alg, utility=alg.get("avg_reward", 0.5)))
        if ciclo % 15 == 0:
            for a, b in archive.archaeology(2):
                bridge.add_thought(f"[ARQUEOLOGIA] {a.content_type} '{a.name[:20]}' + {b.content_type} '{b.name[:20]}'", "archive")
        if r["novos"] > 0:
            notebook.write("experiment", f"Wikipedia: {topic}", tags=[nxt, "wikipedia"])
            notebook.add_hypothesis(f"{topic} se conecta com {", ".join(r['concepts'][:3])}")
            notebook.add_result(f"+{r['novos']} conceitos, {r['edges_created']} arestas", confidence=reward)
        if es["novelty"] > 0.6: notebook.add_insight(f"{topic} e um dominio novo com alta novidade")

    def _step_forja_mine():
        if ciclo % 5 != 0: return
        traces_data = CognitiveTrace.mine_from_cge(cge)
        trace_dicts = [t.to_dict() for t in trace_rec.trace_history[-20:]]
        for ins in insights.discover_from_traces(traces_data):
            bridge.add_thought(f"[INSIGHT] {ins.principle[:100]}", "insight")
        fr = forja_evol.mine_traces(trace_dicts + traces_data)
        for label, key in [("ALGORITMO","algorithms"),("ESTRATEGIA","strategies"),("TEORIA","theories")]:
            if fr.get(key, 0) > 0: bridge.add_thought(f"[{label}] +{fr[key]} novas", "forja")
        if fr.get("algorithms", 0) > 0:
            for a in forja_evol.get_best_algorithms(3):
                bridge.add_thought(f"[ALGORITMO] {a['name']} (r={a['avg_reward']:.2f})", "forja")
                # ForgeValidator: testa cada novo algoritmo
                v = forge_val.validate(a["name"], a)
                if v.approved:
                    bridge.add_thought(f"[VALIDACAO] Aprovado: {a['name']} ({v.reason})", "validate")
                else:
                    bridge.add_thought(f"[VALIDACAO] Rejeitado: {a['name']} ({v.reason})", "validate")
        if fr.get("heuristics", 0) > 0:
            for h in forja_evol.get_best_heuristics(3): bridge.add_thought(f"[HEURISTICA] {h[:80]}", "forja")
        for c in constraints.discover_from_traces(trace_dicts + traces_data):
            bridge.add_thought(f"[RESTRICAO] {c.description[:60]} (dom:{c.domain})", "constraint")
        for p in insight_graph.discover_from_paradigms(list(forja_evol.paradigms.values()), list(forja_evol.theories.values())):
            bridge.add_thought(f"[PRINCIPIO] {p.principle[:80]} ({p.n_domains} dominios)", "principle")
        pl = list(forja_evol.paradigms.values())
        for i in range(0, len(pl)-1, 2):
            bridge.add_thought(f"[CRUZAMENTO] {pl[i].name[:20] if hasattr(pl[i],'name') else pl[i].id[:10]} + {pl[i+1].name[:20]}", "cross_domain")
        for s in stances.discover_from_traces(trace_dicts + traces_data):
            bridge.add_thought(f"[STANCE] Nova: {s.name} (conf={s.confidence:.2f})", "stance")
        for r2 in meta_forja.mine(trace_dicts + traces_data, getattr(forja_evol, 'discovery_log', [])):
            bridge.add_thought(f"[METAFORJA] Regra: {r2.condition[:80]}", "metaforja")
        for p in policies.discover_from_traces(trace_dicts + traces_data):
            bridge.add_thought(f"[POLICY] {p.principle[:80]}", "policy")
        # Worldviews: descobrir de paradigmas
        for wv in worldviews.discover_from_paradigms(list(forja_evol.paradigms.values())):
            bridge.add_thought(f"[WORLDVIEW] Nova: {wv.name}", "worldview")
        # Goal Emergence: metas do UnknownSpace
        new_goals = goal_emergence.tick()
        for g in new_goals:
            bridge.add_thought(f"[META] {g.description} (potencial={g.potential:.2f})", "goal")
        # Symbols: descobrir macros de traces
        new_syms = symbols.discover_macros(trace_dicts + traces_data)
        for s in new_syms:
            seq = " → ".join(s.sequence[:4])
            bridge.add_thought(f"[SIMBOLO] {s.symbol} = {seq} (compressao={s.compression_ratio:.0f}x)", "symbol")
        # ExplainabilitySpace: escanear gaps
        explain.scan()
        open_gaps = explain.get_open_gaps(3)
        for g in open_gaps:
            bridge.add_thought(f"[EXPLICABILIDADE] Gap: {g.phenomenon[:40]} (gap={g.gap_size:.2f})", "explain")
        # CognitiveTension: detectar do CGE
        tension.detect_from_cge()
        press = tension.get_pressing_tensions(2)
        for t in press:
            bridge.add_thought(f"[TENSAO] {t.description[:60]} (forca={t.strength:.2f})", "tension")
        # AnalogyEngine: escanear isomorfismos
        analogy.scan()
        for a in analogy.get_best(2):
            bridge.add_thought(f"[ANALOGIA] {a.source_a}:{a.source_b} :: {a.target_c}:{a.target_d} (sim={a.similarity:.2f})", "analogy")
        # CompressionEngine: comprimir traces em atomos
        new_atoms = compression.mine_from_traces(trace_dicts + traces_data)
        for a in new_atoms:
            bridge.add_thought(f"[COMPRESSAO] {a.name} = {' → '.join(a.original_sequence[:4])} (ratio={a.compression_ratio:.0f}x)", "compression")
        # CognitiveEcology: tick do ecossistema (a cada 5 ciclos)
        if ciclo % 5 == 0:
            ecology.tick()
            new_laws = reasoning_laws.extract_laws()
            for law in new_laws:
                if law is not None:
                    bridge.add_thought(f"[LEI] {law.statement[:60]} (conf={law.confidence:.2f})", "law")
            health = ecology.get_population_health()
            if ciclo % 20 == 0 and health < 0.3:
                bridge.add_thought(f"[ECOLOGY] Saude baixa ({health:.2f}) — auto-cura acionada", "ecology")
            # StructuralAnalogy: isomorfismos profundos a cada 10 ciclos
            if ciclo % 10 == 0:
                analogies = struct_analogy.scan(3)
                for a in analogies:
                    bridge.add_thought(f"[ISO] {a.source_subgraph.root} :: {a.target_subgraph.root} (qualidade={a.isomorphism_quality:.2f}, {len(a.mapping)} mapeamentos)", "analogy")
            # Inventor: gera perguntas novas a cada 7 ciclos
            if ciclo % 7 == 0:
                q = inventor.generate_question()
                if q:
                    inventor.store_question(q)
                    bridge.add_thought(f"[INVENTOR] Pergunta: {q.get('question', '')[:60]}", "inventor")
            # ExperimentDesigner: hipotese do topico atual
            if topic:
                experimenter.add_hypothesis(f"{topic} explica isto", domain="general")
                exp = experimenter.design_experiment(f"hyp_{hashlib.md5(f'{topic} explica isto'.encode()).hexdigest()[:8]}")
                if exp and exp.priority > 0.5:
                    bridge.add_thought(f"[EXPERIMENTO] {exp.description[:50]} (pri={exp.priority:.2f})", "experiment")
            # Elegance: avalia modelos da ecologia
            eco_models = [m for m in ecology.models.values() if m.status == "active"][:3]
            if eco_models:
                candidates = []
                for m in eco_models:
                    candidates.append({
                        "id": m.id, "n_axioms": 3, "n_explanations": m.total_predictions,
                        "n_domains": 1 if m.domain == "general" else 2,
                        "n_contradictions": m.n_incorrect,
                        "reuses": m.n_applications if hasattr(m, 'n_applications') else m.n_predictions,
                    })
                ranked = elegance.rank_by_elegance(candidates)
                if ranked:
                    top = ranked[0]
                    bridge.add_thought(f"[ELEGANCA] {top['c']['id'][:8]} score={top['score']}", "elegance")
            # GapGraph: adiciona lacunas dos diversos detectores
            if ciclo % 5 == 0:
                # De tension
                for t in tension.get_pressing_tensions(2):
                    gaps.add("tension", t.description, domain=t.domain,
                            concepts=t.concepts_involved, source="tension")
                # De explain
                for g in explain.get_open_gaps(2):
                    gaps.add("unexplained", g.phenomenon, domain=g.domain,
                            source="explainability")
                # De unknown
                if unknown:
                    for r in unknown.get_frontier(2):
                        gaps.add("unknown",
                                f"Vazio entre {r.get('concept_a','?')} e {r.get('concept_b','?')}",
                                concepts=[r.get('concept_a',''), r.get('concept_b','')],
                                source="unknown_space")
            # SkillAcquisition: minera traces
            new_skills = skills.mine_from_traces(trace_dicts + traces_data)
            for s in new_skills:
                bridge.add_thought(f"[SKILL] {s.name}: {' → '.join(s.sequence[:4])} ({s.n_executions} exec)", "skill")
            # ExtremeCompression: escaneia e aceita as melhores
            if ciclo % 10 == 0:
                extreme.scan()
                for c in extreme.get_best(min_ratio=2.0, top_k=2):
                    extreme.accept(c.id, ciclo)
                    bridge.add_thought(f"[COMPRESSAO EXTREMA] {c.description[:60]} (ratio={c.compression_ratio:.0f}x, elegancia={c.elegance_score:.2f})", "compress")
        # Communication pipeline (a cada ciclo, prepara explicacao do topico atual)
        topic = _CTX.get("topic", "")
        if topic:
            prof = audience.identify({"vocabulary": 0.5, "math": 0.3, "format": "text"})
            strategy = teacher.recommend(topic, prof)
            quote = quotes.quote(topic.lower())
            plan = discourse.plan(topic, prof, strategy.format,
                                  [quote.text] if quote else None)
            bridge.add_thought(f"[DIDATICA] Explicar {topic} para {prof} via {strategy.format}", "teach")

    def _step_archive():
        r = _CTX["result"]; reward = _CTX["reward"]; nxt = _CTX["next_mode"]; topic = _CTX["topic"]
        model = Model(id=f"model_{topic[:8]}_{ciclo}", description=f"Modelo para {topic}",
            confidence=0.3 + r["novos"]/max(1,r["total_extracted"])*0.4,
            explanatory_power=r["novos"]/max(1,r["total_extracted"]), parsimony=0.8, domain=topic[:20])
        model_eco.add_model(model)
        for eid in list(model_eco.models.keys())[-10:]:
            if eid != model.id:
                child = model_eco.merge(eid, model.id, merge_name=f"merge_{topic[:8]}")
                if child: bridge.add_thought(f"[MERCADO] Fusao: {eid[:8]}+{model.id[:8]}={child.id[:12]}", "model")

    def _step_bridge_update():
        r = _CTX["result"]; es = _CTX["error_signals"]; reward = _CTX["reward"]; nxt = _CTX["next_mode"]; topic = _CTX["topic"]
        bridge._lock.acquire()
        for k in ["notebook","insights","model_eco","forja_evol","stances","meta_forja","failed_traces","archive","meta_evol","constraints","insight_graph"]:
            bridge.state[k] = eval(k).get_state()
        bridge.state["value_system"] = {"priorities": values.get_priorities(5)}
        bridge.state["dag"] = DAG_REGISTRY.get_state()
        bridge.state["policies"] = policies.get_state()
        bridge.state["unknown_space"] = unknown.get_state()
        bridge.state["worldview"] = worldviews.get_state()
        bridge.state["goal_emergence"] = goal_emergence.get_state()
        bridge.state["symbols"] = symbols.get_state()
        bridge.state["canvas"] = canvas.get_state()
        bridge.state["explain"] = explain.get_state()
        bridge.state["latent"] = latent.get_state()
        bridge.state["tension"] = tension.get_state()
        bridge.state["analogy"] = analogy.get_state()
        bridge.state["compression"] = compression.get_state()
        bridge.state["quotes"] = quotes.get_state()
        bridge.state["audience"] = audience.get_state()
        bridge.state["teacher"] = teacher.get_state()
        bridge.state["comm_mem"] = comm_mem.get_state()
        bridge.state["discourse"] = discourse.get_state()
        bridge.state["ecology"] = ecology.get_state()
        bridge.state["reasoning_laws"] = reasoning_laws.get_state()
        bridge.state["budget"] = budget.get_state()
        bridge.state["confidence"] = confidence_tracker.get_state()
        bridge.state["experimenter"] = experimenter.get_state()
        bridge.state["inventor"] = inventor.get_state()
        bridge.state["struct_analogy"] = struct_analogy.get_state()
        bridge.state["elegance"] = elegance.get_state()
        bridge.state["gaps"] = gaps.get_state()
        bridge.state["skills"] = skills.get_state()
        bridge.state["extreme"] = extreme.get_state()
        bridge.state["forge_validator"] = forge_val.get_state()
        bridge.state["cross_domain"] = cross_test.get_state()
        bridge.state["bench"] = bench.generate_report().to_dict() if bench.snapshots else {}
        # Bench: snapshot periodico + relatorio
        report = bench.snapshot(ciclo, cge=cge, forja_evol=forja_evol, meta=meta,
            error_tracker=error_tracker, meta_evol=meta_evol, worldviews=worldviews,
            stances=stances, insights=insights, constraints=constraints,
            policies=policies, goals=goal_emergence, trace_rec=trace_rec,
            fail_db=fail_db)
        if report:
            print(bench.summary())
            bridge.add_thought("[BENCH] Relatorio cognitivo gerado", "bench")
        bridge._lock.release()
        values.record_outcome(topic, "wikipedia", reward)
        recent_r = [m.reward for m in trace_rec.trace_history[-5:]] if trace_rec.trace_history else [0]
        evol_msg = meta_evol.tick(ciclo, recent_r)
        if evol_msg: bridge.add_thought(f"[EVOLUTION] {evol_msg}", "evolution")
        bridge.update(cycle=ciclo, mode=nxt, operator="curiosity", uptime=int(time.time()-start_time))
        bridge.set_from_cge(cge); bridge.set_from_self_model(sm)
        bridge.add_thought(f"[{nxt.upper()}] {topic}: +{r['novos']} nos, {r['edges_created']} edges | reward={reward:.2f} err_PE={es['prediction_error']:.2f} err_N={es['novelty']:.2f}", nxt)

    def _step_auto_save():
        if ciclo % 5 == 0:
            save(cge.nodes, cge.edges)
            bridge.add_thought(f"[AUTO-SAVE] {len(cge.nodes)} nos", "system")

    _STEP_HANDLERS = {
        "select_topic": _step_select_topic, "learn": _step_learn,
        "compute_errors": _step_compute_errors, "post_workspace": _step_post_workspace,
        "meta_reward": _step_meta_reward, "meta_tick": _step_meta_tick,
        "stance": _step_stance, "record_trace": _step_record_trace,
        "build_scene": _step_build_scene, "notebook": _step_notebook,
        "forja_mine": _step_forja_mine, "archive": _step_archive,
        "bridge_update": _step_bridge_update, "auto_save": _step_auto_save,
        "start_trace": lambda: None, "tick_workspace": lambda: None, "idle": lambda: time.sleep(0.1),
    }

    while True:
        ciclo += 1
        dag = DAG_REGISTRY.get_active()
        # Gating Field: neurofield decide quantas portas abrir
        nf_state = meta.neurofield.get_state() if meta.neurofield else None
        steps = dag.apply_gating(nf_state) if nf_state else dag.get_topological_order()
        for step in steps:
            handler = _STEP_HANDLERS.get(step.id)
            if handler:
                try:
                    handler()
                except Exception as e:
                    bridge.add_thought(f"[DAG] Erro em '{step.name}': {e}", "error")
                    import traceback; traceback.print_exc()
                    if step.required: break
        if ciclo % 7 == 0: unknown.scan()
