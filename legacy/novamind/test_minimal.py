import sys, os, json, time
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

from core.world_model import WorldModel
from core.cognitive_executive import CognitiveExecutive
from core.cognitive_modes import CognitiveModeManager
from core.goal_driven_engine import GoalDrivenEngine
from core.unified_self import UnifiedSelf
from core.intrinsic_motivation import IntrinsicMotivationEngine
from core.knowledge_graph import KnowledgeGraph
from core.virtual_lab import VirtualLab

print("1. Loading KG...", flush=True)
kg = KnowledgeGraph()
data = json.load(open(r"D:\novamind_data\knowledge_graph\concepts.json", "r", encoding="utf-8"))
for name, meta in data.items():
    d = meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown"
    kg.add_concept(str(name)[:100], [], domain=d)
rels = json.load(open(r"D:\novamind_data\knowledge_graph\relations.json", "r", encoding="utf-8"))
for r in rels:
    kg.add_causal_relation(r["source"], r["target"], r.get("type", "causa"), confidence=r.get("confidence", 0.5))
print(f"  {len(kg.concepts)}c {len(kg.relations)}r", flush=True)

print("2. WM + VLab...", flush=True)
wm = WorldModel(knowledge_graph=kg)
vl = VirtualLab()
for d in ["Alzheimer", "Parkinson"]:
    vl.load_disease_model(d)
    r = vl.simulate_intervention("baseline", dose=0)
    print(f"  {d}: {len(r.biomarker_change)}", flush=True)

print("3. Executive...", flush=True)
exec = CognitiveExecutive(
    goal_engine=GoalDrivenEngine(),
    self_model=UnifiedSelf(),
    motivation=IntrinsicMotivationEngine(UnifiedSelf()),
    mode_manager=CognitiveModeManager(),
    world_model=wm,
    knowledge_graph=kg,
    virtual_lab=vl,
)
print(f"  {exec.summarize_self()}", flush=True)

print("4. Idle cycles...", flush=True)
for i in range(10):
    print(f"  Cycle {i} starting...", flush=True)
    if i == 3:
        wm.recent_surprise.append(0.9)
    elif i == 7:
        wm.recent_surprise.append(0.1)
    try:
        result = exec.idle_cycle()
        mode = result.get("mode", "?")
        print(f"    mode={mode} surpresa={wm.get_surprise():.2f}", flush=True)
    except Exception as e:
        print(f"    ERRO: {e}", flush=True)

print("5. Done!", flush=True)
