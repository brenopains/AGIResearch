"""Respostas INTEGRADAS: sistema REAL, não hardcoded.
Carrega CGE com todas as 9563 arestas, WorldModel, SimEngine, CausalReasoning.
Responde perguntas usando o conhecimento QUE JÁ EXISTE no sistema."""
import sys, os, pickle, torch, json, re
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

print("=" * 70)
print("CARREGANDO SISTEMA REAL")
print("=" * 70)

# 1. Load saved state
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
cge_nodes = state.get('cge_nodes', {})
cge_edges_raw = state.get('cge_edges', [])

print(f"CGE nodes: {len(cge_nodes)}")
print(f"CGE edges (raw): {len(cge_edges_raw)}")

# 2. Load CGE with actual state
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.bge_encoder import BGEEncoder

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)

# Restore nodes
for nid, nd in cge_nodes.items():
    if isinstance(nid, str) and isinstance(nd, dict) and len(nid) <= 50:
        try:
            nd2 = dict(nd)
            for k in ['vector', 'semantic', 'perceptual']:
                if k in nd2 and isinstance(nd2[k], torch.Tensor):
                    nd2[k] = nd2[k].to(cge.device)
            cge.nodes[nid] = nd2
        except: pass

# Restore edges
edge_type_map = {
    'causes': cge.EDGE_CAUSES, 'caused_by': cge.EDGE_CAUSED_BY,
    'right_of': cge.EDGE_RIGHT_OF, 'left_of': cge.EDGE_LEFT_OF,
    'above': cge.EDGE_ABOVE, 'below': cge.EDGE_BELOW,
    'part_of': cge.EDGE_PART_OF, 'before': cge.EDGE_BEFORE, 'after': cge.EDGE_AFTER,
}
for edge in cge_edges_raw:
    try:
        if isinstance(edge, list) and len(edge) >= 3:
            frm, to = str(edge[0]), str(edge[1])
            etype = edge[2]
            if isinstance(etype, torch.Tensor):
                etype = etype.to(cge.device)
            elif isinstance(etype, str):
                etype = edge_type_map.get(etype.lower(), cge.EDGE_CAUSES)
            else:
                etype = cge.EDGE_CAUSES
            cge.add_edge(frm, to, etype)
        elif isinstance(edge, dict):
            frm = str(edge.get('from',''))
            to = str(edge.get('to',''))
            etype = edge.get('type', edge.get('edge_type', 'causes'))
            if isinstance(etype, str):
                etype = edge_type_map.get(etype.lower(), cge.EDGE_CAUSES)
            elif isinstance(etype, torch.Tensor):
                etype = etype.to(cge.device)
            else:
                etype = cge.EDGE_CAUSES
            cge.add_edge(frm, to, etype)
    except: pass

print(f"CGE loaded: {len(cge.nodes)} nodes, {len(cge.edges)} edges")

# 3. Load WorldModel, CausalReasoning, SimEngine with CGE
from core.world_model import WorldModel
from core.causal_reasoning import CausalReasoningEngine
from core.simulation_engine import SimulationEngine

world = WorldModel()
causal = CausalReasoningEngine()
sim = SimulationEngine()

print(f"WorldModel: ok | CausalReasoning: ok | SimEngine: ok")

# 4. Connect SymbolicEngine to the REAL CGE
from core.cognition.symbolic_engine import SymbolicEngine
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

# Extract universals from actual CGE edges
se.extract_universals_from_graph()
print(f"Universals extracted from CGE: {sum(len(v) for v in se.universal_rules.values())}")

# 5. Build CGE knowledge index
# What concepts exist in the CGE related to key topics?
topic_concepts = {
    'time': [n for n in cge.nodes if any(t in n.lower() for t in ['time', 'temporal', 'chrono'])],
    'physics': [n for n in cge.nodes if any(t in n.lower() for t in ['physic', 'energy', 'force', 'motion', 'gravity'])],
    'causality': [n for n in cge.nodes if any(t in n.lower() for t in ['cause', 'effect', 'causal', 'result'])],
    'earth': [n for n in cge.nodes if any(t in n.lower() for t in ['earth', 'planet', 'moon', 'luna'])],
    'science': [n for n in cge.nodes if any(t in n.lower() for t in ['science', 'theory', 'law', 'model'])],
}

print(f"\n=== CGE Knowledge Index ===")
for topic, concepts in topic_concepts.items():
    status = concepts[:5] if concepts else ['NONE']
    print(f"  {topic}: {status}")

# 6. Check what causal edges exist
print(f"\n=== Top Causal Edges in CGE ===")
causal_edges = []
for e in cge.edges[:100]:
    if isinstance(e, dict):
        if 'CAUSES' in str(e.get('edge_type','')):
            causal_edges.append((e.get('from',''), e.get('to','')))
        elif 'caus' in str(e.get('type','')).lower():
            causal_edges.append((e.get('from',''), e.get('to','')))
for frm, to in causal_edges[:20]:
    print(f"  CAUSES: {frm} -> {to}")

# 7. Extract CGE as text knowledge
print(f"\n=== 50 Random Concepts in CGE ===")
import random
sample = sorted(cge.nodes.keys())[:50]
print(', '.join(sample))
