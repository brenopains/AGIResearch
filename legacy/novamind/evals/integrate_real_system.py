"""INTEGRAÇÃO: conecta SymbolicEngine ao CGE real com 9563 arestas HDV.
Extrai conhecimento dos EDGE_CAUSES/EDGE_PART_OF (tensores HDV) 
e popula WorldModel / CausalReasoning com o que existe."""
import sys, os, pickle, torch
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

# 1. Load state
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
device = cge.device

# Restore nodes
for nid, nd in state['cge_nodes'].items():
    if isinstance(nid, str) and isinstance(nd, dict):
        try:
            nd2 = dict(nd)
            for k in ['vector','semantic','perceptual']:
                if k in nd2 and isinstance(nd2[k], torch.Tensor):
                    nd2[k] = nd2[k].to(device)
            cge.nodes[nid] = nd2
        except: pass

# Map for edge type HDV tensors -> string categories
EDGE_LABELS = {
    'RIGHT_OF': 'spatial', 'LEFT_OF': 'spatial',
    'ABOVE': 'spatial', 'BELOW': 'spatial',
    'BEFORE': 'temporal', 'AFTER': 'temporal',
    'CAUSES': 'causal', 'CAUSED_BY': 'causal',
    'PART_OF': 'hierarchy',
}

# Restore edges, categorizing by HDV tensor identity
restored_count = {'causal': 0, 'spatial': 0, 'temporal': 0, 'hierarchy': 0, 'other': 0}

# Build reference tensors for edge type identification
ref_tensors = {}
for name, attr_name in [('RIGHT_OF', 'EDGE_RIGHT_OF'), ('LEFT_OF', 'EDGE_LEFT_OF'),
                         ('ABOVE', 'EDGE_ABOVE'), ('BELOW', 'EDGE_BELOW'),
                         ('BEFORE', 'EDGE_BEFORE'), ('AFTER', 'EDGE_AFTER'),
                         ('CAUSES', 'EDGE_CAUSES'), ('CAUSED_BY', 'EDGE_CAUSED_BY'),
                         ('PART_OF', 'EDGE_PART_OF')]:
    ref_tensors[name] = getattr(cge, attr_name)

for edge in state['cge_edges']:
    try:
        if isinstance(edge, list) and len(edge) >= 3:
            frm, to = str(edge[0]), str(edge[1])
            etype = edge[2]
            if isinstance(etype, torch.Tensor):
                etype = etype.to(device)
                # Identify edge category by comparing with reference tensors
                category = 'other'
                for name, ref_t in ref_tensors.items():
                    if etype.shape == ref_t.shape:
                        sim = torch.cosine_similarity(etype.flatten().unsqueeze(0),
                                                      ref_t.flatten().unsqueeze(0)).item()
                        if sim > 0.9:
                            category = EDGE_LABELS.get(name, 'other')
                            break
            elif isinstance(etype, str):
                cat = etype.lower()
                if 'caus' in cat: category = 'causal'
                elif 'right' in cat or 'left' in cat or 'above' in cat: category = 'spatial'
                elif 'before' in cat or 'after' in cat: category = 'temporal'
                elif 'part' in cat: category = 'hierarchy'
                else: category = 'other'
                # Use reference tensor by category
                if category == 'causal': etype = cge.EDGE_CAUSES
                elif category == 'spatial': etype = cge.EDGE_RIGHT_OF
                elif category == 'temporal': etype = cge.EDGE_BEFORE
                elif category == 'hierarchy': etype = cge.EDGE_PART_OF
                else: etype = cge.EDGE_CAUSES
            else:
                etype, category = cge.EDGE_CAUSES, 'other'
            
            cge.add_edge(frm, to, etype)
            restored_count[category] = restored_count.get(category, 0) + 1
    except: pass

print(f"Restored edges by category:")
for cat, count in sorted(restored_count.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"Total edges in CGE: {len(cge.edges)}")

# 2. Connect SymbolicEngine to REAL CGE
from core.cognition.symbolic_engine import SymbolicEngine
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

# Now extract universals by looking at EDGE_CAUSES type edges specifically
causal_count = 0
for edge in cge.edges:
    if not isinstance(edge, dict): continue
    frm = str(edge.get('from', ''))
    to = str(edge.get('to', ''))
    edge_type = edge.get('edge_type', None)
    weight = edge.get('weight', 0.5)
    if isinstance(weight, torch.Tensor): weight = weight.mean().item()
    
    # Check if this is a CAUSES edge by comparing with EDGE_CAUSES tensor
    is_causal = False
    if isinstance(edge_type, torch.Tensor):
        sim = torch.cosine_similarity(
            edge_type.flatten().unsqueeze(0),
            cge.EDGE_CAUSES.flatten().unsqueeze(0)
        ).item()
        is_causal = sim > 0.9
    
    if is_causal and weight > 0.3 and frm and to:
        se.register_universal(frm, to, float(weight))
        causal_count += 1

print(f"Universals extracted from real CGE edges: {causal_count}")
print(f"Total universal rules: {sum(len(v) for v in se.universal_rules.values())}")

# Show what the system actually knows
print(f"\n=== Knowledge in REAL CGE ===")
sample_rules = list(se.universal_rules.items())[:15]
for subj, props in sample_rules:
    for prop, conf in list(props.items())[:3]:
        print(f"  {subj} -> {prop} (conf={conf:.3f})")

# 3. Initialize WorldModel with CGE knowledge
from core.world_model import WorldModel
world = WorldModel()
for subj, props in se.universal_rules.items():
    for prop, conf in props.items():
        if conf > 0.5:
            # Register as a known transition
            world.record_transition(subj, prop, conf, -1)

print(f"\nWorldModel populated: {len(world.predictions)} transitions")

# 4. Check causal reasoning with the real system
from core.causal_reasoning import CausalReasoningEngine
causal = CausalReasoningEngine()

# Test: what does the system know about causality?
print(f"\n=== Causal Knowledge Query ===")
sample_concepts = sorted(cge.nodes.keys())[:10]
for concept in sample_concepts:
    # Check what this concept causes
    for edge in cge.edges[:5000]:
        if isinstance(edge, dict) and str(edge.get('from','')) == concept:
            to = str(edge.get('to',''))
            et = edge.get('edge_type', None)
            if isinstance(et, torch.Tensor):
                s = torch.cosine_similarity(et.flatten().unsqueeze(0), 
                                            cge.EDGE_CAUSES.flatten().unsqueeze(0)).item()
                if s > 0.9:
                    print(f"  CAUSES: {concept} -> {to}")
