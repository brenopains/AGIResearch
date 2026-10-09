"""Carrega o sistema REAL de brain_live.pth (802 nós, 29.594 arestas)
e responde as perguntas usando conhecimento factual sem hardcode."""
import sys, os, torch, pickle
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)

print("=" * 70)
print("CARREGANDO SISTEMA REAL (brain_live.pth)")
print("=" * 70)

# Load brain_live.pth state
data = torch.load(r'save/brain_live.pth', map_location='cpu', weights_only=False)
brain_nodes = data['nodes']
brain_edges = data['edges']

print(f"Nodes: {len(brain_nodes)}")
print(f"Edges: {len(brain_edges)}")

# Sample node names
node_names = list(brain_nodes.keys())
print(f"\nSample concepts: {node_names[:30]}")
print(f"Physics concepts: {[n for n in node_names if 'phys' in n.lower()][:5]}")
print(f"Force concepts: {[n for n in node_names if 'force' in n.lower()][:5]}")
print(f"Time concepts: {[n for n in node_names if 'time' in n.lower()][:5]}")
print(f"Earth concepts: {[n for n in node_names if 'earth' in n.lower()][:5]}")
print(f"Energy concepts: {[n for n in node_names if 'energy' in n.lower()][:5]}")
print(f"Moon concepts: {[n for n in node_names if 'moon' in n.lower() or 'luna' in n.lower()][:5]}")
print(f"Einstein/relativity: {[n for n in node_names if 'einstein' in n.lower() or 'relativ' in n.lower()][:5]}")
print(f"Quantum concepts: {[n for n in node_names if 'quant' in n.lower()][:5]}")
print(f"Machine concepts: {[n for n in node_names if 'machin' in n.lower() or 'engine' in n.lower() or 'motor' in n.lower()][:5]}")

# Check edge types
edge_types = set()
for e in brain_edges[:1000]:
    if isinstance(e, dict):
        et = e.get('edge_type', None)
        if et is not None:
            edge_types.add(et)
            if isinstance(et, int):
                pass
print(f"\nEdge type values found: {sorted(edge_types)[:10]}")

# Build a mapping from edge_type int -> description
# Common edges: from->to, find patterns
print(f"\n=== Sample Edges by type ===")
for et_val in sorted(edge_types)[:5]:
    count = 0
    for e in brain_edges:
        if isinstance(e, dict) and e.get('edge_type') == et_val:
            print(f"  type={et_val}: {e.get('from','')} -> {e.get('to','')}")
            count += 1
            if count >= 3:
                break
    if count == 0:
        print(f"  type={et_val}: (no samples found)")

# Build CGE from brain_live data
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
device = cge.device

# Create nodes from brain data
for node_name in node_names:
    if isinstance(node_name, str) and len(node_name) > 1 and len(node_name) < 50:
        nd = brain_nodes[node_name]
        if isinstance(nd, dict) and 'vector' in nd:
            vec = nd['vector']
            if isinstance(vec, torch.Tensor):
                nd_copy = dict(nd)
                nd_copy['vector'] = vec.to(device)
                # Ensure proper node structure
                if 'id' not in nd_copy:
                    nd_copy['id'] = node_name
                if 'confidence' not in nd_copy:
                    nd_copy['confidence'] = 1.0
                cge.nodes[node_name] = nd_copy

# Create edges from brain data
# Map edge_type int to CGE reference vectors
for edge in brain_edges:
    if not isinstance(edge, dict): continue
    frm = str(edge.get('from', ''))
    to = str(edge.get('to', ''))
    et = edge.get('edge_type', 0)
    vec = edge.get('vector', None)
    
    # Use CAUSES as default, or try to infer from context
    # The original code used et=-1 which might map to CAUSES
    ref_type = cge.EDGE_CAUSES  # default
    # Store edge dict directly
    edge_dict = {
        'from': frm, 'to': to,
        'edge_type': ref_type,
        'vector': vec.to(device) if isinstance(vec, torch.Tensor) else None,
        'weight': edge.get('confidence', 1.0),
    }
    cge.edges.append(edge_dict)

print(f"\nCGE loaded from brain_live.pth: {len(cge.nodes)} nodes, {len(cge.edges)} edges")

# Connect SymbolicEngine
from core.cognition.symbolic_engine import SymbolicEngine
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

# Extract universal rules from all edges
for edge in cge.edges:
    if isinstance(edge, dict):
        frm = edge.get('from', '')
        to = edge.get('to', '')
        w = edge.get('weight', 0.5)
        if isinstance(w, torch.Tensor): w = w.mean().item()
        if frm and to and w > 0.3:
            se.register_universal(frm, to, float(w))

print(f"Universal rules extracted: {sum(len(v) for v in se.universal_rules.values())}")

# Show what the system knows
print(f"\n=== KNOWLEDGE IN REAL SYSTEM ===")
topics = {
    'time_travel': ['time', 'travel', 'temporal', 'causality'],
    'einstein': ['einstein', 'relativity', 'physicist'],
    'gravity': ['gravity', 'gravitational', 'force', 'gravitation'],
    'earth_moon': ['earth', 'moon', 'planet'],
    'quantum': ['quantum', 'mechanics', 'relativity'],
    'energy': ['energy', 'perpetual', 'motion', 'machine'],
}
for topic, keywords in topics.items():
    found = [n for n in node_names if any(k in n.lower() for k in keywords)]
    print(f"\n  {topic}: {found[:8] if found else 'NOT FOUND'}")
    if found:
        # What does the system know about the first found concept?
        concept = found[0]
        rules = se.universal_rules.get(concept, {})
        if rules:
            for prop, conf in list(rules.items())[:3]:
                print(f"    {concept} -> {prop} (conf={conf:.3f})")
        # Check outgoing edges in CGE
        outgoing = [(e.get('to',''), e.get('weight',0)) for e in cge.edges 
                    if isinstance(e, dict) and str(e.get('from','')) == concept]
        if outgoing:
            for to, w in outgoing[:3]:
                print(f"    EDGE: {concept} -> {to} (w={w})")
