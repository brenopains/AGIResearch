"""Load REAL CGE state directly, preserving all edge data."""
import sys, os, pickle, torch
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); os.chdir(BASE)

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

# Restore edges DIRECTLY from dicts (bypass add_edge to preserve original vectors)
cat_counts = {}
for edge in state['cge_edges']:
    if not isinstance(edge, dict): continue
    frm = str(edge.get('from',''))
    to = str(edge.get('to',''))
    etype = edge.get('edge_type', None)
    evec = edge.get('vector', None)
    
    # Classify the edge type by comparing with reference vectors
    category = 'other'
    if isinstance(etype, torch.Tensor):
        etype = etype.to(device)
        for name, ref_t in [('CAUSES', cge.EDGE_CAUSES), ('CAUSED_BY', cge.EDGE_CAUSED_BY),
                            ('RIGHT_OF', cge.EDGE_RIGHT_OF), ('LEFT_OF', cge.EDGE_LEFT_OF),
                            ('ABOVE', cge.EDGE_ABOVE), ('BELOW', cge.EDGE_BELOW),
                            ('PART_OF', cge.EDGE_PART_OF), ('BEFORE', cge.EDGE_BEFORE),
                            ('AFTER', cge.EDGE_AFTER)]:
            r = ref_t.flatten().unsqueeze(0)
            e = etype.flatten().unsqueeze(0)
            sim = torch.cosine_similarity(e, r).item()
            if sim > 0.85:
                category = name
                break
    
    # Store the edge dict directly (preserves original vector)
    edge_dict = {
        'from': frm, 'to': to,
        'edge_type': etype if isinstance(etype, torch.Tensor) else cge.EDGE_CAUSES,
        'vector': evec.to(device) if isinstance(evec, torch.Tensor) else None,
        'weight': edge.get('confidence', 1.0),
        'category': category,
    }
    cge.edges.append(edge_dict)
    cat_counts[category] = cat_counts.get(category, 0) + 1

print(f"CGE loaded: {len(cge.nodes)} nodes, {len(cge.edges)} edges")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

# Now extract universal rules from CAUSES edges
from core.cognition.symbolic_engine import SymbolicEngine
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)

causal_count = 0
for edge in cge.edges:
    if edge.get('category') in ('CAUSES', 'CAUSED_BY'):
        se.register_universal(edge['from'], edge['to'], float(edge.get('weight', 0.5)))
        causal_count += 1

print(f"\nCAUSES edges: {causal_count}")
print(f"Universal rules extracted: {sum(len(v) for v in se.universal_rules.values())}")

# Show knowledge sample
print(f"\n=== Sample Knowledge from REAL CGE ===")
for subj, props in list(se.universal_rules.items())[:20]:
    for prop, conf in list(props.items())[:2]:
        print(f"  {subj} -> {prop} (conf={conf:.3f})")

# Initialize WorldModel with transitions
from core.world_model import WorldModel
world = WorldModel()
for subj, props in se.universal_rules.items():
    for prop, conf in props.items():
        world.record_transition(subj, prop, conf, None)
print(f"\nWorldModel: {len(world.predictions)} transitions")

# Save the restored CGE state for later use
torch.save({'cge_nodes': cge.nodes, 'cge_edges': cge.edges}, r'save/cge_restored.pth')
print("\nSaved restored CGE to save/cge_restored.pth")
