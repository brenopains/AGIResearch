"""Debug MT property name matching"""
import sys; sys.path.insert(0, '.')
from core.bge_encoder import BGEEncoder
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.cognition.symbolic_engine import SymbolicEngine

bge = BGEEncoder()
cge = CognitiveGraphEngine(bge_encoder=bge)
se = SymbolicEngine(cge, bge)
se.register_semantic_roles(cge.EDGE_PART_OF, cge.EDGE_CAUSES, None, None, None, None)
se.register_universal('grom', 'has zorb')

entity = 'specific_entity'
neg_prop = 'has zorb'
if entity not in se.cge.nodes: se.cge.create_node(entity, se._bge_vec(entity))
neg_vec = se._bge_vec(neg_prop)
if neg_vec is not None:
    nid = 'not_' + neg_prop.replace(' ', '_')
    if nid not in se.cge.nodes: se.cge.create_node(nid, -neg_vec)
    se.cge.add_edge(entity, nid, se.cge.EDGE_CAUSED_BY)

print('universal_rules:', se.universal_rules)
result = se.modus_tollens(entity, 'grom')
print('Result:', result)
if result:
    print('  Rule:', result.rule, 'Conf:', result.confidence)
else:
    negated = []
    for edge in se.cge.edges:
        if not isinstance(edge, dict): continue
        frm = str(edge.get('from','')).lower()
        to = str(edge.get('to','')).lower()
        if frm == entity.lower() and to.startswith('not_'):
            negated.append(to[4:])
    print('negated_properties:', negated)
    for np in negated:
        for rs, rp in se.universal_rules.items():
            print(f'  "{np}" in {list(rp.keys())} = {np in rp}')
