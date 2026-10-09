import sys, numpy as np
sys.path.insert(0,'.')
from core.program_induction import (
    PrimitiveRegistry, ProgramSynthesizer, CompositionGraph,
    OperatorGroundingBridge, detect_applicable_operators
)
from core.program_induction.primitives import (
    create_recolor_primitive, create_remove_primitive,
    create_move_primitive, create_add_primitive,
)

registry = PrimitiveRegistry()
for fn in [create_recolor_primitive, create_remove_primitive,
           create_move_primitive, create_add_primitive]:
    registry.register(fn())

graph = CompositionGraph()
bridge = OperatorGroundingBridge(registry, graph)
s = ProgramSynthesizer(registry, composition_graph=graph, grounding_bridge=bridge)

print('[TEST 1] Spatial operators')
inp = np.array([[1,2],[3,4]], dtype=int)
out = np.rot90(inp, k=-1)
matches = detect_applicable_operators(inp, out)
print('  Matches:', [m['name'] for m in matches])

print('\n[TEST 2] Bridge proposals')
props = bridge.generate_proposals(inp, out)
for p in props:
    print(f'  {p.name}: conf={p.confidence:.2f} source={p.source}')

print('\n[TEST 3] Synthesis with proposals')
inp = np.array([[1,2],[3,4]], dtype=int)
out = np.fliplr(inp)
progs = s.synthesize(inp, out)
if progs:
    best = progs[0]
    final = best.execute(inp)
    exact = np.array_equal(final, out)
    print(f'  Best: {best.name} score={best.score:.3f} exact={exact} steps={best.n_steps}')

print('\n[TEST 4] Bridge promotion')
for p in bridge.generate_proposals(inp, out):
    if p.confidence == 1.0:
        ok = bridge.promote_if_validated(p, inp, out)
        print(f'  Promoting {p.name}: {ok}')

print('\nRegistry primitives:')
for p in registry.get_all():
    print(f'  {p.name}: {p.description[:50]}')

print('\nCompositionGraph:')
print(f'  Edges: {graph.get_edges()}')
