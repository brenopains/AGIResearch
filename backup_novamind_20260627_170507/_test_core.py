"""Test state_bridge serialization"""
import sys, json
sys.path.insert(0, '.')
from core.state_bridge import StateBridge
from core.cognitive_graph_engine import CognitiveGraphEngine

cge = CognitiveGraphEngine(device='cpu')
b = StateBridge.instance()
b.set_from_cge(cge)
b.add_thought('teste', 'system')
s = b.get_state()
print('OK cge:', s['cge']['n_nodes'], 'thoughts:', len(s['thoughts']))
try:
    j = json.dumps(s)
    print('JSON serialization OK:', len(j), 'bytes')
except Exception as e:
    print('JSON FAIL:', e)
