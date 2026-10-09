"""Debug: test path building in salto sandbox"""
import sys; sys.path.insert(0, '.')
from mtb_world_generator import MTBWorld
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.goal_system import GoalManager
from mtb_sandbox_salto import MTBSandboxSalto

w = MTBWorld(2, 42+137)
cge = CognitiveGraphEngine(device='cpu')
gm = GoalManager(cge=cge)
sandbox = MTBSandboxSalto(cge, gm)
sandbox.load_world(w)

for i in range(30):
    result = sandbox.step_agi()
    mode = sandbox._cognitive_mode.value
    explored = len(sandbox._explored_materials)
    pp = sandbox._path_plan
    action = result.get('action', '?')
    energy = result.get('energy_at_target', 0)
    print(f'{i}: mode={mode} explored={explored}/{len(w.materials)} action={action} pp={pp[:2] if pp else []} pos={sandbox.agi_pos} energy={energy:.1f}')
    if energy > 3:
        print(f'SOLVED!')
        break
