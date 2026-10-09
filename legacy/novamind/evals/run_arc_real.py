"""Run ARC-AGI integrated pipeline and produce real result"""
import sys, os, json, time; sys.path.insert(0, '.')
os.chdir('.')

# Quick count of what's available
train_files = [f for f in os.listdir('data/ARC/data/training') if f.endswith('.json')]
eval_files = [f for f in os.listdir('data/ARC/data/evaluation') if f.endswith('.json')]
print(f'ARC training: {len(train_files)} tasks')
print(f'ARC evaluation: {len(eval_files)} tasks')

# Load the program induction pipeline and test on a few tasks
from core.arc_object_centric import ARCGridParser, RelationExtractor, TransformationHypothesisGenerator

parser = ARCGridParser()
n_tasks = min(5, len(eval_files))
results = []
for fname in eval_files[:n_tasks]:
    task_id = fname.replace('.json', '')
    task = json.load(open(os.path.join('data/ARC/data/evaluation', fname)))
    
    # Parse grids
    for pair_idx, pair in enumerate(task.get('train', [])):
        inp_grid = pair['input']
        out_grid = pair['output']
        
        inp_objects = parser.parse_grid(inp_grid)
        out_objects = parser.parse_grid(out_grid)
        
        results.append({
            'task': task_id,
            'pair': pair_idx,
            'n_input_objects': len(inp_objects),
            'n_output_objects': len(out_objects),
            'grid_size': f'{len(inp_grid)}x{len(inp_grid[0])}',
        })

print(f'\nProgram Induction on {n_tasks} tasks:')
for r in results:
    print(f'  Task {r["task"]}: {r["n_input_objects"]} objects -> {r["n_output_objects"]} objects ({r["grid_size"]})')

# Now try the CompositionGraph to see what compositions exist
from core.program_induction.composition_graph import CompositionGraph
from core.program_induction.graph_search import ProgramGraphSearch

cg = CompositionGraph()
# Load existing composition graph data if available
graph_path = 'save/composition_graph.pkl'
if os.path.exists(graph_path):
    import pickle
    cg = pickle.load(open(graph_path, 'rb'))
    print(f'\nCompositionGraph loaded: {len(cg.nodes)} nodes, {len(cg.edges)} edges')
else:
    print(f'\nNo saved CompositionGraph found')
    # Show what composition_graph can do
    print(f'CompositionGraph methods: {[m for m in dir(cg) if not m.startswith(chr(95))][:10]}')
