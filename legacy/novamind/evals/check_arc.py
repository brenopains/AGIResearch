"""Check ARC data and run ARC test"""
import sys, os, json; sys.path.insert(0, '.')
train_path = 'data/ARC/data/training'
files = [f for f in os.listdir(train_path) if f.endswith('.json')]
print(f'ARC training files: {len(files)}')
task = json.load(open(os.path.join(train_path, files[0])))
print(f'Task keys: {list(task.keys())}')
print(f'Train pairs: {len(task.get(\"train\", []))}')
print(f'Test pairs: {len(task.get(\"test\", []))}')
print(f'Train[0] keys: {list(task[\"train\"][0].keys())}')
print(f'Train[0] grid: {task[\"train\"][0][\"input\"][:3]}...')
print(f'Train[0] output: {task[\"train\"][0][\"output\"][:3]}...')
# Check all file sizes  
print(f'Total ARC tasks: {len(files)}')
print(f'Evaluation files sample: {[f for f in os.listdir(\"data/ARC/data/evaluation\") if f.endswith(\".json\")][:5]}')
eval_files = [f for f in os.listdir('data/ARC/data/evaluation') if f.endswith('.json')]
print(f'Evaluation files: {len(eval_files)}')
