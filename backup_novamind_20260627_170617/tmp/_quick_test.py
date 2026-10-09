"""Quick test of solve_task on first 20 ARC tasks."""
import sys
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\tmp').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
import json, glob, os, numpy as np, time
from core.spelke_genes import solve_task, apply_rule_dict

def exact_match(a, b):
    if a.shape != b.shape: return False
    return bool(np.array_equal(a, b))

ARC = ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\tmp\\arc_data\\data\\training').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
files = sorted(glob.glob(os.path.join(ARC, '*.json')))[:20]

solved = 0
t_total = time.time()
for fp in files:
    t = json.load(open(fp))
    train = [(np.array(p['input']), np.array(p['output'])) for p in t['train']]
    test = (np.array(t['test'][0]['input']), np.array(t['test'][0]['output']))
    ts = time.time()
    rule_seq = solve_task(train, test[0], max_depth=2, beam_width=30)
    if rule_seq:
        g = test[0].copy()
        for r in rule_seq:
            g = apply_rule_dict(g, r, test[0], test[1])
        if exact_match(g, test[1]):
            solved += 1
            names = [r['gene'].name for r in rule_seq]
            print(f'  SOLVED: {os.path.basename(fp)} in {time.time()-ts:.1f}s | rules: {names}')
        else:
            print(f'  wrong: {os.path.basename(fp)} in {time.time()-ts:.1f}s')
    else:
        print(f'  fail: {os.path.basename(fp)} in {time.time()-ts:.1f}s')

print(f'TOTAL: {solved}/20 in {time.time()-t_total:.1f}s')
