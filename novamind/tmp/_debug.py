"""Debug: por que não acha regra para uma task simples?"""
import sys
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\tmp').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
import json, glob, os, numpy as np
from core.spelke_genes import (
    perceive, Context, make_genes, make_predicates,
    apply_rule_dict_with_nodes, cell_err, gen_candidate_rules
)

# Pega uma task fácil (0d3d703e é "recolor by size rank")
fp = ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\tmp\\arc_data\\data\\training\\0d3d703e.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
t = json.load(open(fp))
print(f"Task: {os.path.basename(fp)}")
for i, p in enumerate(t['train']):
    gi, go = np.array(p['input']), np.array(p['output'])
    print(f"  Train {i}: in shape={gi.shape}, out shape={go.shape}")
    print(f"    in  unique: {sorted(np.unique(gi).tolist())}")
    print(f"    out unique: {sorted(np.unique(go).tolist())}")
    if i == 0:
        # mostra exemplo
        print(f"    IN:")
        for row in gi:
            print(f"      {row.tolist()}")
        print(f"    OUT:")
        for row in go:
            print(f"      {row.tolist()}")

# Testa com depth=1 e vê quantas regras geram erro <= alvo
gi, go = np.array(t['train'][0]['input']), np.array(t['train'][0]['output'])
predicates = make_predicates()
candidates = gen_candidate_rules(gi, go, sorted(np.unique(gi).tolist()) + [0], predicates)
print(f"\nCandidatos gerados: {len(candidates)}")

# Aplica cada um e mede erro
nodes, edges = perceive(gi)
err_initial = cell_err(gi, go)
print(f"Erro inicial: {err_initial}")
good_candidates = []
for rule in candidates:
    try:
        out = apply_rule_dict_with_nodes(gi, rule, nodes, edges)
        err = cell_err(out, go)
        if err < err_initial:
            good_candidates.append((rule['gene'].name, rule['pred_name'], rule['pred_param'], rule['gene_param'], err))
    except Exception as e:
        pass
print(f"\nCandidatos que REDUZEM erro: {len(good_candidates)}")
for g in good_candidates[:10]:
    print(f"  {g}")
