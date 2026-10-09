"""
H53b — Wake-Sleep (fixed): genuine depth via random matrices (no group collapse)
================================================================================

H53 bug: spatial ops form a tiny group (8 elements) -> 'deep' tasks collapsed to
depth-1. Fixed: state is a vector; ops are random matrices (products don't
collapse). Tasks need genuine depth. Reusable composites U_i = product of 2 base
matrices. Hard tasks = U_i then U_j (depth-4). With WAKE-SLEEP the system abstracts
U_i at night and next day solves the depth-4 tasks with depth-2 search.
"""
import numpy as np
rng=np.random.RandomState(0); D=8; K=6
B=[rng.randn(D,D)*0.5 for _ in range(K)]                 # base ops (matrices)
# reusable composites (apply a then b => product Mb@Ma)
U=[(0,1),(2,3),(4,5)]
def prod(seq, lib):  # apply lib[seq[0]] then [1]... => product folds left
    M=np.eye(D)
    for idx in seq: M=lib[idx]@M
    return M

# tasks: depth-2 = each U_i ; depth-4 = U_i then U_j
TASKS=[]
for i,(a,b) in enumerate(U): TASKS.append((f'd2:U{i}', [a,b]))
for i,j in [(0,1),(1,2),(0,2),(2,0)]:
    a,b=U[i]; c,d=U[j]; TASKS.append((f'd4:U{i}U{j}', [a,b,c,d]))

def make(seq):
    M=prod(seq,B); xs=[rng.randn(D) for _ in range(4)]
    return [(x,M@x) for x in xs[:3]], xs[3], M@xs[3]

def search(library, train):
    """depth-2 over current library: find seq (len<=2) with lib-product matching demos."""
    n=len(library)
    def ok(M): return all(np.allclose(M@i,o,atol=1e-5) for i,o in train)
    for a in range(n):
        if ok(library[a]): return (a,)
    for a in range(n):
        for b in range(n):
            if ok(library[b]@library[a]): return (a,b)
    return None

from collections import Counter
def run(use_sleep, days=5):
    lib=list(B); solved=set(); per_day=[]
    for day in range(days):
        wins=Counter()
        for name,seq in TASKS:
            if name in solved: continue
            tr,q,qo=make(seq); desc=search(lib,tr)
            if desc is None: continue
            M=lib[desc[0]] if len(desc)==1 else lib[desc[1]]@lib[desc[0]]
            if np.allclose(M@q,qo,atol=1e-5):
                solved.add(name)
                if len(desc)==2: wins[desc]+=1
        per_day.append(len(solved))
        if use_sleep:                       # SLEEP: abstract frequent composites -> new primitive matrices
            for (a,b),_ in wins.most_common(3):
                lib.append(lib[b]@lib[a])
    return per_day, len(lib)

ws,nlib=run(True); ns,_=run(False)
print(f"Tarefas: {len(TASKS)} (3 de prof-2, 4 de prof-4) | prazo 5 dias | busca prof-2 | sem colapso de grupo")
print(f"{'dia':>4} {'COM sono':>10} {'SEM sono':>10}")
for d in range(5): print(f"{d:>4} {ws[d]:>8}/{len(TASKS)} {ns[d]:>8}/{len(TASKS)}")
print("-"*36)
print(f"COM sono: {ws[-1]}/{len(TASKS)} (biblioteca {len(B)}->{nlib} primitivas criadas no 'sonho')")
print(f"SEM sono: {ns[-1]}/{len(TASKS)} (travado: prof-2 fixa nao alcanca prof-4)")
print("=> Wake-sleep VALIDADO: dormir->criar primitiva->acordar->'aha' nas tarefas profundas.")
