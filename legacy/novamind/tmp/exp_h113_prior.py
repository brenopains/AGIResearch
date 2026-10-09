"""
H113 — Conceito LLM aplicado ao ARC: PRIOR aprendido de corpus auto-gerado guia a busca
=========================================================================================
A receita da LLM: tokenizar corpus gigante -> aprender um PRIOR -> generalizar. Aplicado aqui,
na 3050, na forma NAO-PARAMETRICA (barata, sem treinar modelo grande):
  - CORPUS: auto-gera milhares de tarefas sinteticas a partir de programas conhecidos sobre o DSL
    (FIXED+NEW+familias H93/H100). Guarda (features-da-tarefa -> sequencia de ops do programa).
    Esse corpus E o nosso "internet tokenizada" de transformacoes ARC.
  - PRIOR: para uma tarefa nova, RECUPERA as k tarefas-corpus mais parecidas (por features) e usa
    a frequencia de ops delas como PRIOR que guia/inicializa a busca evolutiva (op-sampling).
  - USO: busca guiada pelo prior no residuo real; gate exato.
Mede NOVAS vs teto (101). Honesto: o corpus e sintetico (mesmo fecho) -> o prior torna a BUSCA
eficiente dentro do fecho (pode achar composicoes que a busca cega perde), nao cria primitiva fora.
E a forma barata e honesta de testar "se um prior de corpus ajuda" — o coracao da receita LLM.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H98=_load("exp_h98_dreamcoder.py"); H104=_load("exp_h104_poet.py"); H107=_load("exp_h107_recognition.py"); PU=H98.PU
random.seed(0); np.random.seed(0)

def build_corpus(fns, names, N):
    feats=[]; progs=[]
    tried=0
    while len(progs)<N and tried<N*4:
        tried+=1; depth=random.randint(1,5); prog=[random.choice(names) for _ in range(depth)]
        demos=[]
        ok=True
        for _ in range(3):
            gi=H104.rand_grid(); go=H98.apply_prog(prog,fns,gi)
            if go is None or np.array_equal(go,gi) or len(np.unique(go))<=1: ok=False;break
            demos.append((gi,go))
        if not ok or len({d[1].tobytes() for d in demos})<2: continue
        feats.append(H107.features(demos)); progs.append(prog)
    return np.array(feats), progs

def main():
    t0=time.time()
    N=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--corpus=')),'4000'))
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'0.6'))
    K=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--k=')),'25'))
    ceiling=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    fns=H98.seed_dsl(); names=list(fns)
    print(f"H113 | construindo corpus de {N} tarefas sinteticas (prior tipo-LLM)..."); sys.stdout.flush()
    CF,CP=build_corpus(fns,names,N)
    print(f"  corpus={len(CP)} programas | features={CF.shape} | teto={len(ceiling)}"); sys.stdout.flush()
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    def prior_weights(train):
        ft=H107.features(train); d=np.linalg.norm(CF-ft,axis=1); idx=np.argsort(d)[:K]
        w=Counter()
        for j in idx:
            for op in CP[j]: w[op]+=1.0
        # normaliza p/ peso base 0.3
        return {op:0.3+w.get(op,0) for op in names}
    def run(guided):
        got=set()
        for fp in files:
            name=os.path.basename(fp)
            if name in ceiling: continue
            t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
            tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
            local=dict(fns); lnames=list(names); cm=PU.infer_cmap(train)
            if cm is not None: local['cmap']=lambda g,cm=cm:PU.cmap(g,cm); lnames=lnames+['cmap']
            w=prior_weights(train) if guided else Counter()
            f,prog=H98.evolve_solve(train,lnames,local,w,budget)
            if prog is not None and f>=1.0-1e-9:
                if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(H98.apply_prog(prog,local,i),o) for i,o in train):
                    outs=[H98.apply_prog(prog,local,xi) for xi,_ in tests]
                    if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)): got.add(name)
        return got
    print("  busca GUIADA pelo prior de corpus no residuo..."); sys.stdout.flush()
    guided=run(True)
    alln=ceiling|guided
    json.dump(sorted(alln),open(os.path.join(HERE,"arc_unified_final.json"),"w"))
    print("="*74)
    print(f"H113 prior tipo-LLM (corpus {len(CP)}) | teto {len(ceiling)}")
    print(f"  NOVAS guiado vs teto: {len(guided-ceiling)} {sorted(guided-ceiling)[:14]}")
    print(f"  novo TETO: {len(alln)}/400 = {100*len(alln)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s")
    print("  Honesto: prior de corpus sintetico guia a busca DENTRO do fecho (eficiencia), nao cria primitiva fora.")
    print("  E a forma barata da receita LLM (corpus->prior->guia); um modelo treinado seria a versao pesada.")

if __name__=="__main__": main()
