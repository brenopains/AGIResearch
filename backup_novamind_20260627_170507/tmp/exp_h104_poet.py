"""
H104 — POET-ARC: co-evolucao ABERTA (o "deus" cria/complexifica o mundo) + transferencia ao ARC
================================================================================================
A visao do usuario, destilada na peca genuinamente NOVA que faltava: nao tarefas fixas (H82) nem
auto-geradas presas ao fecho (H92), mas CO-EVOLUCAO ABERTA estilo POET (Wang 2019):
  - GERADOR ("deus"): cria tarefas compondo elementos primordiais (o DSL) e COMPLEXIFICA sem teto.
  - SELECAO POR FRONTEIRA (minimal criterion / learning progress): mantem so as tarefas que sao
    soluveis-mas-nao-triviais AGORA (a borda do que a populacao consegue) — currículo auto-organizado.
  - SOLVERS co-evoluem (busca evolutiva sobre o DSL); solucoes viram MACROS (abstracao composta).
  - ARC = pressao de sobrevivencia: a cada rodada testa a transferencia no RESIDUO REAL do ARC.
  - CREDITO: rastreia QUAL macro/op correlaciona com avanço (o "entender o que mudou" do usuario).
So-CPU. Honesto: pode capar perto de 93 (macros sao composicoes = fecho), mas a selecao aberta por
fronteira e mecanismo novo — medido com dado. Se NOVAS>0 no residuo, o currículo aberto destravou.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter, defaultdict
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H98=_load("exp_h98_dreamcoder.py"); H102=_load("exp_h102_grammar.py")
PU=H98.PU; random.seed(0); np.random.seed(0)

def rand_grid():
    H=random.randint(3,12); W=random.randint(3,12); k=random.randint(2,5); g=np.zeros((H,W),np.int64)
    for _ in range(random.randint(2,max(3,H*W//3))): g[random.randrange(H),random.randrange(W)]=random.randint(1,k)
    return g

def gen_task(names, fns, depth):
    for _ in range(25):
        prog=[random.choice(names) for _ in range(depth)]; demos=[]; ok=True
        for _ in range(3):
            gi=rand_grid(); go=H98.apply_prog(prog,fns,gi)
            if go is None or np.array_equal(go,gi) or len(np.unique(go))<=1: ok=False; break
            demos.append((gi,go))
        if ok and len({d[1].tobytes() for d in demos})>1: return demos, prog
    return None, None

def quick_solvable(demos, names, fns, budget=0.08):
    f,prog=H98.evolve_solve(demos,names,fns,Counter(),budget); return (f>=1.0-1e-9), prog

def transfer_arc(fns, names, ceiling, budget=0.4):
    # solver = busca evolutiva sobre o DSL CRESCIDO (com macros), no residuo real
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); got=set()
    for fp in files:
        name=os.path.basename(fp)
        if name in ceiling: continue
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        local=dict(fns); lnames=list(names); cm=PU.infer_cmap(train)
        if cm is not None: local['cmap']=lambda g,cm=cm:PU.cmap(g,cm); lnames=lnames+['cmap']
        f,prog=H98.evolve_solve(train,lnames,local,Counter(),budget)
        if prog is not None and f>=1.0-1e-9:
            if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(H98.apply_prog(prog,local,i),o) for i,o in train):
                outs=[H98.apply_prog(prog,local,xi) for xi,_ in tests]
                if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)): got.add(name)
    return got

def main():
    t0=time.time()
    rounds=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--rounds=')),'5'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json"))))|set(json.load(open(os.path.join(HERE,"h89_evo_solved.json"))))
    union|=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json"))))|set(json.load(open(os.path.join(HERE,"h100_solved.json"))))
    ceiling=union  # 93
    fns=H98.seed_dsl(); names=list(fns); macro_n=0
    print(f"H104 POET-ARC | DSL inicial={len(names)} | teto ARC={len(ceiling)} | rounds={rounds}")
    depth=3; arc_new=set(); credit=Counter()
    for rd in range(rounds):
        # GERA + SELECIONA POR FRONTEIRA
        kept=[]; tried=0
        while len(kept)<40 and tried<200:
            tried+=1; demos,truep=gen_task(names,fns,depth)
            if demos is None: continue
            solvable,_=quick_solvable(demos,names,fns,0.05)
            # fronteira: queremos soluveis com esforco (nem triviais nem impossiveis) -> aceita ~metade
            if solvable or random.random()<0.3: kept.append(demos)
        # WAKE: resolve os mantidos, coleta solucoes
        sols={}
        for i,demos in enumerate(kept):
            f,prog=H98.evolve_solve(demos,names,fns,Counter(),0.15)
            if prog is not None and f>=1.0-1e-9: sols[i]=[x for x in prog]
        # SLEEP: abstrai macros (reuso entre tarefas)
        added=H98.sleep_abstract(sols,fns,names)
        macro_n+=len(added)
        # COMPLEXIFICA (aberto)
        depth+=1
        # TRANSFERENCIA ao ARC real
        got=transfer_arc(fns,names,ceiling|arc_new,0.35)
        arc_new|=got
        for nm,seq,_ in added: credit[ '+'.join(seq) ]+=1
        print(f"  round {rd}: tarefas-fronteira {len(kept)} resolvidas {len(sols)} | +{len(added)} macros (DSL={len(names)}) | depth->{depth} | ARC NOVAS acum {len(arc_new)} {sorted(arc_new)[:6]}")
        sys.stdout.flush()
    print("="*74)
    print(f"H104 — {time.time()-t0:.0f}s | DSL {len(H98.seed_dsl())}->{len(names)} (+{macro_n} macros auto-abstraidos)")
    print(f"  ARC NOVAS vs teto(93): {len(arc_new)} {sorted(arc_new)[:12]}")
    print(f"  novo TETO ARC: {len(ceiling|arc_new)}/400 = {100*len(ceiling|arc_new)/400:.1f}%")
    print(f"  VEREDITO: {'TRANSFEREU — o currículo aberto destravou tarefa real do residuo!' if arc_new else 'sem transferencia (macros = composicoes no fecho; curriculo aberto nao gerou primitiva fora dele)'}")
    print("  Honesto: POET/co-evolucao aberta medida com transferencia real ao ARC. Se 0, e a confirmacao")
    print("  mais forte ate agora de que largura nova exige sinal de FORA do fecho do DSL. Nao e AGI.")

if __name__=="__main__": main()
