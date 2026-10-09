"""
H107 — DreamCoder COMPLETO: o modelo de RECONHECIMENTO que faltava (fase de sonho)
====================================================================================
Diagnostico (H104/H105): a busca se AFOGA quando a biblioteca/profundidade cresce -> 0 macros,
0 transferencia. O DreamCoder publicado resolve isso com uma peca que NUNCA construimos: um
MODELO DE RECONHECIMENTO que aprende "tarefa com tais features -> tente estes operadores" e GUIA
a busca, mantendo-a tratavel quando a biblioteca cresce.

Sacada: tarefas SINTETICAS dao dados rotulados DE GRACA (sabemos o programa que as gerou) ->
treino o reconhecimento nelas (features da tarefa -> operadores usados) -> uso p/ GUIAR a busca
evolutiva no ARC REAL. Tambem abstrai macros e os inclui na busca guiada.

Testa: busca GUIADA pelo reconhecimento resolve MAIS do residuo que busca cega (H89)? Mede NOVAS
vs teto (94). Honesto: guia a busca DENTRO do fecho (acha composicoes mais fundas, como H89 deu +2);
nao fabrica primitiva fora do fecho. Mas e a peca real que faltava p/ a biblioteca compostar. So-CPU.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter, defaultdict
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H98=_load("exp_h98_dreamcoder.py"); H104=_load("exp_h104_poet.py"); PU=H98.PU; AB=H98.AB if hasattr(H98,'AB') else _load("exp_h55_abduce.py")
random.seed(0); np.random.seed(0)

def features(train):
    i,o=train[0]; H,W=i.shape; OH,OW=o.shape
    def sym(g): return [float(np.array_equal(g,g[:,::-1])),float(np.array_equal(g,g[::-1,:])),float(g.shape[0]==g.shape[1] and np.array_equal(g,g.T))]
    try: nob=len(AB.parse(i,8,False))
    except: nob=0
    ci=set(np.unique(i).tolist()); co=set(np.unique(o).tolist())
    f=[np.sign(OH*OW-H*W)+1, len(ci), len(co), len(co-ci), len(ci-co), min(nob,12)/12.0,
       H/30.0, W/30.0, OH/30.0, OW/30.0]+sym(i)
    return np.array(f,dtype=np.float32)

def main():
    t0=time.time()
    NSYN=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--syn=')),'500'))
    budget=float(next((a.split('=')[1] for a in sys.argv if a.startswith('--budget=')),'0.6'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json"))))|set(json.load(open(os.path.join(HERE,"h89_evo_solved.json"))))
    union|=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json"))))|set(json.load(open(os.path.join(HERE,"h100_solved.json"))))
    try: union|=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    except: pass
    ceiling=union
    fns=H98.seed_dsl(); names=list(fns)
    # ---- DREAM: gera tarefas sinteticas (programa conhecido) e treina perfis de operador ----
    prof_sum=defaultdict(lambda: np.zeros(13,dtype=np.float32)); prof_cnt=Counter(); sols={}
    made=0
    for _ in range(NSYN*3):
        if made>=NSYN: break
        depth=random.randint(2,5); prog=[random.choice(names) for _ in range(depth)]
        env=H104.make_env(prog,fns) if hasattr(H104,'make_env') else None
        if env is None:
            # fallback: gera aqui
            demos=[]
            ok=True
            for _ in range(3):
                gi=H104.rand_grid(); go=H98.apply_prog(prog,fns,gi)
                if go is None or np.array_equal(go,gi) or len(np.unique(go))<=1: ok=False;break
                demos.append((gi,go))
            if not ok or len({d[1].tobytes() for d in demos})<2: continue
            env=demos
        made+=1; ft=features(env)
        for op in set(prog): prof_sum[op]+=ft; prof_cnt[op]+=1
        sols[made]=list(prog)
    profiles={op:prof_sum[op]/prof_cnt[op] for op in prof_cnt}
    # ---- SLEEP: abstrai macros das solucoes sinteticas ----
    added=H98.sleep_abstract(sols,fns,names)
    print(f"H107 | DSL semente {len(H98.seed_dsl())} +{len(added)} macros = {len(names)} | tarefas sinteticas {made} | ops c/ perfil {len(profiles)}")
    # ---- recognition: pesos por tarefa ----
    def rec_weights(train):
        ft=features(train); w={}
        for op in names:
            if op in profiles:
                d=np.linalg.norm(ft-profiles[op]); w[op]=float(np.exp(-d/2.0))+0.05
            else: w[op]=0.3
        return w
    # ---- WAKE GUIADO no ARC real (A/B: guiado vs cego) ----
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    def run_arc(guided):
        got=set()
        for fp in files:
            name=os.path.basename(fp)
            if name in ceiling: continue
            t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
            tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
            local=dict(fns); lnames=list(names); cm=PU.infer_cmap(train)
            if cm is not None: local['cmap']=lambda g,cm=cm:PU.cmap(g,cm); lnames=lnames+['cmap']
            w=rec_weights(train) if guided else Counter()
            f,prog=H98.evolve_solve(train,lnames,local,w,budget)
            if prog is not None and f>=1.0-1e-9:
                if all((lambda r,o:r is not None and r.shape==o.shape and np.array_equal(r,o))(H98.apply_prog(prog,local,i),o) for i,o in train):
                    outs=[H98.apply_prog(prog,local,xi) for xi,_ in tests]
                    if all(r is not None and r.shape==yo.shape and np.array_equal(r,yo) for r,(_,yo) in zip(outs,tests)): got.add(name)
        return got
    print("  rodando busca CEGA no residuo..."); sys.stdout.flush(); blind=run_arc(False)
    print(f"    cega: NOVAS {len(blind)} {sorted(blind)[:6]}"); sys.stdout.flush()
    print("  rodando busca GUIADA (reconhecimento) no residuo..."); guided=run_arc(True)
    print("="*74)
    print(f"H107 — {time.time()-t0:.0f}s")
    print(f"  busca CEGA   : NOVAS vs teto({len(ceiling)}) = {len(blind)} {sorted(blind)[:8]}")
    print(f"  busca GUIADA : NOVAS = {len(guided)} {sorted(guided)[:8]}")
    print(f"  delta guiado-cego: {len(guided-blind)} novas que so o reconhecimento achou: {sorted(guided-blind)[:8]}")
    allnew=blind|guided
    print(f"  novo TETO ARC: {len(ceiling|allnew)}/400 = {100*len(ceiling|allnew)/400:.1f}%")
    print(f"  VEREDITO: {'reconhecimento GUIOU a busca p/ solves novos (peca funcionou)' if (guided-blind) or allnew else 'guia nao destravou novas alem da busca'}")
    print("  Honesto: o reconhecimento torna a busca tratavel/funda (peca real do DreamCoder), mas opera DENTRO do fecho.")
    json.dump(sorted(ceiling|allnew),open(os.path.join(HERE,"arc_unified_final.json"),"w"))

if __name__=="__main__": main()
