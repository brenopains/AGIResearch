"""
H105 — POET consertado: currículo na FRONTEIRA (sem colapso) -> biblioteca cresce de verdade
=============================================================================================
O H104 colapsou porque eu forcei a profundidade pra cima toda rodada -> tarefas impossiveis ->
0 abstracoes. Conserto (POET real, Wang 2019): NAO forcar dificuldade; manter um ARQUIVO de
ambientes e so promover um ambiente mais dificil quando o atual e DOMINADO (minimal criterion +
goal-switching). Assim a abstracao SEMPRE tem material, a biblioteca cresce, e tarefas mais fundas
viram soluveis pelos macros acumulados (o "exponencial" do usuario). Teste de transferencia ao ARC
real a cada rodada. So-CPU. Honesto: macros sao composicoes (fecho); medimos SE crescem e SE transferem.
"""
import os, sys, json, glob, time, random, numpy as np
from collections import Counter, defaultdict
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
HERE=os.path.dirname(__file__); DATA=os.path.join(HERE,"arc_data","data","training")
H98=_load("exp_h98_dreamcoder.py"); H104=_load("exp_h104_poet.py"); PU=H98.PU
random.seed(0); np.random.seed(0)

def make_env(prog, fns):
    demos=[]
    for _ in range(3):
        gi=H104.rand_grid(); go=H98.apply_prog(prog,fns,gi)
        if go is None or np.array_equal(go,gi) or len(np.unique(go))<=1: return None
        demos.append((gi,go))
    if len({d[1].tobytes() for d in demos})<2: return None
    return demos

def solvable(demos, names, fns, budget):
    f,prog=H98.evolve_solve(demos,names,fns,Counter(),budget); return (f>=1.0-1e-9), prog

def main():
    t0=time.time()
    rounds=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--rounds=')),'8'))
    union=set(json.load(open(os.path.join(HERE,"h85_union.json"))))|set(json.load(open(os.path.join(HERE,"h89_evo_solved.json"))))
    union|=set(json.load(open(os.path.join(HERE,"h93_zealot_solved.json"))))|set(json.load(open(os.path.join(HERE,"h100_solved.json"))))
    try: union|=set(json.load(open(os.path.join(HERE,"arc_unified_final.json"))))
    except: pass
    ceiling=union
    fns=H98.seed_dsl(); names=list(fns)
    # arquivo: comeca com programas-semente faceis (depth 2)
    archive=[]
    for _ in range(12):
        prog=[random.choice(names) for _ in range(2)]; env=make_env(prog,fns)
        if env: archive.append({'prog':prog,'env':env,'mastered':False})
    arc_new=set(); hist=[]
    for rd in range(rounds):
        sols={}
        # resolve os do arquivo; marca dominados
        for i,a in enumerate(archive):
            if a['mastered']:
                sols[i]=[x for x in a['prog']]; continue
            ok,prog=solvable(a['env'],names,fns,0.2)
            if ok: a['mastered']=True; sols[i]=[x for x in prog]
        # PROMOVE: a partir dos dominados, gera candidatos +1 op; aceita se na FRONTEIRA (soluvel c/ esforco, nao trivial)
        promoted=0
        for a in [x for x in archive if x['mastered']]:
            if len(archive)>=80: break
            cand=a['prog']+[random.choice(names)]
            env=make_env(cand,fns)
            if env is None: continue
            triv,_=solvable(env,names,fns,0.03)      # trivial?
            hard,_=solvable(env,names,fns,0.25)       # soluvel com esforco?
            if hard and not triv:                     # fronteira
                archive.append({'prog':cand,'env':env,'mastered':False}); promoted+=1
        # ABSTRAI macros (reuso entre tarefas dominadas)
        added=H98.sleep_abstract(sols,fns,names)
        # TRANSFERENCIA ARC
        got=H104.transfer_arc(fns,names,ceiling|arc_new,0.3); arc_new|=got
        hist.append((rd,len([a for a in archive if a['mastered']]),len(archive),len(names),len(added),len(arc_new)))
        print(f"  round {rd}: dominadas {len([a for a in archive if a['mastered']])}/{len(archive)} | +{promoted} fronteira | +{len(added)} macros (DSL={len(names)}) | ARC NOVAS {len(arc_new)} {sorted(arc_new)[:5]}")
        sys.stdout.flush()
    print("="*74)
    print(f"H105 — {time.time()-t0:.0f}s | DSL {len(H98.seed_dsl())}->{len(names)} | arquivo final {len(archive)} ambientes")
    print(f"  biblioteca cresceu exponencial? macros totais = {len(names)-len(H98.seed_dsl())}")
    print(f"  ARC NOVAS vs teto({len(ceiling)}): {len(arc_new)} {sorted(arc_new)[:12]} | novo teto {len(ceiling|arc_new)}/400 = {100*len(ceiling|arc_new)/400:.1f}%")
    grew = (len(names)-len(H98.seed_dsl()))>=3
    print(f"  VEREDITO: curriculo {'NAO colapsou — biblioteca cresceu' if grew else 'ainda fraco'} | transferencia ARC {'SIM (+'+str(len(arc_new))+')' if arc_new else 'nao (macros ficam no fecho)'}")
    print("  Honesto: POET na fronteira gera biblioteca crescente; se ARC NOVAS=0, macros (composicoes) nao furam o fecho — confirma onde a largura nova precisa vir de fora.")

if __name__=="__main__": main()
