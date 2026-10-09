"""
H82 — Evolucao POPULACIONAL + COMUNICACAO sobre o ARC (programacao genetica cultural)
=====================================================================================
A pedido: passar os agentes evolutivos+sociais no ARC. Mapeamento honesto: uma POPULACAO de
programas (genotipo = sequencia de ops do pool estendido) evolui por GERACOES sobre cada tarefa,
selecionada por RECOMPENSA PARCIAL (fracao de celulas certas — gradiente que a busca exaustiva
nao tem). COMUNICACAO = um ARQUIVO COMPARTILHADO de sub-sequencias uteis (blocos), que qualquer
individuo pode reutilizar = transmissao CULTURAL entre a populacao. Une aos 85 ja resolvidos
(21,2%). Honesto: e busca evolutiva DENTRO do fecho (Solomonoff intacto); pode achar programas
profundos (depth 3-5) que a depth-2 perdeu -> +algumas tarefas; nao quebra a parede. Nao e AGI.
"""
import os, json, glob, time, numpy as np
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PU=_load("exp_h54_pulse.py"); H63=_load("exp_h63_efficiency_breadth.py")
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
rng=np.random.RandomState(22)

def current_solved():
    s=set()
    for f in ("h66_base_solved.json","h70_nca_solved.json","h72_rcens_solved.json","h74"):  # base+NCA+RC
        p=os.path.join(os.path.dirname(__file__),f if f.endswith('.json') else f+'.json')
        if os.path.exists(p):
            try: s|=set(json.load(open(p)))
            except Exception: pass
    return s

def apply_chain(pool,prog,g):
    for nm in prog:
        fn=pool.get(nm)
        if fn is None: return None        # bloco compartilhado com op ausente neste pool -> falha graciosa
        g=fn(g)
        if g is None or not isinstance(g,np.ndarray): return None
    return g
def fitness(pool,prog,train,cm=None):
    """recompensa PARCIAL: media da fracao de celulas certas (0 se shape errado). 1.0 = exato."""
    tot=0.0
    for i,o in train:
        r=apply_chain(pool,prog,i)
        if r is None: return 0.0
        if cm is not None: r=PU.cmap(r,cm)
        if r.shape!=o.shape: return 0.0
        tot+=float((r==o).mean())
    return tot/len(train)

def evolve_task(train, pool, names, archive, pop=60, gens=45, maxlen=5):
    """GP cultural: populacao de programas; recompensa parcial; arquivo compartilhado de blocos."""
    P=[tuple(names[i] for i in rng.randint(0,len(names),rng.randint(1,3))) for _ in range(pop)]
    best=None; bestf=-1
    for gconv in range(gens):
        scored=[]
        for prog in P:
            f=fitness(pool,prog,train)
            # injeta colormap do delta (abducao) se ajudar
            if f<1.0:
                outs=[apply_chain(pool,prog,i) for i,_ in train]
                if all(o is not None and o.shape==t.shape for o,(_,t) in zip(outs,train)):
                    cm=PU.infer_cmap([(o,t) for o,(_,t) in zip(outs,train)])
                    if cm is not None:
                        fc=fitness(pool,prog,train,cm)
                        if fc>f: f=fc;
                        if fc>=1.0: return prog,cm,f
            scored.append((f,prog))
            if f>bestf: bestf=f; best=prog
        if bestf>=1.0: return best,None,bestf
        scored.sort(key=lambda x:-x[0]); elite=[p for _,p in scored[:pop//4]]
        # COMUNICACAO: melhores blocos vao p/ o arquivo compartilhado (cultura)
        for _,p in scored[:3]:
            if len(p)>=2: archive.append(p)
        if len(archive)>200: del archive[:len(archive)-200]
        # nova geracao: mutacao + crossover + reuso de blocos do arquivo (transmissao cultural)
        P=list(elite)
        while len(P)<pop:
            op=rng.rand()
            if op<0.3 and archive:                      # reusa bloco compartilhado (comunicacao)
                blk=archive[rng.randint(len(archive))]; base=elite[rng.randint(len(elite))]
                child=(base+blk)[:maxlen]
            elif op<0.65:                               # mutacao
                base=list(elite[rng.randint(len(elite))]); k=rng.rand()
                if k<0.4 and len(base)<maxlen: base.insert(rng.randint(len(base)+1),names[rng.randint(len(names))])
                elif k<0.7 and len(base)>1: base.pop(rng.randint(len(base)))
                else: base[rng.randint(len(base))]=names[rng.randint(len(names))]
                child=tuple(base)
            else:                                        # crossover
                a=elite[rng.randint(len(elite))]; b=elite[rng.randint(len(elite))]
                child=(a[:len(a)//2]+b[len(b)//2:])[:maxlen] or a
            P.append(tuple(child))
    return best,None,bestf

def main():
    t0=time.time(); tasks=[]
    for fp in sorted(glob.glob(os.path.join(DATA,"*.json"))):
        t=json.load(open(fp)); tasks.append((os.path.basename(fp),
            [(np.array(p["input"]),np.array(p["output"])) for p in t["train"]],
            [(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]))
    solved=current_solved()
    print("="*70); print(f"H82 Evolucao populacional + comunicacao no ARC (base {len(solved)}/400)")
    pool=dict(PU.FIXED); pool.update(H63.NEW); names=list(pool)
    archive=[]; new=[]
    unsolved=[(n,tr,te) for n,tr,te in tasks if n not in solved]
    for name,train,tests in unsolved:
        p=dict(pool)
        if train[0][0].shape[0]==train[0][0].shape[1]: p.update(PU.square_ops())
        nm=list(p.keys())
        prog,cm,f=evolve_task(train,p,nm,archive)
        if f>=1.0 and prog is not None:
            ok=True
            for xi,yo in tests:
                r=apply_chain(p,prog,xi)
                if r is not None and cm is not None: r=PU.cmap(r,cm)
                if r is None or r.shape!=yo.shape or not np.array_equal(r,yo): ok=False; break
            if ok: new.append((name,prog,cm))
    total=len(solved)+len(new)
    print(f"  NOVAS pela evolucao+comunicacao: {len(new)} {[n for n,_,_ in new][:12]}")
    print(f"  blocos no arquivo cultural compartilhado: {len(archive)}")
    print(f"  COBERTURA: {len(solved)} -> {total}/400 = {100*total/400:.1f}%   (era 21,2%)")
    print(f"  tempo {time.time()-t0:.0f}s")
    print(f"  Honesto: GP cultural acha programas profundos DENTRO do fecho que a busca anterior perdeu;")
    print(f"  {'somou cobertura real' if len(new)>0 else 'nao somou (o resto exige primitiva fora-do-fecho)'}. Nao quebra Solomonoff, nao e AGI.")

if __name__=="__main__": main()
