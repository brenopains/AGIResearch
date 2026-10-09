"""
H69 — Hibrido MB + abdução: o MB capta a ESTRUTURA local, a abdução CORRIGE simbolicamente
============================================================================================
"MB propoe qual primitiva" = ordenacao -> invariante p/ cobertura (H62 provou). Entao NAO e isso.
Testamos o que PODE adicionar cobertura: duas etapas neuro-simbolicas.
 1. MB (H68) preve a saida por-celula (capta a estrutura/segmentacao local).
 2. ABDUCAO corrige: encontra um transform simbolico simples T tal que T(MB_pred)==saida_real
    em TODOS os demos. Candidatos: colormap inferido (MB acerta ONDE vao as cores mas erra
    QUAIS), ou um op simbolico (faltou um flip/simetria). Se exato nos demos -> aplica no teste.
Mede cobertura nova vs a uniao de 15,0% (H68). GPU. Honesto: continua transducao local +
conserto simbolico; nao sintetiza programa global. Previsao: ganho pequeno (quase-acertos do
MB costumam estar genuinamente errados, nao so com cor trocada). Nao e AGI.
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util
def _load(f):
    p=os.path.join(os.path.dirname(__file__),f); s=importlib.util.spec_from_file_location(f[:-3],p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
MB=_load("exp_h68_mb_arc.py"); PU=_load("exp_h54_pulse.py")
DEVICE=MB.DEVICE; DATA=MB.DATA

def mb_predict(train, tests):
    """treina o readout MB nos demos; retorna predicoes (demos e tests) ou None se nao mesmo-shape."""
    if not all(i.shape==o.shape for i,o in train): return None
    Xs=[]; Ys=[]
    for i,o in train:
        Xs.append(MB.mb_expand(MB.patches(i))); Ys.append(torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long))
    X=torch.cat(Xs); Y=torch.cat(Ys)
    ro=nn.Linear(MB.DHIGH,10).to(DEVICE); opt=torch.optim.Adam(ro.parameters(),5e-3)
    for _ in range(150):
        opt.zero_grad(); F.cross_entropy(ro(X),Y).backward(); opt.step()
    with torch.no_grad():
        dem=[ro(MB.mb_expand(MB.patches(i))).argmax(1).cpu().numpy().reshape(o.shape) for i,o in train]
        tst=[ro(MB.mb_expand(MB.patches(xi))).argmax(1).cpu().numpy().reshape(xi.shape) for xi,_ in tests]
    return dem,tst

def correct(mb_dem, train, mb_tst, tests):
    """acha T simbolico tal que T(MB_pred)==saida em todos demos; aplica no teste. Retorna preds ou None."""
    # candidato 1: colormap (MB acerta estrutura, erra cores)
    cm=PU.infer_cmap([(p,o) for p,(_,o) in zip(mb_dem,train)])
    if cm is not None and all(np.array_equal(PU.cmap(p,cm),o) for p,(_,o) in zip(mb_dem,train)):
        return [PU.cmap(p,cm) for p in mb_tst]
    # candidato 2: um op simbolico simples (faltou flip/simetria/etc.)
    pool=dict(PU.FIXED); pool.update(PU.NEW if hasattr(PU,'NEW') else {})
    for nm,f in pool.items():
        try:
            if all((lambda r,o: r is not None and r.shape==o.shape and np.array_equal(r,o))(f(p),o) for p,(_,o) in zip(mb_dem,train)):
                return [f(p) for p in mb_tst]
        except Exception: continue
    return None

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))  # 57
    h68=set(json.load(open(os.path.join(os.path.dirname(__file__),"h68_solved.json")))) if os.path.exists(os.path.join(os.path.dirname(__file__),"h68_solved.json")) else set()
    union0=base|h68
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    hybrid_new=set(); mb_exact=set()
    for fp in files:
        name=os.path.basename(fp); t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        pred=mb_predict(train,tests)
        if pred is None: continue
        mb_dem,mb_tst=pred
        # MB exato (= H68)
        if all(np.array_equal(p,o) for p,(_,o) in zip(mb_dem,train)):
            if all(np.array_equal(pt,yo) for pt,(_,yo) in zip(mb_tst,tests)): mb_exact.add(name)
            continue
        # senao: tenta conserto simbolico (o HIBRIDO)
        cpred=correct(mb_dem,train,mb_tst,tests)
        if cpred is not None and all(c.shape==yo.shape and np.array_equal(c,yo) for c,(_,yo) in zip(cpred,tests)):
            hybrid_new.add(name)
    allset=base|mb_exact|hybrid_new
    print("="*70)
    print(f"H69 Hibrido MB+abducao ({DEVICE})")
    print(f"  base simbolico: {len(base)}/400 (14,2%) | MB exato (H68): {len(mb_exact)}")
    print(f"  HIBRIDO (MB+conserto simbolico) resolveu a MAIS: {len(hybrid_new-base-mb_exact)} {sorted(hybrid_new-base-mb_exact)[:12]}")
    print(f"  UNIAO TOTAL (simbolico + MB + hibrido): {len(allset)}/400 = {100*len(allset)/400:.1f}%   (era 15,0%)")
    print(f"  tempo {time.time()-t0:.0f}s")
    print(f"\n  Honesto: o conserto simbolico do quase-acerto do MB {'adicionou' if len(hybrid_new-base-mb_exact)>0 else 'NAO adicionou'} cobertura.")
    print(f"  Continua transducao local + cleanup; nao cruza a parede de largura. Nao e AGI.")

if __name__=="__main__": main()
