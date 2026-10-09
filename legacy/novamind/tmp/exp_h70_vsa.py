"""
H70 Fase 2 — VSA/HDC analógico: memória associativa one-shot por-célula (treino-zero)
======================================================================================
VSA/HDC (Kanerva; Plate HRR; NVSA Hersche 2022) é forte em SELECAO/recuperacao (RAVEN multiple-
choice). O ARC e GERATIVO (precisa criar a grade), entao a aplicacao viavel e uma MEMORIA
ASSOCIATIVA one-shot: contexto local (vizinhanca 3x3) -> cor de saida, via bind/bundle. E o
analogo HDC do MB (H68) mas SEM TREINO e SEM ESQUECIMENTO (superposicao). Honesto: capacidade
limitada pelo bundling (crosstalk) -> expectativa modesta, provavel sobreposicao com o MB.
Reusa o esquema do core/novacrush/hdc.py (bind=mult bipolar; bundle=soma+sinal; sim=cosseno).
GPU. So tarefas mesmo-shape. Nao e AGI.
"""
import os, json, glob, time, numpy as np, torch
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
DIM=8192; NCOL=11; K=3   # vizinhanca 3x3; cor 10 + borda(10)
# hipervetores fixos (bipolar) — esquema do hdc.py
COLOR=(torch.randint(0,2,(NCOL,DIM),device=DEVICE)*2-1).float()
POS=(torch.randint(0,2,(K*K,DIM),device=DEVICE)*2-1).float()   # role de cada posicao da vizinhanca

def contexts(g):
    """retorna (H*W, DIM): para cada celula, bundle_k POS[k] ⊗ COLOR[viz_k]."""
    H,W=g.shape; r=K//2
    gp=np.full((H+2*r,W+2*r),10,dtype=np.int64); gp[r:r+H,r:r+W]=g   # cor 10 = fora/borda
    ctx=torch.zeros(H*W,DIM,device=DEVICE); idx=0
    for y in range(H):
        for x in range(W):
            blk=gp[y:y+K,x:x+K].ravel()
            v=(POS*COLOR[blk]).sum(0)            # bundle dos 9 (POS_k ⊗ COLOR_k)
            ctx[idx]=v; idx+=1
    return torch.sign(ctx+1e-6)                  # bipolar

def solve_vsa(train, tests):
    if not all(i.shape==o.shape for i,o in train) or len(train)==0: return None
    # memoria associativa: M = bundle_celulas( ctx ⊗ COLOR[out] )
    M=torch.zeros(DIM,device=DEVICE)
    for i,o in train:
        ctx=contexts(i); outc=COLOR[torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long)]
        M=M+(ctx*outc).sum(0)
    M=torch.sign(M+1e-6)
    def predict(g):
        H,W=g.shape; ctx=contexts(g)
        q=ctx*M                                  # unbind (bind auto-inverso) -> ~COLOR[out]
        sims=q@COLOR.t()                         # cleanup contra as 10 cores
        return sims[:, :10].argmax(1).cpu().numpy().reshape(H,W).astype(np.int64)
    for i,o in train:
        if not np.array_equal(predict(i),o): return None
    return [predict(xi) for xi,_ in tests]

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    vsa=set(); attempted=0
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        attempted+=1
        outs=solve_vsa(train,tests)
        if outs is None: continue
        if all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests)):
            vsa.add(os.path.basename(fp))
    json.dump(sorted(vsa),open(os.path.join(os.path.dirname(__file__),"h70_vsa_solved.json"),"w"))
    novas=vsa-base
    print("="*70)
    print(f"H70 Fase 2 — VSA/HDC associativo one-shot ({DEVICE})")
    print(f"  mesmo-shape tentadas: {attempted}/400 | VSA resolveu: {len(vsa)}/400")
    print(f"  NOVAS vs base (15,0%): {len(novas)} {sorted(novas)[:15]}")
    print(f"  UNIAO (base + VSA): {len(base|vsa)}/400 = {100*len(base|vsa)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s | Honesto: VSA e selecao/memoria, nao geracao; capacidade")
    print(f"  limitada pelo bundling. Mede com dado. Nao e AGI.")

if __name__=="__main__": main()
