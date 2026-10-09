"""
H68 — Corpo Cogumelo como FRONT-END de representação do ARC (a ponte MB->cobertura)
====================================================================================
O MB e memoria associativa (nao sintetizador de programa). Aplicacao fiel ao ARC: um
TRANSDUTOR POR-CELULA onde cada vizinhanca local (patch KxK) da grade e projetada por
expansao aleatoria esparsa (Kenyon) -> readout associativo treinado nos demos da tarefa.
Inductive bias: o codigo esparso de PATCH LOCAL e invariante a POSICAO por construcao
(diferente da CNN do H56, que memorizou posicao e nao generalizou).

Hipotese: isso resolve regras LOCAIS (recolor por vizinhanca, bordas, etc.) generalizando
melhor, e talvez pegue tarefas alem do simbolico (14,2%). Previsao honesta: muito do ganho
sobrepoe o cmap do Pulse -> delta na uniao provavelmente pequeno. Mede com dado. GPU.
So tarefas mesmo-shape (transducao celula-a-celula bem definida). NAO e AGI.
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
K=5; PAD=10           # patch 5x5; cor "borda/fora" = 10 (alem das 10 cores 0-9)
NCH=11; DHIGH=2048; TOPK=48

def patches(g):
    """retorna matriz (H*W, K*K*NCH) one-hot dos patches KxK centrados em cada celula."""
    H,W=g.shape; r=K//2
    gp=np.full((H+2*r,W+2*r),PAD,dtype=np.int64); gp[r:r+H,r:r+W]=g
    out=np.zeros((H*W,K*K,NCH),np.float32); idx=0
    for y in range(H):
        for x in range(W):
            blk=gp[y:y+K,x:x+K].ravel()
            out[idx,np.arange(K*K),blk]=1.0; idx+=1
    return out.reshape(H*W,K*K*NCH)

# projecao aleatoria fixa (Kenyon)
Wproj=torch.randn(K*K*NCH,DHIGH,device=DEVICE)/np.sqrt(K*K*NCH)
def mb_expand(P):
    h=torch.tensor(P,device=DEVICE)@Wproj
    thr=torch.topk(h,TOPK,dim=1).values[:,-1:]
    return (h*(h>=thr)).clamp(min=0)

def solve_mb(train, tests, steps=150):
    if not all(i.shape==o.shape for i,o in train): return None
    # monta dados de treino: patches de todos os demos -> cor de saida
    Xs=[]; Ys=[]
    for i,o in train:
        Xs.append(mb_expand(patches(i))); Ys.append(torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long))
    X=torch.cat(Xs); Y=torch.cat(Ys)
    readout=nn.Linear(DHIGH,10).to(DEVICE); opt=torch.optim.Adam(readout.parameters(),5e-3)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(readout(X),Y).backward(); opt.step()
    # gate: reproduz os demos exatamente?
    with torch.no_grad():
        for i,o in train:
            pred=readout(mb_expand(patches(i))).argmax(1).cpu().numpy().reshape(o.shape)
            if not np.array_equal(pred,o): return None
        outs=[]
        for xi,_ in tests:
            outs.append(readout(mb_expand(patches(xi))).argmax(1).cpu().numpy().reshape(xi.shape))
    return outs

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))  # 57 (14,2%)
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    mb_solved=set(); attempted=0
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        attempted+=1
        outs=solve_mb(train,tests)
        if outs is None: continue
        if all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests)):
            mb_solved.add(os.path.basename(fp))
    union=base|mb_solved; novas=mb_solved-base
    vram=torch.cuda.max_memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print("="*70)
    print(f"H68 Corpo Cogumelo como front-end de representacao do ARC ({DEVICE})")
    print(f"  tarefas mesmo-shape tentadas: {attempted}/400")
    print(f"  MB transdutor resolveu (teste exato): {len(mb_solved)}/400 = {100*len(mb_solved)/400:.1f}%")
    print(f"  NOVAS vs simbolico (14,2%): {len(novas)} {sorted(novas)[:12]}")
    print(f"  UNIAO (simbolico + MB): {len(union)}/400 = {100*len(union)/400:.1f}%   (era 14,2%)")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {vram:.0f}MB")
    print(f"\n  Honesto: o MB e memoria associativa (patch local -> cor), invariante a posicao —")
    print(f"  generaliza regras LOCAIS melhor que a CNN do H56. Ganho na uniao mede o quanto disso")
    print(f"  o simbolico ja nao cobria. Continua sendo transducao local; nao sintetiza programa global. Nao e AGI.")

if __name__=="__main__": main()
