"""
H56-T — Transdutor neural POR-PUZZLE (GPU, sem pre-treino) + AIRV
=================================================================
A metade de TRANSDUCAO do ensemble induction+transduction (Li et al. 2024): em vez de
sintetizar um programa, uma CNN pequena PREVE a grade de saida celula-a-celula. Treinada
no TEMPO DE TESTE so nos demos de cada tarefa, com augmentation pesada (8 simetrias di-
edrais x permutacoes de cor) para induzir invariancia. Predicao com AIRV (augment->infer
->reverse->vote). Pega tarefas perceptuais "difusas" que a busca simbolica nao expressa.

Escopo honesto: so tarefas em que TODOS os demos tem output.shape==input.shape (predicao
celula-a-celula bem definida); senao ABSTEM (o simbolico cobre muitas das que mudam tamanho).
Treino-zero de pre-treino; so TTT por puzzle. Roda na GPU (RTX 3050, <4GB).
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
MAXSZ=30; NCOL=10

# ---------- augmentation ----------
def dihedral(g,k):  # opera so nos 2 primeiros eixos (seguro p/ mapas (h,w,canais))
    if k==0: return g
    if k==1: return np.rot90(g,1)
    if k==2: return np.rot90(g,2)
    if k==3: return np.rot90(g,3)
    if k==4: return g[:,::-1]
    if k==5: return g[::-1,:]
    if k==6: return np.swapaxes(g,0,1)
    if k==7: return np.swapaxes(g[:,::-1],0,1)
def dihedral_inv(g,k):  # inverso da transformacao k
    inv={0:0,1:3,2:2,3:1,4:4,5:5,6:6,7:7}[k]
    return dihedral(g,inv)

def colorperm(g,perm): return perm[g]

# ---------- tensorizacao ----------
def to_tensor(g,dr=0,dc=0):  # SEM canais posicionais (forcam memorizacao); equivariancia a translacao
    H,W=g.shape; t=torch.zeros(11,MAXSZ,MAXSZ)
    for c in range(NCOL): t[c,dr:dr+H,dc:dc+W]=torch.tensor(g==c,dtype=torch.float32)
    t[10,dr:dr+H,dc:dc+W]=1.0                          # mascara de validade
    return t
def target_tensor(g,dr=0,dc=0):
    H,W=g.shape; y=torch.full((MAXSZ,MAXSZ),-100,dtype=torch.long)  # -100 = ignore
    y[dr:dr+H,dc:dc+W]=torch.tensor(g,dtype=torch.long); return y

# ---------- modelo ----------
class Trans(nn.Module):
    def __init__(s,ch=64):
        super().__init__()
        s.inp=nn.Conv2d(11,ch,3,padding=1)
        s.blocks=nn.ModuleList([nn.Conv2d(ch,ch,3,padding=1) for _ in range(5)])
        s.head=nn.Conv2d(ch,NCOL,1)
    def forward(s,x):
        h=F.relu(s.inp(x))
        for b in s.blocks: h=F.relu(h+b(h))
        return s.head(h)

def same_shape_task(train):
    return all(i.shape==o.shape for i,o in train)

def solve_transducer(train, test_inputs, steps=500, seed=0):
    """treina TTT nos demos (so augmentation diedral, segura p/ regras de cor); AIRV no teste.
    retorna lista de grades previstas (uma por test_input) ou None se abstem."""
    if not same_shape_task(train) or len(train)==0: return None
    torch.manual_seed(seed)
    model=Trans().to(DEVICE); opt=torch.optim.Adam(model.parameters(),3e-3)
    # conjunto aumentado: simetrias diedrais (geometria) x TRANSLACOES aleatorias no canvas
    # (sem permutar cor; sem canais posicionais => forca regra LOCAL invariante a posicao)
    rng=np.random.RandomState(seed); X,Y=[],[]
    for i,o in train:
        for k in range(8):
            di=np.ascontiguousarray(dihedral(i,k)); do=np.ascontiguousarray(dihedral(o,k))
            H,W=di.shape
            offs=[(0,0)]+[(rng.randint(0,MAXSZ-H+1),rng.randint(0,MAXSZ-W+1)) for _ in range(3)]
            for dr,dc in offs:
                X.append(to_tensor(di,dr,dc)); Y.append(target_tensor(do,dr,dc))
    Xb=torch.stack(X).to(DEVICE); Yb=torch.stack(Y).to(DEVICE)
    model.train()
    for st in range(steps):
        idx=torch.randint(0,Xb.shape[0],(min(64,Xb.shape[0]),),device=DEVICE)
        logit=model(Xb[idx]); loss=F.cross_entropy(logit,Yb[idx],ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    # gate de confianca: reproduz TODOS os demos exatamente (predicao simples)?
    with torch.no_grad():
        for i,o in train:
            if not np.array_equal(_predict_plain(model,i),o): return None
        outs=[_predict_airv(model,xi) for xi in test_inputs]
    return outs

def _predict_plain(model,g):
    H,W=g.shape
    with torch.no_grad():
        logit=model(to_tensor(g).unsqueeze(0).to(DEVICE))[0]
        return logit[:, :H,:W].argmax(0).cpu().numpy().astype(np.int64)

def _predict_airv(model,g):
    """AIRV: prediz sobre 8 simetrias, reverte, vota por celula. Saida cortada ao shape de g."""
    H,W=g.shape; votes=np.zeros((H,W,NCOL),dtype=np.float32)
    with torch.no_grad():
        for k in range(8):
            gi=np.ascontiguousarray(dihedral(g,k)); h,w=gi.shape
            logit=model(to_tensor(gi).unsqueeze(0).to(DEVICE))[0]      # (10,MAXSZ,MAXSZ)
            prob=F.softmax(logit,0).cpu().numpy()[:, :h,:w]            # (10,h,w)
            pmap=np.transpose(prob,(1,2,0))                            # (h,w,10)
            back=dihedral_inv(pmap,k)                                  # reverte a simetria
            if back.shape[:2]==(H,W): votes+=back
    return votes.argmax(2).astype(np.int64)

def main():
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    solved=0; attempted=0; t0=time.time(); ex=[]
    for n,fp in enumerate(files):
        t=json.load(open(fp))
        train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not same_shape_task(train): continue
        attempted+=1
        outs=solve_transducer(train,[xi for xi,_ in tests])
        if outs is None: continue
        ok=all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests))
        if ok: solved+=1; ex.append(os.path.basename(fp))
    dt=time.time()-t0; vram=torch.cuda.max_memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print("="*64)
    print(f"H56-T Transdutor neural por-puzzle (TTT+AIRV) em {DEVICE}")
    print(f"  tarefas mesmo-shape tentadas: {attempted}/400")
    print(f"  RESOLVIDAS (transdutor sozinho): {solved}/400 = {100*solved/400:.1f}%")
    print(f"  tempo {dt:.0f}s | VRAM pico {vram:.0f}MB")
    print("  exemplos:", ex[:20])
    json.dump(ex,open(os.path.join(os.path.dirname(__file__),"h56_transducer_solved.json"),"w"))

if __name__=="__main__": main()
