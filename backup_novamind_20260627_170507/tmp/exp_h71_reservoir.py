"""
H71 — Reservoir Computing (Echo State espacial) para o ARC: o irmao recorrente do MB
=====================================================================================
Proposta do usuario: um NUCLEO CAOTICO (reservatorio recorrente fixo aleatorio) + leitura
linear treinada. Implementacao fiel e eficiente: reservatorio CONVOLUCIONAL fixo (pesos
aleatorios, raio espectral ~edge-of-chaos), excitado pela grade por T passos; leitura RIDGE
(forma fechada, SEM loop de treino) por-celula -> cor de saida.

Honesto (matematica estabelecida): RC = mapa de features ALEATORIAS fixas + leitura linear =
metodo de kernel. NAO quebra o fecho/Solomonoff; e uma base fixa enorme/temporal. E o primo
RECORRENTE do MB (H68, projecao aleatoria feedforward). Previsao: pega tarefas iterativas/locais,
provavel sobreposicao com NCA(H70)/MB(H68); mede com dado quanto ADICIONA aos ~18%. GPU, rapido.
So tarefas mesmo-shape. Nao e AGI.
"""
import os, json, glob, time, numpy as np, torch
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
NCOL=11; HID=128; T=8
# reservatorio FIXO: conv 3x3 recorrente (pesos aleatorios) + projecao de entrada
Wres=torch.randn(HID,HID,3,3,device=DEVICE)*0.1
# normaliza p/ ~edge-of-chaos (escala pelo norm espectral aproximado)
with torch.no_grad():
    sc=Wres.flatten(1).svd().S.max(); Wres*= (0.95/sc)
Win=torch.randn(HID,NCOL,3,3,device=DEVICE)*0.5

import torch.nn.functional as F
def reservoir_features(g):
    """retorna (H*W, D): features do reservatorio por celula (concat input + estados em alguns T)."""
    H,W=g.shape
    x=torch.zeros(1,NCOL,H,W,device=DEVICE)
    for c in range(NCOL-1):
        x[0,c]=torch.tensor(g==c,dtype=torch.float32,device=DEVICE)
    drive=F.conv2d(x,Win,padding=1)              # excitacao de entrada (fixa)
    r=torch.zeros(1,HID,H,W,device=DEVICE); feats=[x[0].reshape(NCOL,-1).t()]
    for t in range(T):
        r=torch.tanh(F.conv2d(r,Wres,padding=1)+drive)   # dinamica recorrente fixa
        if t in (1,3,T-1): feats.append(r[0].reshape(HID,-1).t())
    return torch.cat(feats,dim=1)                # (H*W, NCOL+3*HID)

def ridge_fit(Phi,Y,lam=1.0):
    D=Phi.shape[1]; A=Phi.t()@Phi+lam*torch.eye(D,device=DEVICE)
    return torch.linalg.solve(A,Phi.t()@Y)       # (D,10) forma fechada

def solve_rc(train, tests):
    if not all(i.shape==o.shape for i,o in train) or len(train)==0: return None
    Phi=[]; Y=[]
    for i,o in train:
        Phi.append(reservoir_features(i))
        oh=torch.zeros(o.size,10,device=DEVICE); oh[torch.arange(o.size),torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long)]=1
        Y.append(oh)
    Phi=torch.cat(Phi); Y=torch.cat(Y)
    Wout=ridge_fit(Phi,Y)
    def predict(g):
        H,W=g.shape; return (reservoir_features(g)@Wout).argmax(1).cpu().numpy().reshape(H,W).astype(np.int64)
    for i,o in train:
        if not np.array_equal(predict(i),o): return None
    return [predict(xi) for xi,_ in tests]

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    nca=set(json.load(open(os.path.join(os.path.dirname(__file__),"h70_nca_solved.json")))) if os.path.exists(os.path.join(os.path.dirname(__file__),"h70_nca_solved.json")) else set()
    prev=base|nca
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    rc=set(); attempted=0
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        attempted+=1
        outs=solve_rc(train,tests)
        if outs is None: continue
        if all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests)): rc.add(os.path.basename(fp))
    json.dump(sorted(rc),open(os.path.join(os.path.dirname(__file__),"h71_rc_solved.json"),"w"))
    vram=torch.cuda.max_memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print("="*70)
    print(f"H71 Reservoir Computing (Echo State espacial) ({DEVICE})")
    print(f"  mesmo-shape tentadas: {attempted}/400 | RC resolveu: {len(rc)}/400")
    print(f"  NOVAS vs base simbolico (15,0%): {len(rc-base)} {sorted(rc-base)[:12]}")
    print(f"  NOVAS vs base+NCA(~18%): {len(rc-prev)} {sorted(rc-prev)[:12]}")
    print(f"  UNIAO (base + NCA + RC): {len(prev|rc)}/400 = {100*len(prev|rc)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {vram:.0f}MB (reservatorio FIXO + ridge fechado = rapido)")
    print(f"  Honesto: RC = features aleatorias fixas + leitura linear (kernel). Adiciona onde a")
    print(f"  dinamica casa; NAO quebra o fecho. Primo recorrente do MB. Nao e AGI.")

if __name__=="__main__": main()
