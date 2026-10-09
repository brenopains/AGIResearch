"""
H70 Fase 1 — Neural Cellular Automata (NCA) per-puzzle para o ARC
=================================================================
Aposta arquitetural validada (Xu 2025 "NCA for ARC-AGI"; Guichard "ARC-NCA"; Hartl 2025):
aprende uma REGRA DE ATUALIZACAO LOCAL ITERATIVA (morfogenese) nos demos da tarefa e aplica
no teste. Captura tarefas ESPACIAIS/DESENVOLVIMENTAIS (preencher, propagar, crescer, contornos)
que o pipeline simbolico+MB nao pega. Equivariancia local por construcao (conv) — sem o
problema de memorizar posicao do H56. Treino-zero de pre-treino; so TTT por puzzle. GPU.

V0: so tarefas mesmo-shape (atualizacao celula-a-celula bem definida). Honesto: nao sintetiza
programa global; mede com dado quanto adiciona alem dos 15,0%. Nao e AGI.
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
SZ=30; NCOL=10; HID=8; CH=NCOL+HID

def dihedral(g,k):
    return [g,np.rot90(g,1),np.rot90(g,2),np.rot90(g,3),g[:,::-1],g[::-1,:],np.swapaxes(g,0,1),np.swapaxes(g[:,::-1],0,1)][k]

def to_state(g):
    H,W=g.shape; s=torch.zeros(CH,SZ,SZ)
    for c in range(NCOL): s[c,:H,:W]=torch.tensor(g==c,dtype=torch.float32)
    return s
def to_target(g):
    H,W=g.shape; y=torch.full((SZ,SZ),-100,dtype=torch.long); y[:H,:W]=torch.tensor(g,dtype=torch.long); return y

class NCA(nn.Module):
    def __init__(s,hidden=96):
        super().__init__()
        s.perceive=nn.Conv2d(CH,hidden,3,padding=1)
        s.rule=nn.Conv2d(hidden,CH,1); nn.init.zeros_(s.rule.weight); nn.init.zeros_(s.rule.bias)  # delta~0 no inicio
    def forward(s,state,T):
        for _ in range(T):
            dx=s.rule(F.relu(s.perceive(state)))
            mask=(torch.rand_like(state[:, :1])<0.85).float()   # update estocastico (Mordvintsev)
            state=state+dx*mask
        return state

def solve_nca(train, tests, T=12, steps=220):
    if not all(i.shape==o.shape for i,o in train) or len(train)==0: return None
    X=[]; Y=[]
    for i,o in train:
        for k in range(8):
            X.append(to_state(np.ascontiguousarray(dihedral(i,k)))); Y.append(to_target(np.ascontiguousarray(dihedral(o,k))))
    X=torch.stack(X).to(DEVICE); Y=torch.stack(Y).to(DEVICE)
    net=NCA().to(DEVICE); opt=torch.optim.Adam(net.parameters(),2e-3)
    net.train()
    for _ in range(steps):
        out=net(X.clone(),T); loss=F.cross_entropy(out[:, :NCOL],Y,ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def predict(g):
        H,W=g.shape
        with torch.no_grad():
            out=net(to_state(g).unsqueeze(0).to(DEVICE),T)
            return out[0,:NCOL,:H,:W].argmax(0).cpu().numpy().astype(np.int64)
    # gate: reproduz os demos exato?
    for i,o in train:
        if not np.array_equal(predict(i),o): return None
    return [predict(xi) for xi,_ in tests]

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    files=sorted(glob.glob(os.path.join(DATA,"*.json")))
    nca_solved=set(); attempted=0
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        attempted+=1
        outs=solve_nca(train,tests)
        if outs is None: continue
        if all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests)):
            nca_solved.add(os.path.basename(fp))
    json.dump(sorted(nca_solved),open(os.path.join(os.path.dirname(__file__),"h70_nca_solved.json"),"w"))
    novas=nca_solved-base; vram=torch.cuda.max_memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print("="*70)
    print(f"H70 Fase 1 — NCA per-puzzle ({DEVICE})")
    print(f"  tarefas mesmo-shape tentadas: {attempted}/400")
    print(f"  NCA resolveu (teste exato): {len(nca_solved)}/400")
    print(f"  NOVAS vs base simbolico+MB (15,0%): {len(novas)} {sorted(novas)[:15]}")
    print(f"  UNIAO (base + NCA): {len(base|nca_solved)}/400 = {100*len(base|nca_solved)/400:.1f}%   (era 15,0%)")
    print(f"  tempo {time.time()-t0:.0f}s | VRAM {vram:.0f}MB")
    print(f"  Honesto: NCA capta regras locais iterativas (propagacao/crescimento); nao sintetiza")
    print(f"  programa global; mede quanto disso o pipeline atual nao cobria. Nao e AGI.")

if __name__=="__main__": main()
