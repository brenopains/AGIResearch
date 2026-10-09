"""
H74 — Extensão tamanho-variável: RC com leitura GLOBAL para saída de tamanho constante
=======================================================================================
Os transdutores (MB/NCA/RC) so faziam mesmo-shape (correspondencia celula-a-celula). Aqui
atacamos a classe tamanho-variavel mais tratavel: tarefas cujo OUTPUT tem SHAPE CONSTANTE em
todos os demos (extrair/sumarizar). Reservatorio processa o input -> features GLOBAIS (mean+max
pooled) -> ridge -> grade de saida fixa (flattened one-hot). Honesto: pouquissimos demos p/ um
output grande -> tende a overfit (gate de demos passa trivial; o teste e o filtro real) ->
expectativa BAIXA. Mede com dado. So conta o que bate no teste held-out e e novo. Nao e AGI.
"""
import os, json, glob, time, numpy as np, torch, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
NCOL=11; HID=128; T=8
g_=torch.Generator(device=DEVICE).manual_seed(0)
Wres=torch.randn(HID,HID,3,3,generator=g_,device=DEVICE)*0.1
with torch.no_grad(): Wres*=(0.95/Wres.flatten(1).svd().S.max())
Win=torch.randn(HID,NCOL,3,3,generator=g_,device=DEVICE)*0.5
def gfeat(g):
    H,W=g.shape; x=torch.zeros(1,NCOL,H,W,device=DEVICE)
    for c in range(NCOL-1): x[0,c]=torch.tensor(g==c,dtype=torch.float32,device=DEVICE)
    drive=F.conv2d(x,Win,padding=1); r=torch.zeros(1,HID,H,W,device=DEVICE); pools=[]
    for t in range(T):
        r=torch.tanh(F.conv2d(r,Wres,padding=1)+drive)
    # features globais: mean + max sobre as celulas + histograma de cores
    gm=r.mean((2,3)).flatten(); gx=r.amax((2,3)).flatten()
    hist=x.mean((2,3)).flatten()
    return torch.cat([gm,gx,hist])
def solve(train,tests):
    shp={o.shape for _,o in train}
    if len(shp)!=1: return None                  # output shape constante
    Ho,Wo=shp.pop()
    if any(i.shape==o.shape for i,o in train): return None   # nao mesmo-shape (isso e o H71)
    Phi=torch.stack([gfeat(i) for i,_ in train])
    Y=torch.stack([torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long) for _,o in train])
    # ridge multi-saida: features -> logits por celula (Ho*Wo*10)
    oh=F.one_hot(Y,10).float().reshape(len(train),-1)
    D=Phi.shape[1]; W=torch.linalg.solve(Phi.t()@Phi+5.0*torch.eye(D,device=DEVICE),Phi.t()@oh)
    def pr(g):
        out=(gfeat(g)@W).reshape(Ho*Wo,10); return out.argmax(1).cpu().numpy().reshape(Ho,Wo).astype(np.int64)
    for i,o in train:
        if not np.array_equal(pr(i),o): return None
    return [pr(xi) for xi,_ in tests]
def main():
    t0=time.time()
    prev=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    for f in ("h70_nca_solved.json","h72_rcens_solved.json"):
        p=os.path.join(os.path.dirname(__file__),f)
        if os.path.exists(p): prev|=set(json.load(open(p)))
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); vs=set(); attempted=0
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if all(i.shape==o.shape for i,o in train): continue   # so tamanho-variavel
        attempted+=1
        outs=solve(train,tests)
        if outs is None: continue
        if all(p.shape==yo.shape and np.array_equal(p,yo) for p,(_,yo) in zip(outs,tests)): vs.add(os.path.basename(fp))
    print("="*70)
    print(f"H74 tamanho-variavel (RC leitura global, output constante) ({DEVICE})")
    print(f"  tarefas tamanho-variavel tentadas: {attempted} | resolveu: {len(vs)} {sorted(vs)[:12]}")
    print(f"  NOVAS vs ensemble atual: {len(vs-prev)}")
    print(f"  UNIAO total: {len(prev|vs)}/400 = {100*len(prev|vs)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s | Honesto: poucos demos p/ output grande -> overfit esperado.")
if __name__=="__main__": main()
