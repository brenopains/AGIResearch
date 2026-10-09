"""
H72 — Ensemble de Reservatórios (RC multi-seed/raio) + voto entre reservatorios consistentes
=============================================================================================
Vários reservatorios (seeds x raios espectrais). Por tarefa: entre os reservatorios cujo ridge
REPRODUZ os demos exato, VOTA-se a predicao do teste (sem usar rotulo do teste p/ selecionar —
voto so sobre predicoes; honesto). Une com base+NCA(H70). RC e barato (ridge fechado).
"""
import os, json, glob, time, numpy as np, torch, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
NCOL=11; HID=128; T=8
CONFIGS=[(s,rho) for s in range(4) for rho in (0.8,0.95,1.1)]   # 12 reservatorios

def make_res(seed,rho):
    g=torch.Generator(device=DEVICE).manual_seed(seed)
    Wres=torch.randn(HID,HID,3,3,generator=g,device=DEVICE)*0.1
    with torch.no_grad():
        sc=Wres.flatten(1).svd().S.max(); Wres*=(rho/sc)
    Win=torch.randn(HID,NCOL,3,3,generator=g,device=DEVICE)*0.5
    return Wres,Win

def feats(g,Wres,Win):
    H,W=g.shape; x=torch.zeros(1,NCOL,H,W,device=DEVICE)
    for c in range(NCOL-1): x[0,c]=torch.tensor(g==c,dtype=torch.float32,device=DEVICE)
    drive=F.conv2d(x,Win,padding=1); r=torch.zeros(1,HID,H,W,device=DEVICE); fs=[x[0].reshape(NCOL,-1).t()]
    for t in range(T):
        r=torch.tanh(F.conv2d(r,Wres,padding=1)+drive)
        if t in (1,3,T-1): fs.append(r[0].reshape(HID,-1).t())
    return torch.cat(fs,1)

def ridge(Phi,Y,lam=1.0):
    D=Phi.shape[1]; return torch.linalg.solve(Phi.t()@Phi+lam*torch.eye(D,device=DEVICE),Phi.t()@Y)

def solve_one(train,tests,Wres,Win):
    Phi=[]; Y=[]
    for i,o in train:
        Phi.append(feats(i,Wres,Win)); oh=torch.zeros(o.size,10,device=DEVICE)
        oh[torch.arange(o.size),torch.tensor(o.ravel(),device=DEVICE,dtype=torch.long)]=1; Y.append(oh)
    W=ridge(torch.cat(Phi),torch.cat(Y))
    def pred(g):
        H,Wd=g.shape; return (feats(g,Wres,Win)@W).argmax(1).cpu().numpy().reshape(H,Wd).astype(np.int64)
    for i,o in train:
        if not np.array_equal(pred(i),o): return None
    return [pred(xi) for xi,_ in tests]

def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    nca=set(json.load(open(os.path.join(os.path.dirname(__file__),"h70_nca_solved.json")))) if os.path.exists(os.path.join(os.path.dirname(__file__),"h70_nca_solved.json")) else set()
    prev=base|nca
    res=[make_res(s,rho) for s,rho in CONFIGS]
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); rc=set()
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        # coleta predicoes dos reservatorios que reproduzem os demos
        cand=[]
        for Wres,Win in res:
            o=solve_one(train,tests,Wres,Win)
            if o is not None: cand.append(o)
        if not cand: continue
        # voto por celula entre reservatorios consistentes
        voted=[]
        for ti in range(len(tests)):
            stack=np.stack([c[ti] for c in cand])  # (R,H,W)
            from scipy import stats
            voted.append(stats.mode(stack,axis=0,keepdims=False).mode)
        if all(v.shape==yo.shape and np.array_equal(v,yo) for v,(_,yo) in zip(voted,tests)): rc.add(os.path.basename(fp))
    json.dump(sorted(rc),open(os.path.join(os.path.dirname(__file__),"h72_rcens_solved.json"),"w"))
    print("="*70)
    print(f"H72 Ensemble de Reservatorios ({len(CONFIGS)} reservatorios; voto) ({DEVICE})")
    print(f"  RC-ensemble resolveu: {len(rc)}/400 | NOVAS vs base+NCA: {len(rc-prev)} {sorted(rc-prev)[:12]}")
    print(f"  UNIAO (base + NCA + RC-ens): {len(prev|rc)}/400 = {100*len(prev|rc)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s | Honesto: ensemble de bases aleatorias; nao quebra o fecho.")

if __name__=="__main__": main()
