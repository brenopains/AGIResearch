"""
H73 — NCA multi-seed: estabiliza/amplia o NCA (que e estocastico) por uniao+voto de seeds
==========================================================================================
O NCA (H70) e estocastico: cada seed resolve um subconjunto diferente. Rodamos K seeds; por
tarefa, entre os seeds cujo NCA REPRODUZ os demos exato, VOTA-se a predicao do teste (sem usar
rotulo do teste p/ selecionar). A uniao multi-seed e legitima e estabiliza o numero. GPU.
"""
import os, json, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DATA=os.path.join(os.path.dirname(__file__),"arc_data","data","training")
SZ=30; NCOL=10; HID=8; CH=NCOL+HID; KSEEDS=3
def dih(g,k): return [g,np.rot90(g,1),np.rot90(g,2),np.rot90(g,3),g[:,::-1],g[::-1,:],np.swapaxes(g,0,1),np.swapaxes(g[:,::-1],0,1)][k]
def to_state(g):
    H,W=g.shape; s=torch.zeros(CH,SZ,SZ)
    for c in range(NCOL): s[c,:H,:W]=torch.tensor(g==c,dtype=torch.float32)
    return s
def to_tgt(g):
    H,W=g.shape; y=torch.full((SZ,SZ),-100,dtype=torch.long); y[:H,:W]=torch.tensor(g,dtype=torch.long); return y
class NCA(nn.Module):
    def __init__(s,h=96):
        super().__init__(); s.p=nn.Conv2d(CH,h,3,padding=1); s.r=nn.Conv2d(h,CH,1)
        nn.init.zeros_(s.r.weight); nn.init.zeros_(s.r.bias)
    def forward(s,st,T):
        for _ in range(T):
            dx=s.r(F.relu(s.p(st))); m=(torch.rand_like(st[:, :1])<0.85).float(); st=st+dx*m
        return st
def solve(train,tests,seed,T=12,steps=200):
    if not all(i.shape==o.shape for i,o in train): return None
    torch.manual_seed(seed)
    X=[];Y=[]
    for i,o in train:
        for k in range(8): X.append(to_state(np.ascontiguousarray(dih(i,k)))); Y.append(to_tgt(np.ascontiguousarray(dih(o,k))))
    X=torch.stack(X).to(DEVICE);Y=torch.stack(Y).to(DEVICE)
    net=NCA().to(DEVICE); opt=torch.optim.Adam(net.parameters(),2e-3); net.train()
    for _ in range(steps):
        out=net(X.clone(),T); loss=F.cross_entropy(out[:, :NCOL],Y,ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def pr(g):
        H,W=g.shape
        with torch.no_grad(): return net(to_state(g).unsqueeze(0).to(DEVICE),T)[0,:NCOL,:H,:W].argmax(0).cpu().numpy().astype(np.int64)
    for i,o in train:
        if not np.array_equal(pr(i),o): return None
    return [pr(xi) for xi,_ in tests]
def main():
    t0=time.time()
    base=set(json.load(open(os.path.join(os.path.dirname(__file__),"h66_base_solved.json"))))
    files=sorted(glob.glob(os.path.join(DATA,"*.json"))); nca=set()
    for fp in files:
        t=json.load(open(fp)); train=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
        tests=[(np.array(p["input"]),np.array(p["output"])) for p in t["test"]]
        if not all(i.shape==o.shape for i,o in train): continue
        cand=[]
        for sd in range(KSEEDS):
            o=solve(train,tests,sd)
            if o is not None: cand.append(o)
        if not cand: continue
        voted=[stats.mode(np.stack([c[ti] for c in cand]),axis=0,keepdims=False).mode for ti in range(len(tests))]
        if all(v.shape==yo.shape and np.array_equal(v,yo) for v,(_,yo) in zip(voted,tests)): nca.add(os.path.basename(fp))
    json.dump(sorted(nca),open(os.path.join(os.path.dirname(__file__),"h73_ncams_solved.json"),"w"))
    print("="*70)
    print(f"H73 NCA multi-seed ({KSEEDS} seeds, voto) ({DEVICE})")
    print(f"  NCA-multiseed resolveu: {len(nca)}/400 | NOVAS vs base: {len(nca-base)}")
    print(f"  UNIAO (base + NCA-ms): {len(base|nca)}/400 = {100*len(base|nca)/400:.1f}%")
    print(f"  tempo {time.time()-t0:.0f}s")
if __name__=="__main__": main()
