"""
H33 — REAL data + a FAIR generalization test (conceding the user's critique)
============================================================================

Prior tests used toy synthetic data, and the L4 'foreign' test was unfair (it
made the answer depend on info the encoders never received -- unsolvable by
construction). Fixed here:

  - REAL data: CIFAR-10 (natural images), via torchvision.
  - FAIR generalization test: train an encoder on 6 BASE classes (supervised).
    Freeze it. Then few-shot transfer (k=1,5,10,50 examples) to 4 HELD-OUT NOVEL
    classes the encoder NEVER trained on -- using a simple nearest-centroid probe
    on the frozen features. The info IS in the pixels; the question is whether the
    learned representation GENERALIZES to recognize categories it never saw.
  - Control: same probe on RANDOM (untrained) features.

This is a legitimate generalization test on real data. Honest reading up front:
strong few-shot transfer = real NARROW generalization (a known, valuable property
of good representations) -- NOT AGI. AGI = open-ended cross-domain generality
(e.g., ARC-AGI level), which no single-dataset transfer test can demonstrate.
"""
import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def load_cifar():
    import torchvision, torchvision.transforms as T
    tf=T.Compose([T.ToTensor()])
    tr=torchvision.datasets.CIFAR10(root="tmp/data",train=True,download=True,transform=tf)
    te=torchvision.datasets.CIFAR10(root="tmp/data",train=False,download=True,transform=tf)
    def to_xy(ds):
        X=torch.tensor(ds.data,dtype=torch.float32).permute(0,3,1,2)/255.0
        y=torch.tensor(ds.targets,dtype=torch.long); return X,y
    return to_xy(tr)+to_xy(te)

class Enc(nn.Module):
    def __init__(s,d=256):
        super().__init__(); s.d=d
        s.c=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),    #16
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2), #8
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),nn.Linear(256*16,d))
    def forward(s,x): return s.c(x)

def main():
    t0=time.time()
    Xtr,ytr,Xte,yte=load_cifar()
    BASE=[0,1,2,3,4,5]; NOVEL=[6,7,8,9]      # train on 6 classes, test transfer to 4 unseen
    print(f"Device: {DEVICE} | CIFAR-10 | base classes {BASE} | NOVEL (held-out) {NOVEL}")
    # ---- train encoder on BASE classes (supervised) ----
    mtr=torch.isin(ytr,torch.tensor(BASE)); Xb,yb=Xtr[mtr].to(DEVICE),ytr[mtr].to(DEVICE)
    remap={c:i for i,c in enumerate(BASE)}; yb=torch.tensor([remap[int(v)] for v in yb.cpu()],device=DEVICE)
    enc=Enc().to(DEVICE); head=nn.Linear(enc.d,len(BASE)).to(DEVICE)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),2e-3); bs=256
    for ep in range(8):
        perm=torch.randperm(len(Xb))
        for i in range(0,len(Xb),bs):
            idx=perm[i:i+bs]; opt.zero_grad(); F.cross_entropy(head(enc(Xb[idx])),yb[idx]).backward(); opt.step()
    enc.eval()
    # ---- few-shot transfer to NOVEL classes (frozen rep, nearest-centroid) ----
    def feats(X):
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),512): out.append(F.normalize(enc(X[i:i+512].to(DEVICE)),1).cpu())
        return torch.cat(out)
    mtr_n=torch.isin(ytr,torch.tensor(NOVEL)); mte_n=torch.isin(yte,torch.tensor(NOVEL))
    Xn_tr,yn_tr=Xtr[mtr_n],ytr[mtr_n]; Xn_te,yn_te=Xte[mte_n],yte[mte_n]
    Fn_tr=feats(Xn_tr); Fn_te=feats(Xn_te)
    rng=np.random.RandomState(0)
    # random-feature control encoder
    enc_r=Enc().to(DEVICE); enc_r.eval()
    def feats_r(X):
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),512): out.append(F.normalize(enc_r(X[i:i+512].to(DEVICE)),1).cpu())
        return torch.cat(out)
    Rn_tr=feats_r(Xn_tr); Rn_te=feats_r(Xn_te)
    def fewshot(Ftr,ytr_,Fte,yte_,k):
        accs=[]
        for trial in range(5):
            cents=[]
            for c in NOVEL:
                idx=np.where(ytr_.numpy()==c)[0]; pick=rng.choice(idx,k,replace=False)
                cents.append(Ftr[pick].mean(0))
            C=F.normalize(torch.stack(cents),1)
            pred=(Fte@C.t()).argmax(1).numpy()
            yt=np.array([NOVEL.index(int(v)) for v in yte_.numpy()])
            accs.append((pred==yt).mean())
        return np.mean(accs),np.std(accs)
    print(f"\nFew-shot transfer to {len(NOVEL)} NOVEL classes (chance {100/len(NOVEL):.0f}%):")
    print(f"{'k/class':>8} {'TRAINED rep':>14} {'RANDOM rep (control)':>22}")
    for k in [1,5,10,50]:
        a,sd=fewshot(Fn_tr,yn_tr,Fn_te,yn_te,k); ar,_=fewshot(Rn_tr,yn_tr,Rn_te,yn_te,k)
        print(f"{k:>8} {a:>12.0%}±{sd:.0%} {ar:>18.0%}")
    vram=torch.cuda.memory_allocated()/1e6 if DEVICE=='cuda' else 0
    print(f"\nVRAM: {vram:.0f} MB | total time: {time.time()-t0:.0f}s")
    print("Reading: TRAINED >> RANDOM and >> chance => the representation GENERALIZES to")
    print("unseen classes (real narrow generalization). That is genuine and valuable -- but")
    print("it is within-domain transfer (images->images), a known property, NOT AGI. AGI =")
    print("open-ended cross-domain generality (ARC-AGI level), which this does not test.")

if __name__=='__main__': main()
