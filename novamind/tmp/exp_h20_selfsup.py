"""
H20 — Autonomous structure discovery (self-supervised, NO labels)
=================================================================

The H7 wall was about SUPERVISED acquisition of a never-labeled primitive. The
truly AUTONOMOUS version: can the system DISCOVER structure in UNLABELED data?
We train a SimCLR-style contrastive encoder where the only signal is "two
augmentations of the same image are the same" -- no class labels ever.

Then we ask: does the learned embedding space organize itself by a real factor?
We run k-means and measure cluster purity w.r.t. COLOR / SHAPE / SIZE.

PRE-REGISTERED expectation (honest): the encoder will discover the most SALIENT
factor (likely color), not necessarily the one we'd want (shape). That itself is
the key insight about autonomy: an unsupervised system discovers what is
statistically salient, not what you asked for. Reported honestly either way.
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle","cross"]; SIZES=[0.30,0.42,0.54]
C,S,Z=4,4,3; G=18; D=64
ALL=list(itertools.product(range(C),range(S),range(Z)))

def shape_mask(shape,r):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; rr=G*r
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=rr*rr: m[y,x]=1
            elif shape=="square" and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="triangle" and -rr<=dy<=rr and abs(dx)<=(dy+rr)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=rr*0.32 or abs(dx)<=rr*0.32) and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
    return m

def render(c,s,z,rng):
    m=shape_mask(SHAPES[s],SIZES[z])
    return np.stack([m*ch for ch in COLOR_RGB[c]],0).astype(np.float32)

def augment(img,rng):
    """Augmentations that PRESERVE identity: noise, brightness, small shift."""
    x=img+rng.standard_normal(img.shape).astype(np.float32)*0.12
    x=x*(1+0.2*(rng.rand()-0.5))                       # brightness jitter
    sh=rng.randint(-2,3,size=2)                         # small translation
    x=np.roll(x,sh,axis=(1,2))
    return x

class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.b=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
                          nn.Flatten(),nn.Linear(64*16,D))
        s.proj=nn.Sequential(nn.Linear(D,D),nn.ReLU(),nn.Linear(D,D))
    def feat(s,x): return s.b(x)                        # representation
    def forward(s,x): return F.normalize(s.proj(s.b(x)),dim=1)  # projection for contrastive

def nt_xent(z1,z2,tau=0.2):
    N=z1.shape[0]; z=torch.cat([z1,z2],0)               # 2N
    sim=z@z.t()/tau
    sim.fill_diagonal_(-1e9)
    targets=torch.cat([torch.arange(N,2*N),torch.arange(0,N)]).to(DEVICE)
    return F.cross_entropy(sim,targets)

def purity(labels_pred, labels_true, k):
    tot=0
    for c in range(k):
        members=labels_true[labels_pred==c]
        if len(members)==0: continue
        tot+=np.bincount(members,minlength=max(labels_true)+1).max()
    return tot/len(labels_true)

def kmeans(X,k,iters=50,seed=0):
    rng=np.random.RandomState(seed); cen=X[rng.choice(len(X),k,replace=False)]
    for _ in range(iters):
        d=((X[:,None,:]-cen[None])**2).sum(-1); a=d.argmin(1)
        for c in range(k):
            if (a==c).any(): cen[c]=X[a==c].mean(0)
    return a

# ---- self-supervised training (NO labels) ----
rng=np.random.RandomState(0)
base=np.array([render(c,s,z,rng) for (c,s,z) in ALL])   # 48 base images
enc=Enc().to(DEVICE); opt=torch.optim.Adam(enc.parameters(),2e-3)
for step in range(1500):
    idx=rng.choice(len(ALL),32)
    v1=torch.tensor(np.array([augment(base[i],rng) for i in idx])).to(DEVICE)
    v2=torch.tensor(np.array([augment(base[i],rng) for i in idx])).to(DEVICE)
    loss=nt_xent(enc(v1),enc(v2))
    opt.zero_grad(); loss.backward(); opt.step()

# ---- evaluate what was discovered (no labels used in training) ----
enc.eval()
with torch.no_grad():
    feats=[]; cols=[]; shp=[]; siz=[]
    for (c,s,z) in ALL:
        for _ in range(20):
            im=torch.tensor(augment(render(c,s,z,rng),rng)[None]).to(DEVICE)
            feats.append(enc.feat(im).cpu().numpy()[0]); cols.append(c); shp.append(s); siz.append(z)
    X=np.array(feats); cols=np.array(cols); shp=np.array(shp); siz=np.array(siz)
X=(X-X.mean(0))/(X.std(0)+1e-6)

print(f"Device: {DEVICE} | self-supervised, NO labels | what did it discover?")
print("-"*56)
for name,lab,k in [("COLOR",cols,4),("SHAPE",shp,4),("SIZE",siz,3)]:
    p=np.mean([purity(kmeans(X,k,seed=s),lab,k) for s in range(3)])
    print(f"  cluster purity vs {name:6s} (k={k}): {p:.0%}   (chance ~{100/k:.0f}%)")
# random-encoder control
enc2=Enc().to(DEVICE); enc2.eval()
with torch.no_grad():
    Xr=np.array([enc2.feat(torch.tensor(augment(render(*ALL[i],rng),rng)[None]).to(DEVICE)).cpu().numpy()[0]
                 for i in range(len(ALL)) for _ in range(20)])
Xr=(Xr-Xr.mean(0))/(Xr.std(0)+1e-6)
pr=np.mean([purity(kmeans(Xr,4,seed=s),cols,4) for s in range(3)])
print(f"\n  CONTROL (random untrained encoder) vs COLOR: {pr:.0%}")
print("Honest read: self-sup discovers the SALIENT factor(s) with no labels; which one is the finding.")
