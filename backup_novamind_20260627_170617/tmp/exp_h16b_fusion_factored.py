"""
H16b — Does multimodality help? FIXED with FACTORED anchors
===========================================================

H16 had a flaw: per-concept anchors meant held-out concepts had untrained
(random) anchors -> held-out test invalid (0%). Fix, using our own H2 lesson:
make the anchor FACTORED:  anchor(c,s,z) = norm( E_color[c] + E_shape[s] + E_size[z] ).
Now EVERY concept (incl. unseen) has a valid anchor composed from attribute
embeddings (each attribute value is seen in training), so compositional
generalization is testable.

Modalities stay complementary (disclosed):
  IMAGE : color + shape only (all sizes render identically)
  AUDIO : size only (frequency)

Readouts: image-only | audio-only | FUSED, on SEEN and HELD-OUT combos.
PRE-REGISTERED: fused > max(image-only, audio-only).
"""
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle","cross"]
C,S,Z=4,4,3; G=18; AUD_BINS=64; D=128; FIXED_R=0.40

def shape_mask(shape,r):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; rr=G*r
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=rr*rr: m[y,x]=1
            elif shape=="square" and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="triangle" and -rr<=dy<=rr and abs(dx)<=(dy+rr)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=rr*0.32 or abs(dx)<=rr*0.32): m[y,x]=1
    return m
MASKS={s:shape_mask(s,FIXED_R) for s in SHAPES}
def render_image(c,s,z,rng,noise=0.06):
    m=MASKS[SHAPES[s]]
    return np.stack([m*ch for ch in COLOR_RGB[c]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise
def render_audio(c,s,z,rng):
    f=180+z*180; f*=(1+0.03*rng.standard_normal())
    t=np.linspace(0,1,1024,endpoint=False)
    wave=np.sin(2*np.pi*f*t)+0.4*np.sin(2*np.pi*2*f*t)+0.2*rng.standard_normal(1024)
    mag=np.abs(np.fft.rfft(wave))[:AUD_BINS]; return (np.log1p(mag).astype(np.float32))/ \
        (np.linalg.norm(np.log1p(mag))+1e-6)

class ImgEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(32*16,D))
    def forward(s,x): return F.normalize(s.c(x),dim=1)
class AudEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.m=nn.Sequential(nn.Linear(AUD_BINS,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,x): return F.normalize(s.m(x),dim=1)

ALL=list(itertools.product(range(C),range(S),range(Z)))
CS=torch.tensor([t[0] for t in ALL]); SS=torch.tensor([t[1] for t in ALL]); ZS=torch.tensor([t[2] for t in ALL])
def cid(t): return ALL.index(t)

def run(seed, steps=1500, per_cell=8, pool=64):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+9)
    held=set()
    for i in rng.permutation(len(ALL)):
        if len(held)>=10: break
        held.add(ALL[i])
    TRAIN=[t for t in ALL if t not in held]; TEST_HO=[t for t in ALL if t in held]
    def cache(combos):
        I=np.array([[render_image(*t,rng) for _ in range(pool)] for t in combos])
        A=np.array([[render_audio(*t,rng) for _ in range(pool)] for t in combos])
        return torch.tensor(I).to(DEVICE),torch.tensor(A).to(DEVICE)
    Itr,Atr=cache(TRAIN)
    ie,ae=ImgEnc().to(DEVICE),AudEnc().to(DEVICE)
    Ecol=nn.Parameter(torch.randn(C,D).to(DEVICE)*0.3)
    Esha=nn.Parameter(torch.randn(S,D).to(DEVICE)*0.3)
    Esiz=nn.Parameter(torch.randn(Z,D).to(DEVICE)*0.3)
    csd,ssd,zsd=CS.to(DEVICE),SS.to(DEVICE),ZS.to(DEVICE)
    def anchors():  # (48,D) factored, normalized
        return F.normalize(Ecol[csd]+Esha[ssd]+Esiz[zsd],dim=1)
    opt=torch.optim.Adam(list(ie.parameters())+list(ae.parameters())+[Ecol,Esha,Esiz],1.5e-3)
    tr_ids=torch.tensor([cid(t) for t in TRAIN],device=DEVICE).repeat_interleave(per_cell)
    base=torch.arange(len(TRAIN),device=DEVICE).repeat_interleave(per_cell)
    for _ in range(steps):
        idx=torch.randint(0,pool,(len(TRAIN)*per_cell,),device=DEVICE)
        anc=anchors()
        zi=ie(Itr[base,idx]); za=ae(Atr[base,idx])
        loss=F.cross_entropy(zi@anc.t()/0.07,tr_ids)+F.cross_entropy(za@anc.t()/0.07,tr_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); ae.eval(); anc=anchors().detach()
    def evalset(combos):
        ci=[];ca=[];cf=[]
        with torch.no_grad():
            for t in combos:
                I=torch.tensor(np.array([render_image(*t,rng) for _ in range(40)])).to(DEVICE)
                A=torch.tensor(np.array([render_audio(*t,rng) for _ in range(40)])).to(DEVICE)
                si=ie(I)@anc.t(); sa=ae(A)@anc.t(); sf=si+sa; g=cid(t)
                ci.append((si.argmax(1).cpu().numpy()==g).mean())
                ca.append((sa.argmax(1).cpu().numpy()==g).mean())
                cf.append((sf.argmax(1).cpu().numpy()==g).mean())
        return np.mean(ci),np.mean(ca),np.mean(cf)
    return evalset(TRAIN),evalset(TEST_HO)

print(f"Device: {DEVICE} | 48 concepts, FACTORED anchors | chance {100/len(ALL):.1f}%")
seen=[];ho=[]
for s in range(5):
    a,b=run(s); seen.append(a); ho.append(b)
seen=np.array(seen); ho=np.array(ho)
def fmt(a,l): print(f"  {l:18s} image-only {a[:,0].mean():.0%}  audio-only {a[:,1].mean():.0%}  FUSED {a[:,2].mean():.0%}")
print("-"*60); fmt(seen,"SEEN combos:"); fmt(ho,"HELD-OUT combos:")
bs_seen=max(seen[:,0].mean(),seen[:,1].mean()); bs_ho=max(ho[:,0].mean(),ho[:,1].mean())
print(f"\nSEEN:     fused {seen[:,2].mean():.0%} vs best single {bs_seen:.0%} -> {'HELPS' if seen[:,2].mean()-bs_seen>0.15 else 'no clear benefit'}")
print(f"HELD-OUT: fused {ho[:,2].mean():.0%} vs best single {bs_ho:.0%} -> {'HELPS' if ho[:,2].mean()-bs_ho>0.15 else 'no clear benefit'}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU")
