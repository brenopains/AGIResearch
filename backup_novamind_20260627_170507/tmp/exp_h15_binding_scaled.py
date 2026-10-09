"""
H15 — Scaled cross-modal binding with a SHARED ANCHOR (H14 follow-up)
====================================================================

H14 (tiny, pairwise A-T / I-T) failed: 17% ~ chance. Hypothesis for the null:
pairwise alignment doesn't FORCE transitivity, and capacity was tiny. Fix:
  (1) more capacity (embed 256, bigger conv/MLP, more classes, more training);
  (2) better method: a single learnable per-class ANCHOR that BOTH audio and
      image are pulled to (same target point) -> transitivity enforced.
A<->I is STILL never trained, so A->I retrieval remains a genuine zero-shot test.

PRE-REGISTERED: success = A->I well above chance AND >> control (audio-anchor only).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

K=12                       # 4 colors x 3 shapes
COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle"]
G=20; AUD_BINS=96; D=256

def shape_mask(shape):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*0.40
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
    return m
def cls_color(k): return COLOR_RGB[k%4]
def cls_shape(k): return SHAPES[k//4]
MASKS={s:shape_mask(s) for s in SHAPES}

def render_image(k,rng,noise=0.06):
    m=MASKS[cls_shape(k)]
    return np.stack([m*c for c in cls_color(k)],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise
def render_audio(k,rng):
    f=150+k*55; f*=(1+0.03*rng.standard_normal())
    t=np.linspace(0,1,2048,endpoint=False)
    wave=np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*2*f*t)+0.3*np.sin(2*np.pi*3*f*t)+0.2*rng.standard_normal(2048)
    mag=np.abs(np.fft.rfft(wave))[:AUD_BINS]; mag=np.log1p(mag).astype(np.float32)
    return mag/(np.linalg.norm(mag)+1e-6)

class ImgEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),nn.Linear(128*16,D))
    def forward(s,x): return F.normalize(s.c(x),dim=1)
class AudEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.m=nn.Sequential(nn.Linear(AUD_BINS,512),nn.ReLU(),
            nn.Linear(512,512),nn.ReLU(),nn.Linear(512,D))
    def forward(s,x): return F.normalize(s.m(x),dim=1)

def run(seed, train_image=True, steps=2000, per_class=24, pool=256):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+9)
    # ---- pre-render a cache ONCE (huge speedup vs rendering every step) ----
    img_pool=torch.tensor(np.array([[render_image(k,rng) for _ in range(pool)] for k in range(K)])).to(DEVICE)  # (K,pool,3,G,G)
    aud_pool=torch.tensor(np.array([[render_audio(k,rng) for _ in range(pool)] for k in range(K)])).to(DEVICE)  # (K,pool,AUD_BINS)

    ie,ae=ImgEnc().to(DEVICE),AudEnc().to(DEVICE)
    anchor=nn.Parameter(F.normalize(torch.randn(K,D),dim=1).to(DEVICE))  # shared per-class anchor
    opt=torch.optim.Adam(list(ie.parameters())+list(ae.parameters())+[anchor],1.5e-3)
    labels=torch.arange(K,device=DEVICE).repeat_interleave(per_class)
    ar=torch.arange(K,device=DEVICE).repeat_interleave(per_class)
    for _ in range(steps):
        idx=torch.randint(0,pool,(K*per_class,),device=DEVICE)
        A=aud_pool[ar,idx]; anc=F.normalize(anchor,dim=1)
        za=ae(A)
        loss=F.cross_entropy(za@anc.t()/0.07, labels)         # audio -> anchor
        if train_image:
            I=img_pool[ar,idx]; zi=ie(I)
            loss=loss+F.cross_entropy(zi@anc.t()/0.07, labels) # image -> SAME anchor
        opt.zero_grad(); loss.backward(); opt.step()
    ie.eval(); ae.eval()
    with torch.no_grad():
        protos=torch.stack([ie(img_pool[k,:40]).mean(0) for k in range(K)])
        P=F.normalize(protos,dim=1)
        correct=total=0
        for k in range(K):
            pred=(ae(aud_pool[k,:40])@P.t()).argmax(1).cpu().numpy()
            correct+=int((pred==k).sum()); total+=len(pred)
    return correct/total

print(f"Device: {DEVICE} | K={K} classes | embed={D} | chance={100/K:.1f}%")
print(f"{'condition':>26} {'audio->image (mean±std)':>26}")
print("-"*56)
hub=np.array([run(s,train_image=True) for s in range(5)])
ctl=np.array([run(s,train_image=False) for s in range(5)])
print(f"{'SHARED-ANCHOR (A+I)':>26} {hub.mean():>16.0%} ± {hub.std():>3.0%}")
print(f"{'CONTROL (audio only)':>26} {ctl.mean():>16.0%} ± {ctl.std():>3.0%}")
ok = hub.mean()>0.4 and (hub.mean()-ctl.mean())>0.2
print(f"\nPRE-REGISTERED verdict: {'BINDING WORKS' if ok else 'still unclear/null'} "
      f"(hub {hub.mean():.0%} vs control {ctl.mean():.0%}, chance {100/K:.0f}%)")
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU")
