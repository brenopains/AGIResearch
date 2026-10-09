"""
H14 — Transitive cross-modal binding (testing the user's Q6)
============================================================

User's Q6: a concept ('dog') is image + sound + word bound together. Can a tiny
system bind modalities into ONE shared space, and bridge a modality pair it was
NEVER trained on?

Tri-modal, self-contained (no downloads):
  IMAGE : 16x16x3 colored shape (one per class)
  AUDIO : a real waveform (sum of class-specific sinusoids + noise) -> FFT feats
  TEXT  : the class label (one-hot -> embedding) = the linguistic 'hub'

PRE-REGISTERED:
  Hypothesis: train A<->T and I<->T only (text is the hub); A<->I NEVER trained.
              Then A->I retrieval works zero-shot via transitivity through text.
  Metric    : audio->image top-1 over 8 image prototypes. chance = 12.5%.
  Success   : well above chance across 5 seeds.
  Control   : train ONLY A<->T (image never anchored) -> A->I must be ~chance.
              (proves the text hub mediates the binding; no leakage.)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

K=8                      # classes
COLOR_RGB=[(1,0,0),(0,1,0),(0,0,1),(1,1,0)]
SHAPES=["circle","square","triangle","cross"]
G=16; AUD_BINS=64

def shape_mask(shape):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*0.40
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r: m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r: m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=r*0.32 or abs(dx)<=r*0.32): m[y,x]=1
    return m
# class k -> (color k%4, shape k//4)  gives 8 distinct (4 colors x 2 shapes)
def cls_color(k): return COLOR_RGB[k%4]
def cls_shape(k): return SHAPES[k//4]
MASKS={s:shape_mask(s) for s in SHAPES}

def render_image(k,rng,noise=0.05):
    m=MASKS[cls_shape(k)]
    return np.stack([m*c for c in cls_color(k)],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise

def render_audio(k,rng):
    """Real waveform: class-specific fundamental + harmonic + jitter+noise -> FFT magnitude."""
    f=160+ k*70                                  # class fundamental (Hz-ish)
    f*= (1+0.03*rng.standard_normal())           # per-sample frequency jitter
    t=np.linspace(0,1,1024,endpoint=False)
    wave=np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*2*f*t)+0.2*rng.standard_normal(1024)
    mag=np.abs(np.fft.rfft(wave))[:AUD_BINS]
    mag=np.log1p(mag).astype(np.float32)
    return mag/ (np.linalg.norm(mag)+1e-6)

def text_vec(k):
    v=np.zeros(K,np.float32); v[k]=1; return v

class ImgEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)
class AudEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.m=nn.Sequential(nn.Linear(AUD_BINS,96),nn.ReLU(),nn.Linear(96,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)
class TxtEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.m=nn.Sequential(nn.Linear(K,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)

def nce(a,b):
    lg=a@b.t()/0.07; lb=torch.arange(len(a),device=DEVICE)
    return (F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2

def run(seed, train_IT=True, steps=900):
    torch.manual_seed(seed); rng=np.random.RandomState(seed+9)
    ie,ae,te=ImgEnc().to(DEVICE),AudEnc().to(DEVICE),TxtEnc().to(DEVICE)
    opt=torch.optim.Adam(list(ie.parameters())+list(ae.parameters())+list(te.parameters()),2e-3)
    Tcache=torch.tensor(np.array([text_vec(k) for k in range(K)])).to(DEVICE)
    for _ in range(steps):
        # one sample per class -> clean contrastive (8x8), distinct positives
        imgs=torch.tensor(np.array([render_image(k,rng) for k in range(K)])).to(DEVICE)
        auds=torch.tensor(np.array([render_audio(k,rng) for k in range(K)])).to(DEVICE)
        zi,za,zt=ie(imgs),ae(auds),te(Tcache)
        loss=nce(za,zt)                          # AUDIO <-> TEXT
        if train_IT: loss=loss+nce(zi,zt)        # IMAGE <-> TEXT (the hub)
        opt.zero_grad(); loss.backward(); opt.step()
    # ----- evaluate A->I (never trained directly) -----
    ie.eval(); ae.eval(); te.eval()
    with torch.no_grad():
        # image prototypes = mean embedding over 40 imgs per class
        protos=[]
        for k in range(K):
            ims=torch.tensor(np.array([render_image(k,rng) for _ in range(40)])).to(DEVICE)
            protos.append(ie(ims).mean(0))
        P=F.normalize(torch.stack(protos),dim=1)            # (K,d)
        correct=0; total=0
        for k in range(K):
            auds=torch.tensor(np.array([render_audio(k,rng) for _ in range(40)])).to(DEVICE)
            pred=(ae(auds)@P.t()).argmax(1).cpu().numpy()
            correct+=int((pred==k).sum()); total+=len(pred)
        return correct/total

print(f"Device: {DEVICE} | classes={K} | chance = {100/K:.1f}%")
print(f"{'condition':>22} {'audio->image acc (mean±std)':>28}")
print("-"*54)
hub=np.array([run(s,train_IT=True) for s in range(5)])
ctl=np.array([run(s,train_IT=False) for s in range(5)])
print(f"{'HUB (A-T + I-T)':>22} {hub.mean():>18.0%} ± {hub.std():>3.0%}")
print(f"{'CONTROL (A-T only)':>22} {ctl.mean():>18.0%} ± {ctl.std():>3.0%}")
print(f"\nVerdict: text-hub binding {'WORKS' if hub.mean()>0.4 and hub.mean()-ctl.mean()>0.2 else 'unclear'}; "
      f"control near chance = {ctl.mean():.0%} (expect ~{100/K:.0f}%).")
print(f"VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB" if DEVICE=='cuda' else "CPU")
