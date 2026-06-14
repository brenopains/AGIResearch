"""
H8 — Ground NovaMind in perception (close the world-model loop)
===============================================================

Until now NovaMind perceives hand-typed TEXT. The user's key intuition was that
a real world model must associate an IMAGE with meaning. Here we wire the
grounded "eye" (the contrastive color+shape encoder) into NovaMind:

   pixels --> GroundedEye --> concept label --> NovaMind.think() --> memory

We even feed a NOVEL combination (never trained as a pair) so the demo chains
THREE capabilities at once:
   compositional grounding (name an unseen combo)
   + symbolic perception (turn it into a concept)
   + episodic memory (store & recall what was 'seen').

Honest scope: toy shapes; "grounding" = nearest-concept retrieval, not deep
understanding. But the LINK from pixels to symbolic memory is real and runs on
the 3050.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
COLOR_RGB={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
COLORS=list(COLOR_RGB); SHAPES=["circle","square","triangle","cross"]; G=14
C,S=len(COLORS),len(SHAPES)

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
MASKS={s:shape_mask(s) for s in SHAPES}

def render(ci,si,rng,noise=0.05):
    m=MASKS[SHAPES[si]]
    return np.stack([m*c for c in COLOR_RGB[COLORS[ci]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*noise
def tv(ci,si):
    v=np.zeros(C+S,np.float32); v[ci]=1; v[C+si]=1; return v

class Img(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.c=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(24*16,d))
    def forward(self,x): return F.normalize(self.c(x),dim=1)
class Txt(nn.Module):
    def __init__(self,d=64):
        super().__init__(); self.m=nn.Sequential(nn.Linear(C+S,64),nn.ReLU(),nn.Linear(64,d))
    def forward(self,x): return F.normalize(self.m(x),dim=1)


class GroundedEye:
    """Trains contrastively, then names an image by nearest concept."""
    def __init__(self, holdout):
        self.ie, self.te = Img().to(DEVICE), Txt().to(DEVICE)
        self.holdout=holdout
        self.concepts=list(itertools.product(range(C),range(S)))

    def train(self, epochs=400, seed=0):
        torch.manual_seed(seed); self.rng=np.random.RandomState(seed)
        train_combos=[c for c in self.concepts if c not in self.holdout]
        opt=torch.optim.Adam(list(self.ie.parameters())+list(self.te.parameters()),3e-3); tau=0.07
        def batch(combos,reps):
            I,T=[],[]
            for (ci,si) in combos:
                for _ in range(reps): I.append(render(ci,si,self.rng)); T.append(tv(ci,si))
            return torch.tensor(np.array(I)).to(DEVICE),torch.tensor(np.array(T)).to(DEVICE)
        for _ in range(epochs):
            im,tx=batch(train_combos,6); zi,zt=self.ie(im),self.te(tx); lg=zi@zt.t()/tau
            lb=torch.arange(len(im),device=DEVICE)
            loss=(F.cross_entropy(lg,lb)+F.cross_entropy(lg.t(),lb))/2
            opt.zero_grad(); loss.backward(); opt.step()
        self.ie.eval(); self.te.eval()
        with torch.no_grad():
            cand=torch.tensor(np.array([tv(*c) for c in self.concepts])).to(DEVICE)
            self.concept_emb=self.te(cand)

    def see(self, ci, si):
        """Return the concept label the eye perceives from an image's pixels."""
        with torch.no_grad():
            img=torch.tensor(render(ci,si,self.rng)[None]).to(DEVICE)
            sims=self.ie(img)@self.concept_emb.t()
            best=int(sims.argmax())
        c,s=self.concepts[best]
        return f"{COLORS[c]} {SHAPES[s]}"


def main():
    from core.mind import NovaMind
    novel=(2,2)  # blue triangle, held out from training pairs
    eye=GroundedEye(holdout={novel})
    print("Training the grounded eye (color+shape contrastive)...")
    eye.train()

    print(f"\n{'='*60}\nShowing NovaMind images (pixels -> concept -> memory)\n{'='*60}")
    mind=NovaMind(use_semantic_recall=True)

    shown=[(0,0),(1,1),(3,3),novel]  # last one is the NEVER-seen-paired combo
    for (ci,si) in shown:
        label=eye.see(ci,si)
        truth=f"{COLORS[ci]} {SHAPES[si]}"
        tag=" (NOVEL combo!)" if (ci,si)==novel else ""
        sentence=f"I see a {label}"
        mind.think(sentence)
        ok = "OK" if label==truth else "miss"
        print(f"  pixels[{truth:16s}] -> eye perceives: '{label}'  [{ok}]{tag}")

    print(f"\n{'-'*60}\nNow ask NovaMind from MEMORY what it saw:")
    for q in ["What did I see", "Did I see a triangle"]:
        r=mind.think(q)['response']['text']
        print(f"  Q: {q:22s} -> {r}")
    print(f"{'='*60}")
    print("The symbolic memory was driven by PIXELS, including a combo never trained as a pair.")


if __name__=='__main__':
    main()
