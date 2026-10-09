"""
worldmodel_3d — See the system's WORLD MODEL in 3D (transparency for the scientist)
===================================================================================

Trains the multimodal binding (image+audio+text -> shared space) and renders the
system's actual internal 'world model' in 3D:
  -每 concept's 3 modalities as points in a 3D PCA of the shared space
    (same-concept modalities clustering = the learned associations)
  - an association graph: concepts linked by embedding similarity

This shows -- honestly -- what the system really represents: a space of toy
concepts and their cross-modal associations. Not a general world; a narrow,
inspectable one. Saved as worldmodel_3d.png (multiple angles).
"""
import os, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
COLORS={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
CN=list(COLORS); SHAPES=["circle","square","triangle"]; G=14; D=64
CONCEPTS=list(itertools.product(range(len(CN)),range(len(SHAPES)))); NC=len(CONCEPTS)
def label(c): return f"{CN[c[0]]} {SHAPES[c[1]]}"

def mask(shape):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; r=G*0.4
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=r*r:m[y,x]=1
            elif shape=="square" and abs(dy)<=r and abs(dx)<=r:m[y,x]=1
            elif shape=="triangle" and -r<=dy<=r and abs(dx)<=(dy+r)*0.5:m[y,x]=1
    return m
MK={s:mask(s) for s in SHAPES}
def img(c,rng): return np.stack([MK[SHAPES[c[1]]]*v for v in COLORS[CN[c[0]]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*0.05
def aud(c,rng):
    f=180+c[1]*90; f*=(1+0.03*rng.standard_normal()); t=np.linspace(0,1,1024,endpoint=False)
    w=np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*(2+c[0])*f*t)+0.2*rng.standard_normal(1024)
    mg=np.abs(np.fft.rfft(w))[:64]; mg=np.log1p(mg).astype(np.float32); return mg/(np.linalg.norm(mg)+1e-6)
def txt(c): v=np.zeros(NC,np.float32); v[CONCEPTS.index(c)]=1; return v

class ImgE(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(32*16,D))
    def forward(s,x): return F.normalize(s.c(x),1)
class AudE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,x): return F.normalize(s.m(x),1)
class TxtE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(NC,64),nn.ReLU(),nn.Linear(64,D))
    def forward(s,x): return F.normalize(s.m(x),1)

rng=np.random.RandomState(0); POOL=120
Ic=torch.tensor(np.array([[img(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
Ac=torch.tensor(np.array([[aud(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
Tc=torch.tensor(np.array([txt(c) for c in CONCEPTS])).to(DEVICE)
ie,ae,te=ImgE().to(DEVICE),AudE().to(DEVICE),TxtE().to(DEVICE)
anchor=nn.Parameter(F.normalize(torch.randn(NC,D),1).to(DEVICE))
opt=torch.optim.Adam(list(ie.parameters())+list(ae.parameters())+list(te.parameters())+[anchor],2e-3)
lab=torch.arange(NC,device=DEVICE).repeat_interleave(12); ar=lab.clone()
print("Treinando o modelo de mundo (binding multimodal)...")
for _ in range(1200):
    idx=torch.randint(0,POOL,(NC*12,),device=DEVICE)
    zi=ie(Ic[ar,idx]); za=ae(Ac[ar,idx]); zt=te(Tc).repeat_interleave(12,0); anc=F.normalize(anchor,1)
    loss=F.cross_entropy(zi@anc.t()/0.07,lab)+F.cross_entropy(za@anc.t()/0.07,lab)+F.cross_entropy(zt@anc.t()/0.07,lab)
    opt.zero_grad(); loss.backward(); opt.step()
for m in (ie,ae,te): m.eval()
with torch.no_grad():
    P={'image':torch.stack([ie(Ic[k,:40]).mean(0) for k in range(NC)]),
       'audio':torch.stack([ae(Ac[k,:40]).mean(0) for k in range(NC)]),
       'text':te(Tc)}
    for k in P: P[k]=F.normalize(P[k],1)
allv=torch.cat([P['image'],P['audio'],P['text']],0).cpu().numpy()
Xc=allv-allv.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False); P3=Xc@Vt[:3].T   # PCA -> 3D
cols=plt.cm.nipy_spectral(np.linspace(0,1,NC)); mk={'image':'o','audio':'^','text':'*'}
fig=plt.figure(figsize=(20,9))
for ai,(elev,azim) in enumerate([(20,35),(20,125)]):
    ax=fig.add_subplot(1,2,ai+1,projection='3d')
    for mi,m in enumerate(['image','audio','text']):
        seg=P3[mi*NC:(mi+1)*NC]
        for k in range(NC): ax.scatter(*seg[k],color=cols[k],marker=mk[m],s=80,edgecolors='k',linewidths=0.3)
    # association edges (text-concept nearest neighbors)
    tp=P['text'].cpu().numpy(); sim=tp@tp.T; np.fill_diagonal(sim,-1)
    seg=P3[2*NC:3*NC]
    for k in range(NC):
        for nb in np.argsort(-sim[k])[:2]:
            ax.plot([seg[k,0],seg[nb,0]],[seg[k,1],seg[nb,1]],[seg[k,2],seg[nb,2]],'gray',alpha=0.3,lw=0.7)
    if ai==0:
        for k in range(NC): ax.text(*seg[k],label(CONCEPTS[k]),fontsize=6)
    ax.set_title(f"MODELO DE MUNDO em 3D (ang {azim}°)\no=imagem ^=audio *=texto; cor=conceito; linhas=associacao")
    ax.view_init(elev=elev,azim=azim)
out=os.path.join(os.path.dirname(__file__),"worldmodel_3d.png"); plt.tight_layout(); plt.savefig(out,dpi=110); plt.close()
# cross-modal sanity
with torch.no_grad():
    acc=np.mean([(P['audio'][k]@P['image'].t()).argmax().item()==k for k in range(NC)])
print(f"binding cross-modal (audio->image): {acc:.0%} | conceitos: {NC}")
print(f"MODELO DE MUNDO 3D salvo em: {out}")
print("Honesto: este e o modelo de mundo REAL do sistema -- um espaco de conceitos de brinquedo")
print("e suas associacoes cross-modais. Inspecionavel, real, e NARROW (nao um mundo geral).")
