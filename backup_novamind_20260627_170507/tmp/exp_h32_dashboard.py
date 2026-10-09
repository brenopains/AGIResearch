"""
H32 — Richer multimodal grounding + a REAL data dashboard
=========================================================

Builds a multimodal binding system (IMAGE + VIDEO + AUDIO + TEXT) over many
concepts, aligned to a shared per-concept anchor (the H15 mechanism that hit
100%). Uses a bigger model + larger GPU cache to push VRAM up. Then it produces a
VISUAL DASHBOARD of the system's REAL internals (no inflated 'AGI' metrics):
  - PCA map of all cross-modal embeddings: shows the 4 modalities of each concept
    clustering together = the learned ASSOCIATIONS.
  - Real cross-modal retrieval accuracy (audio->image, video->text, etc.).
  - World-model graph: concepts linked by embedding similarity.

HONEST scope: richer data -> richer NARROW grounding (real, useful, visible). It
does NOT cross the L4/AGI wall (H31 was decisive). The dashboard shows REAL data.
"""
import os, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

COLORS={"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"yellow":(1,1,0)}
CN=list(COLORS); SHAPES=["circle","square","triangle"]; SIZES=[0.34,0.5]
G=20; AUD=96; D=256
CONCEPTS=list(itertools.product(range(len(CN)),range(len(SHAPES)),range(len(SIZES))))
NC=len(CONCEPTS)
def label(c): return f"{CN[c[0]]} {SHAPES[c[1]]} {'sm' if c[2]==0 else 'lg'}"

def mask(shape,r,off=(0,0)):
    m=np.zeros((G,G),np.float32); cy=(G-1)/2+off[0]; cx=(G-1)/2+off[1]; rr=G*r
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=rr*rr: m[y,x]=1
            elif shape=="square" and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="triangle" and -rr<=dy<=rr and abs(dx)<=(dy+rr)*0.5: m[y,x]=1
    return m
def img(c,rng):
    m=mask(SHAPES[c[1]],SIZES[c[2]]); return np.stack([m*v for v in COLORS[CN[c[0]]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*0.05
def vid(c,rng):  # 3 frames: shape translating (motion)
    fr=[]
    for t in range(3):
        m=mask(SHAPES[c[1]],SIZES[c[2]],off=(0,(t-1)*3)); fr.append(np.stack([m*v for v in COLORS[CN[c[0]]]],0))
    return (np.stack(fr,0)+rng.standard_normal((3,3,G,G)).astype(np.float32)*0.05)
def aud(c,rng):  # frequency encodes (shape,size); harmonics encode color
    f=180+c[1]*90+c[2]*40; f*=(1+0.03*rng.standard_normal()); t=np.linspace(0,1,2048,endpoint=False)
    w=np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*(2+c[0])*f*t)+0.2*rng.standard_normal(2048)
    mg=np.abs(np.fft.rfft(w))[:AUD]; mg=np.log1p(mg).astype(np.float32); return mg/(np.linalg.norm(mg)+1e-6)
def txt(c): v=np.zeros(NC,np.float32); v[CONCEPTS.index(c)]=1; return v

class ImgE(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(128*16,D))
    def forward(s,x): return F.normalize(s.c(x),1)
class VidE(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(64*16,D))
    def forward(s,x):  # x:(B,3,3,G,G) -> mean over frames
        B=x.shape[0]; f=s.c(x.reshape(B*3,3,G,G)).reshape(B,3,D); return F.normalize(f.mean(1),1)
class AudE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(AUD,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,x): return F.normalize(s.m(x),1)
class TxtE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(NC,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,x): return F.normalize(s.m(x),1)

def main():
    rng=np.random.RandomState(0); torch.manual_seed(0)
    POOL=200
    print(f"Device: {DEVICE} | {NC} concepts x 4 modalities | building cache (uses VRAM)...")
    Icache=torch.tensor(np.array([[img(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
    Vcache=torch.tensor(np.array([[vid(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
    Acache=torch.tensor(np.array([[aud(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
    Tcache=torch.tensor(np.array([txt(c) for c in CONCEPTS])).to(DEVICE)
    ie,ve,ae,te=ImgE().to(DEVICE),VidE().to(DEVICE),AudE().to(DEVICE),TxtE().to(DEVICE)
    anchor=nn.Parameter(F.normalize(torch.randn(NC,D),1).to(DEVICE))
    opt=torch.optim.Adam(list(ie.parameters())+list(ve.parameters())+list(ae.parameters())+list(te.parameters())+[anchor],1.5e-3)
    labels=torch.arange(NC,device=DEVICE).repeat_interleave(16); ar=labels.clone()
    nce=lambda z,anc: F.cross_entropy(z@F.normalize(anc,1).t()/0.07, labels)
    for step in range(1500):
        idx=torch.randint(0,POOL,(NC*16,),device=DEVICE)
        zi=ie(Icache[ar,idx]); zv=ve(Vcache[ar,idx]); za=ae(Acache[ar,idx]); zt=te(Tcache).repeat_interleave(16,0)
        loss=nce(zi,anchor)+nce(zv,anchor)+nce(za,anchor)+nce(zt,anchor)
        opt.zero_grad(); loss.backward(); opt.step()
    for m in (ie,ve,ae,te): m.eval()
    # ---- real cross-modal retrieval + embeddings for dashboard ----
    with torch.no_grad():
        protos={}  # modality -> (NC,D) mean concept embedding
        embs={'image':ie,'video':ve,'audio':ae}
        protos['image']=torch.stack([ie(Icache[k,:40]).mean(0) for k in range(NC)])
        protos['video']=torch.stack([ve(Vcache[k,:40]).mean(0) for k in range(NC)])
        protos['audio']=torch.stack([ae(Acache[k,:40]).mean(0) for k in range(NC)])
        protos['text']=te(Tcache)
        for k in protos: protos[k]=F.normalize(protos[k],1)
        # retrieval matrix: query modality -> target modality (top-1 over concepts)
        mods=['image','video','audio','text']; M=np.zeros((4,4))
        for i,q in enumerate(mods):
            for j,t in enumerate(mods):
                pred=(protos[q]@protos[t].t()).argmax(1).cpu().numpy()
                M[i,j]=(pred==np.arange(NC)).mean()
    vram=torch.cuda.memory_allocated()/1e6 if DEVICE=='cuda' else 0

    # ---- DASHBOARD (real data) ----
    allv=torch.cat([protos[m] for m in mods],0).cpu().numpy()
    Xc=allv-allv.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False); P=Xc@Vt[:2].T   # PCA 2D
    cols=plt.cm.nipy_spectral(np.linspace(0,1,NC))
    markers={'image':'o','video':'s','audio':'^','text':'*'}
    fig,axs=plt.subplots(1,3,figsize=(21,6.5))
    # Panel A: cross-modal embedding map (associations)
    for mi,m in enumerate(mods):
        seg=P[mi*NC:(mi+1)*NC]
        for k in range(NC): axs[0].scatter(seg[k,0],seg[k,1],color=cols[k],marker=markers[m],s=70,edgecolors='k',linewidths=0.3)
    axs[0].set_title("ESPACO COMPARTILHADO (associacoes cross-modais)\nmesma cor=mesmo conceito; o/s/^/*=imagem/video/audio/texto")
    # Panel B: retrieval matrix (real)
    im=axs[1].imshow(M,cmap='viridis',vmin=0,vmax=1)
    axs[1].set_xticks(range(4)); axs[1].set_yticks(range(4)); axs[1].set_xticklabels(mods); axs[1].set_yticklabels(mods)
    for i in range(4):
        for j in range(4): axs[1].text(j,i,f"{M[i,j]:.0%}",ha='center',va='center',color='w')
    axs[1].set_title(f"RECUPERACAO CROSS-MODAL real (top-1, acaso {100/NC:.0f}%)\nlinha=consulta -> coluna=alvo"); plt.colorbar(im,ax=axs[1],fraction=0.046)
    # Panel C: world-model graph (concepts linked by text-embedding similarity)
    txtp=protos['text'].cpu().numpy(); sim=txtp@txtp.T; np.fill_diagonal(sim,-1)
    th=np.linspace(0,2*np.pi,NC,endpoint=False); pos=np.stack([np.cos(th),np.sin(th)],1)
    for k in range(NC):
        for nb in np.argsort(-sim[k])[:2]:
            axs[2].plot([pos[k,0],pos[nb,0]],[pos[k,1],pos[nb,1]],'gray',alpha=0.35,lw=0.8)
    for k in range(NC):
        axs[2].scatter(*pos[k],color=cols[k],s=120,edgecolors='k',zorder=3)
        axs[2].text(pos[k,0]*1.12,pos[k,1]*1.12,label(CONCEPTS[k]),fontsize=6,ha='center')
    axs[2].set_title("MODELO DE MUNDO (conceitos ligados por similaridade)"); axs[2].axis('off'); axs[2].set_xlim(-1.4,1.4); axs[2].set_ylim(-1.4,1.4)
    out=os.path.join(os.path.dirname(__file__),"dashboard.png"); plt.tight_layout(); plt.savefig(out,dpi=110); plt.close()

    print(f"\nCross-modal retrieval (real, chance {100/NC:.0f}%):")
    for i,m in enumerate(mods): print(f"   {m:>6} -> "+"  ".join(f"{t}:{M[i,j]:.0%}" for j,t in enumerate(mods)))
    print(f"\nmean off-diagonal cross-modal acc: {(M.sum()-np.trace(M))/12:.0%}")
    print(f"VRAM used: {vram:.0f} MB")
    print(f"DASHBOARD salvo em: {out}")

if __name__=='__main__': main()
