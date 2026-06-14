"""
train_worldmodel — treino real de ~5 min na GPU pra ampliar o modelo de mundo grounded
=======================================================================================
Expande o espaço de conceitos (cor x forma x tamanho) e treina o binding multimodal
(imagem + audio + texto -> espaco compartilhado) por ~300s, usando a GPU intensamente
(batches grandes, cache na GPU). Resultado: um modelo de mundo grounded MAIOR (mais
conceitos ancorados cross-modal). Narrow e real -- nao conhecimento do mundo real, nao AGI.
"""
import time, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
COLORS=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1)]      # 6 cores
SHAPES=["circle","square","triangle","cross","diamond"]       # 5 formas
SIZES=[0.30,0.42,0.54]                                         # 3 tamanhos
G=16; AUD=96; D=192
CONCEPTS=list(itertools.product(range(len(COLORS)),range(len(SHAPES)),range(len(SIZES))))
NC=len(CONCEPTS)

def mask(shape,r):
    m=np.zeros((G,G),np.float32); cy=cx=(G-1)/2; rr=G*r
    for y in range(G):
        for x in range(G):
            dy,dx=y-cy,x-cx
            if shape=="circle" and dy*dy+dx*dx<=rr*rr: m[y,x]=1
            elif shape=="square" and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="triangle" and -rr<=dy<=rr and abs(dx)<=(dy+rr)*0.5: m[y,x]=1
            elif shape=="cross" and (abs(dy)<=rr*0.32 or abs(dx)<=rr*0.32) and abs(dy)<=rr and abs(dx)<=rr: m[y,x]=1
            elif shape=="diamond" and abs(dy)+abs(dx)<=rr: m[y,x]=1
    return m
MK={(s,zi):mask(SHAPES[s],SIZES[zi]) for s in range(len(SHAPES)) for zi in range(len(SIZES))}
def img(c,rng): return np.stack([MK[(c[1],c[2])]*v for v in COLORS[c[0]]],0)+rng.standard_normal((3,G,G)).astype(np.float32)*0.05
def aud(c,rng):
    f=150+c[1]*70+c[2]*30; f*=(1+0.03*rng.standard_normal()); t=np.linspace(0,1,2048,endpoint=False)
    w=np.sin(2*np.pi*f*t)+0.5*np.sin(2*np.pi*(2+c[0])*f*t)+0.2*rng.standard_normal(2048)
    mg=np.abs(np.fft.rfft(w))[:AUD]; mg=np.log1p(mg).astype(np.float32); return mg/(np.linalg.norm(mg)+1e-6)
def txt(c): v=np.zeros(NC,np.float32); v[CONCEPTS.index(c)]=1; return v

class ImgE(nn.Module):
    def __init__(s): super().__init__(); s.c=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(128*16,D))
    def forward(s,x): return F.normalize(s.c(x),1)
class AudE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(AUD,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,x): return F.normalize(s.m(x),1)
class TxtE(nn.Module):
    def __init__(s): super().__init__(); s.m=nn.Sequential(nn.Linear(NC,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,x): return F.normalize(s.m(x),1)

rng=np.random.RandomState(0); POOL=50
print(f"Conceitos no modelo de mundo: {NC} (cor x forma x tamanho) | construindo cache na GPU...")
Ic=torch.tensor(np.array([[img(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
Ac=torch.tensor(np.array([[aud(c,rng) for _ in range(POOL)] for c in CONCEPTS])).to(DEVICE)
Tc=torch.tensor(np.array([txt(c) for c in CONCEPTS])).to(DEVICE)
ie,ae,te=ImgE().to(DEVICE),AudE().to(DEVICE),TxtE().to(DEVICE)
anchor=nn.Parameter(F.normalize(torch.randn(NC,D),1).to(DEVICE))
opt=torch.optim.Adam(list(ie.parameters())+list(ae.parameters())+list(te.parameters())+[anchor],2e-3)
PER=16; lab=torch.arange(NC,device=DEVICE).repeat_interleave(PER); ar=lab.clone()

def evalb():
    ie.eval();ae.eval();te.eval()
    with torch.no_grad():
        Pi=torch.stack([ie(Ic[k,:30]).mean(0) for k in range(NC)]); Pa=torch.stack([ae(Ac[k,:30]).mean(0) for k in range(NC)])
        Pi=F.normalize(Pi,1); Pa=F.normalize(Pa,1)
        a2i=(Pa@Pi.t()).argmax(1).cpu().numpy(); acc=np.mean([a2i[k]==k for k in range(NC)])
    ie.train();ae.train();te.train(); return acc

print(f"Treinando ~5 min na {DEVICE} (batch={NC*PER} por passo)...")
t0=time.time(); step=0; last=0
while time.time()-t0<300:
    idx=torch.randint(0,POOL,(NC*PER,),device=DEVICE)
    zi=ie(Ic[ar,idx]); za=ae(Ac[ar,idx]); zt=te(Tc).repeat_interleave(PER,0); anc=F.normalize(anchor,1)
    loss=F.cross_entropy(zi@anc.t()/0.07,lab)+F.cross_entropy(za@anc.t()/0.07,lab)+F.cross_entropy(zt@anc.t()/0.07,lab)
    opt.zero_grad(); loss.backward(); opt.step(); step+=1
    if time.time()-t0-last>=30:
        last=time.time()-t0; vram=torch.cuda.memory_allocated()/1e6 if DEVICE=='cuda' else 0
        print(f"  t={last:>4.0f}s | passos={step:>5} | loss={loss.item():.3f} | binding(audio->img)={evalb():.0%} | VRAM={vram:.0f}MB")
acc=evalb()
print("-"*60)
print(f"FIM: {NC} conceitos | binding cross-modal final = {acc:.0%} | {step} passos em 5 min")
torch.save({'ie':ie.state_dict(),'ae':ae.state_dict(),'te':te.state_dict(),'anchor':anchor.detach().cpu()},
           'tmp/worldmodel_trained.pt')
print("Modelo de mundo ampliado salvo em tmp/worldmodel_trained.pt")
print("Honesto: repertorio NARROW maior (90 conceitos de brinquedo aterrados cross-modal). Real,")
print("mas nao e conhecimento do mundo real (LLM/imagem treinam em TB/semanas/milhares de GPUs);")
print("nao move o ARC (dominio diferente) nem vira AGI. E um modelo de mundo maior, com nossa arquitetura.")
