"""
world.py v3 — sociedade com CONSTRUCAO (blocos persistentes), COMBATE (matar/guerra),
EMPATIA (gene + compartilhar), economia, terreno. Otimizado (sem syncs no loop quente).
Acoes(12): 0-3 mover,4 comer,5 pegar,6 criar,7 falar,8 trocar,9 ATACAR,10 CONSTRUIR,11 COMPARTILHAR.
Honesto: agentes PERCEBEM blocos (inclusive os que o jogador constroi) — sem trapaca, so percepcao;
nao ha treino com o codigo do jogador. Aprender a copiar estruturas complexas NAO e garantido.
"""
import numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
G=110; P=2600; NMAT=3; NTOOL=12; NTRIBE=4
A_EAT,A_PICK,A_CRAFT,A_TALK,A_TRADE,A_ATTACK,A_BUILD,A_SHARE=4,5,6,7,8,9,10,11; OUT=12
IN=4*9+9+1+NMAT+NTOOL+1+NTRIBE+1+NMAT+1; H=22
_OY=torch.tensor([-1,-1,-1,0,0,0,1,1,1],device=DEV); _OX=torch.tensor([-1,0,1,-1,0,1,-1,0,1],device=DEV)
# arvore PROFUNDA encadeada (12 tiers): cada tier exige o anterior + materiais (distintos p/ evitar subtrair duplo)
RECIPES=[(-1,[0]),(0,[1]),(1,[2]),(2,[0,1]),(3,[1,2]),(4,[0,1,2]),(5,[0,2]),(6,[0,1]),(7,[1,2]),(8,[0,1,2]),(9,[0,2]),(10,[0,1,2])]
def _terrain(g):
    n=torch.randn(1,1,G,G,device=DEV,generator=g)
    for _ in range(6): n=torch.nn.functional.avg_pool2d(torch.nn.functional.pad(n,(1,1,1,1),mode='replicate'),3,1)
    n=(n-n.mean())/(n.std()+1e-6); n=n[0,0]; t=torch.zeros(G,G,dtype=torch.long,device=DEV)
    t[n<-0.8]=1; t[(n>=0.0)&(n<0.9)]=3; t[n>=0.9]=2; return t
class World:
    def __init__(s,seed=0):
        g=torch.Generator(device=DEV).manual_seed(seed); s.g=g
        s.terr=_terrain(g); grass=(s.terr==0).float(); forest=(s.terr==3).float(); rock=(s.terr==2).float(); s.water=(s.terr==1).float()
        s.food=(grass*0.3+forest*0.5)*torch.rand(G,G,device=DEV,generator=g); s.pfood=forest*torch.rand(G,G,device=DEV,generator=g)*0.9
        s.mat=torch.stack([(rock+0.3*grass)*(torch.rand(G,G,device=DEV,generator=g)<0.12).float() for _ in range(NMAT)])
        s.block=torch.zeros(G,G,device=DEV)                 # blocos construidos (tijolos) — persistente, visivel
        s.sig=torch.zeros(G,G,device=DEV)
        s.alive=torch.zeros(P,dtype=torch.bool,device=DEV); s.alive[:1100]=True
        s.x=torch.randint(0,G,(P,),device=DEV,generator=g); s.y=torch.randint(0,G,(P,),device=DEV,generator=g)
        s.energy=torch.zeros(P,device=DEV); s.energy[s.alive]=1.5; s.age=torch.zeros(P,device=DEV)
        s.inv=torch.zeros(P,NMAT,device=DEV); s.tools=torch.zeros(P,NTOOL,device=DEV); s.wealth=torch.zeros(P,device=DEV); s.wealth[s.alive]=2.0
        s.sex=torch.randint(0,2,(P,),device=DEV,generator=g); s.tribe=torch.randint(0,NTRIBE,(P,),device=DEV,generator=g)
        s.social=torch.rand(P,device=DEV,generator=g); s.empathy=torch.rand(P,device=DEV,generator=g)   # EMPATIA evolui
        s.W1=torch.randn(P,IN,H,device=DEV,generator=g)*0.3; s.b1=torch.zeros(P,H,device=DEV)
        s.W2=torch.randn(P,H,OUT,device=DEV,generator=g)*0.3; s.b2=torch.zeros(P,OUT,device=DEV)
        s.act=torch.zeros(P,dtype=torch.long,device=DEV); s.price=torch.ones(NMAT,device=DEV)
        s.goal_bias=torch.zeros(P,OUT,device=DEV)           # CORTEX top-down: vies de objetivo p/ elite (0=sem cortex)
        s.elite_mask=torch.zeros(P,dtype=torch.bool,device=DEV)  # quem o cortex governa (so p/ visualizacao)
        s.freq=1.0; s.ticks=0; s.player=False; s.player_action=A_EAT; s.rain=False
        s.META=0.013; s.REPRO=2.8; s.MUT=0.08; s.MAXAGE=1600.0; s.EAT=0.5; s.FOODPRICE=0.6
    def _patch(s,f): return f[(s.y[:,None]+_OY)%G,(s.x[:,None]+_OX)%G]
    def spawn_player(s):
        s.player=True; s.alive[0]=True; s.x[0]=G//2; s.y[0]=G//2; s.energy[0]=15.0; s.wealth[0]=10.0; s.tribe[0]=0; s.tools[0]=0; s.inv[0]=5.0; s.age[0]=0
    def set_player_action(s,a): s.player_action=int(a)
    def set_frequency(s,f): s.freq=float(max(0.1,min(4.0,f)))
    def place(s,gx,gy,kind):
        gx=int(gx)%G; gy=int(gy)%G
        if kind=='food': s.food[gy,gx]=1.0
        elif kind=='rich': s.pfood[gy,gx]=1.0
        elif kind=='brick': s.block[gy,gx]=1.0          # JOGADOR constroi (Minecraft/Factorio)
        elif kind=='erase': s.block[gy,gx]=0.0
        elif kind.startswith('mat'): s.mat[int(kind[3:])%NMAT,gy,gx]=1.0
        elif kind=='gold':
            d=((s.x-gx)%G).abs()+((s.y-gy)%G).abs(); s.wealth[d<4]+=5.0
        elif kind=='spawn':
            f=(~s.alive).nonzero().flatten()
            if len(f)>0:
                i=int(f[0]); s.alive[i]=True; s.x[i]=gx; s.y[i]=gy; s.energy[i]=2.0; s.wealth[i]=2.0; s.age[i]=0; s.tribe[i]=int(torch.randint(0,NTRIBE,(1,)).item()); s.tools[i]=0; s.inv[i]=0
        elif kind=='cataclysm':
            d=((s.x-gx)%G).abs()+((s.y-gy)%G).abs(); s.alive=s.alive&(d>8)
    def step(s):
        g=s.g; a=s.alive; x=s.x; y=s.y; cell=(y,x); P_=P
        if s.ticks%4==0:
            supply=s.mat.sum((1,2))+s.inv[a].sum(0) if int(a.sum())>0 else s.mat.sum((1,2)); s.price=torch.clamp(3.0/(supply+1.0),0.2,5.0)
        tribe_cell=torch.zeros(NTRIBE,G*G,device=DEV); idx=y*G+x; aw=a.float()
        tribe_cell.index_put_((s.tribe,idx),aw,accumulate=True); tribe_cell=tribe_cell.view(NTRIBE,G,G)
        sigp=s._patch(s.sig).max(1).values; toh=torch.zeros(P_,NTRIBE,device=DEV); toh[torch.arange(P_,device=DEV),s.tribe]=1.0
        sens=torch.cat([s._patch(s.food),s._patch(s.mat[0]),s._patch(s.mat[1]),s._patch(s.mat[2]),s._patch(s.block)],1)  # PERCEBE blocos
        inp=torch.cat([sens,s.energy[:,None],s.inv,s.tools,sigp[:,None],toh,s.wealth[:,None],s.price[None,:].expand(P_,NMAT),s.empathy[:,None]],1)
        h=torch.relu(torch.einsum('pi,pih->ph',inp,s.W1)+s.b1); logits=(torch.einsum('ph,pho->po',h,s.W2)+s.b2)*s.freq + s.goal_bias
        act=torch.distributions.Categorical(logits=logits).sample()
        # APRENDIZADO POR OBSERVACAO: copia a acao do vizinho de MAIOR energia na celula (com EGO: so as vezes)
        key=(torch.clamp(s.energy,0,400)*16).long()+act
        cellbest=torch.full((G*G,),-1.0,device=DEV); cellbest.scatter_reduce_(0,idx,key.float(),reduce='amax',include_self=True)
        bestact=(cellbest[idx].long()%16); imitate=a&(s.social>0.5)&(torch.rand(P_,device=DEV,generator=g)<0.3*s.social)
        act=torch.where(imitate,bestact,act)
        # APRENDER COM O JOGADOR: agentes perto do player observam e copiam a acao dele (com EGO; sem trapaca)
        if s.player and bool(s.alive[0]):
            dxw=torch.minimum((x-s.x[0])%G,(s.x[0]-x)%G); dyw=torch.minimum((y-s.y[0])%G,(s.y[0]-y)%G)
            near=a&((dxw+dyw)<=6)&(s.social>0.4)&(torch.rand(P_,device=DEV,generator=g)<0.45*s.social)
            act=torch.where(near,torch.full_like(act,s.player_action),act)
        if s.player: act[0]=s.player_action
        s.act=act
        dx=torch.where(act==2,1,torch.where(act==3,-1,0)).long(); dy=torch.where(act==0,1,torch.where(act==1,-1,0)).long()
        nx=(x+dx)%G; ny=(y+dy)%G; blocked=(s.water[ny,nx]>0)|(s.block[ny,nx]>0)   # agua E blocos bloqueiam
        mv=(act<4)&a&(~blocked); s.x=torch.where(mv,nx,x); s.y=torch.where(mv,ny,y); x=s.x; y=s.y; cell=(y,x); idx=y*G+x
        eat=(act==A_EAT)&a; tl=s.tools.sum(1)
        gain=s.food[cell]*s.EAT*0.5+(s.tools[:,0]>0).float()*s.pfood[cell]*s.EAT*(1.0+tl); s.energy+=torch.where(eat,gain,torch.zeros_like(gain)); s.energy+=(s.tools[:,3:].sum(1))*0.02*aw
        s.food[cell]=torch.where(eat,s.food[cell]*0.1,s.food[cell])
        pick=(act==A_PICK)&a
        for m in range(NMAT):
            got=pick&(s.mat[m][cell]>0); s.inv[:,m]+=got.float(); s.mat[m][cell]=torch.where(got,0.0,s.mat[m][cell]); s.energy+=got.float()*0.12
        comb=(act==A_CRAFT)&a
        for t,(req,mats) in enumerate(RECIPES):
            ok=comb&(s.tools[:,t]==0)
            if req>=0: ok=ok&(s.tools[:,req]>0)
            for mm in mats: ok=ok&(s.inv[:,mm]>=1)
            for mm in mats: s.inv[:,mm]-=ok.float()
            s.tools[:,t]=torch.where(ok,1.0,s.tools[:,t]); s.energy+=ok.float()*(2.0+t)
        # CONSTRUIR: usa mat0 -> coloca bloco na celula (tijolo a tijolo, persistente, visivel)
        build=(act==A_BUILD)&a&(s.inv[:,0]>=1)&(s.block[cell]==0); s.inv[:,0]-=build.float()
        s.block.index_put_((y,x),torch.maximum(s.block[cell],build.float()),accumulate=False)
        # ECONOMIA: trocar
        tr=(act==A_TRADE)&a; sell=s.inv.argmax(1); hassell=s.inv.gather(1,sell[:,None]).squeeze(1)>0; dosell=tr&hassell
        s.inv[torch.arange(P_,device=DEV),sell]-=dosell.float(); s.wealth+=dosell.float()*s.price[sell]
        buy=s.inv.argmin(1); canbuy=tr&(s.wealth>=s.price[buy]); s.inv[torch.arange(P_,device=DEV),buy]+=canbuy.float(); s.wealth-=canbuy.float()*s.price[buy]
        starving=a&(s.energy<0.6)&(s.wealth>=s.FOODPRICE); s.energy+=starving.float()*1.0; s.wealth-=starving.float()*s.FOODPRICE
        # COMBATE: ataque por celula entre tribos (guerra). dano = ataque de OUTRAS tribos na sua celula
        atkf=torch.zeros(NTRIBE,G*G,device=DEV); atk=(act==A_ATTACK)&a; atkf.index_put_((s.tribe,idx),atk.float(),accumulate=True); atkf=atkf.view(NTRIBE,G*G)
        total_atk=atkf.sum(0)[idx]; own_atk=atkf[s.tribe,idx]; dmg=torch.clamp(total_atk-own_atk,0,10)   # so inimigos ferem
        s.energy=s.energy-dmg*0.8*aw                              # leva dano -> pode MORRER
        s.energy=s.energy+atk.float()*torch.clamp(dmg,0,3)*0.3    # predador ganha ao atacar onde ha inimigo
        # CULTURA
        mytier=s.tools.sum(1); s.sig=s.sig*0.9; s.sig[cell]=torch.maximum(s.sig[cell],mytier*aw)
        learn=(act==A_TALK)&a&(s.social>0.5)&(sigp>mytier)
        for t in range(NTOOL):
            prev=(s.tools[:,t-1]>0) if t>0 else torch.ones(P_,dtype=torch.bool,device=DEV)
            s.tools[:,t]=torch.where(learn&prev&(s.tools[:,t]==0),1.0,s.tools[:,t])
        # EMPATIA: compartilhar energia com a celula (proporcional ao gene); altruismo evolui
        share=(act==A_SHARE)&a; give=share.float()*s.empathy*0.3*torch.clamp(s.energy,0,3)
        pool=torch.zeros(G*G,device=DEV); pool.index_put_((idx,),give,accumulate=True); cnt=torch.zeros(G*G,device=DEV); cnt.index_put_((idx,),aw,accumulate=True)
        s.energy=s.energy-give + aw*pool[idx]/torch.clamp(cnt[idx],1,1e9)
        # nos/eles cooperacao intra
        same=tribe_cell[s.tribe,y,x]-1.0; diff=tribe_cell.sum(0)[y,x]-tribe_cell[s.tribe,y,x]
        s.energy=s.energy+aw*(0.02*torch.clamp(same,0,5)-0.02*torch.clamp(diff,0,5))
        # CLIMA: chuva ciclica; SEM abrigo (>=3 paredes ao redor OU floresta) -> frio/dor progressiva (perde HP)
        s.rain=((s.ticks//350)%3==0)
        if s.rain:
            sheltered=(s._patch(s.block).sum(1)>=3)|(s.terr[cell]==3)
            s.energy=s.energy-(a&(~sheltered)).float()*0.03      # exposicao a chuva: dano progressivo
        s.energy=torch.where(a,s.energy-s.META,s.energy); s.age=torch.where(a,s.age+1,s.age)
        dead=a&((s.energy<=0)|(s.age>s.MAXAGE))
        if s.player: dead[0]=s.energy[0]<=0
        s.alive=a&(~dead)
        for k in('energy','age','wealth'): setattr(s,k,torch.where(dead,torch.zeros_like(getattr(s,k)),getattr(s,k)))
        s.inv=torch.where(dead[:,None],0.0,s.inv); s.tools=torch.where(dead[:,None],0.0,s.tools)
        # reproducao sexual
        rich=s.energy+0.3*s.wealth; free=(~s.alive).nonzero().flatten()
        ma=(s.alive&(rich>s.REPRO)&(s.sex==0)).nonzero().flatten(); fe=(s.alive&(rich>s.REPRO)&(s.sex==1)).nonzero().flatten()
        n=min(len(free),len(ma),len(fe))
        if n>0:
            free=free[:n]; mm=ma[torch.randperm(len(ma),device=DEV,generator=g)[:n]]; ff=fe[torch.randperm(len(fe),device=DEV,generator=g)[:n]]
            k1=(torch.rand(n,IN,H,device=DEV,generator=g)<0.5); s.W1[free]=torch.where(k1,s.W1[mm],s.W1[ff])+torch.randn(n,IN,H,device=DEV,generator=g)*s.MUT
            k2=(torch.rand(n,H,OUT,device=DEV,generator=g)<0.5); s.W2[free]=torch.where(k2,s.W2[mm],s.W2[ff])+torch.randn(n,H,OUT,device=DEV,generator=g)*s.MUT
            s.social[free]=torch.clamp((s.social[mm]+s.social[ff])/2+torch.randn(n,device=DEV,generator=g)*0.1,0,1)
            s.empathy[free]=torch.clamp((s.empathy[mm]+s.empathy[ff])/2+torch.randn(n,device=DEV,generator=g)*0.1,0,1)
            s.tribe[free]=s.tribe[ff]; s.sex[free]=torch.randint(0,2,(n,),device=DEV,generator=g)
            s.energy[mm]*=0.6; s.energy[ff]*=0.6; s.energy[free]=1.5; s.wealth[free]=(s.wealth[mm]+s.wealth[ff])*0.25
            s.x[free]=(s.x[ff]+1)%G; s.y[free]=s.y[ff]; s.age[free]=0; s.alive[free]=True; s.inv[free]=0; s.tools[free]=0
        grass=(s.terr==0).float(); forest=(s.terr==3).float()
        s.food=torch.clamp(s.food+0.004*((grass*0.3+forest*0.5)-s.food),0,1); s.pfood=torch.clamp(s.pfood+0.012*(forest*0.9-s.pfood),0,1)
        if s.ticks%3==0:
            rock=(s.terr==2).float()
            for m in range(NMAT): s.mat[m]=torch.clamp(s.mat[m]+rock*(torch.rand(G,G,device=DEV,generator=g)<0.002).float(),0,1)
        s.ticks+=1
    def snapshot(s,maxn=2600):
        a=s.alive; idx=a.nonzero().flatten()[:maxn]; gv=lambda t:t[idx].cpu().numpy().tolist(); pop=int(a.sum())
        blk=(s.block>0).nonzero().to(torch.int16).cpu().numpy().tolist()
        return {'tick':s.ticks,'pop':pop,'G':G,'NTOOL':NTOOL,'freq':round(s.freq,1),
                'maxtier':int(s.tools[a].sum(1).max().item()) if pop else 0,
                'auto':round(float((s.tools[a,3:].sum(1)>0).float().mean().item()) if pop else 0,2),
                'wealth':round(float(s.wealth[a].sum().item()) if pop else 0,1),'price':[round(p,2) for p in s.price.cpu().numpy().tolist()],
                'empathy':round(float(s.empathy[a].mean().item()) if pop else 0,2),'blocks':len(blk),'rain':bool(s.rain),
                'terr':s.terr[::2,::2].to(torch.uint8).cpu().numpy().tolist(),'blk':blk[:4000],
                'id':idx.cpu().numpy().tolist(),'ax':gv(s.x.to(torch.int16)),'ay':gv(s.y.to(torch.int16)),
                'tribe':gv(s.tribe.to(torch.int8)),'tier':gv(s.tools.sum(1).to(torch.int8)),'act':gv(s.act.to(torch.int8)),
                'mat':[(s.mat[m]>0).nonzero().to(torch.int16).cpu().numpy().tolist()[:500] for m in range(NMAT)],
                'px':int(s.x[0]),'py':int(s.y[0]),'player':s.player,'penergy':round(float(s.energy[0]),1),
                'pwealth':round(float(s.wealth[0]),1),'ptier':int(s.tools[0].sum().item()),'pinv':[int(v) for v in s.inv[0].cpu().numpy().tolist()]}
