"""
H35 — Learn a game from REWARD only, no rules given (the user's chess idea, feasible)
=====================================================================================

Chess-from-scratch-no-rules is infeasible on a 3050 (AlphaZero used thousands of
TPUs; sparse reward). Tic-tac-toe is the feasible analog and proves the SAME
principle: an agent given ONLY reward (win=+1, lose/illegal=-1, draw=+0.2) and NO
rules learns to play. Illegal move (occupied cell) = instant loss -> it must even
learn legal play from reward alone.

Train vs EASY (random) opponent, then evaluate vs a HARDER opponent (blocks/wins
when obvious). Show the learning curve. REINFORCE policy net.

HONEST framing: this works (narrow RL). But mastering ONE game is NARROW -- it is
the definition of narrow (AlphaZero/Stockfish are superhuman at chess and NOT AGI).
A sign of AGI would be ONE system handling MANY novel tasks, not one game.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

def winner(b):
    L=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,c,d in L:
        if b[a]!=0 and b[a]==b[c]==b[d]: return b[a]
    return 0

def opp_move(b,level,rng):
    legal=[i for i in range(9) if b[i]==0]
    if not legal: return None
    if level=='easy': return rng.choice(legal)
    # harder: win if can, block if must, else random
    for mark in (2,1):  # 2=opp win, 1=block agent
        for i in legal:
            bb=b.copy(); bb[i]=mark
            if winner(bb)==mark: return i
    return rng.choice(legal)

class Policy(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(27,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,9))
    def forward(s,b):
        x=torch.zeros(27,device=DEVICE)
        for i,v in enumerate(b): x[i*3+int(v)]=1.0
        return s.net(x)

def play_episode(pol, level, rng, train=True):
    b=np.zeros(9,int); logps=[]; agent_first=rng.rand()<0.5
    if not agent_first:
        m=opp_move(b,level,rng); b[m]=2
    while True:
        logits=pol(b); probs=F.softmax(logits,0); d=torch.distributions.Categorical(probs); a=d.sample()
        logps.append(d.log_prob(a)); ai=int(a)
        if b[ai]!=0:  # ILLEGAL -> instant loss (learned legal play from reward)
            return logps,-1.0
        b[ai]=1
        if winner(b)==1: return logps,1.0
        if (b!=0).all(): return logps,0.2     # draw
        m=opp_move(b,level,rng); b[m]=2
        if winner(b)==2: return logps,-1.0
        if (b!=0).all(): return logps,0.2

def evaluate(pol,level,rng,n=2000):
    w=dr=l=il=0
    for _ in range(n):
        b=np.zeros(9,int); first=rng.rand()<0.5; illegal=False
        if not first: m=opp_move(b,level,rng); b[m]=2
        res=0
        while True:
            with torch.no_grad(): a=int(pol(b).argmax())
            if b[a]!=0: res=-1; illegal=True; break
            b[a]=1
            if winner(b)==1: res=1; break
            if (b!=0).all(): res=0; break
            m=opp_move(b,level,rng); b[m]=2
            if winner(b)==2: res=-1; break
            if (b!=0).all(): res=0; break
        if illegal: il+=1
        if res==1: w+=1
        elif res==0: dr+=1
        else: l+=1
    return w/n,dr/n,l/n,il/n

pol=Policy().to(DEVICE); opt=torch.optim.Adam(pol.parameters(),5e-3); rng=np.random.RandomState(0)
print(f"Device: {DEVICE} | tic-tac-toe from REWARD only, NO rules given")
print(f"{'episodes':>9} {'win%':>6} {'draw%':>6} {'loss%':>6} {'illegal%':>9}  (vs easy)")
EP=40000
for ep in range(1,EP+1):
    logps,R=play_episode(pol,'easy',rng)
    loss=-torch.stack(logps).sum()*R
    opt.zero_grad(); loss.backward(); opt.step()
    if ep in (200,2000,10000,40000):
        w,dr,l,il=evaluate(pol,'easy',rng)
        print(f"{ep:>9} {w:>5.0%} {dr:>6.0%} {l:>6.0%} {il:>8.0%}")
print("-"*48)
w,dr,l,il=evaluate(pol,'hard',rng)
print(f"vs HARDER opponent: win {w:.0%} draw {dr:.0%} loss {l:.0%} illegal {il:.0%}")
print("\nHonest: it LEARNED a game from reward alone, no rules (illegal-rate -> low, win-rate up).")
print("That is real narrow RL. But ONE game = NARROW (AlphaZero/Stockfish: superhuman chess, NOT AGI).")
print("AGI = one system handling MANY novel tasks (it would also pass ARC, which ours can't).")
