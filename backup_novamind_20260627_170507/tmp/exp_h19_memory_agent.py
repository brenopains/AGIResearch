"""
H19 — Minimal AGENCY: remember a state to solve a task (RL, POMDP)
=================================================================

Every prior experiment was supervised & static. This adds the missing piece the
user flagged from the start: an ACTION->consequence loop. A tiny POMDP, memo-proof
by design:

  t=0      : a CUE (0 or 1) is shown (random each episode)
  t=1..T-1 : cue is GONE (blank observations)
  t=T-1    : agent picks action in {0,1}; reward = 1 iff action == cue

The cue is random per episode, so there is NO fixed answer to memorize: the agent
must DEVELOP the strategy 'remember the cue across blank steps, then act on it'.
Trained from scratch by RL (REINFORCE) -- no pretraining, no supervision of the
strategy.

Architecture question (ties to H18): does an agent WITH memory (GRU) discover the
solution, while a MEMORYLESS agent (sees only current obs) cannot?
  Prediction: memory -> ~100%; memoryless -> ~50% (cue is gone at decision time).
HONEST: textbook meta-RL; not novel, not AGI. But it is the first genuine agency
loop here, probing 'remember + solve autonomously'.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
T=5            # episode length; cue at t0, decide at t=T-1
OBS=3          # [cue==0, cue==1, is_decision_step]
B=256          # episodes per update

def make_batch(rng):
    cue=torch.tensor(rng.randint(0,2,B),dtype=torch.long,device=DEVICE)
    obs=torch.zeros(T,B,OBS,device=DEVICE)
    obs[0,torch.arange(B),cue]=1.0          # cue one-hot at t0
    obs[T-1,:,2]=1.0                          # decision-step flag at last step
    return obs,cue

class MemoryAgent(nn.Module):
    def __init__(s,h=32):
        super().__init__(); s.rnn=nn.GRU(OBS,h); s.head=nn.Linear(h,2)
    def forward(s,obs):
        out,_=s.rnn(obs)            # (T,B,h)
        return s.head(out[-1])      # logits at decision step, using full history

class MemorylessAgent(nn.Module):
    def __init__(s,h=32):
        super().__init__(); s.net=nn.Sequential(nn.Linear(OBS,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,obs):
        return s.net(obs[-1])       # sees ONLY the decision-step obs (cue gone)

def train(agent, seed, updates=1500):
    torch.manual_seed(seed); rng=np.random.RandomState(seed)
    opt=torch.optim.Adam(agent.parameters(),3e-3)
    succ=0.0
    for u in range(updates):
        obs,cue=make_batch(rng)
        logits=agent(obs); probs=F.softmax(logits,1)
        dist=torch.distributions.Categorical(probs)
        act=dist.sample()
        reward=(act==cue).float()
        baseline=reward.mean()
        loss=-(dist.log_prob(act)*(reward-baseline)).mean()-0.01*dist.entropy().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        succ=0.98*succ+0.02*reward.mean().item() if u>0 else reward.mean().item()
    # final eval (greedy) on fresh cues
    with torch.no_grad():
        obs,cue=make_batch(rng)
        act=agent(obs).argmax(1)
        return (act==cue).float().mean().item()

print(f"Device: {DEVICE} | POMDP: remember cue across {T-1} blank steps | chance=50%")
print(f"{'agent':>14} {'success on novel cues (mean±std)':>34}")
print("-"*52)
for name,cls in [("MEMORY (GRU)",MemoryAgent),("MEMORYLESS (MLP)",MemorylessAgent)]:
    accs=np.array([train(cls().to(DEVICE),s) for s in range(4)])
    print(f"{name:>14} {accs.mean():>22.0%} ± {accs.std():>3.0%}")
print("\nPrediction: memory agent develops 'remember->act' (~100%); memoryless stuck at ~50%.")
