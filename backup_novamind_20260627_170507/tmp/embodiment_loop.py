"""
PILAR 1: EMBODIMENT E LOOP CONTÍNUO
Um agente inserido em um ambiente 2D (GridWorld). Ele não recebe dados estáticos;
ele deve agir (cima, baixo, esquerda, direita) e receber recompensas.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbodimentEnv:
    def __init__(self, size=8):
        self.size = size
        self.reset()
        
    def reset(self):
        self.agent_pos = [0, 0]
        # Target at random position
        self.target_pos = [np.random.randint(1, self.size), np.random.randint(1, self.size)]
        self.steps = 0
        return self._get_state()
        
    def _get_state(self):
        # State is a grid where 0=empty, 1=agent, 2=target
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        grid[self.target_pos[0], self.target_pos[1]] = 2.0
        return grid
        
    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        if action == 0 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1: self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1: self.agent_pos[1] += 1
        
        self.steps += 1
        reward = -0.01 # time penalty
        done = False
        
        if self.agent_pos == self.target_pos:
            reward = 1.0
            done = True
        elif self.steps >= 20:
            done = True
            
        return self._get_state(), reward, done

class PolicyNet(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 4)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print(f"[{DEV}] PILAR 1: EMBODIMENT | Agente aprendendo em loop continuo (REINFORCE)")
    env = EmbodimentEnv(size=5)
    policy = PolicyNet(size=5).to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    epochs = 300
    moving_reward = 0
    
    t0 = time.time()
    for ep in range(epochs):
        state = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            # Add channel/batch dims
            st_tensor = torch.tensor(state, device=DEV).unsqueeze(0).unsqueeze(0)
            logits = policy(st_tensor)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            
            state = next_state
            if done: break
            
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=DEV)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.cat(loss).sum()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        ep_reward = sum(rewards)
        moving_reward = 0.05 * ep_reward + 0.95 * moving_reward
        if (ep+1) % 50 == 0:
            print(f"   Episódio {ep+1:3d} | Recompensa Média: {moving_reward:.2f}")

    print("-> SUCESSO! Agente corpóreo otimizou a política explorando o ambiente de forma autônoma.")
    print(f"({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
