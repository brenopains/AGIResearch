import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, in_features, action_dim, hidden_dim=512):
        super().__init__()
        # 1024 or 512 in hidden
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch], dim=-1)
        feat = self.net(x)
        
        mean = self.mean_layer(feat)
        # Bounding standard deviation to prevent collapse or explosion
        std = torch.clamp(nn.functional.softplus(self.std_layer(feat)), min=0.1, max=10.0)
        
        dist = Normal(mean, std)
        # Uses rsample for reparameterization trick gradients
        action = dist.rsample()
        
        # Squash to [-1, 1] bounds (Tanh transformation)
        # Note: True PPO/Dreamer corrects log_prob for tanh transform, but a rough estimate is fine for now
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_squashed = torch.tanh(action)
        
        # Correction term for Tanh squashing
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(dim=-1)
        
        return action_squashed, log_prob

class Critic(nn.Module):
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Outputs Expected Free Energy (Value)
        )

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch], dim=-1)
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=32, stoch_classes=32, action_dim=10):
        super().__init__()
        in_features = deter_dim + (stoch_dim * stoch_classes)
        self.actor = Actor(in_features, action_dim)
        self.critic = Critic(in_features)

    def forward(self, deter, stoch):
        action, _ = self.actor(deter, stoch)
        value = self.critic(deter, stoch)
        return action, value
