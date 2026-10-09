import torch
import pytest
from novamind.v10.models.actor_critic import ActorCritic

def test_actor_critic_shapes():
    ac = ActorCritic(deter_dim=512, stoch_dim=32, stoch_classes=32, action_dim=10)
    batch_size = 4
    
    deter = torch.randn(batch_size, 512)
    # Stoch is typically flattened before input [B, 32*32]
    stoch = torch.randn(batch_size, 32 * 32)
    
    action, action_log_prob = ac.actor(deter, stoch)
    
    # Acceptance criteria: Actor outputs action in correct bounds, and Value outputs scalar
    assert action.shape == (batch_size, 10)
    assert action_log_prob.shape == (batch_size,)
    
    value = ac.critic(deter, stoch)
    assert value.shape == (batch_size, 1), f"Expected Critic shape [4, 1], got {value.shape}"

def test_forward_tuple():
    ac = ActorCritic(deter_dim=512, stoch_dim=32, stoch_classes=32, action_dim=10)
    deter = torch.randn(2, 512)
    stoch = torch.randn(2, 32 * 32)
    
    action, value = ac(deter, stoch)
    assert action.shape == (2, 10)
    assert value.shape == (2, 1)
