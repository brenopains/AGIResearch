import pytest
import torch
from novamind.v10.models.rssm import RSSM
from novamind.v10.models.imagination import ImaginationEnv

def test_imagination_rollout_shapes():
    rssm = RSSM()
    # Mock Reward model and value model for the rollouts. 
    # For now, just a dummy reward head that takes (deter, stoch) and outputs reward
    batch_size = 2
    horizon = 15
    
    start_state = rssm.initial_state(batch_size)
    
    # Simple Action Network (Actor)
    # Output logic for 10 dimension action space
    class MockActor(torch.nn.Module):
        def forward(self, deter, stoch):
            return torch.randn(deter.size(0), 10)
    
    actor = MockActor()
    env = ImaginationEnv(rssm, actor, horizon=horizon)
    
    # We rollout from the start_state
    rollout_states, actions = env.rollout(start_state)
    
    assert len(rollout_states) == horizon
    assert rollout_states[0]['deter'].shape == (batch_size, 512)
    assert actions.shape == (horizon, batch_size, 10)
