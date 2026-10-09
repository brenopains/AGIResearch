import torch
import pytest
from novamind.v10.models.rssm import RSSM

def test_rssm_shapes():
    rssm = RSSM(action_dim=10, embed_dim=256, stoch_dim=32, stoch_classes=32, deter_dim=512)
    batch_size = 4
    
    # Initial state
    prev_state = rssm.initial_state(batch_size)
    
    # Action (one-hot or continuous)
    prev_action = torch.zeros(batch_size, 10)
    
    # Observation embedding from JEPA trunk or encoders
    embed = torch.randn(batch_size, 256)
    
    prior, posterior, current_state = rssm.step(prev_state, prev_action, embed)
    
    # Acceptance tests
    assert prior['logits'].shape == (batch_size, 32, 32), "Prior must output logits for categorical latents"
    assert prior['stoch'].shape == (batch_size, 32, 32), "Prior must output sampled stochastic states"
    
    assert posterior['logits'].shape == (batch_size, 32, 32), "Posterior must output logits for categorical latents"
    assert posterior['stoch'].shape == (batch_size, 32, 32), "Posterior must output sampled stochastic states"
    
    assert current_state['deter'].shape == (batch_size, 512), "Deterministic recurrent state shape mismatch"

def test_rssm_imagine_shapes():
    rssm = RSSM()
    batch_size = 2
    horizon = 15
    
    prev_state = rssm.initial_state(batch_size)
    
    # Generate random imagined actions
    action_seq = torch.zeros(horizon, batch_size, 10)
    
    states, priors = rssm.imagine(prev_state, action_seq)
    
    assert len(states) == horizon
    assert states[0]['stoch'].shape == (batch_size, 32, 32)
