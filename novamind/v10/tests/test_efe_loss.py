import torch
import pytest
from novamind.v10.models.loss import compute_expected_free_energy

def test_efe_loss_calculation():
    batch_size = 4
    # Mock Reward (pragmatic value): shape [B]
    expected_reward = torch.randn(batch_size)
    
    # Mock priors (logits)
    prior_logits = torch.randn(batch_size, 32, 32)
    # Mock posteriors (logits), simulating higher certainty
    posterior_logits = prior_logits + torch.randn(batch_size, 32, 32) * 2.0
    
    efe = compute_expected_free_energy(expected_reward, prior_logits, posterior_logits, beta=0.1)
    
    # Acceptance criteria: Outputs a scalar EFE for each batch item or an aggregated mean
    assert efe.shape == (), f"Expected scalar loss but got {efe.shape}"
    
    # EFE should be quantifiable and non-nan
    assert not torch.isnan(efe)
