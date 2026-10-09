import torch
import torch.nn as nn
import pytest
from novamind.v10.models.ewc import EWC

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def test_ewc_penalty_calculation():
    model = DummyModel()
    # Mock parameters and fisher
    old_params = {}
    fisher = {}
    for n, p in model.named_parameters():
        old_params[n] = p.clone().detach()
        # Random fisher, all positive
        fisher[n] = torch.rand_like(p) + 0.1
        
    ewc = EWC(model)
    ewc.register_task(old_params, fisher)
    
    # Change model parameters slightly
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
            
    # Calculate penalty
    penalty = ewc.penalty()
    
    assert penalty.shape == () 
    assert penalty.item() > 0.0, "EWC penalty should be positive when parameters drift"
    assert not torch.isnan(penalty)
    
def test_ewc_no_drift():
    model = DummyModel()
    old_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    
    ewc = EWC(model)
    ewc.register_task(old_params, fisher)
    
    # Calculate penalty with absolutely identical params
    penalty = ewc.penalty()
    
    assert penalty.item() == 0.0, "Penalty should be strictly 0 if parameters have not changed"
