import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from novamind.v10.models.maml import MAMLWrapper

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.fc(x)

def test_maml_adaptation():
    model = SimpleNet()
    maml = MAMLWrapper(model, inner_lr=0.1)
    
    # Task data
    x_support = torch.randn(4, 10)
    y_support = torch.ones(4, 1)
    
    x_query = torch.randn(4, 10)
    y_query = torch.zeros(4, 1)
    
    # Inner loop
    def loss_fn(pred, y):
        return F.mse_loss(pred, y)
        
    adapted_params = maml.inner_loop(x_support, y_support, loss_fn)
    
    # Query performance
    # Apply adapted params explicitly to check shape structure
    pred_query = F.linear(x_query, adapted_params['fc.weight'], adapted_params['fc.bias'])
    meta_loss = loss_fn(pred_query, y_query)
    
    assert meta_loss.requires_grad, "MAML must preserve gradient graph from inner loop"
    
    # Check that outer loop doesn't error
    meta_loss.backward()
    
    for p in model.parameters():
        assert p.grad is not None, "Outer loop gradient failed to flow back to original parameters"
