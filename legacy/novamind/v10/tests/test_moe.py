import torch
import pytest
from novamind.v10.models.moe import SparseMoE

def test_moe_forward_shape():
    batch_size = 4
    seq_len = 16
    d_model = 256
    
    moe = SparseMoE(d_model=d_model, num_experts=8, top_k=2)
    x = torch.randn(batch_size, seq_len, d_model)
    
    out, aux_loss = moe(x)
    
    # Check shape
    assert out.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"
    
    # Check aux loss (load balancing)
    assert aux_loss.shape == (), f"Aux loss should be a scalar"
    assert aux_loss >= 0.0, "Aux loss must be non-negative"

def test_moe_routing():
    d_model = 64
    moe = SparseMoE(d_model=d_model, num_experts=4, top_k=1)
    x = torch.randn(2, d_model) # [2, 64]
    
    # We can check if it routes strictly to 1 expert by inspecting zeros in intermediate, 
    # but practically we verify it executes without error for top_k=1
    out, _ = moe(x)
    assert out.shape == (2, d_model)
