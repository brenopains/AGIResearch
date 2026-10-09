import torch
import pytest
from novamind.v10.models.symbolic import SymbolicHead, PrimitiveBase

def test_symbolic_head_generation():
    embed_dim = 128
    primitives = ['ADD', 'SUB', 'MAP', 'IF', '0', '1', 'VAR_X']
    
    head = SymbolicHead(embed_dim=embed_dim, primitives=primitives)
    
    batch_size = 2
    latent = torch.randn(batch_size, embed_dim)
    
    # Generate a program of max length 5
    programs, log_probs = head.generate_program(latent, max_len=5)
    
    assert len(programs) == batch_size
    assert len(programs[0]) <= 5
    
    # Verify the generated tokens belong to the primitive domain
    for token in programs[0]:
        assert token in primitives + ['<EOS>']
        
    assert log_probs.shape == (batch_size,)
    # Log prob is negative or zero
    assert torch.all(log_probs <= 0)
