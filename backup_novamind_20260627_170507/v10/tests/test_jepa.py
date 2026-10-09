import torch
import pytest
from novamind.v10.models.jepa import JEPATrunk, JEPAPredictor

def test_jepa_trunk_forward():
    batch_size = 2
    seq_len = 16
    embed_dim = 256
    
    trunk = JEPATrunk(embed_dim=embed_dim, num_layers=2)
    # Mock sequence embeddings from any modality (Vision, Audio, Text)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # We can pass context inputs
    out = trunk(x)
    assert out.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}"

def test_jepa_predictor():
    batch_size = 2
    seq_len = 8
    embed_dim = 256
    
    predictor = JEPAPredictor(embed_dim=embed_dim, num_layers=1)
    
    # Context representations from Trunk
    context = torch.randn(batch_size, seq_len, embed_dim)
    
    # In JEPA, predictor receives context + positional embeddings for targets to predict targets
    # For simplicity of test, predict the targets directly given target positions
    pred = predictor(context)
    
    # Output matches sequence dimensions expected to be evaluated against EMAs
    assert pred.shape == (batch_size, seq_len, embed_dim)
