import torch
import pytest
from novamind.v10.models.text_encoder import TextEncoder
from novamind.v10.tokenizer import Tokenizer

def test_embedding_weight_matrix():
    # Tokenizer is expected to have exactly 16384 vocab_size
    tokenizer = Tokenizer()
    encoder = TextEncoder(vocab_size=tokenizer.vocab_size, d_model=256)
    
    assert encoder.embedding.weight.shape == (16384, 256), f"Expected (16384, 256) but got {encoder.embedding.weight.shape}"

def test_forward_pass_shape():
    encoder = TextEncoder(vocab_size=16384, d_model=256)
    batch_size = 4
    seq_len = 16
    
    # Mock input tensor [batch_size, seq_len]
    mock_input = torch.randint(0, 16384, (batch_size, seq_len))
    
    output = encoder(mock_input)
    
    assert output.shape == (batch_size, seq_len, 256), f"Expected {(batch_size, seq_len, 256)} but got {output.shape}"
