import torch
import pytest
from novamind.v10.models.audio_encoder import AudioEncoder

def test_audio_encoder_forward_shape():
    encoder = AudioEncoder(in_channels=80, d_model=256)
    batch_size = 4
    
    # Mock audio tensor [Batch, Channels (Mel-Bins), Time] -> [4, 80, 200]
    mock_audio = torch.randn(batch_size, 80, 200)
    
    # Acceptance criteria (a) & (b): Output sequence is fixed length [B, 50, 256], and forward succeeds
    output = encoder(mock_audio)
    
    assert output.shape == (batch_size, 50, 256), f"Expected {(batch_size, 50, 256)} but got {output.shape}"
