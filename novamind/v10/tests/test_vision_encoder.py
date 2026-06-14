import torch
import pytest
from novamind.v10.models.vision_encoder import VisionVQVAE

def test_vqvae_forward_shape():
    encoder = VisionVQVAE(num_embeddings=512, embedding_dim=64)
    batch_size = 4
    
    # Mock image tensor [B, C, H, W] -> [4, 3, 64, 64]
    mock_image = torch.randn(batch_size, 3, 64, 64)
    
    # Encoder outputs discrete tokens
    quantized_indices, quantized_loss = encoder(mock_image)
    
    # Acceptance criteria (a): Sequence of 64 discrete tokens [B, 64]
    # Since 64x64 image goes to 8x8 spatial resolution through 3 downsamples (or 2 downsamples of stride 2 = 16x16, wait. to get 8x8 it takes 3 downsamples of stride 2: 64->32->16->8)
    # Then 8x8 = 64 tokens.
    assert quantized_indices.shape == (batch_size, 64), f"Expected {(batch_size, 64)} but got {quantized_indices.shape}"
    assert quantized_indices.dtype == torch.long

def test_codebook_size():
    encoder = VisionVQVAE(num_embeddings=512, embedding_dim=64)
    # Acceptance criteria (b): Codebook size is exactly 512
    assert encoder.codebook.weight.shape == (512, 64), f"Codebook size is {encoder.codebook.weight.shape}, expected (512, 64)"
