import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    # Acceptance criteria (c): relies on 1D convolutions.
    def __init__(self, in_channels: int = 80, d_model: int = 256):
        super().__init__()
        
        # We need to map time dimension from 200 down to 50. 
        # A combination of strides achieving a 4x reduction (e.g. two convs with stride 2) works.
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=4, stride=2, padding=1), # 200 -> 100
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=4, stride=2, padding=1),     # 100 -> 50
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F, T] -> [Batch, Features(Mel-Bins=80), Time=200]
        """
        z = self.encoder(x) # [B, d_model, 50]
        
        # Output should be sequence-first or batch-first [B, L, D] for transformers
        return z.permute(0, 2, 1) # [B, 50, d_model]
