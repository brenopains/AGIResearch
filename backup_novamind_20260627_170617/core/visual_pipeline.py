"""
Visual Pipeline: encoding reversivel e reconstrucao fiel de frames.

Pipeline:
  Frame [real] -> ResNet18 -> 512D features -> JL -> 10000D presign -> sign -> bipolar [memoria]
  Bipolar [memoria] -> pinv(JL) -> 512D rec -> decoder -> frame_rec [reconstruido]
  
  Verificacao: MSE/PSNR/SSIM entre frame real e reconstruido.
  
  A Forja opera no bipolar. O decoder ve o que o bind criou.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .coglang import Manifold


class ConvBlock(nn.Module):
    """Conv + BN + ReLU com skip residual."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + F.relu(self.bn(self.conv(x)))


class VisualDecoder(nn.Module):
    """Decoder 512D features -> imagem RGB 224x224 (~4.8M params, com residuais).

    Arquitetura:
      FC 512->4096, reshape 256x4x4,
      up 4->8->16->32->64->128->256, com ConvBlocks residuais apos cada up,
      sigmoid + interpolate 256->224.
    """
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256 * 4 * 4)

        # Upsample: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        ch = [256, 128, 64, 32, 16, 8]
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(ch[i], ch[i+1], 4, 2, 1) for i in range(len(ch)-1)
        ])
        self.blocks = nn.ModuleList([ConvBlock(c) for c in ch[1:]])
        self.to_rgb = nn.Conv2d(8, 3, 3, 1, 1)

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        for up, block in zip(self.up, self.blocks):
            x = block(F.relu(up(x)))
        x = torch.sigmoid(self.to_rgb(x))
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        return x


class VisualPipeline:
    """
    Pipeline visual bidirecional com verificacao de fidelidade.

    encode_frame: frame -> bipolar HDC + presign continuo + 512D features
    decode_bipolar: bipolar HDC -> imagem reconstruida (pinv + decoder)
    verify_fidelity: PSNR/SSIM entre original e reconstruido
    """

    def __init__(self, proj_visual: torch.Tensor, device: str = "cuda"):
        self.device = torch.device(device)
        self.proj = proj_visual.to(self.device).float()
        self.pinv = torch.linalg.pinv(self.proj)
        self.decoder = VisualDecoder().to(self.device)
        self.decoder.eval()

    def encode_frame(self, features: torch.Tensor
                     ) -> Tuple[Manifold, torch.Tensor, torch.Tensor]:
        """Frame -> bipolar + presign + features."""
        f = features.flatten().float().to(self.device)
        presign = torch.matmul(self.proj, f)
        bipolar = torch.sign(presign)
        bipolar[bipolar == 0] = 1
        return Manifold(bipolar, device=self.device), presign, f

    def decode_bipolar(self, bipolar_vec: torch.Tensor) -> torch.Tensor:
        """Vetor bipolar 10000D -> imagem 224x224 RGB [0,1]."""
        b = bipolar_vec.flatten().float().to(self.device)
        f_rec = torch.matmul(self.pinv, b)
        img = self.decoder(f_rec.unsqueeze(0))
        return img.squeeze(0)

    def reconstruct_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """512D features -> imagem."""
        return self.decoder(features.unsqueeze(0)).squeeze(0)

    def verify_fidelity(self, original: torch.Tensor,
                        reconstructed: torch.Tensor) -> dict:
        """PSNR + SSIM + MSE entre original e reconstruido."""
        o = original.float().cpu()
        r = reconstructed.float().cpu()
        mse = float(F.mse_loss(o, r))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
        ssim = self._ssim(o, r)
        return {"psnr": psnr, "ssim": ssim, "mse": mse}

    @staticmethod
    def _ssim(img1: torch.Tensor, img2: torch.Tensor, ws: int = 11) -> float:
        C1, C2 = 0.01**2, 0.03**2
        i1, i2 = img1.unsqueeze(0), img2.unsqueeze(0)
        mu1 = F.avg_pool2d(i1, ws, 1, 0)
        mu2 = F.avg_pool2d(i2, ws, 1, 0)
        s1 = F.avg_pool2d(i1**2, ws, 1, 0) - mu1**2
        s2 = F.avg_pool2d(i2**2, ws, 1, 0) - mu2**2
        s12 = F.avg_pool2d(i1*i2, ws, 1, 0) - mu1*mu2
        num = (2*mu1*mu2 + C1) * (2*s12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
        return float(num.mean() / den.mean())

    def train_decoder(self, dataloader, epochs: int = 10, lr: float = 1e-3):
        """Treina o decoder com pares (features, imagem) do dataloader."""
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for features, images in dataloader:
                f = features.to(self.device)
                i = images.to(self.device)
                pred = self.decoder(f)
                loss = loss_fn(pred, i)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(len(dataloader), 1)
            print(f"  [Decoder] Epoca {epoch+1}/{epochs} loss={avg:.6f}")
        self.decoder.eval()
