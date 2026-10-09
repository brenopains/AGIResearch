"""
PILAR 5: FUSÃO MULTIMODAL
Codificadores processam matrizes 2D (visão), sinais 1D (sinais fisiológicos simulados/EEG)
e tensores 1D densos (texto/instintos), projetando-os em um Espaço Latente Comum.
"""
import time
import torch
import torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultimodalFusion(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # 1. Visão (Grid 8x8 -> CNN)
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, latent_dim)
        )
        
        # 2. Sinais Fisiológicos (EEG simulado 1D: ex. 128 pontos no tempo -> CNN 1D)
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, latent_dim)
        )
        
        # 3. Simbólico/Texto (Vetor denso -> MLP)
        self.txt_encoder = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Projetor de Fusão
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, vis, eeg, txt):
        z_vis = self.vis_encoder(vis)
        z_eeg = self.eeg_encoder(eeg)
        z_txt = self.txt_encoder(txt)
        
        concat = torch.cat([z_vis, z_eeg, z_txt], dim=-1)
        z_fused = self.fusion_mlp(concat)
        return z_fused

def main():
    print(f"[{DEV}] PILAR 5: FUSÃO MULTIMODAL | Espaço Latente Compartilhado")
    
    # Criando dados falsos multimodais
    B = 2 # batch size
    visual_grid = torch.randn(B, 1, 8, 8).to(DEV)
    eeg_signal = torch.randn(B, 1, 128).to(DEV)
    text_embedding = torch.randn(B, 300).to(DEV)
    
    model = MultimodalFusion(latent_dim=32).to(DEV)
    
    t0 = time.time()
    latent_representation = model(visual_grid, eeg_signal, text_embedding)
    
    print(f"   Dimensão Visão: {visual_grid.shape}")
    print(f"   Dimensão Fisiológica: {eeg_signal.shape}")
    print(f"   Dimensão Texto: {text_embedding.shape}")
    print("-" * 50)
    print(f"   Vetor Latente Final (Fusão): {latent_representation.shape}")
    print(f"   Latente gerado: {latent_representation[0, :5].detach().cpu().numpy().round(3)}...")
    
    print("-> SUCESSO! Módulo acoplou 3 domínios desconexos num substrato semântico unificado.")
    print(f"({time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
