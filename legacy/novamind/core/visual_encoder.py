"""
VisualEncoder — Encoder visual com preservacao de posicao via DINOv2.

Diferenca do ResNet18 anterior:
  - Preserva posicao espacial dos objetos (patches ViT)
  - Agrupa patches em regioes (ex: 4 quadrantes)
  - Bind de cada regiao com vetor de posicao
  - Bundle final preserva "o que" + "onde"

Uso:
    ve = VisualEncoder()
    hdv = ve.image_to_hdv("foto.jpg")
    # hdv codifica tanto o conteudo quanto a posicao
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Optional

DIM = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Vetores de posicao (fixos, 8 regioes) ──────────────────
np.random.seed(48)
POSITION_VECTORS = {
    "top_left":       np.sign(np.random.randn(DIM)).astype(np.float32),
    "top_center":     np.sign(np.random.randn(DIM)).astype(np.float32),
    "top_right":      np.sign(np.random.randn(DIM)).astype(np.float32),
    "middle_left":    np.sign(np.random.randn(DIM)).astype(np.float32),
    "middle_center":  np.sign(np.random.randn(DIM)).astype(np.float32),
    "middle_right":   np.sign(np.random.randn(DIM)).astype(np.float32),
    "bottom_left":    np.sign(np.random.randn(DIM)).astype(np.float32),
    "bottom_center":  np.sign(np.random.randn(DIM)).astype(np.float32),
    "bottom_right":   np.sign(np.random.randn(DIM)).astype(np.float32),
    "global":         np.sign(np.random.randn(DIM)).astype(np.float32),
}

# Projecao: patch DINOv2 (384D ViT-S ou 768D ViT-B) → 10000D
np.random.seed(49)


def _make_proj(in_dim: int) -> np.ndarray:
    return np.random.randn(DIM, in_dim).astype(np.float32) / np.sqrt(in_dim)


class VisualEncoder:
    """Encoder visual com posicao via DINOv2.

    Processa imagem → patches DINOv2 → agrupa em regioes →
    HDV por regiao → bind com posicao → bundle final.
    """

    def __init__(self, model_name: str = "dinov2_vits14"):
        self.device = torch.device(DEVICE)
        # Patch grid: DINOv2 ViT-S/14 → 16x16 patches para 224x224
        self.patch_grid = (224 // 14, 224 // 14)  # (16, 16)
        self._load_model(model_name)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, model_name: str):
        """Carrega modelo DINOv2."""
        print(f"[VIS] Carregando {model_name}...")
        if model_name == "dinov2_vits14":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
            self.patch_dim = 384
        elif model_name == "dinov2_vitb14":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(self.device)
            self.patch_dim = 768
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")
        self.model.eval()
        self.proj = _make_proj(self.patch_dim)
        print(f"[VIS] DINOv2 pronto ({self.patch_dim}D, {self.patch_grid})")

    def get_patches(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extrai patches DINOv2 com posicao preservada.

        Returns:
            patches: (n_patches, patch_dim) numpy array
            grid: (h_patches, w_patches)
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # DINOv2 retorna [CLS] + patch tokens
            tokens = self.model.forward_features(image_tensor)["x_norm_patchtokens"]
            # tokens shape: (1, n_patches, patch_dim)
            patches = tokens[0].cpu().numpy()  # (n_patches, patch_dim)

        h, w = self.patch_grid
        return patches.reshape(h, w, self.patch_dim)

    def image_to_hdv(self, image: Image.Image, position_aware: bool = True) -> np.ndarray:
        """Imagem → HDV com posicao (preserva estrutura espacial).

        Se position_aware=False, retorna media simples (compatibilidade).
        """
        img_t = self.transform(image).to(self.device)
        patches = self.get_patches(img_t)  # (16, 16, patch_dim)

        if not position_aware:
            # Fallback: media de todos os patches
            avg = patches.mean(axis=(0, 1))
            hdv = np.sign(self.proj @ avg)
            hdv[hdv == 0] = 1
            return hdv.astype(np.float32)

        # Agrupa patches em 9 regioes (3x3 grid)
        h, w, pd = patches.shape
        rows = np.array_split(patches, 3, axis=0)  # 3 grupos de linhas
        region_hdvs = []

        for row_idx, row_group in enumerate(rows):
            cols = np.array_split(row_group, 3, axis=1)  # 3 grupos de colunas
            for col_idx, col_group in enumerate(cols):
                # Media dos patches nesta regiao
                region_patches = col_group.reshape(-1, pd)
                avg_patch = region_patches.mean(axis=0)

                # Projeta para HDV
                hdv_region = np.sign(self.proj @ avg_patch)
                hdv_region[hdv_region == 0] = 1

                # Nome da posicao
                pos_names = [
                    ["top_left", "top_center", "top_right"],
                    ["middle_left", "middle_center", "middle_right"],
                    ["bottom_left", "bottom_center", "bottom_right"],
                ]
                pos_name = pos_names[row_idx][col_idx]

                # Bind: HDV_regiao * HDV_posicao
                bound = np.sign(hdv_region * POSITION_VECTORS[pos_name])
                bound[bound == 0] = 1
                region_hdvs.append(bound.astype(np.float32))

        # Bundle: soma + sign de todas as regioes
        result = np.sign(np.sum(region_hdvs, axis=0))
        result[result == 0] = 1
        return result.astype(np.float32)

    def image_to_hdv_simple(self, image: Image.Image) -> np.ndarray:
        """Wrapper para compatibilidade."""
        return self.image_to_hdv(image, position_aware=True)

    def get_region_hdvs(self, image: Image.Image) -> dict:
        """Retorna HDVs por regiao (para debug/analise)."""
        img_t = self.transform(image).to(self.device)
        patches = self.get_patches(img_t)
        h, w, pd = patches.shape
        rows = np.array_split(patches, 3, axis=0)
        result = {}
        pos_names = [
            ["top_left", "top_center", "top_right"],
            ["middle_left", "middle_center", "middle_right"],
            ["bottom_left", "bottom_center", "bottom_right"],
        ]
        for ri, row_group in enumerate(rows):
            cols = np.array_split(row_group, 3, axis=1)
            for ci, col_group in enumerate(cols):
                avg = col_group.reshape(-1, pd).mean(axis=0)
                hdv = np.sign(self.proj @ avg)
                hdv[hdv == 0] = 1
                result[pos_names[ri][ci]] = hdv.astype(np.float32)
        return result


# ─── Teste ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))

    ve = VisualEncoder("dinov2_vits14")

    # Teste com imagem gerada (ARC grid)
    from PIL import ImageDraw
    img = Image.new("RGB", (224, 224), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Quadrado vermelho no canto superior esquerdo
    draw.rectangle([10, 10, 100, 100], fill=(255, 0, 0))
    # Quadrado azul no canto inferior direito
    draw.rectangle([120, 120, 210, 210], fill=(0, 0, 255))

    t0 = time.time()
    hdv = ve.image_to_hdv(img)
    print(f"HDV computation: {time.time()-t0:.3f}s")
    print(f"HDV shape: {hdv.shape}, unique: {np.unique(hdv)}")

    # Teste com posicao diferente (vermelho embaixo)
    img2 = Image.new("RGB", (224, 224), (240, 240, 240))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([10, 120, 100, 210], fill=(255, 0, 0))  # vermelho embaixo
    draw2.rectangle([120, 10, 210, 100], fill=(0, 0, 255))   # azul em cima

    hdv2 = ve.image_to_hdv(img2)
    cos = np.dot(hdv, hdv2) / DIM
    print(f"\n'vermelho em cima, azul embaixo' vs 'vermelho embaixo, azul em cima'")
    print(f"Cosine: {cos:.4f} (baixo = posicoes diferentes)")

    # Comparacao: media simples (sem posicao)
    hdv_flat = ve.image_to_hdv(img, position_aware=False)
    hdv2_flat = ve.image_to_hdv(img2, position_aware=False)
    cos_flat = np.dot(hdv_flat, hdv2_flat) / DIM
    print(f"Cosine (sem posicao): {cos_flat:.4f} (alto = posicoes ignoradas)")

    # Teste ARC
    print(f"\nRegioes:")
    regions = ve.get_region_hdvs(img)
    for name, r_hdv in regions.items():
        print(f"  {name}: {r_hdv[:5]}")
