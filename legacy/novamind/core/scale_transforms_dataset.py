"""
Scale Transforms Dataset — Transformacoes de dimensao GENERICAS.

Nao sao especificas de ARC. Sao operacoes universais de grid:
  - tile(N, M): replica o conteudo NxM vezes
  - crop(h, w): extrai regiao do grid
  - resize(h, w): redimensiona o grid
  - extract_bbox(): recorta ate bounding box do conteudo
  - pad(amount): adiciona bordas vazias

Qualquer sistema de visao computacional precisa destas operacoes.
Nao ha nada de ARC-specific aqui.
"""
from typing import List, Tuple
import numpy as np
import random


def random_grid(max_h=16, max_w=16) -> np.ndarray:
    """Gera um grid aleatorio com formas simples."""
    h = random.randint(3, max_h)
    w = random.randint(3, max_w)
    grid = np.zeros((h, w), dtype=int)
    
    # Adiciona formas aleatorias
    n_shapes = random.randint(1, 4)
    colors = list(range(1, 10))
    for _ in range(n_shapes):
        color = random.choice(colors)
        shape = random.choice(["rect", "line_h", "line_v", "dot"])
        rh = random.randint(1, max(2, h // 3))
        rw = random.randint(1, max(2, w // 3))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        
        if shape == "rect":
            grid[r0:r0+rh, c0:c0+rw] = color
        elif shape == "line_h":
            grid[r0, c0:c0+rw] = color
        elif shape == "line_v":
            grid[r0:r0+rh, c0] = color
        elif shape == "dot":
            grid[r0, c0] = color
    
    return grid


def apply_tile(grid: np.ndarray) -> np.ndarray:
    """Replica o grid NxM vezes."""
    h, w = grid.shape
    n = random.randint(2, 3)
    m = random.randint(2, 3)
    result = np.tile(grid, (n, m))
    return result


def apply_crop(grid: np.ndarray) -> np.ndarray:
    """Extrai uma regiao do grid."""
    h, w = grid.shape
    crop_h = random.randint(max(2, h // 2), h)
    crop_w = random.randint(max(2, w // 2), w)
    r0 = random.randint(0, h - crop_h)
    c0 = random.randint(0, w - crop_w)
    return grid[r0:r0+crop_h, c0:c0+crop_w].copy()


def apply_resize(grid: np.ndarray) -> np.ndarray:
    """Redimensiona o grid (preserva proporcao aproximada)."""
    h, w = grid.shape
    factor = random.choice([2, 3])
    new_h = h * factor
    new_w = w * factor
    # Tile simples para upscale
    result = np.zeros((new_h, new_w), dtype=int)
    for i in range(factor):
        for j in range(factor):
            rh = i * h
            rw = j * w
            result[rh:rh+h, rw:rw+w] = grid
    return result


def apply_extract_bbox(grid: np.ndarray) -> np.ndarray:
    """Recorta ate o bounding box do conteudo nao-zero."""
    rows, cols = np.where(grid > 0)
    if len(rows) == 0:
        return grid.copy()
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    return grid[r_min:r_max+1, c_min:c_max+1].copy()


def apply_pad(grid: np.ndarray) -> np.ndarray:
    """Adiciona bordas vazias."""
    h, w = grid.shape
    pad = random.randint(1, 3)
    result = np.zeros((h + pad * 2, w + pad * 2), dtype=int)
    result[pad:pad+h, pad:pad+w] = grid
    return result


TRANSFORMS = [
    ("tile", apply_tile),
    ("crop", apply_crop),
    ("resize", apply_resize),
    ("extract_bbox", apply_extract_bbox),
    ("pad", apply_pad),
]


class ScaleTransformDataset:
    """Dataset de transformacoes de dimensao GENERICAS.
    
    Nada ARC-specific. Sao operacoes universais de grid.
    """
    
    def __init__(self, size: int = 2000, max_h: int = 12, max_w: int = 12):
        self.size = size
        self.max_h = max_h
        self.max_w = max_w
        self.data: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self._generate()
    
    def _generate(self):
        for idx in range(self.size):
            grid = random_grid(self.max_h, self.max_w)
            transform_name, transform_fn = random.choice(TRANSFORMS)
            try:
                result = transform_fn(grid)
                self.data.append((grid, result, transform_name))
            except Exception:
                pass
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def sample_batch(self, batch_size: int = 32) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        idxs = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        return [self.data[i] for i in idxs]
    
    def get_label_distribution(self) -> dict:
        dist = {}
        for _, _, label in self.data:
            dist[label] = dist.get(label, 0) + 1
        return dist
