"""
Geometry & Transformation Dataset — ARC-style grids sinteticos
==============================================================
Gera grids com formas geometricas e transformacoes conhecidas
para treinar o modelo a reconhecer operacoes estruturais.

Tipos de grid gerado:
  1. Formas geometricas (quadrado, circulo, triangulo, linha, L)
  2. Transformacoes: rotate, mirror, translate, scale, recolor, delete, copy
  3. Relacoes logicas: above, below, inside, outside, count
  4. Composicao: combinacao de 2+ transformacoes

Uso:
    dataset = GeometryDataset(size=10000)
    grid, label = dataset[idx]
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


COLORS = list(range(1, 10))  # 1-9 (0 = background)
COLOR_NAMES = ["black", "blue", "red", "green", "yellow",
               "grey", "fuchsia", "orange", "azure", "maroon"]


# ─── Shape Generators ──────────────────────────────────────

def random_color() -> int:
    return random.choice(COLORS)


def generate_rectangle(h: int, w: int) -> np.ndarray:
    """Grid com um retangulo preenchido."""
    grid = np.zeros((h, w), dtype=int)
    rh = random.randint(2, max(2, h // 2))
    rw = random.randint(2, max(2, w // 2))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    color = random_color()
    grid[r0:r0 + rh, c0:c0 + rw] = color
    return grid


def generate_square(h: int, w: int) -> np.ndarray:
    """Grid com um quadrado."""
    grid = np.zeros((h, w), dtype=int)
    size = random.randint(2, min(h, w) // 2)
    r0 = random.randint(0, h - size)
    c0 = random.randint(0, w - size)
    color = random_color()
    grid[r0:r0 + size, c0:c0 + size] = color
    return grid


def generate_line_h(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    length = random.randint(2, w - 1)
    r = random.randint(0, h - 1)
    c0 = random.randint(0, w - length)
    color = random_color()
    grid[r, c0:c0 + length] = color
    return grid


def generate_line_v(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    length = random.randint(2, h - 1)
    c = random.randint(0, w - 1)
    r0 = random.randint(0, h - length)
    color = random_color()
    grid[r0:r0 + length, c] = color
    return grid


def generate_L_shape(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    size = random.randint(2, min(h, w) // 2)
    r0 = random.randint(0, h - size - 1)
    c0 = random.randint(0, w - size - 1)
    color = random_color()
    # Horizontal + vertical
    grid[r0, c0:c0 + size] = color
    grid[r0:r0 + size, c0] = color
    return grid


def generate_multiple_shapes(h: int, w: int, n: int = 3) -> np.ndarray:
    """Grid com N formas geometricas."""
    grid = np.zeros((h, w), dtype=int)
    generators = [generate_rectangle, generate_square,
                  generate_line_h, generate_line_v, generate_L_shape]
    for _ in range(n):
        g = random.choice(generators)
        shape_grid = g(h, w)
        # Sobrepoe (objetos nao se sobrepoem na pratica porque as posicoes sao aleatorias)
        grid = np.where(grid > 0, grid, shape_grid)
    return grid


# ─── Transformation Functions ──────────────────────────────

def apply_rotate_90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-1)


def apply_rotate_180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2)


def apply_rotate_270(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=1)


def apply_mirror_h(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)


def apply_mirror_v(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid)


def apply_translate(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    dr = random.randint(-h // 4, h // 4)
    dc = random.randint(-w // 4, w // 4)
    shifted = np.zeros_like(grid)
    r_src = max(0, -dr)
    r_dst = max(0, dr)
    c_src = max(0, -dc)
    c_dst = max(0, dc)
    r_end = min(h, h - dr)
    c_end = min(w, w - dc)
    shifted[r_dst:r_dst + (r_end - r_src), c_dst:c_dst + (c_end - c_src)] = \
        grid[r_src:r_end, c_src:c_end]
    return shifted


def apply_recolor(grid: np.ndarray) -> np.ndarray:
    """Recolor: troca uma cor por outra."""
    grid = grid.copy()
    old_color = random.choice(COLORS)
    new_color = random.choice([c for c in COLORS if c != old_color])
    grid[grid == old_color] = new_color
    return grid


def apply_delete_color(grid: np.ndarray) -> np.ndarray:
    """Delete: remove objetos de uma cor."""
    grid = grid.copy()
    color = random.choice(COLORS)
    grid[grid == color] = 0
    return grid


def apply_copy(grid: np.ndarray) -> np.ndarray:
    """Copy: duplica todos os objetos, transladando."""
    h, w = grid.shape
    dr = random.randint(2, h // 3)
    dc = random.randint(2, w // 3)
    shifted = np.zeros_like(grid)
    shifted[dr:, dc:] = grid[:h - dr, :w - dc]
    # Combina original + copia
    result = np.where(grid > 0, grid, shifted)
    return result


def apply_crop(grid: np.ndarray) -> np.ndarray:
    """Crop: mantem apenas uma regiao."""
    h, w = grid.shape
    # Encontra bounding box dos objetos
    rows, cols = np.where(grid > 0)
    if len(rows) == 0:
        return grid
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    # Expande um pouco
    pad = random.randint(0, 2)
    r0 = max(0, r_min - pad)
    r1 = min(h, r_max + pad + 1)
    c0 = max(0, c_min - pad)
    c1 = min(w, c_max + pad + 1)
    cropped = np.zeros((h, w), dtype=int)
    cropped[:r1 - r0, :c1 - c0] = grid[r0:r1, c0:c1]
    return cropped


TRANSFORMATIONS = {
    "rotate_90": apply_rotate_90,
    "rotate_180": apply_rotate_180,
    "rotate_270": apply_rotate_270,
    "mirror_h": apply_mirror_h,
    "mirror_v": apply_mirror_v,
    "translate": apply_translate,
    "recolor": apply_recolor,
    "delete_color": apply_delete_color,
    "copy": apply_copy,
    "crop": apply_crop,
}

TRANSFORMATION_NAMES = list(TRANSFORMATIONS.keys())


# ─── Dataset Generator ─────────────────────────────────────

class GeometryTransformDataset:
    """Dataset de grids geometricos com transformacoes conhecidas.

    Cada exemplo: (input_grid, output_grid, transformation_name, params)
    """

    def __init__(self, size: int = 10000, h: int = 20, w: int = 20):
        self.size = size
        self.h = h
        self.w = w
        self.data: List[Tuple[np.ndarray, np.ndarray, str, dict]] = []
        self._generate()

    def _generate(self):
        shape_generators = [
            generate_rectangle, generate_square,
            generate_line_h, generate_line_v,
            generate_L_shape, generate_multiple_shapes,
        ]

        for idx in range(self.size):
            shape_fn = random.choice(shape_generators)
            if shape_fn == generate_multiple_shapes:
                input_grid = shape_fn(self.h, self.w, n=random.randint(3, 8))
            else:
                input_grid = shape_fn(self.h, self.w)

            # Garante que tem pelo menos um objeto
            if np.max(input_grid) == 0:
                input_grid = generate_rectangle(self.h, self.w)

            # Escolhe uma transformacao
            t_name = random.choice(TRANSFORMATION_NAMES)
            t_fn = TRANSFORMATIONS[t_name]

            # Aplica
            output_grid = t_fn(input_grid)

            self.data.append((input_grid, output_grid, t_name, {}))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, dict]:
        return self.data[idx]

    def sample_batch(self, batch_size: int = 32) -> List[Tuple[np.ndarray, np.ndarray, str, dict]]:
        idxs = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        return [self.data[i] for i in idxs]

    def get_label_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for _, _, label, _ in self.data:
            dist[label] = dist.get(label, 0) + 1
        return dist


# ─── Dataset de Relacoes Logicas ───────────────────────────

class RelationalDataset:
    """Dataset de relacoes entre objetos em grid.

    Exemplos:
      - "A above B" -> output mantem relacao
      - "A inside B" -> output verifica
      - "count(red) == 3" -> output mostra contagem
    """

    def __init__(self, size: int = 5000, h: int = 10, w: int = 10):
        self.size = size
        self.h = h
        self.w = w
        self.data: List[Tuple[np.ndarray, np.ndarray, str, dict]] = []
        self._generate()

    def _generate(self):
        relations = ["above", "below", "left_of", "right_of", "inside", "count"]
        for idx in range(self.size):
            rel = random.choice(relations)
            input_grid, output_grid = self._generate_pair(rel)
            self.data.append((input_grid, output_grid, rel, {}))

    def _generate_pair(self, relation: str) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.h, self.w
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)

        if relation == "above":
            c1, c2, c3 = random_color(), random_color(), random_color()
            col = random.randint(1, w - 2)
            r1, r2 = random.randint(0, h // 2 - 1), random.randint(h // 2, h - 1)
            inp[r1, col] = c1
            inp[r2, col] = c2
            out[r1, col] = c1
            out[r2, col] = c2
            # Add check: above relation
            out[0, 0] = c3 if r1 < r2 else 0

        elif relation == "inside":
            # Large rectangle with small object inside
            color_big = random_color()
            color_small = random_color()
            rh, rw = random.randint(4, h - 2), random.randint(4, w - 2)
            r0, c0 = random.randint(0, h - rh), random.randint(0, w - rw)
            inp[r0:r0 + rh, c0:c0 + rw] = color_big
            si, sj = random.randint(r0 + 1, r0 + rh - 2), random.randint(c0 + 1, c0 + rw - 2)
            inp[si, sj] = color_small
            out = inp.copy()

        elif relation == "count":
            color = random_color()
            n = random.randint(2, 5)
            positions = random.sample([(r, c) for r in range(h) for c in range(w)], n)
            for r, c in positions:
                inp[r, c] = color
            # Output: write count as color index (1-9)
            out[0, 0] = min(n, 9)

        else:
            # left_of, right_of, below: symmetric to above
            color = random_color()
            if relation in ("left_of", "right_of"):
                r = random.randint(0, h - 1)
                c1 = random.randint(0, w // 2 - 1)
                c2 = random.randint(w // 2, w - 1)
                inp[r, c1] = color
                inp[r, c2] = color
            else:  # below
                c = random.randint(0, w - 1)
                r1 = random.randint(0, h // 2 - 1)
                r2 = random.randint(h // 2, h - 1)
                inp[r1, c] = color
                inp[r2, c] = color
            out = inp.copy()

        return inp, out

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_batch(self, batch_size: int = 32) -> List[Tuple[np.ndarray, np.ndarray, str, dict]]:
        idxs = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        return [self.data[i] for i in idxs]


# ─── Principal ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Geometry Dataset...")
    geo = GeometryTransformDataset(size=1000, h=8, w=8)
    dist = geo.get_label_distribution()
    print(f"  Total: {len(geo)}")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")

    print("\nGenerating Relational Dataset...")
    rel = RelationalDataset(size=500)
    print(f"  Total: {len(rel)}")

    # Show one example
    inp, out, label, _ = geo[0]
    print(f"\nExample: {label}")
    print("Input:")
    print(inp)
    print("Output:")
    print(out)
