"""Linguagem espacial pequena: correspondência móvel e deslocamento relativo.

O gerador recebe um resíduo sensorial, não nomes de objetos. Seus fragmentos
são parâmetros de um programa executável. Correspondências não estabelecem
identidade de objetos: o resultado é força e distância entre dois fragmentos.
"""

from __future__ import annotations

from collections import deque

import numpy as np

COMPONENT_FRACTION = 0.25
MATCH_TOLERANCE = 1e-7
CHANNELS = ("positive_strength", "negative_strength", "relative_x", "relative_y")


def fragment(residual, sign):
    weights = np.maximum(np.asarray(residual, dtype=np.float64) * sign, 0)
    maximum = float(weights.max())
    if maximum <= 1e-12:
        return None
    mask = weights >= COMPONENT_FRACTION * maximum
    visited = np.zeros_like(mask)
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        points = []
        while queue:
            row, column = queue.popleft()
            points.append((row, column))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = row + dy, column + dx
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        components.append(points)
    points = max(components, key=lambda p: sum(weights[y, x] ** 2 for y, x in p))
    ys, xs = zip(*points)
    y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
    kernel = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    for y, x in points:
        kernel[y - y0, x - x0] = weights[y, x]
    kernel /= np.linalg.norm(kernel)
    return {"kernel": kernel.tolist(), "source_bounds": [y0, x0, y1, x1],
            "source_component_pixels": len(points)}


def make_readout(residual, seed):
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2 or not np.all(np.isfinite(residual)):
        raise ValueError("resíduo deve ser matriz sensorial finita")
    positive, negative = fragment(residual, 1), fragment(residual, -1)
    rng = np.random.default_rng(seed)
    controls = []
    for item in (positive, negative):
        if item is None:
            controls.append(None)
        else:
            shape = np.asarray(item["kernel"]).shape
            kernel = np.abs(rng.normal(size=shape))
            kernel /= np.linalg.norm(kernel)
            controls.append({"kernel": kernel.tolist(), "source_bounds": item["source_bounds"],
                             "source_component_pixels": int(np.prod(shape))})
    return {"op": "signed_patch_matches_and_offset", "shape": list(residual.shape),
            "channels": list(CHANNELS), "positive": positive, "negative": negative,
            "random_positive": controls[0], "random_negative": controls[1],
            "component_fraction": COMPONENT_FRACTION,
            "match_tolerance": MATCH_TOLERANCE,
            "scope": "fragment_correspondences_not_object_identity"}


def match(images, template, tolerance):
    n, height, width = images.shape
    strengths = np.zeros(n)
    centers = np.zeros((n, 2))
    if template is None:
        return strengths, centers
    kernel = np.asarray(template["kernel"], dtype=np.float64)
    kh, kw = kernel.shape
    if kh > height or kw > width:
        raise ValueError("fragmento maior que a observação")
    for start in range(0, n, 64):
        chunk = images[start:start + 64]
        windows = np.lib.stride_tricks.sliding_window_view(chunk, (kh, kw), axis=(1, 2))
        scores = np.einsum("nyxij,ij->nyx", windows, kernel, optimize=False)
        flattened = scores.reshape(len(chunk), -1)
        values = flattened.max(1)
        strengths[start:start + len(chunk)] = values
        tied = scores >= values[:, None, None] - tolerance
        count = tied.sum((1, 2))
        rows = np.arange(height - kh + 1)
        columns = np.arange(width - kw + 1)
        centers[start:start + len(chunk), 0] = (tied.sum(1) @ columns) / count + (kw - 1) / 2
        centers[start:start + len(chunk), 1] = (tied.sum(2) @ rows) / count + (kh - 1) / 2
    strengths[strengths <= tolerance] = 0
    return strengths, centers


def execute(readout, effects, random=False):
    if readout["op"] != "signed_patch_matches_and_offset":
        raise ValueError("operador fora da linguagem")
    effects = np.asarray(effects, dtype=np.float64)
    if not np.all(np.isfinite(effects)):
        raise ValueError("observação não finita")
    shape = tuple(readout["shape"])
    images = effects.reshape(-1, *shape)
    prefix = "random_" if random else ""
    tolerance = float(readout["match_tolerance"])
    positive, pcenter = match(np.maximum(images, 0), readout[prefix + "positive"], tolerance)
    negative, ncenter = match(np.maximum(-images, 0), readout[prefix + "negative"], tolerance)
    relative = (pcenter - ncenter) / np.asarray([shape[1], shape[0]])
    relative[(positive <= 1e-12) | (negative <= 1e-12)] = 0
    return np.column_stack((positive, negative, relative))
