"""
value_maps.py — Inducao de mapas de valor parametrizados de exemplos
======================================================================
As ops de valor existentes (invert_colors com 10-c, threshold_4 com 4)
tem constantes hardcoded. Aqui, INDUZIMOS os parametros dos exemplos.

Familias:
  - afim:     f(c) = a*c + b
  - permutacao: bijecao inferida diretamente
  - threshold:  f(c) = c se c > t senao 0
  - constante:  f(c) = k

Inducao: dados pares (in_pixel, out_pixel) dos exemplos do gap,
         ajusta cada familia e escolhe a de erro zero.

Isto resolve a raiz do T7: o campo de busca agora pode gerar
a primitiva correta mesmo quando os parametros variam.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ValueMapResult:
    """Resultado da inducao de um mapa de valor."""
    name: str = ""
    fn: Optional[Callable] = None
    params: Dict = None
    error: float = 1.0
    n_mapped: int = 0
    n_total: int = 0

    def perfect(self) -> bool:
        return self.error == 0.0 and self.n_mapped >= self.n_total

    def to_primitive_name(self) -> str:
        return f"valuemap_{self.name}"


def induce_value_map(examples: List[Tuple[np.ndarray, np.ndarray]]) -> ValueMapResult:
    """
    Induz o mapa de valor que explica a transformacao pixel a pixel
    entre grids de entrada e saida.

    Args:
        examples: lista de (input_grid, output_grid)

    Returns:
        ValueMapResult com a melhor familia encontrada
    """
    if not examples:
        return ValueMapResult(name="identity", fn=lambda c: c, error=0.0)

    pairs = _extract_value_pairs(examples)
    if len(pairs) < 2:
        return ValueMapResult(name="identity", fn=lambda c: c, error=1.0)

    candidates = [
        _try_affine(pairs),
        _try_permutation(pairs),
        _try_threshold(pairs),
        _try_constant(pairs),
    ]

    best = min(candidates, key=lambda r: r.error)
    return best


def _extract_value_pairs(examples) -> List[Tuple[int, int]]:
    """Extrai pares (valor_entrada, valor_saida) de todos os exemplos."""
    pairs = []
    for inp, out in examples:
        if inp.shape != out.shape:
            continue
        h, w = inp.shape
        for y in range(h):
            for x in range(w):
                v_in = int(inp[y, x])
                v_out = int(out[y, x])
                if v_in != 0 or v_out != 0:
                    pairs.append((v_in, v_out))
    return pairs


def _try_affine(pairs: List[Tuple[int, int]]) -> ValueMapResult:
    """Tenta f(c) = a*c + b. Resolve a, b de dois pares."""
    unique = list(set(pairs))
    min_error = 1.0
    best_a = 1.0
    best_b = 0.0

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            x1, y1 = unique[i]
            x2, y2 = unique[j]
            if x1 == x2:
                continue
            a = (y1 - y2) / (x1 - x2)
            b = y1 - a * x1

            errors = []
            for v_in, v_out in pairs:
                pred = a * v_in + b
                errors.append(abs(pred - v_out))
            mean_err = np.mean(errors)

            if mean_err < min_error:
                min_error = mean_err
                best_a = a
                best_b = b

    if min_error > 0.5:
        return ValueMapResult(name="affine", error=1.0)

    def fn(c):
        return int(round(best_a * c + best_b))

    n_correct = sum(1 for v_in, v_out in pairs if fn(v_in) == v_out)
    return ValueMapResult(
        name="affine",
        fn=fn,
        params={"a": best_a, "b": best_b},
        error=round(min_error, 4),
        n_mapped=n_correct,
        n_total=len(pairs),
    )


def _try_permutation(pairs: List[Tuple[int, int]]) -> ValueMapResult:
    """Tenta bijecao direta: mapeia cada valor_in para valor_out."""
    mapping = {}
    consistent = True
    for v_in, v_out in pairs:
        if v_in in mapping:
            if mapping[v_in] != v_out:
                consistent = False
                break
        else:
            mapping[v_in] = v_out

    if not consistent:
        return ValueMapResult(name="permutation", error=1.0)

    def fn(c):
        return mapping.get(c, c)

    n_correct = sum(1 for v_in, v_out in pairs if fn(v_in) == v_out)
    return ValueMapResult(
        name="permutation",
        fn=fn,
        params={"mapping": mapping},
        error=0.0 if consistent else 1.0,
        n_mapped=n_correct,
        n_total=len(pairs),
    )


def _try_threshold(pairs: List[Tuple[int, int]]) -> ValueMapResult:
    """Tenta f(c) = c se c > t senao 0."""
    vals_in = [p[0] for p in pairs]
    vals_out = [p[1] for p in pairs]
    unique_in = sorted(set(vals_in))

    min_error = 1.0
    best_t = 0

    for t in unique_in:
        errors = []
        for v_in, v_out in pairs:
            pred = v_in if v_in > t else 0
            errors.append(abs(pred - v_out))
        mean_err = np.mean(errors)
        if mean_err < min_error:
            min_error = mean_err
            best_t = t

    if min_error > 0.5:
        return ValueMapResult(name="threshold", error=1.0)

    def fn(c):
        return c if c > best_t else 0

    n_correct = sum(1 for v_in, v_out in pairs if fn(v_in) == v_out)
    return ValueMapResult(
        name="threshold",
        fn=fn,
        params={"t": best_t},
        error=round(min_error, 4),
        n_mapped=n_correct,
        n_total=len(pairs),
    )


def _try_constant(pairs: List[Tuple[int, int]]) -> ValueMapResult:
    """Tenta f(c) = k. k e o valor de saida mais frequente."""
    out_vals = [p[1] for p in pairs]
    if not out_vals:
        return ValueMapResult(name="constant", error=1.0)
    k = max(set(out_vals), key=out_vals.count)

    def fn(c):
        return k

    errors = [abs(fn(v_in) - v_out) for v_in, v_out in pairs]
    mean_err = np.mean(errors) if errors else 1.0
    n_correct = sum(1 for v_in, v_out in pairs if fn(v_in) == v_out)
    return ValueMapResult(
        name="constant",
        fn=fn,
        params={"k": k},
        error=round(mean_err, 4),
        n_mapped=n_correct,
        n_total=len(pairs),
    )
