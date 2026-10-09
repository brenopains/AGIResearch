"""
PrimitiveDiscovery — Descobre TIPOS NOVOS de transformacao da experiencia.

Nao ha classificador fixo. O sistema:
  1. Extrai ASSINATURA ESTRUTURAL do delta (razoes, proporcoes)
  2. Compara com assinaturas de tipos existentes
  3. Se similar a um tipo existente -> reforca
  4. Se NOVO -> Cria novo tipo com nome descritivo

Cada tipo pode gerar uma primitiva com executor generico.
"""
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, field
import math

from .primitive import Primitive, PrimitiveRegistry
from .delta_extractor import SceneDelta, ObjectDelta
from .primitives import get_primitive_factory


@dataclass
class TypeSignature:
    """Assinatura estrutural de um tipo de transformacao.
    
    Usada para comparar deltas e agrupar tipos similares.
    """
    object_ratio: float = 1.0       # n_out / n_in
    shape_h_ratio: float = 1.0      # out_h / in_h
    shape_w_ratio: float = 1.0      # out_w / in_w
    area_ratio: float = 1.0         # total_pixels_out / total_pixels_in
    color_preserved: float = 1.0    # fracao de cores preservadas
    centroid_dr: float = 0.0        # deslocamento centroide
    centroid_dc: float = 0.0
    connectivity_change: float = 0.0 # fracao de objetos que mudaram de forma
    n_examples: int = 0
    
    def similarity(self, other: 'TypeSignature') -> float:
        """Similaridade entre duas assinaturas (0-1)."""
        # Diferenca de razao de objetos
        obj_diff = abs(math.log2(max(0.01, self.object_ratio) / max(0.01, other.object_ratio)))
        obj_score = max(0, 1.0 - obj_diff / 3.0)
        
        # Diferenca de razao de shape
        h_diff = abs(math.log2(max(0.01, self.shape_h_ratio) / max(0.01, other.shape_h_ratio)))
        w_diff = abs(math.log2(max(0.01, self.shape_w_ratio) / max(0.01, other.shape_w_ratio)))
        shape_score = max(0, 1.0 - (h_diff + w_diff) / 6.0)
        
        # Diferenca de area
        area_diff = abs(math.log2(max(0.01, self.area_ratio) / max(0.01, other.area_ratio)))
        area_score = max(0, 1.0 - area_diff / 3.0)
        
        # Similaridade de cor
        color_score = 1.0 - abs(self.color_preserved - other.color_preserved)
        
        # Similaridade de centroide
        dr_diff = abs(self.centroid_dr - other.centroid_dr)
        dc_diff = abs(self.centroid_dc - other.centroid_dc)
        centroid_score = max(0, 1.0 - (dr_diff + dc_diff) / 10.0)
        
        # Ponderacao
        score = (obj_score * 0.25 + shape_score * 0.25 + area_score * 0.15 + 
                color_score * 0.15 + centroid_score * 0.20)
        return round(score, 4)
    
    def to_name(self) -> str:
        """Gera nome descritivo baseado na assinatura."""
        # Identifica padrao dominante
        if self.object_ratio > 1.5:
            return f"scale_{int(round(self.object_ratio))}x"
        if self.object_ratio < 0.5:
            return f"compress_{int(round(1/self.object_ratio))}x"
        if self.shape_h_ratio > 1.2 or self.shape_w_ratio > 1.2:
            return f"expand_{int(round(self.shape_h_ratio))}x{int(round(self.shape_w_ratio))}"
        if self.shape_h_ratio < 0.8 or self.shape_w_ratio < 0.8:
            return f"crop_{int(round(1/self.shape_h_ratio))}x{int(round(1/self.shape_w_ratio))}"
        if self.area_ratio > 1.5:
            return "tile"
        if abs(self.centroid_dr) > 2 or abs(self.centroid_dc) > 2:
            return "shift"
        return "transform"


def extract_signature(delta: SceneDelta, inp: np.ndarray = None, 
                       out: np.ndarray = None) -> TypeSignature:
    """Extrai assinatura estrutural de um delta."""
    in_objs = delta.n_input or (len(delta.objects_removed) + len(delta.objects_transformed) + 
                                 len(delta.objects_unchanged))
    out_objs = delta.n_output or (len(delta.objects_added) + len(delta.objects_transformed) + 
                                  len(delta.objects_unchanged))
    
    in_objs = max(1, in_objs)
    out_objs = max(1, out_objs)
    
    # Shape
    in_h = inp.shape[0] if inp is not None else 1
    in_w = inp.shape[1] if inp is not None else 1
    out_h = out.shape[0] if out is not None else 1
    out_w = out.shape[1] if out is not None else 1
    
    # Area (pixels nao-zero)
    in_px = int((inp > 0).sum()) if inp is not None else 1
    out_px = int((out > 0).sum()) if out is not None else 1
    
    # Cores preservadas
    in_colors = set(int(c) for c in np.unique(inp) if c != 0) if inp is not None else set()
    out_colors = set(int(c) for c in np.unique(out) if c != 0) if out is not None else set()
    preserved = len(in_colors & out_colors) / max(1, max(len(in_colors), len(out_colors))) if (in_colors or out_colors) else 1.0
    
    # Centroide
    def _center(g):
        rows, cols = np.where(g > 0) if g is not None else ([0], [0])
        if len(rows) == 0: return (0.0, 0.0)
        return (float(np.mean(rows)), float(np.mean(cols)))
    
    in_c = _center(inp)
    out_c = _center(out)
    
    # Conectividade
    n_shape_changed = sum(1 for o in delta.objects_transformed if o.shape_changed)
    total_transformed = max(1, len(delta.objects_transformed))
    conn_change = n_shape_changed / total_transformed
    
    return TypeSignature(
        object_ratio=out_objs / in_objs,
        shape_h_ratio=out_h / max(1, in_h),
        shape_w_ratio=out_w / max(1, in_w),
        area_ratio=max(out_px, 1) / max(in_px, 1),
        color_preserved=preserved,
        centroid_dr=out_c[0] - in_c[0],
        centroid_dc=out_c[1] - in_c[1],
        connectivity_change=conn_change,
    )


def _check_spatial_transform(inp: np.ndarray, out: np.ndarray) -> str:
    """Verifica se e transformacao espacial (rotate, flip)."""
    try:
        from .operator_proposals import detect_applicable_operators
        matches = detect_applicable_operators(inp, out)
        if matches:
            return matches[0]["name"]
    except Exception:
        pass
    return ""


class PrimitiveDiscovery:
    """Descobre TIPOS NOVOS de transformacao.
    
    Nao ha classificador fixo. Cada delta gera uma assinatura
    que e comparada com tipos existentes. Se for nova, cria tipo.
    """
    
    def __init__(self, registry: Optional[PrimitiveRegistry] = None):
        self.registry = registry or PrimitiveRegistry()
        from .delta_extractor import DeltaExtractor
        self.delta_extractor = DeltaExtractor()
        self._type_signatures: Dict[str, TypeSignature] = {}
        self._known_types: set = set()
        self._similarity_threshold = 0.6
    
    def observe(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[str]:
        """Observa par input/output, descobre/reforca tipo."""
        delta = self.delta_extractor.extract(input_grid, output_grid)
        if not delta.has_changes:
            return None
        
        # Extrai assinatura do delta
        sig = extract_signature(delta, input_grid, output_grid)
        
        # PASSO 1: Dynamic typing — cria tipo baseado na ESTRUTURA do delta
        struct_type = self._match_or_create(sig)
        
        if struct_type == "unknown":
            return None
        
        # PASSO 2: Se o tipo e novo e tem correspondente espacial, usa o executor dele
        spatial_exec = None
        spatial_name = _check_spatial_transform(input_grid, output_grid)
        if spatial_name:
            # Registra o tipo espacial TAMBEM como tipo separado (mais especifico)
            self._type_signatures[spatial_name] = sig
            self._ensure_primitive(spatial_name, input_grid, output_grid, is_spatial=True)
            self._known_types.add(spatial_name)
        
        # PASSO 3: Garante que o tipo dinamico tem primitiva
        self._ensure_primitive(struct_type, input_grid, output_grid)
        prim = self.registry.get(struct_type)
        if prim:
            prim.n_examples += 1
        
        self._known_types.add(struct_type)
        return struct_type
    
    def _match_or_create(self, sig: TypeSignature) -> str:
        """Encontra tipo existente similar ou cria novo."""
        best_match = None
        best_sim = 0
        
        for type_name, existing_sig in self._type_signatures.items():
            sim = sig.similarity(existing_sig)
            if sim > best_sim:
                best_sim = sim
                best_match = type_name
        
        if best_sim >= self._similarity_threshold and best_match:
            # Reforca tipo existente
            self._type_signatures[best_match] = TypeSignature(
                object_ratio=(self._type_signatures[best_match].object_ratio + sig.object_ratio) / 2,
                shape_h_ratio=(self._type_signatures[best_match].shape_h_ratio + sig.shape_h_ratio) / 2,
                shape_w_ratio=(self._type_signatures[best_match].shape_w_ratio + sig.shape_w_ratio) / 2,
                area_ratio=(self._type_signatures[best_match].area_ratio + sig.area_ratio) / 2,
                color_preserved=(self._type_signatures[best_match].color_preserved + sig.color_preserved) / 2,
                centroid_dr=(self._type_signatures[best_match].centroid_dr + sig.centroid_dr) / 2,
                centroid_dc=(self._type_signatures[best_match].centroid_dc + sig.centroid_dc) / 2,
                connectivity_change=(self._type_signatures[best_match].connectivity_change + sig.connectivity_change) / 2,
                n_examples=self._type_signatures[best_match].n_examples + 1,
            )
            return best_match
        
        # Cria NOVO tipo
        type_name = sig.to_name()
        # Garante unicidade
        if type_name in self._type_signatures:
            i = 2
            while f"{type_name}_{i}" in self._type_signatures:
                i += 1
            type_name = f"{type_name}_{i}"
        
        self._type_signatures[type_name] = sig
        return type_name
    
    def _ensure_primitive(self, struct_type: str, inp: np.ndarray = None, 
                          out: np.ndarray = None, is_spatial: bool = False):
        """Garante que existe primitiva para este tipo."""
        if self.registry.get(struct_type):
            return
        
        if is_spatial:
            # Para tipos espaciais confirmados, usa executor do operador
            from .operator_proposals import get_all_proposals
            for op in get_all_proposals():
                if op["name"] == struct_type:
                    fn = op["function"]
                    prim = Primitive(name=struct_type, structure_type=struct_type,
                                   description=op.get("description", ""))
                    def _me(f):
                        def _e(g, p):
                            try: return f(g)
                            except: return g.copy()
                        return _e
                    prim._execute = _me(fn)
                    prim.confidence = 0.5
                    self.registry.register(prim)
                    return
        
        # Cria primitiva generica com executor baseado na assinatura
        prim = Primitive(
            name=struct_type,
            structure_type=struct_type,
            description=f"dynamic type: {struct_type}",
            solver_fn=lambda i, o: {},
        )
        
        # Cria executor baseado na assinatura
        sig = self._type_signatures.get(struct_type)
        if sig:
            prim._execute = self._make_executor(sig)
        
        prim.confidence = 0.3
        self.registry.register(prim)
    
    def _make_executor(self, sig: TypeSignature):
        """Cria funcao de execucao baseada na assinatura do tipo."""
        def _exec(grid, params):
            try:
                h, w = grid.shape
                target_h = max(1, int(h * sig.shape_h_ratio))
                target_w = max(1, int(w * sig.shape_w_ratio))
                
                # Tile: repete padrao
                if "scale" in str(sig.to_name()) or "tile" in str(sig.to_name()):
                    if sig.shape_h_ratio > 1 or sig.shape_w_ratio > 1:
                        nh = int(round(sig.shape_h_ratio))
                        nw = int(round(sig.shape_w_ratio))
                        return np.tile(grid, (max(1, nh), max(1, nw)))
                
                # Crop: extrai regiao
                if "crop" in str(sig.to_name()) or "compress" in str(sig.to_name()):
                    if sig.shape_h_ratio < 1:
                        crop_h = max(1, int(h * sig.shape_h_ratio))
                    else:
                        crop_h = h
                    if sig.shape_w_ratio < 1:
                        crop_w = max(1, int(w * sig.shape_w_ratio))
                    else:
                        crop_w = w
                    return grid[:crop_h, :crop_w].copy()
                
                # Shift: translada
                if "shift" in str(sig.to_name()):
                    dr = int(round(sig.centroid_dr))
                    dc = int(round(sig.centroid_dc))
                    result = np.zeros_like(grid)
                    r_s = max(0, -dr)
                    r_d = max(0, dr)
                    c_s = max(0, -dc)
                    c_d = max(0, dc)
                    r_e = min(h, h - dr)
                    c_e = min(w, w - dc)
                    result[r_d:r_d+(r_e-r_s), c_d:c_d+(c_e-c_s)] = grid[r_s:r_e, c_s:c_e]
                    return result
                
            except Exception:
                pass
            return grid.copy()
        
        return _exec
    
    def get_signatures(self) -> Dict[str, dict]:
        """Retorna assinaturas dos tipos descobertos."""
        return {name: {
            "object_ratio": round(s.object_ratio, 2),
            "shape_h_ratio": round(s.shape_h_ratio, 2),
            "shape_w_ratio": round(s.shape_w_ratio, 2),
            "area_ratio": round(s.area_ratio, 2),
            "color_preserved": round(s.color_preserved, 2),
            "n_examples": s.n_examples,
        } for name, s in self._type_signatures.items()}
    
    def get_stats(self) -> dict:
        stats = self.registry.get_stats()
        stats["dynamic_types"] = len(self._type_signatures)
        stats["known_types"] = list(self._known_types)[:10]
        return stats
