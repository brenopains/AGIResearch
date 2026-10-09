"""
TransformationLearner — Aprende transformacoes de exemplos visuais.

Nao contem operadores ARC (rotate, flip, etc.).
Apenas: dado input→output, aprender o delta e aplicar.

Mecanismo geral:
  1. Compara HDVs de regioes entre input e output
  2. Detecta mudancas consistentes entre exemplos
  3. Abstrai regra: "regiao X muda de A para B"
  4. Aplica a regra a um novo input

Funciona para qualquer dominio (ARC, diagramas, mapas, UI, etc.)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

DIM = 10000


class RegionChange:
    """Representa uma mudanca em uma regiao especifica."""
    def __init__(self, region: str, from_hdv: np.ndarray, to_hdv: np.ndarray):
        self.region = region
        self.from_hdv = from_hdv.copy()
        self.to_hdv = to_hdv.copy()
        self.delta = np.sign(to_hdv - from_hdv)  # o que mudou

    def cosine(self) -> float:
        return float(np.dot(self.from_hdv, self.to_hdv) / DIM)

    def applies_to(self, region_hdv: np.ndarray) -> bool:
        """Verifica se esta mudanca se aplica a um HDV de regiao."""
        sim = float(np.dot(self.from_hdv, region_hdv) / DIM)
        return sim > 0.6  # threshold: HDV similar o suficiente

    def apply(self, region_hdv: np.ndarray) -> np.ndarray:
        """Aplica a transformacao a um HDV de regiao."""
        # Se o HDV de entrada e similar ao from_hdv, transforma em to_hdv
        result = np.sign(region_hdv + self.delta * 0.5)
        result[result == 0] = 1
        return result.astype(np.float32)


class PositionMapping:
    """Representa um mapeamento de posicao (ex: top_left -> top_right)."""
    def __init__(self, from_region: str, to_region: str):
        self.from_region = from_region
        self.to_region = to_region

    def apply(self, regions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aplica o mapeamento a um dicionario de regioes."""
        result = {}
        mapped = set()
        for src, dst in [(self.from_region, self.to_region)]:
            if src in regions:
                result[dst] = regions[src].copy()
                mapped.add(src)
        for r, hdv in regions.items():
            if r not in mapped:
                result[r] = hdv.copy()
        return result


class Transformation:
    """Uma transformacao aprendida de exemplos."""
    
    def __init__(self):
        self.region_changes: List[RegionChange] = []
        self.position_mappings: List[PositionMapping] = []
        self.confidence: float = 0.0
        self.n_examples: int = 0
    
    def apply(self, input_regions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aplica a transformacao a um novo input."""
        # 1. Aplica mapeamentos de posicao primeiro
        result = dict(input_regions)
        for mapping in self.position_mappings:
            result = mapping.apply(result)
        
        # 2. Aplica mudancas de regiao
        for change in self.region_changes:
            if change.region in result:
                if change.applies_to(result[change.region]):
                    result[change.region] = change.apply(result[change.region])
        
        return result
    
    def similarity(self, predicted: Dict[str, np.ndarray], expected: Dict[str, np.ndarray]) -> float:
        """Similaridade entre predicao e esperado."""
        scores = []
        for region in predicted:
            if region in expected:
                cos = float(np.dot(predicted[region], expected[region]) / DIM)
                scores.append(cos)
        return float(np.mean(scores)) if scores else 0.0


class TransformationLearner:
    """Aprende transformacoes de exemplos (input->output) sem operadores pre-definidos."""
    
    def __init__(self):
        self.learned_transforms: Dict[str, Transformation] = {}
    
    def learn(self, examples: List[Tuple[Dict, Dict]], task_id: str = "") -> Transformation:
        """Aprende uma transformacao a partir de exemplos input->output.
        
        Args:
            examples: lista de (input_regions, output_regions)
            task_id: identificador opcional
        
        Returns:
            Transformation aprendida
        """
        if not examples:
            return Transformation()
        
        transform = Transformation()
        transform.n_examples = len(examples)
        
        # 1. Detecta mudancas consistentes de regiao
        # Para cada regiao, ve se o padrao de mudanca se repete nos exemplos
        region_deltas = defaultdict(list)
        all_regions = set()
        
        for inp, out in examples:
            for region in inp:
                all_regions.add(region)
                if region in out:
                    cos = float(np.dot(inp[region], out[region]) / DIM)
                    if cos < 0.8:  # mudanca significativa
                        region_deltas[region].append({
                            "from": inp[region].copy(),
                            "to": out[region].copy(),
                            "cos": cos,
                        })
        
        # Se a mesma regiao mudou de forma consistente em todos os exemplos
        for region, deltas in region_deltas.items():
            if len(deltas) == transform.n_examples:
                # Verifica consistencia: todas as mudancas na mesma regiao sao similares?
                from_hdvs = [d["from"] for d in deltas]
                to_hdvs = [d["to"] for d in deltas]
                
                # Media dos "from" e "to"
                avg_from = np.mean(from_hdvs, axis=0)
                avg_to = np.mean(to_hdvs, axis=0)
                avg_from = np.sign(avg_from)
                avg_from[avg_from == 0] = 1
                avg_to = np.sign(avg_to)
                avg_to[avg_to == 0] = 1
                
                change = RegionChange(region, avg_from, avg_to)
                transform.region_changes.append(change)
        
        # 2. Detecta mapeamentos de posicao
        # Se o HDV de uma regiao no input aparece em OUTRA regiao no output
        if len(examples) >= 1:
            inp, out = examples[0]
            for src_region, src_hdv in inp.items():
                best_dst = None
                best_sim = 0
                for dst_region, dst_hdv in out.items():
                    if dst_region != src_region:
                        cos = float(np.dot(src_hdv, dst_hdv) / DIM)
                        if cos > best_sim:
                            best_sim = cos
                            best_dst = dst_region
                if best_sim > 0.7 and best_dst:
                    transform.position_mappings.append(
                        PositionMapping(src_region, best_dst)
                    )
        
        # 3. Confianca: quantas mudancas foram detectadas / numero de regioes
        n_changes = len(transform.region_changes) + len(transform.position_mappings)
        n_regions = len(all_regions)
        transform.confidence = min(1.0, n_changes / max(1, n_regions) * transform.n_examples)
        
        if task_id:
            self.learned_transforms[task_id] = transform
        
        return transform
    
    def predict(self, transform: Transformation, input_regions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Aplica transformacao a um novo input e retorna output predito."""
        return transform.apply(input_regions)
    
    def evaluate(self, transform: Transformation, test_input: Dict, test_expected: Dict) -> Dict:
        """Avalia uma transformacao contra um par test esperado."""
        predicted = transform.apply(test_input)
        sim = transform.similarity(predicted, test_expected)
        
        # Decodifica similaridade em score
        # 0.0-0.3: falha, 0.3-0.6: parcial, 0.6-0.8: bom, 0.8-1.0: excelente
        if sim > 0.7:
            verdict = "PASS"
        elif sim > 0.4:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
        
        return {
            "similarity": round(sim, 3),
            "verdict": verdict,
            "predicted": predicted,
            "n_changes": len(transform.region_changes),
            "n_mappings": len(transform.position_mappings),
            "confidence": round(transform.confidence, 3),
        }
    
    def predict_unknown(self, task_id: str, test_input: Dict) -> Optional[Dict]:
        """Prediz output para uma task conhecida."""
        if task_id in self.learned_transforms:
            return self.learned_transforms[task_id].apply(test_input)
        return None
