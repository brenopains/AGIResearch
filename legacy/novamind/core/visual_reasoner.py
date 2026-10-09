"""
VisualMemory + VisualReasoner — Memoria e raciocinio visuo-espacial.

Pipeline:
  1. DINOv2 processa imagem → patches → 9 regioes (3x3)
  2. Cada regiao vira um HDV (bind com vetor de posicao)
  3. VisualMemory armazena HDVs estruturados com metadata
  4. VisualReasoner compara HDV atual com memorias passadas
  5. Detecta transformacoes: rotacao, reflexao, translacao, troca de cor
  6. Alimenta o CGE com conceitos espaciais descobertos
"""
import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

DIM = 10000

# ─── Nomes das 9 regioes ────────────────────────────────────
REGIONS_3X3 = [
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]


class VisualMemory:
    """Armazena HDVs visuais estruturados por regiao.

    Cada entrada: { "id": str, "timestamp": float, "regions": {regiao: hdv_1d}, "metadata": dict }
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.entries: list = []  # mais recente primeiro

    def store(self, regions: Dict[str, np.ndarray], metadata: dict = None) -> str:
        """Armazena uma experiencia visual."""
        entry_id = f"vis_{int(time.time()*1000)}_{len(self.entries)}"
        entry = {
            "id": entry_id,
            "timestamp": time.time(),
            "regions": {k: v.copy() for k, v in regions.items()},
            "metadata": metadata or {},
        }
        self.entries.insert(0, entry)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[:self.capacity]
        return entry_id

    def get_recent(self, n: int = 5) -> list:
        """Retorna as N experiencias visuais mais recentes."""
        return self.entries[:n]

    def get_by_metadata(self, key: str, value) -> list:
        """Busca entradas por metadata."""
        return [e for e in self.entries if e["metadata"].get(key) == value]

    def clear(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)


class VisualReasoner:
    """Compara HDVs visuais, detecta transformacoes espaciais.

    Opera exclusivamente no espaco HDV: compara regioes,
    detecta mudancas de cor/posicao, infere transformacoes.
    """

    def __init__(self, memory: VisualMemory = None):
        self.memory = memory or VisualMemory()
        self._transform_counts = defaultdict(int)  # transformacao -> frequencia

    # ─── Comparacao entre dois HDVs visuais ─────────────────

    def compare_regions(self, regions_a: Dict, regions_b: Dict) -> Dict:
        """Compara duas imagens regiao por regiao.

        Returns:
            dict com similaridade geral + por regiao + regioes que mudaram
        """
        similarities = {}
        deltas = {}
        changed_regions = []

        for region in REGIONS_3X3:
            if region in regions_a and region in regions_b:
                va = regions_a[region]
                vb = regions_b[region]
                cos = float(np.dot(va, vb) / DIM)
                similarities[region] = cos
                deltas[region] = 1 - cos  # 0 = igual, 1 = totalmente diferente
                if cos < 0.5:
                    changed_regions.append(region)

        overall = float(np.mean(list(similarities.values()))) if similarities else 0.0

        return {
            "overall_similarity": overall,
            "regions": similarities,
            "deltas": deltas,
            "changed_regions": changed_regions,
            "n_changed": len(changed_regions),
        }

    # ─── Deteccao de transformacoes ─────────────────────────

    def detect_transformation(self, input_regions: Dict, output_regions: Dict) -> List[str]:
        """Detecta qual transformacao foi aplicada entre input e output.

        Analisa mudancas nas 9 regioes para identificar:
          - color_swap: cores trocadas entre regioes
          - rotation_90: rotacao 90 graus
          - reflection: reflexao horizontal/vertical
          - translation: deslocamento
          - identity: mesma imagem
          - unknown: padrao nao identificado
        """
        transformations = []

        # 1. Identidade (mesma imagem)
        comp = self.compare_regions(input_regions, output_regions)
        if comp["overall_similarity"] > 0.9:
            transformations.append("identity")
            return transformations

        # 2. Troca de cor (mesma regiao, conteudo diferente)
        # Verifica se as mesmas regioes mudaram de cor
        color_swap = False
        for region in REGIONS_3X3:
            if region in input_regions and region in output_regions:
                cos = float(np.dot(input_regions[region], output_regions[region]) / DIM)
                if cos < 0.3:  # cor completamente diferente na mesma posicao
                    color_swap = True
                    break
        if color_swap:
            transformations.append("color_change")

        # 3. Rotacao 90: top_left → top_right, top_right → bottom_right, etc.
        rot_map_90 = {
            "top_left": "top_right", "top_center": "middle_right",
            "top_right": "bottom_right", "middle_right": "bottom_center",
            "bottom_right": "bottom_left", "bottom_center": "middle_left",
            "bottom_left": "top_left", "middle_left": "top_center",
            "middle_center": "middle_center",
        }
        rot_score = self._check_remap(input_regions, output_regions, rot_map_90)
        if rot_score > 0.6:
            transformations.append("rotation_90")

        # 4. Reflexao horizontal: top_left ↔ top_right, etc.
        reflect_h = {
            "top_left": "top_right", "top_center": "top_center",
            "top_right": "top_left", "middle_left": "middle_right",
            "middle_center": "middle_center", "middle_right": "middle_left",
            "bottom_left": "bottom_right", "bottom_center": "bottom_center",
            "bottom_right": "bottom_left",
        }
        ref_h_score = self._check_remap(input_regions, output_regions, reflect_h)
        if ref_h_score > 0.6:
            transformations.append("reflection_horizontal")

        # 5. Reflexao vertical: top_left ↔ bottom_left, etc.
        reflect_v = {
            "top_left": "bottom_left", "top_center": "bottom_center",
            "top_right": "bottom_right", "middle_left": "middle_left",
            "middle_center": "middle_center", "middle_right": "middle_right",
            "bottom_left": "top_left", "bottom_center": "top_center",
            "bottom_right": "top_right",
        }
        ref_v_score = self._check_remap(input_regions, output_regions, reflect_v)
        if ref_v_score > 0.6:
            transformations.append("reflection_vertical")

        if not transformations:
            transformations.append("unknown")

        return transformations

    def _check_remap(self, reg_a: Dict, reg_b: Dict, mapping: Dict) -> float:
        """Verifica se reg_a → reg_b segue um mapeamento de posicoes."""
        scores = []
        for src, dst in mapping.items():
            if src in reg_a and dst in reg_b:
                cos = float(np.dot(reg_a[src], reg_b[dst]) / DIM)
                scores.append(cos)
        return float(np.mean(scores)) if scores else 0.0

    def learn_transformation(self, input_regions: Dict, output_regions: Dict, label: str = None):
        """Aprende uma transformacao a partir de exemplos.

        Armazena na memoria e incrementa contador.
        Se a mesma transformacao aparecer N vezes, poderia criar
        um conceito no CGE via Forja.
        """
        if label is None:
            transforms = self.detect_transformation(input_regions, output_regions)
            label = transforms[0] if transforms else "unknown"

        self._transform_counts[label] += 1

        return {
            "label": label,
            "count": self._transform_counts[label],
            "total_learned": sum(self._transform_counts.values()),
        }

    # ─── Predicao de output a partir de input ──────────────

    def predict_output(self, input_regions: Dict, example_input: Dict, example_output: Dict) -> Optional[Dict]:
        """Prediz o output de um input visual baseado em um exemplo.

        Aplica a mesma transformacao detectada no exemplo.
        """
        # Detecta transformacao no exemplo
        transforms = self.detect_transformation(example_input, example_output)
        if not transforms or transforms[0] == "unknown":
            return None

        transform = transforms[0]

        # Aplica a transformacao inversa ao input
        if transform == "identity":
            return input_regions

        elif transform == "rotation_90":
            # Mapeia regioes do input para onde iriam no output
            rot_map = {
                "top_left": "top_right", "top_center": "middle_right",
                "top_right": "bottom_right", "middle_right": "bottom_center",
                "bottom_right": "bottom_left", "bottom_center": "middle_left",
                "bottom_left": "top_left", "middle_left": "top_center",
                "middle_center": "middle_center",
            }
            predicted = {}
            for src, dst in rot_map.items():
                if src in input_regions and dst in rot_map:
                    predicted[dst] = input_regions[src].copy()
            return predicted

        elif transform == "reflection_horizontal":
            ref_map = {
                "top_left": "top_right", "top_center": "top_center",
                "top_right": "top_left", "middle_left": "middle_right",
                "middle_center": "middle_center", "middle_right": "middle_left",
                "bottom_left": "bottom_right", "bottom_center": "bottom_center",
                "bottom_right": "bottom_left",
            }
            predicted = {}
            for src, dst in ref_map.items():
                if src in input_regions:
                    predicted[dst] = input_regions[src].copy()
            return predicted

        elif transform == "reflection_vertical":
            ref_map = {
                "top_left": "bottom_left", "top_center": "bottom_center",
                "top_right": "bottom_right", "middle_left": "middle_left",
                "middle_center": "middle_center", "middle_right": "middle_right",
                "bottom_left": "top_left", "bottom_center": "top_center",
                "bottom_right": "top_right",
            }
            predicted = {}
            for src, dst in ref_map.items():
                if src in input_regions:
                    predicted[dst] = input_regions[src].copy()
            return predicted

        elif transform == "color_change":
            # Tenta identificar a regra de cor: as mesmas regioes do output
            # mantem as cores do input mas com um mapeamento diferente
            return None  # precisa de mais exemplos

        return None

    def compare_with_stored(self, regions: Dict, memory: VisualMemory, top_k: int = 3) -> List[Dict]:
        """Compara HDV visual atual com experiencias passadas na memoria."""
        scored = []
        for entry in memory.entries:
            comp = self.compare_regions(regions, entry["regions"])
            scored.append({
                "similarity": comp["overall_similarity"],
                "entry": entry,
                "comparison": comp,
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
