import numpy as np
from PIL import Image
from io import BytesIO
import traceback
from typing import Dict, Optional, List, Tuple


class VisualCortex:
    """
    Córtex Visual — processa PIXELS REAIS.

    Extrai features visuais diretamente de frames:
      - bordas (horizontais, verticais, total)
      - histograma de cores (dominantes por quadrante)
      - brilho, contraste
      - textura (fina vs grossa via desvio local)
      - movimento (diferença entre frames consecutivos)
      - complexidade (entropia da imagem)
      - detecção de mudanças de cena

    Gera auto-descrição SEM usar BLIP.
    BLIP é apenas muleta — label de supervisão para bootstrap.
    """

    def __init__(self, buffer_frames: int = 4):
        self._prev_frame: Optional[np.ndarray] = None
        self._buffer: List[np.ndarray] = []
        self._buffer_size = buffer_frames
        self._scene_change_threshold = 0.4
        self.last_features: Dict = {}
        self.last_self_description: str = ""
        self.total_frames_processed = 0

    def process(self, frame_bytes: bytes) -> Dict:
        """
        Processa um frame JPEG/PNG bruto e retorna features visuais.
        """
        try:
            img = Image.open(BytesIO(frame_bytes)).convert('RGB')
            arr = np.array(img, dtype=np.float32)
            h, w, _ = arr.shape
            gray = np.mean(arr, axis=2)
            self.total_frames_processed += 1

            features = {}

            # 1. Brilho e contraste
            features['brightness'] = float(np.mean(gray) / 255.0)
            features['contrast'] = float(np.std(gray) / 128.0)

            # 2. Bordas via gradientes (diferenças finitas)
            gx = np.abs(np.diff(gray, axis=1))
            gy = np.abs(np.diff(gray, axis=0))
            if gx.size > 0 and gy.size > 0:
                gx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
                gy = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
            edge_mag = np.sqrt(gx**2 + gy**2)
            edge_norm = edge_mag / (np.max(edge_mag) + 1e-8)
            features['edges'] = {
                'horizontal': float(np.mean(gy) / 255.0),
                'vertical': float(np.mean(gx) / 255.0),
                'total': float(np.mean(edge_norm)),
            }

            # 3. Cores dominantes (média por quadrante)
            q_h, q_w = h // 2, w // 2
            quadrants = {
                'top_left': arr[:q_h, :q_w],
                'top_right': arr[:q_h, q_w:],
                'bottom_left': arr[q_h:, :q_w],
                'bottom_right': arr[q_h:, q_w:],
            }
            colors = {}
            for qname, qarr in quadrants.items():
                mean_rgb = np.mean(qarr, axis=(0, 1))
                colors[qname] = {
                    'r': float(mean_rgb[0] / 255.0),
                    'g': float(mean_rgb[1] / 255.0),
                    'b': float(mean_rgb[2] / 255.0),
                }
            features['colors'] = colors

            # 4. Cor predominante geral
            mean_rgb = np.mean(arr, axis=(0, 1))
            dominant_channel = max(enumerate(['r', 'g', 'b']), key=lambda x: mean_rgb[x[0]])
            features['dominant_color'] = dominant_channel[1]

            # 5. Textura (desvio local médio)
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(gray, size=8)
            local_std = np.sqrt(uniform_filter((gray - local_mean)**2, size=8))
            mean_texture = float(np.mean(local_std) / 128.0)
            features['texture'] = {
                'fine': min(1.0, mean_texture * 2),
                'coarse': max(0.0, 1.0 - mean_texture * 2),
            }

            # 6. Complexidade (entropia normalizada)
            hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 255))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features['complexity'] = min(1.0, entropy / 6.0)

            # 7. Movimento (diferença com frame anterior)
            motion_detected = False
            motion_magnitude = 0.0
            if self._prev_frame is not None:
                if gray.shape == self._prev_frame.shape:
                    diff = np.abs(gray - self._prev_frame)
                    motion_magnitude = float(np.mean(diff) / 255.0)
                    motion_detected = motion_magnitude > 0.05
                    features['scene_change'] = motion_magnitude > self._scene_change_threshold
                else:
                    features['scene_change'] = False
            else:
                features['scene_change'] = False

            # Nova cena detectada
            is_new_scene = features.get('scene_change', False)

            features['motion'] = {
                'detected': motion_detected,
                'magnitude': motion_magnitude,
            }

            # 8. Simetria horizontal (centro vs laterais)
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            if left_half.shape == right_half.shape:
                symmetry = 1.0 - float(np.mean(np.abs(left_half - right_half)) / 255.0)
            else:
                symmetry = 0.5
            features['symmetry'] = symmetry

            self._prev_frame = gray.copy()

            # Buffer para análise temporal
            self._buffer.append(arr)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

            # Auto-descrição (sem BLIP!)
            desc = self._generate_description(features)
            self.last_features = features
            self.last_self_description = desc

            return {
                'features': features,
                'self_description': desc,
                'dimensions': (w, h),
                'is_new_scene': is_new_scene,
                'frame_index': self.total_frames_processed,
            }

        except Exception as e:
            return {
                'features': {
                    'error': str(e),
                    'brightness': 0.5,
                    'edges': {'total': 0.0, 'horizontal': 0.0, 'vertical': 0.0},
                },
                'self_description': f"[Visão: erro ao processar frame]",
                'dimensions': (0, 0),
                'is_new_scene': False,
                'frame_index': self.total_frames_processed,
            }

    def _generate_description(self, f: Dict) -> str:
        """Gera descrição textual do que o córtex visual detectou."""
        parts = []

        bright = f.get('brightness', 0.5)
        if bright < 0.25:
            parts.append("imagem escura")
        elif bright > 0.75:
            parts.append("imagem clara")
        else:
            parts.append("iluminação moderada")

        edges = f.get('edges', {})
        edge_total = edges.get('total', 0)
        if edge_total > 0.15:
            parts.append("muitas bordas nítidas")
            if edges.get('vertical', 0) > edges.get('horizontal', 0):
                parts.append("predomínio de linhas verticais")
            else:
                parts.append("predomínio de linhas horizontais")
        elif edge_total > 0.05:
            parts.append("bordas suaves")
        else:
            parts.append("poucas bordas")

        dc = f.get('dominant_color', '?')
        color_names = {'r': 'avermelhado', 'g': 'esverdeado', 'b': 'azulado'}
        parts.append(f"tom {color_names.get(dc, 'neutro')}")

        motion = f.get('motion', {})
        if motion.get('detected'):
            mag = motion.get('magnitude', 0)
            if mag > 0.3:
                parts.append("movimento intenso")
            else:
                parts.append("movimento sutil")

        tex = f.get('texture', {})
        if tex.get('fine', 0) > tex.get('coarse', 0):
            parts.append("textura fina/detalhada")
        else:
            parts.append("textura lisa")

        scene = "mudança de cena" if f.get('scene_change') else "mesma cena"

        return f"[Visão: {', '.join(parts)} | {scene}]"

    def get_state(self) -> Dict:
        return {
            'total_frames': self.total_frames_processed,
            'last_description': self.last_self_description,
            'features': self.last_features,
            'buffer_size': len(self._buffer),
        }
