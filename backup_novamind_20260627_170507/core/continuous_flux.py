"""
ContinuousFluxEngine — Percepção Contínua com Derivadas Temporais
==================================================================
Captura o framebuffer em tempo real e processa derivadas temporais:

  frame_0 → frame_1 → frame_2 → ...

  diff_t    = frame_t − frame_{t-1}          [velocidade]
  diff2_t   = diff_t − diff_{t-1}            [aceleração]
  pred_err  = MSE(predicted, frame_t)        [surpresa]

  ACh = σ(pred_err − limiar)                 [foco/noradrenalina]

Substitui o modelo de "tirar fotos" por processamento contínuo
de fluxo visual, como um córtex visual biológico.
"""

import time
import threading
from typing import Optional, Callable, List, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ContinuousFluxEngine:
    """
    Captura contínua de framebuffer com processamento de derivadas temporais.

    Args:
        region: (left, top, width, height) da região a capturar. None = full screen.
        target_fps: quadros por segundo alvo (default 15)
        buffer_seconds: quantos segundos de histórico manter (default 5)
        pred_window: quantos frames usar para extrapolação linear (default 3)
        ach_threshold: limiar de prediction error para ACh = 1.0 (default 0.3)
        device: device torch
    """

    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 15,
        buffer_seconds: int = 5,
        pred_window: int = 3,
        ach_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        self.region = region
        self.target_fps = target_fps
        self.buffer_size = target_fps * buffer_seconds
        self.pred_window = pred_window
        self.ach_threshold = ach_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ring buffer de frames como tensores (gray, uint8)
        self.frames: deque = deque(maxlen=self.buffer_size)
        self.timestamps: deque = deque(maxlen=self.buffer_size)

        # Derivadas
        self.velocity: Optional[np.ndarray] = None        # diff_t
        self.acceleration: Optional[np.ndarray] = None    # diff2_t
        self.prediction_error: float = 0.0                # MSE pred_err
        self.ach: float = 0.0                              # neuromodulador ACh

        # Estado do loop de captura
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_frame: Optional[Callable] = None

        # Último frame em RGB para consumo externo
        self.last_rgb: Optional[np.ndarray] = None

        # Resolução detectada
        self.width: int = 0
        self.height: int = 0

        if HAS_MSS:
            self._sct = mss.mss()
        else:
            self._sct = None

    # ─── Loop de Captura ─────────────────────────────────────────

    def start(self, on_frame: Optional[Callable] = None):
        """Inicia captura em thread separada."""
        if self._running:
            return
        self._running = True
        self._on_frame = on_frame
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _capture_loop(self):
        interval = 1.0 / self.target_fps
        while self._running:
            t0 = time.perf_counter()
            self._capture_frame()
            if self._on_frame:
                self._on_frame()
            elapsed = time.perf_counter() - t0
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _capture_frame(self):
        """Captura um frame do framebuffer e atualiza derivadas."""
        if self._sct:
            monitor = self._sct.monitors[1]  # monitor primário
            if self.region:
                left, top, w, h = self.region
                monitor = {"left": left, "top": top, "width": w, "height": h}
            sct_img = self._sct.grab(monitor)
            img = np.array(sct_img)
            rgb = img[:, :, :3]  # BGRA → RGB (na verdade mss dá BGRA)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            # Converte BGRA → RGB
            rgb = rgb[:, :, ::-1]
        elif HAS_CV2:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.region:
                x, y, w, h = self.region
                rgb = rgb[y:y+h, x:x+w]
        else:
            return

        self.last_rgb = rgb
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        now = time.time()
        self.timestamps.append(now)
        self.frames.append(gray)

        if len(self.frames) >= 3:
            self._compute_derivatives()

    def _compute_derivatives(self):
        """Computa derivadas temporais e prediction error."""
        f_t = self.frames[-1]
        f_t1 = self.frames[-2]
        f_t2 = self.frames[-3] if len(self.frames) >= 3 else f_t1

        # Primeira derivada: velocidade (diferença entre frames consecutivos)
        self.velocity = f_t - f_t1

        # Segunda derivada: aceleração (diferença da diferença)
        diff_t = self.velocity
        diff_t1 = f_t1 - f_t2
        self.acceleration = diff_t - diff_t1

        # Predição: extrapolação linear dos últimos pred_window frames
        if len(self.frames) >= self.pred_window + 1:
            pred = self._linear_extrapolate()
            mse = float(np.mean((pred - f_t) ** 2))
            self.prediction_error = mse
            # ACh = sigmoid(pred_err - threshold)
            self.ach = 1.0 / (1.0 + np.exp(-5.0 * (mse - self.ach_threshold)))
        else:
            self.prediction_error = 0.0
            self.ach = 0.0

    def _linear_extrapolate(self) -> np.ndarray:
        """
        Extrapolação linear: usa últimos N frames para prever o próximo.

        predicted = last + (last - last_N) / N  [tendência linear]
        """
        n = min(self.pred_window, len(self.frames) - 1)
        last = self.frames[-1]
        last_n = self.frames[-(n + 1)]
        trend = (last - last_n) / n
        predicted = last + trend
        return np.clip(predicted, 0.0, 1.0)

    # ─── API de Consulta ─────────────────────────────────────────

    def get_state(self) -> dict:
        """Retorna estado completo do fluxo contínuo."""
        return {
            "fps_actual": self._estimate_fps(),
            "width": self.width,
            "height": self.height,
            "buffer_frames": len(self.frames),
            "velocity_mean": float(np.mean(np.abs(self.velocity))) if self.velocity is not None else 0.0,
            "acceleration_mean": float(np.mean(np.abs(self.acceleration))) if self.acceleration is not None else 0.0,
            "prediction_error": self.prediction_error,
            "ach": self.ach,
            "ach_exceeded": self.ach > 0.7,
            "running": self._running,
        }

    def get_prediction_error_tensor(self) -> Optional[torch.Tensor]:
        """Retorna o prediction error como tensor 1D para log no EpisodicStream."""
        if self.prediction_error > 0:
            return torch.tensor([self.prediction_error, self.ach], dtype=torch.float32)
        return None

    def _estimate_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        recent = list(self.timestamps)[-min(30, len(self.timestamps)):]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / (recent[-1] - recent[0] + 1e-6)

    def get_motion_mask(self, threshold: float = 0.05) -> Optional[np.ndarray]:
        """Retorna máscara binária de movimento (onde |velocity| > threshold)."""
        if self.velocity is None:
            return None
        return (np.abs(self.velocity) > threshold).astype(np.uint8) * 255


class UniformNoiseFlux(ContinuousFluxEngine):
    """
    Versão simplificada para testes sem captura de tela real.
    Gera padrões de noise que simulam movimento.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._test_frame = np.random.randn(100, 100).astype(np.float32) * 0.1 + 0.5

    def _capture_frame(self):
        # Gera um padrão que "anda" diagonalmente
        shift = np.sin(time.time() * 0.5) * 10
        self._test_frame = np.roll(self._test_frame, 1, axis=0)
        self._test_frame = np.roll(self._test_frame, 1, axis=1)
        self._test_frame += np.random.randn(100, 100).astype(np.float32) * 0.01
        self._test_frame = np.clip(self._test_frame, 0, 1)

        now = time.time()
        self.timestamps.append(now)
        self.frames.append(self._test_frame.copy())
        self.last_rgb = (self._test_frame * 255).astype(np.uint8)
        self.width = 100
        self.height = 100

        if len(self.frames) >= 3:
            self._compute_derivatives()
