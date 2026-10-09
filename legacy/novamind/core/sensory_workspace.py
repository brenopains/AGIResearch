"""
SensoryWorkspace — Percepcao continua de audio e video.

Mantem buffers curtos (~5s) de percepcao sensorial.
Alimenta o MentalState continuamente.
Só persiste episodios quando eventos significativos ocorrem.

Uso:
    ws = SensoryWorkspace()
    ws.start()  # inicia threads de captura
    # ... ciclo cognitivo roda ...
    ws.stop()
"""
import os, time, threading, queue, json
from typing import Optional, Dict, Any
import numpy as np

# ─── Config ──────────────────────────────────────────────────
SENSORY_DIR = r"D:\AGI2\sensory"
os.makedirs(SENSORY_DIR, exist_ok=True)

BUFFER_SECONDS = 5      # duracao do buffer sensorial
FPS = 10                 # frames por segundo (video)
SAMPLE_RATE = 16000      # taxa de amostragem (audio)
EPISODE_COOLDOWN = 2.0   # segundos minimos entre episodios

# ─── HDV Encoder ────────────────────────────────────────────
# Reusa o mesmo encoder do cognitive_cycle
_NLP = None
_PROJ = None

def _ensure_encoder():
    global _NLP, _PROJ
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_lg")
        np.random.seed(46)
        _PROJ = np.random.randn(10000, 300).astype(np.float32) / np.sqrt(300)

def _word_vec(text):
    _ensure_encoder()
    doc = _NLP(str(text)[:500])
    vecs = [t.vector for t in doc if t.has_vector and t.vector_norm > 0]
    if not vecs:
        return np.zeros(10000, dtype=np.float32)
    avg = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    hdv = np.sign(_PROJ @ avg)
    hdv[hdv == 0] = 1
    return hdv.astype(np.float32)


class SensoryBuffer:
    """Buffer circular de percepcao sensorial (ultimos N segundos)."""

    def __init__(self, max_frames=FPS * BUFFER_SECONDS):
        self.max_frames = max_frames
        self.frames: list = []  # lista de dicts com timestamp, modality, hdv, metadata
        self._lock = threading.Lock()

    def push(self, modality: str, hdv: np.ndarray, metadata: dict = None):
        with self._lock:
            self.frames.append({
                "timestamp": time.time(),
                "modality": modality,
                "hdv": hdv.copy() if hdv is not None else None,
                "metadata": metadata or {},
            })
            if len(self.frames) > self.max_frames:
                self.frames = self.frames[-self.max_frames:]

    def get_recent(self, seconds: float = 2.0) -> list:
        """Retorna frames dos ultimos N segundos."""
        cutoff = time.time() - seconds
        with self._lock:
            return [f for f in self.frames if f["timestamp"] >= cutoff]

    def get_last(self, modality: str = None):
        """Retorna o ultimo frame (opcionalmente de uma modalidade)."""
        with self._lock:
            for f in reversed(self.frames):
                if modality is None or f["modality"] == modality:
                    return f
        return None

    def detect_change(self, threshold: float = 0.3) -> float:
        """Detecta mudanca significativa no buffer."""
        with self._lock:
            if len(self.frames) < 2:
                return 0.0
            recent = self.frames[-min(10, len(self.frames)):]
            if len(recent) < 2:
                return 0.0
            # Compara HDVs consecutivos da mesma modalidade
            changes = []
            for i in range(1, len(recent)):
                if recent[i]["modality"] == recent[i-1]["modality"]:
                    a = recent[i]["hdv"]
                    b = recent[i-1]["hdv"]
                    if a is not None and b is not None:
                        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                        changes.append(1 - cos)
            return float(np.mean(changes)) if changes else 0.0

    def clear(self):
        with self._lock:
            self.frames = []


class AudioCapture:
    """Captura de audio do microfone em background."""

    def __init__(self, buffer: SensoryBuffer, device=None):
        self.buffer = buffer
        self.device = device
        self._running = False
        self._thread = None
        self._mel_tf = None
        self._proj_audio = None

    def start(self):
        try:
            import sounddevice as sd
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
        except ImportError:
            print("[SENSORY] sounddevice nao instalado. Audio desabilitado.")
            return False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _capture_loop(self):
        import sounddevice as sd
        import torch
        import torchaudio

        self._mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=512, hop_length=128, n_mels=64
        )
        np.random.seed(43)
        self._proj_audio = np.random.randn(10000, 64).astype(np.float32) / 8

        chunk_seconds = 1.0  # 1 segundo por chunk
        chunk_samples = int(SAMPLE_RATE * chunk_seconds)

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=chunk_samples, device=self.device) as stream:
                while self._running:
                    data, _ = stream.read(chunk_samples)
                    if data is None or len(data) == 0:
                        continue
                    # Converte para tensor
                    wav = torch.from_numpy(data.flatten()).float().unsqueeze(0)
                    # Mel spectrogram
                    mel = self._mel_tf(wav).mean(dim=-1).squeeze(0).numpy()
                    # HDV
                    hdv = np.sign(self._proj_audio @ mel)
                    hdv[hdv == 0] = 1
                    self.buffer.push("audio", hdv.astype(np.float32),
                                     {"rms": float(np.sqrt(np.mean(data**2)))})
        except Exception as e:
            print(f"[SENSORY] Audio error: {e}")


class VideoCapture:
    """Captura de video da camera em background."""

    def __init__(self, buffer: SensoryBuffer, camera_id=0):
        self.buffer = buffer
        self.camera_id = camera_id
        self._running = False
        self._thread = None
        self._model = None
        self._proj = None

    def start(self):
        try:
            import cv2
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
        except ImportError:
            print("[SENSORY] OpenCV nao instalado. Video desabilitado.")
            return False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _ensure_vision(self):
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
            resnet.eval()
            self._model = nn.Sequential(*list(resnet.children())[:-1])
            torch.manual_seed(42)
            self._proj = torch.randn(10000, 512, device=device) / np.sqrt(512)
            self._device = device
            self._transform = T.Compose([
                T.ToPILImage(), T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _capture_loop(self):
        import cv2
        import torch

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("[SENSORY] Camera nao encontrada. Video desabilitado.")
            return

        self._ensure_vision()
        frame_interval = 1.0 / FPS
        last_time = time.time()

        while self._running:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.01)
                continue
            last_time = now

            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            try:
                # Converte frame para HDV
                img = self._transform(frame).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feats = self._model(img).squeeze(-1).squeeze(-1)
                hdv = torch.sign(self._proj @ feats)
                hdv[hdv == 0] = 1
                self.buffer.push("video", hdv.cpu().numpy().astype(np.float32), {})
            except Exception as e:
                pass

        cap.release()


class SensoryWorkspace:
    """Workspace sensorial continuo: audio + video.

    Mantem buffers curtos, detecta mudancas, alimenta o MentalState.
    """

    def __init__(self, mental_state=None):
        self.buffer = SensoryBuffer()
        self.audio = AudioCapture(self.buffer)
        self.video = VideoCapture(self.buffer)
        self.ms = mental_state
        self._last_episode_time = 0
        self._running = False

    def start(self):
        """Inicia captura de audio e video em background."""
        print("[SENSORY] Iniciando workspace sensorial...")
        self._running = True
        audio_ok = self.audio.start()
        video_ok = self.video.start()
        if audio_ok or video_ok:
            print(f"[SENSORY] Audio={'OK' if audio_ok else 'OFF'}, Video={'OK' if video_ok else 'OFF'}")
            # Thread de monitoramento
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            return True
        print("[SENSORY] Nenhuma fonte sensorial disponivel")
        return False

    def stop(self):
        """Para captura."""
        self._running = False
        self.audio.stop()
        self.video.stop()
        print("[SENSORY] Workspace sensorial parado")

    def _monitor_loop(self):
        """Monitora o buffer sensorial e alimenta o MentalState."""
        while self._running and self.ms:
            change = self.buffer.detect_change()
            last_audio = self.buffer.get_last("audio")
            last_video = self.buffer.get_last("video")
            recent = self.buffer.get_recent(2.0)

            # Registra estado sensorial no MentalState
            sensory_state = {
                "audio_level": float(last_audio["metadata"].get("rms", 0)) if last_audio else 0,
                "change_detected": change,
                "buffer_frames": len(recent),
                "has_audio": last_audio is not None,
                "has_video": last_video is not None,
                "timestamp": time.time(),
            }
            self.ms.register("sensory", sensory_state)

            # Decisao: vira episodio?
            significant = change > 0.5 or (
                last_audio and last_audio["metadata"].get("rms", 0) > 0.1
            )
            if significant and time.time() - self._last_episode_time > EPISODE_COOLDOWN:
                self._last_episode_time = time.time()
                # Sinaliza que um evento sensorial ocorreu
                self.ms.prediction_error = min(1.0, self.ms.prediction_error + change * 0.5)
                if self.ms._modules.get("memory"):
                    try:
                        mem = self.ms._modules["memory"]
                        desc = f"sensory_event_change_{change:.2f}"
                        if last_audio and last_audio["metadata"].get("rms", 0) > 0.1:
                            desc = f"sensory_event_loud_audio"
                        ep = mem.store_episode({
                            "input": desc, "modality": "sensory",
                            "audio_level": sensory_state["audio_level"],
                            "change": change,
                        }, episode_type="sensory_event",
                        emotional_valence=change * 0.5,
                        surprise=change)
                        if last_audio and last_audio["hdv"] is not None:
                            ep.hdc_vector = last_audio["hdv"]
                    except Exception as e:
                        print(f"[SENSORY] Episodio error: {e}")

            time.sleep(0.5)

    def get_state(self) -> dict:
        """Estado sensorial atual."""
        last_audio = self.buffer.get_last("audio")
        last_video = self.buffer.get_last("video")
        return {
            "running": self._running,
            "buffer_size": len(self.buffer.frames),
            "change": self.buffer.detect_change(),
            "last_audio_rms": float(last_audio["metadata"].get("rms", 0)) if last_audio else 0,
            "has_video": last_video is not None,
        }


# ─── Teste ──────────────────────────────────────────────────
if __name__ == "__main__":
    from core.mental_state import get_mental_state
    from core.memory import MemorySystem

    ms = get_mental_state()
    mem = MemorySystem()
    ms._modules["memory"] = mem

    ws = SensoryWorkspace(ms)
    if ws.start():
        try:
            print("Workspace sensorial rodando. Ctrl+C para parar.")
            while True:
                time.sleep(2)
                s = ws.get_state()
                print(f"  Sensory: buffer={s['buffer_size']} change={s['change']:.3f} audio={s['last_audio_rms']:.4f}")
        except KeyboardInterrupt:
            ws.stop()
    else:
        print("Nao foi possivel iniciar. Instale dependencias:")
        print("  pip install sounddevice opencv-python")
