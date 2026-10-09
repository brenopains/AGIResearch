"""
MentalRenderer — Interface de Renderizadores Plugáveis

O Maestro chama: render(estado_cognitivo, modalidade)
Nunca sabe quem está renderizando.

Renderizadores:
  VisualRenderer: HDV 10000D → imagem 224×224 RGB (VisualPipeline.decode_bipolar)
  AudioRenderer: HDV → waveform (AudioVocoder)
  TextRenderer: HDV → texto (BrocaDecoder)
  VideoRenderer: HDV → multiplos frames (VisualPipeline + temporal)
  CodeRenderer: HDV → codigo (templates)

O estado cognitivo é composto de milhares de HDVs + atenção + contexto.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import torch
import numpy as np
import io, struct, zlib, hashlib, math, random, time


# ─── INTERFACE BASE ──────────────────────────────────────

class BaseRenderer:
    """Todo renderizador herda desta classe."""
    def render(self, cognitive_state: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def learn(self, error: float):
        """Ajuste incremental (opcional)."""
        pass


# ─── VISUAL RENDERER (VisualPipeline wrapper) ────────────

class VisualRenderer(BaseRenderer):
    """HDV 10000D → imagem 224×224 RGB.

    Usa VisualPipeline.decode_bipolar() quando disponivel.
    Fallback: composicao geometrica do estado cognitivo.
    """

    def __init__(self, visual_pipeline=None):
        self.pipeline = visual_pipeline
        self._errors: List[float] = []

    def render(self, state: Dict) -> bytes:
        """Renderiza estado cognitivo como PNG."""
        hdvs = state.get("hdvs", [])
        concept = state.get("focus", "unknown")

        if self.pipeline and hdvs:
            # Usar VisualPipeline real se disponivel
            return self._render_via_pipeline(hdvs, concept)
        else:
            # Fallback: composicao geometrica
            return self._render_compositional(state)

    def learn(self, error: float):
        """Aprende com erro de reconstrucao (incremental)."""
        self._errors.append(error)
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]

    def _render_via_pipeline(self, hdvs: list, concept: str) -> bytes:
        """Usa VisualPipeline.decode_bipolar para renderizar HDV."""
        try:
            vec = torch.tensor(hdvs[0], dtype=torch.float32)
            img = self.pipeline.decode_bipolar(vec)  # (3,224,224)
            # Tensor → PNG bytes
            arr = (img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            H, W = arr.shape[:2]
            px = bytearray(W * H * 3)
            for y in range(H):
                for x in range(W):
                    idx = (y * W + x) * 3
                    px[idx] = arr[y,x,0]; px[idx+1] = arr[y,x,1]; px[idx+2] = arr[y,x,2]
            return _encode_png(px, W, H)
        except Exception as e:
            return self._render_compositional({"focus": concept, "hdvs": hdvs})

    def _render_compositional(self, state: Dict) -> bytes:
        """Fallback: composicao geometrica do estado."""
        focus = state.get("focus", "?")
        hdvs = state.get("hdvs", [])
        attention = state.get("attention", {})
        scene = Scene()

        if hdvs:
            # Renderizar HDV como pontos
            bits = hdvs[0]
            step = max(1, len(bits) // 200)
            for i in range(0, min(len(bits), 10000), step):
                x = (i % 100) / 100.0
                y = (i // 100) / 100.0
                if i < len(bits) and bits[i] > 0:
                    scene.add(Primitive("dot", x, y, cr=0, cg=0.6, cb=1, alpha=0.4))
                else:
                    scene.add(Primitive("dot", x, y, cr=0.5, cg=0.1, cb=0, alpha=0.2))

        # Foco: circulo central
        seed = _seed(focus)
        scene.add(Primitive("circle", 0.5, 0.5, 0.15, 0.15,
                            cr=_col(seed,0), cg=_col(seed,1), cb=_col(seed,2)))

        # Atencao: pontos ao redor com peso
        for key, weight in list(attention.items())[:8]:
            a = random.random() * 2 * math.pi
            d = 0.2 + weight * 0.2
            x = 0.5 + d * math.cos(a)
            y = 0.5 + d * math.sin(a)
            scene.add(Primitive("dot", x, y, cr=weight, cg=0.3, cb=1-weight, alpha=0.6))
            scene.add(Primitive("line", 0.5, 0.5, x-0.5, y-0.5,
                                cr=0.3, cg=0.5, cb=0.8, alpha=0.3))

        return scene.render_png()


# ─── AUDIO RENDERER ───────────────────────────────────────

class AudioRenderer(BaseRenderer):
    """HDV → waveform audio via AudioVocoder ou sintese direta."""

    def __init__(self, vocoder=None):
        self.vocoder = vocoder

    def render(self, state: Dict) -> bytes:
        hdvs = state.get("hdvs", [])
        concept = state.get("focus", "unknown")

        if self.vocoder and hdvs:
            try:
                vec = torch.tensor(hdvs[0], dtype=torch.float32)
                wav = self.vocoder.synthesize(vec, duration=0.3)
                return _wav_bytes(wav, 16000)
            except:
                pass

        # Fallback: tom baseado no conceito
        return _tone_wav(concept, duration=0.3, sr=16000)


# ─── VIDEO RENDERER ───────────────────────────────────────

class VideoRenderer(BaseRenderer):
    """Estado cognitivo temporal → video (sequencia de frames).

    Usa VisualPipeline.decode_bipolar() para renderizar cada frame.
    Se o estado tem scene_graph com to_temporal(), usa animacao procedural.
    """

    def __init__(self, visual_pipeline=None, n_frames: int = 16):
        self.pipeline = visual_pipeline
        self.n_frames = n_frames

    def render(self, state: Dict) -> bytes:
        """Renderiza estado cognitivo como GIF/animacao."""
        hdvs = state.get("hdvs", [])
        scene_graph = state.get("scene_graph", None)

        frames = []
        if scene_graph and hasattr(scene_graph, "to_temporal"):
            temporal = scene_graph.to_temporal(self.n_frames)
            for sg in temporal:
                frame_state = dict(state)
                frame_state["scene_graph"] = sg
                frame_state["hdvs"] = hdvs
                png = self._render_frame(frame_state)
                frames.append(png)
        elif hdvs:
            for i in range(min(self.n_frames, 8)):
                frame_state = dict(state)
                frame_state["_t"] = i / max(self.n_frames, 1)
                png = self._render_frame(frame_state)
                frames.append(png)
        else:
            for i in range(4):
                frames.append(self._render_frame(state))

        # Se tem varios frames, retorna o primeiro + info
        return frames[0] if frames else self._fallback_frame(state)

    def _render_frame(self, state: Dict) -> bytes:
        """Renderiza um frame do video."""
        hdvs = state.get("hdvs", [])
        if self.pipeline and hdvs:
            try:
                vec = torch.tensor(hdvs[0], dtype=torch.float32)
                img = self.pipeline.decode_bipolar(vec)
                arr = (img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                H, W = arr.shape[:2]
                px = bytearray(W * H * 3)
                for y in range(H):
                    for x in range(W):
                        idx = (y * W + x) * 3
                        px[idx] = arr[y,x,0]; px[idx+1] = arr[y,x,1]; px[idx+2] = arr[y,x,2]
                return _encode_png(px, W, H)
            except:
                pass
        return self._fallback_frame(state)

    def _fallback_frame(self, state: Dict) -> bytes:
        """Frame generico via composicao geometrica."""
        concept = state.get("focus", "?")
        scene = Scene()
        seed = _seed(concept)
        t = state.get("_t", 0)
        scene.add(Primitive("circle", 0.5, 0.5, 0.12, 0.12,
                            cr=_col(seed,0), cg=_col(seed,1), cb=_col(seed,2)))
        for i in range(6):
            a = t * 2 * math.pi + i * math.pi / 3
            d = 0.2 + 0.05 * math.sin(t * 4 + i)
            scene.add(Primitive("dot",
                0.5 + d * math.cos(a), 0.5 + d * math.sin(a),
                cr=_col(seed,i+3), cg=_col(seed,i+4), cb=_col(seed,i+5), alpha=0.6))
        return scene.render_png()


# ─── TEXT RENDERER ────────────────────────────────────────

class TextRenderer(BaseRenderer):
    """Estado cognitivo → texto descritivo."""

    def render(self, state: Dict) -> str:
        lines = [f"=== RENDER MENTAL ==="]
        lines.append(f"Foco: {state.get('focus', '?')}")
        lines.append(f"Modo: {state.get('mode', '?')}")
        hdvs = state.get("hdvs", [])
        lines.append(f"HDVs ativos: {len(hdvs)}")

        attention = state.get("attention", {})
        if attention:
            lines.append(f"Atencao: {', '.join(f'{k}:{v:.2f}' for k,v in list(attention.items())[:5])}")

        memories = state.get("memories", [])
        if memories:
            lines.append(f"Memorias: {len(memories)}")

        return '\n'.join(lines)


# ─── CODE RENDERER ────────────────────────────────────────

class CodeRenderer(BaseRenderer):
    """Estado cognitivo → pseudo-codigo do pensamento."""

    def render(self, state: Dict) -> str:
        lines = ["def pensamento_atual():"]
        lines.append(f"    # Foco: {state.get('focus', '?')}")
        lines.append(f"    # Modo: {state.get('mode', '?')}")
        hdvs = state.get("hdvs", [])
        for i, h in enumerate(hdvs[:3]):
            summary = sum(h[:100]) / 100 if len(h) >= 100 else 0
            lines.append(f"    hdv_{i} = HDV(ativacao_media={summary:.3f})")
        attention = state.get("attention", {})
        for k, v in list(attention.items())[:3]:
            lines.append(f"    atencao['{k}'] = {v:.2f}")
        lines.append("    return EstadoCognitivo(...)")
        return '\n'.join(lines)


# ─── PLUGGABLE RENDERER INTERFACE ─────────────────────────

class MentalRenderer:
    """Interface unificada de renderizacao.

    Uso:
        renderer = MentalRenderer(visual=VisualRenderer(pipeline))
        png = renderer.render(state, "image")
        wav = renderer.render(state, "audio")
        txt = renderer.render(state, "text")
    """

    def __init__(self, visual=None, audio=None, text=None, code=None, video=None):
        self.renderers = {
            "image": visual or VisualRenderer(),
            "audio": audio or AudioRenderer(),
            "text": text or TextRenderer(),
            "code": code or CodeRenderer(),
            "video": video or VideoRenderer(),
        }

    def render(self, cognitive_state: Dict, modality: str = "image") -> Any:
        """Renderiza estado cognitivo em uma modalidade."""
        r = self.renderers.get(modality)
        if r is None:
            return f"No renderer for {modality}"
        return r.render(cognitive_state)

    def learn(self, modality: str, error: float):
        """Aprendizado incremental do renderizador."""
        r = self.renderers.get(modality)
        if r:
            r.learn(error)

    def get_state(self) -> Dict:
        return {k: type(v).__name__ for k, v in self.renderers.items()}


# ─── PRIMITIVAS COMPOSICIONAIS (fallback visual) ─────────

@dataclass
class Primitive:
    type: str = "circle"
    x: float = 0.5; y: float = 0.5
    w: float = 0.1; h: float = 0.1
    cr: float = 0.3; cg: float = 0.5; cb: float = 1.0
    alpha: float = 0.7
    rotation: float = 0.0

@dataclass
class Scene:
    primitives: list = None
    bg_r: float = 0.02; bg_g: float = 0.03; bg_b: float = 0.08
    W: int = 224; H: int = 224
    def __post_init__(self):
        if self.primitives is None: self.primitives = []
    def add(self, p): self.primitives.append(p)
    def render_png(self) -> bytes:
        W, H, px = self.W, self.H, bytearray(self.W*self.H*3)
        for i in range(W*H):
            px[i*3]=int(self.bg_r*255); px[i*3+1]=int(self.bg_g*255); px[i*3+2]=int(self.bg_b*255)
        for p in self.primitives:
            cx, cy = int(p.x*W), int(p.y*H)
            rr, gg, bb = int(p.cr*255), int(p.cg*255), int(p.cb*255)
            if p.type == "circle":
                r = max(1, int(p.w*W/2))
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dx*dx+dy*dy <= r*r:
                            x2, y2 = cx+dx, cy+dy
                            if 0<=x2<W and 0<=y2<H:
                                idx=(y2*W+x2)*3; a=p.alpha
                                px[idx]=int(px[idx]*(1-a)+rr*a)
                                px[idx+1]=int(px[idx+1]*(1-a)+gg*a)
                                px[idx+2]=int(px[idx+2]*(1-a)+bb*a)
            elif p.type == "dot":
                for dy in range(-2,3):
                    for dx in range(-2,3):
                        x2, y2 = cx+dx, cy+dy
                        if 0<=x2<W and 0<=y2<H:
                            idx=(y2*W+x2)*3
                            px[idx]=rr; px[idx+1]=gg; px[idx+2]=bb
            elif p.type == "line":
                ex, ey = int((p.x+p.w)*W), int((p.y+p.h)*H)
                st = max(abs(ex-cx), abs(ey-cy), 1)
                for i in range(st+1):
                    t=i/st; x2=int(cx+(ex-cx)*t); y2=int(cy+(ey-cy)*t)
                    if 0<=x2<W and 0<=y2<H:
                        idx=(y2*W+x2)*3; a=p.alpha
                        px[idx]=int(px[idx]*(1-a)+rr*a)
                        px[idx+1]=int(px[idx+1]*(1-a)+gg*a)
                        px[idx+2]=int(px[idx+2]*(1-a)+bb*a)
        return _encode_png(px, W, H)


# ─── UTILITARIOS ──────────────────────────────────────────

def _seed(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

def _col(seed: int, off: int = 0) -> float:
    return random.Random(seed+off*137).random()

def _encode_png(px: bytearray, W: int, H: int) -> bytes:
    def chk(ct, d):
        c = ct + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c)&0xFFFFFFFF)
    raw = b''
    for y in range(H):
        raw += b'\x00' + bytes(px[y*W*3:(y+1)*W*3])
    return (b'\x89PNG\r\n\x1a\n' +
            chk(b'IHDR', struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) +
            chk(b'IDAT', zlib.compress(raw)) +
            chk(b'IEND', b''))

def _tone_wav(concept: str, duration: float = 0.3, sr: int = 16000) -> bytes:
    h = hash(concept) & 0xFFFF
    freq = 100 + (h % 2000)
    n = int(duration * sr)
    samples = bytearray(n * 2)
    for i in range(n):
        t = i / sr
        val = int(math.sin(2*math.pi*freq*t) * 8000 +
                  math.sin(2*math.pi*freq*1.5*t) * 3000)
        if val > 32767: val = 32767
        if val < -32768: val = -32768
        samples[i*2] = val & 0xFF
        samples[i*2+1] = (val >> 8) & 0xFF
    header = struct.pack('<4sI4s4sIHHIIHH', b'RIFF', 36+len(samples),
                         b'WAVE', b'fmt ', 16, 1, 1, sr, sr*2, 2, 16)
    return header + samples

def _wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    if waveform.dtype != np.int16:
        waveform = (waveform * 32767).astype(np.int16)
    samples = waveform.tobytes()
    header = struct.pack('<4sI4s4sIHHIIHH', b'RIFF', 36+len(samples),
                         b'WAVE', b'fmt ', 16, 1, 1, sr, sr*2, 2, 16)
    return header + samples
