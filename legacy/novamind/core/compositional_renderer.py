"""
MentalRenderer — Renderizador de Pensamentos via Primitivas Composicionais

Pipeline:
  CGE → Simulation Engine → NOVO HDV → MentalRenderer → texto/imagem/video/audio/codigo

O renderer NAO reconstroi memoria. Ele RENDERIZA PENSAMENTO.
Usa primitivas (circulo, linha, curva, textura, sombra, movimento) 
compostas como a Forja compoe operadores.
Nao ha templates. Cada renderizacao e gerada do estado atual do CGE.
"""
import math, random, io, struct, zlib, hashlib, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ─── PRIMITIVAS VISUAIS ──────────────────────────────────

@dataclass
class Primitive:
    type: str = "circle"  # circle, line, rect, curve, dot, polygon
    x: float = 0.5
    y: float = 0.5
    w: float = 0.1
    h: float = 0.1
    r: float = 0.0  # rotation radians
    cr: float = 0.3  # color r 0..1
    cg: float = 0.5
    cb: float = 1.0
    alpha: float = 0.7
    points: List[Tuple[float, float]] = field(default_factory=list)  # for polygon
    label: str = ""


@dataclass
class Scene:
    primitives: List[Primitive] = field(default_factory=list)
    bg_r: float = 0.02
    bg_g: float = 0.03
    bg_b: float = 0.08
    W: int = 224
    H: int = 224

    def add(self, p: Primitive):
        self.primitives.append(p)

    def render_png(self) -> bytes:
        """Renderiza como PNG (zero dependencias)."""
        W, H = self.W, self.H
        px = bytearray(W * H * 3)
        # Fundo
        for i in range(W * H):
            px[i*3] = int(self.bg_r * 255)
            px[i*3+1] = int(self.bg_g * 255)
            px[i*3+2] = int(self.bg_b * 255)

        for p in self.primitives:
            cx, cy = int(p.x * W), int(p.y * H)
            rr, gg, bb = int(p.cr*255), int(p.cg*255), int(p.cb*255)
            a = p.alpha

            if p.type == "circle":
                radius = max(1, int(p.w * W / 2))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            px2, py2 = cx + dx, cy + dy
                            if 0 <= px2 < W and 0 <= py2 < H:
                                idx = (py2 * W + px2) * 3
                                px[idx] = int(px[idx]*(1-a) + rr*a)
                                px[idx+1] = int(px[idx+1]*(1-a) + gg*a)
                                px[idx+2] = int(px[idx+2]*(1-a) + bb*a)

            elif p.type == "dot":
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        px2, py2 = cx + dx, cy + dy
                        if 0 <= px2 < W and 0 <= py2 < H:
                            idx = (py2 * W + px2) * 3
                            px[idx] = rr; px[idx+1] = gg; px[idx+2] = bb
                # glow
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        d = dx*dx + dy*dy
                        if 4 <= d <= 9:
                            px2, py2 = cx + dx, cy + dy
                            if 0 <= px2 < W and 0 <= py2 < H:
                                idx = (py2 * W + px2) * 3
                                glow = 0.3
                                px[idx] = int(px[idx]*(1-glow) + rr*glow)
                                px[idx+1] = int(px[idx+1]*(1-glow) + gg*glow)
                                px[idx+2] = int(px[idx+2]*(1-glow) + bb*glow)

            elif p.type == "line":
                ex = int((p.x + p.w) * W)
                ey = int((p.y + p.h) * H)
                steps = max(abs(ex-cx), abs(ey-cy), 1)
                for i in range(steps + 1):
                    t = i / steps
                    px2 = int(cx + (ex-cx)*t); py2 = int(cy + (ey-cy)*t)
                    if 0 <= px2 < W and 0 <= py2 < H:
                        idx = (py2 * W + px2) * 3
                        px[idx] = int(px[idx]*(1-a) + rr*a)
                        px[idx+1] = int(px[idx+1]*(1-a) + gg*a)
                        px[idx+2] = int(px[idx+2]*(1-a) + bb*a)

            elif p.type == "rect":
                rw, rh = int(p.w*W/2), int(p.h*H/2)
                for py2 in range(max(0, cy-rh), min(H, cy+rh)):
                    for px2 in range(max(0, cx-rw), min(W, cx+rw)):
                        idx = (py2 * W + px2) * 3
                        px[idx] = int(px[idx]*(1-a) + rr*a)
                        px[idx+1] = int(px[idx+1]*(1-a) + gg*a)
                        px[idx+2] = int(px[idx+2]*(1-a) + bb*a)

            elif p.type == "curve":
                n = 20
                for i in range(n):
                    t = i / n
                    angle = p.w * math.pi * 2 * t + p.h
                    r2 = p.w * 0.4 * t
                    px2 = int(cx + r2 * math.cos(angle) * W)
                    py2 = int(cy + r2 * math.sin(angle) * H)
                    if 0 <= px2 < W and 0 <= py2 < H:
                        idx = (py2 * W + px2) * 3
                        px[idx] = int(px[idx]*(1-a) + rr*a)
                        px[idx+1] = int(px[idx+1]*(1-a) + gg*a)
                        px[idx+2] = int(px[idx+2]*(1-a) + bb*a)

            elif p.type == "polygon" and len(p.points) >= 3:
                for i in range(len(p.points)):
                    x1, y1 = p.points[i]
                    x2, y2 = p.points[(i+1) % len(p.points)]
                    px1, py1 = int((p.x+x1)*W), int((p.y+y1)*H)
                    px2, py2 = int((p.x+x2)*W), int((p.y+y2)*H)
                    steps = max(abs(px2-px1), abs(py2-py1), 1)
                    for j in range(steps + 1):
                        t = j / steps
                        ppx = int(px1 + (px2-px1)*t)
                        ppy = int(py1 + (py2-py1)*t)
                        if 0 <= ppx < W and 0 <= ppy < H:
                            idx = (ppy * W + ppx) * 3
                            px[idx] = int(px[idx]*(1-a) + rr*a)
                            px[idx+1] = int(px[idx+1]*(1-a) + gg*a)
                            px[idx+2] = int(px[idx+2]*(1-a) + bb*a)

        # PNG encode
        return self._png(px, W, H)

    def _png(self, px: bytearray, W: int, H: int) -> bytes:
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


# ─── GERADOR DE PRIMITIVAS A PARTIR DO CGE ──────────────

def _hash_seed(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

def _color_from_seed(seed: int, i: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed + i * 137)
    return (rng.random(), rng.random(), rng.random())

def _pos_from_seed(seed: int, i: int) -> Tuple[float, float]:
    rng = random.Random(seed + i * 73)
    return (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95))


class ThoughtRenderer:
    """Renderiza pensamentos do CGE como cenas visuais composicionais.

    Nao usa templates. Cada cena e gerada do estado atual do CGE.
    """

    def __init__(self, cge=None):
        self.cge = cge

    def render_concept(self, concept: str) -> Scene:
        """Renderiza um conceito baseado em suas relacoes no CGE."""
        scene = Scene()
        seed = _hash_seed(concept)
        rng = random.Random(seed)

        # Nó central: circulo para o conceito principal
        cr, cg, cb = _color_from_seed(seed, 0)
        scene.add(Primitive("circle", 0.5, 0.5, 0.15, 0.15, cr=cr, cg=cg, cb=cb,
                            label=concept[:8]))

        # Buscar relacoes no CGE
        relations = []
        if self.cge and concept in self.cge.nodes:
            for e in self.cge.edges[-50:]:
                if e["from"] == concept:
                    relations.append(e["to"])
                elif e["to"] == concept:
                    relations.append(e["from"])

        if relations:
            # Filhos conectados por linhas
            angle_step = 2 * math.pi / max(len(relations), 1)
            for i, rel in enumerate(relations[:10]):
                angle = angle_step * i + rng.uniform(-0.2, 0.2)
                dist = 0.25 + rng.uniform(0, 0.1)
                x = 0.5 + dist * math.cos(angle)
                y = 0.5 + dist * math.sin(angle)
                rc, gc, bc = _color_from_seed(seed, i + 1)

                # Linha de conexao
                scene.add(Primitive("line", 0.5, 0.5, x-0.5, y-0.5,
                                    cr=rc, cg=gc, cb=bc, alpha=0.3))

                # No filho
                scene.add(Primitive("dot", x, y, cr=rc, cg=gc, cb=bc))
        else:
            # Sem relacoes: pontos orbitando
            for i in range(3):
                angle = rng.uniform(0, 2*math.pi)
                dist = 0.2 + i * 0.1
                x = 0.5 + dist * math.cos(angle)
                y = 0.5 + dist * math.sin(angle)
                rc, gc, cb = _color_from_seed(seed, i + 5)
                scene.add(Primitive("curve", 0.5, 0.5, 0.1 + i*0.05, angle,
                                    cr=rc, cg=gc, cb=cb, alpha=0.4))

        # Textura de fundo: pontos de ruido
        for _ in range(30):
            scene.add(Primitive("dot", rng.random(), rng.random(),
                                cr=rng.random()*0.3, cg=rng.random()*0.3,
                                cb=rng.random()*0.5, alpha=rng.uniform(0.05, 0.15)))

        return scene

    def render_thoughts(self, concepts: List[str]) -> Scene:
        """Renderiza uma sequencia de pensamentos como cena unica."""
        scene = Scene()
        n = len(concepts)
        if n == 0:
            return scene

        # Layout: grid adaptativo baseado no numero de conceitos
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        for i, concept in enumerate(concepts[:16]):
            seed = _hash_seed(concept)
            cr, cg, cb = _color_from_seed(seed, 0)

            x = (0.5 + (i % cols)) / max(cols, 1)
            y = (0.5 + (i // cols)) / max(rows, 1)

            scene.add(Primitive("circle", x, y, 0.06, 0.06, cr=cr, cg=cg, cb=cb))

            # Conexoes entre conceitos consecutivos
            if i > 0:
                prev_seed = _hash_seed(concepts[i-1])
                pr, pg, pb = _color_from_seed(prev_seed, 0)
                px = (0.5 + ((i-1) % cols)) / max(cols, 1)
                py = (0.5 + ((i-1) // cols)) / max(rows, 1)
                scene.add(Primitive("line", px, py, x-px, y-py,
                                    cr=pr, cg=pg, cb=pb, alpha=0.3))

        return scene

    def render_from_hdv(self, hdv_vector: list, concept: str = "") -> Scene:
        """Renderiza a partir de um vetor HDV (10000 bipolar)."""
        scene = Scene()
        if not hdv_vector:
            return scene

        # Usar os bits do HDV para gerar a cena
        n_bits = min(len(hdv_vector), 200)
        step = max(1, len(hdv_vector) // n_bits)

        points = []
        for i in range(0, min(len(hdv_vector), 10000), step):
            val = hdv_vector[i]
            x = (i % 100) / 100.0
            y = (i // 100) / 100.0 if (i // 100) < 100 else random.random()
            if val > 0:
                scene.add(Primitive("dot", x, y, cr=0, cg=0.6, cb=1, alpha=0.5))
            else:
                scene.add(Primitive("dot", x, y, cr=0.6, cg=0.1, cb=0, alpha=0.3))

        if concept:
            cr, cg, cb = _color_from_seed(_hash_seed(concept), 0)
            scene.add(Primitive("circle", 0.5, 0.5, 0.1, 0.1, cr=cr, cg=cg, cb=cb))

        return scene
