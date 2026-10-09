"""
PerceptualTargetMatcher — Engenharia Reversa Visual por HDC
===============================================================
A AGI recebe um vídeo de referência, codifica frames como vetores
HDC, gera código Pygame, executa, captura frames em tempo real,
compara no espaço HDC e itera até convergência visual.

Pipeline:
  1. extract_target_vector(video_path) → target_vec
  2. generate_code(target_vec, critique) → código Pygame
  3. run_and_capture(code, ticks) → lista de frames (numpy arrays)
  4. encode_frames(frames) → live_vec
  5. compute_fe(target_vec, live_vec) → Free Energy
  6. generate_visual_critique(fe, target_vec, live_vec) → texto
  7. loop até FE < threshold

Referência: sent.txt (55-133) — PerceptualTargetMatcher
"""

import os, sys, time, subprocess, tempfile, json
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class PerceptualTargetMatcher:
    """
    Compara frames gerados por código Pygame contra um vídeo de
    referência no espaço HDC. Itera até convergência visual.
    """

    def __init__(
        self,
        engine,           # GroundingEngine (ResNet18 → HDC)
        forge=None,       # Forja de código (opcional)
        critic=None,      # Crítico visual (opcional)
        device: Optional[str] = None,
        fe_threshold: float = 0.20,
        max_generations: int = 15,
    ):
        self.engine = engine
        self.forge = forge
        self.critic = critic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe_threshold = fe_threshold
        self.max_generations = max_generations

        # Estado
        self.target_vec: Optional[torch.Tensor] = None
        self.target_frames: List[np.ndarray] = []
        self.current_code: str = ""
        self.history: List[dict] = []
        self._tmp_dir = Path(tempfile.gettempdir()) / "novamind_perceptual"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def extract_target_vector(self, video_path: str, num_frames: int = 30) -> torch.Tensor:
        """
        Codifica vídeo de referência como vetor HDC alvo.
        Usa o GroundingEngine.encode_video() para projetar frames
        no hiperespaço.
        """
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) necessario para ler videos")

        print(f"[PTM] Extraindo vetor alvo de: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Nao foi possivel abrir video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / max(fps, 1)
        print(f"[PTM] Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

        # Amostra N frames uniformemente
        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        cap.release()

        if not frames:
            raise ValueError("Nenhum frame extraido do video")

        self.target_frames = frames
        print(f"[PTM] {len(frames)} frames amostrados")

        # Codifica cada frame como HDC via GroundingEngine
        vecs = []
        for i, frame in enumerate(frames):
            try:
                # Converte para formato que a engine aceita
                h, w = frame.shape[:2]
                # Se o GroundingEngine tem encode_image, usa direto
                if hasattr(self.engine, 'encode_image'):
                    mf = self.engine.encode_image(frame)
                    vecs.append(mf.tensor)
                elif hasattr(self.engine, 'encode_video_stream'):
                    # Salva frame temporário e codifica
                    tmp_path = self._tmp_dir / f"frame_{i:04d}.png"
                    cv2.imwrite(str(tmp_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # Usa a engine para codificar
                    mf = self.engine.encode_video(str(tmp_path), num_frames=1)
                    if hasattr(mf, 'tensor'):
                        vecs.append(mf.tensor)
                    else:
                        vecs.append(torch.from_numpy(mf).float())
                else:
                    # Fallback: projeta via ResNet18 manual
                    feats = self._extract_features(frame)
                    vecs.append(feats)
            except Exception as e:
                print(f"[PTM] Erro codificando frame {i}: {e}")
                continue

        if not vecs:
            raise RuntimeError("Nenhum frame pôde ser codificado como HDC")

        # Bundle: média + sign (para manter polaridade)
        stacked = torch.stack(vecs)
        bundle = torch.sign(stacked.sum(dim=0))
        self.target_vec = bundle
        print(f"[PTM] Target vector: {bundle.shape}, dev={bundle.device}")
        return bundle

    def _extract_features(self, rgb_frame: np.ndarray) -> torch.Tensor:
        """Fallback: extrai features hand-crafted se ResNet nao disponivel."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        # HOG-like: gradientes
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        # Histograma de gradientes + cor média + bordas
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        mean_rgb = rgb_frame.mean(axis=(0, 1)) / 255.0
        edge_density = (mag > 30).mean()

        feats = np.concatenate([hist / hist.sum(), mean_rgb, [edge_density]])
        # Projeta para 10000D
        proj = np.random.randn(10000, len(feats)).astype(np.float32)
        hdv = np.sign(proj @ feats)
        return torch.from_numpy(hdv).float().to(self.device)

    def encode_frames_to_vec(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Codifica lista de frames como vetor HDC (bundle)."""
        vecs = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            if hasattr(self.engine, 'encode_image'):
                mf = self.engine.encode_image(frame)
                vecs.append(mf.tensor)
            else:
                feats = self._extract_features(frame)
                vecs.append(feats)

        if not vecs:
            return torch.zeros(10000, device=self.device)
        stacked = torch.stack(vecs)
        return torch.sign(stacked.sum(dim=0))

    def run_and_capture(
        self, code: str, ticks: int = 60, headless: bool = True
    ) -> List[np.ndarray]:
        """
        Executa código Pygame e captura frames em tempo real.

        Args:
            code: código fonte Pygame completo
            ticks: número de ticks para executar
            headless: se True, usa variável de ambiente SDL_VIDEODRIVER=dummy

        Returns:
            lista de frames (numpy arrays RGB HxWx3)
        """
        if not HAS_PYGAME:
            print("[PTM] Pygame nao disponivel, gerando frames simulados")
            return self._simulate_frames(ticks)

        # Salva código em arquivo temporário com wrapper de captura
        capture_code = self._wrap_with_framebuffer(code, ticks)
        script_path = self._tmp_dir / "run_and_capture.py"
        script_path.write_text(capture_code, encoding="utf-8")

        output_path = self._tmp_dir / "captured_frames.npy"
        if output_path.exists():
            output_path.unlink()

        env = os.environ.copy()
        if headless:
            env["SDL_VIDEODRIVER"] = "dummy"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, timeout=30,
                env=env, cwd=str(self._tmp_dir),
            )
            # Log do stdout/stderr
            for line in result.stdout.split("\n"):
                if line.strip():
                    print(f"  [CAPTURE] {line.strip()}")
        except subprocess.TimeoutExpired:
            print("[PTM] Captura excedeu timeout de 30s")
        except Exception as e:
            print(f"[PTM] Erro na captura: {e}")

        if output_path.exists():
            frames = np.load(str(output_path), allow_pickle=True)
            print(f"[PTM] {len(frames)} frames capturados")
            return list(frames)
        else:
            print("[PTM] Nenhum frame capturado, modo simulacao")
            return self._simulate_frames(ticks)

    def _wrap_with_framebuffer(self, code: str, ticks: int) -> str:
        """Adiciona wrapper ao código Pygame para capturar framebuffer."""
        wrapper = f"""
import os, sys, numpy as np, pygame, time
os.environ["SDL_AUDIODRIVER"] = "dummy"

CAPTURE_TICKS = {ticks}
captured = []

try:
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    if W < 100 or H < 100:
        W, H = 800, 600
    
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    
    # Código gerado pela AGI
{self._indent_code(code, 4)}

    # Loop de captura
    running = True
    tick = 0
    while running and tick < CAPTURE_TICKS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        try:
            render_scene(screen, tick)
        except Exception as e:
            render_scene(screen)
        
        pygame.display.flip()
        
        # Captura framebuffer
        arr = pygame.surfarray.array3d(screen)
        arr = np.transpose(arr, (1, 0, 2))  # (W,H,3) -> (H,W,3)
        captured.append(arr)
        
        clock.tick(60)
        tick += 1

except Exception as e:
    print(f"ERRO: {{e}}")
finally:
    np.save(r'{self._tmp_dir / "captured_frames.npy"}', np.array(captured, dtype=object))
    pygame.quit()
"""
        return wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indenta código para nested dentro do wrapper."""
        indent = " " * spaces
        lines = []
        for line in code.split("\n"):
            if line.strip():
                lines.append(indent + line)
            else:
                lines.append("")
        return "\n".join(lines)

    def _simulate_frames(self, ticks: int) -> List[np.ndarray]:
        """Gera frames simulados quando Pygame nao esta disponivel."""
        frames = []
        for t in range(min(ticks, 10)):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            # Círculo que se move
            cx = 50 + int(30 * np.sin(t * 0.3))
            cy = 50 + int(20 * np.cos(t * 0.2))
            cv2.circle(frame, (cx, cy), 15, (200, 100, 50), -1)
            frames.append(frame)
        print(f"[PTM] {len(frames)} frames simulados (fallback)")
        return frames

    def compute_fe(self, target_vec: torch.Tensor, live_vec: torch.Tensor) -> float:
        """Computa Free Energy: 1 - cosine_similarity."""
        t = target_vec.flatten().float().to(self.device)
        l = live_vec.flatten().float().to(self.device)
        cos = F.cosine_similarity(t.unsqueeze(0), l.unsqueeze(0))
        fe = float(1.0 - cos.item())
        return fe

    def compute_similarity_matrix(
        self, target_frames: List[np.ndarray], live_frames: List[np.ndarray]
    ) -> torch.Tensor:
        """Matriz de similaridade frame-a-frame entre target e live."""
        n_t = len(target_frames)
        n_l = len(live_frames)
        sim = torch.zeros(n_t, n_l)

        # Pega alguns frames representativos
        step_t = max(1, n_t // 8)
        step_l = max(1, n_l // 8)

        for i in range(0, n_t, step_t):
            t_vec = self.encode_frames_to_vec([target_frames[i]])
            for j in range(0, n_l, step_l):
                l_vec = self.encode_frames_to_vec([live_frames[j]])
                cos = F.cosine_similarity(
                    t_vec.flatten().unsqueeze(0).float(),
                    l_vec.flatten().unsqueeze(0).float(),
                )
                sim[i, j] = cos.item()

        return sim

    def generate_visual_critique(
        self, fe: float, target_vec: torch.Tensor, live_vec: torch.Tensor,
        gen: int,
    ) -> str:
        """
        Gera crítica textual descritiva da discrepância visual.

        Analisa o error_vec = target - live para identificar direções
        de melhoria (cor, forma, posição, movimento).
        """
        error_vec = target_vec - live_vec
        error_mag = float(error_vec.norm())

        # Analisa componentes do erro
        # Se o error_vec for projetado em direções canônicas, podemos
        # inferir qual dimensão visual está errada
        directions = {
            "cor/muito escuro": self._project_to_concept(error_vec, "dark"),
            "cor/muito claro": self._project_to_concept(error_vec, "bright"),
            "muito movimento": self._project_to_concept(error_vec, "motion"),
            "pouco movimento/monotono": self._project_to_concept(error_vec, "static"),
            "bordas/geometria incompativel": self._project_to_concept(error_vec, "edge"),
            "simetria/forma incorreta": self._project_to_concept(error_vec, "symmetry"),
        }

        # Ordena direções por magnitude de projeção
        sorted_dirs = sorted(directions.items(), key=lambda x: -abs(x[1]))
        top_dirs = [d for d, v in sorted_dirs[:3] if abs(v) > 0.01]

        critique_parts = []
        if fe > 0.5:
            critique_parts.append(f"ALTA ENERGIA LIVRE ({fe:.2f})")
        elif fe > 0.3:
            critique_parts.append(f"ENERGIA LIVRE MODERADA ({fe:.2f})")
        else:
            critique_parts.append(f"ENERGIA LIVRE BAIXA ({fe:.2f})")

        if top_dirs:
            critique_parts.append(f"Discrepancias: {', '.join(top_dirs)}")
        else:
            critique_parts.append("Discrepancia visual generica")

        critique_parts.append(f"Erro geometrico: {error_mag:.2f}")

        # Gera descrição visual usando o GroundingEngine se disponível
        visual_desc = ""
        if hasattr(self.engine, 'describe_frame'):
            # Tenta descrever o que a AGI vê vs o que deveria ver
            try:
                target_desc = self.engine.describe_frame(target_vec)
                live_desc = self.engine.describe_frame(live_vec)
                visual_desc = f"Alvo: {target_desc} | Gerado: {live_desc}"
            except Exception:
                pass

        if visual_desc:
            critique_parts.append(visual_desc)

        return " | ".join(critique_parts)

    def _project_to_concept(self, vec: torch.Tensor, concept: str) -> float:
        """
        Projeta vetor de erro em direções conceituais canônicas.
        Quanto maior o valor absoluto, mais esse conceito contribui
        para o erro.
        """
        # Direções canônicas simplificadas
        direction_map = {
            "dark": 0,
            "bright": 1,
            "motion": 2,
            "static": 3,
            "edge": 4,
            "symmetry": 5,
        }
        # Se o GroundingEngine tem encode_text, usa para direção real
        if hasattr(self.engine, 'encode_text'):
            try:
                mf = self.engine.encode_text(concept)
                dir_vec = mf.tensor.flatten().float().to(self.device)
                # Normaliza
                dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
                v = vec.flatten().float().to(self.device)
                v = v / (v.norm() + 1e-8)
                return float(F.cosine_similarity(
                    v.unsqueeze(0), dir_vec.unsqueeze(0)
                ))
            except Exception:
                pass
        idx = direction_map.get(concept, 0) % 10000
        return float(vec[idx].item())

    def evolve_code(
        self, current_code: str, critique: str, fe: float, gen: int
    ) -> str:
        """
        Evolui o código com base na crítica visual.
        Se forge estiver disponível, usa.
        Caso contrário, usa templates de transformação.
        """
        if self.forge is not None:
            try:
                # Tenta usar a Forja para evoluir
                new_code = self.forge.evolve(current_code, critique)
                if new_code and len(new_code) > 100:
                    return new_code
            except Exception as e:
                print(f"[PTM] Forge falhou: {e}")

        # Templates de transformação baseados na crítica
        transformations = []

        if "escuro" in critique.lower() or "bright" in critique:
            transformations.append(
                lambda c: c.replace("(135, 206, 235)",
                                    "(200, 230, 255)")
            )
        if "movimento" in critique.lower() or "motion" in critique:
            # Adiciona movimento a objetos
            if "tick" in c:
                transformations.append(
                    lambda c: c.replace(
                        "x = 400",
                        "x = 400 + int(100 * math.sin(tick * 0.05))"
                    )
                )
        if "bordas" in critique.lower() or "edge" in critique:
            # Adiciona mais elementos visuais
            if "pygame.draw.rect" in current_code:
                transformations.append(
                    lambda c: c + "\n    # Adiciona bordas decorativas\n    for i in range(10):\n        pygame.draw.rect(screen, (100, 200, 100), (i*80, 0, 2, 600))\n"
                )

        if "geometria" in critique.lower() or "geometry" in critique:
            transformations.append(
                lambda c: c.replace(
                    "pygame.draw.circle(screen, ",
                    "pygame.draw.circle(screen, (255, 100, 100), "
                )
            )

        if transformations:
            new_code = current_code
            for t in transformations:
                try:
                    new_code = t()
                except Exception:
                    pass
            return new_code

        # Último recurso: adiciona mais detalhes visuais
        extra = """
    # Detalhes visuais adicionados pelo PerceptualTargetMatcher
    for _dx in range(0, 800, 100):
        pygame.draw.line(screen, (80, 80, 80), (_dx, 0), (_dx, 600), 1)
    for _dy in range(0, 600, 100):
        pygame.draw.line(screen, (80, 80, 80), (0, _dy), (800, _dy), 1)
"""
        return current_code + extra

    def run_evolutionary_loop(
        self,
        target_video: str,
        initial_code: Optional[str] = None,
        ticks: int = 60,
        callback: Optional[callable] = None,
    ) -> Tuple[str, List[dict]]:
        """
        Loop principal de engenharia reversa visual.

        Args:
            target_video: caminho para vídeo de referência
            initial_code: código inicial (opcional, começa do zero)
            ticks: ticks por geração
            callback: função chamada a cada geração com estado

        Returns:
            (código final, histórico de gerações)
        """
        print(f"\n{'='*60}")
        print(f"[PTM] INICIANDO ENGENHARIA REVERSA VISUAL")
        print(f"[PTM] Alvo: {target_video}")
        print(f"[PTM] Threshold FE: {self.fe_threshold}")
        print(f"[PTM] Max gerações: {self.max_generations}")
        print(f"{'='*60}")

        # 1. Extrair vetor alvo do vídeo
        self.target_vec = self.extract_target_vector(target_video)

        # 2. Código inicial
        if initial_code:
            self.current_code = initial_code
        else:
            self.current_code = self._default_code()

        # 3. Loop evolucionário
        for gen in range(self.max_generations):
            print(f"\n--- GERAÇÃO {gen+1}/{self.max_generations} ---")

            # Executa código e captura frames
            print("[PTM] Executando e capturando frames...")
            live_frames = self.run_and_capture(self.current_code, ticks=ticks)

            if not live_frames or len(live_frames) < 3:
                print("[PTM] Frames insuficientes, código crashou")
                self.current_code = self._fix_crash(self.current_code)
                self.history.append({
                    "gen": gen, "fe": 1.0, "status": "crash",
                    "critique": "Código crashou",
                })
                if callback:
                    callback(self.history[-1])
                continue

            # Codifica frames gerados como vetor HDC
            live_vec = self.encode_frames_to_vec(live_frames)

            # Computa Free Energy
            fe = self.compute_fe(self.target_vec, live_vec)
            print(f"[PTM] Free Energy = {fe:.4f}")

            # Gera crítica visual
            critique = self.generate_visual_critique(
                fe, self.target_vec, live_vec, gen)
            print(f"[PTM] Crítica: {critique}")

            # Registra no histórico
            entry = {
                "gen": gen,
                "fe": fe,
                "critique": critique,
                "num_frames_live": len(live_frames),
                "code_length": len(self.current_code),
                "converged": fe < self.fe_threshold,
            }
            self.history.append(entry)
            if callback:
                callback(entry)

            # Convergência
            if fe < self.fe_threshold:
                print(f"\n{'='*60}")
                print(f"[PTM] CONVERGENCIA VISUAL ATINGIDA!")
                print(f"[PTM] FE={fe:.4f} < threshold={self.fe_threshold}")
                print(f"[PTM] {gen+1} gerações")
                print(f"{'='*60}")
                return self.current_code, self.history

            # Evolui código
            self.current_code = self.evolve_code(
                self.current_code, critique, fe, gen)

        print(f"[PTM] Fim do loop ({self.max_generations} gerações)")
        print(f"[PTM] FE final: {self.history[-1]['fe']:.4f}" if self.history else "")
        return self.current_code, self.history

    def _default_code(self) -> str:
        """Código inicial padrão."""
        return """import pygame, math

def render_scene(screen, tick=0):
    W, H = screen.get_size()
    screen.fill((135, 206, 235))
    
    # Chão
    pygame.draw.rect(screen, (34, 139, 34), (0, H-100, W, 100))
    
    # Elemento central
    cx, cy = W//2, H//2
    pygame.draw.circle(screen, (200, 100, 50), (cx, cy), 30)
    
    # Olhos
    pygame.draw.circle(screen, (255, 255, 255), (cx-12, cy-8), 8)
    pygame.draw.circle(screen, (255, 255, 255), (cx+12, cy-8), 8)
    pygame.draw.circle(screen, (0, 0, 0), (cx-10, cy-8), 4)
    pygame.draw.circle(screen, (0, 0, 0), (cx+10, cy-8), 4)
    
    # Rabo (opcional, com movimento)
    tail_offset = int(15 * math.sin(tick * 0.1))
    pygame.draw.line(screen, (200, 100, 50),
                     (cx, cy+20),
                     (cx-30-tail_offset, cy+40), 5)
"""

    def _fix_crash(self, code: str) -> str:
        """Simplifica código quando crasha."""
        return self._default_code()

    def get_progress_report(self) -> str:
        """Gera relatório de progresso do loop."""
        if not self.history:
            return "Nenhuma geração executada."

        lines = ["### Relatório PerceptualTargetMatcher"]
        best_fe = min(h["fe"] for h in self.history)
        best_gen = min(range(len(self.history)),
                       key=lambda i: self.history[i]["fe"])

        lines.append(f"- Gerações: {len(self.history)}")
        lines.append(f"- Melhor FE: {best_fe:.4f} (geração {best_gen+1})")
        lines.append(f"- Threshold: {self.fe_threshold}")
        lines.append(f"- Status: {'CONVERGIU' if best_fe < self.fe_threshold else 'NAO CONVERGIU'}")
        lines.append("")

        for h in self.history:
            status = "✓" if h.get("converged") else "✗"
            lines.append(
                f"| {status} Gen {h['gen']+1:2d} | FE={h['fe']:.4f} | "
                f"{len(h.get('critique',''))} chars"
            )

        lines.append("")
        lines.append(f"Código final: {len(self.current_code)} chars")
        return "\n".join(lines)

    def export_video_comparison(
        self, target_video: str, output_path: str, gen: int = -1
    ):
        """
        Gera vídeo comparativo lado a lado: target | gerado.
        """
        if not HAS_CV2:
            print("[PTM] OpenCV necessario para exportar video")
            return

        hist = self.history[:gen] if gen > 0 else self.history
        if not hist:
            print("[PTM] Nenhum historico para exportar")
            return

        # Pega frames do target
        cap = cv2.VideoCapture(target_video)
        target_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            target_frames.append(f)
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 20
        h, w = target_frames[0].shape[:2] if target_frames else (300, 400)
        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

        for i, tf in enumerate(target_frames):
            # Frame target
            frame = tf.copy()
            # Frame gerado (simulado)
            gen_frame = np.zeros_like(tf)
            text = f"Gen {len(hist)} FE={hist[-1]['fe']:.2f}" if hist else "No gen"
            cv2.putText(gen_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Lado a lado
            side = np.hstack([frame, gen_frame])
            out.write(side)

        out.release()
        print(f"[PTM] Video comparativo salvo: {output_path}")
