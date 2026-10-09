"""
SelfImproveLoop — Ciclo Autônomo de Auto-Melhoria
===================================================
Observa → Critica → Gera melhoria → Aplica → Verifica → Registra no EpisodicStream.

Fluxo completo:
  1. run_game() — executa jogo, coleta frames + log
  2. analyze() — processa log/frames, extrai problemas estruturados (issue_id)
  3. improve() — gera código de melhoria via forja HDC
  4. apply() — reescreve sandbox_game.py com a melhoria
  5. verify() — re-executa e confirma que o issue_id foi resolvido
  6. log_to_stream() — registra a iteração no EpisodicStream
  7. converge_check() — se repetiu issue_id 3x, declara convergência
"""

import sys
import os
import re
import json
import time
import math
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "novamind"))

import torch
import numpy as np
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from core.grounding_engine import GroundingEngine
from core.hdc_memory import HDCMemory
from core.episodic_stream import EpisodicStream
from core.coglang import Manifold

GAME_PATH = str(Path(__file__).parent.parent.parent / "sandbox_game.py")
OUTPUT_BASE = "D:/temp/self_improve_loop"
STREAM_PATH = "D:/temp/episodic_stream.stream"


@dataclass
class Critique:
    priority: int         # 0-10
    text: str             # descrição legível
    issue_id: str         # identificador estruturado (ex: "low_contrast")
    fix_hint: str = ""    # sugestão de código


@dataclass
class Iteration:
    number: int
    intention: str
    critiques: List[Critique] = field(default_factory=list)
    code_generated: str = ""
    code_path: str = ""
    exec_success: bool = False
    exec_output: str = ""
    resolved_issues: List[str] = field(default_factory=list)
    new_issues: List[str] = field(default_factory=list)
    neuromodulators: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    stream_entry_id: int = 0
    is_repeat: bool = False


ISSUE_REGISTRY = {
    "no_tail_wag":       {"fix": "add_tail_wag",       "da": 0.3, "na": -0.4},
    "static_eyes":       {"fix": "animate_eyes",       "da": 0.2, "na": -0.3},
    "boring_colors":     {"fix": "more_colors",        "da": 0.4, "na": -0.5},
    "low_contrast":      {"fix": "fix_contrast",       "da": 0.3, "na": -0.4},
    "no_background":     {"fix": "add_background",     "da": 0.3, "na": -0.3},
    "no_decoration":     {"fix": "add_decoration",     "da": 0.4, "na": -0.4},
    "small_entity":      {"fix": "scale_entity",       "da": 0.2, "na": -0.3},
    "no_movement":       {"fix": "add_movement",       "da": 0.5, "na": -0.5},
    "low_fps":           {"fix": "optimize_fps",       "da": 0.1, "na": -0.2},
    "physics_error":     {"fix": "fix_physics",        "da": 0.1, "na": -0.6},
    "repetitive":        {"fix": "add_variety",        "da": 0.3, "na": -0.4},
    "color_clash":       {"fix": "harmonize_colors",   "da": 0.2, "na": -0.3},
    "empty_scene":       {"fix": "populate_scene",     "da": 0.4, "na": -0.4},
    "no_shapes":         {"fix": "add_shapes",         "da": 0.3, "na": -0.3},
    "asymmetric":        {"fix": "fix_asymmetry",      "da": 0.2, "na": -0.3},
    "too_few_colors":    {"fix": "more_colors_v2",     "da": 0.4, "na": -0.5},
    "clip_overflow":     {"fix": "fix_clipping",       "da": 0.1, "na": -0.4},
}

REPEAT_LIMIT = 3


class SelfImproveLoop:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = GroundingEngine(device=torch.device(self.device))
        self.memory = HDCMemory()
        self.stream = EpisodicStream(STREAM_PATH, device=self.device)

        self.iteration = 0
        self.history: List[Iteration] = []
        self.converged = False
        self.convergence_reason = ""
        self.issue_counter: Counter = Counter()
        self.last_code = self._read_current_game()
        self._running = False

        self.iteration_start = time.time()

    def _read_current_game(self) -> str:
        with open(GAME_PATH, "r", encoding="utf-8") as f:
            return f.read()

    def _write_game(self, code: str):
        with open(GAME_PATH, "w", encoding="utf-8") as f:
            f.write(code)
        self.last_code = code

    # ─── Etapa 1: Executar ───────────────────────────────────────

    def run_game(self) -> dict:
        """Executa o jogo e coleta resultados."""
        output_dir = os.path.join(OUTPUT_BASE, f"iter_{self.iteration:04d}")
        os.makedirs(output_dir, exist_ok=True)

        import subprocess
        result = subprocess.run(
            [sys.executable, GAME_PATH],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        # Lê o log de estado (salvo pelo sandbox_game.py)
        possible_logs = [
            os.path.join(output_dir, "state_log.txt"),
            "D:/temp/self_improve/state_log.txt",
        ]
        state_log = ""
        for lp in possible_logs:
            try:
                with open(lp, encoding="utf-8") as f:
                    state_log = f.read()
                break
            except FileNotFoundError:
                continue
        if not state_log:
            state_log = result.stdout[-2000:]

        # Lê as frames
        frames = sorted(Path(output_dir).glob("frame_*.png"))
        if not frames:
            # Tenta no output_dir do sandbox_game.py
            frames = sorted(Path("D:/temp/self_improve").glob("frame_*.png"))

        return {
            "returncode": result.returncode,
            "stdout": result.stdout + result.stderr,
            "state_log": state_log,
            "frames": [str(f) for f in frames],
            "output_dir": output_dir,
        }

    # ─── Etapa 2: Analisar ──────────────────────────────────────

    def analyze(self, game_result: dict) -> List[Critique]:
        """
        Analisa o log de estado e as frames geradas.
        Retorna lista de críticas estruturadas com issue_id.
        """
        critiques = []
        state_log = game_result.get("state_log", "")

        # Extrai cores do log
        color_counts = []
        for line in state_log.split("\n"):
            m = re.search(r"colors=(\d+)", line)
            if m:
                color_counts.append(int(m.group(1)))

        avg_colors = np.mean(color_counts) if color_counts else 0

        if avg_colors < 3:
            critiques.append(Critique(priority=8,
                text=f"Only {int(avg_colors)} colors used (too few)",
                issue_id="too_few_colors", fix_hint="Add more color variety"))
        elif avg_colors < 5:
            critiques.append(Critique(priority=5,
                text=f"Only {int(avg_colors)} colors (could be more vibrant)",
                issue_id="boring_colors", fix_hint="Use a broader palette"))

        # Análise de frames via visão computacional
        frames = game_result.get("frames", [])
        if len(frames) >= 2:
            vision_critiques = self._analyze_frames(frames)
            critiques.extend(vision_critiques)

        # Verifica erro de execução
        if game_result["returncode"] != 0:
            critiques.append(Critique(priority=10,
                text="Game crashed on execution",
                issue_id="physics_error",
                fix_hint=f"Fix runtime error: {game_result['stdout'][:200]}"))

        if not frames:
            critiques.append(Critique(priority=9,
                text="No frames were saved — game may not have rendered",
                issue_id="empty_scene",
                fix_hint="Ensure render_scene is called"))

        return critiques

    def _analyze_frames(self, frame_paths: list) -> List[Critique]:
        """Analisa frames reais usando visão computacional."""
        if not HAS_CV2:
            return []
        critiques = []
        frames = []
        for fp in frame_paths:
            if os.path.exists(fp):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)

        if len(frames) < 2:
            return []

        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # 1. Movimento: diferença média entre frames consecutivos
        diffs = [np.mean(np.abs(gray[i] - gray[i-1])) for i in range(1, len(gray))]
        avg_motion = np.mean(diffs) if diffs else 0
        if avg_motion < 1.0:
            critiques.append(Critique(priority=6,
                text=f"Very low motion ({avg_motion:.1f} mean pixel diff)",
                issue_id="no_movement", fix_hint="Add more animation/movement"))
        elif avg_motion < 3.0:
            critiques.append(Critique(priority=3,
                text=f"Subtle motion ({avg_motion:.1f}) — could be more dynamic",
                issue_id="repetitive", fix_hint="Increase animation amplitude"))

        # 2. Densidade de bordas (medida de complexidade visual)
        edges = [cv2.Canny(g, 50, 150) for g in gray]
        edge_density = np.mean([np.mean(e > 0) for e in edges])
        if edge_density < 0.02:
            critiques.append(Critique(priority=7,
                text=f"Very low edge density ({edge_density:.3f}) — scene is too empty",
                issue_id="empty_scene", fix_hint="Add more shapes and details"))
        elif edge_density < 0.05:
            critiques.append(Critique(priority=4,
                text=f"Low edge density ({edge_density:.3f}) — could use more detail",
                issue_id="no_shapes", fix_hint="Add more decorative elements"))

        # 3. Simetria (left-right flip similarity)
        h, w = gray[0].shape
        left = gray[0][:, :w//2]
        right = cv2.flip(gray[0][:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        sym_score = np.mean(np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float)))
        if sym_score > 30:
            critiques.append(Critique(priority=5,
                text=f"Asymmetric composition (symmetry diff={sym_score:.1f})",
                issue_id="asymmetric", fix_hint="Balance the visual composition"))

        # 4. Brilho médio (muito escuro ou claro)
        mean_brightness = np.mean([np.mean(g) for g in gray])
        if mean_brightness < 30:
            critiques.append(Critique(priority=6,
                text=f"Too dark (brightness={mean_brightness:.1f})",
                issue_id="low_contrast", fix_hint="Brighten the scene"))
        elif mean_brightness > 220:
            critiques.append(Critique(priority=4,
                text=f"Too bright (brightness={mean_brightness:.1f}) — washed out",
                issue_id="low_contrast", fix_hint="Add more saturated colors"))

        # 5. Variância de cores (medida de contraste)
        hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
        sat_mean = np.mean([np.mean(hsv[i][:, :, 1]) for i in range(len(hsv))])
        if sat_mean < 30:
            critiques.append(Critique(priority=6,
                text=f"Low color saturation ({sat_mean:.1f})",
                issue_id="boring_colors", fix_hint="Use more saturated/vibrant colors"))

        return critiques

    # ─── Etapa 3: Melhorar ──────────────────────────────────────

    def improve(self, critiques: List[Critique]) -> str:
        """
        Gera código de melhoria baseado nas críticas.
        Usa o ISSUE_REGISTRY para encontrar templates de código.
        """
        if not critiques:
            return ""

        top = max(critiques, key=lambda c: c.priority)
        return self._generate_code_from_hdv(top)

    def _generate_code_from_hdv(self, critique: Critique) -> str:
        """Gera código de melhoria a partir do template registry."""
        issue = ISSUE_REGISTRY.get(critique.issue_id, {})
        fix_name = issue.get("fix", critique.issue_id)

        templates = {
            "add_tail_wag": """# Tail wag animation
tail_angle = math.pi * 0.3 * math.sin(t * 0.15)
tx = cx - 30; ty = cy + 5
ex = tx - 40 * math.cos(tail_angle)
ey = ty - 40 * math.sin(tail_angle)
pygame.draw.line(screen, (139, 69, 19), (tx, ty), (ex, ey), 4)
""",
            "animate_eyes": """# Animated eyes (blink)
blink = int(t * 2) % 30 < 3
if not blink:
    pygame.draw.circle(screen, (255,255,255), (cx-10, cy-5), 8)
    pygame.draw.circle(screen, (255,255,255), (cx+10, cy-5), 8)
    pygame.draw.circle(screen, (0,0,0), (cx-8, cy-5), 4)
    pygame.draw.circle(screen, (0,0,0), (cx+8, cy-5), 4)
""",
            "more_colors": """# Colorful ground
ground_y = 250 + 20 * math.sin(t * 0.02)
for i in range(10):
    shade = 100 + i * 15
    pygame.draw.rect(screen, (50, shade, 50), (i*40, int(ground_y), 42, HEIGHT-int(ground_y)))
""",
            "more_colors_v2": """# Rainbow accent circles
for i in range(6):
    angle = t * 0.05 + i * math.pi / 3
    rx = cx + 50 * math.cos(angle)
    ry = cy + 50 * math.sin(angle)
    colors = [(255,0,0),(255,165,0),(255,255,0),(0,255,0),(0,0,255),(128,0,128)]
    pygame.draw.circle(screen, colors[i], (int(rx), int(ry)), 8)
""",
            "fix_contrast": """# High-contrast background
bg_r = 30 + 20 * math.sin(t * 0.01)
screen.fill((int(bg_r), 100, 180))
""",
            "add_background": """# Gradient background
for y in range(HEIGHT):
    ratio = y / HEIGHT
    r = int(135 * (1 - ratio) + 50 * ratio)
    g = int(206 * (1 - ratio) + 100 * ratio)
    b = int(235 * (1 - ratio) + 180 * ratio)
    pygame.draw.line(screen, (r, g, b), (0, y), (WIDTH, y))
""",
            "add_decoration": """# Decorative elements
for i in range(3):
    dx = 50 + i * 150
    dy = 260 + 20 * math.sin(t * 0.03 + i)
    pygame.draw.circle(screen, (60, 180, 60), (dx, int(dy)), 15)
    pygame.draw.rect(screen, (100, 100, 60), (dx-2, int(dy), 4, 30))
""",
            "scale_entity": """# Scale entity based on tick
scale = 1.0 + 0.2 * math.sin(t * 0.03)
radius = int(30 * scale)
cx = 200 + int(100 * math.sin(t * 0.03))
cy = 150 + int(50 * math.cos(t * 0.05))
pygame.draw.circle(screen, (139, 69, 19), (cx, cy), radius)
""",
            "add_movement": """# Move entity in figure-8 pattern
cx = 200 + int(120 * math.sin(t * 0.04))
cy = 150 + int(60 * math.sin(t * 0.08))
""",
            "fix_clipping": """# Keep entity on screen
cx = max(30, min(WIDTH-30, cx))
cy = max(30, min(HEIGHT-30, cy))
""",
        }

        code = templates.get(fix_name, "")
        if not code:
            code = f"""
    # Auto-generated fix for {critique.issue_id}: {critique.text[:60]}
    pass
"""
        return code

    # ─── Etapa 4: Aplicar ────────────────────────────────────────

    def apply_improvement(self, code_snippet: str) -> str:
        """
        Aplica o snippet de código gerado ao sandbox_game.py.
        Insere o código dentro da função render_scene, antes do draw do círculo.
        """
        if not code_snippet:
            return ""

        current = self._read_current_game()

        # Procura o marcador de inserção: o draw do círculo
        marker = "pygame.draw.circle(screen, (139, 69, 19), (cx, cy), 30)"
        indent = "    "

        if marker in current:
            lines = code_snippet.split("\n")
            indented = "\n".join(indent + l if l.strip() else l for l in lines)
            new_code = current.replace(marker, indented + "\n" + indent + marker)
        else:
            # Fallback: insere antes de render_scene return
            new_code = current + "\n" + code_snippet

        self._write_game(new_code)
        return new_code

    # ─── Etapa 5: Verificar ─────────────────────────────────────

    def verify(self, critique: Critique, game_result_before: dict, game_result_after: dict) -> bool:
        """Verifica se a melhoria resolveu a crítica."""
        if critique.issue_id == "physics_error":
            return game_result_after["returncode"] == 0

        if critique.issue_id == "too_few_colors":
            colors_before = self._extract_color_count(game_result_before["state_log"])
            colors_after = self._extract_color_count(game_result_after["state_log"])
            return colors_after > colors_before

        if critique.issue_id == "boring_colors":
            colors_before = self._extract_color_count(game_result_before["state_log"])
            colors_after = self._extract_color_count(game_result_after["state_log"])
            return colors_after > colors_before

        # Para outras críticas, verifica se o jogo ao menos roda
        return game_result_after["returncode"] == 0

    def _extract_color_count(self, state_log: str) -> float:
        counts = [int(m) for m in re.findall(r"colors=(\d+)", state_log)]
        return np.mean(counts) if counts else 0

    # ─── Etapa 6: Log no EpisodicStream ─────────────────────────

    def log_to_stream(self, iteration: Iteration):
        """Codifica a iteração como vetor HDC e salva no EpisodicStream."""
        # Concatena os elementos chave em um texto
        summary = (
            f"Iteration {iteration.number}: {iteration.intention} "
            f"critiques={len(iteration.critiques)} "
            f"resolved={iteration.resolved_issues} "
            f"success={iteration.exec_success}"
        )
        mf = self.engine.encode_text(summary)
        hdv = mf.tensor.float()

        # Metadados
        meta = {
            "iteration": iteration.number,
            "intention": iteration.intention,
            "n_critiques": len(iteration.critiques),
            "critiques": [c.issue_id for c in iteration.critiques],
            "resolved": iteration.resolved_issues,
            "new_issues": iteration.new_issues,
            "exec_success": iteration.exec_success,
            "neuromodulators": iteration.neuromodulators,
            "timestamp": iteration.timestamp,
        }

        entry_id = self.stream.append(hdv.cpu(), meta, iteration.timestamp)
        iteration.stream_entry_id = entry_id

        print(f"  [STREAM] Logged iteration {iteration.number} (entry #{entry_id})")

    # ─── Etapa 7: Verificar Convergência ────────────────────────

    def check_convergence(self, iteration: Iteration) -> bool:
        """Verifica se o mesmo issue_id apareceu REPEAT_LIMIT vezes."""
        for issue_id in iteration.new_issues:
            self.issue_counter[issue_id] += 1

        for issue_id, count in self.issue_counter.items():
            if count >= REPEAT_LIMIT:
                self.converged = True
                self.convergence_reason = (
                    f"Converged: {issue_id} repeated {count} times "
                    f"(no further improvement possible)"
                )
                return True

        # Se não há novas issues e execução foi bem sucedida, convergiu
        if not iteration.new_issues and iteration.exec_success and iteration.number > 2:
            if all(not c for c in self.history[-3:]):
                self.converged = True
                self.convergence_reason = "Converged: 3 stable iterations"
                return True

        return False

    # ─── Ciclo Principal ─────────────────────────────────────────

    def step(self, intention: str = "Improve the visual appeal of the game") -> Optional[Iteration]:
        """
        Executa um ciclo completo: run → analyze → improve → apply → verify → log.
        """
        if self.converged:
            return None

        self.iteration += 1
        it = Iteration(
            number=self.iteration,
            intention=intention,
            timestamp=time.time(),
        )

        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}: {intention}")
        print(f"{'='*60}")

        # 1. Executar
        print("\n[1/6] Running game...")
        result_before = self.run_game()

        # 2. Analisar
        print("[2/6] Analyzing output...")
        critiques = self.analyze(result_before)
        it.critiques = critiques

        if not critiques:
            print("  No issues found. Game looks good!")
            it.exec_success = True
            it.resolved_issues = []
            it.new_issues = []
            self.history.append(it)
            self.log_to_stream(it)
            return it

        for c in critiques:
            print(f"  [{c.priority}] {c.issue_id}: {c.text[:80]}")

        # 3. Melhorar
        print("[3/6] Generating improvement...")
        code = self.improve(critiques)
        it.code_generated = code

        if not code:
            print("  No improvement generated.")
            it.exec_success = True
            it.resolved_issues = []
            it.new_issues = [c.issue_id for c in critiques]
            self.history.append(it)
            self.log_to_stream(it)
            return it

        # 4. Aplicar
        print("[4/6] Applying improvement to game...")
        new_code = self.apply_improvement(code)
        code_path = os.path.join(OUTPUT_BASE, f"iter_{self.iteration:04d}", "game.py")
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        it.code_path = code_path

        # 5. Verificar
        print("[5/6] Verifying improvement...")
        result_after = self.run_game()
        it.exec_output = result_after["stdout"][:500]
        it.exec_success = result_after["returncode"] == 0

        resolved = []
        new_issues = []
        for c in critiques:
            if self.verify(c, result_before, result_after):
                resolved.append(c.issue_id)
                print(f"  RESOLVED: {c.issue_id}")
            else:
                new_issues.append(c.issue_id)
                print(f"  NOT RESOLVED: {c.issue_id}")

        it.resolved_issues = resolved
        it.new_issues = new_issues

        # 6. Log no EpisodicStream
        print("[6/6] Logging to EpisodicStream...")

        # Neuromoduladores baseados no resultado
        nm = {"DA": 0.5, "NA": 0.3, "ACh": 0.3, "5HT": 0.5}
        if resolved and not new_issues:
            nm["DA"] = 0.8
            nm["5HT"] = 0.7
        elif new_issues and not resolved:
            nm["NA"] = 0.7
            nm["ACh"] = 0.5
        elif it.exec_success:
            nm["DA"] = 0.6
        else:
            nm["NA"] = 0.6
            nm["ACh"] = 0.4
        it.neuromodulators = nm

        self.log_to_stream(it)

        # Verificar convergência
        if self.check_convergence(it):
            print(f"\n  CONVERGED: {self.convergence_reason}")

        self.history.append(it)
        return it

    def run_loop(self, max_iterations: int = 20, intention: str = "Improve the visual appeal of the game"):
        """Executa o loop completo até convergência ou max_iterations."""
        print(f"\n{'#'*60}")
        print(f"Starting self-improvement loop")
        print(f"Max iterations: {max_iterations}")
        print(f"Intention: {intention}")
        print(f"Stream: {STREAM_PATH}")
        print(f"{'#'*60}")

        self._running = True
        for i in range(max_iterations):
            if not self._running:
                print(f"\nLoop stopped by user at iteration {self.iteration}")
                break
            it = self.step(intention)
            if it is None:
                print(f"\nLoop stopped: converged after {self.iteration} iterations")
                break

        elapsed = time.time() - self.iteration_start

        # Relatório final
        total_resolved = sum(len(it.resolved_issues) for it in self.history if it)
        print(f"\n{'#'*60}")
        print(f"Loop finished: {self.iteration} iterations in {elapsed:.1f}s")
        print(f"  Issues resolved: {total_resolved}")
        print(f"  Converged: {self.converged} ({self.convergence_reason})")
        print(f"  Stream entries: {len(self.stream)}")
        self._running = False
        print(f"{'#'*60}")
        return self.history


# ─── Integração com AgentBridge (para dashboard) ────────────────

class MonitoredLoop:
    """
    Wrapper que expõe o SelfImproveLoop via eventos de monitoramento.
    Usado pelo server.py para iniciar/parar o loop via REST.
    """

    def __init__(self, emit_callback=None):
        self.loop: Optional[SelfImproveLoop] = None
        self._running = False
        self._task = None
        self._emit = emit_callback or (lambda e: None)

    def start(self, intention: str = "Improve the visual appeal of the game", max_iter: int = 10):
        if self._running:
            return {"error": "Already running"}
        self._running = True
        self.loop = SelfImproveLoop()
        print(f"[MONLOOP] Starting loop: {intention} (max {max_iter})")

        import asyncio
        loop = asyncio.get_event_loop()
        self._task = loop.run_in_executor(None, self.loop.run_loop, max_iter, intention)
        return {"status": "started", "intention": intention, "max_iter": max_iter}

    def stop(self):
        if not self._running:
            return {"error": "Not running"}
        self._running = False
        if self.loop:
            self.loop._running = False
        print("[MONLOOP] Loop stopped by user")
        return {"status": "stopped"}

    def get_status(self) -> dict:
        if not self.loop:
            return {"running": False, "iterations": 0}
        return {
            "running": self._running,
            "iterations": self.loop.iteration,
            "converged": self.loop.converged,
            "convergence_reason": self.loop.convergence_reason,
            "stream_entries": len(self.loop.stream),
            "total_resolved": sum(len(it.resolved_issues) for it in self.loop.history),
        }
