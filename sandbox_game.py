"""
Simple Pygame Sandbox — canvas modificavel pelo AGI.
O AGI observa, critica e reescreve este jogo para melhora-lo.
"""

import sys, os, math, json
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import numpy as np

WIDTH, HEIGHT = 400, 300
BG_COLOR = (135, 206, 235)
FPS = 60
MAX_TICKS = 200


def render_scene(screen, tick, t):
    """Renderiza a cena do jogo. AGI pode modificar esta funcao."""
    screen.fill(BG_COLOR)

    # Circulo que se move
    cx = 200 + int(100 * math.sin(t * 0.03))
    cy = 150 + int(50 * math.cos(t * 0.05))
    pygame.draw.circle(screen, (139, 69, 19), (cx, cy), 30)

    # Olhos
    pygame.draw.circle(screen, (255, 255, 255), (cx - 10, cy - 5), 8)
    pygame.draw.circle(screen, (255, 255, 255), (cx + 10, cy - 5), 8)
    pygame.draw.circle(screen, (0, 0, 0), (cx - 8, cy - 5), 4)
    pygame.draw.circle(screen, (0, 0, 0), (cx + 8, cy - 5), 4)

    # Rabo (linha que oscila)
    tail_len = 30 + 10 * math.sin(t * 0.1)
    tail_angle = math.pi * 0.25 * math.sin(t * 0.15)
    tx = cx - 30
    ty = cy + 5
    ex = tx - tail_len * math.cos(tail_angle)
    ey = ty - tail_len * math.sin(tail_angle)
    pygame.draw.line(screen, (139, 69, 19), (tx, ty), (ex, ey), 4)


def run_game(headless: bool = False, output_dir: str = "D:/temp/self_improve") -> dict:
    """Executa o jogo por MAX_TICKS ticks. Salva screenshots e log."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "state_log.txt")

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AGI Sandbox")
    clock = pygame.time.Clock()

    frames_saved = []
    with open(log_path, "w", encoding="utf-8") as f_log:
        for tick in range(MAX_TICKS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    f_log.write(f"[STATE] tick={tick} event=QUIT\n")
                    break

            t = tick / FPS
            render_scene(screen, tick, t)

            state_line = f"[STATE] tick={tick} t={t:.2f} {_extract_state(screen)}"
            f_log.write(state_line + "\n")

            if tick in (50, 100, 150, 199):
                frame_path = os.path.join(output_dir, f"frame_{tick:04d}.png")
                pygame.image.save(screen, frame_path)
                frames_saved.append(str(frame_path))

            if not headless:
                pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()
    return {"log_path": log_path, "frames": frames_saved, "total_ticks": MAX_TICKS, "output_dir": output_dir}


def _extract_state(screen) -> str:
    arr = pygame.surfarray.array3d(screen)
    colors = np.unique(arr.reshape(-1, 3), axis=0)
    return f"colors={len(colors)} w={arr.shape[1]} h={arr.shape[0]}"


if __name__ == "__main__":
    result = run_game()
    print(json.dumps(result, indent=2))
