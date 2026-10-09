"""Banco pareado de experiências visuais; nenhuma variável do jogo entra no aprendiz.

Cada situação é clonada antes de quatro programas anônimos. Todas as ramificações
contam no orçamento. Guardamos também os 16 quadros anteriores à situação.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
from ale_py import ALEInterface, LoggerMode, roms  # noqa: E402

ALEInterface.setLoggerMode(LoggerMode.Error)
PREFIXES = (8, 32, 128)
DEV_GAMES = ("breakout", "pong")


def gray(ale: ALEInterface) -> np.ndarray:
    rgb = ale.getScreenRGB()
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)


def small(frame: np.ndarray, side: int) -> np.ndarray:
    if side == 42:
        return frame.reshape(42, 5, 32, 5).mean((1, 3), dtype=np.float32) / 255.0
    if side == 21:
        return frame.reshape(21, 10, 16, 10).mean((1, 3), dtype=np.float32) / 255.0
    raise ValueError("side deve ser 21 ou 42")


def make_ale(game: str, seed: int) -> ALEInterface:
    if game not in DEV_GAMES:
        raise ValueError("Somente jogos de desenvolvimento nesta bancada")
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("frame_skip", 4)
    ale.setInt("max_num_frames_per_episode", 108000)
    ale.loadROM(roms.get_rom_path(game))
    return ale


def branch(ale: ALEInterface, action: int, horizon: int, budget: list[int]):
    images, rewards = [], []
    total = 0.0
    stopped = False
    for t in range(1, horizon + 1):
        if budget[0] <= 0:
            raise RuntimeError("orçamento de passos esgotado")
        total += float(ale.act(action))
        budget[0] -= 1
        if ale.game_over():
            stopped = True
        if t in PREFIXES:
            images.append(small(gray(ale), 21))
            rewards.append(total)
        if stopped:
            break
    for future in PREFIXES:
        if future > t:
            images.append(small(gray(ale), 21))
            rewards.append(total)
    assert len(images) == len(PREFIXES)
    return np.stack(images), np.asarray(rewards, dtype=np.float32)


def collect(game: str, output: Path, seed: int = 7, episodes: int = 8,
            roots_per_episode: int = 8, interval: int = 24,
            budget_steps: int = 50000) -> dict:
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    if len(actions) < 4:
        raise ValueError("são necessários quatro comandos anônimos")
    rng = np.random.default_rng(seed)
    budget = [budget_steps]
    histories, end_images, returns, groups = [], [], [], []
    replay_failures = []
    for episode in range(episodes):
        ale.reset_game()
        current = gray(ale)
        history = deque([current.copy() for _ in range(16)], maxlen=16)
        collected = 0
        main_step = 0
        while collected < roots_per_episode:
            if budget[0] <= 4 * PREFIXES[-1] + interval + 2 * PREFIXES[0]:
                raise RuntimeError("orçamento de passos insuficiente para concluir um conjunto pareado")
            if ale.game_over():
                ale.reset_game()
                current = gray(ale)
                history = deque([current.copy() for _ in range(16)], maxlen=16)
            action = actions[int(rng.integers(len(actions)))]
            ale.act(action)
            budget[0] -= 1
            current = gray(ale)
            history.append(current.copy())
            main_step += 1
            if main_step < 32 or main_step % interval:
                continue
            state = ale.cloneSystemState()
            # A mesma ação duas vezes testa se o par é uma experiência controlada.
            if collected == 0:
                repeat = []
                for _ in range(2):
                    ale.restoreSystemState(state)
                    im, reward = branch(ale, actions[0], PREFIXES[0], budget)
                    repeat.append((im[0], reward[0]))
                replay_failures.append(not (
                    np.array_equal(repeat[0][0], repeat[1][0])
                    and repeat[0][1] == repeat[1][1]
                ))
                if replay_failures[-1]:
                    raise RuntimeError("o snapshot não reproduz o mesmo programa; contraste causal indisponível")
            outcomes, gains = [], []
            for a in actions:
                ale.restoreSystemState(state)
                ims, rew = branch(ale, a, PREFIXES[-1], budget)
                outcomes.append(ims)
                gains.append(rew)
            ale.restoreSystemState(state)
            histories.append(np.stack(history).astype(np.uint8))
            end_images.append(np.stack(outcomes, axis=1))  # horizonte, ação, y, x
            returns.append(np.stack(gains, axis=1))        # horizonte, ação
            groups.append(episode)
            collected += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, histories=np.stack(histories),
                        end_images=np.stack(end_images).astype(np.float32),
                        returns=np.stack(returns), episode=np.asarray(groups),
                        actions=np.asarray([a.value for a in actions]), prefixes=np.asarray(PREFIXES))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "game": game, "seed": seed, "roots": len(histories), "episodes": episodes,
        "actions": len(actions), "budget_steps": budget_steps,
        "steps_used": budget_steps - budget[0], "replay_failures": int(sum(replay_failures)),
        "dataset_sha256": digest, "source": "ALE 0.12.1, frame_skip=4, sticky=0",
        "visibility": "somente pixels, recompensa, ações e tempo ao aprendiz",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=DEV_GAMES)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--budget", type=int, default=50000)
    args = parser.parse_args()
    path = args.out or HERE / "resultados" / f"{args.game}_bank.npz"
    print(json.dumps(collect(args.game, path, args.seed, args.episodes,
                             args.roots, budget_steps=args.budget), indent=2))
