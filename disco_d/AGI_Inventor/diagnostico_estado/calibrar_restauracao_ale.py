"""Tela em cache versus estado restaurado; semente29 fora do teste137."""

import json
from collections import deque

import numpy as np

from coleta import gray, make_ale
from coletar_experiencia_cruzada import StepBudget, digest, play
from experiencia_cruzada import save_json
from piloto_planos_propostos import HERE, sha


def main():
    path = HERE / "resultados" / "calibracao_restauracao_ale.json"
    if path.exists():
        raise RuntimeError("calibração já existe; preservar")
    world, rng, budget = make_ale("breakout", 29), np.random.default_rng(29), StepBudget(1000)
    actions = list(world.getMinimalActionSet()[:4])
    frame = gray(world)
    history = deque([frame.copy() for _ in range(16)], maxlen=16)
    for _ in range(128):
        budget.act(world, actions[int(rng.integers(0, 4))])
        if world.game_over():
            world.reset_game()
            frame = gray(world)
            history = deque([frame.copy() for _ in range(16)], maxlen=16)
        else:
            history.append(gray(world))
    state, initial_frame = world.cloneSystemState(), gray(world).copy()
    word = [2, 1, 1, 1, 2, 0, 2, 0, 3, 1, 3, 1, 0, 0, 0, 0]
    first = play(world, actions, word, history, budget)
    final_frame = gray(world).copy()
    world.restoreSystemState(state)
    restored = world.cloneSystemState()
    displayed = gray(world).copy()
    equal_state = bool(restored.equals(state))
    repeated = play(world, actions, word, history, budget)
    equal_replay = all(np.array_equal(a, b) for a, b in zip(first, repeated))
    record = {"status": "measured_display_cache_and_state_separately", "world_seed": 29,
              "restored_state_equals_original": equal_state, "full_program_replay_equal": bool(equal_replay),
              "display_after_restore_equals_root": bool(np.array_equal(displayed, initial_frame)),
              "display_after_restore_equals_branch_final": bool(np.array_equal(displayed, final_frame)),
              "initial_frame_sha256": digest(initial_frame), "branch_final_frame_sha256": digest(final_frame),
              "display_after_restore_sha256": digest(displayed), "ale_calls": budget.used,
              "source_sha256": sha(HERE / "calibrar_restauracao_ale.py"), "agent_received_internal_state": False,
              "reserved137_outcomes_used": 0, "not_a_capability_experiment": True}
    save_json(path, record)
    print(json.dumps(record, indent=2))
    if not equal_state or not equal_replay:
        raise RuntimeError("a restauração não foi validada")


if __name__ == "__main__":
    main()
