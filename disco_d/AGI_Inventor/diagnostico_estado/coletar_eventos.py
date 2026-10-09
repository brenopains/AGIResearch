"""Coleta própria para leitores de eventos congelados; sem modificar o coletor anterior."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import json
import time
from collections import deque
import numpy as np
from piloto_leitor_eventos import HERE, PROPOSAL, BODIES, TEST_SEED, fresh_bank, require, save_new, sha, verify_frozen
from coletar_prospectiva import Budget, play
from coleta import gray, make_ale


def collect(game):
    verify_frozen()
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    output = fresh_bank(game)
    require(not list((HERE / "resultados").glob(output.name + "*")), "coleta iniciada antes; preservar")
    frozen_sha, proposal_sha = sha(BODIES), sha(PROPOSAL)
    budget = Budget(proposal)
    programs = np.asarray(proposal["programs"], int)
    ale, rng = make_ale(game, TEST_SEED), np.random.default_rng(TEST_SEED)
    actions = list(ale.getMinimalActionSet()[:4])
    require(len(actions) == 4, "alfabeto físico insuficiente")
    records, hs, rs, vs, roots, episodes = [], [], [], [], [], []
    event_path = output.with_suffix(".jsonl")
    checkpoint = output.with_name(output.name + "_checkpoint.json")
    stream = event_path.open("x", encoding="utf-8")

    def emit(record):
        records.append(record)
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()

    def snapshot(status):
        value = {"status": status, "completed_roots": len(roots), "ale_calls": budget.used,
            "proposal_sha256": proposal_sha, "bodies_sha256_before_collection": frozen_sha,
            "can_resume_exact_ale_state": False, "partial_chunks_not_eligible_for_evaluation": True}
        temporary = checkpoint.with_name(checkpoint.name + ".partial")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(checkpoint)

    certified = 0
    try:
        for episode in range(proposal["episodes"]):
            ale.reset_game()
            first = gray(ale)
            history = deque([first.copy() for _ in range(16)], maxlen=16)
            for point in range(8):
                for _ in range(128):
                    budget.act(ale, actions[int(rng.integers(0, 4))])
                    if ale.game_over():
                        ale.reset_game()
                        first = gray(ale)
                        history = deque([first.copy() for _ in range(16)], maxlen=16)
                    else:
                        history.append(gray(ale))
                root = 8 * episode + point
                emit({"event": "walk_checkpoint", "episode": episode, "root": root, "ale_calls": 128})
                if point not in proposal["checkpoints"]:
                    continue
                state = ale.cloneSystemState()
                branches = []
                for j, word in enumerate(programs):
                    ale.restoreSystemState(state)
                    require(ale.cloneSystemState().equals(state), "restauração física divergiu")
                    row = play(ale, actions, word, history, budget)
                    branches.append(row)
                    emit({"event": "common_program_measured", "root": root, "episode": episode, "program": j,
                        "word": word.tolist(), "rewards": row[1].tolist(), "valid": row[2].tolist(), "ale_calls": row[3]})
                    if j == 0 and point == proposal["checkpoints"][0]:
                        ale.restoreSystemState(state)
                        replay = play(ale, actions, word, history, budget)
                        require(all(np.array_equal(a, b) for a, b in zip(row, replay)), "replay integral divergiu")
                        certified += 1
                        emit({"event": "full_program_replay_certified", "root": root, "ale_calls": replay[3]})
                root_h = np.stack([row[0] for row in branches])
                root_r = np.stack([row[1] for row in branches])
                root_v = np.stack([row[2] for row in branches])
                chunk = output.with_name(output.name + f"_root_{root:02d}.npz")
                np.savez_compressed(chunk, histories=root_h, rewards=root_r, valid=root_v,
                    episode=episode, root=root, programs=programs)
                hs.append(root_h)
                rs.append(root_r)
                vs.append(root_v)
                roots.append(root)
                episodes.append(episode)
                ale.restoreSystemState(state)
                snapshot("partial_collecting_not_evaluated")
            print(json.dumps({"game": game, "seed": TEST_SEED, "episode": episode,
                "roots": len(roots), "ale_calls": budget.used}), flush=True)
        budget.check()
        require(len(roots) == 32 and certified == 8 and budget.used == sum(r["ale_calls"] for r in records), "contabilidade incompleta")
        require(frozen_sha == sha(BODIES) and proposal_sha == sha(PROPOSAL), "congelamento mudou durante coleta")
        np.savez_compressed(output.with_suffix(".npz"), histories=np.stack(hs), rewards=np.stack(rs), valid=np.stack(vs),
            programs=programs, roots=np.asarray(roots), episode=np.asarray(episodes))
        stream.close()
        save_new(output.with_suffix(".json"), {"status": "event_evaluation_bank_complete", "game": game, "seed": TEST_SEED,
            "roots": 32, "programs_per_root": 8, "certified_replays": certified, "ale_calls": budget.used,
            "wall_seconds": time.perf_counter() - budget.started, "maximum_sampled_rss_bytes": budget.peak,
            "proposal_sha256": proposal_sha, "collector_sha256": sha(HERE / "coletar_eventos.py"),
            "bank_sha256": sha(output.with_suffix(".npz")), "events_sha256": sha(event_path),
            "bodies_sha256_before_collection": frozen_sha, "human_level_goal_achieved": False})
        snapshot("collection_finished_complete_bank_separate")
        return {"game": game, "seed": TEST_SEED, "status": "event_evaluation_bank_complete",
            "ale_calls": budget.used, "wall_seconds": time.perf_counter() - budget.started, "certified_replays": certified}
    except BaseException as error:
        stream.close()
        snapshot("interrupted_inconclusive")
        save_new(output.with_name(output.name + "_interrupcao.json"), {"status": "inconclusive", "error": repr(error),
            "ale_calls": budget.used, "do_not_restart_without_accounting_previous_calls": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    print(json.dumps(collect(parser.parse_args().game), ensure_ascii=False, indent=2))
