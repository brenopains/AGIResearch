"""Experiência nova cruzada, com histórias intermediárias e orçamento físico."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import argparse
import ctypes
import json
import time
from collections import deque
import numpy as np
from piloto_prospectivo import HERE, BODIES, PROPOSAL, SEEDS, bank_path, require, save_new, sha, verify, verify_bodies
from coleta import gray, make_ale
from coletar_transicoes_mistas import compressed_history


def rss_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise RuntimeError("não foi possível medir a memória do coletor")
    return int(counters.WorkingSetSize)


class Budget:
    def __init__(self, proposal):
        self.proposal, self.used, self.started, self.peak = proposal, 0, time.perf_counter(), 0

    def check(self):
        require(time.perf_counter() - self.started < self.proposal["maximum_collection_wall_seconds"], "teto de tempo da coleta atingido")
        current = rss_bytes()
        self.peak = max(self.peak, current)
        require(current < self.proposal["maximum_worker_rss_bytes"], "teto de memória do coletor atingido")

    def act(self, ale, action):
        require(self.used < self.proposal["maximum_calls_per_bank"], "teto de chamadas ALE atingido")
        if self.used % 128 == 0:
            self.check()
        reward = float(ale.act(action))
        self.used += 1
        return reward


def play(ale, actions, word, history, budget):
    branch = deque([frame.copy() for frame in history], maxlen=16)
    hs, rewards, valid, elapsed, ended = [compressed_history(branch)], [], [True], 0, False
    for symbol in word:
        gain = 0.
        for _ in range(8):
            if ended:
                break
            gain += budget.act(ale, actions[int(symbol)])
            elapsed += 1
            branch.append(gray(ale))
            ended = bool(ale.game_over())
        rewards.append(gain)
        valid.append(not ended)
        hs.append(compressed_history(branch))
    return np.stack(hs), np.asarray(rewards), np.asarray(valid), elapsed


def collect(game, seed):
    proposal = verify()
    require(seed in SEEDS, "semente fora do protocolo")
    output = bank_path(game, seed)
    require(not list((HERE / "resultados").glob(output.name + "*")), "coleta iniciada antes; preservar")
    frozen_sha = sha(BODIES) if seed == 157 else None
    if seed == 157:
        verify_bodies()
    elif seed == 151:
        require(all(bank_path(g, 149).with_suffix(".json").exists() for g in ("breakout", "pong")), "construção deve preceder validação")
    budget = Budget(proposal)
    programs = np.asarray(proposal["programs"][:8 if seed == 157 else 4], int)
    ale, rng = make_ale(game, seed), np.random.default_rng(seed)
    actions = list(ale.getMinimalActionSet()[:4])
    require(len(actions) == 4, "alfabeto físico insuficiente")
    records, hs, rs, vs, roots, episodes = [], [], [], [], [], []
    event_path = output.with_suffix(".jsonl")
    checkpoint = output.with_name(output.name + "_checkpoint.json")
    event_stream = event_path.open("x", encoding="utf-8")

    def emit(record):
        records.append(record)
        event_stream.write(json.dumps(record, allow_nan=False) + "\n")
        event_stream.flush()

    def snapshot(status):
        value = {"status": status, "completed_roots": len(roots), "ale_calls": budget.used,
            "proposal_sha256": sha(PROPOSAL), "can_resume_exact_ale_state": False,
            "partial_chunks_not_eligible_for_evaluation": True}
        temporary = checkpoint.with_name(checkpoint.name + ".partial")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(checkpoint)

    certified = 0
    try:
        for episode in range(8):
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
                    require(ale.cloneSystemState().equals(state), "restauração de estado divergiu")
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
                np.savez_compressed(chunk, histories=root_h, rewards=root_r, valid=root_v, episode=episode, root=root, programs=programs)
                hs.append(root_h)
                rs.append(root_r)
                vs.append(root_v)
                roots.append(root)
                episodes.append(episode)
                ale.restoreSystemState(state)
                snapshot("partial_collecting_not_evaluated")
            print(json.dumps({"game": game, "seed": seed, "episode": episode, "roots": len(roots), "ale_calls": budget.used}), flush=True)
        budget.check()
        require(len(roots) == 32 and certified == 8 and budget.used == sum(r["ale_calls"] for r in records), "contabilidade incompleta")
        np.savez_compressed(output.with_suffix(".npz"), histories=np.stack(hs), rewards=np.stack(rs), valid=np.stack(vs),
            programs=programs, roots=np.asarray(roots), episode=np.asarray(episodes))
        event_stream.close()
        save_new(output.with_suffix(".json"), {"status": "prospective_bank_complete", "game": game, "seed": seed,
            "roots": 32, "programs_per_root": len(programs), "certified_replays": certified, "ale_calls": budget.used,
            "wall_seconds": time.perf_counter() - budget.started, "maximum_sampled_rss_bytes": budget.peak,
            "proposal_sha256": sha(PROPOSAL), "collector_sha256": sha(HERE / "coletar_prospectiva.py"),
            "bank_sha256": sha(output.with_suffix(".npz")), "events_sha256": sha(event_path),
            "frozen_bodies_sha256_before_collection": frozen_sha, "human_level_goal_achieved": False})
        snapshot("collection_finished_complete_bank_separate")
        return {"game": game, "seed": seed, "status": "prospective_bank_complete", "ale_calls": budget.used,
            "wall_seconds": time.perf_counter() - budget.started, "certified_replays": certified}
    except BaseException as error:
        event_stream.close()
        snapshot("interrupted_inconclusive")
        save_new(output.with_name(output.name + "_interrupcao.json"), {"status": "inconclusive", "error": repr(error),
            "ale_calls": budget.used, "do_not_restart_without_accounting_previous_calls": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), ensure_ascii=False, indent=2))
