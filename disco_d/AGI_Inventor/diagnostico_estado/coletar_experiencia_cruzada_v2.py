"""Programas comuns em todas as raízes; contador físico e checkpoint por raiz."""

import argparse
import hashlib
import json
import time
from collections import deque

import numpy as np

from coleta import gray, make_ale
from coletar_transicoes_mistas import compressed_history
from experiencia_cruzada_v2 import CHECKPOINTS, PROPOSAL, TEST_SEED, TRAIN_SEED, base, manifest_path, save_json, verify_proposal
from piloto_planos_propostos import HERE, sha


class StepBudget:
    def __init__(self, limit=50000):
        self.limit, self.used = limit, 0

    def act(self, ale, action):
        if self.used >= self.limit:
            raise RuntimeError("orçamento de chamadas ALE esgotado")
        result = float(ale.act(action))
        self.used += 1
        return result


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def play(ale, actions, word, history, budget):
    branch = deque([f.copy() for f in history], maxlen=16)
    rewards, valid, ended, elapsed = [], [True], False, 0
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
    return np.asarray(rewards), np.asarray(valid), elapsed, compressed_history(branch), digest(branch[-1])


def collect(game, seed):
    proposal = verify_proposal()
    output = base(game, seed)
    if seed not in (TRAIN_SEED, TEST_SEED):
        raise ValueError("semente fora do protocolo")
    if any(output.with_suffix(s).exists() for s in (".npz", ".json", ".jsonl")):
        raise RuntimeError("banco existe; preservar")
    if seed == TEST_SEED:
        manifest = json.loads(manifest_path(game).read_text(encoding="utf-8"))
        for p, expected in manifest["source_hashes"].items():
            if sha(HERE / p) != expected:
                raise RuntimeError("ajuste não está congelado")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    programs = np.asarray(proposal["programs"][:4] if seed == TRAIN_SEED else proposal["programs"])
    ale, rng, budget = make_ale(game, seed), np.random.default_rng(seed), StepBudget()
    prior_path = HERE / "resultados" / f"{game}_experiencia_cruzada_{seed}_interrupcao.json"
    prior_calls = json.loads(prior_path.read_text(encoding="utf-8"))["ale_calls"] if prior_path.exists() else 0
    if not 0 <= prior_calls < 50000:
        raise RuntimeError("custo anterior inválido")
    budget = StepBudget(50000 - prior_calls)
    actions = list(ale.getMinimalActionSet()[:4])
    if len(actions) != 4:
        raise RuntimeError("alfabeto físico insuficiente")
    histories, raw_history_hashes, rewards, valid, lengths, endpoints, final_hashes, roots, episodes = [], [], [], [], [], [], [], [], []
    events, walks, certified = [], 0, 0
    event_path = output.with_suffix(".jsonl")
    checkpoint = output.with_name(output.name + "_checkpoint.json")

    def emit(event):
        events.append(event)
        with event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def snapshot(status):
        path = output.with_name(output.name + f"_checkpoint_{len(histories):02d}.npz")
        if not path.exists():
            temporary = path.with_name(path.stem + "_partial.npz")
            np.savez_compressed(temporary, histories=np.asarray(histories), rewards=np.asarray(rewards), valid=np.asarray(valid),
                                lengths=np.asarray(lengths), roots=np.asarray(roots), episode=np.asarray(episodes), programs=programs)
            temporary.replace(path)
        save_json(checkpoint, {"status": status, "complete_roots": len(histories), "ale_calls_used": budget.used,
                               "proposal_sha256": sha(PROPOSAL), "partial_bank_sha256": sha(path),
                               "partial_bank": path.relative_to(HERE).as_posix(),
                               "can_resume_exact_ale_state": False, "not_eligible_as_finished_experiment": True})

    try:
        for episode in range(8):
            ale.reset_game()
            first = gray(ale)
            history = deque([first.copy() for _ in range(16)], maxlen=16)
            for point in range(8):
                for _ in range(128):
                    budget.act(ale, actions[int(rng.integers(0, 4))])
                    walks += 1
                    if ale.game_over():
                        ale.reset_game()
                        first = gray(ale)
                        history = deque([first.copy() for _ in range(16)], maxlen=16)
                    else:
                        history.append(gray(ale))
                root = 8 * episode + point
                emit({"event": "walk_checkpoint", "episode": episode, "root": root, "ale_calls": 128, "selected": point in CHECKPOINTS})
                if point not in CHECKPOINTS:
                    continue
                state, initial = ale.cloneSystemState(), compressed_history(history)
                raw_hash = digest(np.stack(history))
                rows = []
                for j, word in enumerate(programs):
                    ale.restoreSystemState(state)
                    require_frame = bool(ale.cloneSystemState().equals(state))
                    if not require_frame:
                        raise RuntimeError("estado clonado não foi restaurado")
                    before = budget.used
                    row = play(ale, actions, word, history, budget)
                    rows.append(row)
                    emit({"event": "common_program_executed", "root": root, "episode": episode, "program_index": j,
                          "word": word.tolist(), "rewards_by_block": row[0].tolist(), "valid": row[1].tolist(),
                          "ale_calls": budget.used - before, "final_frame_sha256": row[4],
                          "initial_history_sha256": digest(initial), "raw_initial_history_sha256": raw_hash})
                    if j == 0 and point == CHECKPOINTS[0]:
                        ale.restoreSystemState(state)
                        replay = play(ale, actions, word, history, budget)
                        if any(not np.array_equal(a, b) for a, b in zip(row, replay)):
                            raise RuntimeError("replay divergiu")
                        certified += 1
                        emit({"event": "full_program_replay_certified", "root": root, "ale_calls": replay[2]})
                histories.append(initial)
                raw_history_hashes.append(raw_hash)
                rewards.append(np.stack([r[0] for r in rows]))
                valid.append(np.stack([r[1] for r in rows]))
                lengths.append([r[2] for r in rows])
                endpoints.append(np.stack([r[3] for r in rows]))
                final_hashes.append([r[4] for r in rows])
                roots.append(root)
                episodes.append(episode)
                ale.restoreSystemState(state)
                snapshot("partial_collecting_not_evaluated")
            print(json.dumps({"game": game, "seed": seed, "episode": episode, "roots": len(histories), "ale_calls": budget.used}), flush=True)
        if walks != 8192 or budget.used != sum(e["ale_calls"] for e in events) or certified != 8 or len(histories) != 32:
            raise RuntimeError("coleta/contabilidade incompleta")
        np.savez_compressed(output.with_suffix(".npz"), histories=np.stack(histories), raw_history_sha256=np.asarray(raw_history_hashes),
                            rewards=np.stack(rewards), valid=np.stack(valid), lengths=np.asarray(lengths),
                            end_histories=np.stack(endpoints), final_frame_sha256=np.asarray(final_hashes),
                            programs=programs, roots=np.asarray(roots), episode=np.asarray(episodes))
        report = {"status": "crossed_real_programs_measured", "game": game, "seed": seed, "roots": 32,
                  "programs_per_root": len(programs), "certified_replays": certified, "ale_calls": budget.used,
                  "walk_calls": walks, "previous_attempt_ale_calls": prior_calls,
                  "ale_calls_including_previous_attempt": budget.used + prior_calls, "proposal_sha256": sha(PROPOSAL), "bank_sha256": sha(output.with_suffix(".npz")),
                  "events_sha256": sha(event_path), "cpu_seconds": time.process_time() - start_cpu,
                  "wall_seconds": time.perf_counter() - start_wall, "human_level_goal_achieved": False,
                  "fit_manifest_sha256_before_test_collection": sha(manifest_path(game)) if seed == TEST_SEED else None}
        save_json(output.with_suffix(".json"), report)
        snapshot("collection_finished_complete_bank_separate")
        return report
    except BaseException as exc:
        snapshot("interrupted_inconclusive")
        save_json(output.with_name(output.name + "_interrupcao.json"), {"status": "inconclusive", "error": repr(exc), "ale_calls": budget.used})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    parser.add_argument("seed", type=int, choices=(TRAIN_SEED, TEST_SEED))
    args = parser.parse_args()
    print(json.dumps(collect(args.game, args.seed), indent=2))
