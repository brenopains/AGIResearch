"""Reproduz snapshots ALE e certifica rótulos antes de salvar históricos."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from aquisicao import frozen_model, prepare_episode
from coleta import gray, make_ale
from desacordo_ativo import (ARM_BUDGET, EPISODES, POOL, SEGMENTS,
                              SEGMENT_STEPS, SHARED_BUDGET, candidate,
                              check_replay, verify)

HERE = Path(__file__).resolve().parent


def candidate_with_history(ale, actions, initial, initial_history, macro, budget):
    """Mesmo laço de candidate(), com cópia do histórico ao final."""
    ale.restoreSystemState(initial)
    history = deque((frame.copy() for frame in initial_history), maxlen=16)
    before = budget[0]
    for command in macro:
        for _ in range(SEGMENT_STEPS):
            if budget[0] <= 0:
                raise RuntimeError("orçamento comum esgotado")
            ale.act(actions[command])
            budget[0] -= 1
            history.append(gray(ale))
            if ale.game_over():
                return {"macro": macro, "terminal": True, "steps": before - budget[0]}
    from desacordo_ativo import features_from_history
    return {"macro": macro, "terminal": False, "steps": before - budget[0],
            "snapshot": ale.cloneSystemState(),
            "features": features_from_history(history),
            "history": np.stack(history).astype(np.uint8)}


def replay(game, seed=59):
    report_path = HERE / "resultados" / f"{game}_desacordo_ativo.json"
    event_path = report_path.with_suffix(".jsonl")
    original = json.loads(report_path.read_text(encoding="utf-8"))
    if original["seed"] != seed or original["complete_episodes"] != EPISODES:
        raise RuntimeError("relatório original incompatível")
    current = hashlib.sha256((HERE / "desacordo_ativo.py").read_bytes()).hexdigest()
    if current != original["source_sha256"]["desacordo_ativo.py"]:
        raise RuntimeError("fonte do piloto mudou; replay não é prova")
    if hashlib.sha256(event_path.read_bytes()).hexdigest() != original["events_sha256"]:
        raise RuntimeError("JSONL do piloto mudou")
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    decisive, reference_hash = frozen_model(game)
    if reference_hash != original["reference_bank_sha256"]:
        raise RuntimeError("banco do decisor mudou")
    ale = make_ale(game, seed)
    actions = list(ale.getMinimalActionSet()[:4])
    rng = np.random.default_rng(seed)
    shared = [SHARED_BUDGET]
    budgets = {"disagreement": [ARM_BUDGET], "random": [ARM_BUDGET]}
    histories, values, episodes, macros_out, origins = [], [], [], [], []
    seen_states = {}
    checked_events = 0
    replay_checked = False
    for episode in range(EPISODES):
        initial, initial_history = prepare_episode(ale, actions, rng, shared)
        pool_rng = np.random.default_rng(seed * 1000 + episode * 11 + 1)
        macros = [tuple(map(int, macro)) for macro in
                  pool_rng.choice(4, size=(POOL, SEGMENTS), replace=True)]
        seen = set()
        for i, macro in enumerate(macros):
            while macro in seen:
                macro = tuple(int(a) for a in pool_rng.integers(0, 4, SEGMENTS))
            seen.add(macro)
            macros[i] = macro
        items = [candidate_with_history(ale, actions, initial, initial_history,
                                        macro, shared) for macro in macros]
        valid = [item for item in items if not item["terminal"]]
        expected_row = original["episodes"][episode]
        if len(valid) != expected_row["valid"]:
            raise RuntimeError(f"episódio {episode}: número de candidatos divergiu")
        if not replay_checked:
            check_replay(ale, actions, valid[0], shared)
            replay_checked = True
        group = [event for event in events if event["episode"] == episode]
        if len(group) != 8:
            raise RuntimeError(f"episódio {episode}: número de eventos divergiu")
        for event in group:
            item = valid[event["pool_index"]]
            if tuple(event["macro"]) != item["macro"]:
                raise RuntimeError(f"episódio {episode}: macro divergiu")
            item["votes"], item["disagreement"] = event["votes"], event["disagreement"]
            actual = verify(ale, actions, item, decisive, budgets[event["arm"]])
            if (not np.array_equal(np.asarray(actual["values"]), np.asarray(event["values"]))
                    or actual["chosen"] != event["chosen"]
                    or actual["steps"] != event["steps"]):
                raise RuntimeError(f"episódio {episode}, sonda {event['probe']}: rótulo divergiu")
            checked_events += 1
            key = (episode, item["macro"])
            if key in seen_states:
                if not np.array_equal(values[seen_states[key]], np.asarray(actual["values"])):
                    raise RuntimeError("dois braços discordam no mesmo snapshot")
                origins[seen_states[key]].append(event["arm"])
            else:
                seen_states[key] = len(histories)
                histories.append(item["history"])
                values.append(np.asarray(actual["values"], np.float32))
                episodes.append(episode)
                macros_out.append(item["macro"])
                origins.append([event["arm"]])
        print(json.dumps({"game": game, "episode": episode,
                          "events_checked": checked_events, "unique_states": len(histories)}), flush=True)
    spent = {"shared": SHARED_BUDGET - shared[0],
             "disagreement": ARM_BUDGET - budgets["disagreement"][0],
             "random": ARM_BUDGET - budgets["random"][0]}
    if (spent["shared"] != original["shared_steps"]
            or spent["disagreement"] != original["active_steps"]
            or spent["random"] != original["random_steps"]):
        raise RuntimeError("contabilidade ALE divergiu")
    output = HERE / "resultados" / f"{game}_estados_ativos.npz"
    np.savez_compressed(output, histories=np.stack(histories),
                        values=np.stack(values), episode=np.asarray(episodes),
                        macros=np.asarray(macros_out),
                        source=np.array(str(event_path)))
    metadata = {"game": game, "seed": seed, "verified_events": checked_events,
                "unique_states": len(histories), "steps": spent,
                "bank_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "events_sha256": original["events_sha256"],
                "source_code_sha256": current,
                "arm_membership": origins}
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    print(json.dumps(replay(args.game), ensure_ascii=False, indent=2))
