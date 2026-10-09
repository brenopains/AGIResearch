"""Recontagem do diagnóstico por ajuste Cholesky e execução para a frente."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import json
import numpy as np
from diagnosticar_pool_funcoes import HERE, MANIFEST, RESULT, EVENTS, CONSTRUCTION, BODIES, GENERATORS
from piloto_funcoes_contraexemplos import (FunctionTransport, PairFunctions, GeneratedTransport,
    append_slots, data_for_episodes, load_bank, root_view, require, sha, save_new, verify_proposal)
from auditar_funcoes_contraexemplos import coefficients, forward, independent_functions
from auditar_semantica_prefixo import scores, equal_report


def scalar_fields(record, expected):
    require(set(record) == set(expected), "campos diferem")
    for key, value in expected.items():
        if isinstance(value, float):
            require(np.isclose(record[key], value, atol=1e-8, rtol=1e-7), f"número difere: {key}")
        else:
            require(record[key] == value, f"valor difere: {key}")


def reexecute(phi_grid, reward, matrices, bank, ids):
    rows = []
    for k, row in enumerate(ids):
        branches = []
        for side in (0, 1):
            phi, actions, actual_rewards, valid = phi_grid[k, side], bank["actions"][row, side], bank["rewards"][row, side], bank["valid"][row, side]
            predicted = float(forward(phi[:1], reward, matrices, [actions.tolist()], [[]])[0])
            functions = [np.zeros(len(reward[0]))]
            for action in actions[::-1]:
                functions.append(reward[action] + matrices[action] @ functions[-1])
            functions.reverse()
            readout = reward_error = transport_error = 0.
            for t, action in enumerate(actions):
                if not valid[t]:
                    continue
                r = float(phi[t] @ reward[action])
                re = r - float(actual_rewards[t])
                # Adjacent scalar suffix predictions instead of vector residual dot.
                te = float(phi[t] @ functions[t]) - r - float(valid[t + 1]) * float(phi[t + 1] @ functions[t + 1])
                readout += r
                reward_error += re
                transport_error += te
            branches.append({"actual": float(actual_rewards.sum()), "predicted": predicted,
                "reward_error": reward_error, "transport_error": transport_error,
                "boundary_error": 0. if valid[0] else predicted, "real_readout": readout})
        rows.append({"bank_row": int(row), "episode": int(bank["episode"][row]),
            **{field: branches[0][field] - branches[1][field] for field in branches[0]}})
    return rows


def audit():
    verify_proposal()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        require(sha(HERE / path) == expected, f"fonte congelada mudou: {path}")
    audit_manifest = HERE / "resultados" / "pool_funcoes_auditoria_manifesto.json"
    snapshot = {"scope": "post_diagnostic_independent_recount_no_new_gate", "source_hashes":
        {name: sha(HERE / name) for name in ("auditar_pool_funcoes.py", "auditar_funcoes_contraexemplos.py",
         "auditar_semantica_prefixo.py", MANIFEST.relative_to(HERE).as_posix(), RESULT.relative_to(HERE).as_posix(), EVENTS.relative_to(HERE).as_posix())}}
    if audit_manifest.exists():
        require(json.loads(audit_manifest.read_text(encoding="utf-8")) == snapshot, "auditor mudou; preservar")
    else:
        save_new(audit_manifest, snapshot)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(result["events_sha256"] == sha(EVENTS) and result["manifest_sha256"] == sha(MANIFEST), "hash divergente")
    require(result["fit_calls"] == result["events"] == len(records) == 256, "pool incompleto")
    require(result["previous_gate_preserved"] == "failed_development_gate" and not result["human_level_goal_achieved"] and result["ale_calls"] == 0, "escopo mudou")
    construction = json.loads(CONSTRUCTION.read_text(encoding="utf-8"))["construction"]
    bodies = json.loads(BODIES.read_text(encoding="utf-8"))["bodies"]
    maximum, checked = 0., 0
    for game in ("breakout", "pong"):
        bank = load_bank(game, 97)
        data = data_for_episodes(bank, [0, 1, 2, 3])
        validation = data_for_episodes(bank, [4, 5])
        view = root_view(bank, [4, 5])
        ids = view["ids"]
        base = FunctionTransport.load(HERE / construction[game]["base"])
        baseline = GeneratedTransport.load(HERE / bodies[game]["no_generated_functions"])
        width = len(base.initial_prior)
        raw_grid = bank["histories"][ids]
        shape = raw_grid.shape
        flat = raw_grid.reshape(-1, *shape[3:])
        p, pn = base.features(data["source"]), base.features(data["successor"])
        vp, vpn, gb = base.features(validation["source"]), base.features(validation["successor"]), base.features(flat)
        baseline_phi = append_slots(gb, np.empty((len(gb), 0)), []).reshape(*shape[:3], -1)
        baseline_rows = reexecute(baseline_phi, baseline.reward, baseline.transport, bank, ids)
        for recorded, actual in zip(result["games"][game]["baseline"]["rows"], baseline_rows):
            scalar_fields(recorded, actual)
        y = np.asarray(view["actual"], float)
        bp = np.asarray([row["predicted"] for row in baseline_rows])
        def costs(prediction):
            return np.asarray([abs(a) / 2 if abs(b) <= 1e-12 else max(-a, 0) if b >= -1e-12 else max(a, 0) for a, b in zip(y, prediction)])
        baseline_cost = costs(bp)
        for arm in GENERATORS:
            functions = PairFunctions.load(HERE / construction[game]["dictionaries"][arm])
            group = [r for r in records if r["game"] == game and r["arm"] == arm]
            require([r["candidate"] for r in group] == list(range(32)), "candidatos omitidos/duplicados")
            g, gn = functions.apply(data["source"]), functions.apply(data["successor"])
            vg, vgn, grid_g = functions.apply(validation["source"]), functions.apply(validation["successor"]), functions.apply(flat)
            require(np.allclose(grid_g, independent_functions(functions, flat, data["source"]), rtol=1e-9, atol=1e-8), "corpo sensorial difere")
            all_costs = [baseline_cost]
            for index, record in enumerate(group):
                reward, matrices = coefficients(append_slots(p, g, [index]), append_slots(pn, gn, [index]), data, .05)
                phi = append_slots(vp, vg, [index])
                nphi = append_slots(vpn, vgn, [index])
                rp = np.asarray([float(row @ reward[action]) for row, action in zip(phi, validation["actions"])])
                residual = np.asarray([row @ matrices[action] - c * nr for row, nr, action, c in
                    zip(phi, nphi, validation["actions"], validation["continuation"])])
                local = {"observed_transitions": len(rp), "nonzero_reward_transitions": int(np.count_nonzero(validation["reward"])),
                    "reward_mse": float(np.mean((rp - validation["reward"]) ** 2)),
                    "old_coordinate_transport_mse": float(np.mean(residual[:, :width] ** 2)),
                    "active_new_coordinate_transport_mse": float(np.mean(residual[:, width] ** 2))}
                scalar_fields(record["local_error"], local)
                grid = append_slots(gb, grid_g, [index]).reshape(*shape[:3], -1)
                rows = reexecute(grid, reward, matrices, bank, ids)
                for recorded, actual in zip(record["rows"], rows):
                    scalar_fields(recorded, actual)
                    for field in ("predicted", "reward_error", "transport_error", "boundary_error", "real_readout"):
                        maximum = max(maximum, abs(recorded[field] - actual[field]))
                prediction = np.asarray([r["predicted"] for r in rows])
                all_costs.append(costs(prediction))
                q = record["queries"]
                equal_report(q["composed"], scores(y, prediction), "composição")
                equal_report(q["real_future_history_readout_oracle"], scores(y, [r["real_readout"] for r in rows]), "leitura real")
                equal_report(q["remove_reward_error_oracle"], scores(y, [r["predicted"] - r["reward_error"] for r in rows]), "oráculo")
                re, te, be = [np.asarray([r[field] for r in rows]) for field in ("reward_error", "transport_error", "boundary_error")]
                terms = {"reward_squared": float(np.mean(re * re)), "transport_squared": float(np.mean(te * te)),
                    "boundary_squared": float(np.mean(be * be)), "reward_transport_cross": float(np.mean(2 * re * te)),
                    "reward_boundary_cross": float(np.mean(2 * re * be)), "transport_boundary_cross": float(np.mean(2 * te * be))}
                scalar_fields(q["mse_decomposition"], terms)
                stored = result["games"][game]["generators"][arm]["candidates"][index]
                require(stored["queries"] == record["queries"] and stored["local_error"] == record["local_error"] and stored["expression"] == functions.expressions[index], "JSON/evento/AST divergentes")
                checked += 1
            all_costs = np.asarray(all_costs)
            improved = all_costs[1:] < baseline_cost - 1e-12
            expected = {"baseline_error_positions": np.flatnonzero(baseline_cost > 1e-12).tolist(),
                "root_improved_by_any_candidate": np.flatnonzero(improved.any(0)).tolist(),
                "root_not_improved_by_any_candidate": np.flatnonzero((baseline_cost > 1e-12) & ~improved.any(0)).tolist(),
                "maximum_roots_improved_by_one_candidate": int(improved.sum(1).max()),
                "baseline_fair_regret": float(baseline_cost.mean()),
                "answer_conditioned_pool_oracle_fair_regret": float(all_costs.min(0).mean()),
                "oracle_is_not_an_executable_decision_rule": True}
            scalar_fields(result["games"][game]["generators"][arm]["decision_opportunity"], expected)
        del bank
    output = {"status": "audit_passed", "candidate_bodies_recounted": checked,
        "paired_predictions_recounted": checked * 8, "maximum_scalar_discrepancy": maximum,
        "manifest_sha256": sha(audit_manifest), "result_sha256": sha(RESULT),
        "previous_gate_preserved": "failed_development_gate", "human_level_goal_achieved": False}
    save_new(HERE / "resultados" / "pool_funcoes_auditoria.json", output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
