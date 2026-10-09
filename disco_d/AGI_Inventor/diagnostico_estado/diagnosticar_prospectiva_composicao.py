"""Separar leitura e transporte nos corpos congelados; oráculos não são agentes."""
from piloto_prospectivo import (np, json, HERE, PROPOSAL, BODIES, RESULT as PRIOR_RESULT,
    GeneratedTransport, bank_path, load_bank, require, save_new, sha, verify_bodies)
from decompor_erros_transporte import trajectory
from auditar_funcoes_contraexemplos import forward
from diagnosticar_semantica_prefixo import metrics
import argparse
import time

STEM = HERE / "resultados" / "prospectiva_composicao"
MANIFEST = STEM.with_name(STEM.name + "_manifesto.json")
RESULT, EVENTS = STEM.with_suffix(".json"), STEM.with_suffix(".jsonl")
ARMS = ("future_pairs", "no_generated_functions", "random_directions")


def prepare():
    verify_bodies()
    require(not any(p.exists() for p in (MANIFEST, RESULT, EVENTS)), "diagnóstico existe; preservar")
    paths = ["diagnosticar_prospectiva_composicao.py", "decompor_erros_transporte.py", "auditar_funcoes_contraexemplos.py",
        "resultados/prospectiva_auditoria.json", PROPOSAL.relative_to(HERE).as_posix(),
        BODIES.relative_to(HERE).as_posix(), PRIOR_RESULT.relative_to(HERE).as_posix()]
    for game in ("breakout", "pong"):
        for suffix in (".npz", ".json", ".jsonl"):
            paths.append(bank_path(game, 157).with_suffix(suffix).relative_to(HERE).as_posix())
    save_new(MANIFEST, {"status": "frozen_post_evaluation_diagnostic_not_confirmation", "arms": list(ARMS),
        "horizons": [4, 8, 16], "all_roots": True, "no_fit": True, "no_ale": True,
        "maximum_wall_seconds": 120, "previous_gate_unchanged": True,
        "source_hashes": {p: sha(HERE / p) for p in paths}})
    return {"status": "diagnostic_frozen", "manifest_sha256": sha(MANIFEST)}


def run():
    frozen = verify_bodies()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        require(sha(HERE / path) == expected, "fonte/dado/corpo mudou")
    require(not RESULT.exists() and not EVENTS.exists(), "diagnóstico já calculado")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    start = time.perf_counter()
    reports, records, maximum, replay_maximum = {}, [], 0., 0.
    for game in ("breakout", "pong"):
        bank = load_bank(game, 157)
        shape = bank["histories"].shape
        reports[game] = {}
        for arm in ARMS:
            require(time.perf_counter() - start < 120, "orçamento esgotado")
            body = GeneratedTransport.load(HERE / frozen["bodies"][game][arm])
            phi = body.features(bank["histories"].reshape(-1, *shape[3:])).reshape(*shape[:3], -1)
            reports[game][arm] = {}
            for partition, offset in (("known", 0), ("new", 4)):
                curves = {}
                for horizon in (4, 8, 16):
                    rows = []
                    for i in range(32):
                        for p in (offset, offset + 2):
                            branches = []
                            for side in (p, p + 1):
                                word = bank["programs"][side, :horizon]
                                observed_phi = phi[i, side, :horizon + 1]
                                reward, valid = bank["rewards"][i, side, :horizon], bank["valid"][i, side, :horizon + 1]
                                branch = trajectory(body, observed_phi, word, reward, valid)
                                maximum = max(maximum, branch["identity_error"])
                                reconstructed = float(forward(observed_phi[:1], body.reward, body.transport, [word.tolist()], [[]])[0])
                                actual_readout = sum(float(observed_phi[t] @ body.reward[a]) for t, a in enumerate(word) if valid[t])
                                error = max(abs(branch["predicted"] - reconstructed), abs(branch["real_readout"] - actual_readout))
                                replay_maximum = max(replay_maximum, error)
                                require(error < 1e-8, "reexecução independente diverge")
                                branches.append(branch)
                            delta = {field: branches[0][field] - branches[1][field] for field in
                                ("actual", "predicted", "reward_error", "transport_error", "boundary_error", "real_readout")}
                            row = {"event": "prospective_paired_error_attribution", "game": game, "arm": arm, "partition": partition,
                                "horizon": horizon, "bank_row": i, "left": p, "right": p + 1, "episode": int(bank["episode"][i]), **delta}
                            rows.append(row)
                            records.append(row)
                    actual = np.asarray([r["actual"] for r in rows])
                    reward_error, transport_error, boundary_error = [np.asarray([r[field] for r in rows]) for field in
                        ("reward_error", "transport_error", "boundary_error")]
                    queries = {"composed": metrics(actual, [r["predicted"] for r in rows]),
                        "real_future_histories_oracle": metrics(actual, [r["real_readout"] for r in rows]),
                        "remove_reward_error_oracle": metrics(actual, [r["predicted"] - r["reward_error"] for r in rows])}
                    terms = {"reward_squared": float(np.mean(reward_error ** 2)), "transport_squared": float(np.mean(transport_error ** 2)),
                        "boundary_squared": float(np.mean(boundary_error ** 2)), "reward_transport_cross": float(np.mean(2 * reward_error * transport_error)),
                        "reward_boundary_cross": float(np.mean(2 * reward_error * boundary_error)),
                        "transport_boundary_cross": float(np.mean(2 * transport_error * boundary_error))}
                    require(abs(sum(terms.values()) - queries["composed"]["mse"]) < 1e-8, "MSE não reconcilia")
                    for metric in ("mse", "fair_tie_regret"):
                        require(np.isclose(queries["composed"][metric], prior["games"][game][partition]["curves"][str(horizon)][arm][metric], atol=1e-10), "previsão anterior mudou")
                    curves[str(horizon)] = {"queries": queries, "mse_decomposition": terms}
                reports[game][arm][partition] = curves
        del bank
    with EVENTS.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    output = {"status": "post_evaluation_diagnostic_complete_no_promotion", "games": reports, "event_count": len(records),
        "manifest_sha256": sha(MANIFEST), "events_sha256": sha(EVENTS), "maximum_identity_error": maximum,
        "maximum_independent_forward_readout_error": replay_maximum, "wall_seconds": time.perf_counter() - start,
        "fit_calls": 0, "ale_calls": 0, "previous_gate_preserved": prior["status"], "human_level_goal_achieved": False}
    require(output["wall_seconds"] < 120, "orçamento esgotado")
    save_new(RESULT, output)
    return {"status": output["status"], "wall_seconds": output["wall_seconds"], "events": len(records),
        "maximum_identity_error": maximum, "maximum_independent_forward_readout_error": replay_maximum,
        "primary16": {g: {p: r["future_pairs"][p]["16"] for p in ("known", "new")} for g, r in reports.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    mode = parser.parse_args().mode
    print(json.dumps(prepare() if mode == "prepare" else run(), ensure_ascii=False, indent=2))
