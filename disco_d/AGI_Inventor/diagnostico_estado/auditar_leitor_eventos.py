"""Auditoria independente: equações normais, grafo, matriz aumentada e chamadas físicas."""
import os
import platform
import sys
if sys.platform == "win32":
    platform.machine = platform._get_machine_win32
    platform.system = lambda: "Windows"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
import hashlib
import json
from collections import deque
import numpy as np
from piloto_leitor_eventos import (HERE, STEM, PROPOSAL, BODIES, DIAGNOSTIC, RESULT, EVENTS,
    BASES, READERS, ARMS, GeneratedTransport, EventReader, load_bank, load_fresh, fresh_bank,
    transitions, view, verify_frozen, require, save_new, sha)
from auditar_semantica_prefixo import scores, equal_report
from auditar_funcoes_contraexemplos import forward


def solve(x, y, alpha=.05):
    factor = np.linalg.cholesky(x.T @ x + alpha * np.eye(x.shape[1]))
    return np.linalg.solve(factor.T, np.linalg.solve(factor, x.T @ y))


def independent_design(p, pn, mode):
    out = np.zeros((len(p), 2 * p.shape[1]))
    d = p.shape[1]
    if mode == "before":
        out[:, :d] = p
    elif mode == "after":
        out[:, :d] = pn
    elif mode == "delta":
        out[:, 0] = 1
        out[:, 1:d] = pn[:, 1:] - p[:, 1:]
    elif mode == "potential":
        out[:, :d] = pn - p
    elif mode == "transition":
        out[:, :d], out[:, d:] = p, pn
    elif mode == "action_prior":
        out[:, 0] = 1
    else:
        require(mode == "zero", "leitor desconhecido")
    return out


def permute(value, actions):
    result = value.copy()
    rng = np.random.default_rng(10311)
    for action in range(4):
        ids = np.flatnonzero(actions == action)
        result[ids] = value[rng.permutation(ids)]
    return result


def independent_graph(data):
    adjacency, conflict = {}, []
    node_groups = [[hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in data[key]]
        for key in ("source", "successor")]
    for i, (u, v, reward) in enumerate(zip(*node_groups, data["reward"])):
        require(float(reward).is_integer(), "recompensa não inteira")
        adjacency.setdefault(u, [])
        adjacency.setdefault(v, [])
        values = {u: 0}
        queue = deque([u])
        while queue and v not in values:
            at = queue.popleft()
            for other, delta in adjacency[at]:
                if other not in values:
                    values[other] = values[at] + delta
                    queue.append(other)
        if v in values:
            residual = values[v] - int(reward)
            if residual:
                conflict.append({"edge": i, "constraint_residual": residual})
        else:
            adjacency[u].append((v, int(reward)))
            adjacency[v].append((u, -int(reward)))
    remaining, components = set(adjacency), 0
    while remaining:
        components += 1
        queue = deque([remaining.pop()])
        while queue:
            for other, _ in adjacency[queue.popleft()]:
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
    return {"observed_nodes": len(adjacency), "connected_components": components,
        "conflict_count": len(conflict), "first_conflicts": conflict[:8], "compatible_on_observed_graph": not conflict,
        "verification_scope": "exact_integer_constraints_on_observed_history_graph", "universal_conservativity_proven": False}


def local_scores(y, p):
    errors = [(float(a) - float(b)) ** 2 for a, b in zip(y, p)]
    nonzero = [e for a, e in zip(y, errors) if a != 0]
    return {"count": len(y), "mse": float(np.mean(errors)), "mae": float(np.mean([abs(float(a)-float(b)) for a,b in zip(y,p)])),
        "observed_nonzero": int(np.count_nonzero(y)), "nonzero_mse": float(np.mean(nonzero)) if nonzero else None,
        "rounded_match_fraction_not_exact_proof": float(np.mean([round(float(b)) == a for a,b in zip(y,p)]))}


def matrix_values(phi, words, alive, terminal, field):
    d = len(field)
    augmented = np.zeros((len(alive), 2*d, 2*d))
    augmented[:, :d, :d] = alive
    augmented[:, :d, d:] = terminal
    augmented[:, d:, d:] = np.eye(d)
    objective = np.tile(field, 2)
    # Backward products, whereas the runner transports alive/dead states forward.
    cache, result = {}, []
    for z, word in zip(phi, words):
        key = tuple(word)
        if key not in cache:
            vector = objective.copy()
            for action in reversed(key):
                vector = augmented[action] @ vector
            cache[key] = vector[:d] - field
        result.append(float(z @ cache[key]))
    return np.asarray(result)


def physical_audit(game):
    bank = load_fresh(game)
    stem = fresh_bank(game)
    certificate = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in stem.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    require(sum(r["ale_calls"] for r in events) == certificate["ale_calls"] <= 50000, "chamadas físicas divergem")
    require(certificate["wall_seconds"] < 120 and certificate["maximum_sampled_rss_bytes"] < 6 * 1024 ** 3, "teto físico excedido")
    require(sum(e["event"] == "walk_checkpoint" for e in events) == 64, "caminhada incompleta")
    require(sum(e["event"] == "full_program_replay_certified" for e in events) == certificate["certified_replays"] == 8, "replay ausente")
    rows = [e for e in events if e["event"] == "common_program_measured"]
    require(len(events) == 64 + 8 + len(rows) and len(rows) == 256, "contagem/espécie de eventos diverge")
    for i, root in enumerate(bank["roots"]):
        group = [r for r in rows if r["root"] == int(root)]
        require([r["program"] for r in group] == list(range(8)), "programas duplicados/omitidos")
        for j, row in enumerate(group):
            require(np.array_equal(row["word"], bank["programs"][j]) and np.array_equal(row["rewards"], bank["rewards"][i,j])
                and np.array_equal(row["valid"], bank["valid"][i,j]), "banco/evento diverge")
            require(row["episode"] == int(bank["episode"][i]), "episódio diverge")
        with np.load(stem.with_name(stem.name + f"_root_{root:02d}.npz"), allow_pickle=False) as chunk:
            require(np.array_equal(chunk["histories"], bank["histories"][i]) and np.array_equal(chunk["rewards"], bank["rewards"][i])
                and np.array_equal(chunk["valid"], bank["valid"][i]) and np.array_equal(chunk["programs"], bank["programs"]), "checkpoint diverge")
    return bank, certificate["ale_calls"]


def audit():
    frozen = verify_frozen()
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    held = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    require(diagnostic["frozen_bodies_sha256"] == result["bodies_sha256"] == sha(BODIES)
        and result["proposal_sha256"] == sha(PROPOSAL) and result["events_sha256"] == sha(EVENTS), "hash do relatório diverge")
    require(len(held) == result["event_count"] == 9216, "comparações finais incompletas")
    require(not result["human_level_goal_achieved"] and result["no_future_observations_used_at_decision"]
        and result["fit_calls_during_evaluation"] == 0 and result["primary_basis"] == "no_generated_functions"
        and result["secondary_basis_not_promoted_to_primary"], "escopo exagerado")
    max_coefficient, max_prediction, evaluated, physical, states, graph_reports = 0., 0., 0, 0, [], {}
    for game in ("breakout", "pong"):
        train = transitions(load_bank(game, 149))
        graph_reports[game] = independent_graph(train)
        equal_report(graph_reports[game], frozen["construction"][game]["observed_integer_graph"], "grafo independente")
        fresh, calls = physical_audit(game)
        physical += calls
        for basis in BASES:
            files = frozen["bodies"][game][basis]
            body = GeneratedTransport.load(HERE / files["feature"])
            p, pn = body.features(train["source"]), body.features(train["successor"])
            coefficients = {}
            for arm in READERS:
                reader = EventReader.load(HERE / files["readers"][arm])
                mode = "transition" if arm in ("shuffled_successor", "shuffled_rewards") else arm
                xx = independent_design(p, permute(pn, train["actions"]) if arm == "shuffled_successor" else pn, mode)
                yy = permute(train["reward"], train["actions"]) if arm == "shuffled_rewards" else train["reward"]
                coefficient = np.zeros((4, xx.shape[1]))
                if mode == "potential":
                    coefficient[:, :p.shape[1]] = solve(xx[:, :p.shape[1]], yy)
                    require(np.allclose(reader.field, coefficient[0, :p.shape[1]], atol=1e-9, rtol=1e-9), "campo não reproduz")
                elif mode != "zero":
                    for action in range(4):
                        ids = train["actions"] == action
                        coefficient[action] = solve(xx[ids], yy[ids])
                error = float(np.max(np.abs(reader.coefficient - coefficient)))
                max_coefficient = max(max_coefficient, error)
                require(error < 1e-8, "equação normal não reproduz")
                coefficients[arm] = coefficient
                definition = json.loads((HERE / files["readers"][arm]).with_suffix(".json").read_text(encoding="utf-8"))
                require(definition["feature_reference"]["sha256"] == sha(HERE / files["feature"]) and definition["mode"] == mode, "definição executável diverge")
            for seed in (151, 157):
                bank = load_bank(game, seed)
                data = transitions(bank)
                bp, bn = body.features(data["source"]), body.features(data["successor"])
                for arm in READERS:
                    mode = "transition" if arm in ("shuffled_successor", "shuffled_rewards") else arm
                    x = independent_design(bp, bn, mode)
                    predicted = np.array([row @ coefficients[arm][a] for row,a in zip(x,data["actions"])])
                    report = diagnostic["games"][game][str(seed)][basis][arm]
                    equal_report(local_scores(data["reward"], predicted), report["local"], "leitura local")
                    require(report["available_before_action"] == (arm in ("before", "action_prior", "zero")), "timing errado")
                    for episode in range(8):
                        ids = data["episodes"] == episode
                        equal_report(local_scores(data["reward"][ids], predicted[ids]), report["per_episode"][str(episode)], "episódio")
                    reward_matrix = np.zeros_like(bank["rewards"], float)
                    for index, prediction in zip(data["bank_indices"], predicted):
                        reward_matrix[tuple(index)] = prediction
                    for partition, offset in (("known",0),("new",4)) if seed == 157 else (("known",0),):
                        cases = view(bank, offset)
                        values = np.array([sum(reward_matrix[i,a])-sum(reward_matrix[i,b]) for i,a,b in cases["bank_cases"]])
                        equal_report(scores(cases["actual"], values), report["whole_program_with_real_observations_diagnostic"][partition], "reconhecimento por programa")
            terminal = []
            for action in range(4):
                ids = train["actions"] == action
                terminal.append(solve(p[ids], (1-train["continuation"][ids,None])*pn[ids]))
            terminal = np.asarray(terminal)
            field = coefficients["potential"][0, :p.shape[1]]
            shuffled_field = solve(pn-p, permute(train["reward"], train["actions"]))
            with np.load(HERE / files["planner"], allow_pickle=False) as params:
                require(np.allclose(params["field"], field, atol=1e-9) and np.allclose(params["shuffled_field"], shuffled_field, atol=1e-9)
                    and np.allclose(params["terminal_transport"], terminal, atol=1e-9), "parâmetros de planejamento divergem")
            solution, _, rank, _ = np.linalg.lstsq(pn-p, train["reward"], rcond=1e-10)
            residual = (pn-p) @ solution-train["reward"]
            report = frozen["construction"][game]["bases"][basis]
            require(int(rank) == report["least_squares_difference_rank"] and np.isclose(float(np.mean(residual**2)), report["minimum_training_mse_with_numeric_rank_cutoff"]), "resíduo da família diverge")
            for partition, offset in (("known",0),("new",4)):
                cell = result["games"][game][basis][partition]
                for horizon in (4,8,16):
                    cases = view(fresh,offset,horizon)
                    phi = body.features(cases["history"])
                    for arm in ARMS:
                        if arm == "zero":
                            predicted = np.zeros(64)
                        elif arm == "old_composed":
                            predicted = forward(phi, body.reward, body.transport, cases["left"], cases["right"])
                        else:
                            ff = shuffled_field if arm == "shuffled_field" else field
                            tt = np.zeros_like(terminal) if arm == "no_terminal_transport" else terminal
                            zz = np.broadcast_to(body.initial_prior, phi.shape) if arm == "no_context" else phi
                            predicted = matrix_values(zz,cases["left"],body.transport,tt,ff)-matrix_values(zz,cases["right"],body.transport,tt,ff)
                        group = [r for r in held if r["game"] == game and r["basis"] == basis and r["partition"] == partition
                            and r["horizon"] == horizon and r["arm"] == arm]
                        require(len(group) == 64, "eventos duplicados/omitidos")
                        for i, (row,a,b) in enumerate(cases["bank_cases"]):
                            require((group[i]["bank_row"],group[i]["left"],group[i]["right"]) == (row,a,b)
                                and group[i]["actual_delta"] == float(cases["actual"][i]), "identidade do caso diverge")
                        error = float(np.max(np.abs(predicted-np.array([r["predicted_delta"] for r in group]))))
                        max_prediction = max(max_prediction,error)
                        require(error < 1e-8, "matriz aumentada não reproduz")
                        equal_report(scores(cases["actual"],predicted),cell["curves"][str(horizon)][arm],"planejamento")
                        evaluated += 64
                values = cases["actual"].reshape(32,2)
                active = sorted(map(int,set(cases["episodes"][np.abs(cases["actual"]) > 1e-12])))
                opposite = [bool(np.any(values[:,i]<-1e-12) and np.any(values[:,i]>1e-12)) for i in range(2)]
                coverage = {"opposite_preferences_per_pair":opposite,"episodes_with_contrast":active,
                    "non_tied_comparisons":int(np.count_nonzero(cases["actual"])),"sufficient":bool(all(opposite) and len(active)>=3)}
                require(coverage == cell["coverage"], "cobertura diverge")
                checks = {"contextual_coverage": coverage["sufficient"]}
                primary = cell["curves"]["16"]["potential"]
                for arm in ARMS:
                    if arm != "potential":
                        for metric in ("mse","fair_tie_regret"):
                            checks[f"{metric}_vs_{arm}"] = bool(primary[metric] < cell["curves"]["16"][arm][metric]-1e-12)
                status = "inconclusive_coverage" if not coverage["sufficient"] else "passed" if all(checks.values()) else "failed"
                require(checks == cell["gate_checks"] and status == cell["status"], "portão diverge")
                if basis == "no_generated_functions":
                    states.append(status)
    verdict = "event_field_gate_passed_not_agi" if all(s == "passed" for s in states) else "event_field_inconclusive_coverage" if "inconclusive_coverage" in states else "event_field_gate_failed"
    require(verdict == result["status"], "veredito agregado diverge")
    output = {"status":"event_reader_audit_passed","readout_bodies_checked":36,"held_comparisons_checked":evaluated,
        "maximum_coefficient_error":max_coefficient,"maximum_augmented_matrix_prediction_error":max_prediction,
        "fresh_ale_calls":physical,"certified_physical_replays":16,"integer_graphs":graph_reports,
        "result_sha256":sha(RESULT),"diagnostic_sha256":sha(DIAGNOSTIC),"human_level_goal_achieved":False}
    save_new(STEM.with_name(STEM.name+"_auditoria.json"),output)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
