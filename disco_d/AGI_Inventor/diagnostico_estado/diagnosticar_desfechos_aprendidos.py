"""Localizar erro local e de composição; futuros reais somente no diagnóstico."""

import argparse
import json

import numpy as np

from avaliar import KernelRidge
from avaliar_estado_preditivo import encode, load
from coordenadas_consequencias import ConsequenceReadout
from desfechos_aprendidos import OutcomeModel, OutcomePlanner
from futuros_distribuidos import make_edges, weights
from piloto_desfechos_aprendidos import HERE, POLICIES, proposal_path, stem
from piloto_planos_propostos import sha, proposal_path as old_proposals


def run(game):
    proposal = json.loads(proposal_path(game).read_text(encoding="utf-8"))
    for p, expected in proposal["source_hashes"].items():
        if sha(HERE / p) != expected:
            raise RuntimeError(f"arquivo congelado mudou: {p}")
    models = {name: OutcomeModel.load(stem(game).with_name(stem(game).name + f"_{name}.npz"))
              for name in POLICIES}
    old = json.loads(old_proposals(game).read_text(encoding="utf-8"))
    path = HERE / "resultados" / f"{game}_oraculo_planos_propostos_103.npz"
    bank = np.load(path)
    basis = np.load(HERE / "resultados" / f"{game}_base_estado_preditivo.npz")
    real_z = encode(bank["signatures"].reshape(32, 2, 17, 4, 336), basis["mean"], basis["basis"])
    _, _, signature, valid, _ = load(game, 97)
    z = encode(signature, basis["mean"], basis["basis"])
    reward = np.load(HERE / "resultados" / f"{game}_horizontes_97.npz")["rewards"]
    edges = make_edges(z, valid, reward)
    credit = ConsequenceReadout().fit(edges)
    rows, cases = [], []
    for j, policy in enumerate(("distributed_search", "reward_search")):
        for k, root in enumerate(bank["roots"]):
            plan = old["policies"][policy]["plans"][int(root)]
            for t, action in enumerate(plan):
                if bank["valid"][k, j, t]:
                    cases.append((int(bank["episode"][k]), action, real_z[k, j, t], real_z[k, j, t + 1],
                                  bank["rewards"][k, j, t], float(bank["valid"][k, j, t + 1])))
    for action in range(4):
        selected = [c for c in cases if c[1] == action]
        if not selected:
            continue
        current, successor = np.stack([c[2] for c in selected]), np.stack([c[3] for c in selected])
        actual_r, actual_c = np.asarray([c[4] for c in selected]), np.asarray([c[5] for c in selected])
        episode = np.asarray([c[0] for c in selected])
        surviving = actual_c > 0
        categories = models["conditional_outcomes"].outcomes[action]
        represented = np.asarray([np.any((categories["reward"] == r) & (categories["continuation"] == c))
                                  for r, c in zip(actual_r, actual_c)])
        forecasts = {}
        for name, model in models.items():
            q = model.posterior(current, action)
            o = model.outcomes[action]
            alive = q * o["continuation"]
            mass = alive.sum(1, keepdims=True)
            predicted_state = (alive @ o["successor"]) / np.maximum(mass, 1e-12)
            predicted_state[mass[:, 0] < 1e-12] = current[mass[:, 0] < 1e-12]
            forecasts[name] = (q @ o["reward"], q @ o["continuation"], predicted_state)
        e = edges[action]
        q = weights(current, e["source"])
        alive = q * e["continuation"]
        mass = alive.sum(1, keepdims=True)
        next_state = alive @ e["successor"] / np.maximum(mass, 1e-12)
        next_state[mass[:, 0] < 1e-12] = current[mass[:, 0] < 1e-12]
        forecasts["source_neighbors"] = (q @ e["reward"], q @ e["continuation"], next_state)
        ridge = credit.predict_action(current, action)
        forecasts["ridge_local_credit"] = (ridge[:, 0], ridge[:, 1], None)
        scores = {}
        for name, (r, c, state) in forecasts.items():
            errors = (r - actual_r) ** 2
            scores[name] = {"reward_mse": float(errors.mean()), "reward_bias": float((r - actual_r).mean()),
                            "continuation_brier": float(np.mean((c - actual_c) ** 2)),
                            "surviving_state_mse": float(np.mean((state[surviving] - successor[surviving]) ** 2))
                            if state is not None and surviving.any() else None,
                            "per_episode_reward_mse": {str(ep): float(errors[episode == ep].mean())
                                                       for ep in sorted(set(episode))}}
        target_support = e["successor"][e["continuation"] > 0]
        category_support = categories["successor"][categories["continuation"] > 0]
        # Future-dependent bounds are support diagnostics, not an agent's prediction.
        oracle = {}
        if surviving.any():
            for name, support in (("all_successors", target_support), ("category_successors", category_support)):
                oracle[name] = float(KernelRidge.distances(successor[surviving], support).min(1).mean() / successor.shape[1])
        rows.append({"action": action, "cases": len(selected), "surviving_cases": int(surviving.sum()),
                     "unrepresented_reward_termination_events": int((~represented).sum()),
                     "scores": scores, "future_dependent_support_mse": oracle})
    trajectories = {}
    for j, policy in enumerate(("distributed_search", "reward_search")):
        plans = [old["policies"][policy]["plans"][int(i)] for i in bank["roots"]]
        observed = bank["rewards"][:, j].sum(1)
        results = {}
        for name, model in models.items():
            local = np.zeros(32)
            for k, plan in enumerate(plans):
                for t, action in enumerate(plan):
                    if bank["valid"][k, j, t]:
                        query = real_z[k:k + 1, j, t]
                        local[k] += float((model.posterior(query, action) @ model.outcomes[action]["reward"])[0])
            composed = np.diag(OutcomePlanner(model).plan_values(real_z[:, j, 0], plans))
            results[name] = {
                "real_trajectory_local_credit": {"mse": float(np.mean((local - observed) ** 2)),
                                                  "bias": float((local - observed).mean())},
                "real_initial_composed": {"mse": float(np.mean((composed - observed) ** 2)),
                                          "bias": float((composed - observed).mean())}}
        trajectories[policy] = {"cases": 32, "observed_mean": float(observed.mean()), "models": results}
    cycles = {}
    for name, model in models.items():
        planner = OutcomePlanner(model)
        cycles[name] = []
        for action, o in enumerate(model.outcomes):
            self_mass = np.diag(planner.transition(action, action)) * o["continuation"]
            cycles[name].append({"action": action, "max_reward_positive_self_mass":
                                 float(self_mass[o["reward"] > 0].max()) if np.any(o["reward"] > 0) else None,
                                 "mean_reward": float(model.priors[action] @ o["reward"])})
    report = {"status": "post_pilot_diagnostic_not_confirmation", "game": game, "train_seed": 97,
              "test_seed": 103, "new_ale_steps": 0, "cases": len(cases), "by_action": rows,
              "real_trajectory_comparisons": trajectories, "model_cycles_not_environment_cycles": cycles,
              "source_sha256": sha(HERE / "diagnosticar_desfechos_aprendidos.py"), "bank_sha256": sha(path),
              "proposals_sha256": sha(proposal_path(game)), "human_level_goal_achieved": False,
              "scope": "oracle_future_states_only_for_error_measurement_no_new_policy_or_promotion"}
    output = HERE / "resultados" / f"{game}_diagnostico_desfechos.json"
    if output.exists():
        raise RuntimeError("diagnóstico já existe; preservar")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    r = run(parser.parse_args().game)
    print(json.dumps({"game": r["game"], "cases": r["cases"],
                      "by_action": [{k: v for k, v in row.items() if k != "scores"} | {
                          "scores": {name: {k: v for k, v in data.items() if not k.startswith("per_episode")}
                                     for name, data in row["scores"].items()}} for row in r["by_action"]],
                      "trajectory": r["real_trajectory_comparisons"]}, indent=2))
