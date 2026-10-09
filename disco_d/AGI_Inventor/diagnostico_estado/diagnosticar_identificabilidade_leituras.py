"""Diagnóstico posterior: a amostra exige contexto ou admite valor só da palavra?"""

import json
from collections import defaultdict, deque

import numpy as np

from avaliar_leituras_por_programa import bank_path, manifest_path, reference_path, stem
from piloto_planos_propostos import HERE, sha


def word_potential(words, differences):
    """V(P)-V(Q)=d: existência no grafo finito, não previsão de palavras novas."""
    adjacency = defaultdict(list)
    for (p, q), value in zip(words, differences):
        adjacency[p].append((q, -float(value)))
        adjacency[q].append((p, float(value)))
    potential, components = {}, 0
    for node in adjacency:
        if node in potential:
            continue
        components += 1
        potential[node], queue = 0., deque([node])
        while queue:
            current = queue.popleft()
            for other, offset in adjacency[current]:
                if other not in potential:
                    potential[other] = potential[current] + offset
                    queue.append(other)
    residual = np.asarray([potential[p] - potential[q] - d for (p, q), d in zip(words, differences)])
    return {"nodes": len(adjacency), "edges": len(words), "components": components,
            "maximum_cycle_inconsistency": float(np.max(np.abs(residual))),
            "program_only_potential_can_interpolate_this_finite_sample": bool(np.max(np.abs(residual)) < 1e-12)}


def crossings(plans, differences, length):
    groups, words = defaultdict(list), []
    for i in range(len(plans)):
        p, q = tuple(int(a) for a in plans[i, 0, :length]), tuple(int(a) for a in plans[i, 1, :length])
        words.append((p, q))
        groups[min((p, q), (q, p))].append(float(differences[i]) if p <= q else -float(differences[i]))
    repeated = [values for values in groups.values() if len(values) > 1]
    return {"prefix_blocks": length, "roots": len(plans), "distinct_unordered_pairs": len(groups),
            "repeated_pair_groups": len(repeated), "roots_in_repeated_pair_groups": sum(map(len, repeated)),
            "repeated_pairs_with_different_returns": sum(max(v) - min(v) > 1e-12 for v in repeated),
            "repeated_pairs_requiring_opposite_decisions": sum(min(v) < -1e-12 and max(v) > 1e-12 for v in repeated),
            "nonzero_real_contrasts": int(np.sum(np.abs(differences) > 1e-12)),
            "finite_word_potential": word_potential(words, differences)}


def diagnostic(game):
    with np.load(bank_path(game, True)) as archive:
        train_plans, train_returns = archive["actions"], archive["rewards"].cumsum(-1)
    with np.load(bank_path(game)) as archive:
        plans, returns, episode = archive["actions"], archive["rewards"].cumsum(-1), archive["episode"]
    report_path = stem(game).with_name(stem(game).name + "_novas_103.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows, contrasts = {}, {}
    for label, ps, rewards in (("train97", train_plans, train_returns), ("known_development103", plans, returns)):
        rows[label] = [crossings(ps, rewards[:, 0, length - 1] - rewards[:, 1, length - 1], length)
                       for length in (1, 2, 4, 8, 16)]
    for horizon in (32, 64, 128):
        curve = report["curves"][str(horizon)]
        primary = np.asarray(curve["quadratic"]["choices"])
        real = returns[:, :, horizon // 8 - 1]
        contrasts[str(horizon)] = {
            "actual_non_tied_pairs": int(np.sum(real[:, 0] != real[:, 1])),
            "episodes_with_actual_contrast": int(sum(np.any(real[episode == e, 0] != real[episode == e, 1]) for e in range(8))),
            "mean_pair_value_range": float((real.max(1) - real.min(1)).mean()),
            "decisions_changed_by_quadratic_vs_affine": int(np.sum(primary != curve["affine"]["choices"])),
            "decisions_changed_by_history_vs_two_frames": int(np.sum(primary != curve["two_frames"]["choices"])),
            "decisions_changed_by_pixels_vs_word_only": int(np.sum(primary != curve["word_only"]["choices"]))}
    return {"game": game, "crossed_coverage": rows, "diagnostics": contrasts,
            "primary_vs_controls": {k: report["comparisons"][k] for k in
                                    ("affine", "two_frames", "shuffled_targets", "shuffled_histories", "word_only", "raw_rbf", "old_constant_program_index")},
            "source_hashes": {p.relative_to(HERE).as_posix(): sha(p) for p in
                              (bank_path(game, True), bank_path(game), report_path, manifest_path(game), reference_path(game))}}


if __name__ == "__main__":
    result = {"status": "post_result_diagnostic_not_a_changed_gate", "games": [diagnostic(g) for g in ("breakout", "pong")],
              "limits": ["a_word_only_interpolant_does_not_predict_unseen_words",
                         "absence_of_crossings_does_not_prove_context_is_useless",
                         "this_diagnoses_experiment_identification_not_the_unique_cause_of_model_failure"],
              "human_level_goal_achieved": False, "source_sha256": sha(HERE / "diagnosticar_identificabilidade_leituras.py")}
    output = HERE / "resultados" / "identificabilidade_leituras_por_programa.json"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise RuntimeError("diagnóstico existente diverge; preservar")
    else:
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
