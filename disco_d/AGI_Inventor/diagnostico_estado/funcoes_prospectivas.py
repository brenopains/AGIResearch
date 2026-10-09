"""Variáveis propostas por consequências futuras confundidas, sem nomes de mundo.

Mesmo alfabeto limitado do gerador anterior; novidade testada é a origem
prospectiva dos parâmetros, não uma alegação de técnica inédita ou AGI.
"""
import numpy as np
from funcoes_contraexemplos import PairFunctions, ATOMS, PRODUCTS
from transporte_funcoes import distances


def crossfit_future(phi, actions, target, episodes, alpha=.05):
    phi, actions, target, episodes = map(np.asarray, (phi, actions, target, episodes))
    if phi.ndim != 2 or target.ndim != 2 or len(target) != len(phi) or actions.shape != episodes.shape or actions.shape != (len(phi),):
        raise ValueError("contrato de experiência inválido")
    if not all(np.isfinite(v).all() for v in (phi, target)):
        raise ValueError("experiência não finita")
    prediction = np.empty_like(target, dtype=float)
    solves = 0
    for episode in sorted(set(episodes)):
        for action in sorted(set(actions)):
            train = (episodes != episode) & (actions == action)
            held = (episodes == episode) & (actions == action)
            if held.any() and not train.any():
                raise ValueError("pergunta sem repetição em outro episódio")
            if not held.any():
                continue
            x = phi[train]
            coefficient = np.linalg.solve(x.T @ x + alpha * np.eye(phi.shape[1]), x.T @ target[train])
            prediction[held] = phi[held] @ coefficient
            solves += 1
    return prediction, solves


def propose_future(source, actions, successor_phi, continuation, episodes, phi, shuffled=False):
    source, actions, successor_phi, continuation, episodes, phi = map(np.asarray,
        (source, actions, successor_phi, continuation, episodes, phi))
    raw = source.reshape(len(source), -1).astype(float)
    if successor_phi.shape != phi.shape or continuation.shape != (len(raw),) or not np.isin(continuation, [0, 1]).all():
        raise ValueError("consequência incompatível")
    target = continuation[:, None] * successor_phi
    rng = np.random.default_rng(10179)
    if shuffled:
        for action in sorted(set(actions)):
            ids = np.flatnonzero(actions == action)
            target[ids] = target[rng.permutation(ids)]
    scale = np.std(target, axis=0)
    active = scale > 1e-10
    if not active.any():
        return PairFunctions(np.empty((0, raw.shape[1])), np.empty(0), [], []), {
            "status": "no_varying_future_question", "atoms": 0, "products": 0,
            "active_future_coordinates": [], "candidate_pairs": 0, "crossfit_solves": 0}
    normalized = target[:, active] / scale[active]
    forecast, solves = crossfit_future(phi, actions, normalized, episodes)
    residual = normalized - forecast
    pairs = []
    for action in sorted(set(actions)):
        ids = np.flatnonzero(actions == action)
        residual_dist = distances(residual[ids], residual[ids])
        context_dist = distances(phi[ids], phi[ids])
        future_dist = distances(normalized[ids], normalized[ids])
        rows, cols = np.triu_indices(len(ids), 1)
        keep = (episodes[ids[rows]] != episodes[ids[cols]]) & (future_dist[rows, cols] > 1e-12)
        for a, b in zip(rows[keep], cols[keep]):
            score = float(residual_dist[a, b] / (1 + context_dist[a, b]))
            pairs.append((score, int(ids[a]), int(ids[b])))
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    directions, offsets, origins, conflicts = [], [], [], 0
    for score, i, j in pairs:
        difference = raw[i] - raw[j]
        denominator = float(difference @ difference)
        if denominator <= 1e-14:
            conflicts += 1
            continue
        direction = 4 * difference / denominator
        offset = float((raw[i] + raw[j]) @ direction / 2)
        if any(np.allclose(direction, old, rtol=1e-8, atol=1e-12) and np.isclose(offset, old_offset, rtol=1e-8, atol=1e-12)
               for old, old_offset in zip(directions, offsets)):
            continue
        directions.append(direction)
        offsets.append(offset)
        origins.append({"kind": "observed_pair", "rows": [i, j], "episodes": [int(episodes[i]), int(episodes[j])],
            "action": int(actions[i]), "question": "continuation_times_observed_successor_functions",
            "shuffled_future_labels": bool(shuffled), "future_targets_recorded": target[[i, j]].tolist(),
            "selection_future_residual_score": score})
        if len(directions) == ATOMS:
            break
    directions = np.asarray(directions, float).reshape(-1, raw.shape[1])
    expressions = [{"op": "unit", "units": [i]} for i in range(len(directions))]
    products = [(i, j) for i in range(min(6, len(directions))) for j in range(i + 1, min(6, len(directions)))][:PRODUCTS]
    expressions.extend({"op": "product", "units": [i, j]} for i, j in products)
    return PairFunctions(directions, np.asarray(offsets), expressions, origins), {
        "status": "bounded_prospective_functions_generated", "atoms": len(directions), "products": len(products),
        "active_future_coordinates": np.flatnonzero(active).tolist(), "candidate_pairs": len(pairs),
        "crossfit_solves": solves, "raw_aliases_encountered": conflicts,
        "generation_episodes": sorted(map(int, set(episodes))), "validation_targets_used": 0}
