"""Diagnóstico offline do mesmo banco para informação, contraste e escolha.

Não aprende observações de episódios reservados. Modelos com resultado verdadeiro
dos ramos são somente oráculos diagnósticos, nunca componentes do agente.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = list(itertools.combinations(range(4), 2))


def views(histories: np.ndarray) -> dict[str, np.ndarray]:
    h = histories.astype(np.float32) / 255.0
    n, t = h.shape[:2]
    c21 = h.reshape(n, t, 21, 10, 16, 10).mean((3, 5))
    c42 = h.reshape(n, t, 42, 5, 32, 5).mean((3, 5))
    last21 = c21[:, -1].reshape(n, -1)
    prev21 = c21[:, -2].reshape(n, -1)
    old = np.concatenate([last21, last21 - prev21], axis=1)
    last42 = c42[:, -1].reshape(n, -1)
    prev42 = c42[:, -2].reshape(n, -1)
    return {
        "r0_21x16_2": old,
        "hist_21x16_16": c21.reshape(n, -1),
        "res_42x32_2": np.concatenate([last42, last42 - prev42], axis=1),
        "hist_42x32_16": c42.reshape(n, -1),
        "native_160x210_2": np.concatenate(
            [h[:, -1].reshape(n, -1),
             (h[:, -1] - h[:, -2]).reshape(n, -1)], axis=1),
        "c21_history": c21,
    }


class KernelRidge:
    """Mesmo estimador e regularização para todos os braços observacionais."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @staticmethod
    def distances(x, y):
        d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T
        return np.maximum(d, 0)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.mean = x.mean(0)
        self.x = x - self.mean
        self.ym = y.mean(0)
        d = self.distances(self.x, self.x)
        nonzero = d[np.triu_indices(len(d), 1)]
        nonzero = nonzero[nonzero > 1e-15]
        self.scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        k = np.exp(-d / self.scale)
        self.coeff = np.linalg.solve(k + self.alpha * np.eye(len(k)), y - self.ym)
        return self

    def predict(self, x: np.ndarray):
        xx = np.asarray(x, np.float64) - self.mean
        return np.exp(-self.distances(xx, self.x) / self.scale) @ self.coeff + self.ym


def contrast(end_images: np.ndarray, horizon_index: int):
    e = end_images[:, horizon_index].reshape(len(end_images), 4, -1)
    return np.stack([e[:, a] - e[:, b] for a, b in PAIRS], axis=1)


def pixel_weights(y: np.ndarray) -> np.ndarray:
    w = np.mean(np.abs(y), axis=(0, 1))
    floor = 0.1 * max(float(w.max()), 1e-10)
    w = np.maximum(w, floor)
    return w / w.sum()


def loss(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    n, p, d = y.shape
    yp = np.asarray(prediction).reshape(n, p, d)
    return float(np.mean(np.sum((y - yp) ** 2 * weights, axis=2)))


def fit_predict(x, y, fit_indices, eval_indices):
    shape = y.shape[1:]
    model = KernelRidge().fit(x[fit_indices], y[fit_indices].reshape(len(fit_indices), -1))
    return model.predict(x[eval_indices]).reshape(len(eval_indices), *shape)


def group_split(episode):
    groups = sorted(set(map(int, episode)))
    if len(groups) < 8:
        raise ValueError("precisamos de oito episódios independentes para construção/seleção/teste")
    tr = np.flatnonzero(np.isin(episode, groups[:4]))
    val = np.flatnonzero(np.isin(episode, groups[4:6]))
    test = np.flatnonzero(np.isin(episode, groups[6:8]))
    if min(len(tr), len(val), len(test)) < 4:
        raise ValueError("partições de episódios sem amostras suficientes")
    return tr, val, test


def placebo_action_labels(end_images: np.ndarray, train: np.ndarray, seed: int):
    """Atribui desfechos a programas aleatórios DENTRO da mesma situação de treino."""
    rng = np.random.default_rng(seed)
    altered = end_images.copy()
    for i in train:
        altered[i] = np.take(end_images[i], rng.permutation(4), axis=1)
    return altered


def score_action_choice(x, returns, tr, test):
    true = returns[test]
    gap = true.max(1) - true.min(1)
    informative = gap > 1e-8
    if not informative.any():
        return {"informative_roots": 0, "status": "sem_contraste_de_retorno"}
    predicted = KernelRidge().fit(x[tr], returns[tr]).predict(x[test])
    choices = np.argmax(predicted, axis=1)
    chosen = true[np.arange(len(test)), choices]
    regret = (true.max(1) - chosen)[informative] / gap[informative]
    random_regret = ((true.max(1) - true.mean(1)) / np.maximum(gap, 1e-8))[informative]
    return {"informative_roots": int(informative.sum()),
            "relative_regret": float(regret.mean()),
            "uniform_regret": float(random_regret.mean()),
            "oracle_programs": 4,
            "scope": "somente quatro programas e horizonte medido"}


def memory_candidates(c21):
    """Leituras executáveis de uma faixa anterior; nenhum nome de entidade."""
    for lag in (2, 4, 8, 15):
        for y in range(0, 21, 3):
            for x in range(0, 16, 4):
                patch = c21[:, -1 - lag, y:min(y + 3, 21), x:min(x + 4, 16)]
                yield (lag, y, x), patch.mean((1, 2))


def append_memory(base, candidate, tr):
    mean = float(candidate[tr].mean())
    std = max(float(candidate[tr].std()), 1e-6)
    # Dá ao único registro de memória peso observável, sem excluir o estado base.
    scale = max(float(np.median(KernelRidge.distances(base[tr], base[tr]))), 1e-6) ** 0.5
    q = (candidate - mean) / std * 0.5 * scale
    return np.concatenate([base, q[:, None]], axis=1)


def propose_memory(base, c21, y, weights, episode, tr, val, test):
    """Encontra contraexemplos de r0, constrói leitura histórica e testa fora."""
    # O sensor escalar vem apenas dos contrastes de construção.
    pair_var = np.var(y[tr], axis=0).mean(1)
    pair = int(np.argmax(pair_var))
    centered = y[tr, pair] - y[tr, pair].mean(0)
    if np.linalg.norm(centered) < 1e-12:
        return {"status": "nenhum_contraste_visual_para_gerar_sensor"}
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    target = y[:, pair] @ vt[0]
    residual = np.zeros(len(tr))
    for group in np.unique(episode[tr]):
        held = tr[episode[tr] == group]
        other = tr[episode[tr] != group]
        if len(other) < 4:
            continue
        residual[np.searchsorted(tr, held)] = target[held] - KernelRidge().fit(
            base[other], target[other, None]).predict(base[held]).ravel()
    if np.std(residual) < 1e-10:
        return {"status": "sem_residuo_para_reparar"}
    ranked = []
    for spec, candidate in memory_candidates(c21):
        v = candidate[tr]
        if v.std() < 1e-6:
            continue
        correlation = float(np.corrcoef(v, residual)[0, 1])
        if np.isfinite(correlation):
            ranked.append((abs(correlation), spec, candidate))
    ranked.sort(key=lambda item: -item[0])
    if not ranked:
        return {"status": "nenhuma_leitura_de_memoria_distinguivel"}
    baseline_val = loss(y[val], fit_predict(base, y, tr, val), weights)
    scored = []
    for _correlation, spec, candidate in ranked[:8]:
        augmented = append_memory(base, candidate, tr)
        error = loss(y[val], fit_predict(augmented, y, tr, val), weights)
        scored.append((error, spec, candidate))
    val_error, spec, candidate = min(scored, key=lambda item: item[0])
    fit = np.sort(np.concatenate([tr, val]))
    augmented = append_memory(base, candidate, fit)
    baseline_test = loss(y[test], fit_predict(base, y, fit, test), weights)
    augmented_test = loss(y[test], fit_predict(augmented, y, fit, test), weights)
    return {"status": "candidata_testada", "sensor_pareado": PAIRS[pair],
            "programa": {"operacao": "media_de_regiao_passada",
                         "atraso_quadros": spec[0], "linha": spec[1], "coluna": spec[2],
                         "altura": min(3, 21 - spec[1]), "largura": min(4, 16 - spec[2])},
            "erro_validacao_base": baseline_val,
            "erro_validacao_memoria": val_error,
            "erro_teste_base": baseline_test,
            "erro_teste_memoria": augmented_test,
            "ganho_teste": 1.0 - augmented_test / max(baseline_test, 1e-12),
            "promovida": bool(val_error < baseline_val and augmented_test < baseline_test)}


def evaluate(bank: Path) -> dict:
    data = np.load(bank)
    groups = data["episode"]
    tr, val, test = group_split(groups)
    raw = data["histories"]
    feature = views(raw)
    out = {"bank": str(bank), "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
           "groups": {"construction": sorted(set(map(int, groups[tr]))),
                      "selection": sorted(set(map(int, groups[val]))),
                      "evaluation": sorted(set(map(int, groups[test])))},
           "root_count": len(groups), "horizons": {}}
    views_to_test = ["r0_21x16_2", "hist_21x16_16", "res_42x32_2",
                     "hist_42x32_16", "native_160x210_2"]
    for hi, horizon in enumerate(data["prefixes"]):
        y = contrast(data["end_images"], hi)
        w = pixel_weights(y[tr])
        zero = loss(y[test], np.zeros_like(y[test]), w)
        root_effect = np.mean(np.abs(y), axis=(1, 2))
        permuted = placebo_action_labels(data["end_images"], tr, 11 + hi)
        placebo = contrast(permuted, hi)
        scores = {}
        fit = np.sort(np.concatenate([tr, val]))
        for name in views_to_test:
            x = feature[name]
            pred = fit_predict(x, y, fit, test)
            scores[name] = {
                "contrast_mse": loss(y[test], pred, w),
                "return_choice": score_action_choice(x, data["returns"][:, hi], fit, test),
            }
        placebo_fit = np.sort(np.concatenate([tr, val]))
        # O placebo conserva a aparência/situação e quebra só programa→desfecho.
        scores["placebo_r0"] = {"contrast_mse": loss(
            y[test], fit_predict(feature["r0_21x16_2"], placebo, placebo_fit, test), w)}
        memory = propose_memory(feature["r0_21x16_2"], feature["c21_history"],
                                y, w, groups, tr, val, test) if hi == 2 else None
        out["horizons"][str(int(horizon))] = {
            "zero_prediction_mse": zero, "visual_effect_mean": float(root_effect[test].mean()),
            "reward_informative_roots": int(np.sum(np.ptp(data["returns"][test, hi], axis=1) > 0)),
            "scores": scores, "memory_proposal": memory,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.bank)
    target = args.out or args.bank.with_name(args.bank.stem + "_diagnostico.json")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
