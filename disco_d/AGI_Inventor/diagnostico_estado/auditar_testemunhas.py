"""Aplica o contrato de testemunha ao comitê Atari já treinado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from avaliar import group_split, views
from desacordo_ativo import committee
from testemunha import Hipotese, Intervalo, encontrar, incorporar

HERE = Path(__file__).resolve().parent


def run(game):
    bank = HERE / "resultados" / f"{game}_ciclo.npz"
    data = np.load(bank)
    episode = data["episode"]
    values = data["returns"].reshape(-1, 4, 4).max(axis=2)
    feature = views(data["histories"])
    members, committee_hash = committee(game)
    tr, val, test = group_split(episode)
    fit = np.sort(np.concatenate((tr, val)))
    residuals = []
    for i in fit:
        own = int(episode[i])
        for index in (own, 6 + own):
            name, model = members[index]
            residuals.append(np.abs(values[i] - model.predict(feature[name][i:i + 1])[0]))
    radius = np.quantile(np.asarray(residuals), 0.9, axis=0)
    predictions = np.stack([model.predict(feature[name][test]) for name, model in members])
    rows = []
    for local, root in enumerate(test):
        hypotheses = []
        for member, (name, _model) in enumerate(members):
            forecast = predictions[member, local]
            hypotheses.append(Hipotese(
                f"{name}:omitido-{member % 6}",
                {action: Intervalo(float(forecast[action] - radius[action]),
                                   float(forecast[action] + radius[action]))
                 for action in range(4)},
                origem=f"banco:{committee_hash}:episodios-0a5",
                exata=False))
        query = encontrar(hypotheses, [0, 1, 2, 3])
        compatible_full = sum(
            all(h.previsoes[a].contem(float(values[root, a])) for a in range(4))
            for h in hypotheses)
        # Diagnóstico offline: responder somente quando uma consulta foi proposta.
        remaining, path = hypotheses, []
        unused = [0, 1, 2, 3]
        while len(remaining) > 1:
            proposed = encontrar(remaining, unused)
            if proposed["status"] != "testemunha":
                break
            action = proposed["consulta"]
            outcome = incorporar(remaining, action, float(values[root, action]))
            path.append({"proposta": proposed, "resultado": outcome})
            unused.remove(action)
            remaining = [h for h in remaining if h.nome in outcome["sobreviventes"]]
            if not remaining:
                break
        rows.append({"episode": int(episode[root]), "root": int(root),
                     "first_query_status": query["status"],
                     "first_query": query if query["status"] == "testemunha" else None,
                     "jointly_compatible_models": compatible_full,
                     "queries_executed_offline": len(path),
                     "last_status": (path[-1]["resultado"]["status"] if path else query["status"]),
                     "path": path})
    report = {"game": game, "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
              "committee_bank_sha256": committee_hash,
              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                for name in ("testemunha.py", "auditar_testemunhas.py",
                                             "desacordo_ativo.py")},
              "calibration_member_error_q90": radius.tolist(),
              "test_roots": len(test),
              "roots_with_witness": sum(r["first_query_status"] == "testemunha" for r in rows),
              "roots_with_any_jointly_compatible_model": sum(
                  r["jointly_compatible_models"] > 0 for r in rows),
              "roots_all_models_refuted_after_query": sum(
                  r["last_status"] == "modelos_inadequados" for r in rows),
              "rows": rows}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    args = parser.parse_args()
    result = run(args.game)
    target = HERE / "resultados" / f"{args.game}_testemunhas.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
