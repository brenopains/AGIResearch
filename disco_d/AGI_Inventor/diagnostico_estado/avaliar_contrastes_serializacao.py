"""Entrada de avaliação congelada: só converter escalares NumPy para JSON.

O módulo original continua com seus hashes intactos. Não altera modelos,
portões, dados, previsões ou consultas: corrige apenas dumps no namespace
do avaliador, sem modificar a biblioteca json ou NumPy globalmente.
"""

import argparse
import json as standard_json

import numpy as np

import avaliar_contrastes_programas as frozen
from piloto_planos_propostos import sha


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"tipo sem serialização explícita: {type(value).__name__}")


class ScalarJson:
    loads = staticmethod(standard_json.loads)

    @staticmethod
    def dumps(value, **kwargs):
        return standard_json.dumps(value, default=numpy_scalar, **kwargs)


def run(game):
    original_json = frozen.json
    source_path = frozen.HERE / "avaliar_contrastes_programas.py"
    original_hash = sha(source_path)
    try:
        frozen.json = ScalarJson
        result = frozen.evaluate(game)
    finally:
        frozen.json = original_json
    if sha(source_path) != original_hash:
        raise RuntimeError("fonte congelada mudou")
    result_path = frozen.stem(game).with_name(frozen.stem(game).name + "_novas_103.json")
    record_path = frozen.stem(game).with_name(frozen.stem(game).name + "_serializacao.json")
    record = {"status": "numpy_scalar_serialization_only", "source_frozen_sha256": original_hash,
              "adapter_source_sha256": sha(frozen.HERE / "avaliar_contrastes_serializacao.py"),
              "unchanged_evaluation_report_sha256": sha(result_path),
              "models_retrained": False, "gates_changed": False, "new_ale_steps": 0}
    record_path.write_text(standard_json.dumps(record, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=("breakout", "pong"))
    result = run(parser.parse_args().game)
    print(ScalarJson.dumps({"game": result["game"], "status": result["status"],
          "horizon128": {name: {"regret": row["mean_regret"], "mse": row["contrast_mse"]}
                          for name, row in result["curves"]["128"].items()},
          "word_overlap": result["overlapping_complete_words"]}, indent=2))
