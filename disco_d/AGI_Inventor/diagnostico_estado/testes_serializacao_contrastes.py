import json

import numpy as np

from avaliar_contrastes_serializacao import ScalarJson

assert json.loads(ScalarJson.dumps({"flag": np.bool_(True), "n": np.int64(4), "x": np.float64(.25)}, allow_nan=False)) == {"flag": True, "n": 4, "x": .25}
for value in (np.float64(np.nan), np.float64(np.inf), object()):
    try:
        ScalarJson.dumps({"value": value}, allow_nan=False)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("tipo inválido ou valor não finito foi mascarado")
print("serializacao_contrastes_ok: tipos_explicitos, significado_preservado, nao_finitos_rejeitados")
