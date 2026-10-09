"""Extrai mudanças observadas e aplica seus corpos ao estado consultado.

Um corpo é x -> x + delta. Sua escolha depende dos estados que originaram
as experiências; não cadastra objetos ou regras. Não é garantia de validade
fora da amostra nem descoberta de uma transformação científica inédita.
"""

from __future__ import annotations

import numpy as np

from futuros_distribuidos import weights


class LocalOperators:
    def __init__(self, source, successor, neighbors=8, shuffled=False):
        self.source = np.asarray(source, float)
        self.delta = np.asarray(successor, float) - self.source
        if self.source.ndim != 2 or self.delta.shape != self.source.shape or not len(self.source):
            raise ValueError("experiências incompatíveis")
        if not np.isfinite(self.source).all() or not np.isfinite(self.delta).all():
            raise ValueError("experiências não finitas")
        self.neighbors = neighbors
        if shuffled:
            self.delta = self.delta[np.random.default_rng(9777).permutation(len(self.delta))]

    def predict(self, query):
        query = np.asarray(query, float)
        return query + weights(query, self.source, self.neighbors) @ self.delta

    def candidates(self, query):
        query = np.asarray(query, float)
        probability = weights(query, self.source, self.neighbors)
        ids = np.argsort(-probability, axis=1)[:, :min(self.neighbors, len(self.source))]
        selected = np.take_along_axis(probability, ids, axis=1)
        # Cada alternativa é um corpo inteiro, não uma mistura de coordenadas
        # de diferentes alternativas. A média é somente predict(), acima.
        states = query[:, None] + self.delta[ids]
        return states, selected, ids

    def definition(self, index):
        return {"operator": "add_observed_change", "input_type": f"real_vector_{self.source.shape[1]}",
                "body": {"add": ["input", self.delta[index].tolist()]},
                "origin_state": self.source[index].tolist(),
                "scope": "candidate_local_operator_unpromoted"}
