"""Seleciona intervenções por previsões incompatíveis, sem escore de promessa."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Hashable, Mapping


@dataclass(frozen=True)
class Intervalo:
    minimo: float
    maximo: float

    def __post_init__(self):
        if not (isfinite(self.minimo) and isfinite(self.maximo)
                and self.minimo <= self.maximo):
            raise ValueError("intervalo inválido")

    def contem(self, valor: float) -> bool:
        return self.minimo <= valor <= self.maximo

    def disjunto(self, outro: "Intervalo") -> bool:
        return self.maximo < outro.minimo or outro.maximo < self.minimo


@dataclass(frozen=True)
class Hipotese:
    nome: str
    previsoes: Mapping[Hashable, Intervalo]
    origem: str
    exata: bool


def encontrar(hipoteses: list[Hipotese], consultas: list[Hashable],
             custo: Mapping[Hashable, float] | None = None):
    if len({h.nome for h in hipoteses}) != len(hipoteses):
        raise ValueError("nomes de hipóteses devem ser únicos")
    if not hipoteses:
        return {"status": "sem_modelos"}
    custo = custo or {}
    ordenadas = sorted(enumerate(consultas), key=lambda row: (custo.get(row[1], 1), row[0]))
    for _, consulta in ordenadas:
        for i, left in enumerate(hipoteses):
            if consulta not in left.previsoes:
                continue
            for right in hipoteses[i + 1:]:
                if consulta not in right.previsoes:
                    continue
                a, b = left.previsoes[consulta], right.previsoes[consulta]
                if a.disjunto(b):
                    return {"status": "testemunha", "consulta": consulta,
                            "par": [left.nome, right.nome],
                            "previsoes": [[a.minimo, a.maximo], [b.minimo, b.maximo]],
                            "custo": float(custo.get(consulta, 1)),
                            "alcance": "exato" if left.exata and right.exata else "estatistico"}
    return {"status": "indistinguiveis_no_dominio", "consultas": len(consultas),
            "modelos": len(hipoteses)}


def incorporar(hipoteses: list[Hipotese], consulta: Hashable, observacao: float):
    if not isfinite(observacao):
        raise ValueError("observação inválida")
    sobreviventes = [h for h in hipoteses
                     if consulta in h.previsoes and h.previsoes[consulta].contem(observacao)]
    eliminadas = [h.nome for h in hipoteses if h not in sobreviventes]
    if not sobreviventes:
        status = "modelos_inadequados"
    elif len(sobreviventes) == 1:
        status = "uma_hipotese_restante"
    else:
        status = "hipoteses_restantes"
    return {"status": status, "consulta": consulta, "observacao": observacao,
            "sobreviventes": [h.nome for h in sobreviventes], "eliminadas": eliminadas}
