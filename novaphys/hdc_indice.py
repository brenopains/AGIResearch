"""
novaphys.hdc_indice — o indice consultavel, e ele e MULTIMODAL por construcao
==============================================================================

DE ONDE VEM
-------------
Do usuario: "de um lado extrai o puro, do outro digere", com o HDC servindo a
busca sob demanda — nao gerando, SERVINDO o gerador. E: "precisa ser
multimodal".

[[sob-demanda-funciona]] mostrou que ordenar a loja NA HORA DA TAREFA bate
fundir tudo e bate o sorteio por 12-20 desvios, com o ranqueador por COMPRESSAO.
Mas compressao so compara dentro do MESMO espaco de representacao: sonda de
texto nao ranqueia bloco de fluxo. Por isso o ranqueador vira HDC.

O QUE HDC RESOLVE AQUI QUE COMPRESSAO NAO RESOLVE
---------------------------------------------------
Um hipervetor de dimensao fixa nao se importa com a origem. Texto vira
hipervetor, fluxo T x K vira hipervetor, e os dois vivem no MESMO espaco —
entao `sim(sonda_de_texto, bloco_de_fluxo)` e uma operacao definida. Esse e o
unico caminho medido para a consulta multimodal que o usuario pede.

A LICAO REGISTRADA QUE ESTE CODIGO HONRA
------------------------------------------
`prion_hdc.py` documenta que a primeira versao usava so `bind` e deu
similaridade ~0,004 ENTRE TUDO — nivel de acaso — porque bind produz vetor
DISSIMILAR aos operandos (e o proposito dele: criar simbolo novo). Similaridade
estrutural exige SUPERPOSICAO. Aqui bind so amarra papel-a-valor dentro de um
termo, e o feixe (soma com sinal) e quem cria a similaridade entre itens.

E o outro limite medido, que define o PAPEL: HDC ordena (erro 2,6 d contra
11,3 d do sorteio) e nao gera (2/400; recombinar partes destroi sinal, porque
o cleanup so recupera combinacao de atomos JA presentes). Este modulo e so o
ordenador. Quem gera e outro.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np

DIM = 10000
SEMENTE = 20260917

Fluxo = Sequence[Sequence[float]]


class Espaco:
    """
    Espaco bipolar comum a todas as modalidades. Atomos sao gerados sob demanda
    a partir do NOME, por hash — assim qualquer simbolo de qualquer modalidade
    tem um atomo estavel sem tabela pre-declarada, e duas execucoes concordam.
    """

    def __init__(self, dim: int = DIM, semente: int = SEMENTE) -> None:
        self.dim = dim
        self.semente = semente
        self._cache: Dict[str, np.ndarray] = {}

    def atomo(self, nome: str) -> np.ndarray:
        v = self._cache.get(nome)
        if v is None:
            h = hashlib.blake2b(f"{self.semente}:{nome}".encode(),
                                digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(h, "big"))
            v = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
            self._cache[nome] = v
        return v

    # bind amarra papel a valor DENTRO de um termo; nao cria similaridade
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).astype(np.int8)

    def desloca(self, v: np.ndarray, k: int) -> np.ndarray:
        return np.roll(v, k)


def feixe(vs: Sequence[np.ndarray], dim: int = DIM) -> np.ndarray:
    """Soma com sinal. E daqui que vem TODA a similaridade entre itens."""
    if len(vs) == 0:
        return np.zeros(dim, dtype=np.int8)
    s = np.sum(np.stack([v.astype(np.int32) for v in vs]), axis=0)
    return np.where(s >= 0, 1, -1).astype(np.int8)


def sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / len(a))


# ───────────────────────────────────────────── encoders por modalidade
def de_texto(e: Espaco, s: str, n: int = 3, passo: int = 1) -> np.ndarray:
    """Fluxo de simbolos -> hipervetor. Feixe dos n-gramas."""
    if len(s) < n:
        return e.atomo("vazio")
    termos = [e.atomo("g:" + s[i:i + n]) for i in range(0, len(s) - n + 1, passo)]
    return feixe(termos, e.dim)


def _quantiza(x: float, mu: float, sd: float, niveis: int = 8) -> int:
    z = (x - mu) / (sd if sd > 1e-9 else 1.0)
    k = int(np.clip(round(z * 2), -(niveis // 2), niveis // 2))
    return k


def de_fluxo(e: Espaco, fluxo: Fluxo, niveis: int = 8,
             atrasos: Sequence[int] = (0, 1, 3)) -> np.ndarray:
    """
    T x K -> hipervetor, NO MESMO ESPACO do texto.

    Cada instante contribui com o feixe de (canal (x) nivel), deslocado pelo
    ATRASO, para que forma temporal entre num item que so tem dimensao fixa.
    Padroniza por canal de proposito: a FORMA e o que deve sobreviver, nao a
    escala — que e exatamente o que `substrato.gerar_fluxo` embaralha entre os
    canais 0-2 e 3-5.
    """
    A = np.asarray(fluxo, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        return e.atomo("vazio")
    mu, sd = A.mean(axis=0), A.std(axis=0)
    termos: List[np.ndarray] = []
    T, K = A.shape
    for d in atrasos:
        for t in range(d, T):
            for k in range(K):
                niv = _quantiza(A[t - d, k], mu[k], sd[k], niveis)
                v = e.bind(e.atomo(f"c:{k}"), e.atomo(f"n:{niv}"))
                termos.append(e.desloca(v, d))
    return feixe(termos, e.dim)


def de_par(e: Espaco, fluxo: Fluxo, alvo: Sequence[int],
           atrasos: Sequence[int] = (0, 1, 3)) -> np.ndarray:
    """
    Fluxo MAIS o alvo — a 'transformacao' amarrada ao 'material'.

    O alvo entra ligado ao instante, entao dois episodios com a MESMA regra
    (alvo segue o latente do mesmo jeito) ficam proximos mesmo com canais de
    escalas diferentes. E a aposta de forma-sobre-superficie.
    """
    A = np.asarray(fluxo, dtype=np.float64)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    termos: List[np.ndarray] = []
    T, K = A.shape
    for t in range(T):
        a = e.atomo(f"a:{1 if alvo[t] > 0 else -1}")
        for k in range(K):
            niv = _quantiza(A[t, k], mu[k], sd[k])
            termos.append(e.bind(e.atomo(f"c:{k}"),
                                 e.bind(e.atomo(f"n:{niv}"), a)))
    return feixe(termos, e.dim)


# ───────────────────────────────────────────────────────── a loja
class Loja:
    """Guarda tudo. Nada e descartado. Ordena sob demanda."""

    def __init__(self, espaco: Espaco | None = None) -> None:
        self.e = espaco or Espaco()
        self.itens: List[dict] = []

    def guarda(self, rotulo: str, vetor: np.ndarray, carga) -> None:
        self.itens.append({"rotulo": rotulo, "v": vetor, "carga": carga})

    def ordena(self, sonda: np.ndarray) -> List[tuple]:
        """Ranking para ESTA tarefa. Sem limiar, sem descarte."""
        r = [(sim(sonda, it["v"]), it["rotulo"], it) for it in self.itens]
        r.sort(key=lambda x: -x[0])
        return r


def de_par_inv(e: Espaco, fluxo: Fluxo, alvo: Sequence[int],
               atrasos: Sequence[int] = (0, 1), niveis: int = 8) -> np.ndarray:
    """
    Versao INVARIANTE de `de_par`. Tres invariancias, e cada uma conserta um
    jeito de o indice enxergar superficie em vez de forma.

    O DIAGNOSTICO QUE ESTE CODIGO ATACA ([[indice-ve-superficie]]): `de_par`
    amarra alvo a NIVEL quantizado, e no substrato o canal 0 e `1,7*s` enquanto
    o canal 3 e `-2,4*s + 3,0` — o material B codifica o latente com SINAL
    INVERTIDO. Sob 'regime' o alvo e +1 com latente alto, o que em A e nivel
    ALTO ligado a +1 e em B e nivel BAIXO ligado a +1. A mesma transformacao
    produzia associacoes ANTICORRELACIONADAS, e a similaridade despencava
    (mesma transf +0,0988 contra mesmo material +0,5188).

      1. ESCALA   z-score por canal dentro do episodio (ja havia).
      2. SINAL    orienta cada canal pelo sinal da correlacao dele com o alvo.
                  Alinhar o sinal antes de comparar e, em si, um passo de
                  abducao — e usa so o que o episodio guardado ja tem.
      3. PAPEL    amarra ao POSTO do canal por |correlacao|, nao ao INDICE.
                  "O canal mais associado ao alvo" ocupa o mesmo lugar nos dois
                  materiais ainda que seja o canal 0 num e o 3 no outro. Em uso
                  multimodal e o que permite um canal de video e um de audio
                  ocuparem o mesmo papel.

    O atraso 1 entra para que uma regra sobre MUDANCA ('deriva') tambem possa
    se expressar como associacao consistente, e nao so uma regra sobre nivel.
    """
    A = np.asarray(fluxo, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        return e.atomo("vazio")
    T, K = A.shape
    y = np.asarray([1.0 if a > 0 else -1.0 for a in alvo[:T]])
    mu, sd = A.mean(axis=0), A.std(axis=0)
    Z = (A - mu) / np.where(sd > 1e-9, sd, 1.0)

    r = np.zeros(K)
    for k in range(K):
        if np.std(Z[:, k]) > 1e-9 and np.std(y) > 1e-9:
            r[k] = np.corrcoef(Z[:, k], y)[0, 1]
    r = np.nan_to_num(r)

    Zo = Z * np.sign(np.where(np.abs(r) < 1e-9, 1.0, r))   # invariancia 2
    ordem = np.argsort(-np.abs(r))                          # invariancia 3
    papel = np.empty(K, dtype=int)
    papel[ordem] = np.arange(K)

    lim = niveis // 2
    termos: List[np.ndarray] = []
    for d in atrasos:
        for t in range(d, T):
            a = e.atomo(f"a:{int(y[t])}")
            for k in range(K):
                niv = int(np.clip(round(Zo[t - d, k] * 2), -lim, lim))
                v = e.bind(e.atomo(f"p:{papel[k]}"),
                           e.bind(e.atomo(f"n:{niv}"), a))
                termos.append(e.desloca(v, d))
    return feixe(termos, e.dim)
