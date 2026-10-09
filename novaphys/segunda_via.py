"""
novaphys.segunda_via — corroboracao entre mundos como SEGUNDA VIA de admissao
===============================================================================

O PROBLEMA, QUE E O MAIS ANTIGO DO PROJETO
--------------------------------------------
O usuario perguntou: "como tu ainda ta escrevendo primitiva? ja nao conseguimos
extrair, sintetizar e combinar?" A auditoria respondeu com numero:

    codecs no nucleo que resolve ARC          23
    definidos a mao por mim                   23
    vindos de modulo de geracao                0

E a razao NAO e que a geracao falha em produzir. Ela produz —
[[inventar-a-grandeza]] (o sistema inventou h = x vy - y vx de trajetorias),
[[gramatica-propria]] (derivou sozinho a direcao que eu tinha posto a mao),
[[segunda-ordem-compoe]], [[ciclo-fecha-sem-rotulo]]. A razao e
[[o-criterio-nao-seleciona]]: os vocabularios gerados **nunca entram em pilha
vencedora**, e perdem em BITS, nao em orcamento. E
[[paisagem-e-penhasco]] explica por que nao ha como subir: trocar um marcador de
pilha exata salta o erro de 0 para mediana 46 em 96% dos casos, sem valores
intermediarios. Um operador novo nao tem gradiente ate a pilha vencedora.

A IDEIA A TESTAR
------------------
[[corrobora-entre-mundos]] mediu que concordancia entre mundos independentes
bate credito calibrado em todas as doses e nao custa calibracao nenhuma. Se um
operador e util em varios mundos, isso e evidencia que NAO depende de ele ganhar
em bits no ARC sozinho. Dai a segunda via: admitir por corroboracao.

MAS ANTES DE CONSTRUIR A VIA, A PRE-CONDICAO
----------------------------------------------
Este projeto tem doze nulos, e boa parte veio de construir antes de checar se o
sinal existia. Entao aqui nao se constroi nada: mede-se primeiro se evidencia
entre mundos diz ALGUMA coisa sobre utilidade no ARC.

A PONTE, E ELA E EXATA
------------------------
Um estado normalizado do atlas e 8x8 com valores 0..3 — isto e, **um Grid do ARC
valido**. Entao os 23 codecs REAIS do ARC rodam, sem adaptacao, sobre pares
(estado, operador(estado)) montados de CIFAR, HAR, moda, texto e ARC. Nao e
analogia: e o mesmo codigo, o mesmo `propor/bits/prever`, a mesma regua `custo`.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  A0  PRE-CONDICAO. Os 23 codecs VARIAM em corroboracao entre mundos (nem todos
      em 0, nem todos em 5). Sem variacao a medida nao tem resolucao e nada
      depois significa coisa alguma.
  A1  corroboracao entre mundos correlaciona com uso no ARC (rho de ordem
      >= 0,40 nas 105 pilhas vencedoras em cache)
      -> evidencia entre mundos CARREGA informacao sobre utilidade no ARC, e a
         segunda via tem chao. So entao vale construir.
  A2  |rho| < 0,40 -> evidencia entre mundos nao diz nada sobre o ARC; a
      segunda via e infundada e eu paro aqui em vez de construir por estetica.
  A3  rho <= -0,40 -> o que atravessa mundos e justamente o que NAO serve no
      ARC, e isso seria o achado mais interessante dos tres.

  CONFUNDIDOR NOMEADO, e ele e o risco principal: tanto "funciona em muitos
  mundos" quanto "aparece em muitas pilhas" podem ser efeito de GENERALIDADE
  BARATA — um codec generico e barato entra em tudo por preco, nao por
  corroboracao. Controle: repetir a correlacao parcializando os BITS do codec.
  Se o rho morrer ao controlar bits, o sinal era preco.

  `copia` fica FORA da correlacao principal, e nao por conveniencia: ele e o
  elemento identidade da pilha (76% das vencedoras, e `prever` devolve a
  entrada). Medir "explica alem de copia" e depois perguntar quanto `copia`
  explica alem de si mesmo e erro de categoria. O numero COM ele e reportado
  do mesmo jeito, para nao esconder nada.

PREVISAO: A1 a 0,45, A2 a 0,40, A3 a 0,15. Mecanismo para A1: codec que captura
relacao estrutural (d4, mapa_cor, periodo) funciona em qualquer grade, e essas
sao tambem as relacoes comuns no ARC. Mecanismo para A2: o ARC e um mundo de
objetos e cor, e os codecs mais usados depois de `copia` (programa 26%,
cor_por_prop 10%) sao especificos disso — nao ha razao para eles brilharem em
espectro de giroscopio.
"""

from __future__ import annotations

import collections
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys.descricao import CODECS, custo, dims
from novaphys.modelo_mundo import (MUNDOS, N_CRUS, N_MIN, OPERADORES,
                                   normaliza)

CACHE_PILHAS = "resultados/loja_fragmentos.json"
N_PARES = 8          # pares por celula (dominio x operador)
MAX_MODELOS = 6      # modelos examinados por codec — orcamento casado
MARGEM = 0.05        # ganho minimo sobre a base para contar como "funciona"


def _grade(a: np.ndarray):
    """np.ndarray 2-D de inteiros -> Grid do ARC (tupla de tuplas)."""
    return tuple(tuple(int(v) for v in linha) for linha in a)


def pares_do_mundo(nome: str, op: str, n: int = N_PARES):
    """Pares (entrada, saida) de um mundo real sob um operador partilhado."""
    fn = MUNDOS[nome]
    crus = fn(N_CRUS)
    ns = [x for x in (normaliza(a) for a in crus) if x is not None][:n]
    fo = OPERADORES[op]
    out = []
    for s in ns:
        d = fo(s)
        if d.shape == s.shape:
            out.append((_grade(s), _grade(d)))
    return out


def explica(codec, treino) -> float:
    """
    Fracao de celulas que o codec explica, pela REGUA DO PROPRIO NUCLEO.

    Usa `custo`, que e a mesma funcao que decide as 105 do ARC — nao uma medida
    paralela inventada aqui. `err` e o numero de celulas nao explicadas ou
    erradas; o total e a soma de h*w sobre os pares.
    """
    total = sum(dims(b)[0] * dims(b)[1] for _a, b in treino)
    if total == 0:
        return 0.0
    try:
        ms = list(codec.propor(treino) or [])[:MAX_MODELOS]
    except Exception:
        return 0.0
    melhor = None
    for m in ms:
        try:
            _bits, err = custo([(codec, m)], treino)
        except Exception:
            continue
        if melhor is None or err < melhor:
            melhor = err
    if melhor is None:
        return 0.0
    return 1.0 - melhor / total


def preco(codec, treino) -> float:
    """Bits do modelo mais barato — para parcializar o confundidor de PRECO."""
    try:
        ms = list(codec.propor(treino) or [])[:MAX_MODELOS]
    except Exception:
        return float("nan")
    bs = []
    for m in ms:
        try:
            bs.append(codec.bits(m))
        except Exception:
            continue
    return min(bs) if bs else float("nan")


def _rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Correlacao de ordem de Spearman, com empates por posto medio."""
    def postos(v):
        ordem = sorted(range(len(v)), key=lambda i: v[i])
        p = [0.0] * len(v)
        i = 0
        while i < len(ordem):
            j = i
            while j + 1 < len(ordem) and v[ordem[j + 1]] == v[ordem[i]]:
                j += 1
            m = (i + j) / 2.0
            for k in range(i, j + 1):
                p[ordem[k]] = m
            i = j + 1
        return p
    pa, pb = postos(list(a)), postos(list(b))
    ma, mb = statistics.mean(pa), statistics.mean(pb)
    num = sum((x - ma) * (y - mb) for x, y in zip(pa, pb))
    den = (sum((x - ma) ** 2 for x in pa)
           * sum((y - mb) ** 2 for y in pb)) ** 0.5
    return num / den if den else 0.0


def _parcial(x, y, z) -> float:
    """rho(x,y) controlando z — o confundidor de preco."""
    rxy, rxz, ryz = _rho(x, y), _rho(x, z), _rho(y, z)
    den = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    return (rxy - rxz * ryz) / den if den > 1e-12 else float("nan")


def experimento() -> Dict:
    print("=" * 78)
    print("SEGUNDA VIA — evidencia entre mundos diz algo sobre o ARC?")
    print("=" * 78)

    if not os.path.exists(CACHE_PILHAS):
        print("  sem %s — nao ha como medir uso no ARC." % CACHE_PILHAS)
        return {"veredito": "sem cache"}
    pilhas = json.load(open(CACHE_PILHAS))
    uso = collections.Counter()
    for it in pilhas:
        for x in set(e for e in it["estrutura"] if isinstance(e, str)):
            uso[x] += 1
    print("  uso no ARC: %d pilhas vencedoras em cache" % len(pilhas))

    mundos = []
    for nome in MUNDOS:
        try:
            crus = MUNDOS[nome](N_CRUS)
            ns = [x for x in (normaliza(a) for a in crus) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            mundos.append(nome)
    print("  mundos: %s" % ", ".join(mundos))
    print("  %d codecs REAIS do ARC, rodados sem adaptacao nesses mundos\n"
          % len(CODECS))

    # pares por (mundo, operador), montados uma vez
    treinos: Dict[Tuple[str, str], list] = {}
    for m in mundos:
        for op in OPERADORES:
            t = pares_do_mundo(m, op)
            if t:
                treinos[(m, op)] = t

    porn = {c.nome: c for c in CODECS}
    base = {}
    for ch, t in treinos.items():
        base[ch] = explica(porn["copia"], t)

    linhas = []
    print("  %-14s %8s %12s %10s" % ("codec", "bits", "corrobora", "uso ARC"))
    print("  " + "-" * 48)
    for c in CODECS:
        mundos_ok = set()
        precos = []
        for (m, op), t in treinos.items():
            e_ = explica(c, t)
            if e_ > base[(m, op)] + MARGEM:
                mundos_ok.add(m)
            p = preco(c, t)
            if p == p:
                precos.append(p)
        pm = statistics.median(precos) if precos else float("nan")
        linhas.append((c.nome, pm, len(mundos_ok), uso.get(c.nome, 0)))
        print("  %-14s %8.1f %12d %10d"
              % (c.nome, pm, len(mundos_ok), uso.get(c.nome, 0)), flush=True)

    # ---- A0: a medida tem resolucao?
    cor = [l[2] for l in linhas]
    print("\n  A0 corroboracao: min %d, max %d, distintos %d"
          % (min(cor), max(cor), len(set(cor))))
    if len(set(cor)) < 3:
        print("  A0 FALHOU: a medida nao separa os codecs. Rodada invalida.")
        return {"veredito": "A0 falhou", "linhas": linhas}
    print("     . separa; segue.")

    # ---- A1: correlaciona com uso no ARC?
    sem_copia = [l for l in linhas if l[0] != "copia"]
    x = [l[2] for l in sem_copia]
    y = [l[3] for l in sem_copia]
    z = [l[1] if l[1] == l[1] else 0.0 for l in sem_copia]
    r = _rho(x, y)
    r_todos = _rho([l[2] for l in linhas], [l[3] for l in linhas])
    rp = _parcial(x, y, z)
    print("\n  rho(corroboracao, uso no ARC) sem copia : %+.2f  (n=%d)"
          % (r, len(x)))
    print("  rho COM copia (reportado, nao usado)     : %+.2f" % r_todos)
    print("  rho parcial, controlando BITS do codec   : %+.2f" % rp)
    print("     (se o rho morre ao controlar bits, o sinal era PRECO)")

    print("\n" + "=" * 78)
    if r >= 0.40:
        ver = "A1"
        print("A1 . EVIDENCIA ENTRE MUNDOS CARREGA INFORMACAO SOBRE O ARC")
        print("     (rho %+.2f). A segunda via tem chao — vale construir." % r)
        if abs(rp) < 0.2:
            print("     MAS o rho parcial e %+.2f: o sinal era PRECO, nao" % rp)
            print("     corroboracao. A via seria 'admita o que e barato'.")
    elif r <= -0.40:
        ver = "A3"
        print("A3 . INVERSAO: o que atravessa mundos e justamente o que NAO")
        print("     serve no ARC (rho %+.2f). Achado, nao fracasso." % r)
    else:
        ver = "A2"
        print("A2 . evidencia entre mundos NAO diz nada sobre o ARC (rho")
        print("     %+.2f). A segunda via e infundada; paro aqui em vez de" % r)
        print("     construir por estetica. Decimo terceiro nulo.")
    return {"veredito": ver, "rho": r, "rho_com_copia": r_todos,
            "rho_parcial": rp, "linhas": linhas}


if __name__ == "__main__":
    r = experimento()
    json.dump(r, open("resultados/segunda_via.json", "w"), indent=1,
              ensure_ascii=False, default=str)
