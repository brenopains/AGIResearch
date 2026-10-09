"""
novaphys.mente_inteira — a mente com as pecas no ciclo, num quadro-negro comum
================================================================================

A COBRANCA, E ELA ESTAVA CERTA
--------------------------------
O usuario, em 24/09: "estavamos tentando garimpar no ARC-2 como o resto do mercado
sem usar 99% do que tinhamos feito e registrado no ARQUITETURA.md?" Sim. A "mente"
que foi ao exame era nucleo + Sistema 1 + sonho. O invariante que acha estrutura
verdadeira em 342/400 tarefas, o critico com zero alarme falso, a digestora, o HDC
sob demanda, os motores de deducao — nada disso estava no ciclo. Medido por
importacao: zero modulos da mente importavam digestora, busca sob demanda,
acervo, atlas ou linguagem.

A FORMA: QUADRO-NEGRO
-----------------------
Cada departamento da fabrica escreve suas respostas candidatas num quadro comum;
o critico veta as que violam o que se conserva nos exemplos; o arbitro escolhe as
duas tentativas que a regra do ARC permite. Nenhum departamento precisa saber dos
outros — e isso que deixa plugar peca nova sem reescrever a mente.

  DEPARTAMENTO   O QUE FAZ                                     MEDIDA QUE O SUSTENTA
  s2             nucleo: busca pilha de codecs, MDL,          [[arc-numero-oficial]] 105/400
                 simula no treino antes de responder
  construtor     DIGESTORA (objetos e relacoes) -> HDC        [[construtor]] (medindo)
                 (alinha, guarda e puxa sob demanda) ->
                 constroi a regra comparando os exemplos ->
                 simula
  celobj         deducao: quais celulas de fundo viram a      censo: 37 tarefas do perfil
                 cor C, lida do diff
  s1             intuicao renascida do sonho das regras do    [[sonho-ensina]] (D1, +7)
                 nucleo, com a maior simetria que os
                 exemplos permitem
  CRITICO        a invariante mais simples do treino julga    `critica_arc`: alarme falso
                 cada resposta candidata, SEM gabarito        0%, precisao 100%
  ARBITRO        regras explicitas antes de intuicao; o que
                 o critico vetou nao entra; 2 tentativas

O QUE AINDA NAO ESTA NO CICLO, E POR QUE — dito, nao escondido
----------------------------------------------------------------
  revisao por surpresa (`autonomo`)   e de mundo SEQUENCIAL (leis que mudam no
                                      tempo); no ARC cada tarefa e isolada. Entra
                                      quando a mente viver um fluxo de tarefas.
  atlas / credito / corroboracao      e memoria ENTRE mundos; aqui ha um mundo so.
  linguagem aterrada                  precisa de narrador; ARC nao tem fala.
  descoberta de leis (orbitas)        a operacao dela JA entrou: e o critico e o
                                      construtor ("ache o que se conserva").

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  Q0  PRE-CONDICAO. O critico, rodando em todas as candidatas, mantem alarme
      falso <= 10% nas candidatas CERTAS. Se ele vetar resposta certa com
      frequencia, o arbitro fica pior que sem ele.
  Q1  A MENTE INTEIRA SUPERA A MELHOR PECA: o arbitro resolve mais que o melhor
      departamento sozinho (em qualquer conjunto de avaliacao).
  Q2  e igual a melhor peca -> as pecas nao se complementam; plugar nao somou.

PREVISAO: Q1 a 0,65 no ARC-AGI-1 de avaliacao (os conjuntos resolvidos por s2 e
s1 ja se mostraram disjuntos no treino) e 0,35 no ARC-AGI-2 (tao pouco resolvido
que qualquer soma e ruido).
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from typing import Dict, List, Optional

import numpy as np

QUADRO = "resultados/quadro"
PASTAS = {
    "arc2_aval": r"D:\agi_dados\arc_agi_2\data\evaluation",
    "arc1_aval": None,          # preenchido por eter.PASTA_AVAL
    "arc1_treino": None,        # preenchido por promessa.PASTA
}
PRIORIDADE = ("s2", "construtor", "celobj", "s1")     # regra explicita antes de intuicao
TIPO = {"s2": "regra", "construtor": "regra", "celobj": "regra", "s1": "intuicao"}
PRAZO_S = 60


def _pasta(conjunto: str) -> str:
    if conjunto == "arc1_aval":
        from novaphys.eter import PASTA_AVAL
        return PASTA_AVAL
    if conjunto == "arc1_treino":
        from novaphys.promessa import PASTA
        return PASTA
    return PASTAS[conjunto]


def _arq(conjunto, fonte, fatia=-1):
    d = os.path.join(QUADRO, conjunto)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fonte + ("" if fatia < 0 else "_f%d" % fatia) + ".jsonl")


def _grade(g):
    return [list(map(int, l)) for l in g]


# ================================================= os departamentos escrevem
def _s2(tr, entradas):
    import novaphys.descricao as D
    orig = D.custo
    prazo = time.time() + PRAZO_S

    def com_prazo(pilha, treino):
        if time.time() > prazo:
            raise TimeoutError
        return orig(pilha, treino)
    D.custo = com_prazo
    try:
        fm, pilha, err = D.montar(tr, teto=4000)
        if err != 0 or not pilha:
            return None
        r = [D.aplicar((fm, pilha), x) for x in entradas]
        return None if any(x is None for x in r) else [_grade(x) for x in r]
    except Exception:
        return None
    finally:
        D.custo = orig


_MEM = None


def _construtor(tr, entradas):
    global _MEM
    from novaphys.construtor import Memoria, resolve
    if _MEM is None:
        _MEM = Memoria()                  # a memoria HDC atravessa as tarefas
    r, _d = resolve(tr, entradas, _MEM)
    return r


def _celobj(tr, entradas):
    from novaphys import celobj
    try:
        regra = celobj.deduzir(tr)
        if regra is None:
            return None
        r = [celobj.aplicar(regra, x) for x in entradas]
        return None if any(x is None for x in r) else [_grade(x) for x in r]
    except Exception:
        return None


_PESOS = None


def _s1(tr, entradas):
    global _PESOS
    from novaphys.arc2_mente import pesos_do_sonho
    from novaphys.mente_unica import Problema, sistema1
    if _PESOS is None:
        _PESOS = pesos_do_sonho()
    p = Problema("x", tr, [(x, x) for x in entradas])
    r, _rede, _g, _fm = sistema1(_PESOS, p)
    return r


DEPARTAMENTOS = {"s2": _s2, "construtor": _construtor, "celobj": _celobj, "s1": _s1}


def escreve(conjunto: str, fonte: str, fatia: int = -1, total: int = 1) -> None:
    """Um departamento escreve no quadro. Retomavel e fatiavel."""
    from novaphys.arc import carregar
    arq = _arq(conjunto, fonte, fatia)
    feito = set()
    if os.path.exists(arq):
        feito = {json.loads(l)["nome"] for l in open(arq, encoding="utf-8")}
    T = list(carregar(_pasta(conjunto)))
    if fatia >= 0:
        T = [t for t in T if zlib.crc32(t[0].encode()) % total == fatia]
    f = DEPARTAMENTOS[fonte]
    t0 = time.time()
    print("[%s/%s] %d tarefas, ja feitas %d" % (conjunto, fonte, len(T), len(feito)),
          flush=True)
    for k, (nome, tr, te) in enumerate(T):
        if nome in feito:
            continue
        try:
            r = f(tr, [x for x, _ in te])
        except Exception:
            r = None
        with open(arq, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"nome": nome, "resp": r}) + "\n")
        if (k + 1) % 20 == 0:
            print("  %3d/%d | %.0f min" % (k + 1, len(T), (time.time() - t0) / 60),
                  flush=True)
    print("[%s/%s] terminado | %.0f min" % (conjunto, fonte, (time.time() - t0) / 60),
          flush=True)


# ============================================================ o critico
def invariante(tr):
    """A invariante mais simples do treino (uma vez por tarefa)."""
    from novaphys import invariante_arc as IA
    try:
        ac = IA.acha(tr, IA.candidatos(list(range(len(IA.NOMES)))))
        return ac[0][2] if ac else None
    except Exception:
        return None


def veta(c, entrada, resposta) -> bool:
    """
    A invariante mais simples do treino julga a resposta, sem gabarito.
    `critica_arc` mediu: alarme falso 0%, precisao 100%, pega ~1/5 das erradas.
    """
    from novaphys import invariante_arc as IA
    if c is None:
        return False
    try:
        x = tuple(tuple(l) for l in entrada)
        p = tuple(tuple(l) for l in resposta)
        return IA.Q(IA.grandezas(p), c) != IA.Q(IA.grandezas(x), c)
    except Exception:
        return False


# ============================================================ o arbitro
def arbitra(conjunto: str) -> Dict:
    import glob
    from novaphys.arc import carregar
    quadro: Dict[str, Dict[str, list]] = {}
    for fonte in PRIORIDADE:
        for a in glob.glob(os.path.join(QUADRO, conjunto, fonte + "*.jsonl")):
            base = os.path.basename(a)[:-6]              # so fonte.jsonl ou fonte_fK.jsonl:
            if base != fonte and not (base.startswith(fonte + "_f")   # backup antigo
                                      and base[len(fonte) + 2:].isdigit()):  # nao vota
                continue
            for l in open(a, encoding="utf-8"):
                d = json.loads(l)
                quadro.setdefault(d["nome"], {})[fonte] = d["resp"]
    T = {n: (tr, te) for n, tr, te in carregar(_pasta(conjunto))}
    solo = {f: 0 for f in PRIORIDADE}
    veto = {"certa_vetada": 0, "errada_vetada": 0, "certa_ok": 0, "errada_ok": 0}
    mente, resolvidas = 0, []
    for nome, (tr, te) in T.items():
        gab = [y for _x, y in te]
        cands = []
        c_inv = invariante(tr) if quadro.get(nome) else None
        for fonte in PRIORIDADE:
            r = quadro.get(nome, {}).get(fonte)
            if r is None:
                continue
            certa = all(np.array_equal(np.array(ri), np.array(y)) for ri, y in zip(r, gab))
            solo[fonte] += int(certa)
            vetada = any(veta(c_inv, x, ri) for (x, _y), ri in zip(te, r))
            veto[("certa_" if certa else "errada_") + ("vetada" if vetada else "ok")] += 1
            if not vetada and not any(r == c for _f, c, _k in cands):
                cands.append((fonte, r, certa))
        duas = cands[:2]
        if any(k for _f, _r, k in duas):
            mente += 1
            resolvidas.append((nome, [f for f, _r, k in duas if k][0]))
    n = len(T)
    melhor = max(solo.values()) if solo else 0
    print("=" * 72)
    print("MENTE INTEIRA — %s (%d tarefas, %d com alguma candidata no quadro)"
          % (conjunto, n, len(quadro)))
    print("=" * 72)
    for f in PRIORIDADE:
        print("  %-11s sozinho  %3d  (%5.1f%%)" % (f, solo[f], 100 * solo[f] / n))
    print("  %-11s          %3d  (%5.1f%%)   <- 2 tentativas, critico + arbitro"
          % ("MENTE", mente, 100 * mente / n))
    fa = veto["certa_vetada"] / max(veto["certa_vetada"] + veto["certa_ok"], 1)
    de = veto["errada_vetada"] / max(veto["errada_vetada"] + veto["errada_ok"], 1)
    print("\n  critico: vetou %d certas (alarme falso %.1f%%) e %d erradas (pega %.1f%%)"
          % (veto["certa_vetada"], 100 * fa, veto["errada_vetada"], 100 * de))
    print("  Q0 %s alarme falso <= 10%%" % ("." if fa <= 0.10 else "FALHOU ."))
    print("  Q1 %s mente (%d) %s melhor peca sozinha (%d)"
          % ("." if mente > melhor else "FALHOU .", mente,
             ">" if mente > melhor else "<=", melhor))
    r = {"conjunto": conjunto, "n": n, "solo": solo, "mente": mente,
         "veto": veto, "resolvidas": resolvidas}
    json.dump(r, open(os.path.join(QUADRO, conjunto, "arbitro.json"), "w"), indent=1)
    return r


if __name__ == "__main__":
    if sys.argv[1] == "arbitra":
        arbitra(sys.argv[2])
    else:
        conj, fonte = sys.argv[1], sys.argv[2]
        fatia = int(sys.argv[3]) if len(sys.argv) > 3 else -1
        total = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        escreve(conj, fonte, fatia, total)
