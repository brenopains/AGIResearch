"""
novaphys.arc2_mente — a mente inteira no ARC-AGI-2, com a regra oficial
=========================================================================

O CONJUNTO QUE MEDE A AMBICAO
-------------------------------
[[ambicao-do-usuario]]: "resolver problemas que IAs atuais nao conseguem, que so
humanos ate entao se mostraram capazes". O ARC-AGI-2 foi construido com essa
definicao — humanos a 66% na amostra de teste, o melhor placar privado do ARC
Prize 2025 em 24% com computacao enorme. 120 tarefas publicas de avaliacao.

`arc2_nucleo.py` mede o nucleo sozinho (Sistema 2). Isto mede a mente com as
pecas que tem medida positiva:

  SISTEMA 2        o nucleo, biblioteca de base            [[arc-numero-oficial]]
  SISTEMA 1 SONHO  renascido dos sonhos das regras do     [[sonho-ensina]] (D1, +7)
                   nucleo, ajustado em cada tarefa, com a
                   maior simetria que os exemplos permitem [[sistema1-transducao]]
  FORMA + S1       o nucleo decide a geometria da saida,   `mente_unica.sistema1`
                   o Sistema 1 preenche

DUAS TENTATIVAS, A REGRA OFICIAL — e a ordem e decidida SEM gabarito
---------------------------------------------------------------------
  1a  a regra do Sistema 2, se houver pilha que reproduz o treino exato
      (regra exata e verificavel vem primeiro)
  2a  o Sistema 1 renascido do sonho
  Se o Sistema 2 nao tiver regra, as duas tentativas sao do Sistema 1: a do
  sonho e a do Sistema 1 do zero (duas intuicoes com passado diferente).

Para atribuir o ganho, cada fonte tambem e contada sozinha. O gabarito so e
usado pelo experimento, de fora, para corrigir.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
Nao ha barra de sucesso honesta a pre-registrar alem da propria contagem: e o
primeiro numero da mente neste conjunto. O que fica escrito antes e a PREVISAO,
para ser cobrada: MENTE resolve entre 1 e 4 das 120 (1% a 3%). Mecanismo: o
Sistema 1 resolveu 13% do ARC-AGI-1 de mesma dimensao do zero, e o ARC-AGI-2 foi
desenhado para derrubar justamente o que e perceptual raso e o que e busca; o
nucleo sozinho esta em 0 de 40 ate a hora em que isto foi escrito.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time

import numpy as np

PASTA2 = r"D:\agi_dados\arc_agi_2\data"
SAIDA = "resultados/arc_agi_2/mente_%s.jsonl"
PESOS_SONHO = "resultados/arc_agi_2/pesos_sonho.pt"


def pesos_do_sonho():
    """Sistema 1 renascido dos sonhos das regras do nucleo (cache em disco)."""
    import torch
    from novaphys.sonho import placebo_de, pretreina
    if os.path.exists(PESOS_SONHO):
        return torch.load(PESOS_SONHO, map_location="cpu", weights_only=False)
    pares = pickle.load(open("resultados/sonho_pares_nucleo.pkl", "rb"))
    real, _falso = placebo_de(pares, random.Random(1))
    p = pretreina(real)
    os.makedirs(os.path.dirname(PESOS_SONHO), exist_ok=True)
    torch.save({k: v.cpu() for k, v in p.items()}, PESOS_SONHO)
    return p


def rodar(conjunto: str = "evaluation", fatia: int = -1, total: int = 1) -> None:
    from novaphys.arc import carregar
    from novaphys.mente_unica import Problema, sistema1
    import zlib
    arq = SAIDA % (conjunto if fatia < 0 else "%s_f%d" % (conjunto, fatia))
    feito = set()
    if os.path.exists(arq):
        feito = {json.loads(l)["nome"] for l in open(arq, encoding="utf-8")}
    pesos = pesos_do_sonho()                        # antes: o nucleo termina enquanto isso
    s2 = {}
    a2 = "resultados/arc_agi_2/nucleo_%s.jsonl" % conjunto
    if os.path.exists(a2):                          # o nucleo ja rodou a parte
        for l in open(a2, encoding="utf-8"):
            d = json.loads(l)
            s2[d["nome"]] = d
    T = [(n, tr, te) for n, tr, te in carregar(os.path.join(PASTA2, conjunto))]
    if fatia >= 0:
        T = [t for t in T if zlib.crc32(t[0].encode()) % total == fatia]
    print("  nucleo lido para %d tarefas" % len(s2), flush=True)
    print("ARC-AGI-2 %s: %d tarefas nesta fatia | ja feitas %d"
          % (conjunto, len(T), len(feito)), flush=True)
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(T):
        if nome in feito:
            continue
        p = Problema(nome, tr, te)
        n2 = s2.get(nome, {}).get("ok", False)       # o nucleo acertou?
        tem_regra = s2.get(nome, {}).get("err") == 0
        # O POR QUE da falha e o que localiza o gargalo, nao so o acerto:
        #   S2  err==0 e errou -> AMBIGUO;  err>0 -> INEXPRIMIVEL;  None -> prazo
        #   S1  nao ajusta os exemplos -> falta CAPACIDADE
        #       ajusta e erra o teste  -> DECOROU em vez de abstrair
        r_sonho, rede_s, _g, fm_s = sistema1(pesos, p)
        ok_sonho = p.corrige(r_sonho)
        r_zero, rede_z, _g, fm_z = sistema1(None, p)
        ok_zero = p.corrige(r_zero)
        mesma = all(len(a) == len(b) and len(a[0]) == len(b[0]) for a, b in tr)
        if tem_regra:
            mente = n2 or ok_sonho                     # 1a regra, 2a sonho
        else:
            mente = ok_sonho or ok_zero                # duas intuicoes
        with open(arq, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"nome": nome, "nucleo": n2, "s1_sonho": ok_sonho,
                                 "s1_zero": ok_zero, "mente": mente,
                                 "s2_err": s2.get(nome, {}).get("err"),
                                 "mesma_dim": mesma,
                                 "s1_sonho_ajusta": rede_s is not None,
                                 "s1_zero_ajusta": rede_z is not None,
                                 "forma": str(fm_s[0]) if fm_s else None}) + "\n")
        if (k + 1) % 10 == 0:
            print("  %3d/%d | %.0f min" % (k + 1, len(T), (time.time() - t0) / 60),
                  flush=True)
    print("fatia terminada | %.0f min" % ((time.time() - t0) / 60), flush=True)


def resumo(conjunto: str = "evaluation", total: int = 1) -> None:
    import glob
    linhas = []
    for a in glob.glob("resultados/arc_agi_2/mente_%s*.jsonl" % conjunto):
        linhas += [json.loads(l) for l in open(a, encoding="utf-8")]
    n = len(linhas)
    c = {k: sum(d[k] for d in linhas) for k in ("nucleo", "s1_zero", "s1_sonho", "mente")}
    print("=" * 70)
    print("ARC-AGI-2 %s — %d tarefas" % (conjunto, n))
    for k, v in c.items():
        print("  %-9s %3d  (%.1f%%)" % (k, v, 100 * v / max(n, 1)))
    print("  humanos (amostra de teste do ARC Prize): 66%")
    print("  resolvidas pela mente: %s" % sorted(d["nome"] for d in linhas if d["mente"]))

    # O MAPA DAS FALHAS: onde, exatamente, cada tarefa nao resolvida trava.
    from collections import Counter
    def s2_estado(d):
        e = d.get("s2_err")
        return "PRAZO" if e is None else ("AMBIGUO" if e == 0 else "INEXPRIMIVEL")
    def s1_estado(d):
        if "s1_sonho_ajusta" not in d:
            return "?"
        return "DECOROU" if d["s1_sonho_ajusta"] else "NAO_AJUSTA"
    falhas = [d for d in linhas if not d["mente"]]
    tab = Counter((s2_estado(d), s1_estado(d)) for d in falhas)
    print("\n  MAPA DAS FALHAS (%d tarefas nao resolvidas)" % len(falhas))
    print("  %-14s %11s %11s" % ("Sistema 2 \\ S1", "DECOROU", "NAO_AJUSTA"))
    for e2 in ("INEXPRIMIVEL", "AMBIGUO", "PRAZO"):
        print("  %-14s %11d %11d" % (e2, tab[(e2, "DECOROU")], tab[(e2, "NAO_AJUSTA")]))
    dim = Counter(("mesma" if d.get("mesma_dim") else "muda") for d in falhas)
    print("  dimensao: %s" % dict(dim))
    print("  DECOROU    = o Sistema 1 reproduz os exemplos e erra o teste: nao abstraiu")
    print("  NAO_AJUSTA = nem os exemplos ele reproduz: falta capacidade/forma")


if __name__ == "__main__":
    if "--resumo" in sys.argv:
        resumo()
    else:
        f = int(sys.argv[1]) if len(sys.argv) > 1 else -1
        t = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        rodar("evaluation", f, t)
