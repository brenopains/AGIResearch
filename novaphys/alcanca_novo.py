"""
novaphys.alcanca_novo — a ladeira alcanca material que os BITS rejeitam?
=========================================================================

O TESTE DECISIVO, E ELE FECHA UMA PERGUNTA DE MESES
-----------------------------------------------------
[[gradiente-na-admissao]] mediu que amolecer a reivindicacao winner-take-all da
uma derivada em w=0 que separa codec util de placebo de mesma cobertura em 4,8
desvios. E deixou o limite explicito:

    "o rotulo de 'util' veio das pilhas vencedoras em cache, QUE FORAM
     PRODUZIDAS PELO CRITERIO DE BITS. Entao o que esta medido e que o
     gradiente RECUPERA o que os bits ja escolheram — nao que ele acha o
     que os bits REJEITAM."

Achar o rejeitado era o objetivo inteiro. Isto testa exatamente isso.

O MATERIAL REJEITADO JA EXISTE, MEDIDO, COM PLACEBO CASADO
------------------------------------------------------------
Nao precisei inventar nada: `propriedade_inventada.py` ja tem 80 propriedades
GERADAS (85 no vocabulario, 5 minhas de base) e dois pares de codecs:

    cor_por_prop_mais / apaga_prop_mais       sobre as propriedades INVENTADAS
    cor_por_prop_placebo / apaga_prop_placebo sobre propriedades ARBITRARIAS

E ja esta medido que os bits os rejeitam: [[o-criterio-nao-seleciona]] (os
vocabularios novos NUNCA entram em pilha vencedora, e perdem em BITS, nao em
orcamento) e [[vocabulario-grande-piora]] (com as 85 propriedades o sistema
PERDE 3 tarefas). O par inventado/placebo e o controle que decide se o gradiente
esta vendo ESTRUTURA ou so vendo "mais um codec".

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  R0  PRE-CONDICAO. Os quatro codecs produzem previsao nas tarefas (nao voltam
      vazios). Se o material rejeitado nem preve celula, nao ha o que admitir e
      a pergunta nao chega a ser feita.
  R1  A LADEIRA ALCANCA O NOVO: a derivada em w=0 e mais negativa para os
      codecs INVENTADOS do que para os PLACEBO casados, >= 2 desvios pareados
      -> o gradiente admite material que o criterio de bits rejeita. O canal de
         importacao deixa de depender de mim, e ai vale medir se PAGA (R4).
  R2  inventado ~ placebo -> a ladeira existe mas nao alcanca material novo:
      ela so reordena o que os bits ja aprovam. A parede mudou de lugar, nao
      caiu, e [[gradiente-na-admissao]] vira um resultado sobre recuperacao.
  R3  placebo MAIS negativo que inventado -> a ladeira prefere o arbitrario;
      seria o pior desfecho e o mais informativo.

  E SO SE R1 PASSAR:
  R4  PAGA: `montar` com os 23 codecs mais os candidatos escolhidos PELO
      GRADIENTE resolve >= +2 tarefas sobre a base, com orcamento casado, e o
      braco PLACEBO (mesmo numero de codecs arbitrarios) ganha <= 0.
      Referencia dura: [[vocabulario-grande-piora]] mediu que jogar o
      vocabulario INTEIRO dentro PERDE 3 tarefas. Se a selecao por gradiente
      virar -3 em positivo, o ganho e da SELECAO, que e onde o gargalo estava.

PREVISAO: R1 a 0,50, R2 a 0,35, R3 a 0,15. Mecanismo para R1: o gradiente nao
olha bits, so acerto de celula; se as propriedades inventadas carregam estrutura
real (e [[vocabulario-grande-piora]] descreveu as geradas como "5 operadores
novos, categoria metrica que eu nao tinha"), o acerto aparece. Mecanismo para
R2: as propriedades inventadas podem particionar os objetos quase como as
minhas, e ai os codecs preveem as MESMAS celulas e o gradiente nao tem o que
separar. Risco de medida: os codecs `mais` tem 85 propriedades para varrer e o
`propor` deles e caro — se eu cortar modelos demais por orcamento, corto
justamente o modelo bom e leio R2 por pobreza de busca. Por isso o teto de
modelos e o MESMO para inventado e placebo, e esta dito.
"""

from __future__ import annotations

import json
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, dims
from novaphys.gradiente import (_monta_tensores, _perda_e_grad, _previsoes,
                                ajusta)
from novaphys.promessa import PASTA
from novaphys.propriedade_inventada import (ApagaPorPropMais,
                                            ApagaPorPropPlacebo,
                                            CorPorPropMais,
                                            CorPorPropPlacebo)

MAX_MODELOS = 4      # o MESMO teto para inventado e placebo — orcamento casado
N_TAREFAS = 150


def _prev_de(codec, treino, max_modelos: int = MAX_MODELOS):
    """Melhor previsao por celula deste codec, pelo mesmo criterio do nucleo."""
    try:
        ms = list(codec.propor(treino) or [])[:max_modelos]
    except Exception:
        return None
    melhor, n_melhor = None, -1
    for m in ms:
        try:
            ps = [codec.prever(m, a) for a, _b in treino]
        except Exception:
            continue
        n = sum(1 for p, (_a, b) in zip(ps, treino)
                for (i, j), v in p.items()
                if i < dims(b)[0] and j < dims(b)[1] and v == b[i][j])
        if n > n_melhor:
            melhor, n_melhor = ps, n
    return melhor


def experimento(n_tarefas: int = N_TAREFAS, semente: int = 20260923) -> Dict:
    rnd = random.Random(semente)
    tarefas = [(n, tr, te) for n, tr, te in carregar(PASTA)][:n_tarefas]

    cands = [("INVENTADO:cor", CorPorPropMais()),
             ("INVENTADO:apaga", ApagaPorPropMais()),
             ("PLACEBO:cor", CorPorPropPlacebo()),
             ("PLACEBO:apaga", ApagaPorPropPlacebo())]

    print("=" * 78)
    print("ALCANCA NOVO — a ladeira pega o que os BITS rejeitam?")
    print("=" * 78)
    print("  candidatos: 2 sobre propriedades INVENTADAS, 2 PLACEBO casados")
    print("  teto de modelos IGUAL nos quatro (%d) — orcamento casado\n"
          % MAX_MODELOS)

    grad = defaultdict(list)
    pareado = []
    vivos = defaultdict(int)
    usadas = 0

    for k, (nome, tr, _te) in enumerate(tarefas):
        try:
            fms = forma.propor(tr)
            fm = fms[0] if fms else None
            reb = [(forma.base(fm, a), b) for a, b in tr] if fm else tr
            if any(g is None or dims(g) != dims(b) for g, b in reb):
                reb = tr
        except Exception:
            reb = tr
        try:
            base_nps = _previsoes(reb)
        except Exception:
            continue
        if len(base_nps) < 3:
            continue

        extras = []
        for cn, c in cands:
            ps = _prev_de(c, reb)
            if ps and any(p for p in ps):
                extras.append((cn, ps))
                vivos[cn] += 1
        # DEFEITO MEU, CORRIGIDO: eu exigia os QUATRO vivos na MESMA tarefa e
        # so 1 de 60 satisfazia. O pre-registro dizia "os quatro codecs produzem
        # previsao" — e produzem (28/13/16/13 de 60). Exigir coincidencia na
        # mesma tarefa era invencao minha no codigo, nao barra escrita. E o
        # pareamento correto e por TIPO de qualquer forma: cor contra cor,
        # apaga contra apaga, que e comparar codecs da mesma familia.
        tem = {c.split(":")[1] for c, _ in extras}
        pares = [t for t in tem
                 if ("INVENTADO:" + t) in dict(extras)
                 and ("PLACEBO:" + t) in dict(extras)]
        if not pares:
            continue
        usadas += 1

        nps = base_nps + extras
        votos, alvos = _monta_tensores(nps, reb)
        d = {}
        for ci in range(len(base_nps), len(nps)):
            cn = nps[ci][0]
            fix = np.zeros(len(nps), dtype=bool)
            fix[ci] = True
            w2, _ = ajusta(votos, alvos, fixos=fix)
            w2[ci] = 0.0
            _L, g2 = _perda_e_grad(votos, alvos, w2)
            d[cn] = float(g2[ci])
            grad[cn].append(float(g2[ci]))
        for t in pares:                    # pareado POR TIPO
            i_, p_ = d.get("INVENTADO:" + t), d.get("PLACEBO:" + t)
            if i_ is not None and p_ is not None:
                pareado.append(p_ - i_)    # positivo = inventado desce mais
        if (k + 1) % 15 == 0:
            print("    %2d/%d tarefas varridas (%d usadas)"
                  % (k + 1, len(tarefas), usadas), flush=True)

    print("\n  R0 os quatro codecs preveem em quantas tarefas:")
    for cn, _c in cands:
        print("     %-18s %d" % (cn, vivos[cn]))
    print("     tarefas com ao menos UM par (inventado+placebo do mesmo"
          " tipo): %d" % usadas)
    if usadas < 10:
        print("  R0 FALHOU: material rejeitado nem preve. Pergunta nao se faz.")
        return {"veredito": "R0 falhou", "usadas": usadas}
    print("     . preveem; segue.\n")

    print("  derivada dL/dw em w=0 (mais NEGATIVA = mais vale admitir)")
    for cn, _c in cands:
        v = grad[cn]
        if v:
            print("     %-18s %+.5f +- %.5f  (n=%d)"
                  % (cn, statistics.mean(v),
                     statistics.pstdev(v) / (len(v) ** 0.5), len(v)))
    m = statistics.mean(pareado)
    s = statistics.pstdev(pareado) / (len(pareado) ** 0.5) or 1e-9
    sep = m / s
    print("\n  separacao PAREADA por tarefa (placebo - inventado): %+.5f +- %.5f"
          % (m, s))
    print("  = %.1f desvios  (n=%d tarefas)" % (sep, len(pareado)))

    print("\n" + "=" * 78)
    if sep >= 2.0:
        ver = "R1"
        print("R1 . A LADEIRA ALCANCA O NOVO (%.1f desvios). O gradiente" % sep)
        print("     admite material que o criterio de BITS rejeita.")
        print("     Agora vale medir se PAGA (R4).")
    elif sep <= -2.0:
        ver = "R3"
        print("R3 . A LADEIRA PREFERE O ARBITRARIO (%.1f desvios)." % sep)
        print("     Pior desfecho, e o mais informativo.")
    else:
        ver = "R2"
        print("R2 . a ladeira NAO alcanca material novo (%.1f desvios)." % sep)
        print("     Ela reordena o que os bits ja aprovam. A parede mudou de")
        print("     lugar, nao caiu — e [[gradiente-na-admissao]] passa a ser")
        print("     um resultado sobre RECUPERACAO, nao sobre importacao.")
    return {"veredito": ver, "separacao_desvios": sep, "n": len(pareado),
            "grad": {cn: [statistics.mean(v),
                          statistics.pstdev(v) / (len(v) ** 0.5), len(v)]
                     for cn, v in grad.items() if v},
            "usadas": usadas}


if __name__ == "__main__":
    r = experimento()
    json.dump(r, open("resultados/alcanca_novo.json", "w"), indent=1,
              ensure_ascii=False, default=str)
