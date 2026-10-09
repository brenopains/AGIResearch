"""
novaphys.paga — R4: admitir por gradiente resolve MAIS tarefas?
=================================================================

A PERGUNTA QUE FECHA A CADEIA
-------------------------------
[[gradiente-na-admissao]]  ha ladeira e ela aponta para os uteis (4,8 desvios)
[[gradiente-fora-da-amostra]]  e ela ALCANCA material que os bits rejeitam
                               (2,9 desvios) — mas so avaliada fora da amostra
Falta o unico que decide: **isso resolve mais tarefas do ARC?**

Referencia dura, e ela e contraria: [[vocabulario-grande-piora]] mediu que
jogar o vocabulario inteiro dentro **PERDE 3 tarefas**. Se selecao por gradiente
virar esse -3 em positivo, o ganho e da SELECAO — exatamente onde
[[o-criterio-nao-seleciona]] disse que estava o gargalo.

O VAZAMENTO QUE EU QUASE COMETI, E COMO FICA EVITADO
------------------------------------------------------
O gradiente precisa de avaliacao FORA DA AMOSTRA — foi essa a licao de
[[gradiente-fora-da-amostra]]. Mas usar os pares de TESTE para decidir o que
admitir seria vazamento: teste e onde eu meco se a tarefa foi resolvida. Entao a
selecao e **deixa-um-de-fora DENTRO DO TREINO**: o modelo do candidato e
ajustado em `treino \\ {t}` e o gradiente dele e medido em `t`, para cada par t.
O teste nunca e tocado ate a hora de conferir a resposta.

QUATRO BRACOS, ORCAMENTO CASADO
---------------------------------
  BASE       os 23 codecs do nucleo
  TUDO       23 + os 2 codecs inventados, SEMPRE (sem selecao). E a reproducao
             de [[vocabulario-grande-piora]] dentro deste desenho, e serve de
             referencia do que "mais vocabulario" custa sozinho.
  GRADIENTE  23 + so os inventados cuja derivada deixa-um-de-fora e negativa
  PLACEBO    23 + os 2 codecs de propriedade ARBITRARIA, sempre

  O mesmo `teto` nos quatro. Bracos com mais codecs gastam mais busca por passo
  e portanto buscam MENOS — isso nao e defeito do desenho, e o custo real de
  ampliar vocabulario, e foi justamente ele que fez o braco "tudo" perder 3
  tarefas no registro antigo.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  R4a PRE-CONDICAO. BASE resolve um numero compativel com o registro (26,2% em
      [[arc-numero-oficial]]) na fatia usada. Se BASE estiver fora disso, o
      instrumento nao e o mesmo e a comparacao com o -3 antigo nao vale.
  R4  PAGA: GRADIENTE >= BASE + 2 tarefas **e** PLACEBO <= BASE
      -> admitir por gradiente PAGA, o ganho e da selecao, e o canal de
         importacao deixa de depender de mim escrever codec.
  R5  GRADIENTE ~ BASE (diferenca em [-1, +1]) -> a ladeira admite material
      util mas admitir nao muda resolucao: o gargalo esta num terceiro lugar,
      depois da admissao.
  R6  GRADIENTE < BASE -> admitir CUSTA, e a selecao por gradiente nao salva o
      que [[vocabulario-grande-piora]] mediu.
  R7  PLACEBO > BASE -> qualquer codec a mais ajuda e o efeito nao e do
      material; invalida R4 mesmo se R4 passar.

PREVISAO: R5 a 0,45, R4 a 0,30, R6 a 0,20, R7 a 0,05. Mecanismo para R5, e por
isso ele e o favorito: [[paisagem-e-penhasco]] diz que a pilha exata e fragil a
troca de UMA peca; admitir um codec bom no conjunto nao garante que exista pilha
exata que o use, e a busca continua tendo que achar essa pilha num espaco que
cresceu. Admissao e necessaria e pode nao ser suficiente. Mecanismo para R4: as
propriedades inventadas generalizam +0,123 sobre as arbitrarias no teste, entao
ha estrutura real para a pilha aproveitar.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, aplicar, custo, dims, montar
from novaphys.gradiente import _monta_tensores, _perda_e_grad, _previsoes, ajusta
from novaphys.promessa import PASTA
from novaphys.propriedade_inventada import (ApagaPorPropMais,
                                            ApagaPorPropPlacebo,
                                            CorPorPropMais, CorPorPropPlacebo)

MAX_MODELOS = 4
TETO = 4000
N_TAREFAS = 150


def _modelo(c, treino):
    try:
        ms = list(c.propor(treino) or [])[:MAX_MODELOS]
    except Exception:
        return None
    best, n_ = None, -1
    for m in ms:
        try:
            ps = [c.prever(m, a) for a, _b in treino]
        except Exception:
            continue
        n = sum(1 for p, (_a, b) in zip(ps, treino)
                for (i, j), v in p.items()
                if i < dims(b)[0] and j < dims(b)[1] and v == b[i][j])
        if n > n_:
            best, n_ = m, n
    return best


def grad_loo(c, treino) -> Optional[float]:
    """
    Derivada dL/dw do candidato em w=0, por DEIXA-UM-DE-FORA dentro do treino.

    Para cada par t: ajusta o modelo do candidato em `treino \\ {t}`, monta a
    mistura mole com os codecs base (tambem ajustados sem t) e mede o gradiente
    em t. O teste nunca entra — e por isso que isto pode decidir admissao sem
    vazar para a medida de tarefa resolvida.
    """
    if len(treino) < 2:
        return None
    gs = []
    for t in range(len(treino)):
        fora = [treino[t]]
        dentro = [p for i, p in enumerate(treino) if i != t]
        try:
            base = _previsoes(dentro)
        except Exception:
            continue
        if len(base) < 3:
            continue
        base_f = []
        for cn, _ps in base:
            cc = next((x for x in CODECS if x.nome == cn), None)
            if cc is None:
                continue
            m = _modelo(cc, dentro)
            if m is None:
                continue
            try:
                base_f.append((cn, [cc.prever(m, a) for a, _b in fora]))
            except Exception:
                continue
        if len(base_f) < 3:
            continue
        mc = _modelo(c, dentro)
        if mc is None:
            continue
        try:
            pc = [c.prever(mc, a) for a, _b in fora]
        except Exception:
            continue
        if not any(pc):
            continue
        nps = base_f + [("CAND", pc)]
        votos, alvos = _monta_tensores(nps, fora)
        fix = np.zeros(len(nps), dtype=bool)
        fix[-1] = True
        w, _ = ajusta(votos, alvos, fixos=fix)
        w[-1] = 0.0
        _L, g = _perda_e_grad(votos, alvos, w)
        gs.append(float(g[-1]))
    return statistics.mean(gs) if gs else None


def resolve(tr, te, codecs, teto: int) -> bool:
    try:
        fm, pilha, err = montar(tr, codecs=codecs, teto=teto)
    except Exception:
        return False
    if err != 0 or not pilha:
        return False
    try:
        return all(aplicar((fm, pilha), x) == y for x, y in te)
    except Exception:
        return False


def experimento(n_tarefas: int = N_TAREFAS, teto: int = TETO) -> Dict:
    tarefas = list(carregar(PASTA))[:n_tarefas]
    inventados = [CorPorPropMais(), ApagaPorPropMais()]
    placebos = [CorPorPropPlacebo(), ApagaPorPropPlacebo()]
    base = list(CODECS)

    print("=" * 78)
    print("R4 — admitir por gradiente resolve MAIS tarefas?")
    print("=" * 78)
    print("  %d tarefas, teto %d em TODOS os bracos (orcamento casado)" %
          (len(tarefas), teto))
    print("  selecao por DEIXA-UM-DE-FORA dentro do treino; teste nunca tocado\n")

    res = {k: [] for k in ("BASE", "TUDO", "GRADIENTE", "PLACEBO")}
    n_sel = []
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(tarefas):
        ok_base = resolve(tr, te, base, teto)
        ok_tudo = resolve(tr, te, base + inventados, teto)
        if ok_base:
            res["BASE"].append(nome)
        if ok_tudo:
            res["TUDO"].append(nome)
        if resolve(tr, te, base + placebos, teto):
            res["PLACEBO"].append(nome)
        sel = []
        for c in inventados:
            g = grad_loo(c, tr)
            if g is not None and g < 0:
                sel.append(c)
        n_sel.append(len(sel))
        # REAPROVEITAMENTO, e ele e exato e nao aproximacao: com selecao vazia o
        # conjunto do GRADIENTE E o da BASE; com os dois selecionados E o do
        # TUDO. `montar` e deterministico dado conjunto e teto, entao rodar de
        # novo daria o mesmo resultado — so custaria 20s. O montar so acontece
        # no caso intermediario, que e o unico conjunto novo.
        if not sel:
            ok_g = ok_base
        elif len(sel) == len(inventados):
            ok_g = ok_tudo
        else:
            ok_g = resolve(tr, te, base + sel, teto)
        if ok_g:
            res["GRADIENTE"].append(nome)
        if (k + 1) % 5 == 0:
            print("    %3d/%d | BASE=%d TUDO=%d GRAD=%d PLC=%d | %.0f min"
                  % (k + 1, len(tarefas), len(res["BASE"]), len(res["TUDO"]),
                     len(res["GRADIENTE"]), len(res["PLACEBO"]),
                     (time.time() - t0) / 60), flush=True)

    b = len(res["BASE"])
    print("\n" + "=" * 78)
    for m in ("BASE", "TUDO", "GRADIENTE", "PLACEBO"):
        print("  %-10s resolve %3d/%d  (%+d sobre a base)"
              % (m, len(res[m]), len(tarefas), len(res[m]) - b))
    print("\n  candidatos admitidos pelo gradiente: media %.2f de 2"
          % (statistics.mean(n_sel) if n_sel else 0))
    so_g = sorted(set(res["GRADIENTE"]) - set(res["BASE"]))
    perd = sorted(set(res["BASE"]) - set(res["GRADIENTE"]))
    print("  so o GRADIENTE resolve: %s" % (so_g or "nenhuma"))
    print("  o GRADIENTE perdeu:     %s" % (perd or "nenhuma"))

    g_ = len(res["GRADIENTE"]) - b
    p_ = len(res["PLACEBO"]) - b
    t_ = len(res["TUDO"]) - b
    print("\n  referencia: [[vocabulario-grande-piora]] mediu -3 jogando o")
    print("  vocabulario inteiro dentro; aqui o braco TUDO deu %+d." % t_)
    print("=" * 78)
    if p_ > 0:
        ver = "R7"
        print("R7 . o PLACEBO tambem ganha (%+d): qualquer codec a mais ajuda" % p_)
        print("     e o efeito nao e do material. Invalida a leitura do R4.")
    elif g_ >= 2 and p_ <= 0:
        ver = "R4"
        print("R4 . ADMITIR POR GRADIENTE PAGA (%+d tarefas, placebo %+d)."
              % (g_, p_))
        print("     O ganho e da SELECAO, que e onde o gargalo estava.")
    elif -1 <= g_ <= 1:
        ver = "R5"
        print("R5 . admitir NAO muda resolucao (%+d)." % g_)
        print("     A ladeira admite material util, mas admitir nao basta: o")
        print("     gargalo esta num terceiro lugar, DEPOIS da admissao.")
    else:
        ver = "R6"
        print("R6 . admitir CUSTA (%+d): a selecao por gradiente nao salva o" % g_)
        print("     que [[vocabulario-grande-piora]] ja tinha medido.")
    return {"veredito": ver, "n": len(tarefas), "teto": teto,
            "resolve": {m: len(v) for m, v in res.items()},
            "delta": {"TUDO": t_, "GRADIENTE": g_, "PLACEBO": p_},
            "so_gradiente": so_g, "gradiente_perdeu": perd,
            "admitidos_media": statistics.mean(n_sel) if n_sel else 0}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TAREFAS
    r = experimento(n_tarefas=n)
    json.dump(r, open("resultados/paga.json", "w"), indent=1,
              ensure_ascii=False, default=str)


# ------------------------------------------------------------- em paralelo
def _uma_tarefa(args):
    """
    Uma tarefa, os quatro bracos. Funcao de topo de modulo porque o Pool do
    Windows usa `spawn` e precisa importa-la pelo nome.

    Mesma logica de `experimento`, incluindo o reaproveitamento exato (selecao
    vazia = BASE, selecao cheia = TUDO). Tarefas sao independentes, entao
    paralelizar nao muda nenhum resultado — so o relogio.
    """
    nome, tr, te, teto = args
    inventados = [CorPorPropMais(), ApagaPorPropMais()]
    placebos = [CorPorPropPlacebo(), ApagaPorPropPlacebo()]
    base = list(CODECS)
    ok_b = resolve(tr, te, base, teto)
    ok_t = resolve(tr, te, base + inventados, teto)
    ok_p = resolve(tr, te, base + placebos, teto)
    sel = []
    for c in inventados:
        g = grad_loo(c, tr)
        if g is not None and g < 0:
            sel.append(c)
    if not sel:
        ok_g = ok_b
    elif len(sel) == len(inventados):
        ok_g = ok_t
    else:
        ok_g = resolve(tr, te, base + sel, teto)
    return nome, ok_b, ok_t, ok_g, ok_p, len(sel)


def paralelo(n_tarefas: int = N_TAREFAS, teto: int = TETO,
             nucleos: int = 4) -> Dict:
    from multiprocessing import Pool
    tarefas = list(carregar(PASTA))[:n_tarefas]
    print("=" * 78)
    print("R4 — admitir por gradiente resolve MAIS tarefas?  (%d nucleos)"
          % nucleos)
    print("=" * 78)
    print("  %d tarefas, teto %d em TODOS os bracos (orcamento casado)"
          % (len(tarefas), teto))
    print("  selecao por DEIXA-UM-DE-FORA dentro do treino; teste nunca tocado\n",
          flush=True)
    res = {k: [] for k in ("BASE", "TUDO", "GRADIENTE", "PLACEBO")}
    n_sel = []
    t0 = time.time()
    with Pool(nucleos) as pool:
        for k, (nome, b, t, g, p, ns) in enumerate(pool.imap_unordered(
                _uma_tarefa, [(n, tr, te, teto) for n, tr, te in tarefas])):
            for etq, ok in (("BASE", b), ("TUDO", t), ("GRADIENTE", g),
                            ("PLACEBO", p)):
                if ok:
                    res[etq].append(nome)
            n_sel.append(ns)
            if (k + 1) % 10 == 0:
                print("    %3d/%d | BASE=%d TUDO=%d GRAD=%d PLC=%d | %.0f min"
                      % (k + 1, len(tarefas), len(res["BASE"]),
                         len(res["TUDO"]), len(res["GRADIENTE"]),
                         len(res["PLACEBO"]), (time.time() - t0) / 60),
                      flush=True)
    b = len(res["BASE"])
    g_ = len(res["GRADIENTE"]) - b
    p_ = len(res["PLACEBO"]) - b
    t_ = len(res["TUDO"]) - b
    print("\n" + "=" * 78)
    for m in ("BASE", "TUDO", "GRADIENTE", "PLACEBO"):
        print("  %-10s resolve %3d/%d  (%+d sobre a base)"
              % (m, len(res[m]), len(tarefas), len(res[m]) - b))
    print("\n  admitidos pelo gradiente: media %.2f de 2; distribuicao %s"
          % (statistics.mean(n_sel),
             {k: n_sel.count(k) for k in sorted(set(n_sel))}))
    so_g = sorted(set(res["GRADIENTE"]) - set(res["BASE"]))
    perd = sorted(set(res["BASE"]) - set(res["GRADIENTE"]))
    print("  so o GRADIENTE resolve: %s" % (so_g or "nenhuma"))
    print("  o GRADIENTE perdeu:     %s" % (perd or "nenhuma"))
    print("  referencia [[vocabulario-grande-piora]]: -3; aqui TUDO deu %+d" % t_)
    print("=" * 78)
    if p_ > 0:
        ver = "R7"
        print("R7 . o PLACEBO tambem ganha (%+d): efeito nao e do material." % p_)
    elif g_ >= 2 and p_ <= 0:
        ver = "R4"
        print("R4 . ADMITIR POR GRADIENTE PAGA (%+d, placebo %+d)." % (g_, p_))
    elif -1 <= g_ <= 1:
        ver = "R5"
        print("R5 . admitir NAO muda resolucao (%+d): o gargalo esta DEPOIS" % g_)
        print("     da admissao.")
    else:
        ver = "R6"
        print("R6 . admitir CUSTA (%+d)." % g_)
    r = {"veredito": ver, "n": len(tarefas), "teto": teto,
         "resolve": {m: len(v) for m, v in res.items()},
         "delta": {"TUDO": t_, "GRADIENTE": g_, "PLACEBO": p_},
         "so_gradiente": so_g, "gradiente_perdeu": perd,
         "admitidos": {k: n_sel.count(k) for k in sorted(set(n_sel))}}
    json.dump(r, open("resultados/paga.json", "w"), indent=1,
              ensure_ascii=False, default=str)
    return r
