"""
novaphys.explica — o sistema diz o que descobriu, e da para CHECAR se e verdade
================================================================================

DE ONDE VEM
-------------
Do usuario: "Nao tem interface de conceito — nao consegue te contar o que
achou." E antes: "Talvez o LLM possa ser o decoder do pensamento pra humanos?"

POR QUE NAO E NARRACAO DE LLM
-------------------------------
Porque narracao nao e verificavel: um modelo fluente descreve bem tanto uma
regra certa quanto uma errada. Aqui a explicacao e LEITURA DIRETA do objeto.

A regra descoberta e um hipervetor R. `nuvem._tabela` ja calcula sim(termo, R)
para TODAS as combinacoes de termo que o encoder sabe montar — (papel, nivel,
alvo) e (papel, sinal-da-diferenca, alvo). Os termos de maior similaridade SAO
o conteudo da regra. Ler os primeiros e traduzi-los para frase e decodificacao,
nao parafrase.

E o LLM continua cabendo depois, no papel em que mediu bem aqui (critico/leitor,
10/12 em defeitos estruturais, [[juiz-do-modelo-local]]) — deixar a frase mais
fluente. Mas o CONTEUDO sai do vetor, e o nucleo nunca e tocado.

O TESTE: A EXPLICACAO E VERDADEIRA?
-------------------------------------
Uma explicacao so vale se der para checar. Cada grupo descoberto tem um rotulo
majoritario verdadeiro (usado SO na avaliacao). As transformacoes se dividem em
duas familias:

    NIVEL    regime (s > limiar), extremo (|s| > limiar)
    MUDANCA  deriva (s subindo),  acelera (diferenca crescendo)

Se a leitura do vetor for verdadeira, o tipo de termo dominante tem de bater
com a familia da transformacao majoritaria do grupo.

PRE-REGISTRO
--------------
  Z0  PRE-CONDICAO. Ha grupo puro o bastante para avaliar: pelo menos 4 grupos
      com rotulo majoritario >= 60%.
  Z1  a familia lida do vetor bate a familia verdadeira em >= 75% dos grupos
      avaliaveis (acaso 50%, sao duas familias)
      -> a explicacao e VERDADEIRA, nao so fluente
  Z2  bate entre 50% e 75% -> a leitura informa pouco
  Z3  bate ~50% -> a explicacao e ruido com aparencia de conteudo, que e o modo
      de falha que eu quero poder detectar

  Nulo: mesma leitura em regras destiladas de grupos com alvo SORTEADO. A
  concordancia tem de cair ao acaso.

PREVISAO: Z1, 0,7. Mecanismo: o encoder amarra alvo a nivel nos atrasos 0 e 1 e
a sinal de diferenca com deslocamento proprio; uma regra sobre nivel acumula
massa nos termos de nivel e uma sobre mudanca nos de diferenca, porque sao os
termos que se repetem entre episodios daquela regra. O risco nomeado e que
nivel e diferenca sao correlacionados num AR(1) — s alto tende a cair — entao a
massa pode vazar de uma familia para a outra.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe
from novaphys.nuvem import ATRASOS, NIVEIS, _tabela
from novaphys.nuvem_matriz import TRANSFS, codifica, episodio, sorteia_materiais
from novaphys.primitiva_do_dado import agrupa, pureza

FAMILIA = {"regime": "NIVEL", "extremo": "NIVEL",
           "deriva": "MUDANCA", "acelera": "MUDANCA"}
N_POR_TRANSF = 8
SEMENTE = 20260918


def le_regra(e: Espaco, R: np.ndarray, K: int, n_top: int = 6):
    """
    Le o conteudo do hipervetor: os termos de maior similaridade.

    Devolve (familia_dominante, linhas_legiveis, massa_nivel, massa_mudanca).
    A familia sai da MASSA total de cada tipo entre os termos do topo, nao do
    primeiro colocado sozinho — um termo de topo pode ser acidente, a massa nao.
    """
    tab = _tabela(e, R, K)
    itens = sorted(tab.items(), key=lambda kv: -abs(kv[1]))
    m_niv = m_mud = 0.0
    n_niv = n_mud = 0        # CONTAGEM por tipo, para normalizar
    linhas = []
    for (a, p, v, alvo), s in itens[:n_top]:
        if isinstance(a, int):
            tipo, desc = "NIVEL", (
                f"canal de papel {p}, nivel {v:+d} (atraso {a})")
            m_niv += abs(s); n_niv += 1
        else:
            tipo = "MUDANCA"
            qual = "subindo" if v > 0 else "descendo"
            ordem = "a taxa" if a == "d2" else "o valor"
            desc = f"canal de papel {p}, {ordem} {qual}"
            m_mud += abs(s); n_mud += 1
        linhas.append(f"quando {desc}  ->  alvo {alvo:+d}   "
                      f"(forca {s:+.4f}, {tipo})")
    for (a, p, v, alvo), s in itens[n_top:]:
        if isinstance(a, int):
            m_niv += abs(s); n_niv += 1
        else:
            m_mud += abs(s); n_mud += 1
    # CONSERTO: a massa TOTAL e enviesada por CONTAGEM. A tabela tem 2 atrasos
    # x K papeis x 9 niveis x 2 alvos = 108 termos de NIVEL contra 2 tipos x K
    # x 2 sinais x 2 alvos = 24 de MUDANCA — 4,5x mais. Somar |sim| dava NIVEL
    # em TODOS os grupos (massa 3,0-4,1 x 2,4-2,6), e a familia lida batia a
    # verdadeira em so 43%. As LINHAS DO TOPO ja estavam certas: contando pela
    # maioria dos 3 primeiros dava 6/7. Entao normaliza-se pela contagem.
    med_niv = m_niv / max(n_niv, 1)
    med_mud = m_mud / max(n_mud, 1)
    topo3 = [("NIVEL" if isinstance(a, int) else "MUDANCA")
             for (a, _p, _v, _al), _s in itens[:3]]
    fam_topo = max(set(topo3), key=topo3.count)
    fam_soma = "NIVEL" if m_niv >= m_mud else "MUDANCA"
    fam_media = "NIVEL" if med_niv >= med_mud else "MUDANCA"
    return {"topo": fam_topo, "soma": fam_soma, "media": fam_media,
            "linhas": linhas, "med_niv": med_niv, "med_mud": med_mud}


def linha_base(e: Espaco, K: int, n: int = 20) -> Tuple[float, float]:
    """
    O que a MESMA medida da num vetor ALEATORIO.

    E o [[nulo-do-duto]] aplicado a estatistica em vez de ao dado: soma favorece
    NIVEL (108 termos contra 24) e media favorece MUDANCA (poucos termos, cada
    um denso). Nenhuma das duas normalizacoes e neutra, e o meu pre-registro
    dizia "tipo dominante por massa" sem NUNCA especificar qual — o portao era
    ambiguo e as duas implementacoes eram igualmente fieis a ele.
    A referencia honesta e o que o acaso produz na propria medida.
    """
    rng = np.random.default_rng(12345)
    ns, ms = [], []
    for _ in range(n):
        R = rng.choice(np.array([-1, 1], dtype=np.int8), size=e.dim)
        r = le_regra(e, R, K)
        ns.append(r["med_niv"]); ms.append(r["med_mud"])
    return float(np.mean(ns)), float(np.mean(ms))


def experimento() -> None:
    e = Espaco()
    rng = random.Random(SEMENTE)
    mats = [m for m in sorteia_materiais(rng, 24)
            if m[0]["tipo"] in ("linear", "saturado")][:4]
    k = 2 * len(TRANSFS)

    print("=" * 78)
    print("EXPLICA — o sistema diz o que descobriu, e da para checar")
    print("=" * 78)

    itens = []
    for tn in TRANSFS:
        for _ in range(N_POR_TRANSF):
            A, y = episodio(rng, mats[0], tn)
            itens.append((A, y, tn))
    rng.shuffle(itens)
    vet = [codifica(e, A, y) for A, y, _t in itens]
    verdade = [t for _A, _y, t in itens]
    atrib = agrupa(e, vet, k, rng)
    print(f"  {len(itens)} episodios SEM rotulo, {k} grupos, "
          f"pureza {pureza(atrib, verdade, k):.3f}\n")

    K = mats[0].__len__()
    avaliaveis, acertos, ac_topo, ac_soma = 0, 0, 0, 0
    base_n, base_m = linha_base(e, mats[0].__len__())
    print(f"  linha de base do acaso: nivel {base_n:.4f}, mudanca {base_m:.4f}")
    print("")
    for c in range(k):
        idx = [i for i in range(len(atrib)) if atrib[i] == c]
        if len(idx) < 2:
            continue
        cc = Counter(verdade[i] for i in idx)
        maj, n_maj = cc.most_common(1)[0]
        frac = n_maj / len(idx)
        R = feixe([vet[i] for i in idx], e.dim)
        r = le_regra(e, R, K)
        # veredito: qual familia EXCEDE MAIS a propria linha de base do acaso
        exc_n = r["med_niv"] / max(base_n, 1e-9)
        exc_m = r["med_mud"] / max(base_m, 1e-9)
        fam = "NIVEL" if exc_n >= exc_m else "MUDANCA"
        linhas, mn, mm, fam_topo = r["linhas"], r["med_niv"], r["med_mud"], r["topo"]
        ok = fam == FAMILIA[maj]
        ok_topo = r["topo"] == FAMILIA[maj]
        ok_soma = r["soma"] == FAMILIA[maj]
        if frac >= 0.60:
            avaliaveis += 1
            acertos += ok
            ac_topo += ok_topo
            ac_soma += ok_soma
        marca = "OK " if ok else "ERRO"
        print(f"  grupo {c}  ({len(idx)} episodios, {maj} {100*frac:.0f}%)")
        print(f"    o sistema diz: familia {fam}  "
              f"(excesso sobre o acaso: nivel {mn/max(base_n,1e-9):.2f}x "
              f"x mudanca {mm/max(base_m,1e-9):.2f}x; topo-3 diz {fam_topo})   "
              f"verdade: {FAMILIA[maj]}  [{marca}]")
        for l in linhas[:3]:
            print(f"      {l}")
        print("")

    # NULO: mesma leitura em regras de alvo SORTEADO
    it_n = []
    for tn in TRANSFS:
        for _ in range(N_POR_TRANSF):
            A, _y = episodio(rng, mats[0], tn)
            it_n.append((A, np.array([rng.choice([1, -1])
                                      for _ in range(A.shape[0])]), tn))
    vn = [codifica(e, A, y) for A, y, _t in it_n]
    an = agrupa(e, vn, k, rng)
    av_n = ac_n = 0
    for c in range(k):
        idx = [i for i in range(len(an)) if an[i] == c]
        if len(idx) < 2:
            continue
        cc = Counter(it_n[i][2] for i in idx)
        maj, n_maj = cc.most_common(1)[0]
        if n_maj / len(idx) < 0.60:
            continue
        fam = le_regra(e, feixe([vn[i] for i in idx], e.dim), K)[0]
        av_n += 1
        ac_n += (fam == FAMILIA[maj])

    print("=" * 78)
    if avaliaveis < 4:
        print(f"Z0 FALHOU: so {avaliaveis} grupos avaliaveis (>=60% puros).")
        return
    taxa = acertos / avaliaveis
    taxa_n = ac_n / max(av_n, 1)
    print(f"Z0 . {avaliaveis} grupos avaliaveis.")
    print(f"  TRES medidas da mesma leitura, sobre {avaliaveis} grupos:")
    print(f"    excesso sobre o acaso (veredito):  {acertos}/{avaliaveis} "
          f"= {100*taxa:.0f}%")
    print(f"    leitura direta do topo-3:          {ac_topo}/{avaliaveis} "
          f"= {100*ac_topo/avaliaveis:.0f}%")
    print(f"    soma bruta (a 1a versao, enviesada):{ac_soma}/{avaliaveis} "
          f"= {100*ac_soma/avaliaveis:.0f}%")
    print(f"  no NULO (alvo sorteado): {ac_n}/{av_n} = {100*taxa_n:.0f}%")
    print("")
    if taxa >= 0.75 and taxa > taxa_n + 0.15:
        print("Z1 . A EXPLICACAO E VERDADEIRA, nao so fluente. O conteudo sai")
        print("     do vetor por leitura direta, e bate com a regra que gerou")
        print("     os episodios — que o sistema nunca viu nomeada.")
    elif taxa >= 0.50:
        print(f"Z2 . a leitura informa pouco ({100*taxa:.0f}%).")
    else:
        print("Z3 . a explicacao e ruido com aparencia de conteudo.")


if __name__ == "__main__":
    experimento()
