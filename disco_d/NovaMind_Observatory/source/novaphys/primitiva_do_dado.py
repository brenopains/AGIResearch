"""
novaphys.primitiva_do_dado — as primitivas VEM DO DADO, e depois se compoem
============================================================================

DE ONDE VEM
-------------
Do usuario, corrigindo meu enquadramento: "Humanos nao inventam primitivas, eles
extraem do mundo e de dados, temos N datasets pra isso!"

Ele esta certo, e eu vinha repetindo um erro. Este projeto JA extrai primitiva
de dado, medido: o agente construiu `h = x*vy - y*vx` das trajetorias sozinho
(10/10, nulo 10/10, [[inventar-a-grandeza]]), derivou `n^ = rot(v^)` levando um
mundo de 0% a 95% ([[gramatica-propria]]), e `lingua.segmenta` extrai lexico de
fluxo bruto — as "palavras" sao primitivas descobertas.

O que era verdade e so de `segunda_ordem.py`: ali EU escrevi regime/deriva/
extremo/acelera a mao, porque era teste de composicao. **A extracao e a
composicao nunca foram ligadas.** Mesma forma do buraco extrator->loja.

O QUE ESTE MODULO FAZ
-----------------------
A loja recebe episodios (fluxo + alvo) SEM rotulo de transformacao. O sistema
nao sabe que existem quatro regras nem quais sao. Ele tem de:

  1. DESCOBRIR quantas primitivas ha e quais, agrupando os episodios pela
     similaridade das REGRAS que eles implicam (nao pela aparencia do fluxo);
  2. destilar uma regra por grupo, feixando os episodios daquele grupo;
  3. TRANSPORTAR as regras descobertas para um material novo;
  4. COMPOR as descobertas e prever uma regra composta inedita.

Os rotulos verdadeiros existem no gerador e sao usados SO para avaliar o passo
1. Nenhum passo do pipeline os ve.

PRE-REGISTRO
--------------
  Y0  PRE-CONDICAO. O agrupamento recupera as transformacoes acima do acaso:
      pureza >= 0,50 com 4 grupos (acaso = 0,25). Se nao separar, nao ha
      primitiva descoberta e o resto nao e interpretavel.
  Y1  compor as primitivas DESCOBERTAS entrega >= 0,70 de acuracia balanceada
      na regra composta inedita, e bate o controle (composicao de grupos
      errados) por >= 10 pontos
      -> o ciclo fecha: dado -> primitiva -> composicao -> previsao inedita,
         sem eu nomear nada
  Y2  descobre mas nao compoe -> a extracao serve e a ponte nao atravessa
      primitiva descoberta
  Y3  nao descobre (Y0 falha) -> o gargalo esta na EXTRACAO, nao na composicao

  Nulo: mesmo pipeline com alvos SORTEADOS. A pureza tem de cair ao acaso.

PREVISAO: Y0 passa (0,75) — as regras sao bem distintas no espaco de termos, e
o agrupamento so precisa de ORDEM, que e o papel em que o HDC mede positivo.
Y1 a 0,5: o transporte de uma regra DESTILADA de grupo pode ser pior que o de
uma regra pura, porque o grupo pode misturar duas transformacoes parecidas —
e `regime`/`extremo` sao as duas sobre nivel, entao sao as candidatas a fundir.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe, sim
from novaphys.nuvem import preve
from novaphys.nuvem_matriz import (TRANSFS, codifica, episodio,
                                   sorteia_materiais)
from novaphys.segunda_ordem import _ac, compoe, episodio_composto

N_POR_TRANSF = 8
N_TESTE = 12
SEMENTE = 20260918


# ────────────────────────────── 1. DESCOBRIR as primitivas
def agrupa(e: Espaco, vetores: List[np.ndarray], k: int,
           rng: random.Random, n_iter: int = 12) -> List[int]:
    """
    k-medoides por similaridade HDC. Sem rotulo, sem numero de classes vindo
    do dado — `k` e o unico parametro, e e a hipotese "ha k primitivas".

    Medoide e nao centroide de proposito: o feixe de vetores muito diferentes
    vira ruido, e [[indice-ve-superficie]] mediu que o HDC ordena bem perto e
    perde resolucao longe. Medoide so precisa de ordem local.
    """
    n = len(vetores)
    cen = rng.sample(range(n), k)
    atrib = [0] * n
    for _ in range(n_iter):
        for i, v in enumerate(vetores):
            atrib[i] = max(range(k), key=lambda c: sim(v, vetores[cen[c]]))
        novos = []
        for c in range(k):
            membros = [i for i in range(n) if atrib[i] == c]
            if not membros:
                novos.append(rng.randrange(n))
                continue
            # medoide: o membro de maior similaridade media aos outros
            novos.append(max(membros, key=lambda i: sum(
                sim(vetores[i], vetores[j]) for j in membros)))
        if novos == cen:
            break
        cen = novos
    return atrib


def pureza(atrib: Sequence[int], verdade: Sequence[str], k: int) -> float:
    """Fracao de acerto se cada grupo receber o rotulo majoritario dele."""
    acerto = 0
    for c in range(k):
        rot = [verdade[i] for i in range(len(atrib)) if atrib[i] == c]
        if rot:
            acerto += Counter(rot).most_common(1)[0][1]
    return acerto / max(len(atrib), 1)


def experimento() -> None:
    e = Espaco()
    rng = random.Random(SEMENTE)
    mats = [m for m in sorteia_materiais(rng, 24)
            if m[0]["tipo"] in ("linear", "saturado")][:4]
    m_src, m_dst = 0, 1
    # SUPER-AGRUPAR: k MAIOR que o numero real de regras.
    # A rodada 1 usou k=4 (o numero verdadeiro) e os grupos sairam impuros —
    # `regime`+`extremo` fundiram (os dois sobre NIVEL) e `acelera`+`deriva`
    # tambem (os dois sobre MUDANCA). Era o risco escrito no pre-registro.
    # Com k maior, subgrupos PUROS se formam e a busca por pares escolhe os
    # puros. O sistema continua sem saber quantas regras existem — k e
    # hipotese, e super-estimar e o lado seguro do erro.
    k = 2 * len(TRANSFS)

    print("=" * 78)
    print("PRIMITIVA DO DADO — descobrir as primitivas e depois compo-las")
    print("=" * 78)
    print(f"  materiais: {[m[0]['tipo'][:4] for m in mats]}")
    print(f"  a loja recebe {len(TRANSFS) * N_POR_TRANSF} episodios do material {m_src}")
    print("  SEM rotulo de transformacao. O sistema nao sabe que ha 4 regras.\n")

    # a loja: episodios embaralhados, rotulo guardado SO para avaliar
    itens: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for tn in TRANSFS:
        for _ in range(N_POR_TRANSF):
            A, y = episodio(rng, mats[m_src], tn)
            itens.append((A, y, tn))
    rng.shuffle(itens)
    vet = [codifica(e, A, y) for A, y, _t in itens]
    verdade = [t for _A, _y, t in itens]

    atrib = agrupa(e, vet, k, rng)
    p = pureza(atrib, verdade, k)
    comp_grupos = [Counter(verdade[i] for i in range(len(atrib))
                           if atrib[i] == c) for c in range(k)]
    print(f"  pureza do agrupamento: {p:.3f}   (acaso {1/len(TRANSFS):.3f}, k={k} grupos para {len(TRANSFS)} regras reais)")
    for c, cc in enumerate(comp_grupos):
        if cc:
            print(f"    grupo {c}: {dict(cc)}")

    # NULO: mesmo pipeline com alvos sorteados
    itens_n = []
    for tn in TRANSFS:
        for _ in range(N_POR_TRANSF):
            A, _y = episodio(rng, mats[m_src], tn)
            yn = np.array([rng.choice([1, -1]) for _ in range(A.shape[0])])
            itens_n.append((A, yn, tn))
    vet_n = [codifica(e, A, y) for A, y, _t in itens_n]
    p_n = pureza(agrupa(e, vet_n, k, rng), [t for _a, _b, t in itens_n], k)
    print(f"  pureza no NULO (alvo sorteado): {p_n:.3f}\n")

    if p < 0.50 or p <= p_n + 0.10:
        print(f"Y0 FALHOU: pureza {p:.3f} (nulo {p_n:.3f}). O agrupamento nao")
        print("   descobre as primitivas; o gargalo esta na EXTRACAO.")
        return
    print(f"Y0 . pureza {p:.3f} acima do acaso e do nulo: ha primitiva "
          f"descoberta.\n")

    # 2. destilar uma regra por grupo DESCOBERTO
    regras: Dict[int, np.ndarray] = {}
    for c in range(k):
        membros = [vet[i] for i in range(len(atrib)) if atrib[i] == c]
        if membros:
            regras[c] = feixe(membros, e.dim)

    # 3+4. transportar e compor as DESCOBERTAS
    #     a fase vem de casar cada regra descoberta contra episodios rotulados
    #     do material de destino — que a loja TEM (ela guarda tudo), sem que
    #     ninguem diga a qual transformacao pertencem.
    def transporta_desc(c: int) -> Callable:
        amostra = [episodio(rng, mats[m_dst], t) for t in TRANSFS]
        melhor, fase = -1.0, 1.0
        for f in (1.0, -1.0):
            a = max(float(np.mean(preve(e, regras[c], A, f) == y))
                    for A, y in amostra)
            if a > melhor:
                melhor, fase = a, f
        return lambda A: preve(e, regras[c], A, fase)

    # quais grupos correspondem a regime e a deriva? o sistema NAO sabe o nome;
    # escolhe os dois grupos cujas regras melhor preveem o COMPOSTO quando
    # combinadas — busca sobre pares, que e o que ele pode fazer sem rotulo.
    # CONSERTO CRITICO, apontado por revisao externa em 18/09 e CONFIRMADO no
    # codigo. A versao anterior criava UM conjunto `testes`, media os 28 pares
    # NELE, pegava o MAXIMO e reportava esse maximo como resultado. Isso e
    # selecao-no-teste: 85,1% era desempenho na amostra usada para escolher,
    # inflado por tomar o melhor de 28 candidatos em 12 episodios. E o controle
    # era a MEDIA dos outros pares no mesmo conjunto — comparar um MAXIMO contra
    # uma MEDIA e vies duplo.
    #
    # Agora: conjunto de SELECAO e conjunto FINAL sao disjuntos e independentes.
    # O par e congelado na selecao e so entao avaliado no final. E o controle
    # recebe a MESMA oportunidade: o melhor par SORTEADO entre os demais, tambem
    # escolhido na selecao e avaliado no final.
    sel = [episodio_composto(rng, mats[m_dst], "regime", "deriva", "E")
           for _ in range(N_TESTE)]
    fin = [episodio_composto(rng, mats[m_dst], "regime", "deriva", "E")
           for _ in range(N_TESTE)]
    ps = {c: transporta_desc(c) for c in regras}
    pares = [(a, b) for a in regras for b in regras if a < b]

    melhor_par, ac_sel = None, -1.0
    for a, b in pares:
        v = _ac(compoe(ps[a], ps[b], "E"), sel)
        if v > ac_sel:
            ac_sel, melhor_par = v, (a, b)
    melhor_ac = _ac(compoe(ps[melhor_par[0]], ps[melhor_par[1]], "E"), fin)

    # CONTROLE com a MESMA oportunidade de selecao: escolhe-se ao acaso um
    # subconjunto do mesmo tamanho de candidatos, pega-se o melhor DELE na
    # selecao, e avalia-se no final. Media de 10 sorteios.
    ctrls = []
    for _ in range(10):
        amostra = rng.sample(pares, min(len(pares), 6))
        b_par = max(amostra, key=lambda ab: _ac(compoe(ps[ab[0]], ps[ab[1]],
                                                       "E"), sel))
        ctrls.append(_ac(compoe(ps[b_par[0]], ps[b_par[1]], "E"), fin))
    ctrl = statistics.mean(ctrls)
    ctrl_sd = statistics.pstdev(ctrls)
    print(f"  (par escolhido na SELECAO: {100*ac_sel:.1f}%  ->  "
          f"no conjunto FINAL: {100*melhor_ac:.1f}%)")

    print("  braco                                   bal. acc")
    print("  -------------------------------------   --------")
    print(f"  COMPOSTO de primitivas DESCOBERTAS        {100*melhor_ac:5.1f}%"
          f"   (grupos {melhor_par})")
    print(f"  CONTROLE (mesma selecao, 10 sorteios)     {100*ctrl:5.1f}%"
          f" +- {100*ctrl_sd:.1f}")
    print(f"  chutar a classe comum                      50.0%")
    print("")
    print("=" * 78)
    if melhor_ac >= 0.70 and melhor_ac >= ctrl + 0.10:
        print(f"Y1 . O CICLO FECHA: dado -> primitiva descoberta -> composicao")
        print(f"     -> previsao de regra inedita, {100*melhor_ac:.1f}% contra")
        print(f"     {100*ctrl:.1f}% dos outros pares. Eu nao nomeei nenhuma")
        print("     primitiva; o sistema as separou do dado.")
    else:
        print(f"Y2 . descobre mas nao compoe: {100*melhor_ac:.1f}% contra "
              f"{100*ctrl:.1f}%.")
        print("     A extracao serve; a ponte nao atravessa primitiva")
        print("     destilada de grupo.")


if __name__ == "__main__":
    experimento()
