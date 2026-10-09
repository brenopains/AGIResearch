"""
novaphys.politicas — o valor de uma representação é estimável antes de ser usada?
==================================================================================

`gargalo.py` fechou dois elos e deixou um quebrado:

    o conteúdo do campo tem efeito causal      provado, 21-26pp a tamanho e
                                               orçamento casados
    tamanho do campo, por si                   refutado, +0pp ao dobrar o
                                               espaço de busca sem conteúdo novo
    escolher QUAL estrutura guardar            quebrado

Quebrado com números: piso `(v-v)` 33%, política atual (frequência) `(v*v)`
49%, teto `(v/v)` 54%. A política escolhe a segunda de quatro e deixa 21pp
na mesa. E `selecao_cega.py` já mostrou que nenhum sinal disponível NA ORIGEM
— frequência, compressão, generalidade, estabilidade, pico — prevê a ordem.

A PERGUNTA QUE DECIDE TODAS AS POLÍTICAS DE UMA VEZ
----------------------------------------------------
Propostas de política (consolidação imediata · população de hipóteses ·
população + evidência transdomínio) diferem em QUANDO e COM QUE EVIDÊNCIA
comprometer. Mas todas dependem da mesma pressuposição:

    a utilidade medida em algumas físicas prevê a utilidade em outras

Se prevê, a política "suspenda, meça, e comprometa com a que ganhou" funciona,
e o mecanismo é simplesmente medir e esperar. Se não prevê, NENHUMA política
pode funcionar — o valor de uma representação não seria estimável nem depois
de seis físicas de evidência. Não adianta comparar quatro políticas antes de
saber isso.

É a generalização de `selecao_cega.py`: lá, sinais da ORIGEM. Aqui, evidência
TRANSDOMÍNIO a posteriori, que é o melhor sinal concebível dentro desta
arquitetura.

O DESENHO
----------
Doze módulos do simulador, partidos ANTES de olhar qualquer número, com as
quatro categorias representadas dos dois lados para não confundir partição
com categoria:

    DECISÃO     quantum · materia · quimica · fluidos · fusao · portal
    AVALIAÇÃO   bec · ressonancia · gravidade · relatividade · vacuo · caret

Nenhum módulo de AVALIAÇÃO participa de nenhuma decisão. A política escolhe
com os seis primeiros e é medida nos seis últimos.

Quatro campos, cada um com UM molde, todos com 9 termos e 594 candidatos —
tamanho e orçamento de busca idênticos, como em `gargalo.py`. Só o conteúdo
muda. Mais o campo vazio como referência.

Pareado: as medições de intervenção são colhidas UMA vez por (módulo,
semente) e todos os campos são avaliados sobre exatamente os mesmos pares.

PRÉ-REGISTRO, ANTES DE RODAR
------------------------------
    H1  a ordem em DECISÃO prevê a ordem em AVALIAÇÃO
        critério: o topo coincide E a concordância de pares é ≥ 5 de 6
        → política de evidência transdomínio funciona; vale construir as
          quatro políticas, e o mecanismo é medir e esperar

    H0  a ordem não se transfere
        → nenhuma política de consolidação pode funcionar nesta arquitetura;
          o gargalo não é QUANDO comprometer, é que a informação não existe.
          O caminho passa a ser não comprometer nunca — seleção por contexto,
          não consolidação global

    H2  heterogeneidade entre módulos
        se o vencedor variar de módulo para módulo, a média esconde o
        fenômeno e "consolidar UMA estrutura" é a pergunta errada; a resposta
        seria manter várias e escolher por contexto

MINHA PREVISÃO, escrita antes de ver o resultado
--------------------------------------------------
Espero que H1 passe na média, porque `(v/v)` e `(v*v)` venceram `(v+v)` e
`(v-v)` de forma consistente nas cinco físicas de `gargalo.py`. Mas espero
que H2 TAMBÉM apareça: suspeito que o vencedor por módulo varie bastante e
que a vantagem média de `(v/v)` venha de poucos módulos, não de superioridade
uniforme. Se as duas se confirmarem juntas, a leitura correta não é "achamos
a boa política" — é "consolidação global é o enquadramento errado".

Registro isso agora para não poder me dizer depois que eu já sabia.

──────────────────────────────────────────────────────────────────────────────
LIDO DEPOIS DE RODAR
──────────────────────────────────────────────────────────────────────────────
H1 PASSOU, e sem folga nenhuma para interpretação:

    ordem em DECISÃO    (v/v) > (v*v) > (v+v) > (v-v)
    ordem em AVALIAÇÃO  (v/v) > (v*v) > (v+v) > (v-v)
    concordância 6/6 · topo coincide

    piso   (v-v)                    49%
    P0     sortear                  54%
    P1     frequência → (v*v)       56%
    P3     transdomínio → (v/v)     63%   ← alcança o teto
    teto   (v/v)                    63%

H2 disparou "HETEROGÊNEO" e ESTÁ ERRADO — pelo artefato que eu havia sinalizado
antes de ver estes números, não depois. `max()` desempata pela ordem da lista,
então módulos onde todos os braços empatam carimbavam um vencedor fictício. Dos
4 módulos que o código creditou a `(v*v)`, três eram empates puros:

    quimica        0%  0%  0%  0%    módulo opaco, nem o campo vazio explica
    relatividade  67% 67% 67% 67%    empate total
    caret         75% 75% 75% 75%    empate total
    bec           20% 25% 20% 25%    empate no topo entre (/) e (-)

Contando só vencedor ESTRITO (margem > 2pp sobre o segundo):

    (v/v)  6x    quantum · materia · fluidos · ressonancia · gravidade · vacuo
    (v*v)  1x    fusao
    (v+v)  1x    portal
    (v-v)  0x
    sem sinal: quimica · bec · relatividade · caret

Dominância 6/8 = 75%. NÃO é heterogêneo — `(v/v)` é amplamente superior, e a
média representa o fenômeno em vez de escondê-lo. Minha previsão pré-registrada
de que H2 apareceria estava ERRADA.

O MECANISMO DA FALHA, agora identificado
------------------------------------------
`(v*v)` vence estritamente em UM módulo: `fusao`. Que é exatamente o módulo
onde a política de frequência a consolidou, em `descoberta_fisica.py` (5 de 8
compostas).

    a consolidação por frequência na origem seleciona a estrutura que serve
    ao MÓDULO DE ORIGEM, não a que serve em geral

Isso não é "critério subótimo". É sobreajuste ao domínio de descoberta, com
mecanismo nomeado e localizado. E explica de uma vez os quatro resultados
anteriores: por que `(v/v)` transferiu melhor no controle pareado, por que
nenhum sinal da origem previa a ordem, e por que o placebo empatava.

LIMITES DESTE RESULTADO
-------------------------
  · quatro candidatas. "Escolher a melhor de 4" é fácil; nada aqui diz que a
    ordem se transfere com uma população grande de estruturas descobertas.
  · P3 usa como critério de decisão a MESMA métrica da avaliação, medida em
    outros módulos. É a pergunta certa (a métrica generaliza entre físicas?)
    mas não é uma política implementada — é o teto do que uma política dessas
    poderia atingir.
  · as estruturas foram dadas, não descobertas. O elo "descobrir → suspender →
    medir → comprometer" continua sem ser rodado ponta a ponta.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics as st
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import PhysicsWorld
from .intervencao import E, Molde, compor
from .descoberta_fisica import intervir, custo, N_VARS, Par

DECISAO = ['quantum', 'materia', 'quimica', 'fluidos', 'fusao', 'portal']
AVALIACAO = ['bec', 'ressonancia', 'gravidade', 'relatividade', 'vacuo', 'caret']
CONTEUDOS = ['*', '/', '+', '-']
ORCAMENTOS = (54, 594)


def molde_binario(op: str) -> Molde:
    """(v0 op v1) — 3 nós, 2 slots, 6 instâncias, 594 candidatos com N_VARS=3."""
    return Molde(E(op, E('V', 0), E('V', 1)), 2, op)


def campo_de(m: Optional[Molde]) -> List[E]:
    c: List[E] = [E('V', i) for i in range(N_VARS)]
    if m is not None:
        c.extend(m.instancias(N_VARS))
    return c


def cobertura(dados: Dict[str, List[Par]], campo: Sequence[E],
              base: Sequence[E], ordem: random.Random) -> Dict[int, float]:
    """
    Fração das medições em que a busca acha estrutura que bate qualquer
    variável crua por margem > 5.

    Critério corrigido: sem o `tam() > 1` de `placebo.py`, que zerava o
    crédito justamente quando um molde consolidado explicava sozinho — uma
    instância de molde é `E('C',…)` e tem tam 1, que é o ponto do MDL.
    """
    cands = compor(campo)
    ordem.shuffle(cands)
    ok = {k: 0 for k in ORCAMENTOS}
    for _k, pares in sorted(dados.items()):
        base_c = min(custo(a, pares) for a in base)
        best, bc, i_k = None, math.inf, 0
        ks = sorted(ORCAMENTOS)
        for i, e in enumerate(cands, 1):
            c = custo(e, pares)
            if c < bc:
                best, bc = e, c
            while i_k < len(ks) and i >= ks[i_k]:
                if best is not None and base_c - bc > 5:
                    ok[ks[i_k]] += 1
                i_k += 1
        while i_k < len(ks):
            if best is not None and base_c - bc > 5:
                ok[ks[i_k]] += 1
            i_k += 1
    n = len(dados)
    return {k: v / n for k, v in ok.items()}


def concordancia(ord_a: Sequence[str], ord_b: Sequence[str]) -> Tuple[int, int]:
    """Pares ordenados igualmente nas duas listas, de todos os pares."""
    pos_a = {x: i for i, x in enumerate(ord_a)}
    pos_b = {x: i for i, x in enumerate(ord_b)}
    tot = ok = 0
    for x, y in itertools.combinations(ord_a, 2):
        tot += 1
        if (pos_a[x] < pos_a[y]) == (pos_b[x] < pos_b[y]):
            ok += 1
    return ok, tot


def relatorio(n_int: int = 32, passos: int = 80,
              sementes: Sequence[int] = (0, 1, 2, 3),
              orcamento: int = 594) -> Dict[str, Any]:
    w = PhysicsWorld()
    base = [E('V', i) for i in range(N_VARS)]
    moldes = {op: molde_binario(op) for op in CONTEUDOS}

    print("═" * 78)
    print("POLÍTICAS — o valor de uma representação é estimável antes de ser usada?")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  H1  topo coincide E concordância ≥ 5/6 → evidência transdomínio serve")
    print("  H0  não se transfere → nenhuma política pode funcionar")
    print("  H2  se o vencedor variar por módulo, consolidar UMA é a pergunta errada")
    print()
    print(f"DECISÃO    {' · '.join(DECISAO)}")
    print(f"AVALIAÇÃO  {' · '.join(AVALIACAO)}")
    print(f"sementes {list(sementes)}  ·  orçamento de busca fixo: {orcamento} candidatos")
    print("todos os campos: 9 termos, 594 candidatos — só o conteúdo muda")

    # (módulo, conteúdo) → lista de coberturas, uma por semente
    cel: Dict[Tuple[str, str], List[float]] = {}
    vaz: Dict[str, List[float]] = {}

    for mod in DECISAO + AVALIACAO:
        for s in sementes:
            rng = random.Random(s * 137 + 9)
            cap = w.capacidades()
            m = next(x for x in cap['modulos'] if x['id'] == mod)
            mans = [k for k in m['manoplas']
                    if k.get('tipo') == 'range' and 'min' in k][:N_VARS]
            if len(mans) < N_VARS:
                continue
            dados = intervir(w, mod, mans, n_int, passos, rng)
            if not dados:
                continue
            ordem = random.Random(hash((s, mod)) & 0xFFFF)
            vaz.setdefault(mod, []).append(
                cobertura(dados, campo_de(None), base, ordem)[orcamento])
            for op in CONTEUDOS:
                ordem = random.Random(hash((s, mod, op)) & 0xFFFF)
                c = cobertura(dados, campo_de(moldes[op]), base, ordem)[orcamento]
                cel.setdefault((mod, op), []).append(c)
        if (mod, CONTEUDOS[0]) in cel:
            linha = "  ".join(
                f"{op} {st.mean(cel[(mod, op)]):>4.0%}" for op in CONTEUDOS)
            faixa = "DEC" if mod in DECISAO else "AVA"
            venc = max(CONTEUDOS, key=lambda o: st.mean(cel[(mod, o)]))
            print(f"  {faixa} {mod:<14} vazio {st.mean(vaz[mod]):>4.0%}   "
                  f"{linha}   vence ({venc})")

    def media(mods: Sequence[str], op: str) -> float:
        v = [st.mean(cel[(m, op)]) for m in mods if (m, op) in cel]
        return st.mean(v) if v else 0.0

    dec = {op: media(DECISAO, op) for op in CONTEUDOS}
    ava = {op: media(AVALIACAO, op) for op in CONTEUDOS}
    ord_d = sorted(CONTEUDOS, key=lambda o: -dec[o])
    ord_a = sorted(CONTEUDOS, key=lambda o: -ava[o])

    print("\n" + "═" * 78)
    print(f"{'conteúdo':<12}{'DECISÃO':>12}{'AVALIAÇÃO':>12}")
    print("─" * 78)
    for op in CONTEUDOS:
        print(f"(v{op}v){'':<6}{dec[op]:>11.0%}{ava[op]:>12.0%}")
    print(f"\nordem em DECISÃO    {' > '.join(f'(v{o}v)' for o in ord_d)}")
    print(f"ordem em AVALIAÇÃO  {' > '.join(f'(v{o}v)' for o in ord_a)}")

    ok, tot = concordancia(ord_d, ord_a)
    topo = ord_d[0] == ord_a[0]
    print(f"concordância de pares {ok}/{tot}   ·   topo coincide: "
          f"{'sim' if topo else 'NÃO'}")

    print("\n" + "═" * 78)
    print("O QUE CADA POLÍTICA RENDERIA NOS MÓDULOS DE AVALIAÇÃO")
    escolha_freq = '*'          # o que a política atual consolida, por frequência
    escolha_dec = ord_d[0]      # o que a evidência transdomínio escolheria
    teto, piso = ord_a[0], ord_a[-1]
    aleat = st.mean(ava[o] for o in CONTEUDOS)
    print(f"  piso   pior conteúdo (v{piso}v)                    {ava[piso]:>5.0%}")
    print(f"  P0     sortear um conteúdo                    {aleat:>5.0%}")
    print(f"  P1     frequência na origem  → (v{escolha_freq}v)          {ava[escolha_freq]:>5.0%}")
    print(f"  P3     evidência transdomínio → (v{escolha_dec}v)          {ava[escolha_dec]:>5.0%}")
    print(f"  teto   melhor conteúdo (v{teto}v)                  {ava[teto]:>5.0%}")

    ganho = ava[escolha_dec] - ava[escolha_freq]
    resta = ava[teto] - ava[escolha_dec]

    print("\n" + "═" * 78)
    if topo and ok >= 5:
        print("H1 · A ORDEM SE TRANSFERE. Medir a utilidade em algumas físicas prevê")
        print("a utilidade em outras. Uma política que suspende, mede e só então")
        print("compromete tem sinal para funcionar.")
        print(f"Ganho sobre a política atual: {ganho:+.0%}. Ainda na mesa: {resta:.0%}.")
    else:
        print("H0 · A ORDEM NÃO SE TRANSFERE. Seis físicas de evidência não predizem")
        print("as outras seis. Nenhuma política de consolidação global pode funcionar")
        print("nesta arquitetura — não por escolher mal o critério, mas porque a")
        print("informação para escolher não existe nem depois do fato.")
        print("O caminho deixa de ser 'consolidar melhor' e passa a ser")
        print("'não consolidar' — manter várias e selecionar por contexto.")

    # VENCEDOR ESTRITO. `max()` desempata pela ordem da lista, então um módulo
    # onde todos os braços empatam — inclusive em zero — carimbaria um vencedor
    # que é artefato de ordenação, e H2 leria heterogeneidade onde há apenas
    # ausência de sinal. Exige-se margem real sobre o segundo colocado.
    MARGEM = 0.02
    vencedores: Dict[str, str] = {}
    sem_sinal: List[str] = []
    for mod in DECISAO + AVALIACAO:
        if (mod, CONTEUDOS[0]) not in cel:
            continue
        v = sorted(((st.mean(cel[(mod, o)]), o) for o in CONTEUDOS), reverse=True)
        if v[0][0] - v[1][0] > MARGEM:
            vencedores[mod] = v[0][1]
        else:
            sem_sinal.append(mod)
    dist = {o: sum(1 for v in vencedores.values() if v == o) for o in CONTEUDOS}
    print()
    print(f"H2 · vencedor ESTRITO por módulo: " +
          "  ".join(f"(v{o}v) {dist[o]}x" for o in CONTEUDOS))
    if sem_sinal:
        print(f"     sem vencedor estrito (empate ou módulo opaco): "
              f"{' · '.join(sem_sinal)}")
    domin = max(dist.values()) / max(len(vencedores), 1)
    if domin < 0.6:
        print(f"     HETEROGÊNEO ({domin:.0%}). Nenhum conteúdo domina; a média")
        print("     esconde o fenômeno. 'Qual estrutura consolidar' é a pergunta")
        print("     errada — a resposta seria manter várias e escolher por contexto.")
    else:
        print(f"     Um conteúdo domina {domin:.0%} dos módulos com sinal —")
        print("     a vantagem é ampla, não uma média de casos discordantes.")
    return {'celulas': cel, 'decisao': dec, 'avaliacao': ava}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--passos', type=int, default=80)
    ap.add_argument('--sementes', type=int, default=4)
    a = ap.parse_args()
    relatorio(n_int=a.n, passos=a.passos, sementes=tuple(range(a.sementes)))
