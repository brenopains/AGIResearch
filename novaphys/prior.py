"""
novaphys.prior — quanto vale um prior humano sobre objetivos, e vale a pena?
=============================================================================

A PERGUNTA, E POR QUE ELA VEM ANTES DE BAIXAR QUALQUER MODELO
---------------------------------------------------------------
`objetivo.py` gera 3.486 hipoteses de objetivo de uma gramatica. A maioria e
absurda: `coluna=1 & coluna=6` e impossivel, `linha=3 & linha=5` idem. Um
humano nunca consideraria essas — ele parte de um punhado de condicoes de
vitoria plausiveis e testa entre elas.

Isso sugere que um prior humano cortaria o custo de descobrir o objetivo. Mas
"sugere" nao e numero, e obter o prior de verdade custa baixar um modelo de
~2,5 GB. Entao mede-se o TETO primeiro, de graca:

    se um conjunto PEQUENO e PLAUSIVEL de hipoteses converge muito mais
    rapido que a gramatica inteira, o prior vale a pena. Se nao converge,
    modelo nenhum ajuda e o download seria desperdicio.

O CONTROLE QUE SEPARA CONTEUDO DE TAMANHO
-------------------------------------------
Um conjunto plausivel e menor. Menor converge mais rapido por construcao —
menos hipoteses para eliminar. Se eu comparasse so contra a gramatica
inteira, mediria tamanho e chamaria de prior.

Por isso ha tres bracos:

    GRAMATICA   as 3.486 geradas
    PLAUSIVEL   ~30 condicoes de vitoria que fazem sentido num jogo de grade
    ALEATORIO   ~30 sorteadas da gramatica, MESMO TAMANHO do plausivel

O que interessa e PLAUSIVEL contra ALEATORIO. Se empatarem, o ganho era
tamanho e o prior nao tem conteudo. Se PLAUSIVEL vencer, o conteudo importa e
ai vale buscar o prior de verdade num modelo.

O QUE ESTE ARQUIVO NAO E
--------------------------
Nao e simulacao de um modelo pequeno. Eu escrevo a lista PLAUSIVEL, e eu sou
um modelo de linguagem — entao isto mede o TETO do que um prior humano
poderia dar, nao o que um modelo de 1B daria. O modelo real seria pior que
isto, nunca melhor.

Se o teto for baixo, esta decidido sem baixar nada. Se for alto, o numero
diz quanto se estaria perseguindo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  H0  PRE-CONDICAO. O objetivo verdadeiro tem que estar no conjunto
      PLAUSIVEL, para cada um dos 5 objetivos. Se nao estiver, o braco nao
      pode convergir e a comparacao mede a minha lista, nao o prior.

  H1  PLAUSIVEL converge >= 2x mais rapido que ALEATORIO de mesmo tamanho
      -> o conteudo do prior importa; vale buscar num modelo

  H2  PLAUSIVEL ~ ALEATORIO
      -> era tamanho, nao conteudo. Um prior humano nao acrescenta e o
         download seria desperdicio

  H3  PLAUSIVEL pior que ALEATORIO
      -> minha nocao de plausivel esta errada para este mundo

MINHA PREVISAO, escrita antes de rodar: H1, mas com margem menor do que
parece obvio. A razao e que a gramatica ja e podada por observacao muito
rapido — hipoteses impossiveis como `coluna=1 & coluna=6` morrem na primeira
vitoria, porque nunca sao verdadeiras. Prior humano evita CONSIDERAR o
absurdo, mas o absurdo ja se elimina sozinho barato.

O que o prior deve realmente comprar e outra coisa: distinguir entre as
hipoteses SOBREVIVENTES plausiveis, que e onde a gramatica fica presa. Se
esse for o mecanismo, o ganho aparece no fim da curva, nao no comeco.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Dict, List, Optional, Tuple

from novaphys.composto import (LADO, Mundo, OBJETIVOS, Modelo, ACOES,
                               gerar_hipoteses)


# ══════════════════════════════════════════════ o prior humano
def plausiveis() -> List[Tuple[str, Callable]]:
    """
    Condicoes de vitoria que fazem sentido num jogo de grade com coleta.
    Escrito por mim, que sou um modelo de linguagem — ou seja, isto e o TETO
    do que um prior humano daria, nao o que um modelo de 1B daria.

    O criterio de inclusao e "um jogador humano consideraria isto uma regra
    de vitoria possivel", nao "isto e o objetivo verdadeiro". Por isso a
    lista tem coisas que nao sao o objetivo de nenhum mundo.
    """
    fora: List[Tuple[str, Callable]] = []
    for k in range(1, 4):
        fora.append((f"coletar {k}", lambda e, k=k: e[1] >= k))
    fora.append(("coletar tudo", lambda e: e[1] >= 4))
    cantos = [(0, 0), (0, LADO - 1), (LADO - 1, 0), (LADO - 1, LADO - 1)]
    for c in cantos:
        fora.append((f"chegar no canto {c}", lambda e, c=c: e[0] == c))
    fora.append(("chegar em qualquer canto",
                 lambda e: e[0] in cantos))
    fora.append(("chegar na ultima linha", lambda e: e[0][0] == LADO - 1))
    fora.append(("chegar na primeira linha", lambda e: e[0][0] == 0))
    fora.append(("chegar na ultima coluna", lambda e: e[0][1] == LADO - 1))
    fora.append(("chegar na primeira coluna", lambda e: e[0][1] == 0))
    fora.append(("chegar na borda",
                 lambda e: e[0][0] in (0, LADO - 1)
                 or e[0][1] in (0, LADO - 1)))
    fora.append(("voltar ao inicio", lambda e: e[0] == (0, 0)))
    fora.append(("chegar ao meio",
                 lambda e: e[0] == (LADO // 2, LADO // 2)))
    for c in cantos:
        for k in (1, 2):
            fora.append((f"canto {c} com {k}",
                         lambda e, c=c, k=k: e[0] == c and e[1] >= k))
    for k in (1, 2):
        fora.append((f"ultima linha com {k}",
                     lambda e, k=k: e[0][0] == LADO - 1 and e[1] >= k))
        fora.append((f"borda com {k}",
                     lambda e, k=k: (e[0][0] in (0, LADO - 1)
                                     or e[0][1] in (0, LADO - 1))
                     and e[1] >= k))
    fora.append(("percorrer toda a grade", lambda e: False))
    fora.append(("nunca vencer", lambda e: False))
    return fora


def amostra_gramatica(n: int, semente: int, obj: str, m: "Mundo"
                      ) -> List[Tuple[str, Callable]]:
    """
    Controle: n hipoteses DISTINTAS da gramatica, com o objetivo verdadeiro
    garantido dentro.

    A primeira versao apenas sorteava n de 3.486 — e assim quase nunca
    continha a resposta. Quando as sobreviventes zeravam, o episodio contava
    como "nao convergiu", e eu comparei o prior contra um conjunto que NAO
    PODE VENCER. O 1,48x que atribui a conteudo media so ter a resposta
    contra nao ter.
    """
    g = gerar_hipoteses()
    rng = random.Random(semente)
    est = [((i, j), b, ()) for i in range(LADO) for j in range(LADO)
           for b in range(5)]
    alvo = tuple(m.venceu((q, b, ())) for q, b, _ in est)
    verdadeiras = [h for h in g if tuple(h[1](e) for e in est) == alvo]
    if not verdadeiras:
        raise ValueError(f"gramatica nao expressa {obj}")

    # TAMANHO EFETIVO, nao bruto. Medindo os conjuntos:
    #     PLAUSIVEL   30 brutas ->  28 distintas
    #     ALEATORIO   30 brutas ->  10 distintas   <- o vies
    #     GRAMATICA 3486 brutas -> 324 distintas
    # Sortear 30 da gramatica da 10 hipoteses funcionalmente distintas, nao
    # 30: metade dos pares e identica porque expressoes como
    # `linha=3 & coluna=5` nunca sao verdadeiras em estado alcancavel e
    # colapsam todas em "nunca vence". Com a convergencia definida como
    # "sobreviventes com a mesma assinatura", esse conjunto ja NASCE quase
    # convergido — e vencia por isso, nao por ser mais discriminavel.
    # O controle certo casa DISTINTAS, sorteando brutas ate atingir n.
    fora = [verdadeiras[0]]
    vistas = {alvo}
    pool = [h for h in g if h not in verdadeiras]
    rng.shuffle(pool)
    for h in pool:
        if len(vistas) >= n:
            break
        sig = tuple(h[1](e) for e in est)
        if sig not in vistas:
            vistas.add(sig)
            fora.append(h)
    return fora


# ══════════════════════════════════════════════ episodio
def rodar(m: Mundo, hip: List, max_acoes: int, semente: int
          ) -> Optional[int]:
    """
    Explora com o melhor agente que temos (modelo + parcial + novidade) e
    devolve em quantas acoes as sobreviventes viram funcionalmente uma so.
    """
    rng = random.Random(semente)
    mod = Modelo()
    cont: Dict[tuple, int] = {}
    vivos = list(range(len(hip)))
    amostra = [((i, j), b, ()) for i in range(LADO) for j in range(LADO)
               for b in range(5)]
    assin = [tuple(f(e) for e in amostra) for _n, f in hip]
    m.reiniciar()

    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        melhor, cands = None, []
        for a in range(4):
            dy, dx = ACOES[a]
            alv = (e0[0][0] + dy, e0[0][1] + dx)
            if 0 <= alv[0] < LADO and 0 <= alv[1] < LADO:
                cor = e0[2][alv[0]][alv[1]]
            else:
                cor = None
            if cor is not None and cor not in mod.regra:
                pont = -1
            else:
                prox = mod.prever(e0, a, True)
                pont = cont.get(prox, 0) if prox is not None else 10 ** 6
            if melhor is None or pont < melhor:
                melhor, cands = pont, [a]
            elif pont == melhor:
                cands.append(a)
        a = rng.choice(cands)

        e1 = m.passo(a)
        cont[e1] = cont.get(e1, 0) + 1
        mod.aprender(e0, a, e1)

        venceu = m.venceu(e1)
        vivos = [k for k in vivos if hip[k][1](e1) == venceu]
        if vivos and len({assin[k] for k in vivos}) == 1:
            return t
        if not vivos:
            return None                    # verdadeiro fora do conjunto
        if venceu:
            m.reiniciar()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mundos", type=int, default=20)
    ap.add_argument("--acoes", type=int, default=3000)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    pl = plausiveis()
    gr = gerar_hipoteses()

    print("=" * 72)
    print("PRIOR — quanto vale um prior humano sobre objetivos?")
    print("=" * 72)
    print(f"GRAMATICA  {len(gr):>5} hipoteses")
    print(f"PLAUSIVEL  {len(pl):>5} hipoteses  (teto do prior humano)")
    est0 = [((i, j), b, ()) for i in range(LADO) for j in range(LADO)
            for b in range(5)]
    n_dist = len({tuple(f(e) for e in est0) for _n, f in pl})
    n_gr = len({tuple(f(e) for e in est0) for _n, f in gr})
    print(f"           {n_dist:>5} DISTINTAS no plausivel, "
          f"{n_gr} distintas na gramatica")
    print(f"ALEATORIO  {n_dist:>5} distintas  (controle casado em tamanho "
          f"EFETIVO, verdadeira garantida)")
    print()

    # H0: o objetivo verdadeiro esta no conjunto plausivel?
    falta = []
    for obj in OBJETIVOS:
        m = Mundo(a.semente, objetivo=obj)
        est = [((i, j), b, ()) for i in range(LADO) for j in range(LADO)
               for b in range(5)]
        alvo = tuple(m.venceu((p, b, ())) for p, b, _ in est)
        if not any(tuple(f(e) for e in est) == alvo for _n, f in pl):
            falta.append(obj)
    print(f"H0 · objetivo verdadeiro no conjunto PLAUSIVEL: "
          f"{'todos os 5' if not falta else 'FALTA ' + str(falta)}")
    if falta:
        print("     O braco nao pode convergir nesses; a comparacao mediria")
        print("     a minha lista, nao o prior. Abortando.")
        return
    print()

    print(f"{'objetivo':16s} {'GRAMATICA':>11} {'PLAUSIVEL':>11} "
          f"{'ALEATORIO':>11}")
    print("-" * 54)
    tg: List[int] = []
    tp: List[int] = []
    ta: List[int] = []
    for obj in OBJETIVOS:
        cg, cp, ca = [], [], []
        for k in range(a.mundos):
            m0 = Mundo(a.semente + k, objetivo=obj)
            al = amostra_gramatica(n_dist, a.semente + 7 + k, obj, m0)
            for conj, acc in ((gr, cg), (pl, cp), (al, ca)):
                m = Mundo(a.semente + k, objetivo=obj)
                r = rodar(m, conj, a.acoes, a.semente + 500 + k)
                acc.append(r if r is not None else a.acoes)
        mg = sum(cg) / len(cg)
        mp = sum(cp) / len(cp)
        ma = sum(ca) / len(ca)
        cv = sum(1 for x in cp if x < a.acoes) / len(cp)
        tg += cg
        tp += cp
        ta += ca
        print(f"{obj:16s} {mg:>11.0f} {mp:>11.0f} {ma:>11.0f}"
              f"   conv {cv:.0%}"
              + ("" if cv >= 0.60 else "  <- censurado, nao ler"),
              flush=True)
    print("-" * 54)
    MG = sum(tg) / len(tg)
    MP = sum(tp) / len(tp)
    MA = sum(ta) / len(ta)
    print(f"{'MEDIA':16s} {MG:>11.0f} {MP:>11.0f} {MA:>11.0f}")

    print()
    print(f"PLAUSIVEL vs GRAMATICA  {MG/MP:>6.2f}x   (mistura conteudo e tamanho)")
    print(f"PLAUSIVEL vs ALEATORIO  {MA/MP:>6.2f}x   <- isola o CONTEUDO")
    print()
    if MA / MP >= 2.0:
        print("H1 · o conteudo do prior importa. Vale buscar num modelo.")
    elif MA / MP <= 0.9:
        print("H3 · minha nocao de plausivel esta errada para este mundo.")
    else:
        print("H2 · era TAMANHO, nao conteudo. Um prior humano nao")
        print("     acrescenta aqui, e baixar modelo seria desperdicio.")


if __name__ == "__main__":
    main()
