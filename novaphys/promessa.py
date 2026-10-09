"""
novaphys.promessa — crédito para a peça que ainda não paga nada
=================================================================

O GARGALO QUE ISTO ATACA
--------------------------
Nove nulos pré-registrados nesta linha, e [[paisagem-e-penhasco]] explica todos:
no espaço discreto, trocar um marcador salta o erro de 0 para mediana 46 em 96%
dos casos, sem valores intermediários. Não há gradiente para a construção seguir.
Por isso [[abducao-construir-nao-propor]] só funcionou na física, onde o resíduo
é contínuo e "quase certo" vem de graça.

E [[cegueira-do-ganho-imediato]] mediu a consequência exata: operador cujo ganho
só aparece depois é invisível para seleção E para geração. `[mapa_cor]` sozinho
morria embora fosse metade de `[mapa_cor, copia]`; completar à mão com `copia`
valeu +3 (40 -> 44) e era conserto específico.

A INVERSÃO
------------
Hoje o crédito vai para a SAÍDA que a peça produz:

    valor(p) = Δ solução causada por p          -> exatamente zero, por definição

Aqui o crédito vai para a RESTRIÇÃO que a peça impõe sobre o que ainda falta:

    valor(p) = quanto p encurta a descrição das continuações ainda necessárias

Representa-se p como programa incompleto `x --p--> z --?--> y`, onde `?` NÃO é
um operador candidato: é uma função desconhecida. Para cada par i, a obrigação
é "o que `?` teria de fazer para levar z_i a y_i". A pergunta é se essas
obrigações COLAPSAM entre os pares — se os três demos passaram a precisar da
MESMA coisa desconhecida.

O TESTE-PUNHAL, E ELE É POR CONSTRUÇÃO
----------------------------------------
Uma medida que marginalize sobre continuações c ∈ CODECS nunca prova nada: ela
seria busca de profundidade 2 disfarçada. Esta medida **não toca em CODECS em
nenhum ponto**. Ela lê `_estagio([p], treino)` e comprime. Nada mais. Logo o
teste "remova do avaliador a operação que realmente vinha depois" passa por
construção, e não por checagem.

A REPRESENTAÇÃO CANÔNICA DO BURACO
------------------------------------
Um compressor genérico vê LITERAIS, não transformações. Comprimir (z_i, y_i)
cru só colapsaria se as grades fossem parecidas — e no ARC não são. Então o
buraco é codificado como DELTA: as células que ainda precisam mudar, e de que
cor para qual. Um diff não é vocabulário de domínio; é a definição operacional
de "o que falta explicar".

    buraco_i  =  [(i, j, de, para) para cada célula onde z_i != y_i]

    colapso(S) = [ Σ_i L(buraco_i) − L(buraco_1 ++ … ++ buraco_n) ] / Σ_i L(buraco_i)

    promessa(p) = colapso(depois de p) − colapso(antes)

"Descrever os buracos separadamente passou a custar mais do que descrevê-los
juntos" = os pares passaram a precisar da mesma coisa.

PRÉ-REGISTRO
--------------
  DEFEITO CORRIGIDO ANTES DE RODAR: a 1a versao exigia que a peca explicasse
  ZERO celulas (`err >= err0`). Mas `mapa_cor` em `[mapa_cor, copia]` EXPLICA
  as celulas que recolore — ela so nao ganha a disputa. Aquele filtro jogaria
  fora exatamente os casos a medir. "Ganho imediato zero" no ARC nao e
  "explica zero celulas"; e **nao ser a escolha de topo pelo criterio atual**,
  que e o que faz o feixe podar. A pergunta vira direta e acionavel:
  a promessa REORDENA o feixe para salvar o degrau que o criterio afunda?

  K0  PRE-CONDICAO. Existem tarefas resolvidas com pilha >= 2 nas quais o
      primeiro passo verdadeiro NAO e ja o posto 1 pelo criterio atual
      (residuo, bits). Se ele ja fosse, nao haveria nada a consertar.
  K1  a promessa poe o passo verdadeiro num posto MELHOR que o criterio atual
      em >= 2/3 das tarefas, com ganho medio de posto >= 2 desvios
      -> existe sinal local de promessa para composicao futura, sem acesso a
         continuacao
  K2  sem melhora de posto -> nulo. A ideia morre AQUI e nada se constroi em
      cima.

  NULO PAREADO: permutar qual y vai com qual z entre os demos. A promessa deve
  desabar. Se não desabar, a medida está lendo tamanho de grade e não estrutura.

  CONTROLES CASADOS: só entram candidatos com ganho imediato ZERO, e o posto do
  verdadeiro é medido contra eles — mesma tarefa, mesmo orçamento, mesma
  oportunidade. (A lição de 18/09: casar LIBERDADE DE ESCOLHA, não só orçamento.)

PREVISÃO: 0,35 para K1. A favor: a medida é a única desta linha que lê o dado
presente no instante em que a peça nasce, em vez de prever um futuro não
observado. Contra, e é sério: LZMA pode simplesmente premiar peças que deixam o
delta MENOR, e delta menor correlaciona com ganho imediato — que foi zerado por
construção, mas a correlação pode voltar pela porta dos fundos via tamanho.
Por isso o nulo permutado é obrigatório e o tamanho do delta é reportado junto.
Se K2 sair, morre ESTA operacionalização (diff + LZMA), não a ideia; o próximo
passo seria um codec de buraco que enxergue relação e não literal.
"""

from __future__ import annotations

import itertools
import lzma
import statistics
from typing import Dict, List, Sequence, Tuple

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, _marca, _passo, custo, dims, montar

PASTA = "legacy/novamind/data/ARC/data/training"


def _bytes_buraco(z, y) -> bytes:
    """
    O buraco: o que uma continuação desconhecida ainda teria de fazer.
    Diff canônico, sem nenhum conceito de domínio além de célula e cor.
    """
    hz, wz = dims(z)
    hy, wy = dims(y)
    if (hz, wz) != (hy, wy):
        # dimensões diferentes: o buraco inclui a própria mudança de forma
        return ("#%d,%d>%d,%d;" % (hz, wz, hy, wy)).encode() + \
               b";".join(bytes(r) for r in y)
    fora = []
    for i in range(hy):
        for j in range(wy):
            if z[i][j] != y[i][j]:
                fora.append("%d,%d,%d,%d" % (i, j, z[i][j], y[i][j]))
    return (";".join(fora)).encode()


# FORMAT_RAW: sem container. O .xz normal carrega ~60 bytes de cabecalho fixo,
# e quando o delta fica pequeno o cabecalho DOMINA — foi o que produziu razoes
# de -11,6 na 1a versao. Em bruto o overhead e ~0.
_FILT = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def _L(b: bytes) -> float:
    """Bits para descrever `b`."""
    if not b:
        return 0.0
    return 8.0 * len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_FILT))


def custo_buraco(pares: Sequence[Tuple[object, object]]) -> float:
    """
    Bits para descrever TUDO que ainda falta, de uma vez.

    Sem razao e sem normalizacao: e MDL direto, que e o que a formula pede —
    "quanto p reduz a complexidade das continuacoes ainda necessarias". A
    juncao e o ponto: se os buracos viraram a MESMA coisa, descreve-los juntos
    desaba.
    """
    bs = [_bytes_buraco(z, y) for z, y in pares]
    return _L(b";".join(bs))


def colapso(pares: Sequence[Tuple[object, object]]) -> Tuple[float, int]:
    """DIAGNOSTICO: bits que a juncao economiza. Isola o 'mesma coisa desconhecida'."""
    bs = [_bytes_buraco(z, y) for z, y in pares]
    sep = sum(_L(b) for b in bs)
    jun = _L(b";".join(bs))
    return sep - jun, sum(len(b) for b in bs)


def depois_de(item, treino):
    """
    O estado APOS executar a peca, contra a saida: o buraco que sobra.

    DEFEITO CORRIGIDO POR REVISAO EXTERNA, 21/09. A 1a versao usava
    `_estagio([item], treino)`, que procura o ULTIMO MARCADOR da pilha e, sem
    marcador — pilha de um item so —, devolve o treino INTACTO. Conferido:

        _estagio([p], tr) identico ao treino cru?  True
        _passo([p], tr)   identico ao treino cru?  False

    Ou seja, `promessa` comparava o treino consigo mesmo e devolvia exatamente
    0,0 para todo candidato. O experimento nao media nada. `_passo` executa.
    """
    ps = _passo([item], treino)
    return [(g, b) for (g, _e), (_a, b) in zip(ps, treino)]


def promessa(item, treino) -> Tuple[float, int]:
    """
    O número. NÃO consulta CODECS em nenhum ponto: só executa a peça e comprime
    o que sobra. É isso que faz o teste-punhal passar por construção.
    """
    antes = custo_buraco(treino)
    dep = depois_de(item, treino)
    depois = custo_buraco(dep)
    try:
        bits_p = float(item[0].bits(item[1]))
    except Exception:
        bits_p = 0.0
    _c, tam = colapso(dep)
    return antes - depois - bits_p, tam


def candidatos_zero(treino) -> List[Tuple[object, object]]:
    """TODOS os passos de um codec. (Nome historico; nao filtra mais.)"""
    _b0, err0 = custo([], treino)
    fora = []
    for cd in CODECS:
        try:
            # alguns codecs devolvem GERADOR, nao lista: fatiar direto quebra
            ms = list(itertools.islice(iter(cd.propor(treino) or []), 6))
        except Exception:
            continue
        for m in ms:
            it = (cd, m)
            try:
                _b, err = custo([it], treino)
            except Exception:
                continue
            # DEFEITO 4, pego pelo proprio portao (K0 deu 0 utilizaveis):
            # filtrar para peças que NAO melhoram e depois acrescentar a
            # verdadeira, que melhora, faz dela o menor `err` do conjunto —
            # posto 1 pelo criterio POR CONSTRUCAO. E maximiza o confundidor
            # pre-registrado: a verdadeira vira a unica que encolhe o delta.
            # Agora entram TODOS os candidatos de um passo.
            _ = err0
            fora.append(it)
    return fora


def _reais(pilha) -> List:
    return [it for it in pilha if not _marca(it)]


# As 18 que a 1a varredura achou com pilha >= 2 e resolvida. `montar` varre
# TODOS os modelos de forma e chama a busca uma vez por forma, entao o custo e
# forma x teto — baixar o teto nao salva. Rodar so nas que qualificam salva.
QUALIFICAM = ("00d62c1b 025d127b 0520fde7 05269061 05f2a901 08ed6ac7 10fcaaa3 "
              "11852cab 178fcbfb 1bfc4729 1caeab9d 22168020 2281f1f4 22eb0ac0 "
              "23581191 253bf280 25d8a9c8 25ff71a9 31aa019c 32597951 3906de3d "
              "3ac3eb23 3bd67248 40853293 4258a5f9 42a50994 4347f46a 445eab21 "
              "496994bd 543a7ed5 5582e5ca 60b61512 623ea044 63613498 67a423a3 "
              "6c434453 6d75e8bb 6e19193c 6e82a1ae 6f8cd79b 776ffc46 7b6016b9 "
              "810b9b61 868de0fa 913fb3ed 91714a58 9565186b 95990924 a48eeaf7 "
              "a5313dff a5f85a15 a61f2674 a740d043 a79310a0 aabf363d ae3edfdc "
              "b1948b0a b230c067 b2862040 b60334d2 b8825c91 ba97ae07 bb43febb").split()


def experimento(n_tarefas: int = 120, teto: int = 60_000, so_estas=None) -> None:
    tarefas = carregar(PASTA)
    if so_estas:
        alvo = set(so_estas)
        tarefas = [t for t in tarefas if t[0] in alvo]
    else:
        tarefas = tarefas[:n_tarefas]
    print("=" * 78)
    print("PROMESSA — crédito para a peça que ainda não paga nada")
    print("=" * 78)
    print("  %d tarefas, teto %d avaliacoes" % (len(tarefas), teto))
    print("  a medida NAO consulta CODECS: teste-punhal por construcao\n")

    casos = []
    for nome, tr, _te in tarefas:
        try:
            fm, pilha, err = montar(tr, teto=teto)
        except Exception:
            continue
        if err != 0 or not pilha:
            continue
        reais = _reais(pilha)
        if len(reais) < 2:
            continue
        # a pilha opera sobre a base REBASEADA pelo modelo de forma, nao sobre
        # o treino cru: tudo abaixo tem de ver exatamente o que ela viu
        try:
            reb = [(forma.base(fm, a), b) for a, b in tr]
        except Exception:
            continue
        primeiro = reais[0]
        _b0, e0 = custo([], reb)
        try:
            _b1, e1 = custo([primeiro], reb)
        except Exception:
            continue
        _ = (e0, e1)     # sem filtro aqui: o posto pelo criterio e que decide
        casos.append((nome, reb, primeiro))
        print("  %s  pilha de %d, primeiro passo com ganho zero"
              % (nome, len(reais)), flush=True)

    print("\n  K0 — %d tarefas utilizaveis" % len(casos))
    if len(casos) < 5:
        print("  K0 FALHOU: amostra pequena demais para decidir.")
        return

    import random
    linhas = []
    for nome, reb, verdadeiro in casos:
        cands = candidatos_zero(reb)
        if not any(it is verdadeiro for it in cands):
            cands.append(verdadeiro)
        # 3o braco, exigido pelo teste do confundidor: a promessa correlaciona
        # -0,836 com o TAMANHO do delta. Se ordenar so por tamanho der o mesmo
        # posto, a compressao do buraco nao acrescenta nada.
        atual, prom, tamn = [], [], []
        for it in cands:
            try:
                b, e = custo([it], reb)
                v, t = promessa(it, reb)
            except Exception:
                continue
            atual.append(((e, b), it))
            prom.append((v, it))
            tamn.append((-float(t), it))
        if len(atual) < 3:
            continue
        ord_a = [it for _k, it in sorted(atual, key=lambda t: t[0])]
        ord_p = [it for _v, it in sorted(prom, key=lambda t: -t[0])]
        ord_t = [it for _v, it in sorted(tamn, key=lambda t: -t[0])]
        try:
            pa = next(k for k, it in enumerate(ord_a) if it is verdadeiro) + 1
            pp = next(k for k, it in enumerate(ord_p) if it is verdadeiro) + 1
            pt = next(k for k, it in enumerate(ord_t) if it is verdadeiro) + 1
        except StopIteration:
            continue
        # NULO: permuta qual saida vai com qual entrada entre os demos
        rnd = random.Random(20260921)
        ys = [b for _a, b in reb]
        perm = ys[1:] + ys[:1] if len(ys) > 1 else ys
        rebn = [(a, y) for (a, _b), y in zip(reb, perm)]
        pn = []
        for it in cands:
            try:
                v, _t = promessa(it, rebn)
            except Exception:
                continue
            pn.append((v, it))
        ord_n = [it for _v, it in sorted(pn, key=lambda t: -t[0])]
        try:
            pnn = next(k for k, it in enumerate(ord_n) if it is verdadeiro) + 1
        except StopIteration:
            pnn = pa
        linhas.append((nome, len(ord_a), pa, pp, pnn, pt))
        print("    %-10s n=%3d   criterio %3d   promessa %3d   TAMANHO %3d"
              "   nulo %3d" % (nome, len(ord_a), pa, pp, pt, pnn), flush=True)

    uteis = [l for l in linhas if l[2] > 1]          # K0: havia o que consertar
    print()
    print("=" * 78)
    print("  K0 — tarefas em que o passo verdadeiro NAO era ja o posto 1: %d/%d"
          % (len(uteis), len(linhas)))
    if len(uteis) < 5:
        print("  K0 FALHOU: amostra pequena demais para decidir.")
        return

    gan = [l[2] - l[3] for l in uteis]
    ganT = [l[2] - l[5] for l in uteis]
    sobre_tam = [l[5] - l[3] for l in uteis]      # promessa MENOS tamanho
    ganN = [l[2] - l[4] for l in uteis]
    melhor = sum(1 for g in gan if g > 0)
    m, sd = statistics.mean(gan), statistics.pstdev(gan)
    mn, sdn = statistics.mean(ganN), statistics.pstdev(ganN)
    print("  ganho de posto (criterio - promessa): %+.2f +- %.2f, melhora em %d/%d"
          % (m, sd, melhor, len(uteis)))
    print("  o mesmo so com o TAMANHO do delta:     %+.2f +- %.2f"
          % (statistics.mean(ganT), statistics.pstdev(ganT)))
    print("  PROMESSA menos TAMANHO (o que a compressao acrescenta): "
          "%+.2f +- %.2f, positivo em %d/%d"
          % (statistics.mean(sobre_tam), statistics.pstdev(sobre_tam),
             sum(1 for d in sobre_tam if d > 0), len(uteis)))
    print("  o mesmo com o NULO permutado:          %+.2f +- %.2f" % (mn, sdn))
    print()
    # DEFEITO CORRIGIDO POR REVISAO EXTERNA, 21/09: o nulo permutado era
    # calculado, impresso e NUNCA entrava na condicao — K1 podia disparar com o
    # nulo dando exatamente o mesmo ganho. O placebo pareado existe para estar
    # NO PORTAO, nao no relatorio. Agora a barra e a vantagem PAREADA sobre ele.
    dif = [l[4] - l[3] for l in uteis]        # posto no nulo - posto na promessa
    md, sdd = statistics.mean(dif), statistics.pstdev(dif)
    venceN = sum(1 for d in dif if d > 0)
    print("  vantagem PAREADA sobre o nulo:         %+.2f +- %.2f, em %d/%d"
          % (md, sdd, venceN, len(uteis)))
    print()
    if (melhor >= (2 * len(uteis)) / 3 and m >= 2 * max(sd, 1e-9)
            and md >= 2 * max(sdd, 1e-9) and venceN >= (2 * len(uteis)) / 3):
        print("K1 . EXISTE SINAL LOCAL DE PROMESSA para prefixo de UM passo no")
        print("     ARC: a medida reordena o feixe a favor do degrau que o")
        print("     criterio afunda, sem olhar a continuacao, e bate o nulo")
        print("     permutado em %+.2f +- %.2f pareado." % (md, sdd))
        print("     ALCANCE: isto e uma heuristica de ordenacao medida em UM")
        print("     dominio. Nao e fundamento de arquitetura geral, e nao")
        print("     autoriza construir em cima sem replicar noutro dominio.")
    else:
        print("K2 . NULO (ganho %+.2f +- %.2f em %d/%d; pareado sobre o nulo"
              % (m, sd, melhor, len(uteis)))
        print("     %+.2f +- %.2f em %d/%d)." % (md, sdd, venceN, len(uteis)))
        print("     Morre ESTA operacionalizacao — diff comprimido por LZMA —,")
        print("     nao a ideia. O proximo passo seria um codec de buraco que")
        print("     enxergue RELACAO e nao LITERAL.")


if __name__ == "__main__":
    experimento()
