"""
novaphys.xadrez_digestora — extrair nutriente cognitivo de gente jogando xadrez
================================================================================

A COBRANCA (usuario, 24/09)
-----------------------------
"falta extracao de nutriente cognitivo" de mundo humano, "sem um deus escrevendo
a mao". Em todo sucesso ate aqui EU escolhi as variaveis ([[mente-inteira-arc2]]).
Este e o primeiro teste do laco de extracao em dado de GENTE: partidas reais do
Lichess (2013, CC0), com um verificador perfeito (as regras) e conceitos humanos
NOMEADOS guardados como gabarito cego (os temas do banco de problemas: garfo,
cravada, espeto, ataque descoberto, xeque duplo, peca pendurada).

O QUE A MAQUINA RECEBE — dito por inteiro
-------------------------------------------
  - as REGRAS (lances legais, quem ataca quem): e o que se diz a quem aprende
  - UMA operacao generica de contrafactual: "o que esta peca atacaria se a da
    frente nao estivesse la" (serve para qualquer mundo com simulador)
  - o tipo de cada peca, sem VALOR (ninguem diz que dama vale mais que cavalo)
O que NAO recebe: nenhum tema, nenhum valor, nenhuma nocao de lance bom.

A EXTRACAO — um principio, nao um vocabulario
-----------------------------------------------
Cada lance e um EVENTO: o que ele cria de relacao nova no tabuleiro (a peca que
andou passa a atacar X, uma outra peca minha passa a atacar Y porque a linha
abriu, o inimigo passa a atacar a que andou...). Relacao abstraida so por quem
(a que andou / outra minha / do inimigo), o tipo de relacao (ataca, defende,
ataca-atraves, captura), o tipo e o lado do alvo, e se o alvo esta defendido.

Candidato a nutriente = uma relacao criada, ou um PAR de relacoes criadas pelo
mesmo lance (a menor "relacao entre relacoes"; o teto em pares e orcamento de
computacao, declarado aqui).

RELEVANCIA vem das ESCOLHAS HUMANAS — o sinal que faltou no HAR
([[protocolo-nao-percepcao]]). Atracao A(c) = quantas vezes o humano escolheu um
lance com c / quantas vezes seria escolhido se ele sorteasse entre os lances
legais. Admissao (os dois portoes que o projeto ja validou, mais o de chunk):
  - REPETE: em >= 30 partidas distintas de treino (jan/2013)
  - PREVE FORA: A >= 2 no treino (1 bit, o limiar do lexico) E A >= 1,5 em
    partidas de OUTRO mes (fev/2013) que a extracao nunca viu
  - PAR SO SE ACRESCENTA: A(par) >= 1,25 x a maior A das partes

PLACEBOS
----------
  N2  mesmas posicoes, mesmas regras, mas o lance "escolhido" e um lance legal
      SORTEADO: mundo igual, sem intencao humana. Deve admitir ~nada.
  N1  para a ancora: 200 sorteios de conjuntos do MESMO tamanho, tirados do
      mesmo reservatorio de candidatos e casados em frequencia.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  X0  PRE-CONDICAO: >= 10 nutrientes admitidos das partidas humanas; o placebo
      N2 admite <= 2.
  X1  ANCORA: para >= 3 dos 6 temas, o melhor "lift" (P(tema | nutriente
      dispara em algum lance do RESOLVEDOR na solucao) / P(tema)) — ajustado
      antes de rodar: no garfo, cravada e xeque duplo o motivo muitas vezes
      esta no 2o ou 3o lance da solucao, nao no 1o — entre os admitidos passa o
      percentil 95 de N1 -> a extracao guiada por relevancia humana ACHA o que
      humanos nomeiam, sem nunca ter visto o nome
  X2  1 ou 2 temas
  X3  zero -> nulo: o que a escolha humana atrai nao e o que humanos nomeiam
  Y   (secundario) no mes nao visto, somar os nutrientes ao baseline de
      superficie (tipo da peca movida x tipo capturado) melhora a posicao do
      lance humano no ranking, com IC de bootstrap pareado acima de zero.

PREVISAO: X1 a 0,40, X2 a 0,35, X3 a 0,25; Y passa a 0,55. Minha calibragem:
17 de 27 previsoes erradas, todas otimistas.

DEFEITO DE INSTRUMENTO ACHADO DEPOIS DA 1a RODADA: Y foi medido nas partidas de
fevereiro, as MESMAS que o portao "preve fora" usou para admitir — selecao
contaminada. Corrigido: fevereiro so admite, Y e medido em MARCO, que a extracao
nunca toca. O veredito da ancora (X) nao usa fevereiro e nao muda.

RESULTADO DA 1a PERGUNTA (24/09): X3 (0 de 6), Y PASSA (+0,091 em marco, IC
+0,086 a +0,095). Pos-hoc: os conceitos EXISTEM no reservatorio — os padroes
mais alinhados a cada tema sao a definicao de livro (cravada = "ataca o rei
ATRAVES de cavalo/bispo"; espeto = "ataca atraves da dama a torre de tras";
ataque descoberto = "OUTRA peca minha passa a atacar") — mas a ATRACAO das
escolhas nao acompanha tema nenhum (rho -0,18 a +0,18). Amador escolhe captura e
xeque; conceito humano tem nome pelo que PRODUZ depois.

CONSEQUENCIA — 2a pergunta, pre-registrada ANTES de rodar
-----------------------------------------------------------
Relevancia pelo DESFECHO ADIADO, nao pela frequencia da escolha
([[cegueira-do-ganho-imediato]]; na linguagem aterrada o sentido veio da
consequencia). Desfecho sem VALOR de peca: D_k = (pecas que o inimigo perde) -
(pecas que eu perco) nos k meios-lances depois do lance, k = 4 (2 e 6 so
relatados). V(c) = media de D_4 nos lances humanos com c - media geral.
  C0  PRE-CONDICAO: >= 10 padroes com V >= +0,25 no treino E >= +0,15 em
      fevereiro, em >= 30 partidas; placebo (D embaralhado entre posicoes)
      admite <= 2.
  C1  no reservatorio, rho(V, lift do tema) >= +0,25 para >= 3 dos 6 temas ->
      o desfecho adiado e o sinal de relevancia que recupera o que humanos
      nomeiam (a atracao deu -0,18 a +0,18)
  C2  1 ou 2 temas
  C3  nenhum -> desfecho de pecas tambem nao e o que da nome aos conceitos
PREVISAO: C1 a 0,35, C2 a 0,40, C3 a 0,25.

RESULTADO CORRIGIDO: C2 de novo — so peca pendurada (+0,31); cravada -0,24,
ataque descoberto -0,34. Amador nao converte cravada em peca.

IMAGINACAO — 3a pergunta, pre-registrada ANTES de rodar
---------------------------------------------------------
Conceito tem nome pelo que produz sob JOGO BOM, que amador nao mostra. Entao a
consequencia vem da SIMULACAO, com as regras como modelo de mundo — e o VALOR das
pecas tambem tem de ser extraido, nao dado:
  VALOR   regressao logistica do resultado da partida na diferenca de
          quantidade de cada tipo de peca (lance 30), 20.000 partidas de
          jan-mar/2013. Ancora cega: a tabela humana 1/3/3/5/9.
    V1  ordem P < {C,B} < T < D certa E cada valor a +-40% da tabela humana
    V2  ordem certa, valores fora da faixa | V3 ordem errada
    PLACEBO: resultados embaralhados entre partidas -> sem ordem
  IMAGINADO  para cada lance legal (nao so o humano) de 3.000 posicoes reais:
          meu lance, TODAS as respostas do adversario, e depois so capturas ate
          aquietar (quiescencia), avaliado com os valores EXTRAIDOS. Valor do
          lance = pior caso sobre as respostas. V_sim(c) = media, sobre lances
          com c, de (valor do lance - media dos lances da posicao).
    S1  rho(V_sim, lift do tema) >= +0,25 para >= 3 dos 6 temas -> simular com
        o que se extraiu recupera o que humanos nomeiam (atracao -0,18..+0,18;
        desfecho de amador: 1 tema)
    S2  1 ou 2 | S3 nenhum
    PLACEBO: valores de lance embaralhados dentro da posicao -> rho ~ 0
    ABLACAO (relatada): todas as pecas valendo 1 — o valor extraido importa?
  PREVISAO: V1 a 0,65; S1 a 0,45, S2 a 0,35, S3 a 0,20.

RESULTADO DA 3a (24/09): VALOR V1 — de 21.000 partidas, sem a tabela: peao 1,
cavalo 2,05, bispo 2,51, torre 3,13, dama 6,29 (ordem certa, bispo > cavalo como
dizem os fortes; escala comprimida). IMAGINACAO S3 — rho <= 0,05 em todos os
temas, igual ao placebo: o topo e "captura a dama indefesa"; o ganho IMEDIATO
domina de novo. Tres sinais de relevancia (escolha, desfecho observado, desfecho
imaginado) falharam do MESMO jeito: magnitude de ganho nao escolhe conceito.

CAUSA — 4a pergunta, pergunta NOVA (nao e resgate), pre-registrada ANTES de rodar
-----------------------------------------------------------------------------------
Hipotese: humano nomeia a CAUSA MINIMA de um ganho ADIADO, nao o ganho. Abducao
como na fisica ([[abducao-construir-nao-propor]]): o termo que explica a anomalia.
So lances SEM captura; G = valor imaginado - media da posicao (valores extraidos);
causa(c) = E(c) = P(c | G >= +1,5) / P(c | lance sem captura). Dados NOVOS:
posicoes de fevereiro, que a imaginacao nunca tocou.
  D1  rho(log E, lift do tema) >= +0,25 para >= 3 dos 5 temas sem captura (garfo,
      cravada, espeto, descoberto, xeque duplo) -> conceito = causa de ganho adiado
  D2  1 ou 2 | D3 nenhum
  PLACEBO: G embaralhado dentro da posicao.
PREVISAO: D1 a 0,40, D2 a 0,35, D3 a 0,25.

RESULTADO DA 4a (24/09): D3 — rho <= +0,16; e o PLACEBO da -0,22 no ataque
descoberto, quase igual ao real (-0,29): a estatistica "rho sobre o reservatorio
inteiro" tem estrutura que nao vem do sinal. Parte dos quatro nulos e a REGUA.
Paro de cacar sinal de relevancia aqui (padrao de reenquadrar otimista).

NOMEAR — 5a pergunta, pre-registrada ANTES de rodar
-----------------------------------------------------
O genio da vila aprende a PALAVRA "garfo" de um professor, com POUCOS exemplos —
nome e transmissao cultural, nao descoberta solitaria. E o principio do
construtor ([[construtor]]): comparar exemplos e ficar com o que se CONSERVA.
Para cada tema: 5 problemas com o tema (tirados de 10.000 de referencia); o
conceito e o padrao presente em >= 4 dos 5, o mais RARO em geral (Occam por
bits); reconhece em 20.000 problemas de teste nunca vistos. 20 sorteios.
  LINHA DE BASE: o mesmo procedimento sobre SUPERFICIE (que peca andou, qual foi
  tomada). PLACEBO: 5 problemas QUAISQUER rotulados com o nome.
  N1  lift medio >= 2, acima da superficie E acima do p95 do placebo em >= 4 dos
      6 temas -> a representacao extraida + comparacao aprende o NOME como humano
  N2  2 ou 3 | N3 <= 1
PREVISAO: N1 a 0,50, N2 a 0,35, N3 a 0,15.

RESULTADO DA 5a (24/09): N2 (2 de 6). Espeto lift 13,6 e xeque duplo 17,7 com
5 exemplos (superficie 1,9; placebo 1,9). Garfo e cravada caem no generico
("da xeque") porque a peca cravada VARIA entre os exemplos e cada tipo e um
padrao diferente — nenhum se repete em 4 de 5.

NOMEAR COM VARIAVEL — 6a pergunta, pre-registrada ANTES de rodar
------------------------------------------------------------------
A operacao do carimbo do ARC levada a outro mundo: cada atributo da relacao
(quem, tipo do alvo, peca da frente, defendido) EXATO ou QUALQUER; depois de
generalizar, duas relacoes distintas que viram a mesma forma geral marcam "*2".
Mesma regra de conceito (conservado em >= 4 de 5, o mais raro), mesmas linhas de
base e placebo, e DADOS NOVOS: o bloco seguinte de 40.000 problemas (linhas
40.000-80.000), que nenhuma pergunta avaliou.
  N1  >= 4 dos 6 temas | N2 2 ou 3 | N3 <= 1 (mesmos criterios da 5a)
PREVISAO: N1 a 0,45, N2 a 0,40, N3 a 0,15.

1a RODADA (24/09) COM DEFEITO: o codigo contava a captura do PROPRIO lance em
D_k, contra o texto acima ("depois do lance") — o ganho imediato de novo, que era
o que esta pergunta excluia. Deu C2 (so espeto, +0,29). Corrigido para comecar
DEPOIS do lance; o numero que vale e o da rodada corrigida.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

PASTA = r"D:\agi_dados\xadrez"
TREINO = "lichess_db_standard_rated_2013-01.pgn.zst"
FORA = "lichess_db_standard_rated_2013-02.pgn.zst"
MARCO = "lichess_db_standard_rated_2013-03.pgn.zst"      # so para Y
PROBLEMAS = "lichess_db_puzzle.csv.zst"
TEMAS = ("fork", "pin", "skewer", "discoveredAttack", "doubleCheck", "hangingPiece")
SAIDA = "resultados/xadrez_digestora.json"


# ============================================================ dados crus
def _texto(arq: str):
    import zstandard
    f = open(os.path.join(PASTA, arq), "rb")
    return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f),
                            encoding="utf-8")


def jogos(arq: str, n: int) -> List[List[str]]:
    """As n primeiras partidas com >= 20 lances, como listas de UCI."""
    import chess.pgn
    r = _texto(arq)
    out = []
    while len(out) < n:
        g = chess.pgn.read_game(r)
        if g is None:
            break
        ms = [m.uci() for m in g.mainline_moves()]
        if len(ms) >= 20:
            out.append(ms)
    return out


def problemas(n: int) -> List[Tuple[str, List[str], set]]:
    import csv
    r = csv.reader(_texto(PROBLEMAS))
    next(r)
    out = []
    for lin in r:
        out.append((lin[1], lin[2].split(), set(lin[7].split())))
        if len(out) >= n:
            break
    return out


# =============================================== o evento: relacoes criadas
def relacoes(b) -> set:
    """Quem ataca/defende quem, e quem ataca ATRAVES de quem (contrafactual)."""
    import chess
    out = set()
    b = b.copy(stack=False)       # tirar peca limpa a pilha de lances: nunca no original
    pm = b.piece_map()
    for a, p in pm.items():
        ats = b.attacks(a)
        for t in ats:
            q = pm.get(t)
            if q is None:
                continue
            if q.color == p.color:
                out.add((a, t, "=", ""))
                continue
            out.add((a, t, ">", ""))
            if p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                b.remove_piece_at(t)                  # "e se a da frente sumisse?"
                novo = b.attacks(a)
                b.set_piece_at(t, q)
                for u in novo:
                    r = pm.get(u)
                    if u != t and u not in ats and r is not None and r.color != p.color:
                        out.add((a, u, "~", q.symbol().lower()))
    return out


def assinatura(b, mv, antes: Optional[set] = None) -> frozenset:
    """O que o lance CRIA de relacao, abstraido de casa e de identidade."""
    cor = b.turn
    if antes is None:
        antes = relacoes(b)
    itens = Counter()
    if b.is_capture(mv):
        sq = mv.to_square
        if b.is_en_passant(mv):
            sq = mv.to_square + (-8 if cor else 8)
        cap = b.piece_at(sq)
        itens["Mx%s:%s" % (cap.symbol().lower(),
                           "d" if b.is_attacked_by(not cor, sq) else "n")] += 1
    f, t_ = mv.from_square, mv.to_square

    def ren(s):
        return t_ if s == f else s
    antes_r = {(ren(a), ren(t), k, e) for a, t, k, e in antes}
    b.push(mv)
    try:
        for a, t, k, e in relacoes(b) - antes_r:
            pa, pt = b.piece_at(a), b.piece_at(t)
            quem = "M" if a == t_ else ("m" if pa.color == cor else "d")
            lado = "M" if t == t_ else ("P" if pt.color == cor else "I")
            it = quem + k + lado + pt.symbol().lower() + e
            if k != "=":
                it += ":d" if b.is_attacked_by(pt.color, t) else ":n"
            itens[it] += 1
    finally:
        b.pop()
    s = set(itens)
    s |= {it + "*2" for it, c in itens.items() if c >= 2}
    return frozenset(s)


def pedacos(sig: frozenset) -> set:
    """Candidatos: cada relacao criada, e cada par de relacoes do mesmo lance."""
    s = sorted(sig)
    out = {(x,) for x in s}
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            out.add((s[i], s[j]))
    return out


def superficie(b, mv) -> str:
    p = b.piece_at(mv.from_square).symbol().lower()
    c = b.piece_at(mv.to_square) if b.is_capture(mv) else None
    return p + ("x" + c.symbol().lower() if c else "") if not b.is_en_passant(mv) \
        else p + "xp"


# ======================================================== o trabalhador
def _partida(args):
    """Conta, numa partida, o que o humano escolheu e o que estava disponivel."""
    import chess
    k, ms, semente, guarda = args
    rnd = random.Random(semente)
    b = chess.Board()
    brancas, pretas = [], []                  # pecas de cada lado antes de cada lance
    bb = chess.Board()
    for u in ms:
        brancas.append(bin(bb.occupied_co[chess.WHITE]).count("1"))
        pretas.append(bin(bb.occupied_co[chess.BLACK]).count("1"))
        bb.push(chess.Move.from_uci(u))
    brancas.append(bin(bb.occupied_co[chess.WHITE]).count("1"))
    pretas.append(bin(bb.occupied_co[chess.BLACK]).count("1"))
    desfechos = []
    plies = [i for i in range(6, min(len(ms), 80))]
    alvo = set(rnd.sample(plies, min(8, len(plies))))
    esc, disp, sorteio = Counter(), Counter(), Counter()
    sup_esc, sup_disp = Counter(), Counter()
    posicoes = []
    for i, u in enumerate(ms):
        mv = chess.Move.from_uci(u)
        if i in alvo:
            legais = list(b.legal_moves)
            if len(legais) >= 2 and mv in legais:
                antes = relacoes(b)
                sigs = [assinatura(b, m, antes) for m in legais]
                sups = [superficie(b, m) for m in legais]
                h = legais.index(mv)
                r = rnd.randrange(len(legais))
                cont = Counter()
                for sg in sigs:
                    cont.update(pedacos(sg))
                n = len(legais)
                for c, v in cont.items():
                    disp[c] += v / n
                esc.update(pedacos(sigs[h]))
                sorteio.update(pedacos(sigs[r]))
                for s_, v in Counter(sups).items():
                    sup_disp[s_] += v / n
                sup_esc[sups[h]] += 1
                eu_br = b.turn
                d_k = []
                for kk in (2, 4, 6):            # DEPOIS do lance: o proprio lance nao conta
                    i1 = i + 1
                    j = min(i1 + kk, len(ms))
                    meu = (brancas if eu_br else pretas)
                    dele = (pretas if eu_br else brancas)
                    d_k.append((dele[i1] - dele[j]) - (meu[i1] - meu[j]))
                desfechos.append((sigs[h], d_k))
                if guarda:
                    posicoes.append((sigs, sups, h))
        b.push(mv)
    return k, esc, disp, sorteio, sup_esc, sup_disp, posicoes, desfechos


def _problema(args):
    import chess
    fen, ms, temas = args
    try:
        b = chess.Board(fen)
        sig = set()
        for i, u in enumerate(ms):                    # pares: adversario; impares: resolvedor
            mv = chess.Move.from_uci(u)
            if i % 2 == 1:
                sig |= assinatura(b, mv)
            b.push(mv)
        return frozenset(sig), temas
    except Exception:
        return None, temas


# ============================================================ o experimento
def _conta(res):
    esc, disp, srt = Counter(), Counter(), Counter()
    jogos_de, jogos_srt = defaultdict(set), defaultdict(set)
    sup_e, sup_d = Counter(), Counter()
    pos = []
    for k, e, d, s, se, sd, p, _des in res:
        esc.update(e)
        disp.update(d)
        srt.update(s)
        for c in e:
            jogos_de[c].add(k)
        for c in s:
            jogos_srt[c].add(k)
        sup_e.update(se)
        sup_d.update(sd)
        pos += p
    return esc, disp, srt, jogos_de, jogos_srt, sup_e, sup_d, pos


def _admite(esc, disp, jogos_de, esc_fora, disp_fora, min_jogos=30):
    A = {c: esc[c] / disp[c] for c in esc if disp[c] > 0}
    A_f = {c: esc_fora[c] / disp_fora[c] for c in esc_fora if disp_fora.get(c, 0) > 0}
    pool = [c for c in A if len(jogos_de[c]) >= min_jogos]
    adm = []
    for c in pool:
        if A[c] < 2 or A_f.get(c, 0) < 1.5:
            continue
        if len(c) == 2 and A[c] < 1.25 * max(A.get((c[0],), 0), A.get((c[1],), 0)):
            continue
        adm.append(c)
    return adm, pool, A, A_f


def _lifts(conj, dispara, por_tema, base, min_n=30):
    """Melhor lift por tema entre os nutrientes do conjunto (indice invertido)."""
    melhor = {t: (1.0, None) for t in TEMAS}
    for c in conj:
        n = dispara.get(c, 0)
        if n < min_n:
            continue
        for t in TEMAS:
            lf = por_tema[c][t] / n / base[t] if base[t] else 0
            if lf > melhor[t][0]:
                melhor[t] = (lf, c)
    return melhor


def main(n_treino=3000, n_fora=1500, n_prob=40000, trab=6, n_marco=1500):
    from multiprocessing import Pool
    t0 = time.time()
    print("lendo partidas...", flush=True)
    J = jogos(TREINO, n_treino)
    F = jogos(FORA, n_fora)
    M = jogos(MARCO, n_marco)
    P = problemas(n_prob)
    print("  %d partidas de treino (jan/2013), %d fora (fev/2013), %d problemas"
          % (len(J), len(F), len(P)), flush=True)
    with Pool(trab) as pool:
        rt = pool.map(_partida, [(k, ms, k, False) for k, ms in enumerate(J)], 20)
        print("  treino contado | %.0f s" % (time.time() - t0), flush=True)
        rf = pool.map(_partida, [(k, ms, 10 ** 6 + k, False) for k, ms in enumerate(F)], 20)
        rm = pool.map(_partida, [(k, ms, 2 * 10 ** 6 + k, True) for k, ms in enumerate(M)], 20)
        print("  fora contado | %.0f s" % (time.time() - t0), flush=True)
        rp = pool.map(_problema, P, 200)
        print("  problemas | %.0f s" % (time.time() - t0), flush=True)
    esc, disp, srt, jd, jd_srt, sup_e, sup_d, _ = _conta(rt)
    esc_f, disp_f, srt_f, _jf, _js, _se, _sd, _pf = _conta(rf)
    pos_f = _conta(rm)[-1]                        # Y: marco, nunca tocado

    adm, pool_c, A, A_f = _admite(esc, disp, jd, esc_f, disp_f)
    # placebo N2: o "escolhido" e sorteado — mesmas regras, sem intencao
    adm_n2, _p2, _A2, _ = _admite(srt, disp, jd_srt, srt_f, disp_f)
    avers = sorted((c for c in pool_c if A[c] <= 0.5 and A_f.get(c, 9) <= 0.67),
                   key=lambda c: A[c])

    sig_p = [s for s, _t in rp]
    temas_p = [t for _s, t in rp]
    n_ok = sum(s is not None for s in sig_p)
    base = {t: sum(t in tt for s, tt in rp if s is not None) / max(n_ok, 1) for t in TEMAS}
    no_pool = set(pool_c)
    dispara, por_tema = Counter(), defaultdict(Counter)
    for sg, tt in rp:
        if sg is None:
            continue
        for c in pedacos(sg):
            if c in no_pool:
                dispara[c] += 1
                for t in TEMAS:
                    if t in tt:
                        por_tema[c][t] += 1
    real = _lifts(adm, dispara, por_tema, base)

    # N1: sorteios do mesmo reservatorio, casados em frequencia (quintis de jogos)
    rnd = random.Random(20260924)
    q = sorted(pool_c, key=lambda c: len(jd[c]))
    faixas = [q[i * len(q) // 5:(i + 1) * len(q) // 5] for i in range(5)]
    faixa_de = {c: i for i, f in enumerate(faixas) for c in f}
    pedido = Counter(faixa_de[c] for c in adm)
    nulos = {t: [] for t in TEMAS}
    for _ in range(200):
        conj = [c for i, n in pedido.items() for c in rnd.sample(faixas[i], min(n, len(faixas[i])))]
        L = _lifts(conj, dispara, por_tema, base)
        for t in TEMAS:
            nulos[t].append(L[t][0])
    passa = []
    print("=" * 78)
    print("XADREZ — nutrientes extraidos de partidas humanas | %.0f s" % (time.time() - t0))
    print("=" * 78)
    print("  reservatorio (>=30 partidas): %d | ADMITIDOS %d | placebo N2 admitiu %d"
          % (len(pool_c), len(adm), len(adm_n2)))
    print("  aversoes (humano evita, A<=0,5 nos dois meses): %d" % len(avers))
    for c in sorted(adm, key=lambda c: -A[c])[:25]:
        print("    A=%5.1f fora=%5.1f  %d partidas  %s" % (A[c], A_f.get(c, 0), len(jd[c]), " + ".join(c)))
    print("  ANCORA (temas humanos, nunca vistos; %d problemas):" % n_ok)
    for t in TEMAS:
        lf, c = real[t]
        p95 = sorted(nulos[t])[int(0.95 * len(nulos[t])) - 1]
        ok = lf > p95
        passa.append(ok)
        print("    %-17s base %.3f | lift %.2f (N1 p95 %.2f) %s  %s"
              % (t, base[t], lf, p95, "PASSA" if ok else "-----",
                 " + ".join(c) if c else ""))
    n_passa = sum(passa)
    # POS-HOC, rotulado como tal: o maximo de um conjunto de 334 nao separa de
    # sorteios do mesmo tamanho (27% do reservatorio). Pergunta sem esse ponto
    # cego: no reservatorio inteiro, a ATRACAO humana acompanha o alinhamento
    # com o tema? (Spearman entre log A e lift, padroes com >= 30 disparos)
    def _posto(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k_, i in enumerate(o):
            r[i] = k_
        return r
    print("  POS-HOC (nao e veredito): atracao x alinhamento no reservatorio")
    for t in TEMAS:
        cs = [c for c in pool_c if dispara.get(c, 0) >= 30]
        a_ = _posto([math.log(A[c]) for c in cs])
        l_ = _posto([por_tema[c][t] / dispara[c] / base[t] for c in cs])
        n_ = len(cs)
        ma, ml = sum(a_) / n_, sum(l_) / n_
        cov = sum((x - ma) * (y - ml) for x, y in zip(a_, l_))
        va = sum((x - ma) ** 2 for x in a_) ** .5
        vl = sum((y - ml) ** 2 for y in l_) ** .5
        top = sorted(cs, key=lambda c: -por_tema[c][t] / dispara[c])[:3]
        print("    %-17s rho %+.2f (n=%d) | mais alinhados: %s" % (
            t, cov / (va * vl), n_, " ; ".join(" + ".join(c) for c in top)))
    x0 = len(adm) >= 10 and len(adm_n2) <= 2

    # Y: o lance humano no mes nao visto, superficie vs superficie + nutriente
    As = {s: sup_e[s] / sup_d[s] for s in sup_e if sup_d[s] > 0}
    admset = set(adm)
    ganhos = []
    for sigs, sups, h in pos_f:
        def pr(scores):
            v = scores[h]
            maior = sum(x > v for x in scores)
            igual = sum(x == v for x in scores) - 1
            return 1 - (maior + igual / 2) / (len(scores) - 1)
        s0 = [math.log(As.get(s, 0.5)) for s in sups]
        s1 = []
        for s, sg in zip(s0, sigs):
            best = max((math.log(A[c]) for c in pedacos(sg) if c in admset), default=0.0)
            s1.append(s + best)
        ganhos.append(pr(s1) - pr(s0))
    rb = random.Random(7)
    boots = sorted(sum(rb.choice(ganhos) for _ in ganhos) / len(ganhos) for _ in range(1000))
    media = sum(ganhos) / len(ganhos)
    y = boots[25] > 0
    print("  Y: posicao do lance humano (0-1) no mes nao visto: ganho %+.4f "
          "(IC95 %+.4f a %+.4f) em %d posicoes -> %s"
          % (media, boots[25], boots[974], len(ganhos), "PASSA" if y else "nao passa"))
    veredito = ("X0 FALHOU" if not x0 else "X1" if n_passa >= 3 else
                "X2" if n_passa >= 1 else "X3")
    print("  VEREDITO: %s (%d de 6 temas) | Y %s" % (veredito, n_passa, "passa" if y else "nao"))
    os.makedirs("resultados", exist_ok=True)
    json.dump({"admitidos": [[list(c), A[c], A_f.get(c), len(jd[c])] for c in adm],
               "n2": [list(c) for c in adm_n2],
               "aversoes": [[list(c), A[c]] for c in avers[:50]],
               "ancora": {t: [real[t][0], list(real[t][1]) if real[t][1] else None,
                              sorted(nulos[t])[189]] for t in TEMAS},
               "Y": [media, boots[25], boots[974]], "veredito": veredito},
              open(SAIDA, "w"), indent=1)


def _rho(x, y):
    def posto(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k_, i in enumerate(o):
            r[i] = k_
        return r
    a_, l_ = posto(x), posto(y)
    n_ = len(a_)
    ma, ml = sum(a_) / n_, sum(l_) / n_
    cov = sum((u - ma) * (w - ml) for u, w in zip(a_, l_))
    va = sum((u - ma) ** 2 for u in a_) ** .5
    vl = sum((w - ml) ** 2 for w in l_) ** .5
    return cov / (va * vl) if va and vl else 0.0


def _valor(desfechos, ik=1, embaralha=None):
    """V(c) = media de D_k nos lances humanos com c - media geral; e as partidas."""
    ds = [d[ik] for _sg, d in desfechos]
    if embaralha is not None:
        embaralha.shuffle(ds)
    media = sum(ds) / len(ds)
    soma, n = Counter(), Counter()
    for (sg, _d), dk in zip(desfechos, ds):
        for c in pedacos(sg):
            soma[c] += dk
            n[c] += 1
    return {c: soma[c] / n[c] - media for c in n}, n


def consequencia(n_treino=3000, n_fora=1500, n_prob=40000, trab=6):
    from multiprocessing import Pool
    t0 = time.time()
    J, F, P = jogos(TREINO, n_treino), jogos(FORA, n_fora), problemas(n_prob)
    with Pool(trab) as pool:
        rt = pool.map(_partida, [(k, ms, k, False) for k, ms in enumerate(J)], 20)
        rf = pool.map(_partida, [(k, ms, 10 ** 6 + k, False) for k, ms in enumerate(F)], 20)
        rp = pool.map(_problema, P, 200)
    print("contado | %.0f s" % (time.time() - t0), flush=True)
    _e, _d, _s, jd, _js, _se, _sd, _p = _conta(rt)
    des_t = [x for r in rt for x in r[7]]
    des_f = [x for r in rf for x in r[7]]
    pool_c = [c for c in jd if len(jd[c]) >= 30]
    sig_p = [sg for sg, _t in rp]
    n_ok = sum(sg is not None for sg in sig_p)
    base = {t: sum(t in tt for sg, tt in rp if sg is not None) / n_ok for t in TEMAS}
    no_pool = set(pool_c)
    dispara, por_tema = Counter(), defaultdict(Counter)
    for sg, tt in rp:
        if sg is None:
            continue
        for c in pedacos(sg):
            if c in no_pool:
                dispara[c] += 1
                for t in TEMAS:
                    if t in tt:
                        por_tema[c][t] += 1
    print("=" * 78)
    print("XADREZ — relevancia pelo DESFECHO ADIADO (pecas, sem valor) | %.0f s"
          % (time.time() - t0))
    print("=" * 78)
    veredito = None
    for ik, k in enumerate((2, 4, 6)):
        V, nV = _valor(des_t, ik)
        Vf, _ = _valor(des_f, ik)
        adm = [c for c in pool_c if V.get(c, 0) >= 0.25 and Vf.get(c, 0) >= 0.15]
        Vp, _ = _valor(des_t, ik, random.Random(99))
        Vpf, _ = _valor(des_f, ik, random.Random(98))
        adm_p = [c for c in pool_c if Vp.get(c, 0) >= 0.25 and Vpf.get(c, 0) >= 0.15]
        cs = [c for c in pool_c if dispara.get(c, 0) >= 30 and c in V]
        rhos = {t: _rho([V[c] for c in cs],
                        [por_tema[c][t] / dispara[c] / base[t] for c in cs]) for t in TEMAS}
        n1 = sum(r >= 0.25 for r in rhos.values())
        print("  k=%d%s | admitidos %d (placebo %d) | rho por tema: %s | >= +0,25: %d"
              % (k, " (PRE-REGISTRADO)" if k == 4 else "", len(adm), len(adm_p),
                 " ".join("%s %+.2f" % (t[:6], r) for t, r in rhos.items()), n1))
        if k == 4:
            c0 = len(adm) >= 10 and len(adm_p) <= 2
            veredito = ("C0 FALHOU" if not c0 else "C1" if n1 >= 3 else
                        "C2" if n1 >= 1 else "C3")
            for c in sorted(adm, key=lambda c: -V[c])[:15]:
                print("      V=%+.2f fev=%+.2f %4d partidas  %s"
                      % (V[c], Vf[c], len(jd[c]), " + ".join(c)))
    print("  VEREDITO (k=4): %s" % veredito)


# ================================================ 3a pergunta: VALOR e IMAGINACAO
MESES_VALOR = ("lichess_db_standard_rated_2013-01.pgn.zst",
               "lichess_db_standard_rated_2013-02.pgn.zst",
               "lichess_db_standard_rated_2013-03.pgn.zst")
TIPOS = (1, 2, 3, 4, 5)          # peao, cavalo, bispo, torre, dama (python-chess)
HUMANO = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}   # ANCORA CEGA: so o relatorio le isto


def _com_resultado(arq, n, ply=30):
    import chess
    import chess.pgn
    r = _texto(arq)
    X, y = [], []
    while len(y) < n:
        g = chess.pgn.read_game(r)
        if g is None:
            break
        res = g.headers.get("Result")
        if res not in ("1-0", "0-1"):
            continue
        b = g.board()
        ms = list(g.mainline_moves())
        if len(ms) <= ply:
            continue
        for m in ms[:ply]:
            b.push(m)
        X.append([len(b.pieces(t, chess.WHITE)) - len(b.pieces(t, chess.BLACK))
                  for t in TIPOS])
        y.append(1.0 if res == "1-0" else 0.0)
    return X, y


def _logistica(X, y, iters=50):
    import numpy as np
    X = np.column_stack([np.ones(len(X)), np.array(X, float)])
    y = np.array(y)
    w = np.zeros(X.shape[1])
    for _ in range(iters):                     # Newton (IRLS)
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, X.T @ (y - p))
    return w[1:]


def valores(n_por_mes=7000, verbose=True):
    """O VALOR das pecas, extraido de quem ganhou — sem a tabela humana."""
    X, y = [], []
    for arq in MESES_VALOR:
        a, b = _com_resultado(arq, n_por_mes)
        X += a
        y += b
    w = _logistica(X, y)
    v = {t: w[i] / w[0] for i, t in enumerate(TIPOS)}
    yp = list(y)
    random.Random(5).shuffle(yp)
    wp = _logistica(X, yp)
    vp = {t: wp[i] / wp[0] if wp[0] else 0 for i, t in enumerate(TIPOS)}
    ordem = v[1] < min(v[2], v[3]) and max(v[2], v[3]) < v[4] < v[5]
    faixa = all(abs(v[t] - HUMANO[t]) <= 0.4 * HUMANO[t] for t in TIPOS)
    ver = "V1" if ordem and faixa else "V2" if ordem else "V3"
    if verbose:
        nome = {1: "peao", 2: "cavalo", 3: "bispo", 4: "torre", 5: "dama"}
        print("VALOR extraido de %d partidas decididas (lance 30):" % len(y))
        for t in TIPOS:
            print("  %-7s %5.2f   (humano %d | placebo %+.2f)"
                  % (nome[t], v[t], HUMANO[t], vp[t]))
        print("  VEREDITO: %s" % ver)
    return v, ver


_VAL: Dict[int, float] = {}


def _aval(b, cor) -> float:
    return sum(_VAL[t] * (len(b.pieces(t, cor)) - len(b.pieces(t, not cor)))
               for t in TIPOS)


def _quieto(b, alfa, beta, prof):
    """So capturas ate aquietar: o 'e depois' da imaginacao."""
    parado = _aval(b, b.turn)
    if prof == 0 or parado >= beta:
        return parado
    alfa = max(alfa, parado)
    for m in b.generate_legal_captures():
        b.push(m)
        sc = -_quieto(b, -beta, -alfa, prof - 1)
        b.pop()
        if sc >= beta:
            return sc
        alfa = max(alfa, sc)
    return alfa


def valor_imaginado(b, m) -> float:
    """Meu lance, TODA resposta do adversario, capturas ate aquietar; pior caso."""
    b.push(m)
    try:
        resp = list(b.legal_moves)
        if not resp:
            return 1000.0 if b.is_checkmate() else 0.0
        pior = float("inf")
        for r in resp:
            b.push(r)
            if b.is_checkmate():
                sc = -1000.0
            else:
                sc = _quieto(b, -1e9, 1e9, 4)       # b.turn == eu
            b.pop()
            pior = min(pior, sc)
        return pior
    finally:
        b.pop()


def _imagina(args):
    import chess
    k, ms, semente, val, val1 = args
    rnd = random.Random(semente)
    b = chess.Board()
    plies = list(range(8, min(len(ms), 70)))
    alvo = set(rnd.sample(plies, min(2, len(plies))))
    out = []
    for i, u in enumerate(ms):
        if i in alvo:
            legais = list(b.legal_moves)
            if len(legais) >= 2:
                antes = relacoes(b)
                sigs = [assinatura(b, m, antes) for m in legais]
                _VAL.clear()
                _VAL.update(val)
                v = [valor_imaginado(b, m) for m in legais]
                if val1 is None:
                    v1 = v
                else:
                    _VAL.clear()
                    _VAL.update(val1)
                    v1 = [valor_imaginado(b, m) for m in legais]
                out.append((k, sigs, v, v1))
        b.push(chess.Move.from_uci(u))
    return out


def imaginado(n_jogos=1500, n_prob=40000, trab=6):
    from multiprocessing import Pool
    t0 = time.time()
    val, ver_v = valores()
    val1 = {t: 1.0 for t in TIPOS}
    J = jogos(TREINO, n_jogos)
    P = problemas(n_prob)
    with Pool(trab) as pool:
        res = pool.map(_imagina, [(k, ms, 3 * 10 ** 6 + k, val, val1)
                                  for k, ms in enumerate(J)], 5)
        rp = pool.map(_problema, P, 200)
    pos = [x for r in res for x in r]
    print("imaginado em %d posicoes | %.0f s" % (len(pos), time.time() - t0), flush=True)

    def v_sim(qual, embaralha=None):
        soma, n, jogos_c = Counter(), Counter(), defaultdict(set)
        for k, sigs, v, v1 in pos:
            vv = list(v if qual == 0 else v1)
            vv = [max(-20.0, min(20.0, x)) for x in vv]   # mate vira teto, nao domina
            if embaralha is not None:
                embaralha.shuffle(vv)
            mu = sum(vv) / len(vv)
            for sg, x in zip(sigs, vv):
                for c in pedacos(sg):
                    soma[c] += x - mu
                    n[c] += 1
                    jogos_c[c].add(k)
        return {c: soma[c] / n[c] for c in n}, jogos_c

    V, jc = v_sim(0)
    V1, _ = v_sim(1)
    Vp, _ = v_sim(0, random.Random(11))
    pool_c = [c for c in V if len(jc[c]) >= 30]
    n_ok = sum(sg is not None for sg, _t in rp)
    base = {t: sum(t in tt for sg, tt in rp if sg is not None) / n_ok for t in TEMAS}
    no_pool = set(pool_c)
    dispara, por_tema = Counter(), defaultdict(Counter)
    for sg, tt in rp:
        if sg is None:
            continue
        for c in pedacos(sg):
            if c in no_pool:
                dispara[c] += 1
                for t in TEMAS:
                    if t in tt:
                        por_tema[c][t] += 1
    cs = [c for c in pool_c if dispara.get(c, 0) >= 30]
    print("=" * 78)
    print("XADREZ — relevancia pela consequencia IMAGINADA | reservatorio %d, "
          "%d com >= 30 disparos | %.0f s" % (len(pool_c), len(cs), time.time() - t0))
    print("=" * 78)
    n1 = 0
    for nome, VV in (("valor EXTRAIDO", V), ("todas valem 1 (ablacao)", V1),
                     ("PLACEBO embaralhado", Vp)):
        rhos = {t: _rho([VV[c] for c in cs],
                        [por_tema[c][t] / dispara[c] / base[t] for c in cs]) for t in TEMAS}
        k_ = sum(r >= 0.25 for r in rhos.values())
        if nome.startswith("valor"):
            n1 = k_
        print("  %-24s %s | >= +0,25: %d" % (
            nome, " ".join("%s %+.2f" % (t[:6], r) for t, r in rhos.items()), k_))
    top = sorted(cs, key=lambda c: -V[c])[:15]
    print("  padroes de maior consequencia imaginada:")
    for c in top:
        print("    %+6.2f  %s" % (V[c], " + ".join(c)))
    ver = "S1" if n1 >= 3 else "S2" if n1 >= 1 else "S3"
    print("  VEREDITO: valor %s | imaginacao %s" % (ver_v, ver))


def causa(n_jogos=1500, n_prob=40000, trab=6):
    """
    4a pergunta (pre-registrada no docstring ANTES de rodar): conceito = a CAUSA
    minima de um ganho ADIADO. So lances SEM captura; ganho imaginado G; os
    padroes que explicam os lances de G alto, contra os lances sem captura em
    geral. Posicoes de FEVEREIRO — a imaginacao da 3a pergunta nunca as tocou.
    """
    import pickle
    from multiprocessing import Pool
    t0 = time.time()
    val = {1: 1.0, 2: 2.05, 3: 2.51, 4: 3.13, 5: 6.29}     # os EXTRAIDOS (V1)
    F = jogos(FORA, n_jogos)
    P = problemas(n_prob)
    with Pool(trab) as pool:
        res = pool.map(_imagina, [(k, ms, 4 * 10 ** 6 + k, val, None)
                                  for k, ms in enumerate(F)], 5)
        rp = pool.map(_problema, P, 200)
    pos = [(k, sigs, v) for r in res for k, sigs, v, _v1 in r]
    os.makedirs("resultados", exist_ok=True)
    pickle.dump(pos, open("resultados/xadrez_causa_pos.pkl", "wb"))
    print("imaginado em %d posicoes de fevereiro | %.0f s" % (len(pos), time.time() - t0),
          flush=True)

    def sem_captura(sg):
        return not any(it.startswith("Mx") for it in sg)

    def enriquecimento(embaralha=None):
        alto, todos, jogos_c = Counter(), Counter(), defaultdict(set)
        n_alto = n_todos = 0
        for k, sigs, v in pos:
            vv = [max(-20.0, min(20.0, x)) for x in v]
            if embaralha is not None:
                embaralha.shuffle(vv)
            mu = sum(vv) / len(vv)
            for sg, x in zip(sigs, vv):
                if not sem_captura(sg):
                    continue
                cs = pedacos(sg)
                n_todos += 1
                todos.update(cs)
                for c in cs:
                    jogos_c[c].add(k)
                if x - mu >= 1.5:
                    n_alto += 1
                    alto.update(cs)
        E = {c: (alto[c] / max(n_alto, 1)) / (todos[c] / n_todos) for c in todos}
        return E, jogos_c, n_alto, n_todos

    E, jc, n_alto, n_todos = enriquecimento()
    Ep, _, _, _ = enriquecimento(random.Random(13))
    pool_c = [c for c in E if len(jc[c]) >= 30]
    n_ok = sum(sg is not None for sg, _t in rp)
    temas5 = ("fork", "pin", "skewer", "discoveredAttack", "doubleCheck")
    base = {t: sum(t in tt for sg, tt in rp if sg is not None) / n_ok for t in temas5}
    no_pool = set(pool_c)
    dispara, por_tema = Counter(), defaultdict(Counter)
    for sg, tt in rp:
        if sg is None:
            continue
        for c in pedacos(sg):
            if c in no_pool:
                dispara[c] += 1
                for t in temas5:
                    if t in tt:
                        por_tema[c][t] += 1
    cs = [c for c in pool_c if dispara.get(c, 0) >= 30]
    print("=" * 78)
    print("XADREZ — conceito como CAUSA de ganho adiado | %d lances sem captura, "
          "%d de ganho alto | %d padroes | %.0f s" % (n_todos, n_alto, len(cs),
                                                     time.time() - t0))
    print("=" * 78)
    n1 = 0
    for nome, EE in (("CAUSA (enriquecimento)", E), ("PLACEBO embaralhado", Ep)):
        rhos = {t: _rho([math.log(max(EE[c], 1e-3)) for c in cs],
                        [por_tema[c][t] / dispara[c] / base[t] for c in cs]) for t in temas5}
        k_ = sum(r >= 0.25 for r in rhos.values())
        if nome.startswith("CAUSA"):
            n1 = k_
        print("  %-24s %s | >= +0,25: %d" % (
            nome, " ".join("%s %+.2f" % (t[:6], r) for t, r in rhos.items()), k_))
    print("  as causas mais fortes de ganho adiado (sem captura):")
    for c in sorted(cs, key=lambda c: -E[c])[:15]:
        melhor = max(temas5, key=lambda t: por_tema[c][t] / dispara[c] / base[t])
        print("    E=%5.2f  %-34s tema mais alinhado: %s (lift %.1f)" % (
            E[c], " + ".join(c), melhor, por_tema[c][melhor] / dispara[c] / base[melhor]))
    ver = "D1" if n1 >= 3 else "D2" if n1 >= 1 else "D3"
    print("  VEREDITO: %s" % ver)


def _superficie_problema(args):
    """Linha de base: so O QUE andou e O QUE foi capturado nos lances do resolvedor."""
    import chess
    fen, ms, temas = args
    try:
        b = chess.Board(fen)
        sig = set()
        for i, u in enumerate(ms):
            mv = chess.Move.from_uci(u)
            if i % 2 == 1:
                sig.add("anda:" + b.piece_at(mv.from_square).symbol().lower())
                if b.is_capture(mv) and not b.is_en_passant(mv):
                    sig.add("toma:" + b.piece_at(mv.to_square).symbol().lower())
            b.push(mv)
        return frozenset(sig), temas
    except Exception:
        return None, temas


def _conceito(exemplos, freq, n_ref):
    """
    Compara os exemplos e fica com o que se CONSERVA (em >= 4 de 5); entre o que
    se conserva, o mais INFORMATIVO (o mais raro em geral) — Occam por bits.
    """
    conta = Counter()
    for sg in exemplos:
        conta.update(pedacos(sg))
    conserva = [c for c, n in conta.items() if n >= len(exemplos) - 1]
    if not conserva:
        return None
    return min(conserva, key=lambda c: (freq.get(c, 0) / n_ref, -len(c), str(c)))


def nomear(n_prob=40000, k=5, sorteios=20, trab=6):
    from multiprocessing import Pool
    t0 = time.time()
    P = problemas(n_prob)
    with Pool(trab) as pool:
        rel = pool.map(_problema, P, 200)
        sup = pool.map(_superficie_problema, P, 200)
    idx = [i for i in range(len(P)) if rel[i][0] is not None and sup[i][0] is not None]
    rnd = random.Random(20260924)
    rnd.shuffle(idx)
    ref, teste = idx[:10000], idx[10000:]           # ref: frequencia geral e exemplos
    print("%d problemas: %d de referencia, %d de teste | %.0f s"
          % (len(idx), len(ref), len(teste), time.time() - t0), flush=True)

    def avalia(sigs, conceito, t):
        disparam = [i for i in teste if set(conceito) <= sigs[i][0]]
        base = sum(t in sigs[i][1] for i in teste) / len(teste)
        if not disparam:
            return 1.0, 0.0
        p = sum(t in sigs[i][1] for i in disparam) / len(disparam)
        rec = sum(t in sigs[i][1] for i in disparam) / max(1, sum(t in sigs[i][1] for i in teste))
        return p / base, rec

    freq_rel, freq_sup = Counter(), Counter()
    for i in ref:
        freq_rel.update(pedacos(rel[i][0]))
        freq_sup.update(pedacos(sup[i][0]))
    print("=" * 78)
    print("XADREZ — NOMEAR com %d exemplos (como quem aprende de um professor) | "
          "media de %d sorteios" % (k, sorteios))
    print("=" * 78)
    passa = 0
    for t in TEMAS:
        com = [i for i in ref if t in rel[i][1]]
        L = {"rel": [], "sup": [], "plc": []}
        exemplo_conceito = Counter()
        for _ in range(sorteios):
            ex = rnd.sample(com, k)
            c = _conceito([rel[i][0] for i in ex], freq_rel, len(ref))
            L["rel"].append(avalia(rel, c, t) if c else (1.0, 0.0))
            if c:
                exemplo_conceito[c] += 1
            cs = _conceito([sup[i][0] for i in ex], freq_sup, len(ref))
            L["sup"].append(avalia(sup, cs, t) if cs else (1.0, 0.0))
            exp = rnd.sample(ref, k)                   # PLACEBO: 5 quaisquer, com o nome
            cp = _conceito([rel[i][0] for i in exp], freq_rel, len(ref))
            L["plc"].append(avalia(rel, cp, t) if cp else (1.0, 0.0))
        m = {q: sum(x[0] for x in v) / len(v) for q, v in L.items()}
        r = {q: sum(x[1] for x in v) / len(v) for q, v in L.items()}
        plc95 = sorted(x[0] for x in L["plc"])[int(0.95 * sorteios) - 1]
        ok = m["rel"] >= 2 and m["rel"] > m["sup"] and m["rel"] > plc95
        passa += ok
        top = exemplo_conceito.most_common(1)[0][0] if exemplo_conceito else None
        print("  %-17s lift rel %5.2f (revoc %.2f) | superficie %5.2f | placebo p95 %5.2f %s"
              % (t, m["rel"], r["rel"], m["sup"], plc95, "PASSA" if ok else "-----"))
        if top:
            print("      conceito mais escolhido: %s" % " + ".join(top))
    ver = "N1" if passa >= 4 else "N2" if passa >= 2 else "N3"
    print("  VEREDITO: %s (%d de 6)" % (ver, passa))


def _abre(it):
    """'M~Ikn:d*2' -> (quem, rel, lado, alvo, frente, marca, dobro)."""
    dobro = it.endswith("*2")
    base = it[:-2] if dobro else it
    marca = None
    if ":" in base:
        base, marca = base.split(":")
    quem, rel = base[0], base[1]
    if rel == "x":
        return quem, rel, "I", base[2], "", marca, dobro
    return quem, rel, base[2], base[3], base[4:], marca, dobro


def _fecha(q, r, l, a, f, m):
    s = q + r + (a if r == "x" else l + a + f)
    return s + (":" + m if m is not None else "")


def generaliza(sig: frozenset) -> frozenset:
    """
    Cada atributo EXATO ou QUALQUER ('?') — a operacao do carimbo do ARC
    ([[construtor]]) levada a outro mundo. Depois de generalizar, CONTA: duas
    relacoes distintas que viram a mesma forma geral marcam '*2' ("ataca DUAS
    pecas inimigas quaisquer" e o garfo; ninguem escreveu isso).
    """
    conta = Counter()
    for it in sig:
        if it.endswith("*2"):
            continue
        q, r, l, a, f, m, _d = _abre(it)
        for q2 in {q, "?"} if q in "Mm" else {q}:
            for a2 in (a, "?"):
                for f2 in ((f, "?") if f else ("",)):
                    for m2 in ((m, "?") if m is not None else (None,)):
                        conta[_fecha(q2, r, l, a2, f2, m2)] += 1
    out = set(conta)
    out |= {g + "*2" for g, n in conta.items() if n >= 2}
    return frozenset(out)


def _conceito2(exemplos, freq_item, ref_sigs, cache):
    """Conservado em >= 4 de 5: itens gerais e pares deles; o mais raro em geral."""
    k = len(exemplos)
    conta = Counter()
    for sg in exemplos:
        conta.update(sg)
    singles = [x for x, n in conta.items() if n >= k - 1]
    cands = [(x,) for x in singles]
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            par = (singles[i], singles[j])
            if sum(set(par) <= sg for sg in exemplos) >= k - 1:
                cands.append(par)
    if not cands:
        return None

    def freq(c):
        if len(c) == 1:
            return freq_item.get(c[0], 0)
        if c not in cache:
            cache[c] = sum(set(c) <= sg for sg in ref_sigs)
        return cache[c]
    return min(cands, key=lambda c: (freq(c), -len(c), str(c)))


def nomear2(inicio=40000, n_prob=40000, k=5, sorteios=20, trab=6):
    from multiprocessing import Pool
    t0 = time.time()
    P = problemas(inicio + n_prob)[inicio:]          # BLOCO NOVO, nunca avaliado
    with Pool(trab) as pool:
        rel = pool.map(_problema, P, 200)
        sup = pool.map(_superficie_problema, P, 200)
    idx = [i for i in range(len(P)) if rel[i][0] is not None and sup[i][0] is not None]
    rnd = random.Random(20260925)
    rnd.shuffle(idx)
    ref, teste = idx[:10000], idx[10000:]
    gen = {i: generaliza(rel[i][0]) for i in idx}
    print("%d problemas novos (linhas %d-%d) | %.0f s"
          % (len(idx), inicio, inicio + n_prob, time.time() - t0), flush=True)
    freq_g, freq_s = Counter(), Counter()
    for i in ref:
        freq_g.update(gen[i])
        freq_s.update(sup[i][0])
    ref_g = [gen[i] for i in ref]
    ref_s = [sup[i][0] for i in ref]
    cache_g, cache_s = {}, {}

    def avalia(sigs_de, temas_de, conceito, t):
        base = sum(t in temas_de(i) for i in teste) / len(teste)
        disp = [i for i in teste if set(conceito) <= sigs_de(i)]
        if not disp:
            return 1.0, 0.0
        acerto = sum(t in temas_de(i) for i in disp)
        return (acerto / len(disp)) / base, acerto / max(1, base * len(teste))

    temas_de = lambda i: rel[i][1]
    print("=" * 78)
    print("XADREZ — NOMEAR com %d exemplos, atributo EXATO ou QUALQUER | %d sorteios"
          % (k, sorteios))
    print("=" * 78)
    passa = 0
    for t in TEMAS:
        com = [i for i in ref if t in rel[i][1]]
        L = {"rel": [], "sup": [], "plc": []}
        esc = Counter()
        for _ in range(sorteios):
            ex = rnd.sample(com, k)
            c = _conceito2([gen[i] for i in ex], freq_g, ref_g, cache_g)
            L["rel"].append(avalia(lambda i: gen[i], temas_de, c, t) if c else (1.0, 0.0))
            if c:
                esc[c] += 1
            cs = _conceito2([sup[i][0] for i in ex], freq_s, ref_s, cache_s)
            L["sup"].append(avalia(lambda i: sup[i][0], temas_de, cs, t) if cs else (1.0, 0.0))
            exp = rnd.sample(ref, k)
            cp = _conceito2([gen[i] for i in exp], freq_g, ref_g, cache_g)
            L["plc"].append(avalia(lambda i: gen[i], temas_de, cp, t) if cp else (1.0, 0.0))
        m = {q: sum(x[0] for x in v) / len(v) for q, v in L.items()}
        r = {q: sum(x[1] for x in v) / len(v) for q, v in L.items()}
        plc95 = sorted(x[0] for x in L["plc"])[int(0.95 * sorteios) - 1]
        ok = m["rel"] >= 2 and m["rel"] > m["sup"] and m["rel"] > plc95
        passa += ok
        print("  %-17s lift %5.2f (revoc %.2f) | superficie %5.2f | placebo p95 %5.2f %s"
              % (t, m["rel"], r["rel"], m["sup"], plc95, "PASSA" if ok else "-----"))
        for c, n in esc.most_common(2):
            print("      %2dx  %s" % (n, " + ".join(c)))
    ver = "N1" if passa >= 4 else "N2" if passa >= 2 else "N3"
    print("  VEREDITO: %s (%d de 6)" % (ver, passa))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nomear2":
        nomear2()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "nomear":
        nomear()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "causa":
        causa()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "valores":
        valores()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "imaginado":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
        imaginado(n)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "consequencia":
        consequencia()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "fumaca":
        main(60, 30, 2000, 4)
    else:
        main()
