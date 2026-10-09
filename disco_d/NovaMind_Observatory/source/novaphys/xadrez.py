"""
novaphys.xadrez — que ORDEM LOGICA a definicao de vitoria exige?
==================================================================

POR QUE XADREZ, E NAO PARA JOGAR
----------------------------------
Forca de jogo aqui e irrelevante — esta resolvida ha decadas e nao diz nada
sobre AGI. O que xadrez tem, e nenhum mundo desta sessao teve, e o seguinte:

    "vitoria" e uma frase QUANTIFICADA.

Xeque-mate nao e "o rei esta atacado". E "o rei esta atacado E PARA TODO
movimento dele, continua atacado". Todos os objetivos que este projeto testou
ate hoje eram PROPOSICIONAIS — `bolso >= 3` em `objetivo.py`, predicados sobre
features estaticas do estado. Nunca medimos o degrau para primeira ordem.

Esta e a pergunta: para a maquina descobrir sozinha o que e ganhar, a
linguagem de objetivos precisa de quantificador, ou features estaticas em
quantidade suficiente dao conta?

O TESTE, E ELE E CRUEL DE PROPOSITO
-------------------------------------
Aprender o que e vitoria no final REI+TORRE contra REI. Testar a regra
aprendida em REI+DAMA contra REI — um jogo que ela nunca viu, com a mesma
nocao de mate e geometria de ataque diferente.

    L1  PROPOSICIONAL   13 features estaticas: atacado, borda, canto,
                        oposicao, distancias, alinhamentos, proteccao.
                        NENHUMA quantifica sobre movimentos.
    L2  QUANTIFICADA    as mesmas 13 mais UMA: `sem_saida` = para todo
                        movimento do rei preto, e ilegal.

Hipotese = conjuncao de ate 4 literais (feature ou negacao). Mesma linguagem,
mesmo comprimento, mesmo enumerador nos dois bracos. A UNICA diferenca e se o
quantificador esta disponivel.

Em L2 a resposta existe e tem 2 literais: `atacado E sem_saida`. Em L1 ela
NAO EXISTE — a pergunta e se L1 consegue FALSIFICAR bem o bastante em um jogo,
e se essa falsificacao sobrevive ao outro jogo.

E essa a distincao que interessa e que nunca medimos:

    ajuste EXTENSIONAL   acerta os casos que viu
    correcao INTENSIONAL acerta porque capturou a regra

Um sistema que so faz o primeiro parece que entendeu ate mudar o jogo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  X0  PRE-CONDICAO. Os dois conjuntos (KRK e KQK) tem mate e nao-mate em
      quantidade utilizavel, e a resposta de L2 (`atacado E sem_saida`) tem
      que dar 100% nos dois. Se nao der, minhas features estao erradas e
      nada mais no arquivo vale.

  X1  L1 chega a 100% em KRK  e  DESABA em KQK
      -> quantificador e necessario. Features estaticas fingem entender.
         O ajuste extensional nao transfere, e essa e a medida do degrau.

  X2  L1 chega a 100% em KRK  e  SEGURA em KQK (>= 95%)
      -> features estaticas bastam; a nocao de vitoria e geometrica e o
         quantificador e conveniencia, nao necessidade.

  X3  L1 nao chega a 100% nem em KRK
      -> proposicional nao expressa mate nem no jogo em que treinou. Degrau
         ainda maior do que eu esperava.

MINHA PREVISAO, escrita antes de rodar: X1. Espero L1 acima de 99% em KRK
(mate em KRK e geometricamente muito restrito: rei na borda, torre dando
xeque na linha da borda, reis em oposicao) e espero que caia para algo entre
70% e 95% em KQK, porque a dama da mate a distancia e com o rei protegendo de
perto, geometria que a regra de torre nao cobre. Aposto tambem que NENHUMA
conjuncao L1 de 4 literais atinge 100% em KRK — as posicoes de mate exigem
"torre nao capturavel", que e ela mesma uma frase quantificada disfarcada.
Se eu estiver certo nas duas, o resultado e X3 e nao X1, e ai o degrau e
maior do que o titulo deste arquivo sugere.

O QUE ISSO NAO E
------------------
Nao e "a AGI aprendeu xadrez". A familia de features continua sendo escolha
minha, como em todo arquivo desta sessao. O que se mede aqui e uma propriedade
da LINGUAGEM, nao da minha esperteza ao escolher features: se um quantificador
e necessario, nenhuma escolha de features estaticas resolve, e o resultado e
robusto a eu ter sido burro ou esperto na lista.
"""

from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import chess


# ══════════════════════════════════════════════ geracao das posicoes
def categoria(t: chess.Board) -> str:
    """Verdade de campo, do python-chess. Nao e a minha logica que rotula."""
    if t.is_checkmate():
        return "mate"
    if t.is_stalemate():
        return "afogamento"
    n = sum(1 for _ in t.legal_moves)
    if t.is_check():
        return "xeque_1" if n == 1 else "xeque_fuga"
    return "livre"


def posicoes(peca: int, limite: int, semente: int, dificil: bool
             ) -> List[Tuple[chess.Board, bool]]:
    """
    Posicoes legais de REI+<peca> contra REI, pretas para jogar.

    ERRO DE AMOSTRAGEM na primeira versao, e ele quase virou conclusao: eu
    sorteava metade mate e metade NAO-MATE ALEATORIO. Negativo aleatorio esta
    geometricamente longe de mate, entao a linguagem proposicional acertava
    99,61% e transferia — parecia que features estaticas bastavam.

    Nao bastam; a amostra e que era facil. `dificil=True` monta o conjunto com
    QUASE-ERROS, as posicoes que so a regra certa separa:

        mate         rei atacado e sem saida
        afogamento   sem saida e NAO atacado   <- so `atacado` separa de mate
        xeque_1      atacado com UMA fuga so   <- so o quantificador separa
        xeque_fuga   atacado com varias fugas

    E a mesma licao de `prior.py`: hipotese plausivel e quase-erro da verdade,
    e conjunto de teste sem quase-erro nao mede nada.
    """
    rng = random.Random(semente)
    todas = [(a, b, c) for a in range(64) for b in range(64) for c in range(64)
             if len({a, b, c}) == 3 and chess.square_distance(a, c) > 1]
    rng.shuffle(todas)
    alvo = ["mate", "afogamento", "xeque_1", "xeque_fuga"] if dificil         else ["mate", "livre"]
    cota = limite // len(alvo)
    conta: Dict[str, int] = {k: 0 for k in alvo}
    fora: List[Tuple[chess.Board, bool]] = []
    for rb, pb, rp in todas:
        t = chess.Board(None)
        t.set_piece_at(rb, chess.Piece(chess.KING, chess.WHITE))
        t.set_piece_at(pb, chess.Piece(peca, chess.WHITE))
        t.set_piece_at(rp, chess.Piece(chess.KING, chess.BLACK))
        t.turn = chess.BLACK
        if not t.is_valid():
            continue
        c = categoria(t)
        if c not in conta or conta[c] >= cota:
            continue
        conta[c] += 1
        fora.append((t, c == "mate"))
        if all(v >= cota for v in conta.values()):
            break
    print(f"     composicao: {conta}")
    return fora


# ══════════════════════════════════════════════ a linguagem de objetivos
def _casas(t: chess.Board) -> Tuple[int, int, int]:
    rb = t.king(chess.WHITE)
    rp = t.king(chess.BLACK)
    pb = next(s for s in chess.SQUARES
              if (p := t.piece_at(s)) and p.color == chess.WHITE
              and p.piece_type != chess.KING)
    return rb, pb, rp


def features_estaticas() -> List[Tuple[str, Callable]]:
    """
    Treze features geometricas. Nenhuma delas quantifica sobre movimentos:
    todas se leem olhando a posicao parada.
    """
    def f(nome):
        def dec(g):
            return (nome, g)
        return dec

    fs: List[Tuple[str, Callable]] = []

    def add(nome, g):
        fs.append((nome, g))

    add("atacado", lambda t: t.is_check())
    add("borda", lambda t: (lambda rp: chess.square_file(rp) in (0, 7)
                            or chess.square_rank(rp) in (0, 7))(
        t.king(chess.BLACK)))
    add("canto", lambda t: (lambda rp: chess.square_file(rp) in (0, 7)
                            and chess.square_rank(rp) in (0, 7))(
        t.king(chess.BLACK)))
    add("reis_dist1", lambda t: chess.square_distance(
        t.king(chess.WHITE), t.king(chess.BLACK)) == 1)
    add("reis_dist2", lambda t: chess.square_distance(
        t.king(chess.WHITE), t.king(chess.BLACK)) == 2)
    add("reis_longe", lambda t: chess.square_distance(
        t.king(chess.WHITE), t.king(chess.BLACK)) >= 3)
    add("oposicao", lambda t: (lambda rb, rp: chess.square_distance(rb, rp) == 2
                               and (chess.square_file(rb) ==
                                    chess.square_file(rp)
                                    or chess.square_rank(rb) ==
                                    chess.square_rank(rp)))(
        t.king(chess.WHITE), t.king(chess.BLACK)))
    add("peca_adj_rei", lambda t: chess.square_distance(
        _casas(t)[1], t.king(chess.BLACK)) == 1)
    add("peca_protegida", lambda t: chess.square_distance(
        _casas(t)[0], _casas(t)[1]) == 1)
    add("mesma_linha", lambda t: chess.square_rank(_casas(t)[1]) ==
        chess.square_rank(t.king(chess.BLACK)))
    add("mesma_coluna", lambda t: chess.square_file(_casas(t)[1]) ==
        chess.square_file(t.king(chess.BLACK)))
    add("mesma_diagonal", lambda t: abs(
        chess.square_file(_casas(t)[1]) - chess.square_file(
            t.king(chess.BLACK))) == abs(
        chess.square_rank(_casas(t)[1]) - chess.square_rank(
            t.king(chess.BLACK))))
    add("peca_na_borda", lambda t: (lambda pb: chess.square_file(pb) in (0, 7)
                                    or chess.square_rank(pb) in (0, 7))(
        _casas(t)[1]))
    return fs


def feature_quantificada() -> Tuple[str, Callable]:
    """A UNICA coisa que L2 tem a mais: o quantificador universal."""
    return ("sem_saida", lambda t: not any(t.legal_moves))


# ══════════════════════════════════════════════ inducao da regra
def tabela(dados, fs) -> List[Tuple[Tuple[bool, ...], bool]]:
    return [(tuple(bool(g(t)) for _n, g in fs), y) for t, y in dados]


def melhor_conjuncao(tab, n_f: int, max_lit: int
                     ) -> Tuple[float, Optional[Tuple], int]:
    """
    Enumera conjuncoes de ate `max_lit` literais (feature ou negacao) e
    devolve a melhor acuracia, uma regra que a atinge, e quantas regras
    empatam no topo. Enumeracao exaustiva: nada de heuristica escondida.
    """
    total = len(tab)
    melhor, regra, quantas = -1.0, None, 0
    lits = [(i, s) for i in range(n_f) for s in (True, False)]
    for k in range(1, max_lit + 1):
        for comb in itertools.combinations(lits, k):
            if len({i for i, _s in comb}) != k:
                continue        # nao usar a mesma feature duas vezes
            acertos = 0
            for v, y in tab:
                p = all(v[i] == s for i, s in comb)
                if p == y:
                    acertos += 1
            a = acertos / total
            if a > melhor + 1e-12:
                melhor, regra, quantas = a, comb, 1
            elif abs(a - melhor) <= 1e-12:
                quantas += 1
        if melhor >= 1.0:
            break
    return melhor, regra, quantas


def teto(tab) -> Tuple[float, int, int]:
    """
    Teto de QUALQUER funcao booleana destas features, nao so de conjuncoes.

    Agrupa as posicoes por vetor de features. Se um grupo tem mate e nao-mate
    ao mesmo tempo, nenhuma regra escrita nessa linguagem separa os dois — sao
    posicoes INDISTINGUIVEIS para ela. O melhor possivel e chutar a maioria de
    cada grupo. Isso e exato e nao depende de eu ter escolhido bem a familia
    de regras: e um limite da LINGUAGEM.
    """
    grupos: Dict[Tuple[bool, ...], List[int]] = {}
    for v, y in tab:
        g = grupos.setdefault(v, [0, 0])
        g[1 if y else 0] += 1
    bons = sum(max(g) for g in grupos.values())
    colididos = sum(min(g) * 2 for g in grupos.values()
                    if g[0] and g[1])
    return bons / len(tab), len(grupos), colididos


def aplica(regra, v) -> bool:
    return all(v[i] == s for i, s in regra)


def nome_regra(regra, fs) -> str:
    return " E ".join(("" if s else "NAO ") + fs[i][0] for i, s in regra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--literais", type=int, default=4)
    ap.add_argument("--facil", action="store_true")
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 74)
    print("XADREZ — que ordem logica a definicao de VITORIA exige?")
    print("=" * 74)

    modo = "FACIL (negativo aleatorio)" if a.facil else "DIFICIL (quase-erros)"
    print(f"\namostra {modo}")
    krk = posicoes(chess.ROOK, a.n, a.semente, not a.facil)
    kqk = posicoes(chess.QUEEN, a.n, a.semente + 1, not a.facil)
    est = features_estaticas()
    quant = feature_quantificada()
    l1, l2 = est, est + [quant]

    n_m = sum(1 for _t, y in krk if y)
    n_m2 = sum(1 for _t, y in kqk if y)
    print(f"\nX0 · KRK {len(krk)} posicoes, {n_m} mates")
    print(f"     KQK {len(kqk)} posicoes, {n_m2} mates")
    print(f"     L1 = {len(l1)} features estaticas")
    print(f"     L2 = L1 + `sem_saida` (o quantificador)")

    # a resposta em L2 tem que dar 100% nos dois, senao as features estao erradas
    for rot, dados in (("KRK", krk), ("KQK", kqk)):
        ok = sum(1 for t, y in dados
                 if (t.is_check() and not any(t.legal_moves)) == y)
        print(f"     `atacado E sem_saida` em {rot}: {ok/len(dados):.1%}")
        if ok != len(dados):
            print("\n     X0 FALHOU. A resposta conhecida nao classifica.")
            return

    t1, t2 = tabela(krk, l1), tabela(krk, l2)
    print(f"\ninduzindo em KRK, conjuncoes de ate {a.literais} literais, "
          f"enumeracao exaustiva\n")

    c1, g1, x1c = teto(t1)
    c2, g2, x2c = teto(t2)
    print(f"  TETO de qualquer funcao das features (nao so conjuncoes):")
    print(f"    L1  {c1:7.2%}   {g1} vetores distintos, "
          f"{x1c} posicoes indistinguiveis")
    print(f"    L2  {c2:7.2%}   {g2} vetores distintos, "
          f"{x2c} posicoes indistinguiveis\n")

    a1, r1, q1 = melhor_conjuncao(t1, len(l1), a.literais)
    a2, r2, q2 = melhor_conjuncao(t2, len(l2), a.literais)
    print(f"  L1 proposicional  {a1:7.2%} em KRK   "
          f"({q1} regras empatam)   {nome_regra(r1, l1)}")
    print(f"  L2 quantificada   {a2:7.2%} em KRK   "
          f"({q2} regras empatam)   {nome_regra(r2, l2)}")

    # transferencia: a MESMA regra, no jogo que ela nunca viu
    v1, v2 = tabela(kqk, l1), tabela(kqk, l2)
    x1 = sum(1 for v, y in v1 if aplica(r1, v) == y) / len(v1)
    x2 = sum(1 for v, y in v2 if aplica(r2, v) == y) / len(v2)
    print(f"\ntransferindo a MESMA regra para KQK (jogo nunca visto)\n")
    print(f"  L1 proposicional  {x1:7.2%} em KQK   "
          f"queda de {(a1 - x1) * 100:5.1f} pontos")
    print(f"  L2 quantificada   {x2:7.2%} em KQK   "
          f"queda de {(a2 - x2) * 100:5.1f} pontos")

    print("\n" + "=" * 74)
    if c1 < 0.999:
        print(f"X3 · proposicional nao expressa mate NEM no jogo em que")
        print(f"     treinou, e o limite NAO e a familia de regras: NENHUMA")
        print(f"     funcao das 13 features passa de {c1:.2%}, porque {x1c}")
        print(f"     posicoes tem vetor de features IDENTICO e rotulo")
        print(f"     diferente. Com o quantificador o teto vai a {c2:.2%}.")
        print(f"     O degrau para primeira ordem nao e conveniencia; e")
        print(f"     expressividade, e custa UM literal.")
    elif x1 >= 0.95:
        print("X2 · features estaticas bastam. Vitoria e geometrica e o")
        print("     quantificador e conveniencia, nao necessidade.")
    else:
        print("X1 · o quantificador e NECESSARIO. Features estaticas fingem")
        print(f"     entender ({a1:.2%} no jogo de treino) e desabam quando o")
        print(f"     jogo muda ({x1:.2%}). Ajuste extensional nao transfere.")


if __name__ == "__main__":
    main()
