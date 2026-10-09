"""
novaphys.objetivo — aprender COMO o mundo funciona vs descobrir O QUE vencer
=============================================================================

A ASSIMETRIA QUE O TWIN MEDIU, EM MINIATURA
---------------------------------------------
O paper Twin (arXiv 2608.14490) constroi um modelo de mundo executavel em
tempo de teste e chega a 93,3 no ARC-AGI-3, contra 7,8 do mesmo modelo jogando
direto. O achado central deles nao e o placar:

    "Construir um modelo de mundo utilizavel e mais simples do que se
     antecipava; o problema mais dificil e inferir o objetivo certo."

A razao e estrutural: a DINAMICA e verificavel contra cada interacao — toda
transicao observada e uma restricao dura. O OBJETIVO nunca e declarado, entao
so aparece por exploracao.

Todo o nosso trabalho ate agora assumiu objetivo dado: "produza esta grade".
Este arquivo mede a outra metade.

DUAS VERSOES, E A DIFERENCA ENTRE ELAS E O EXPERIMENTO
--------------------------------------------------------
  FECHADA   7 hipoteses de objetivo pre-declaradas, a verdadeira entre elas.
            Isso e MULTIPLA ESCOLHA, nao inferencia. Rodou primeiro e deu
            razao media 0,7x — o objetivo convergindo ANTES da dinamica.
            Quebrado por objetivo, os compostos ("porta com uma chave") deram
            1,4 a 1,8x e os triviais ("pegar uma chave") deram 0,0 a 0,2x. A
            media agregava coisas incomparaveis.

  ABERTA    os predicados sao GERADOS de uma gramatica: atomos sobre posicao,
            linha, coluna, contagem de chaves e regioes, mais todas as
            conjuncoes de dois. Milhares de hipoteses; o objetivo verdadeiro
            e um ponto la dentro, sem marcacao.

A comparacao entre as duas mede quanto do problema era multipla escolha
disfarcada de inferencia.

NA VERSAO ABERTA A CONVERGENCIA MUDA DE DEFINICAO
---------------------------------------------------
Com espaco grande, muitas hipoteses ficam observacionalmente equivalentes
para sempre: "pos=(3,4)" e "pos=(3,4) & linha=3" nunca se separam, porque
nenhum estado alcancavel as distingue. Exigir "sobrar uma" seria exigir o
impossivel, e o experimento nunca convergiria por construcao.

A definicao honesta e "sobrar poucas o bastante para agir": <= `sobrev`
hipoteses compativeis com tudo que foi observado. E parametro declarado, nao
ajuste posterior.

PRE-REGISTRO, ANTES DE RODAR A VERSAO ABERTA
----------------------------------------------
  G0  PRE-CONDICAO. O objetivo verdadeiro tem que ser expressavel na
      gramatica gerada, para cada um dos 7 mundos. Se nao for, o experimento
      mede a minha gramatica e nao a assimetria. Conferido antes de medir.

  G1  o objetivo custa >= 3x mais acoes que a dinamica
      -> a assimetria do Twin reproduz em escala pequena e sem LLM

  G2  custos parecidos (fator < 3)

  G3  o objetivo ainda converge antes da dinamica
      -> nem com espaco aberto; algo no meu mundo nao e representativo

MINHA PREVISAO, escrita antes de rodar a versao aberta: G1. Na fechada deu
0,7x, mas la eliminar 6 alternativas de uma lista minha era barato. Com
milhares de hipoteses cada acao elimina uma fracao pequena, e as conjuncoes
raras sobrevivem por muito tempo.

O que eu nao sei prever e o TAMANHO do fator — e e ele que interessa, porque
mede quanto custa descobrir o que se quer comparado a descobrir como o mundo
funciona.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, List, Optional, Set, Tuple

LADO = 7
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

OBJETIVOS = ["porta", "todas_chaves", "uma_chave", "duas_chaves",
             "porta_com_uma", "porta_com_todas", "canto"]


# ══════════════════════════════════════════════ o mundo
class Mundo:
    """Regras e objetivo escondidos. O agente so ve transicoes."""

    def __init__(self, semente: int, n_chaves: int = 3,
                 objetivo: str = "porta_com_todas"):
        rng = random.Random(semente)
        self.paredes: Set[Tuple[int, int]] = set()
        for _ in range(LADO * 2):
            self.paredes.add((rng.randrange(LADO), rng.randrange(LADO)))
        livres = [(i, j) for i in range(LADO) for j in range(LADO)
                  if (i, j) not in self.paredes]
        rng.shuffle(livres)
        self.inicio = livres[0]
        # CONECTIVIDADE. A auditoria achou um mundo em que a terceira chave
        # estava murada: 0 vitorias, 2.639 hipoteses vivas. Objetivo
        # impossivel nao testa nada — nem exploracao nem inferencia. Porta e
        # chaves so podem ficar na componente alcancavel do inicio.
        alc = self._alcancaveis(self.inicio)
        cand = [c for c in livres[1:] if c in alc]
        if len(cand) < 1 + n_chaves:
            raise ValueError("mundo sem espaco alcancavel suficiente")
        self.porta = cand[0]
        self.chaves0 = frozenset(cand[1:1 + n_chaves])
        self.objetivo = objetivo
        self.n_chaves = n_chaves
        self.reiniciar()

    def _alcancaveis(self, origem) -> Set[Tuple[int, int]]:
        vis = {origem}
        pilha = [origem]
        while pilha:
            y, x = pilha.pop()
            for dy, dx in ACOES:
                n = (y + dy, x + dx)
                if (0 <= n[0] < LADO and 0 <= n[1] < LADO
                        and n not in self.paredes and n not in vis):
                    vis.add(n)
                    pilha.append(n)
        return vis

    def estados_alcancaveis(self) -> List[tuple]:
        """Todo (pos, chaves restantes, bolso) que o agente pode ocupar."""
        alc = self._alcancaveis(self.inicio)
        fora = []
        ch = sorted(self.chaves0)
        for mask in range(1 << len(ch)):
            rest = frozenset(c for i, c in enumerate(ch) if not mask >> i & 1)
            bolso = len(ch) - len(rest)
            for p in alc:
                if p in rest:
                    continue          # estar na chave = ja pegou
                fora.append((p, rest, bolso))
        return fora

    def reiniciar(self) -> None:
        self.pos = self.inicio
        self.chaves = set(self.chaves0)
        self.bolso = 0

    def estado(self) -> tuple:
        return (self.pos, frozenset(self.chaves), self.bolso)

    def passo(self, a: int) -> tuple:
        dy, dx = ACOES[a]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if 0 <= ny < LADO and 0 <= nx < LADO and (ny, nx) not in self.paredes:
            self.pos = (ny, nx)
        if self.pos in self.chaves:
            self.chaves.discard(self.pos)
            self.bolso += 1
        return self.estado()

    def venceu(self, e: Optional[tuple] = None) -> bool:
        pos, _ch, bolso = e if e else self.estado()
        return alvo_verdadeiro(self.objetivo, pos, bolso, self.porta,
                               self.n_chaves)


def alvo_verdadeiro(nome, pos, bolso, porta, n_chaves) -> bool:
    if nome == "porta":
        return pos == porta
    if nome == "todas_chaves":
        return bolso >= n_chaves
    if nome == "uma_chave":
        return bolso >= 1
    if nome == "duas_chaves":
        return bolso >= 2
    if nome == "porta_com_uma":
        return pos == porta and bolso >= 1
    if nome == "porta_com_todas":
        return pos == porta and bolso >= n_chaves
    if nome == "canto":
        return pos in ((0, 0), (0, LADO - 1), (LADO - 1, 0),
                       (LADO - 1, LADO - 1))
    raise ValueError(nome)


# ══════════════════════════════════════════════ modelo de dinamica
class Modelo:
    """
    Hipotese sobre COMO o mundo funciona, aprendida so de transicoes.

    Nao chuta: DEDUZ. Cada transicao diz, para a celula alvo, se havia parede
    ou nao — restricao dura. So e aceito se reproduz TODAS as transicoes ja
    vistas, o mesmo criterio de validacao por replay do Twin.
    """

    def __init__(self) -> None:
        self.parede: Set[Tuple[int, int]] = set()
        self.livre: Set[Tuple[int, int]] = set()

    def aprender(self, e0: tuple, a: int, e1: tuple) -> None:
        p0 = e0[0]
        p1 = e1[0]
        dy, dx = ACOES[a]
        alvo = (p0[0] + dy, p0[1] + dx)
        dentro = 0 <= alvo[0] < LADO and 0 <= alvo[1] < LADO
        if p1 == p0 and dentro:
            self.parede.add(alvo)
        elif p1 == alvo:
            self.livre.add(alvo)

    def prever(self, e0: tuple, a: int) -> Optional[tuple]:
        p0, ch0, b0 = e0
        dy, dx = ACOES[a]
        alvo = (p0[0] + dy, p0[1] + dx)
        if not (0 <= alvo[0] < LADO and 0 <= alvo[1] < LADO):
            return (p0, ch0, b0)
        if alvo in self.parede:
            return (p0, ch0, b0)
        if alvo not in self.livre:
            return None                       # lacuna: nao inventa
        if alvo in ch0:
            return (alvo, ch0 - {alvo}, b0 + 1)
        return (alvo, ch0, b0)

    def cobertura(self) -> int:
        return len(self.parede) + len(self.livre)


# ══════════════════════════════════════════════ objetivos GERADOS
def atomos(n_chaves: int) -> List[Tuple[str, Callable]]:
    fora: List[Tuple[str, Callable]] = []
    for i in range(LADO):
        for j in range(LADO):
            fora.append((f"pos=({i},{j})",
                         lambda e, i=i, j=j: e[0] == (i, j)))
    for i in range(LADO):
        fora.append((f"linha={i}", lambda e, i=i: e[0][0] == i))
        fora.append((f"coluna={i}", lambda e, i=i: e[0][1] == i))
    for k in range(1, n_chaves + 1):
        fora.append((f"bolso>={k}", lambda e, k=k: e[2] >= k))
        fora.append((f"bolso=={k}", lambda e, k=k: e[2] == k))
    fora.append(("bolso==0", lambda e: e[2] == 0))
    fora.append(("semChaves", lambda e: len(e[1]) == 0))
    fora.append(("canto", lambda e: e[0] in ((0, 0), (0, LADO - 1),
                                             (LADO - 1, 0),
                                             (LADO - 1, LADO - 1))))
    fora.append(("borda", lambda e: e[0][0] in (0, LADO - 1)
                 or e[0][1] in (0, LADO - 1)))
    return fora


def atomos_quantificados() -> List[Tuple[str, Callable]]:
    """
    O DEGRAU DE ORDEM LOGICA. Atomos que quantificam sobre ACOES, nao sobre
    features estaticas do estado.

    `xadrez.py` mediu o custo de nao ter isso: 66 de 668 posicoes tem vetor de
    features identico e rotulo diferente, e nenhuma funcao das features chega
    a 100%. Um unico atomo quantificado zera as colisoes.

    Aqui a lacuna e outra e vale dizer com precisao, senao a medida vira
    propaganda: os atomos deste arquivo incluem `pos=(i,j)`, entao o estado ja
    esta totalmente determinado e NAO ha colisao. O quantificador nao compra
    EXPRESSIVIDADE neste mundo — compra TRANSFERENCIA. "Beco sem saida" vale
    em qualquer labirinto; "pos=(3,5)" nao vale em nenhum outro.

    CUSTO REAL, e ele e a parte interessante: para avaliar "para toda acao..."
    o agente precisa ter TENTADO todas as acoes daquela celula. Enquanto uma
    delas for lacuna no modelo, o atomo e INDEFINIDO — devolve None — e a
    hipotese nao pode ser eliminada nem confirmada. O quantificador se paga em
    exploracao. Nao recebe o mundo de graca, como o planejador de
    `distante.py` recebia.
    """
    def saidas(e, mod) -> Optional[int]:
        p0 = e[0]
        n = 0
        for dy, dx in ACOES:
            alvo = (p0[0] + dy, p0[1] + dx)
            if not (0 <= alvo[0] < LADO and 0 <= alvo[1] < LADO):
                continue                      # fora do tabuleiro: nao anda
            if alvo in mod.parede:
                continue
            if alvo not in mod.livre:
                return None                   # lacuna: indefinido, nao chuta
            n += 1
        return n

    fora: List[Tuple[str, Callable]] = []
    for k in range(0, 5):
        fora.append((f"saidas=={k}",
                     lambda e, mod, k=k: (lambda s: None if s is None
                                          else s == k)(saidas(e, mod))))
    fora.append(("saidas<=1",
                 lambda e, mod: (lambda s: None if s is None else s <= 1)(
                     saidas(e, mod))))
    return fora


def gerar_hipoteses(n_chaves: int) -> List[Tuple[str, Callable]]:
    """Atomos e todas as conjuncoes de dois. Nenhuma marcada como verdadeira."""
    at = atomos(n_chaves)
    fora = list(at)
    for i in range(len(at)):
        for j in range(i + 1, len(at)):
            fa = at[i][1]
            fb = at[j][1]
            fora.append((f"{at[i][0]} & {at[j][0]}",
                         lambda e, fa=fa, fb=fb: fa(e) and fb(e)))
    return fora


# ══════════════════════════════════════════════ exploradores
def acao_acaso(modelo, e, cont, rng) -> int:
    return rng.randrange(4)


def acao_novidade(modelo, e, cont, rng) -> int:
    """
    Vai para onde AINDA NAO ESTEVE. Usa o proprio modelo de dinamica para
    prever aonde cada acao leva, e escolhe a que leva ao estado menos
    visitado. Se o modelo nao sabe prever (lacuna), isso e o mais novo que
    existe — vai la.

    Nao ha nada de LLM aqui nem de recompensa: e contagem de visitas sobre o
    modelo que o proprio agente deduziu. A exploracao ao acaso nunca achou
    "pegar as tres chaves" em 30 mundos e 240 mil acoes porque estados raros
    quase nao sao visitados por acaso. Esta politica mira exatamente isso.
    """
    melhor, melhores = None, []
    for a in range(4):
        prox = modelo.prever(e, a)
        if prox is None:
            pont = -1                         # desconhecido: novidade maxima
        elif prox == e:
            pont = 10 ** 9                    # bateu na parede: inutil
        else:
            pont = cont.get(prox, 0)
        if melhor is None or pont < melhor:
            melhor, melhores = pont, [a]
        elif pont == melhor:
            melhores.append(a)
    return rng.choice(melhores)


def acao_discriminante(modelo, e, cont, rng, hip=None, vivos=None):
    """
    Escolhe a acao que mais DIVIDE as hipoteses sobreviventes.

    A medicao que motivou isto: numa corrida tipica, 20 acoes matam ~150
    hipoteses gotejando, e a UNICA acao que produz vitoria mata 2.542 de
    2.545 — 99,9% de uma vez. Toda a informacao esta concentrada num tipo de
    evento, e o explorador por novidade chega nele por acaso.

    Aqui a acao e escolhida por eliminacao GARANTIDA. Para o estado que cada
    acao alcanca, conta-se quantas sobreviventes dizem "vitoria" ali (v) e
    quantas dizem que nao (n). Qualquer que seja a resposta do mundo, o lado
    errado morre inteiro — entao a eliminacao garantida e min(v, n), e ela e
    maxima quando as hipoteses estao mais divididas.

    Isso e projetar o experimento em vez de vagar. Empate cai em novidade,
    porque quando nada discrimina o melhor a fazer e ampliar o alcancavel.
    """
    if hip is None or vivos is None or not vivos:
        return acao_novidade(modelo, e, cont, rng)
    melhor, cands = -1, []
    for a in range(4):
        prox = modelo.prever(e, a)
        if prox is None:
            continue
        v = 0
        for k in vivos:
            if hip[k][1](prox):
                v += 1
        n = len(vivos) - v
        corte = min(v, n)
        if corte > melhor:
            melhor, cands = corte, [a]
        elif corte == melhor:
            cands.append(a)
    if melhor <= 0 or not cands:
        return acao_novidade(modelo, e, cont, rng)
    return rng.choice(cands)


def acao_planejada(modelo, e, cont, rng, hip=None, vivos=None):
    """
    Vai ATE o estado onde as sobreviventes mais discordam, planejando no
    modelo aprendido.

    `acao_discriminante` olhava um passo a frente e ficava pior que novidade
    (120 contra 99). O motivo nao era a ideia, era o alcance: o evento
    informativo — a vitoria — esta a ~20 passos, e nas celulas adjacentes
    TODAS as hipoteses concordam que ali nao e vitoria. `min(v,n)` dava zero
    em toda acao e caia de volta em novidade.

    Aqui a busca e em largura sobre o modelo: enumera estados alcancaveis,
    pontua cada um pela divisao que causaria (min(v,n) entre as
    sobreviventes), e devolve o primeiro passo do caminho ate o melhor. Isso
    e projetar o experimento e depois ir executa-lo.
    """
    if hip is None or vivos is None or len(vivos) < 2:
        return acao_novidade(modelo, e, cont, rng)

    from collections import deque
    fila = deque([(e, None, 0)])
    visto = {e}
    melhor, melhor_a = 0, None
    while fila:
        est, prim, d = fila.popleft()
        if d > 25:
            break
        if prim is not None:
            # UM EXPERIMENTO SO INFORMA SE FOR INEDITO.
            # A versao anterior pontuava o estado previsto sem memoria de que
            # ele ja tinha sido testado. A instrumentacao pegou: o planejador
            # achava plano discriminativo em 355/400 acoes, vencia 22 vezes
            # contra 5 da novidade — e visitava 33 estados distintos contra
            # 198, terminando com MAIS hipoteses vivas (41 contra 12). Ele
            # achava um jeito eficiente de vencer e repetia. Vinte e duas
            # vitorias no mesmo estado confirmam o que a primeira ja
            # confirmou.
            # O corte agora e descontado pelas visitas: um estado ja visto
            # nao divide nada, porque a resposta dele ja esta no historico.
            v = sum(1 for k in vivos if hip[k][1](est))
            corte = min(v, len(vivos) - v) / (1 + cont.get(est, 0))
            if corte > melhor:
                melhor, melhor_a = corte, prim
        for a in range(4):
            prox = modelo.prever(est, a)
            if prox is None or prox in visto:
                continue
            visto.add(prox)
            fila.append((prox, a if prim is None else prim, d + 1))
        if len(visto) > 4000:
            break
    if melhor_a is None:
        return acao_novidade(modelo, e, cont, rng)
    return melhor_a


EXPLORADORES = {"acaso": acao_acaso, "novidade": acao_novidade,
                "discriminante": acao_discriminante,
                "planejada": acao_planejada}


# ══════════════════════════════════════════════ episodio
def episodio(m: Mundo, max_acoes: int, semente: int, aberto: bool,
             sobrev: int, hip: Optional[List] = None,
             explorador: str = "acaso"
             ) -> Tuple[Optional[int], Optional[int], int]:
    """
    Explora registrando as duas curvas na MESMA moeda: acoes tomadas.
    `explorador` escolhe a politica; a contabilidade e identica.
    """
    rng = random.Random(semente)
    modelo = Modelo()
    escolher = EXPLORADORES[explorador]
    cont: dict = {}
    # ASSINATURA. "Sobram <= 5" era inalcancavel para todas_chaves: a classe
    # de hipoteses logicamente iguais a bolso>=3 tem 12 membros e o agente
    # a encontrava inteira, perfeitamente, e a regua dizia "nao convergiu".
    # O criterio certo: TODAS as sobreviventes dao a mesma resposta em TODO
    # estado alcancavel — sao funcionalmente o mesmo objetivo. Isso e
    # calculado uma vez por mundo, como assinatura de cada hipotese.
    assin: List[tuple] = []
    if aberto:
        est = m.estados_alcancaveis()
        assin = [tuple(f(e) for e in est) for _n, f in hip]
    hist: List[Tuple[tuple, int, tuple]] = []
    vivos: List[int] = list(range(len(hip))) if aberto else []
    nomes: List[str] = [] if aberto else list(OBJETIVOS)
    din_ok: Optional[int] = None
    obj_ok: Optional[int] = None
    m.reiniciar()

    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        if explorador in ("discriminante", "planejada"):
            a = escolher(modelo, e0, cont, rng, hip, vivos)
        else:
            a = escolher(modelo, e0, cont, rng)
        e1 = m.passo(a)
        cont[e1] = cont.get(e1, 0) + 1
        hist.append((e0, a, e1))
        modelo.aprender(e0, a, e1)

        if din_ok is None and len(hist) >= 20:
            if (modelo.cobertura() >= LADO * LADO - 6
                    and all(modelo.prever(x, b) == y for x, b, y in hist)):
                din_ok = t

        venceu = m.venceu(e1)
        if aberto:
            vivos = [k for k in vivos if hip[k][1](e1) == venceu]
            # convergiu = uma unica assinatura entre as sobreviventes
            n = len({assin[k] for k in vivos}) if vivos else 0
            alvo = 1
        else:
            nomes = [h for h in nomes
                     if alvo_verdadeiro(h, e1[0], e1[2], m.porta,
                                        m.n_chaves) == venceu]
            n, alvo = len(nomes), 1
        if obj_ok is None and n <= alvo:
            obj_ok = t
        if venceu:
            m.reiniciar()

    return din_ok, obj_ok, (len(vivos) if aberto else len(nomes))


def nome_verdadeiro(obj: str, m: Mundo) -> str:
    p = f"pos=({m.porta[0]},{m.porta[1]})"
    return {"porta": p,
            "todas_chaves": f"bolso>={m.n_chaves}",
            "uma_chave": "bolso>=1",
            "duas_chaves": "bolso>=2",
            "canto": "canto",
            "porta_com_uma": f"{p} & bolso>=1",
            "porta_com_todas": f"{p} & bolso>={m.n_chaves}"}[obj]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mundos", type=int, default=30)
    ap.add_argument("--acoes", type=int, default=4000)
    ap.add_argument("--semente", type=int, default=20260904)
    ap.add_argument("--aberto", action="store_true")
    ap.add_argument("--sobrev", type=int, default=5)
    ap.add_argument("--explorador", default="acaso",
                    choices=list(EXPLORADORES))
    a = ap.parse_args()

    hip = gerar_hipoteses(3) if a.aberto else None
    print("=" * 72)
    print("OBJETIVO ABERTO — o sistema GERA os predicados" if a.aberto
          else "OBJETIVO FECHADO — 7 hipoteses pre-declaradas")
    print("=" * 72)
    print(f"{a.mundos} mundos {LADO}x{LADO}, ate {a.acoes} acoes · "
          f"explorador: {a.explorador}")

    if a.aberto:
        nomes_hip = {n for n, _f in hip}
        print(f"{len(hip):,} hipoteses geradas · convergiu = sobreviventes "
              f"funcionalmente identicas em todo estado alcancavel")
        falta = [o for o in OBJETIVOS
                 if nome_verdadeiro(o, Mundo(a.semente, objetivo=o))
                 not in nomes_hip]
        print(f"G0 · objetivo verdadeiro expressavel: "
              f"{'todos os 7' if not falta else 'FALTA ' + str(falta)}")
        if falta:
            print("     Sem isso mede-se a gramatica, nao a assimetria. Abortando.")
            return
    print()

    print(f"{'objetivo verdadeiro':22s} {'dinamica':>10} {'objetivo':>10} "
          f"{'razao':>8}")
    print("-" * 54)
    TD: List[int] = []
    TO: List[int] = []
    for obj in OBJETIVOS:
        ds: List[int] = []
        os_: List[int] = []
        for k in range(a.mundos):
            m = Mundo(a.semente + k, objetivo=obj)
            d, o, _r = episodio(m, a.acoes, a.semente + 1000 + k,
                                a.aberto, a.sobrev, hip, a.explorador)
            if d is not None:
                ds.append(d)
            if o is not None:
                os_.append(o)
        # CENSURA: quem nao convergiu conta como >= horizonte. Descartar
        # esses casos puxava a media para baixo exatamente no objetivo mais
        # dificil (todas_chaves, 0/30) — foi o erro da rodada anterior.
        md = (sum(ds) + (a.mundos - len(ds)) * a.acoes) / a.mundos
        mo = (sum(os_) + (a.mundos - len(os_)) * a.acoes) / a.mundos
        raz = mo / md
        TD.append(md)
        TO.append(mo)
        print(f"{obj:22s} {md:>10.0f} {mo:>10.0f} {raz:>8.1f}x"
              f"   ({len(ds)}/{a.mundos} din, {len(os_)}/{a.mundos} obj)")

    print("-" * 54)
    MD = sum(TD) / len(TD)
    MO = sum(TO) / len(TO)
    print(f"{'MEDIA (censurada)':22s} {MD:>10.0f} {MO:>10.0f} {MO/MD:>8.1f}x")

    print("\n" + "=" * 72)
    if MO / MD >= 3:
        print(f"G1 · o objetivo custa {MO/MD:.1f}x mais acoes que a dinamica.")
        print("     A assimetria do Twin reproduz sem LLM.")
    elif MO / MD < 1:
        print("G3 · o objetivo ainda converge ANTES da dinamica.")
    else:
        print(f"G2 · custos parecidos ({MO/MD:.1f}x).")
    if a.aberto:
        print("\nNa versao FECHADA a razao media foi 0,7x. A diferenca entre")
        print("as duas mede quanto do problema era multipla escolha.")


if __name__ == "__main__":
    main()
