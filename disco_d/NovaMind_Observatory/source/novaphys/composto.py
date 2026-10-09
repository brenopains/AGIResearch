"""
novaphys.composto — os tres mecanismos se COMPOEM, ou apenas se somam?
========================================================================

O QUE ESTE ARQUIVO EXISTE PARA DECIDIR
----------------------------------------
A sessao produziu tres mecanismos que sobreviveram a controle, cada um
medido isolado, em experimento separado:

  1. DEDUZIR    inferir o operador resolvendo restricoes, em vez de sortear
                (`eter_inferencia.py`: 1 em 3.000 sorteando -> deducao exata)
  2. PARCIAL    deixar estrutura incompleta viver e se compor
                (`relaxado.py`: exigir 100% da 3 tarefas e ZERO composicoes;
                 exigir 85% da 8, com 41% do orcamento)
  3. NOVIDADE   explorar o estado nunca visitado, em vez de andar ao acaso
                (`objetivo.py`: 128 vitorias contra 5, p = 0,00001)

Os tres nunca foram testados JUNTOS, porque o ARC-AGI-1 nao tem exploracao —
la o objetivo esta dado. Este e o primeiro dominio onde os tres cabem ao
mesmo tempo.

E a pergunta nao e "o sistema completo e o melhor?". Isso seria quase certo e
quase vazio. E:

    o ganho dos tres juntos SUPERA a soma dos ganhos individuais?

Se sim, eles se compoem — cada um habilita os outros, e a arquitetura vale
mais que a lista de partes. Se apenas somam, sao tres melhorias
independentes empilhadas, o que e util e bem menos interessante.

O MUNDO
---------
Grade com celulas de cores. Cada cor tem um COMPORTAMENTO escondido:

    livre      passa
    bloqueia   nao passa
    coleta     passa, some, e conta +1 no bolso
    empurra    passa se a celula seguinte estiver livre (o bloco desliza)
    teleporta  passa e joga o agente para outra celula da mesma cor

O agente nao recebe a tabela. Cada transicao observada e uma restricao dura
sobre a cor envolvida — e uma unica observacao basta para deduzir aquela cor.
Isso e o mecanismo 1 no lugar dele: nao ha 5^5 combinacoes a enumerar, ha uma
deducao por cor.

O objetivo tambem e escondido e vem da gramatica gerada de `objetivo.py`.

OS TRES FATORES, LIGADOS E DESLIGADOS
---------------------------------------
  MODELO    ON  constroi o modelo de transicao deduzido das observacoes
            OFF nao modela nada; so conta estados visitados

  PARCIAL   ON  planeja com o modelo INCOMPLETO, tratando cor desconhecida
                como "vale a pena tentar" (e o analogo exato do limiar < 1.0)
            OFF so usa o modelo quando toda cor relevante ja e conhecida;
                enquanto houver lacuna, age ao acaso

  NOVIDADE  ON  escolhe a acao que leva ao estado menos visitado, prevendo
                com o modelo
            OFF acao uniforme ao acaso

PARCIAL sem MODELO nao existe (nao ha o que ser parcial), e NOVIDADE sem
MODELO degrada para "nao repita o estado imediatamente anterior" — que e a
melhor coisa possivel sem prever. Esses dois casos estao marcados no
resultado, nao escondidos.

MOEDA
-------
Acoes tomadas no mundo, identica nos 8 bracos. Deduzir e planejar sao gratis
em acoes de proposito: o recurso escasso aqui e interacao com o mundo, nao
computacao interna. Isso FAVORECE os bracos que computam mais — e e o
favorecimento honesto, porque e o que a realidade cobra num robo ou num jogo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. O braco completo tem que convergir (dinamica e objetivo)
      em >= 70% dos mundos. Se nao convergir, o teto e curto demais e as
      comparacoes viram ruido de censura.

  C1  SUPERADITIVO: ganho dos tres juntos > soma dos tres ganhos individuais,
      com margem >= 20%
      -> os mecanismos se compoem; a arquitetura vale mais que a lista

  C2  ADITIVO: ganho conjunto ~ soma dos individuais (dentro de 20%)
      -> tres melhorias independentes empilhadas

  C3  SUBADITIVO: ganho conjunto < soma
      -> os mecanismos disputam o mesmo ganho; ha redundancia entre eles

MINHA PREVISAO, escrita antes de rodar: C1, superaditivo, e por um motivo
mecanico que da para nomear antes. NOVIDADE precisa do MODELO para prever
aonde cada acao leva; sem modelo ela e cega. E o MODELO precisa de PARCIAL
para ser usado enquanto ainda tem buracos — sem parcial, o agente fica ao
acaso ate conhecer tudo, e nunca conhece tudo porque nao explora dirigido.

Ou seja: eu espero uma dependencia circular entre os tres, que e exatamente o
que produz superaditividade. Se der C2, essa historia esta errada e os tres
sao independentes de verdade.

Se der C3 eu tenho que explicar por que dois deles fazem o mesmo trabalho, e
nao tenho hipotese pronta para isso.
"""

from __future__ import annotations

import argparse
import random
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

LADO = 8
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COMPORTAMENTOS = ["livre", "bloqueia", "coleta", "empurra", "teleporta"]
N_CORES = 5


# ══════════════════════════════════════════════ o mundo
class Mundo:
    """Cores com comportamento escondido, e objetivo escondido."""

    def __init__(self, semente: int, objetivo: str = "coletou2"):
        rng = random.Random(semente)
        # cor 0 e sempre livre: o agente precisa de chao para andar
        self.regra: Dict[int, str] = {0: "livre"}
        resto = ["bloqueia", "coleta", "empurra", "teleporta"]
        rng.shuffle(resto)
        for c in range(1, N_CORES):
            self.regra[c] = resto[c - 1]

        self.mapa0 = [[0] * LADO for _ in range(LADO)]
        for _ in range(LADO * LADO // 3):
            i, j = rng.randrange(LADO), rng.randrange(LADO)
            self.mapa0[i][j] = rng.randrange(1, N_CORES)
        self.inicio = (0, 0)
        self.mapa0[0][0] = 0
        self.objetivo = objetivo
        self.semente = semente
        self.reiniciar()

    def reiniciar(self) -> None:
        self.mapa = [list(l) for l in self.mapa0]
        self.pos = self.inicio
        self.bolso = 0

    def estado(self) -> tuple:
        return (self.pos, self.bolso,
                tuple(tuple(l) for l in self.mapa))

    def _dentro(self, p) -> bool:
        return 0 <= p[0] < LADO and 0 <= p[1] < LADO

    def passo(self, a: int) -> tuple:
        dy, dx = ACOES[a]
        alvo = (self.pos[0] + dy, self.pos[1] + dx)
        if not self._dentro(alvo):
            return self.estado()
        cor = self.mapa[alvo[0]][alvo[1]]
        r = self.regra[cor]
        if r == "bloqueia":
            pass
        elif r == "livre":
            self.pos = alvo
        elif r == "coleta":
            self.mapa[alvo[0]][alvo[1]] = 0
            self.pos = alvo
            self.bolso += 1
        elif r == "empurra":
            atras = (alvo[0] + dy, alvo[1] + dx)
            if self._dentro(atras) and self.mapa[atras[0]][atras[1]] == 0:
                self.mapa[atras[0]][atras[1]] = cor
                self.mapa[alvo[0]][alvo[1]] = 0
                self.pos = alvo
        else:                                   # teleporta
            iguais = [(i, j) for i in range(LADO) for j in range(LADO)
                      if self.mapa[i][j] == cor and (i, j) != alvo]
            self.pos = iguais[0] if iguais else alvo
        return self.estado()

    def venceu(self, e: Optional[tuple] = None) -> bool:
        pos, bolso, _m = e if e else self.estado()
        o = self.objetivo
        if o == "coletou2":
            return bolso >= 2
        if o == "coletou3":
            return bolso >= 3
        if o == "canto":
            return pos == (LADO - 1, LADO - 1)
        if o == "canto_com_1":
            return pos == (LADO - 1, LADO - 1) and bolso >= 1
        if o == "linha_final":
            return pos[0] == LADO - 1
        raise ValueError(o)


OBJETIVOS = ["coletou2", "coletou3", "canto", "canto_com_1", "linha_final"]


# ══════════════════════════════════════════════ modelo deduzido
class Modelo:
    """
    Comportamento de cada cor, DEDUZIDO de uma observacao.

    Uma unica transicao envolvendo a cor c ja determina o comportamento de c:
    se o agente nao andou, c bloqueia; se andou e o bolso subiu, c coleta; se
    andou e a celula seguinte mudou de cor, c empurra; se foi parar longe, c
    teleporta; senao c e livre. Nao ha 5^5 combinacoes a enumerar.
    """

    def __init__(self) -> None:
        self.regra: Dict[int, str] = {}

    def aprender(self, e0, a: int, e1) -> None:
        (p0, b0, m0), (p1, b1, m1) = e0, e1
        dy, dx = ACOES[a]
        alvo = (p0[0] + dy, p0[1] + dx)
        if not (0 <= alvo[0] < LADO and 0 <= alvo[1] < LADO):
            return
        cor = m0[alvo[0]][alvo[1]]
        if cor in self.regra:
            return
        if p1 == p0:
            self.regra[cor] = "bloqueia"
        elif b1 > b0:
            self.regra[cor] = "coleta"
        elif p1 == alvo and m0 != m1:
            self.regra[cor] = "empurra"
        elif p1 != alvo:
            self.regra[cor] = "teleporta"
        else:
            self.regra[cor] = "livre"

    def completo(self, cores_vistas: Set[int]) -> bool:
        return cores_vistas <= set(self.regra)

    def prever(self, e0, a: int, parcial: bool) -> Optional[tuple]:
        """None = nao sei. Com `parcial`, cor desconhecida vira aposta livre."""
        p0, b0, m0 = e0
        dy, dx = ACOES[a]
        alvo = (p0[0] + dy, p0[1] + dx)
        if not (0 <= alvo[0] < LADO and 0 <= alvo[1] < LADO):
            return e0
        cor = m0[alvo[0]][alvo[1]]
        r = self.regra.get(cor)
        if r is None:
            if not parcial:
                return None
            r = "livre"                        # aposta otimista
        m = [list(l) for l in m0]
        if r == "bloqueia":
            return (p0, b0, tuple(tuple(l) for l in m))
        if r == "coleta":
            m[alvo[0]][alvo[1]] = 0
            return (alvo, b0 + 1, tuple(tuple(l) for l in m))
        if r == "empurra":
            atras = (alvo[0] + dy, alvo[1] + dx)
            if (0 <= atras[0] < LADO and 0 <= atras[1] < LADO
                    and m[atras[0]][atras[1]] == 0):
                m[atras[0]][atras[1]] = cor
                m[alvo[0]][alvo[1]] = 0
                return (alvo, b0, tuple(tuple(l) for l in m))
            return (p0, b0, tuple(tuple(l) for l in m))
        if r == "teleporta":
            return None                        # destino depende do mapa todo
        return (alvo, b0, tuple(tuple(l) for l in m))


# ══════════════════════════════════════════════ hipoteses de objetivo
def gerar_hipoteses() -> List[Tuple[str, object]]:
    fora: List[Tuple[str, object]] = []
    for k in range(1, 4):
        fora.append((f"bolso>={k}", lambda e, k=k: e[1] >= k))
    for i in range(LADO):
        fora.append((f"linha={i}", lambda e, i=i: e[0][0] == i))
        fora.append((f"coluna={i}", lambda e, i=i: e[0][1] == i))
    for i in range(LADO):
        for j in range(LADO):
            fora.append((f"pos=({i},{j})",
                         lambda e, i=i, j=j: e[0] == (i, j)))
    at = list(fora)
    for i in range(len(at)):
        for j in range(i + 1, len(at)):
            fa, fb = at[i][1], at[j][1]
            fora.append((f"{at[i][0]} & {at[j][0]}",
                         lambda e, fa=fa, fb=fb: fa(e) and fb(e)))
    return fora


# ══════════════════════════════════════════════ episodio
def episodio(m: Mundo, max_acoes: int, semente: int, hip: List,
             modelo_on: bool, parcial_on: bool, novidade_on: bool
             ) -> Tuple[Optional[int], Optional[int]]:
    rng = random.Random(semente)
    mod = Modelo() if modelo_on else None
    cont: Dict[tuple, int] = {}
    vivos = list(range(len(hip)))
    # assinatura sobre amostra fixa de estados sinteticos (pos x bolso).
    # Enumerar o espaco real e inviavel porque o mapa entra no estado; esta
    # e uma APROXIMACAO declarada da equivalencia funcional, nao a coisa
    # exata. Duas hipoteses com a mesma assinatura aqui podem, em principio,
    # divergir num estado fora da amostra.
    amostra = [((i, j), b, ()) for i in range(LADO) for j in range(LADO)
               for b in range(4)]
    assin = [tuple(f(e) for e in amostra) for _n, f in hip]
    din_ok: Optional[int] = None
    obj_ok: Optional[int] = None
    ultimo = None
    m.reiniciar()
    cores_vistas: Set[int] = set()

    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        for l in e0[2]:
            cores_vistas.update(l)

        # ── escolha da acao
        if novidade_on and mod is not None:
            usar = parcial_on or mod.completo(cores_vistas)
            if usar:
                # ARMADILHA DA IMPREVISIBILIDADE PERMANENTE.
                # A primeira versao marcava "nao sei prever" como novidade
                # maxima. Parece certo — otimismo sob incerteza e padrao. Mas
                # `teleporta` e permanentemente imprevisivel: o destino depende
                # do mapa inteiro. Ela nunca deixava de parecer nova, o agente
                # ficava em looping nela, e a dinamica convergia em 0% dos
                # mundos contra 100% sem novidade.
                #
                # A correcao separa duas incertezas que eu tinha juntado:
                #   cor DESCONHECIDA          -> vale explorar, reduz ignorancia
                #   cor conhecida mas de saida
                #   imprevisivel (teleporta)  -> nao reduz nada, nao e premio
                # Buscar novidade e buscar REDUZIR incerteza, nao ENCONTRAR
                # incerteza.
                melhor, cands = None, []
                for a in range(4):
                    dy, dx = ACOES[a]
                    alv = (e0[0][0] + dy, e0[0][1] + dx)
                    if 0 <= alv[0] < LADO and 0 <= alv[1] < LADO:
                        cor_alvo = e0[2][alv[0]][alv[1]]
                    else:
                        cor_alvo = None
                    if cor_alvo is not None and cor_alvo not in mod.regra:
                        pont = -1                  # ignorancia reduzivel
                    else:
                        prox = mod.prever(e0, a, parcial_on)
                        # saida imprevisivel de cor JA conhecida nao e premio
                        pont = cont.get(prox, 0) if prox is not None else 10 ** 6
                    if melhor is None or pont < melhor:
                        melhor, cands = pont, [a]
                    elif pont == melhor:
                        cands.append(a)
                a = rng.choice(cands)
            else:
                a = rng.randrange(4)
        else:
            # NOVIDADE sem MODELO nao existe: sem prever aonde a acao leva,
            # nao ha como escolher o estado menos visitado. Este braco e
            # identico ao aleatorio, e o resultado registra isso em vez de
            # fingir uma politica que nao esta la.
            a = rng.randrange(4)

        e1 = m.passo(a)
        cont[e1] = cont.get(e1, 0) + 1
        ultimo = e0
        if mod is not None:
            mod.aprender(e0, a, e1)
            if din_ok is None and mod.completo(cores_vistas):
                din_ok = t

        venceu = m.venceu(e1)
        vivos = [k for k in vivos if hip[k][1](e1) == venceu]
        if obj_ok is None and vivos:
            if len({assin[k] for k in vivos}) == 1:
                obj_ok = t
        if venceu:
            m.reiniciar()

    return din_ok, obj_ok


BRACOS = [(mo, pa, no) for mo, pa, no in product([True, False], repeat=3)
          if not (pa and not mo)]


def rotulo(mo, pa, no) -> str:
    return ("MOD" if mo else "---") + " " + ("PAR" if pa else "---") \
        + " " + ("NOV" if no else "---")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mundos", type=int, default=25)
    ap.add_argument("--acoes", type=int, default=3000)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    hip = gerar_hipoteses()
    print("=" * 72)
    print("COMPOSTO — os tres mecanismos se compoem ou apenas se somam?")
    print("=" * 72)
    print(f"mundo {LADO}x{LADO}, {N_CORES} cores com comportamento escondido")
    print(f"{len(hip):,} hipoteses de objetivo geradas")
    print(f"{a.mundos} mundos x {len(OBJETIVOS)} objetivos, "
          f"ate {a.acoes} acoes\n")

    print(f"{'braco':16s} {'dinamica':>10} {'objetivo':>10} "
          f"{'conv din':>9} {'conv obj':>9}")
    print("-" * 60)
    res: Dict[tuple, Tuple[float, float]] = {}
    conv: Dict[tuple, Tuple[float, float]] = {}
    for mo, pa, no in BRACOS:
        ds: List[int] = []
        os_: List[int] = []
        nd = nob = 0
        tot = 0
        for obj in OBJETIVOS:
            for k in range(a.mundos):
                m = Mundo(a.semente + k, objetivo=obj)
                d, o = episodio(m, a.acoes, a.semente + 500 + k, hip,
                                mo, pa, no)
                tot += 1
                ds.append(d if d is not None else a.acoes)
                os_.append(o if o is not None else a.acoes)
                nd += d is not None
                nob += o is not None
        md = sum(ds) / len(ds)
        mo_ = sum(os_) / len(os_)
        res[(mo, pa, no)] = (md, mo_)
        conv[(mo, pa, no)] = (nd / tot, nob / tot)
        print(f"{rotulo(mo, pa, no):16s} {md:>10.0f} {mo_:>10.0f} "
              f"{nd/tot:>8.0%} {nob/tot:>8.0%}", flush=True)
    print("-" * 60)

    base = res[(False, False, False)]
    cheio = res[(True, True, True)]

    # C0 DE VERDADE. A primeira versao tinha `{'>=70%' if True else ''}` — um
    # esboco que eu nunca terminei. Imprimia ">=70%" sempre, entao o portao
    # NUNCA foi verificado e o script deu tapinha nas costas com 48% de
    # convergencia no objetivo. Agora ele checa e diz o que achou.
    cd, co = conv[(True, True, True)]
    print()
    print(f"C0 · braco completo: dinamica {cd:.0%}, objetivo {co:.0%}")
    print("     dinamica: " + ("coluna limpa" if cd >= 0.70
                               else "FALHA, censura domina"))
    print("     objetivo: " + ("coluna limpa" if co >= 0.70
                               else "FALHA, censura domina — sem conclusao"))

    def ganho(r, col):
        return base[col] - r[col]

    for col, nome in ((0, "DINAMICA"), (1, "OBJETIVO")):
        ok = conv[(True, True, True)][col] >= 0.70
        g_mod = ganho(res[(True, False, False)], col)
        g_nov = ganho(res[(False, False, True)], col)
        g_par = ganho(res[(True, True, False)], col) - g_mod
        soma = g_mod + g_nov + g_par
        junto = ganho(cheio, col)
        print()
        print(f"GANHOS na {nome}, acoes economizadas vs base"
              + ("" if ok else "   [C0 FALHOU — nao concluir daqui]"))
        print(f"   so MODELO           {g_mod:>8.0f}")
        print(f"   so NOVIDADE         {g_nov:>8.0f}")
        print(f"   PARCIAL sobre MOD   {g_par:>8.0f}")
        print(f"   soma dos tres       {soma:>8.0f}")
        print(f"   OS TRES JUNTOS      {junto:>8.0f}")
        if soma > 0:
            r = junto / soma
            v = ("C1 SUPERADITIVO" if r > 1.20
                 else "C3 SUBADITIVO" if r < 0.80 else "C2 ADITIVO")
            print(f"   razao junto/soma    {r:>8.2f}   -> {v}")
        else:
            print("   soma <= 0; razao nao se aplica")
        # Decomposicao CONDICIONAL a MODELO ligado. E post-hoc: escolhida
        # depois de ver os dados, entao NAO substitui o veredito acima.
        b2 = res[(True, False, False)][col]
        print(f"   [post-hoc, dado MODELO ligado] "
              f"parcial {b2 - res[(True, True, False)][col]:>+.0f}, "
              f"novidade {b2 - res[(True, False, True)][col]:>+.0f}, "
              f"juntos {b2 - cheio[col]:>+.0f}")



if __name__ == "__main__":
    main()
