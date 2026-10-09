"""
novaphys.vivo — tudo e vivo, inclusive o que era cenario
=========================================================

O QUE EU TINHA FEITO ERRADO
-----------------------------
`coevolucao.py` fez as plantas serem CENARIO: reapareciam em posicao
aleatoria, densidade fixa, indiferentes a tudo. Depois chamei o outro braco
de "coevolucao" — mas eram agentes identicos se comendo num mundo cujo
recurso continuava estatico.

Isso codifica o problema dentro do proprio teste. A literatura diz que o
plato vem de REGRAS FIXAS E AMBIENTE ESTATICO, e eu deixei o ambiente
estatico e testei se coevolucao ajuda. Nao havia coevolucao em lugar nenhum.

O MECANISMO QUE FALTAVA, E ELE E REAL
---------------------------------------
Planta nao comida nao e dispersa, logo nao se reproduz. Isso acopla as duas
populacoes nos DOIS sentidos:

    planta indefesa demais  -> comida rapido demais, some
    planta defendida demais -> nao e comida, nao e dispersa, some
    o otimo depende de QUANTOS e COMO sao os herbivoros
    e os herbivoros dependem de QUAIS plantas existem

Nenhum dos dois lados tem otimo fixo. O espaco de problemas nao fecha por
construcao — que e exatamente a condicao que Tierra e Avida nao tinham.

O DESENHO, DUAS VARIAVEIS ISOLADAS
------------------------------------
    braco CENARIO   plantas reaparecem sozinhas, densidade fixa (o antigo)
    braco VIVO      plantas tem genoma, so se reproduzem por dispersao no
                    intestino, e evoluem defesa e nutricao

Mesmos herbivoros, mesma economia de energia, mesma mutacao. A unica
diferenca e se a planta e organismo ou mobilia.

MEDIDA
--------
Complexidade efetiva dos HERBIVOROS (regras que realmente disparam), como
antes. Mais a diversidade genetica das plantas, que diz se o lado vegetal
esta de fato evoluindo em vez de convergir e travar.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  V0  PRE-CONDICAO. As duas populacoes sobrevivem >= 200 geracoes no braco
      VIVO. Se extinguir, nao testei nada — e foi o que aconteceu tres vezes
      no arquivo anterior.

  V1  CENARIO estabiliza e VIVO continua subindo
      -> "tudo tem que ser vivo" tem base empirica; o eixo e acoplamento
         mutuo, nao riqueza

  V2  os dois estabilizam
      -> nem acoplamento mutuo sustenta nesta escala

  V3  VIVO extingue -> nao testei; reportar como falha de instrumento, nao
      como resultado

MINHA PREVISAO: V2, mas com menos confianca que antes. O acoplamento aqui e
real e bidirecional, e essa e a primeira vez nesta sessao que o ambiente
responde ao agente. Se eu estiver errado, e o resultado mais interessante da
linha.

SEMENTE PRE-REGISTRADA: 20260826
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

SEMENTE = 20260826
LADO = 24
N_BICHOS = 120
N_PLANTAS = 400
MAX_POP = 400
MAX_PLANTAS = 700
GERACOES = 400
PASSOS = 40

CUSTO_REGRA = 0.005
CUSTO_VIVER = 0.03
E_INICIAL = 8.0
E_REPRODUZ = 10.0
MAX_REGRAS = 24

# SENSORES DE VERDADE, nao so um numero.
# A primeira tentativa de testar o teto sensorio-motor mudou a constante
# `N_SENS` mas deixou a lista de sensores codificada a mao com 8 itens — o
# genoma passava a referenciar indices inexistentes e estourava. Aqui os
# sensores sao funcoes de fato, e cortar a lista muda o que o bicho PERCEBE.
def _sensores(b, plantas, viz, rng, LADO, E_INICIAL, E_REPRODUZ):
    aqui = plantas.get((b.x, b.y))
    perto = [plantas[p] for p in viz if p in plantas]
    return [
        bool(perto),                                   # 0 planta por perto
        aqui is not None,                              # 1 planta aqui
        b.e < E_INICIAL * 0.5,                         # 2 energia baixa
        aqui is not None and aqui.defesa > 1.0,        # 3 planta defendida
        bool(b.gut),                                   # 4 comeu ha pouco
        b.e > E_REPRODUZ * 0.8,                        # 5 energia alta
        rng.random() < 0.5,                            # 6 ruido
        True,                                          # 7 sempre
        # --- alem de 8: informacao direcional e graduada
        ((b.x + 1) % LADO, b.y) in plantas,            # 8 planta a direita
        ((b.x - 1) % LADO, b.y) in plantas,            # 9 planta a esquerda
        (b.x, (b.y + 1) % LADO) in plantas,            # 10 planta acima
        (b.x, (b.y - 1) % LADO) in plantas,            # 11 planta abaixo
        len(perto) >= 3,                               # 12 muita planta perto
        aqui is not None and aqui.nutri > 4.0,         # 13 planta nutritiva
        b.e < E_INICIAL * 0.25,                        # 14 energia critica
        len(b.gut) >= 2,                               # 15 barriga cheia
    ]


N_SENS_MAX = 16
N_SENS = 8
N_ACOES = 6          # 0-3 mover, 4 comer, 5 parado
GESTACAO = 6         # passos ate a semente sair


class Planta:
    """
    Genoma: defesa (custa energia de quem come) e nutricao. Reproduz-se de
    dois modos, e o segundo e o acoplamento:
        1. broto local, taxa baixa — a planta sozinha quase nao se espalha
        2. semente dispersa pelo herbivoro que a comeu, alguns passos depois
    """
    __slots__ = ("defesa", "nutri", "x", "y")

    def __init__(self, defesa, nutri, x, y):
        self.defesa, self.nutri = defesa, nutri
        self.x, self.y = x, y

    def filho(self, rng, x, y):
        d = min(3.0, max(0.0, self.defesa + rng.gauss(0, 0.25)))
        n = min(6.0, max(0.5, self.nutri + rng.gauss(0, 0.25)))
        return Planta(d, n, x, y)


class Bicho:
    __slots__ = ("regras", "x", "y", "e", "disparos", "gut")

    def __init__(self, regras, x, y):
        self.regras = regras
        self.x, self.y = x, y
        self.e = E_INICIAL
        self.disparos = set()
        self.gut: List[Tuple[int, Planta]] = []   # (passos restantes, planta)

    def agir(self, sens):
        for i, (s, neg, a) in enumerate(self.regras):
            self.e -= CUSTO_REGRA
            if (sens[s] != neg):
                self.disparos.add(i)
                return a
        return N_ACOES - 1


def nascer(rng):
    n = rng.randint(1, 3)
    return [(rng.randrange(N_SENS), rng.random() < 0.5, rng.randrange(N_ACOES))
            for _ in range(n)]


def mutar(regras, rng):
    r = [list(x) for x in regras]
    op = rng.random()
    if op < 0.35 and len(r) < MAX_REGRAS:
        r.insert(rng.randrange(len(r) + 1),
                 [rng.randrange(N_SENS), rng.random() < 0.5,
                  rng.randrange(N_ACOES)])
    elif op < 0.55 and len(r) > 1:
        r.pop(rng.randrange(len(r)))
    elif op < 0.8:
        i, j = rng.randrange(len(r)), rng.randrange(3)
        r[i][j] = (rng.randrange(N_SENS) if j == 0 else
                   (rng.random() < 0.5 if j == 1 else rng.randrange(N_ACOES)))
    elif len(r) > 1:
        i, j = rng.randrange(len(r)), rng.randrange(len(r))
        r[i], r[j] = r[j], r[i]
    return [tuple(x) for x in r]


def rodar(vivo: bool, semente: int, geracoes: int = GERACOES):
    rng = random.Random(semente)
    pop = [Bicho(nascer(rng), rng.randrange(LADO), rng.randrange(LADO))
           for _ in range(N_BICHOS)]
    plantas: Dict[Tuple[int, int], Planta] = {}
    for _ in range(N_PLANTAS):
        x, y = rng.randrange(LADO), rng.randrange(LADO)
        plantas[(x, y)] = Planta(0.5, 4.0, x, y)
    hist = []

    for g in range(geracoes):
        for _ in range(PASSOS):
            for b in pop:
                viz = [((b.x + dx) % LADO, (b.y + dy) % LADO)
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                sens = _sensores(b, plantas, viz, rng, LADO,
                                 E_INICIAL, E_REPRODUZ)[:N_SENS]
                a = b.agir(sens)
                if a == 0:
                    b.x = (b.x + 1) % LADO
                elif a == 1:
                    b.x = (b.x - 1) % LADO
                elif a == 2:
                    b.y = (b.y + 1) % LADO
                elif a == 3:
                    b.y = (b.y - 1) % LADO
                elif a == 4:
                    pl = plantas.pop((b.x, b.y), None)
                    if pl is not None:
                        b.e += pl.nutri - pl.defesa      # defesa custa
                        if vivo:
                            b.gut.append((GESTACAO, pl))
                b.e -= CUSTO_VIVER
                # semente sai onde o bicho estiver agora
                if vivo and b.gut:
                    resta = []
                    for t, pl in b.gut:
                        if t <= 1:
                            if (b.x, b.y) not in plantas \
                                    and len(plantas) < MAX_PLANTAS:
                                plantas[(b.x, b.y)] = pl.filho(rng, b.x, b.y)
                        else:
                            resta.append((t - 1, pl))
                    b.gut = resta

            if vivo:
                # broto local raro: sozinha a planta quase nao se espalha
                for pl in list(plantas.values()):
                    if rng.random() < 0.004 and len(plantas) < MAX_PLANTAS:
                        nx = (pl.x + rng.randint(-1, 1)) % LADO
                        ny = (pl.y + rng.randint(-1, 1)) % LADO
                        if (nx, ny) not in plantas:
                            plantas[(nx, ny)] = pl.filho(rng, nx, ny)
            else:
                while len(plantas) < N_PLANTAS:
                    x, y = rng.randrange(LADO), rng.randrange(LADO)
                    if (x, y) not in plantas:
                        plantas[(x, y)] = Planta(0.5, 4.0, x, y)

            pop = [b for b in pop if b.e > 0]
            if not pop or (vivo and not plantas):
                break

        if not pop:
            hist.append((g, 0.0, 0, 0, 0.0, 0.0))
            break
        filhos = []
        for b in pop:
            if b.e >= E_REPRODUZ and len(pop) + len(filhos) < MAX_POP:
                b.e /= 2
                f = Bicho(mutar(b.regras, rng), b.x, b.y)
                f.e = b.e
                filhos.append(f)
        pop.extend(filhos)
        if len(pop) > MAX_POP:
            pop.sort(key=lambda b: -b.e)
            pop = pop[:MAX_POP]
        efet = st.mean(len(b.disparos) for b in pop)
        defs = [p.defesa for p in plantas.values()]
        nuts = [p.nutri for p in plantas.values()]
        hist.append((g, efet, len(pop), len(plantas),
                     st.mean(defs) if defs else 0.0,
                     st.pstdev(defs) if len(defs) > 1 else 0.0))
        for b in pop:
            b.disparos = set()
        if vivo and not plantas:
            break
    return hist


def incl(h, i0, i1):
    seg = [x[1] for x in h[i0:i1]]
    if len(seg) < 10:
        return None
    m = len(seg) // 2
    return st.mean(seg[m:]) - st.mean(seg[:m])


def relatorio(sementes: Sequence[int] = (0, 1, 2)) -> None:
    print("=" * 78)
    print("VIVO — plantas como organismos, nao como mobilia")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  V0  as duas populacoes sobrevivem >= 200 geracoes no braco VIVO")
    print("  V1  CENARIO estabiliza e VIVO continua subindo")
    print("  V2  os dois estabilizam")
    print("  V3  VIVO extingue -> falha de instrumento, nao resultado")
    print("\nMINHA PREVISAO: V2, com menos confianca. O acoplamento aqui e")
    print("bidirecional de verdade — e a primeira vez que o ambiente responde.")

    res = {}
    for vivo, nome in ((False, "CENARIO"), (True, "VIVO")):
        hs = [rodar(vivo, SEMENTE + s) for s in sementes]
        res[nome] = hs
        print(f"\n--- {nome}")
        print(f"{'ger':>6}{'efetivas':>10}{'bichos':>8}{'plantas':>9}"
              f"{'defesa':>9}{'dp def':>8}")
        for g in (0, 25, 50, 100, 200, 300, GERACOES - 1):
            v = [h[g] for h in hs if g < len(h)]
            if not v:
                print(f"{g:>6}   extinto")
                continue
            print(f"{g:>6}{st.mean(x[1] for x in v):>10.2f}"
                  f"{st.mean(x[2] for x in v):>8.0f}"
                  f"{st.mean(x[3] for x in v):>9.0f}"
                  f"{st.mean(x[4] for x in v):>9.2f}"
                  f"{st.mean(x[5] for x in v):>8.2f}")

    print("\n" + "=" * 78)
    print("INCLINACAO da complexidade efetiva dos herbivoros")
    print(f"{'braco':<10}{'inicio (0-100)':>18}{'fim (200-400)':>18}")
    inc = {}
    for nome in ("CENARIO", "VIVO"):
        a = [incl(h, 0, 100) for h in res[nome]]
        b = [incl(h, 200, GERACOES) for h in res[nome]]
        a = [x for x in a if x is not None]
        b = [x for x in b if x is not None]
        inc[nome] = (st.mean(a) if a else None, st.mean(b) if b else None)
        fa = "—" if inc[nome][0] is None else f"{inc[nome][0]:+.3f}"
        fb = "—" if inc[nome][1] is None else f"{inc[nome][1]:+.3f}"
        print(f"{nome:<10}{fa:>18}{fb:>18}")

    print()
    vf = inc["VIVO"][1]
    cf = inc["CENARIO"][1]
    if vf is None:
        print("V3 · O BRACO VIVO NAO CHEGOU AO FIM. Nao testei nada — isso e")
        print("     falha de instrumento, nao resultado sobre coevolucao.")
    elif cf is not None and vf > 0.02 and cf <= 0.02:
        print("V1 · O ACOPLAMENTO SUSTENTA. Com as plantas vivas a complexidade")
        print("     continua subindo onde o cenario estatico estagnou.")
        print("     'Tudo tem que ser vivo' ganha base empirica.")
    elif cf is not None and vf <= 0.02 and cf <= 0.02:
        print("V2 · OS DOIS ESTABILIZAM. Nem acoplamento mutuo bidirecional")
        print("     sustenta crescimento nesta escala.")
    else:
        print(f"Parcial: VIVO {vf:+.3f} · CENARIO {cf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()
    relatorio(sementes=tuple(range(a.sementes)))
