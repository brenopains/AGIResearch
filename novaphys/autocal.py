"""
novaphys.autocal — o metabolismo vai para o genoma, e os nichos existem
========================================================================

A CRITICA QUE ORIGINOU ISTO, E ELA PROCEDE
--------------------------------------------
Nos arquivos anteriores eu FIXEI o custo de viver, o custo de pensar, o
limiar de reproducao e a densidade de comida — e depois varri esses valores
a mao ate o mundo nao extinguir. Tres rodadas morreram por eu ter escolhido
um numero global errado para todo mundo.

Isso e o projetista escolhendo o ponto de operacao. Nao e o sistema achando.

O QUE MUDA AQUI
-----------------
    1. METABOLISMO NO GENOMA
       custo de viver, custo de pensar, limiar de reproducao e eficiencia
       digestiva sao herdaveis e mutam. Nao existe valor certo a descobrir:
       cada linhagem carrega o seu, e quem escolhe mal morre sozinho.

    2. NICHOS
       tres tipos de recurso, cada um exigindo uma enzima diferente. Um bicho
       bom em um e ruim nos outros (soma das enzimas limitada). Estrategia
       que falha num lugar prospera noutro, e extincao de um ramo nao leva o
       mundo junto.

O QUE NAO MUDA, E EU NAO VOU FINGIR QUE MUDA
----------------------------------------------
Autocalibracao empurra a minha escolha UM NIVEL ACIMA. Eu ainda decido o que
e herdavel, a faixa de cada gene e a taxa de mutacao. E o mesmo regresso da
maquina de referencia que ja apareceu nesta sessao: nao fecha, so recua.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. A populacao sobrevive 400 geracoes SEM eu ajustar
      nenhuma constante de energia. Se extinguir, autocalibracao nao
      resolveu nem a viabilidade.

  A1  os genes metabolicos DIVERSIFICAM em vez de convergir para um valor
      -> autocalibracao de fato aconteceu, e ha nichos ocupados

  A2  a complexidade efetiva cresce onde antes estagnava
      -> autocalibracao e a peca que faltava

  A3  sobrevive e diversifica, mas a complexidade continua plana
      -> autocalibracao resolve VIABILIDADE, nao abertura. Sao problemas
         diferentes, e so o primeiro foi resolvido.

MINHA PREVISAO: A3. Espero que os genes metabolicos diversifiquem de verdade
e que a populacao pare de extinguir — isso a critica acerta. Mas nao espero
que a complexidade cognitiva cresca, porque nada aqui ainda cria pressao por
POLITICA mais elaborada, so por metabolismo melhor ajustado.

SEMENTE PRE-REGISTRADA: 20260826
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from typing import Dict, List, Sequence, Tuple

SEMENTE = 20260826
LADO = 28
N_INICIAL = 150
MAX_POP = 500
GERACOES = 400
PASSOS = 40
MAX_REGRAS = 24
N_TIPOS = 3            # nichos: tres recursos, tres enzimas
N_SENS = 12
N_ACOES = 6

# FAIXAS dos genes metabolicos — a minha escolha que sobra, declarada
FAIXA = {
    "viver": (0.005, 0.20),
    "pensar": (0.001, 0.05),
    "reproduz": (5.0, 30.0),
}


class Rec:
    __slots__ = ("tipo", "x", "y")

    def __init__(self, tipo, x, y):
        self.tipo, self.x, self.y = tipo, x, y


class Bicho:
    __slots__ = ("regras", "x", "y", "e", "disparos", "viver", "pensar",
                 "reproduz", "enz", "idade")

    def __init__(self, regras, x, y, viver, pensar, reproduz, enz):
        self.regras = regras
        self.x, self.y = x, y
        self.viver, self.pensar, self.reproduz = viver, pensar, reproduz
        self.enz = enz                      # soma <= 1, reparte entre tipos
        self.e = reproduz * 0.6             # nasce com fracao do proprio limiar
        self.disparos = set()
        self.idade = 0

    def agir(self, sens):
        for i, (s, neg, a) in enumerate(self.regras):
            self.e -= self.pensar
            if sens[s] != neg:
                self.disparos.add(i)
                return a
        return N_ACOES - 1


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def nascer(rng):
    regras = [(rng.randrange(N_SENS), rng.random() < 0.5, rng.randrange(N_ACOES))
              for _ in range(rng.randint(1, 3))]
    enz = [rng.random() for _ in range(N_TIPOS)]
    s = sum(enz) or 1.0
    return Bicho(regras, rng.randrange(LADO), rng.randrange(LADO),
                 rng.uniform(*FAIXA["viver"]), rng.uniform(*FAIXA["pensar"]),
                 rng.uniform(*FAIXA["reproduz"]), [e / s for e in enz])


def mutar(p: Bicho, rng) -> Bicho:
    r = [list(x) for x in p.regras]
    op = rng.random()
    if op < 0.30 and len(r) < MAX_REGRAS:
        r.insert(rng.randrange(len(r) + 1),
                 [rng.randrange(N_SENS), rng.random() < 0.5,
                  rng.randrange(N_ACOES)])
    elif op < 0.50 and len(r) > 1:
        r.pop(rng.randrange(len(r)))
    elif op < 0.75:
        i, j = rng.randrange(len(r)), rng.randrange(3)
        r[i][j] = (rng.randrange(N_SENS) if j == 0 else
                   (rng.random() < 0.5 if j == 1 else rng.randrange(N_ACOES)))
    elif len(r) > 1:
        i, j = rng.randrange(len(r)), rng.randrange(len(r))
        r[i], r[j] = r[j], r[i]
    v = _clamp(p.viver * rng.lognormvariate(0, 0.12), *FAIXA["viver"])
    t = _clamp(p.pensar * rng.lognormvariate(0, 0.12), *FAIXA["pensar"])
    q = _clamp(p.reproduz * rng.lognormvariate(0, 0.12), *FAIXA["reproduz"])
    enz = [max(0.0, e + rng.gauss(0, 0.08)) for e in p.enz]
    s = sum(enz) or 1.0
    return Bicho([tuple(x) for x in r], p.x, p.y, v, t, q, [e / s for e in enz])


def rodar(semente: int, geracoes: int = GERACOES):
    rng = random.Random(semente)
    pop = [nascer(rng) for _ in range(N_INICIAL)]
    # nichos espaciais: cada tipo predomina numa faixa do mundo
    recs: Dict[Tuple[int, int], Rec] = {}

    def repor():
        while len(recs) < 500:
            x, y = rng.randrange(LADO), rng.randrange(LADO)
            if (x, y) in recs:
                continue
            tipo = min(N_TIPOS - 1, (x * N_TIPOS) // LADO)
            if rng.random() < 0.25:
                tipo = rng.randrange(N_TIPOS)     # mistura, nicho nao e puro
            recs[(x, y)] = Rec(tipo, x, y)
    repor()

    hist = []
    for g in range(geracoes):
        for _ in range(PASSOS):
            for b in pop:
                viz = [((b.x + dx) % LADO, (b.y + dy) % LADO)
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                aqui = recs.get((b.x, b.y))
                sens = [
                    any(p in recs for p in viz),
                    aqui is not None,
                    b.e < b.reproduz * 0.3,
                    b.e > b.reproduz * 0.8,
                    aqui is not None and aqui.tipo == 0,
                    aqui is not None and aqui.tipo == 1,
                    aqui is not None and aqui.tipo == 2,
                    ((b.x + 1) % LADO, b.y) in recs,
                    ((b.x - 1) % LADO, b.y) in recs,
                    (b.x, (b.y + 1) % LADO) in recs,
                    rng.random() < 0.5,
                    True,
                ]
                a = b.agir(sens)
                if a == 0:
                    b.x = (b.x + 1) % LADO
                elif a == 1:
                    b.x = (b.x - 1) % LADO
                elif a == 2:
                    b.y = (b.y + 1) % LADO
                elif a == 3:
                    b.y = (b.y - 1) % LADO
                elif a == 4 and aqui is not None:
                    del recs[(b.x, b.y)]
                    b.e += 6.0 * b.enz[aqui.tipo]     # enzima decide o ganho
                b.e -= b.viver
                b.idade += 1
            repor()
            pop = [b for b in pop if b.e > 0]
            if not pop:
                break
        if not pop:
            hist.append(None)
            break
        filhos = []
        for b in pop:
            if b.e >= b.reproduz and len(pop) + len(filhos) < MAX_POP:
                b.e /= 2
                f = mutar(b, rng)
                f.e = b.e
                f.x, f.y = b.x, b.y
                filhos.append(f)
        pop.extend(filhos)
        if len(pop) > MAX_POP:
            pop.sort(key=lambda b: -b.e)
            pop = pop[:MAX_POP]
        hist.append({
            "g": g, "n": len(pop),
            "efet": st.mean(len(b.disparos) for b in pop),
            "tam": st.mean(len(b.regras) for b in pop),
            "viver": st.mean(b.viver for b in pop),
            "viver_dp": st.pstdev([b.viver for b in pop]) if len(pop) > 1 else 0,
            "reproduz": st.mean(b.reproduz for b in pop),
            "esp": st.mean(max(b.enz) for b in pop),      # especializacao
            "esp_dp": st.pstdev([max(range(N_TIPOS), key=lambda i: b.enz[i])
                                 for b in pop]) if len(pop) > 1 else 0,
        })
        for b in pop:
            b.disparos = set()
    return hist


def relatorio(sementes: Sequence[int] = (0, 1, 2)) -> None:
    print("=" * 78)
    print("AUTOCALIBRACAO — metabolismo no genoma, nichos no mundo")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  A0  sobrevive 400 geracoes SEM eu ajustar constante de energia")
    print("  A1  genes metabolicos DIVERSIFICAM em vez de convergir")
    print("  A2  a complexidade cresce onde antes estagnava")
    print("  A3  sobrevive e diversifica, mas complexidade plana")
    print("      -> autocalibracao resolve VIABILIDADE, nao abertura")
    print("\nMINHA PREVISAO: A3.")
    print("\nfaixas que EU escolhi (o regresso que sobra, declarado):")
    for k, (lo, hi) in FAIXA.items():
        print(f"     {k:<10} {lo} a {hi}")

    hs = [rodar(SEMENTE + s) for s in sementes]
    vivos = [h for h in hs if h and h[-1] is not None]
    print(f"\n{len(vivos)}/{len(hs)} sementes chegaram ao fim")

    print(f"\n{'ger':>6}{'pop':>7}{'efetivas':>10}{'tamanho':>9}"
          f"{'viver':>9}{'dp viver':>10}{'reproduz':>10}{'espec':>8}")
    for g in (0, 25, 50, 100, 200, 300, GERACOES - 1):
        v = [h[g] for h in vivos if g < len(h) and h[g]]
        if not v:
            print(f"{g:>6}   extinto")
            continue
        print(f"{g:>6}{st.mean(x['n'] for x in v):>7.0f}"
              f"{st.mean(x['efet'] for x in v):>10.2f}"
              f"{st.mean(x['tam'] for x in v):>9.2f}"
              f"{st.mean(x['viver'] for x in v):>9.3f}"
              f"{st.mean(x['viver_dp'] for x in v):>10.3f}"
              f"{st.mean(x['reproduz'] for x in v):>10.2f}"
              f"{st.mean(x['esp'] for x in v):>8.2f}")

    print("\n" + "=" * 78)
    if not vivos:
        print("A0 · REPROVADA. Extinguiu mesmo com metabolismo herdavel.")
        print("     Autocalibracao nao resolveu nem a viabilidade.")
        return
    print(f"A0 · aprovada — {len(vivos)}/{len(hs)} sobrevivem sem eu ajustar nada")

    ini = [h[10] for h in vivos if len(h) > 10 and h[10]]
    fim = [h[-1] for h in vivos if h[-1]]
    dp0 = st.mean(x["viver_dp"] for x in ini)
    dp1 = st.mean(x["viver_dp"] for x in fim)
    esp0 = st.mean(x["esp"] for x in ini)
    esp1 = st.mean(x["esp"] for x in fim)
    print(f"\nA1 · diversidade do custo de viver: {dp0:.3f} -> {dp1:.3f}")
    print(f"     especializacao enzimatica media: {esp0:.2f} -> {esp1:.2f}")
    diversificou = dp1 > 0.01 or esp1 > esp0 + 0.05
    print(f"     {'diversificou' if diversificou else 'CONVERGIU — sem nichos'}")

    meio = [h[len(h) // 2] for h in vivos if h[len(h) // 2]]
    e_meio = st.mean(x["efet"] for x in meio)
    e_fim = st.mean(x["efet"] for x in fim)
    print(f"\nA2/A3 · complexidade efetiva: meio {e_meio:.2f} -> fim {e_fim:.2f}"
          f"  ({e_fim - e_meio:+.2f})")
    if e_fim > e_meio + 0.15:
        print("     A2 · CRESCE. Autocalibracao era a peca que faltava.")
    elif diversificou:
        print("     A3 · autocalibracao resolveu VIABILIDADE e produziu nichos,")
        print("     mas nao abertura. Sao dois problemas, e so um caiu.")
    else:
        print("     nem diversificou nem cresceu.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()
    relatorio(sementes=tuple(range(a.sementes)))
