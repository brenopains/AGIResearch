"""
novaphys.cortical — o mapa retina-cortex: trocar coordenada em vez de aprender
===============================================================================

DE ONDE VEM
-------------
Bressloff, Cowan, Golubitsky, Thomas e Wiener (Phil Trans R Soc B, 2001)
mostraram que as quatro "form constants" de Kluver — grades e favos, teias,
tuneis e funis, espirais — nao sao conteudo alucinado: sao os modos de
instabilidade espontanea do V1, e correspondem aos subgrupos axiais de E(2).
O que faz aquilo virar espiral na experiencia visual e um fato estrutural: o
mapa retina->cortex e aproximadamente LOG-POLAR.

E nessas coordenadas:

    rotacao em torno da fovea  ->  TRANSLACAO no eixo angular
    escala                     ->  TRANSLACAO no eixo radial
    translacao                 ->  deixa de ser simples

O cerebro nao APRENDE invariancia a rotacao. Ele muda de coordenada e a
rotacao vira translacao, que a arquitetura ja tem de graca. E paga o preco:
perde a translacao, e compra ela de volta movendo o olho.

POR QUE ISSO TEM ENDERECO AQUI
--------------------------------
`eficiencia.py` mediu uma rede convolucional saturando em EXATAMENTE 33,3% em
quatro tamanhos de dado — quatro pontos iguais ao decimo. Nao era curva de
aprendizado: convolucao e equivariante a translacao por construcao e NAO PODE
expressar rotacao. Acertava 100% das 4 leis por-celula e 0% das 8 espaciais.

Se a leitura estiver certa, o conserto nao e mais dado nem rede maior. E
trocar a coordenada. Mesma rede, mesmos pesos, mesma quantidade de dado.

O TESTE
---------
As mesmas 10 leis atomicas de `eficiencia.py`, uma de cada vez, com dado
abundante (N grande de proposito: isto e sobre EXPRESSIVIDADE, nao sobre
eficiencia de dado). Duas coordenadas, a mesma convolucional:

    CARTESIANO   conv 3x3 na grade, como antes
    LOG-POLAR    conv 3x3 na grade reamostrada em (angulo, raio), com
                 vizinhanca CIRCULAR no eixo angular — porque o angulo e
                 periodico e o cortex tambem e

TETO DE REAMOSTRAGEM. Ir e voltar de log-polar perde informacao numa grade
6x6. Entao mede-se primeiro a IDENTIDADE no ida-e-volta: se o teto for baixo,
qualquer numero abaixo dele nao diz nada sobre a rede. Sem esse teto o
experimento e vazio.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. O teto de reamostragem tem que ser alto (>= 90%) e o
      braco CARTESIANO tem que reproduzir o achado anterior: ~100% nas leis
      por-celula e ~0% em rot90/transp.

  C1  LOG-POLAR acerta rot90 e CARTESIANO nao
      -> coordenada, e nao aprendizado, decide o que e expressavel. O truque
         do cortex e real e transportavel.

  C2  LOG-POLAR tambem falha em rot90
      -> ou a reamostragem estraga, ou a leitura do mapa esta errada.

MINHA PREVISAO, escrita antes de rodar, e ela NAO e "vai melhorar": espero C1
no rot90 e espero PERDER dl01 e dl10, as translacoes, que o cartesiano acerta.
Log-polar compra rotacao e escala e vende translacao. Espero tambem que espH,
espV e transp continuem falhando nos DOIS, porque reflexao vira reflexao
angular e convolucao nao expressa reflexao em coordenada nenhuma.

Ou seja: aposto que o placar TOTAL do log-polar seja IGUAL OU PIOR que o do
cartesiano. Se eu estiver certo, o achado nao e um ganho — e a TROCA. Cada
sistema de coordenadas torna um conjunto de leis gratuito e outro impossivel,
e nenhum e universal.

E ai a consequencia e a parte interessante, porque ela e um argumento
estrutural e nao um numero: se coordenada define expressividade e nenhuma
coordenada serve para tudo, entao o desenho certo nao e escolher a melhor —
e rodar VARIAS EM PARALELO sobre a mesma cena. Que e, na descricao do
usuario, "modulos diferentes que veem a mesma coisa de jeitos que nao
conversam".
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from novaphys.eficiencia import ATOMOS, LADO, N_CORES, Grid, grade

# COBERTURA. Com 16x5 o mapa nao alcancava as 36 celulas: o ida-e-volta da
# identidade caia a 0% e o braco polar estaria predizendo MENOS informacao que
# o cartesiano — comparacao injusta, pega pelo portao C0. Varri a configuracao
# ate achar uma que cobre as 36 E mantem rot90 == deslocamento angular exato.
# 64x8 = 512 bins para 36 celulas e sobreamostragem pesada, que e exatamente o
# que o cortex faz: fator de magnificacao cortical.
NA = 64          # bins angulares — 90 graus = deslocamento de 16, exato
NR = 8           # bins radiais


# ══════════════════════════════════════════════ o mapa retina -> cortex
def _tabela() -> List[List[Tuple[int, int]]]:
    """
    Para cada bin (angulo, raio), a celula cartesiana correspondente.
    Espacamento LOG no raio, que e o que o mapa retino-cortical faz — muita
    resolucao no centro, pouca na periferia.

    GEOMETRIA, e a primeira versao estava errada. Eu centrava em (LADO-1)/2 =
    2,5 e usava round(). Nos angulos alinhados aos eixos a coordenada caia
    exatamente em 2,5, e `round` do Python usa arredondamento bancario
    (round(2.5)=2, round(3.5)=4). Empate assimetrico quebra a simetria de
    rotacao e o mapa deixava de comutar com o giro de 90 graus — a verificacao
    pegou: rot90 nao virava deslocamento angular.

    Correcao: centro em LADO/2, celula (i,j) tratada como o ponto (i+0,5,
    j+0,5), indice por floor, e os angulos deslocados de MEIO bin
    (ia + 0,5) para que nenhum caia sobre um eixo. O deslocamento de meio bin
    preserva a simetria de 90 graus porque NA e multiplo de 4.
    """
    c = LADO / 2.0
    rmin, rmax = 0.5, 3.6
    tab: List[List[Tuple[int, int]]] = []
    for ia in range(NA):
        th = 2 * math.pi * (ia + 0.5) / NA
        linha = []
        for ir in range(NR):
            rho = rmin * (rmax / rmin) ** (ir / max(NR - 1, 1))
            y = math.floor(c + rho * math.sin(th))
            x = math.floor(c + rho * math.cos(th))
            linha.append((min(max(y, 0), LADO - 1), min(max(x, 0), LADO - 1)))
        tab.append(linha)
    return tab


TAB = _tabela()


def para_polar(g: Grid) -> np.ndarray:
    return np.array([[g[y][x] for (y, x) in linha] for linha in TAB],
                    dtype=np.int64)


def de_polar(p: np.ndarray) -> Grid:
    """Volta por voto majoritario: varios bins podem cair na mesma celula."""
    votos: Dict[Tuple[int, int], List[int]] = {}
    for ia in range(NA):
        for ir in range(NR):
            votos.setdefault(TAB[ia][ir], []).append(int(p[ia][ir]))
    g = [[0] * LADO for _ in range(LADO)]
    for (y, x), vs in votos.items():
        g[y][x] = max(set(vs), key=vs.count)
    return tuple(tuple(l) for l in g)


def teto_reamostragem(rng: random.Random, n: int) -> float:
    """Quanto a IDENTIDADE sobrevive ao ida-e-volta. Sem isto nada vale."""
    ok = 0
    for _ in range(n):
        g = grade(rng)
        if de_polar(para_polar(g)) == g:
            ok += 1
    return ok / n


# ══════════════════════════════════════════════ as duas redes
def cod_cart(g: Grid) -> np.ndarray:
    t = np.zeros((N_CORES, LADO, LADO), dtype=np.float32)
    for i in range(LADO):
        for j in range(LADO):
            t[g[i][j], i, j] = 1.0
    return t


def cod_polar(g: Grid) -> np.ndarray:
    p = para_polar(g)
    t = np.zeros((N_CORES, NA, NR), dtype=np.float32)
    for ia in range(NA):
        for ir in range(NR):
            t[p[ia][ir], ia, ir] = 1.0
    return t


class ConvCircular:
    """Fabrica de rede. No polar o eixo angular usa padding CIRCULAR."""

    @staticmethod
    def montar(circular: bool):
        import torch.nn as nn
        modo = "circular" if circular else "zeros"

        def bloco(ent, sai, k=3):
            return nn.Conv2d(ent, sai, k, padding=1, padding_mode=modo)
        return nn.Sequential(
            bloco(N_CORES, 64), nn.ReLU(),
            bloco(64, 64), nn.ReLU(),
            bloco(64, 64), nn.ReLU(),
            nn.Conv2d(64, N_CORES, 1),
        )


def medir(lei: str, polar: bool, n: int, epocas: int, semente: int) -> float:
    import torch
    import torch.nn as nn
    torch.manual_seed(semente)
    rng = random.Random(semente)
    fn = dict(ATOMOS)[lei]

    def par():
        a = grade(rng)
        return a, fn(a)

    tr = [par() for _ in range(n)]
    te = [par() for _ in range(300)]
    cod = cod_polar if polar else cod_cart
    alvo = (lambda g: para_polar(g)) if polar else \
        (lambda g: np.array(g, dtype=np.int64))

    rede = ConvCircular.montar(polar)
    X = torch.from_numpy(np.stack([cod(a) for a, _ in tr]))
    Y = torch.from_numpy(np.stack([alvo(b) for _, b in tr]))
    opt = torch.optim.Adam(rede.parameters(), lr=3e-3)
    perda = nn.CrossEntropyLoss()
    for _ in range(epocas):
        opt.zero_grad()
        perda(rede(X), Y).backward()
        opt.step()

    rede.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.stack([cod(a) for a, _ in te]))
        p = rede(Xt).argmax(1).numpy()
    ok = sum(1 for k, (_a, b) in enumerate(te) if (p[k] == alvo(b)).all())
    return ok / len(te)


POR_CELULA = {"cic1", "cic2", "tr12", "tr34"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--epocas", type=int, default=800)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 74)
    print("CORTICAL — trocar de coordenada em vez de aprender a invariancia")
    print("=" * 74)
    rng = random.Random(a.semente)
    teto = teto_reamostragem(rng, 400)
    print(f"\nC0 · grade {LADO}x{LADO} -> polar {NA}x{NR}")
    print(f"     teto de reamostragem (identidade ida-e-volta): {teto:.1%}")
    if teto < 0.90:
        print("     C0 FALHOU: o ida-e-volta ja perde demais. Qualquer numero")
        print("     abaixo disso nao diz nada sobre a rede. Abortando.")
        return
    print(f"     N={a.n} de proposito alto: isto e sobre EXPRESSIVIDADE\n")

    print(f"{'lei':8s} {'tipo':10s} {'CARTESIANO':>11s} {'LOG-POLAR':>11s}")
    print("-" * 44)
    placar = {"cart": 0, "pol": 0}
    ganhou, perdeu = [], []
    for nome, _f in ATOMOS:
        c = medir(nome, False, a.n, a.epocas, a.semente)
        p = medir(nome, True, a.n, a.epocas, a.semente)
        tipo = "por-celula" if nome in POR_CELULA else "espacial"
        placar["cart"] += int(c >= 0.95)
        placar["pol"] += int(p >= 0.95)
        if p >= 0.95 > c:
            ganhou.append(nome)
        if c >= 0.95 > p:
            perdeu.append(nome)
        print(f"{nome:8s} {tipo:10s} {c:>10.1%} {p:>11.1%}", flush=True)
    print("-" * 44)
    print(f"{'leis >= 95%':19s} {placar['cart']:>10d} {placar['pol']:>11d}")

    print(f"\nso o LOG-POLAR resolve : {ganhou}")
    print(f"so o CARTESIANO resolve: {perdeu}")

    print("\n" + "=" * 74)
    if "rot90" in ganhou:
        print("C1 · a COORDENADA decide o que e expressavel, nao o")
        print("     aprendizado. A mesma rede, os mesmos pesos, o mesmo dado:")
        print("     rotacao era impossivel no cartesiano e e de graca no")
        print("     log-polar, porque la ela E uma translacao.")
        if perdeu:
            print(f"\n     E o preco apareceu: {perdeu} funcionavam no")
            print("     cartesiano e quebraram no polar. Nenhuma coordenada")
            print("     serve para tudo — o cerebro paga esse preco tambem, e")
            print("     compra a translacao de volta movendo o olho.")
            print("\n     CONSEQUENCIA ESTRUTURAL: se coordenada define")
            print("     expressividade e nenhuma e universal, o desenho certo")
            print("     nao e escolher a melhor — e rodar VARIAS em paralelo")
            print("     sobre a mesma cena.")
    else:
        print("C2 · log-polar tambem falha em rot90. Ou a reamostragem")
        print("     estraga, ou a leitura do mapa esta errada.")


if __name__ == "__main__":
    main()
