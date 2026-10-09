"""
novaphys.prion_hdc — HDC como guia de busca, com recompensa e punicao
======================================================================

POR QUE AQUI, E NAO ANTES
---------------------------
O HDC atravessou a sessao inteira sendo tratado como componente do projeto e
nunca foi usado em nada. Eu disse que so o usaria onde ele pudesse ser medido
contra uma alternativa e PERDER. Este e o lugar.

`prion_busca.py` varreu 837 leis de profundidade 2 e achou `x^3` (41,7 d
contra 53 observados). A tendencia e monotona na potencia de x — mediana
15,0 -> 16,1 -> 18,3 -> 22,3 — e o melhor ainda fica 21% curto. Ou seja, a
resposta esta MAIS FUNDO, e profundidade 3 tem dezenas de milhares de leis.

Exaustivo morre. O mesmo gargalo do ARC.

O QUE O HDC OFERECE QUE ENUMERACAO NAO
----------------------------------------
A descoberta e uma regularidade ESTRUTURAL: leis que contem mais x vao melhor.
Enumeracao nao tem nocao de "parecido com" — ela testa `(x*x)` e depois
`(x*y)` sem que o sucesso da primeira diga nada sobre a terceira.

HDC tem: `bind` e `bundle` dao um espaco onde expressoes que compartilham
estrutura ficam proximas. Entao o que foi aprendido em algumas candidatas
generaliza para candidatas nunca avaliadas.

RECOMPENSA E PUNICAO
----------------------
Cada lei avaliada recebe |duracao prevista - 53|. As melhores sao agregadas
num prototipo BOM, as piores num prototipo RUIM. Candidatas nao avaliadas sao
ordenadas por  sim(v, BOM) - sim(v, RUIM).  Isso e reforco no sentido
estreito e honesto: o sinal do mundo reordena a busca.

OS TRES BRACOS, ORCAMENTO CASADO
----------------------------------
Todos avaliam o MESMO numero de leis do MESMO conjunto. So a ORDEM muda.

    exaustivo   ordem canonica da gramatica
    aleatorio   ordem sorteada, semente pre-registrada
    HDC         semeia com N leis, aprende bom/ruim, ordena o resto

Sem orcamento casado, "guiado" venceria por avaliar mais — o erro que
`gargalo.py` pegou e que custou um experimento inteiro.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  H0  PRE-CONDICAO. A similaridade HDC tem que refletir estrutura: `(x*x)`
      mais proxima de `((x*x)*x)` do que de `(y+z)`. Se nao refletir, o
      codificador esta errado e nada abaixo se le.

  H1  a orcamento casado, o HDC acha lei melhor que a ordem aleatoria
      -> generalizacao estrutural funciona como guia

  H2  empate com aleatorio
      -> o HDC nao acrescenta nada aqui, e a representacao nao ajuda a busca

  H3  em profundidade 3, o HDC acha lei MELHOR que `x^3` (41,7 d)
      -> uma estrutura que a enumeracao exaustiva nunca alcancou

SEMENTE PRE-REGISTRADA: 20260815

MINHA PREVISAO: H1 fraco e H3 provavel. O sinal estrutural aqui e forte e
simples (potencia de x), entao qualquer representacao que capture composicao
deveria pega-lo. Se der H2, a conclusao e que num sinal TAO simples o HDC nao
tem vantagem sobre sorteio, o que tambem e informativo.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prion import BASE
from .prion_busca import (DIA_DOSE, DUR_MAX, INICIO_TRAT, OPS, FOLHAS,
                          avaliar, duracao_prevista, trajetoria)
from .prion_fases import (DUR_CTRL, DUR_TRAT, INICIO_CTRL, cal_beta,
                          inicio_clinico)

DIM = 10000
SEMENTE = 20260815
N_POOL = 30000        # leis candidatas de profundidade <= 3
N_SEMEIA = 150        # avaliadas antes de aprender
ORCAMENTO = 600       # avaliacoes TOTAIS por braco


# ══════════════════════════════════════════════ codificador HDC
class Codificador:
    """
    VSA bipolar. `bind` = produto elemento a elemento (involutivo, preserva
    distancia); posicao pelo deslocamento ciclico, para que (a-b) e (b-a)
    nao colidam; `bundle` = soma com sinal.
    """

    def __init__(self, dim: int = DIM, semente: int = SEMENTE):
        rng = np.random.default_rng(semente)
        self.dim = dim
        simbolos = list(FOLHAS) + list(OPS)
        self.atomo = {s: rng.choice([-1, 1], size=dim).astype(np.int8)
                      for s in simbolos}

    def _bind(self, a, b):
        return (a * b).astype(np.int8)

    def _pos(self, v, k: int):
        return np.roll(v, k)

    def codificar(self, expr: str) -> np.ndarray:
        """
        FEIXE de sub-termos marcados por profundidade, nao bind puro.

        A primeira versao usava so `bind`, e H0 a reprovou com similaridade
        ~0,004 entre TUDO — nivel de acaso. O motivo e conceitual: bind
        produz um vetor DISSIMILAR aos operandos; e o proposito dele, criar
        um simbolo novo para a composicao. Uma arvore codificada so por bind
        nao compartilha nada com nenhuma outra.

        Similaridade estrutural precisa de SUPERPOSICAO. Aqui cada simbolo
        entra deslocado pela sua profundidade — o deslocamento mantem
        posicao distinguivel — e o feixe soma tudo. Duas leis com muitos `x`
        compartilham muitos termos e ficam proximas; a multiplicidade
        sobrevive porque o feixe soma antes de tomar o sinal.
        """
        termos: List[np.ndarray] = []
        self._coletar(expr, 0, termos)
        return feixe(termos) if termos else self.atomo["1"]

    def _coletar(self, e: str, prof: int, saida: List[np.ndarray]) -> None:
        e = e.strip()
        if e in self.atomo:
            saida.append(self._pos(self.atomo[e], prof))
            return
        if e.startswith("(") and e.endswith(")"):
            corpo = e[1:-1]
            d = 0
            for i, c in enumerate(corpo):
                if c == "(":
                    d += 1
                elif c == ")":
                    d -= 1
                elif d == 0 and c in OPS:
                    saida.append(self._pos(self.atomo[c], prof))
                    self._coletar(corpo[:i], prof + 1, saida)
                    self._coletar(corpo[i + 1:], prof + 1, saida)
                    return
        saida.append(self._pos(self.atomo["1"], prof))


def sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / len(a))


def feixe(vs: Sequence[np.ndarray]) -> np.ndarray:
    if not len(vs):
        return np.zeros(DIM, dtype=np.int8)
    s = np.sum(np.stack([v.astype(np.int32) for v in vs]), axis=0)
    return np.where(s >= 0, 1, -1).astype(np.int8)


# ══════════════════════════════════════════════ gramatica ate profundidade 3
def sortear_expr(rng: random.Random, prof: int) -> str:
    if prof <= 0 or rng.random() < 0.3:
        return rng.choice(FOLHAS)
    a = sortear_expr(rng, prof - 1)
    b = sortear_expr(rng, prof - 1)
    return f"({a}{rng.choice(OPS)}{b})"


def gerar_pool(n: int, semente: int) -> List[str]:
    rng = random.Random(semente)
    vistas = set()
    while len(vistas) < n:
        e = sortear_expr(rng, 3)
        if e not in FOLHAS:
            vistas.add(e)
    return sorted(vistas)


# ══════════════════════════════════════════════ o experimento
def relatorio() -> None:
    print("=" * 78)
    print("PRION + HDC — representacao composicional como guia de busca")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  H0  similaridade HDC reflete estrutura")
    print("  H1  HDC > aleatorio a orcamento casado")
    print("  H2  empate -> HDC nao acrescenta nada aqui")
    print("  H3  HDC acha lei melhor que x^3 (41,7 d) em profundidade 3")
    print(f"\nsemente pre-registrada {SEMENTE} · dim {DIM} · "
          f"orcamento {ORCAMENTO} avaliacoes por braco")

    cod = Codificador()

    # ── H0
    print("\n" + "-" * 78)
    print("H0 · a similaridade reflete estrutura?")
    pares = [("(x*x)", "((x*x)*x)"), ("(x*x)", "(y+z)"),
             ("(x*y)", "(x*z)"), ("(x*y)", "((y+z)/(y+z))")]
    v = {e: cod.codificar(e) for pr in pares for e in pr}
    for a, b in pares:
        print(f"     sim({a:<10}, {b:<16}) = {sim(v[a], v[b]):+.3f}")
    ok0 = sim(v["(x*x)"], v["((x*x)*x)"]) > sim(v["(x*x)"], v["(y+z)"])
    print(f"     {'aprovada' if ok0 else 'REPROVADA — codificador errado'}")
    if not ok0:
        return

    # ── monta o mundo uma vez
    b = cal_beta(INICIO_CTRL)
    p = dict(BASE); p["beta"] = b
    lo, hi = 0.0, 0.995
    for _ in range(45):
        m = (lo + hi) / 2
        ti = inicio_clinico(p, k=m, inicio=DIA_DOSE, sustentado=False)
        if ti is None or ti > INICIO_TRAT:
            hi = m
        else:
            lo = m
    k = (lo + hi) / 2
    t_c, S_c = trajetoria(p, 0.0, 0.0, True, DUR_CTRL * 3)
    t_t, S_t = trajetoria(p, k, DIA_DOSE, False, DUR_MAX)

    pool = gerar_pool(N_POOL, SEMENTE)
    print(f"\npool: {len(pool)} leis de profundidade <= 3 "
          f"(exaustivo em prof 2 tinha 837)")

    cache: Dict[str, Optional[float]] = {}

    def pontuar(ex: str) -> Optional[float]:
        if ex in cache:
            return cache[ex]
        gc = avaliar(ex, S_c)
        gt = avaliar(ex, S_t) if gc is not None else None
        d = duracao_prevista(gc, t_c, gt, t_t, DUR_CTRL) \
            if (gc is not None and gt is not None) else None
        cache[ex] = d
        return d

    def rodar(ordem: Sequence[str], orc: int) -> Tuple[Optional[float], str, int]:
        melhor, qual, usados = None, "", 0
        for ex in ordem:
            if usados >= orc:
                break
            usados += 1
            d = pontuar(ex)
            if d is None:
                continue
            if melhor is None or abs(d - DUR_TRAT) < abs(melhor - DUR_TRAT):
                melhor, qual = d, ex
        return melhor, qual, usados

    print("\n" + "-" * 78)
    print(f"{'braco':<14}{'melhor duracao':>16}{'lei':>26}{'erro':>10}")
    print("-" * 78)

    res = {}
    # exaustivo: ordem canonica
    d1, e1, _ = rodar(pool, ORCAMENTO)
    res["exaustivo"] = (d1, e1)
    # aleatorio
    rr = random.Random(SEMENTE + 1)
    emb = list(pool); rr.shuffle(emb)
    d2, e2, _ = rodar(emb, ORCAMENTO)
    res["aleatorio"] = (d2, e2)

    # HDC: semeia, aprende bom/ruim, ordena o resto
    rs = random.Random(SEMENTE + 2)
    sementes = rs.sample(pool, N_SEMEIA)
    pontos = [(pontuar(e), e) for e in sementes]
    validos = [(d, e) for d, e in pontos if d is not None]
    validos.sort(key=lambda r: abs(r[0] - DUR_TRAT))
    corte = max(3, len(validos) // 5)
    bom = feixe([cod.codificar(e) for _d, e in validos[:corte]])
    ruim = feixe([cod.codificar(e) for _d, e in validos[-corte:]])
    resto = [e for e in pool if e not in set(sementes)]
    escore = [(sim(cod.codificar(e), bom) - sim(cod.codificar(e), ruim), e)
              for e in resto]
    escore.sort(reverse=True)
    ordem_hdc = [e for _s, e in escore]
    d3, e3, _ = rodar(ordem_hdc, ORCAMENTO - N_SEMEIA)
    # o melhor do HDC inclui o que ele viu na semeadura
    if validos and (d3 is None or
                    abs(validos[0][0] - DUR_TRAT) < abs(d3 - DUR_TRAT)):
        d3, e3 = validos[0]
    res["HDC"] = (d3, e3)

    for nome in ("exaustivo", "aleatorio", "HDC"):
        d, e = res[nome]
        if d is None:
            print(f"{nome:<14}{'nenhuma':>16}")
            continue
        err = 100 * (d - DUR_TRAT) / DUR_TRAT
        print(f"{nome:<14}{d:>13.1f} d{e:>26}{err:>+9.0f}%")

    print("\n" + "=" * 78)
    dh = res["HDC"][0]
    da = res["aleatorio"][0]
    if dh is None or da is None:
        print("comparacao impossivel")
        return
    gh, ga = abs(dh - DUR_TRAT), abs(da - DUR_TRAT)
    if gh < ga * 0.9:
        print(f"H1 · O HDC GUIA MELHOR. Erro {gh:.1f} d contra {ga:.1f} d do")
        print("     sorteio, mesmo numero de avaliacoes, mesmo conjunto.")
    elif ga < gh * 0.9:
        print(f"H1 · INVERTIDO — o sorteio vence ({ga:.1f} contra {gh:.1f}).")
        print("     A ordem aprendida e pior que nenhuma ordem.")
    else:
        print(f"H2 · EMPATE ({gh:.1f} contra {ga:.1f}). Num sinal estrutural")
        print("     tao simples, o HDC nao supera o sorteio.")

    print()
    if dh is not None and abs(dh - DUR_TRAT) < abs(41.7 - DUR_TRAT):
        print(f"H3 · ACHOU ALEM DO EXAUSTIVO. {dh:.1f} d contra 41,7 d do melhor")
        print(f"     de profundidade 2 — lei `{res['HDC'][1]}`, estrutura que a")
        print("     enumeracao exaustiva nunca alcancou.")
    else:
        print(f"H3 · nao superou `x^3` (41,7 d). A profundidade extra nao pagou")
        print("     dentro deste orcamento.")


if __name__ == "__main__":
    relatorio()
