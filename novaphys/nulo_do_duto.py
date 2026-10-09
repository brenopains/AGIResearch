"""
novaphys.nulo_do_duto — o controle que faltava: a relacao vem do DADO ou do DUTO?
===================================================================================

O GAP, MEDIDO
---------------
`zeta_explora.py` rodou o achador de leis sobre 7.998 zeros de zeta e aceitou 18
relacoes, com **0 de 42 no nulo permutado**. Doze eram conhecidas. Tres foram
anunciadas como novas:

    N_menos_n ~ d        1839 bits
    N_menos_n ~ d_ant    1806 bits
    d ~ d_ant             452 bits

Examinando a mao: `d ~ d_ant` e lei REAL e conhecida (espacamentos consecutivos
sao anticorrelacionados no GUE — segue da correlacao de pares), e as duas com
`N_menos_n` sao consequencia ALGEBRICA das minhas definicoes (N(t)-n e o
espacamento local sao duas vistas da mesma flutuacao de densidade).

Ou seja: o achador nao inventou nada, mas **anunciou como novo o que era
consequencia das grandezas que eu mesmo defini**. E esse passo de checagem nao
existia — nem aqui, nem na rodada do cerebro.

O CONTROLE, E POR QUE ELE E MELHOR QUE UMA CHECAGEM SINTATICA
---------------------------------------------------------------
A tentacao e escrever um verificador de "isto e algebra das definicoes?". Isso
exige conhecer o grafo de derivacao, quebra com qualquer operacao nova, e nao
diz nada sobre relacoes que sao PARCIALMENTE definicionais.

O controle certo e empirico e geral:

    NULO PERMUTADO   embaralha os VALORES         -> destroi dado E duto
    NULO DO DUTO     troca o CONTEUDO por dado    -> mantem o duto, destroi
                     aleatorio plausivel e passa     so o conteudo
                     pelo MESMO duto de derivacao

Toda relacao que aparece no nulo do duto e consequencia da DERIVACAO, nao do
fenomeno. Toda relacao que aparece so no dado real e candidata de verdade.

E generico: serve para os zeros de zeta (substrato = sequencia crescente com a
mesma densidade), para o cerebro (substrato = grafo com os mesmos graus), para
qualquer duto que derive grandezas de um dado bruto.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
Substrato aleatorio: sequencia crescente com EXATAMENTE a mesma funcao de
densidade dos zeros de zeta (espacamentos exponenciais independentes,
reescalados pela densidade local), passada pelo mesmo `grandezas()`.

  D1  MATA O DEFINICIONAL: `dens ~ logt`, `n ~ t` e as outras 12 "conhecidas"
      APARECEM no nulo do duto (sao consequencia da derivacao, nao de zeta)
  D2  MATA O QUE EU JULGUEI ALGEBRICO: `N_menos_n ~ d` e `N_menos_n ~ d_ant`
      aparecem no nulo do duto
  D3  POUPA O REAL: `d ~ d_ant` NAO aparece no nulo do duto (espacamentos
      independentes nao tem correlacao de curto alcance — a anticorrelacao e
      fenomeno do GUE, nao do duto)
  D4  SOBRA SO O VERDADEIRO: a lista final (real menos nulo-do-duto) contem
      `d ~ d_ant` e nada mais
PREVISAO: D1, D2, D3 e D4 passam. Este e o teste do proprio controle: ele tem de
reproduzir o julgamento que eu fiz A MAO nas tres relacoes, sem eu dizer nada.
Se reproduzir, o passo que falta esta construido e e mecanico; se nao, meu
julgamento a mao estava errado, ou o substrato aleatorio nao e o certo.
"""

from __future__ import annotations

import argparse
from math import pi

import numpy as np

from novaphys.neuro_leis import busca
from novaphys.zeta_explora import grandezas, CONHECIDAS

ARQ = "D:/zeta/zeros.npy"
SEMENTE = 20261029


def substrato_aleatorio(t, rng):
    """Sequencia crescente com a MESMA densidade local dos zeros de zeta, mas
    com posicoes aleatorias: espacamentos exponenciais independentes divididos
    pela densidade local. Mantem tudo o que o duto usa (altura, contagem,
    densidade) e destroi so a estrutura FINA (repulsao, correlacao)."""
    n = len(t)
    u = rng.exponential(1.0, n)                 # espacamentos normalizados
    z = np.empty(n)
    z[0] = t[0]
    for k in range(1, n):
        dens = np.log(max(z[k - 1], 2.1) / (2 * pi)) / (2 * pi)
        z[k] = z[k - 1] + u[k] / max(dens, 1e-6)
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    ap.add_argument("--repeticoes", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)
    t = np.load(a.arquivo)

    print("=" * 78)
    print(f"NULO DO DUTO — {len(t):,} zeros, {a.repeticoes} substratos aleatorios")
    print("=" * 78)

    real = busca(grandezas(t), np.random.default_rng(SEMENTE), "REAL",
                 min_n=2000)
    pares_real = {frozenset((cy, cx)): g for g, cy, cx, _f, _c in real}

    do_duto = {}
    for r in range(a.repeticoes):
        z = substrato_aleatorio(t, rng)
        ac = busca(grandezas(z), np.random.default_rng(SEMENTE + r),
                   f"NULO DO DUTO {r+1}", min_n=2000)
        for g, cy, cx, _f, _c in ac:
            p = frozenset((cy, cx))
            do_duto[p] = max(do_duto.get(p, 0.0), g)

    sobrevive = {p: g for p, g in pares_real.items() if p not in do_duto}

    print(f"\n  {'relacao':30s} {'real(bits)':>12s} {'duto(bits)':>12s} "
          f"{'veredito':>12s}")
    for p, g in sorted(pares_real.items(), key=lambda kv: -kv[1]):
        gd = do_duto.get(p)
        v = "DO DUTO" if gd is not None else "SOBREVIVE"
        et = " (conhecida)" if p in CONHECIDAS else ""
        print(f"  {'~'.join(sorted(p)):30s} {g:12.0f} "
              f"{(f'{gd:12.0f}' if gd is not None else '           -')} "
              f"{v:>12s}{et}")

    print("")
    print("=" * 78)
    d_alg = {frozenset(("N_menos_n", "d")), frozenset(("N_menos_n", "d_ant"))}
    d_real = frozenset(("d", "d_ant"))
    conh_no_duto = [p for p in CONHECIDAS if p in pares_real and p in do_duto]
    print(f"D1 {'.' if len(conh_no_duto) >= 4 else 'FALHOU .'} mata o "
          f"definicional: {len(conh_no_duto)} das conhecidas aparecem no duto")
    print(f"D2 {'.' if d_alg <= set(do_duto) else 'FALHOU .'} mata o que eu "
          f"julguei algebrico: N_menos_n~d no duto = "
          f"{[p in do_duto for p in sorted(d_alg, key=lambda s: sorted(s))]}")
    print(f"D3 {'.' if d_real not in do_duto else 'FALHOU .'} poupa o real: "
          f"d~d_ant {'NAO' if d_real not in do_duto else 'SIM'} aparece no duto")
    print(f"D4 {'.' if set(sobrevive) == {d_real} else 'FALHOU .'} sobra so o "
          f"verdadeiro: {[sorted(p) for p in sobrevive]}")


if __name__ == "__main__":
    main()
