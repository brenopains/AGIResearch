"""
novaphys.dois_eixos — material e transformacao sao separaveis no indice?
=========================================================================

DE ONDE VEM
-------------
Do criterio do usuario para o que conta como mente artificial: imaginar vidro
estilhacando, os cacos virando metal, depois uma nuvem de algodao, e entao a
NUVEM estilhacando. A ultima parte e a que morde: aplicar uma transformacao
aprendida num material a um material que nunca foi visto sofrendo aquilo.

Isso so e FORMULAVEL se material e transformacao forem eixos SEPARAVEIS na
loja. Meco a estrutura do indice ANTES de construir gerador, para nao confundir
"gerador fraco" com "pedido sem endereco" — dois fracassos que se parecem por
fora e pedem consertos opostos.

  MATERIAL       = qual grupo de canais (0-2 = A, 3-5 = B) de
                   `substrato.gerar_fluxo`. Superficie, e ele inverte SINAL:
                   canal 0 e `1,7*s`, canal 3 e `-2,4*s + 3,0`.
  TRANSFORMACAO  = 'regime' (alvo depende de cruzar um limiar) x
                   'deriva' (alvo depende de estar SUBINDO)

RODADA 1 (registrada em [[indice-ve-superficie]]) — `de_par` REPROVOU
----------------------------------------------------------------------
    de_par   mesma TRANSF +0,0988   mesmo MATERIAL +0,5188   dif -0,4201
    nulo                                                     dif -0,0942
Agrupava por MATERIAL, e mais forte no real que no nulo. Causa rastreada:
`de_par` amarra alvo a NIVEL quantizado, e com o sinal invertido a MESMA
transformacao gera associacoes ANTICORRELACIONADAS entre materiais.

E dois defeitos meus, que esta rodada corrige:
  (a) o nulo da 1a versao NAO EXISTIA: `transfs=("ruido","ruido")` colidia na
      mesma chave do dicionario e eu comparava uma lista consigo mesma (+1,0).
  (b) o portao J0 exigia que o nulo nao separasse em `de_fluxo` — impossivel,
      porque `gerar_fluxo` produz o fluxo IDENTICO para toda estrutura e so o
      ALVO muda. `de_fluxo` ignora o alvo, logo real e nulo sao o mesmo dado
      (-0,1689 x -0,1677 confirmou). A linha `de_fluxo` fica no relatorio como
      REFERENCIA e nao entra em portao nenhum.

RODADA 2 — PRE-REGISTRO, com `de_par_inv`
-------------------------------------------
`de_par_inv` (em `hdc_indice.py`) adiciona tres invariancias: escala (z-score),
SINAL (orienta cada canal pelo sinal da correlacao dele com o alvo) e PAPEL
(amarra ao POSTO do canal por |correlacao|, nao ao INDICE).

  K0  PRE-CONDICAO. No nulo (alvo sorteado) `de_par_inv` nao separa:
      |diferenca| < 0,05. Com alvo aleatorio as correlacoes sao ~0, a
      orientacao e aleatoria e os postos sao arbitrarios — se ainda assim
      separar, a separacao vem do DUTO.
  K1  real: mesma TRANSF - mesmo MATERIAL > +0,02
      -> EXISTE eixo de transformacao atravessando a superficie; a nuvem
         estilhacando passa a ser enderecavel
  K2  ainda negativa, mas |dif| bem menor que os -0,4201 da rodada 1
      -> a orientacao era parte da causa, e sobra superficie a tirar
  K3  praticamente igual a rodada 1 -> a inversao de sinal NAO era a causa e
      o meu diagnostico estava errado

PREVISAO: K1. O mecanismo da rodada 1 foi identificado nos coeficientes do
gerador, nao inferido de fora, e as tres invariancias atacam exatamente ele.
Ressalva que mantenho: o canal 2 e `0,9*s*s` (quadratico) e o canal 5 e ruido
puro — os materiais diferem ALEM do sinal, e isso nenhuma das tres conserta.
0,6 em K1. Placar do dia: 6 previsoes, 5 erradas (a unica certa foi a
pessimista da rodada 1).
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Tuple

from novaphys.hdc_indice import Espaco, de_fluxo, de_par, de_par_inv, sim
from novaphys.substrato import gerar_fluxo

N_EP = 24
SEMENTE = 20260917


def episodio(rng: random.Random, material: str, transf: str):
    """Mesmo gerador; o MATERIAL e qual grupo de canais se enxerga."""
    fluxo, alvo = gerar_fluxo(rng, T=200, K=6, estrutura=transf)
    col = (0, 1, 2) if material == "A" else (3, 4, 5)
    return [[linha[c] for c in col] for linha in fluxo], alvo


def _celulas(e: Espaco, rng, transfs: Tuple[str, str]) -> Dict[str, List]:
    """
    Codifica N episodios de cada celula (material x transformacao).

    A chave leva a POSICAO e nao o nome da estrutura: no nulo as duas celulas
    tem a mesma estrutura e colidiriam, que foi o bug da rodada 1.
    """
    fora: Dict[str, List] = {}
    for mat in ("A", "B"):
        for j, tr in enumerate(transfs):
            vinv, vpar, vflu = [], [], []
            for _ in range(N_EP):
                fl, al = episodio(rng, mat, tr)
                vinv.append(de_par_inv(e, fl, al))
                vpar.append(de_par(e, fl, al))
                vflu.append(de_fluxo(e, fl))
            fora[f"{mat}|{j}"] = [vinv, vpar, vflu]
    return fora


def _cruz(va: List, vb: List) -> float:
    return statistics.mean(sim(a, b) for a, b in zip(va, vb))


def mede(rotulo: str, transfs: Tuple[str, str], semente: int) -> Dict:
    e = Espaco()
    rng = random.Random(semente)
    c = _celulas(e, rng, transfs)

    print(f"\n  {rotulo}")
    print("    codificacao    mesma TRANSF    mesmo MATERIAL   diferenca")
    print("    ------------   -------------   --------------   ---------")
    saida = {}
    for i, nome in ((0, "de_par_inv"), (1, "de_par"), (2, "de_fluxo*")):
        mT = _cruz(c["A|0"][i], c["B|0"][i])
        mM = _cruz(c["A|0"][i], c["A|1"][i])
        saida[nome.rstrip("*")] = mT - mM
        print(f"    {nome:12s}   {mT:+.4f}         {mM:+.4f}         "
              f"{mT - mM:+.4f}")
    return saida


def experimento() -> None:
    print("=" * 78)
    print("DOIS EIXOS (rodada 2) — o encoder invariante separa transformacao?")
    print("=" * 78)
    print(f"  {N_EP} episodios por celula; material = canais 0-2 (A) x 3-5 (B)")
    print("  * de_fluxo e REFERENCIA: ignora o alvo, logo real e nulo sao o")
    print("    mesmo dado por construcao. Nao entra em portao.")

    real = mede("REAL   regime x deriva", ("regime", "deriva"), SEMENTE)
    nulo = mede("NULO   alvo sorteado (duas instanciacoes independentes)",
                ("ruido", "ruido"), SEMENTE + 1)

    print("\n" + "=" * 78)
    d_real, d_nulo = real["de_par_inv"], nulo["de_par_inv"]
    print(f"  de_par_inv:  real {d_real:+.4f}   nulo {d_nulo:+.4f}")
    print(f"  de_par    :  real {real['de_par']:+.4f}   "
          f"nulo {nulo['de_par']:+.4f}   (rodada 1 dera -0,4201)")
    print("")

    if abs(d_nulo) >= 0.05:
        print(f"K0 FALHOU: o nulo ja separa ({d_nulo:+.4f}) com alvo")
        print("   sorteado. A separacao vem do DUTO, nao da estrutura.")
        return
    print(f"K0 . o nulo nao separa ({d_nulo:+.4f}).")

    if d_real > 0.02:
        print(f"K1 . EXISTE EIXO DE TRANSFORMACAO ({d_real:+.4f}).")
        print("     O indice aproxima MESMA transformacao atraves de materiais")
        print("     que invertem o sinal do latente. 'Esta forma, naquele")
        print("     material' passa a ser enderecavel — e a pergunta do")
        print("     gerador passa a valer a pena.")
    elif d_real > -0.15:
        print(f"K2 . parcial ({d_real:+.4f} contra -0,4201 da rodada 1).")
        print("     A inversao de sinal era parte da causa, e sobra superficie.")
        print("     Suspeito nomeado: canal 2 e 0,9*s*s (quadratico) e canal 5")
        print("     e ruido puro — os materiais diferem ALEM do sinal.")
    else:
        print(f"K3 . sem mudanca ({d_real:+.4f}). A inversao de sinal NAO era")
        print("     a causa e o meu diagnostico da rodada 1 estava errado.")


if __name__ == "__main__":
    experimento()
