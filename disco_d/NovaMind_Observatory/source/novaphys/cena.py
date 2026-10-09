"""
novaphys.cena — quantos objetos cabem numa cena mental?
=========================================================

DE ONDE VEM
-------------
Do usuario, descrevendo o que ele considera mente de nivel humano: "Um
engenheiro, fisico e demais faz isso com milhares de objetos mentais cada um com
seu significado, cola eles na cena mental, atribui um significado a cena, e a
mente comeca a operar em cima desses pre supostos (coisas que ainda nao
existem)."

E mostrando os diagramas do relatorio do DeepSeek: caixa e modulo, seta e fluxo,
aninhamento e hierarquia, destaque e "foi isto que mudou". Isso e composicao de
CENA: objetos com PAPEL, colados juntos, e a cena inteira ganhando significado.

POR QUE ISSO E MEDIVEL AQUI
-----------------------------
HDC foi construido para papel-preenchedor: `bind(papel, valor)` e depois feixe.
E o caso em que ele MEDE POSITIVO neste projeto (ordenar/casar, erro 2,6 d
contra 11,3 d do sorteio). Uma cena e exatamente um feixe de bindings.

Entao a pergunta "a AGI consegue montar cenas mentais?" vira uma pergunta com
numero: **quantos pares papel-preenchedor cabem antes de a recuperacao
degradar?** Se o usuario fala em milhares e a capacidade for dezenas, o buraco
esta medido em ordens de grandeza, nao em opiniao.

A CONTA FEITA ANTES DE RODAR (a licao que eu acabei de pagar caro em
[[escolher-alvo-sem-folga]]: calcular o efeito possivel ANTES)
-----------------------------------------------------------------
Num espaco bipolar de dimensao D, o feixe de n vetores aleatorios guarda cada um
com razao sinal-ruido ~ sqrt(D/n). Para recuperar contra um codebook de V
candidatos e preciso que o alvo vença V-1 distratores, o que exige
sqrt(D/n) crescer com log(V). Com D=10.000 isso da CAPACIDADE DA ORDEM DE
DEZENAS, nao de milhares. Entao eu ja sei que o resultado mostra buraco; a
medida serve para dizer de QUANTAS ordens de grandeza, e como ele escala.

PRE-REGISTRO
--------------
  C0  PRE-CONDICAO. Com 2 pares a recuperacao e >= 0,95. Se nem duas coisas
      cabem, o mecanismo nao e de cena e nada abaixo interessa.
  C1  existe um n* onde a recuperacao cruza 0,90 -> a CAPACIDADE esta medida
  C2  a capacidade cresce ~linear com a dimensao D -> da para comprar
      capacidade com memoria, e o custo por objeto fica conhecido
  C3  a capacidade NAO cresce com D -> comprar memoria nao resolve e o
      mecanismo de cena precisa ser outro

  Nulo: recuperar um preenchedor que NAO esta na cena. Tem de dar ~1/V.

  Segunda pergunta, e e a do usuario: uma cena com ESTRUTURA (papeis que se
  repetem, como "entrada/saida" ligando varios modulos) cabe melhor ou pior que
  uma cena de pares independentes? Diagrama real tem estrutura repetida.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe, sim

SEMENTE = 20260918
N_REP = 12


def monta_cena(e: Espaco, pares: Sequence[Tuple[str, str]]) -> np.ndarray:
    """A cena e o feixe dos bindings papel-preenchedor. Nada mais."""
    return feixe([e.bind(e.atomo(f"papel:{p}"), e.atomo(f"obj:{o}"))
                  for p, o in pares], e.dim)


def consulta_rapida(e: Espaco, cena: np.ndarray, papel: str,
                    M: np.ndarray) -> int:
    """
    Igual a `consulta`, mas o cleanup e UMA multiplicacao de matriz contra o
    codebook empilhado, em vez de 4096 comparacoes uma a uma. So velocidade;
    o resultado e identico.
    """
    sonda = e.bind(e.atomo(f"papel:{papel}"), cena).astype(np.float32)
    return int(np.argmax(M @ sonda))


def consulta(e: Espaco, cena: np.ndarray, papel: str,
             codebook: Sequence[str]) -> str:
    """
    "O que ocupa este papel na cena?"

    Desliga o papel do feixe e faz cleanup contra o codebook. E recall
    associativo — o HDC nunca inventa preenchedor novo, so recupera o que ja
    existe no codebook, que e o limite medido dele como GERADOR (2/400).
    """
    sonda = e.bind(e.atomo(f"papel:{papel}"), cena)
    return max(codebook, key=lambda o: sim(sonda, e.atomo(f"obj:{o}")))


def mede_capacidade(dim: int, v_codebook: int = 4096,
                    ns: Sequence[int] = (2, 8, 32, 100, 300, 800, 2000),
                    n_consulta: int = 24) -> Dict:
    """
    RODADA 2. A rodada 1 parou em 50 objetos e deu 1,000 em TODAS as dimensoes —
    faixa curta demais, e o "a capacidade nao cresce" era artefato de teto. E o
    codebook de 64 tornava o nulo inutil: com 50 objetos na cena, 78% do
    codebook JA estava nela. Agora o codebook tem 4096 e a faixa vai a 2000.
    Consulta-se uma AMOSTRA de `n_consulta` papeis por cena, para caber no tempo.

    E registro que minha conta pre-registrada errou PARA O PESSIMISTA: previ
    capacidade "da ordem de dezenas" e nao achei o limite nem aos 50.
    """
    e = Espaco(dim=dim)
    rng = np.random.default_rng(SEMENTE)
    objs = [f"o{i}" for i in range(v_codebook)]
    papeis = [f"p{i}" for i in range(max(ns) + 2)]
    M = np.stack([e.atomo(f"obj:{o}") for o in objs]).astype(np.float32)
    out = {}
    for n in ns:
        acertos, nulos = [], []
        for _ in range(N_REP):
            esc = rng.choice(v_codebook, size=n, replace=False)
            pares = [(papeis[i], objs[esc[i]]) for i in range(n)]
            cena = monta_cena(e, pares)
            amostra = pares if n <= n_consulta else [
                pares[i] for i in rng.choice(n, size=n_consulta, replace=False)]
            idx = {o: i for i, o in enumerate(objs)}
            ok = sum(consulta_rapida(e, cena, p, M) == idx[o]
                     for p, o in amostra) / len(amostra)
            acertos.append(ok)
            # NULO: perguntar por um papel que NAO esta na cena
            fora = papeis[n]
            na_cena = {idx[o] for _p, o in pares}
            nulos.append(float(consulta_rapida(e, cena, fora, M) in na_cena))
        out[n] = (statistics.mean(acertos), statistics.pstdev(acertos),
                  statistics.mean(nulos))
    return out


def experimento() -> None:
    print("=" * 78)
    print("CENA — quantos objetos com papel cabem numa cena mental?")
    print("=" * 78)
    print("  codebook de 4096 objetos; a cena e o feixe de bind(papel, objeto)")
    print("  consulta: 'o que ocupa o papel p?' por desligamento + cleanup\n")

    dims = (2000, 10000, 40000)
    tabelas = {}
    for d in dims:
        tabelas[d] = mede_capacidade(d)

    print("  objetos   " + "".join(f"D={d:<12}" for d in dims))
    print("  -------   " + "".join("-" * 14 for _ in dims))
    for n in sorted(tabelas[dims[0]]):
        linha = f"  {n:7d}   "
        for d in dims:
            m, sd, _nu = tabelas[d][n]
            linha += f"{m:.3f}+-{sd:.2f}  "
        print(linha)

    print("\n  nulo (papel ausente cai num objeto da cena; acaso = "
          f"{1/4096:.4f}):")
    for d in dims:
        nus = [tabelas[d][n][2] for n in sorted(tabelas[d])]
        print(f"    D={d}: {min(nus):.3f} a {max(nus):.3f}")

    print("\n" + "=" * 78)
    if tabelas[10000][2][0] < 0.95:
        print(f"C0 FALHOU: com 2 pares a recuperacao ja e "
              f"{tabelas[10000][2][0]:.3f}.")
        return
    print(f"C0 . com 2 pares a recuperacao e {tabelas[10000][2][0]:.3f}.")

    caps = {}
    for d in dims:
        cap = 0
        for n in sorted(tabelas[d]):
            if tabelas[d][n][0] >= 0.90:
                cap = n
        caps[d] = cap
    print("")
    for d in dims:
        print(f"  D={d:6d}  ->  capacidade (>=0,90): {caps[d]} objetos")

    d0, d1 = dims[0], dims[-1]
    razao_d = d1 / d0
    razao_c = caps[d1] / max(caps[d0], 1)
    print(f"\n  a dimensao cresceu {razao_d:.0f}x e a capacidade {razao_c:.1f}x")
    if razao_c >= 0.6 * razao_d:
        print("C2 . a capacidade cresce ~LINEAR com a dimensao: da para")
        print("     comprar capacidade com memoria, e o custo por objeto e")
        print("     conhecido.")
    elif razao_c > 1.5:
        print("C2 parcial: cresce, mas SUBLINEAR — cada objeto a mais custa")
        print("     mais memoria que o anterior.")
    else:
        print("C3 . a capacidade NAO cresce com a dimensao. Comprar memoria")
        print("     nao resolve; a cena precisa de outro mecanismo.")

    alvo = 1000
    if caps[d1] > 0:
        # extrapolacao pela razao medida, declarada como extrapolacao
        exp = math.log(razao_c) / math.log(razao_d) if razao_c > 1 else 0.0
        if exp > 0:
            d_nec = d1 * (alvo / caps[d1]) ** (1 / exp)
            print(f"\n  EXTRAPOLACAO (nao medida): para {alvo} objetos numa"
                  f" cena seriam ~{d_nec:.0e} dimensoes,")
            print(f"  ou {d_nec*1/8e9:.1f} GB por cena em int8.")


if __name__ == "__main__":
    experimento()
