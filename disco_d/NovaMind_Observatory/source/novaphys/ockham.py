"""
novaphys.ockham — selecionar sem usar ajuste ao treino como criterio
======================================================================

O PROBLEMA MEDIDO
-------------------
`segmenta.py` deu oito lentes ao motor e mediu o preco: 127 de 400 tarefas
acham uma regra que reproduz TODO o treino, e apenas 6 sobrevivem ao teste.
Precisao de 4,7%. Sobreajuste de 21x.

Oito lentes x dez propriedades = 80 familias de hipotese por tarefa, contra 2
a 4 pares de treino. Com essa liberdade, ajustar o treino deixa de ser
evidencia.

E ha os dois extremos medidos no MESMO projeto, com o mesmo verificador:

    deducao exata     1 resposta determinada   precisao 100%
    multi-lente      80 familias               precisao   4,7%

A precisao e funcao direta de quanta liberdade a busca teve. Nao e sobre o
motor; e sobre o tamanho do espaco em relacao ao dado.

O QUE NAO PODE SER FEITO
--------------------------
Filtrar as lentes pelo desempenho no TESTE. Isso e vazamento: se o teste
seleciona, ele deixa de medir. Foi o erro que `propositor.py` cometeu e que
custou uma conclusao inteira. O teste continua intocado ate o fim.

O QUE ESTE ARQUIVO FAZ
------------------------
Dois criterios EXTERNOS ao ajuste, nesta ordem:

  1. VALIDACAO CRUZADA nos proprios pares de treino (deixe-um-de-fora).
     Para cada par i, deduz-se a regra usando TODOS OS OUTROS e testa-se em i.
     Uma regra que so existe porque viu o par i nao sobrevive a isso.

     E a resposta estatistica certa para "ajustar nao e evidencia", e nao toca
     no teste. Exige >= 3 pares de treino, senao deixar um de fora nao sobra
     material para deduzir.

  2. NAVALHA DE OCKHAM entre as sobreviventes: prefere-se a lente mais simples
     e a propriedade mais simples. A ordem de complexidade e declarada e fixa,
     escrita antes de rodar:

       lentes    cc4cor < cc8cor < cc4any < cc8any < porcor < linhas <
                 colunas < paineis
       props     global < cor < tam < e_maior < rank_tam < altura < largura <
                 n_mesmo_tam < n_cores < n_buracos

     `global` primeiro porque "todos os objetos sofrem o mesmo" nao usa
     variavel nenhuma; `n_buracos` por ultimo porque e a mais especifica.

O QUE SE MEDE
---------------
    ajustou o TREINO        quantas acham alguma regra (era 127)
    sobreviveu a VALIDACAO  quantas resistem ao deixe-um-de-fora
    passou no TESTE         quantas acertam a grade nunca vista
    PRECISAO = teste / sobreviveu   contra 4,7% de teste/ajustou

A precisao e o numero que importa. Mais tarefas com precisao baixa nao e
progresso — e a mesma sorte com outra roupa.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  O0  PRE-CONDICAO. As 6 de `segmenta.py` precisam sobreviver, ou pelo menos
      as que tenham >= 3 pares de treino. Tarefa com 2 pares nao pode passar
      pela validacao cruzada por construcao, e isso e uma PERDA CONHECIDA, nao
      um bug — tem que ser contada e reportada separada.

  O1  precisao >= 25%  E  >= 5 tarefas
      -> validacao cruzada e o selecionador certo: corta o falso sem cortar o
         verdadeiro
  O2  precisao >= 25%  E  < 5 tarefas
      -> corta o falso E o verdadeiro junto; o criterio e severo demais para
         2-4 exemplos
  O3  precisao < 25%
      -> nem validacao cruzada separa; o sobreajuste nao vem de ver o par
         especifico, vem do tamanho do espaco

MINHA CALIBRAGEM, com dez pontos: dez previsoes sobre mecanismos hoje, DUAS
certas. As oito erradas foram todas otimistas.

PREVISAO: O2. Espero a precisao subir bastante — de 4,7% para algo entre 20% e
40% — e espero PERDER tarefas, ficando entre 3 e 5. Razao concreta: o ARC da 2
a 4 pares de treino, e deixar um de fora com 3 pares significa deduzir com 2.
Muitas regras verdadeiras precisam dos 3 para ficarem determinadas.

Se der O2, a leitura e util e desconfortavel: o criterio certo existe e o ARC
nao da dado suficiente para aplica-lo. E ai o gargalo nao e o metodo de
selecao — e o numero de exemplos, que e propriedade do benchmark e nao nossa.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.segmenta import LENTES, aplicar, deduzir

# Ordem de complexidade, declarada ANTES de rodar. Ockham precisa de uma
# ordem fixa, senao "mais simples" vira o que der o melhor numero.
ORDEM_LENTE = {n: i for i, (n, _f) in enumerate(LENTES)}
ORDEM_PROP = {p: i for i, p in enumerate(
    ("global", "cor", "tam", "e_maior", "rank_tam", "altura", "largura",
     "n_mesmo_tam", "n_cores", "n_buracos"))}


def valida_cruzada(treino, lente) -> bool:
    """
    Deixe-um-de-fora nos proprios pares de treino. Uma regra que so existe
    porque viu o par i nao sobrevive a ser deduzida sem ele.
    """
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir(resto, lente)
        if r is None:
            return False
        try:
            if aplicar(r, treino[i][0], lente) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    cands: List[Tuple[int, int, str, object]] = []
    ajustou = False
    for ln, lente in LENTES:
        try:
            r = deduzir(tr, lente)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in tr):
                continue
            ajustou = True
            if not valida_cruzada(tr, lente):
                continue
            cands.append((ORDEM_LENTE[ln], ORDEM_PROP.get(r[0], 99), ln,
                          (r, lente)))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None, len(tr)
    cands.sort(key=lambda z: (z[0], z[1]))       # Ockham: a mais simples
    _a, _b, ln, (r, lente) = cands[0]
    try:
        ok = all(aplicar(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    return nome, ajustou, True, (ln if ok else None), len(tr)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("OCKHAM — selecionar sem usar ajuste ao treino como criterio")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _v, _o, _k in res if aj]
    sobrev = [n for n, _aj, v, _o, _k in res if v]
    ok = {n: o for n, _aj, _v, o, _k in res if o}
    curtos = [n for n, aj, _v, _o, k in res if aj and k < 3]

    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400   (era 127)")
    print(f"  sobreviveu a VALIDACAO : {len(sobrev):3d}/400")
    print(f"  passou no TESTE        : {len(ok):3d}/400   "
          f"({time.time()-t0:.0f}s)")
    p_antes = 6 / max(len(ajust), 1)
    p_dep = len(ok) / max(len(sobrev), 1)
    print(f"\n  precisao ANTES (teste/ajustou)     : {p_antes:.1%}")
    print(f"  precisao DEPOIS (teste/sobreviveu) : {p_dep:.1%}")
    print(f"\n  perda conhecida: {len(curtos)} tarefas com < 3 pares de")
    print("  treino nao podem passar pela validacao por construcao")
    antes6 = {"00dbd492", "37d3e8b2", "72a961c9", "ae58858e", "b1fc8b8e",
              "f45f5ca7"}
    print(f"\nO0 · das 6 de `segmenta`, mantidas: {sorted(antes6 & set(ok))}")
    print(f"     perdidas: {sorted(antes6 - set(ok))}")
    for k, v in Counter(ok.values()).most_common():
        print(f"    {v:2d}  lente {k}")
    print(f"\n  tarefas: {sorted(ok)}")

    print("\n" + "=" * 78)
    if p_dep >= 0.25 and len(ok) >= 5:
        print("O1 · validacao cruzada e o selecionador certo: corta o falso")
        print("     sem cortar o verdadeiro.")
    elif p_dep >= 0.25:
        print("O2 · corta o falso E o verdadeiro junto. O criterio e severo")
        print("     demais para 2 a 4 exemplos — e o gargalo deixa de ser o")
        print("     metodo de selecao e passa a ser o NUMERO DE EXEMPLOS, que")
        print("     e propriedade do benchmark e nao nossa.")
    else:
        print("O3 · nem validacao cruzada separa. O sobreajuste nao vem de")
        print("     ver o par especifico — vem do TAMANHO DO ESPACO.")


if __name__ == "__main__":
    main()
