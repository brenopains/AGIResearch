"""
novaphys.porque_falha — parar de propor e medir POR QUE as 40 nao descrevem
=============================================================================

O DIAGNOSTICO QUE EU VENDI, E QUE O DADO REJEITOU
---------------------------------------------------
Eu afirmei, com confianca e em cima de uma analogia com a fisica: "na grade nao
falta codec, falta PROPRIEDADE para por dentro dele". Construi 85 propriedades
(80 novas, por 7 operadores relacionais), e medi:

    base        12/80
    tudo (85)    9/80   -3
    barato (K)  12/80   +0     corte por MDL do mapa, particoes distintas
    sorteado(K) 12/80   +0     corte ao acaso
    propriedades em pilha vencedora: NENHUMA

A ultima linha e a que mata: em nenhuma das 12 resolvidas o codec de propriedade
APARECEU. Nao e que a selecao seja indiferente — e que esse codec nunca e a
resposta nestas tarefas. O diagnostico estava errado.

O ERRO DE METODO, QUE E PIOR QUE O ERRO
-----------------------------------------
Eu tinha a medida "40 de 60 nao conseguem descrever o treino" e PULEI direto para
uma causa por analogia, em vez de medir a causa. Este modulo faz o que devia ter
sido feito primeiro: caracterizar as falhas ANTES de propor conserto.

O QUE SE MEDE, SO DESCRITIVO, SEM HIPOTESE DE CONSERTO
--------------------------------------------------------
Para cada tarefa que falha com residuo > 0:
  - a saida muda de DIMENSAO em relacao a entrada?
  - o residuo esta no FUNDO ou dentro de objeto?
  - quantos objetos tem a entrada? e a saida?
  - o residuo esta CONCENTRADO numa regiao ou espalhado?
  - a saida usa cor que NAO existe na entrada?
  - a contagem de cores da saida e funcao da contagem da entrada?
  - quanto do residuo cai em celulas que a pilha NAO REIVINDICOU vs
    reivindicou e errou? (falta de cobertura x erro de conteudo)

Nenhuma dessas medidas propoe solucao. Elas so dizem em que grupos as 40 se
dividem — e o grupo maior e onde vale gastar o proximo esforco.

SEM PORTAO DE PASSA/FALHA, DE PROPOSITO
-----------------------------------------
Isto e diagnostico, nao experimento: nao ha hipotese para confirmar ou refutar, e
inventar um portao aqui seria teatro. O que vale e a distribuicao. A disciplina
que se aplica e outra: `agi2-disciplina-de-portao` item 8 — "inexprimivel e
hipotese, nao diagnostico" — e este modulo existe porque eu violei exatamente
isso pela segunda vez.
"""

from __future__ import annotations

import argparse
from collections import Counter

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.descricao import objetos, fundo


def _reivindicadas(fm, pilha, a):
    """Celulas que a pilha escreve para a entrada `a`."""
    try:
        g = D.aplicar((fm, pilha), a)
    except Exception:
        return None
    return g


def caracteriza(tr, te, fm, pilha, err):
    f = {}
    a0, b0 = tr[0]
    ha, wa = dims(a0)
    hb, wb = dims(b0)
    f["dim_muda"] = (ha, wa) != (hb, wb)
    f["dim_muda_sempre"] = all(dims(a) != dims(b) for a, b in tr)
    f["saida_menor"] = hb * wb < ha * wa
    f["saida_maior"] = hb * wb > ha * wa

    bg = fundo(a0)
    f["n_obj_entrada"] = min(len(objetos(a0, bg)), 9)
    f["n_obj_saida"] = min(len(objetos(b0, fundo(b0))), 9)

    cor_a = {c for l in a0 for c in l}
    cor_b = {c for l in b0 for c in l}
    f["cor_nova_na_saida"] = bool(cor_b - cor_a)
    f["menos_cores_na_saida"] = len(cor_b) < len(cor_a)

    # falta de COBERTURA (celula que ninguem escreveu) x ERRO de conteudo
    faltou = escreveu_errado = 0
    no_fundo = no_objeto = 0
    for a, b in tr:
        g = _reivindicadas(fm, pilha, a)
        h, w = dims(b)
        bga = fundo(a)
        for i in range(h):
            for j in range(w):
                if g is None or dims(g) != (h, w):
                    faltou += 1
                    continue
                if g[i][j] != b[i][j]:
                    escreveu_errado += 1
                    if i < dims(a)[0] and j < dims(a)[1] and a[i][j] == bga:
                        no_fundo += 1
                    else:
                        no_objeto += 1
    tot = max(faltou + escreveu_errado, 1)
    f["residuo_e_dimensao"] = faltou > escreveu_errado
    f["residuo_no_fundo"] = no_fundo > no_objeto
    f["fracao_residuo"] = round((faltou + escreveu_errado)
                                / max(sum(dims(b)[0] * dims(b)[1]
                                          for _a, b in tr), 1), 2)
    f["pilha_vazia"] = not pilha
    f["tam_pilha"] = min(len(pilha), 6)
    f["codecs"] = tuple(it[0].nome for it in pilha if not D._marca(it))[:4]
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"POR QUE FALHA — {len(tarefas)} tarefas, teto {a.teto} "
          f"(diagnostico, sem portao)")
    print("=" * 78)

    grupos = Counter()
    campos = Counter()
    n_falha = 0
    codecs_usados = Counter()
    exemplos = {}
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, teto=a.teto)
        except Exception:
            grupos["explodiu"] += 1
            continue
        if pilha and err == 0:
            if all(D.aplicar((fm, pilha), x) == y for x, y in te):
                grupos["RESOLVIDA"] += 1
            else:
                grupos["decorou (residuo 0, teste errado)"] += 1
            continue
        n_falha += 1
        f = caracteriza(tr, te, fm, pilha, err)
        codecs_usados[f["codecs"]] += 1
        for nome, v in f.items():
            if isinstance(v, bool) and v:
                campos[nome] += 1
        # o grupo grosso: o que explica melhor a falha
        if f["pilha_vazia"]:
            g = "A. nao montou pilha nenhuma"
        elif f["residuo_e_dimensao"]:
            g = "B. nem chegou a produzir a grade do tamanho certo"
        elif f["dim_muda_sempre"]:
            g = "C. dimensao muda e a pilha erra o conteudo"
        elif f["residuo_no_fundo"]:
            g = "D. erra no FUNDO (o que devia aparecer do nada)"
        else:
            g = "E. erra DENTRO de objeto (o que devia mudar)"
        grupos[g] += 1
        exemplos.setdefault(g, []).append(tarefa)
        if (k + 1) % 25 == 0:
            print(f"    {k+1} tarefas...", flush=True)

    print("")
    print("=" * 78)
    print("  COMO AS TAREFAS SE DIVIDEM")
    for g, n in grupos.most_common():
        print(f"    {g:46s} {n:3d}")
    print(f"\n  entre as {n_falha} que falharam com residuo > 0, "
          f"quantas tem cada tracO:")
    for nome, n in campos.most_common():
        print(f"    {nome:30s} {n:3d}/{n_falha}  ({100*n/max(n_falha,1):.0f}%)")
    print("\n  pilhas mais comuns nas falhas:")
    for cs, n in codecs_usados.most_common(6):
        print(f"    {n:3d}x  {' -> '.join(cs) if cs else '(vazia)'}")
    print("\n  exemplos por grupo:")
    for g, ts in sorted(exemplos.items()):
        print(f"    {g[:3]} {ts[:5]}")


if __name__ == "__main__":
    main()
