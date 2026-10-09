"""
novaphys.critica_arc — a invariante como critica: saber que a resposta esta errada
====================================================================================

DE ONDE VEM
-------------
`invariante_arc`: a operacao "ache o que se conserva", trazida das orbitas, achou
invariantes entrada -> saida em 342/400 tarefas, e 97,4% delas valem no par de
teste (nulo: 16%). Estrutura verdadeira — mas estrutura BARATA ("o que nao muda").
A pergunta que falta: ela SERVE para alguma coisa?

O usuario descreveu o que quer: "sistemas de criterio rigidos onde a propria AGI
deve simular para entender que nao funcionou". Aqui o portao e interno: o nucleo
responde; a invariante do treino JULGA a resposta. Resposta que viola o que a
transformacao conserva nos pares de treino deve estar errada — sem olhar o gabarito.

O DESENHO
-----------
Fase pesada (`--roda`, uma vez, 3 processos): o nucleo (`descricao.montar`) em
todas as 400 tarefas; guarda a resposta para cada entrada de teste e se ela esta
certa. Fase leve (padrao): para cada resposta, a invariante MAIS SIMPLES aceita no
treino (e, a parte, TODAS as aceitas) — a resposta a viola?

PRE-REGISTRO, ANTES DE RODAR
------------------------------
Em "fecha treino" (o nucleo reproduz todos os pares de treino — a resposta em que
ele CONFIA):
  CR0  ALARME FALSO: das respostas CERTAS, a invariante mais simples acusa <= 10%
  CR1  DETECCAO: das respostas ERRADAS, a invariante mais simples acusa >= 25%
Em todas as respostas:
  CR2  PRECISAO: das respostas acusadas, >= 90% estao de fato erradas

PREVISAO: CR0 passa (~3%, porque a invariante vale no teste 97% das vezes). CR1
e o incerto: um modelo que fecha o treino tende a preservar o que o treino
preserva; aposto ~30% (passa raspando). CR2 passa.
Se CR1 passar, o sistema ganha o que nao tinha: saber, sem gabarito, que uma
resposta confiante esta errada — o gatilho para procurar outra.

RESULTADO (nucleo em 2908 s, 40/400 no teste, igual a remedicao):
  fecha treino: 56 respostas, 40 certas, 16 erradas
      invariante mais simples: alarme falso 0/40, detecta 2/16, precisao 2/2
      todas as aceitas:        alarme falso 8/40, detecta 9/16, precisao 9/17
  todas as respostas: 375, 49 certas, 326 erradas
      mais simples: alarme falso 0/49, detecta 67/326, precisao 67/67
CR0 passou (0%). CR2 passou (100%). CR1 FALHOU: 12,5% (previ ~30%, otimista).
Leitura: e o que eu temia — um modelo que FECHA o treino ja preserva o que o treino
preserva, entao a critica barata quase nunca pega o erro confiante. Onde ela vale:
e um detector de erro GRATIS e PERFEITAMENTE preciso para ~1/5 das respostas
erradas em geral (67/326, zero alarme falso). E pegou 8ee62060 — resposta confiante
e errada do nucleo numa tarefa que OUTRO motor resolve: a critica pode arbitrar
entre motores. Usar TODAS as invariantes pega 9/16, mas acusa 20% das certas.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import invariante_arc as IA

ARQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predicoes_aval.json")


def _uma(args):
    from novaphys import descricao as Dd
    nome, tr, te = args
    try:
        fm, pilha, err = Dd.montar(tr)
        if not pilha:
            return nome, None, None, None
        preds, oks = [], []
        for x, y in te:
            p = Dd.aplicar((fm, pilha), x)
            preds.append([list(l) for l in p] if p is not None else None)
            oks.append(bool(p is not None and p == y))
        return nome, err, preds, oks
    except Exception:
        return nome, None, None, None


def roda(nucleos):
    from multiprocessing import Pool
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(_uma, tarefas, chunksize=1)
    with open(ARQ, "w") as f:
        json.dump(res, f)
    ok = sum(1 for _n, e, _p, o in res if o and e == 0 and all(o))
    print(f"  ({time.time()-t0:.0f}s)  nucleo passou no teste: {ok}/400  -> {ARQ}")


def analisa():
    tarefas = {n: (tr, te) for n, tr, te in carregar(PASTA_AVAL)}
    with open(ARQ) as f:
        dados = json.load(f)
    cands = IA.candidatos(list(range(len(IA.NOMES))))
    linhas = []
    for nome, err, preds, oks in dados:
        if preds is None:
            continue
        tr, te = tarefas[nome]
        ac = IA.acha(tr, cands)
        for (x, _y), p, ok in zip(te, preds, oks):
            if p is None:
                continue
            p = tuple(tuple(l) for l in p)
            fx, fp = IA.grandezas(x), IA.grandezas(p)
            top = bool(ac) and IA.Q(fp, ac[0][2]) != IA.Q(fx, ac[0][2])
            alguma = any(IA.Q(fp, c) != IA.Q(fx, c) for _k, _g, c in ac)
            linhas.append((nome, err == 0, ok, top, alguma))

    def bloco(rot, rs):
        certas = [r for r in rs if r[2]]
        erradas = [r for r in rs if not r[2]]
        print(f"\n  {rot}: {len(rs)} respostas, {len(certas)} certas, {len(erradas)} erradas")
        out = {}
        for i, nm in ((3, "mais simples"), (4, "todas")):
            fa = sum(1 for r in certas if r[i])
            de = sum(1 for r in erradas if r[i])
            acus = fa + de
            out[nm] = (fa / max(len(certas), 1), de / max(len(erradas), 1),
                       de / max(acus, 1))
            print(f"    invariante {nm:12s} alarme falso {fa:3d}/{len(certas):<3d}   "
                  f"detecta {de:3d}/{len(erradas):<3d}   precisao {de}/{acus}")
        return out

    print("=" * 76)
    print("CRITICA ARC — a invariante do treino julga a resposta do nucleo")
    print("=" * 76)
    fecha = bloco("FECHA TREINO (resposta confiante)", [r for r in linhas if r[1]])
    todas = bloco("TODAS as respostas", linhas)
    print("")
    print("=" * 76)
    fa, de, _ = fecha["mais simples"]
    _, _, pr = todas["mais simples"]
    print(f"CR0 {'.' if fa <= 0.10 else 'FALHOU .'} alarme falso nas certas: {100*fa:.1f}% (<= 10%)")
    print(f"CR1 {'.' if de >= 0.25 else 'FALHOU .'} detecta erradas confiantes: {100*de:.1f}% (>= 25%)")
    print(f"CR2 {'.' if pr >= 0.90 else 'FALHOU .'} precisao do alarme: {100*pr:.1f}% (>= 90%)")
    print("\n  respostas confiantes ERRADAS que a invariante pegou:")
    for r in linhas:
        if r[1] and not r[2] and r[3]:
            print(f"    {r[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roda", action="store_true")
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    if a.roda:
        roda(a.nucleos)
    else:
        analisa()


if __name__ == "__main__":
    main()
