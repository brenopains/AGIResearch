"""novaphys.criterio — bits no treino preve acerto no teste?
Ver PRE_REGISTRO_criterio.md. Deixe-um-de-fora nos pares de TREINO: nao usa a
saida de teste em ponto nenhum."""
from __future__ import annotations
import argparse, random, time
import numpy as np

from novaphys import descricao as D
from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    g = b = 0
    for p in pos:
        for n in neg:
            g += 1
            b += 1.0 if p > n else (0.5 if p == n else 0.0)
    return b / g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 76)
    print(f"CRITERIO — deixe-um-de-fora no TREINO, {len(tarefas)} tarefas, teto {a.teto}")
    print("=" * 76)
    linhas = []
    t0 = time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        # resolve o teste?
        resolve = False
        res0 = None
        try:
            fm, pilha, err = D.montar(tr, codecs=D.CODECS, teto=a.teto)
            if pilha:
                res0 = err
                resolve = err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te)
        except Exception:
            pass
        # deixe-um-de-fora
        acertos = tot = 0
        if len(tr) >= 3:
            for i in range(len(tr)):
                resto = [p for j, p in enumerate(tr) if j != i]
                try:
                    fm2, p2, _e2 = D.montar(resto, codecs=D.CODECS, teto=a.teto)
                    if p2 and D.aplicar((fm2, p2), tr[i][0]) == tr[i][1]:
                        acertos += 1
                except Exception:
                    pass
                tot += 1
        gen = acertos / tot if tot else float("nan")
        linhas.append((tarefa, resolve, gen, res0, len(tr)))
        if (k + 1) % 20 == 0:
            print(f"    {k+1} tarefas, {time.time()-t0:.0f}s", flush=True)

    val = [(r, g, e) for _t, r, g, e, n in linhas if g == g]
    pos = [g for r, g, _e in val if r]
    neg = [g for r, g, _e in val if not r]
    A = auc(pos, neg)
    rng = random.Random(7)
    falso = [rng.random() for _ in val]
    Af = auc([f for f, (r, _g, _e) in zip(falso, val) if r],
             [f for f, (r, _g, _e) in zip(falso, val) if not r])
    dec = [g for r, g, e in val if not r and e == 0]
    print("\n" + "=" * 76)
    print(f"  tarefas com >= 3 pares de treino: {len(val)}")
    print(f"  RESOLVE o teste ({len(pos)}): gen_treino mediana "
          f"{np.median(pos) if pos else float('nan'):.2f}")
    print(f"  NAO resolve   ({len(neg)}): gen_treino mediana "
          f"{np.median(neg) if neg else float('nan'):.2f}")
    print(f"  das que NAO resolvem, com residuo ZERO ({len(dec)}): mediana "
          f"{np.median(dec) if dec else float('nan'):.2f}")
    print()
    print(f"C0 {'.' if pos and np.median(pos) >= 0.6 else 'FALHOU .'} sanidade: "
          f"resolvidas tem gen_treino alta")
    print(f"C1 {'.' if A >= 0.75 else 'FALHOU .'} o sinal separa: AUC {A:.3f} (>= 0,75)")
    print(f"C2 {'.' if dec and np.median(dec) <= 0.34 else 'FALHOU .'} especifico da "
          f"decoracao: mediana {np.median(dec) if dec else float('nan'):.2f} (<= 0,34)")
    print(f"C3 {'.' if abs(Af-0.5) < 0.12 else 'FALHOU .'} nulo sorteado: AUC {Af:.3f} (~0,50)")


if __name__ == "__main__":
    main()
