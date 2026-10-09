"""novaphys.ligacao — LIGAR as quatro pecas medidas. Nenhum mecanismo novo.

A CRITICA E A SURPRESA DO ARC. Na fisica a surpresa e "previsao falhou contra o
ruido do instrumento"; no ARC nao ha ruido, mas a critica da o mesmo sinal —
interno, gratuito, e com precisao 100% ela NUNCA acusa inocente.

  nucleo responde -> critica julga -> ACUSOU? -> gasta MAIS orcamento AQUI
                                                 e poe o rejeitado na sala

O ganho esperado nao e achar mecanismo novo: e REALOCAR ORCAMENTO. Ver
PRE_REGISTRO_ligacao.md.
"""
from __future__ import annotations
import argparse, random, time

from novaphys import descricao as D
from novaphys import invariante_arc as IA
from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL


def critica_acusa(tr, entrada, resposta, cands):
    """A invariante do TREINO julga a resposta. Sem gabarito."""
    ac = IA.acha(tr, cands)
    if not ac:
        return False
    fx, fp = IA.grandezas(entrada), IA.grandezas(resposta)
    return IA.Q(fp, ac[0][2]) != IA.Q(fx, ac[0][2])      # a mais simples


def roda(tarefas, teto, extras=None, etq=""):
    """extras: {tarefa: teto_maior}. Devolve (resolvidas, acusadas, custo)."""
    extras = extras or {}
    ok, acus, custo = set(), set(), 0
    cands = IA.candidatos(list(range(len(IA.NOMES))))
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(tarefas):
        t = extras.get(nome, teto)
        custo += t
        try:
            fm, pilha, err = D.montar(tr, codecs=D.CODECS, teto=t)
            if not pilha:
                continue
            preds = [D.aplicar((fm, pilha), x) for x, _y in te]
            if err == 0 and all(p == y for p, (_x, y) in zip(preds, te)):
                ok.add(nome)
                continue
            for (x, _y), p in zip(te, preds):
                if p is None:
                    continue
                try:
                    if critica_acusa(tr, x, tuple(tuple(l) for l in p), cands):
                        acus.add(nome)
                except Exception:
                    pass
        except Exception:
            pass
        if (k + 1) % 25 == 0:
            print(f"    [{etq}] {k+1} tarefas, {len(ok)} ok, {len(acus)} acusadas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return ok, acus, custo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=80)
    ap.add_argument("--teto", type=int, default=400)
    ap.add_argument("--fator", type=int, default=4, help="orcamento extra nas acusadas")
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 74)
    print(f"LIGACAO — a critica realoca o orcamento. {len(tarefas)} tarefas")
    print("=" * 74)

    import json, os
    CACHE = "D:/agi_memoria/ligacao_parcial.json"
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        base, acus, custo0 = set(d["base"]), set(d["acus"]), d["custo0"]
        print(f"  (base recuperada do cache: {len(base)} ok, {len(acus)} acusadas)")
    else:
        base, acus, custo0 = roda(tarefas, a.teto, etq="base")
        os.makedirs("D:/agi_memoria", exist_ok=True)
        json.dump({"base": sorted(base), "acus": sorted(acus), "custo0": custo0},
                  open(CACHE, "w"))
    print(f"\n  BASE: {len(base)}/{len(tarefas)} | a critica acusou {len(acus)} tarefas"
          f" | custo {custo0}")

    # quem a critica acusou ganha orcamento; o resto fica igual
    extra = {n: a.teto * a.fator for n in acus}
    com, _ac2, custo1 = roda(tarefas, a.teto, extra, etq="realocado")
    json.dump({"com": sorted(com), "custo1": custo1},
              open("D:/agi_memoria/ligacao_realoc.json", "w"))
    gasto_extra = custo1 - custo0

    # PLACEBO: o MESMO orcamento extra, em tarefas SORTEADAS
    rng = random.Random(20260917)
    nao = [n for n, _t, _e in tarefas if n not in acus]
    sorteadas = set(rng.sample(nao, min(len(acus), len(nao))))
    plac, _a3, custo2 = roda(tarefas, a.teto,
                             {n: a.teto * a.fator for n in sorteadas}, etq="placebo")

    print("\n" + "=" * 74)
    print(f"  base             {len(base):3d}/{len(tarefas)}   custo {custo0}")
    print(f"  realocado        {len(com):3d}   NOVAS {len(com-base)}, perdeu {len(base-com)}"
          f"   custo {custo1} (+{gasto_extra})")
    print(f"  placebo sorteado {len(plac):3d}   NOVAS {len(plac-base)}, perdeu {len(base-plac)}"
          f"   custo {custo2}")
    n1, n2 = len(com - base), len(plac - base)
    acertou = len(acus - base)
    print()
    print(f"L0 . sanidade: base {len(base)}")
    print(f"L1 {'.' if n1>=2 else 'FALHOU .'} realocar paga: {n1} novas (>= 2)")
    print(f"L2 {'.' if n1-n2>=2 else 'FALHOU .'} bate o placebo: {n1}-{n2} = {n1-n2} (>= +2)")
    print(f"L3 {'.' if acus and acertou/len(acus)>=0.9 else 'FALHOU .'} a critica acerta: "
          f"{acertou}/{len(acus)} das acusadas nao estavam resolvidas "
          f"({100*acertou/max(len(acus),1):.0f}%)")


if __name__ == "__main__":
    main()
