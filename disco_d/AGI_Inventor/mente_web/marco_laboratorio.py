"""
mente_web.marco_laboratorio — o laboratorio de uma vida, conferido contra a verdade da arena
=============================================================================================
Uso:  python marco_laboratorio.py vidas/viva
So LE (laboratorio.pkl, arena.json, ontogenia.jsonl); nunca escreve na mente.

A mente nunca soube o que a arena e. Este instrumento sabe (a camera sai da semente da
arena) e pergunta, com estados percebidos de episodios NOVOS de cada universo:
  - que conceitos ela inventou, e quais deles SAO s^2 ou h (|corr| >= 0,99);
  - que conceito a teoria que explica cada universo usa hoje;
  - quantas vezes ela foi a arena, e o que o laboratorio fez com os videos da internet;
  - (transplante 11) quantos experimentos ela montou, e se o FILME que ela imagina com a
    teoria atual bate com o filme real da arena melhor do que imaginar tudo reto.
Criterios dos transplantes 10 e 11: PRE_REGISTRO_vida1.md.
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as AR  # noqa: E402
import laboratorio as LB  # noqa: E402


def main():
    pasta = sys.argv[1]
    lab = pickle.load(open(os.path.join(pasta, "laboratorio.pkl"), "rb"))
    st = json.load(open(os.path.join(pasta, "arena.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(pasta, "arena.json")) else None
    print(f"laboratorio: {lab.contas}")
    print(f"teorias vivas: {len(lab.teorias)} · episodios guardados: {len(lab.eps)}")
    fontes = collections.Counter(e["fonte"] for e in lab.eps.values())
    print(f"episodios por fonte: {dict(fontes)}")
    acoes = collections.Counter()
    arq = os.path.join(pasta, "ontogenia.jsonl")
    if os.path.exists(arq):
        for linha in open(arq, encoding="utf-8"):
            if '"ev": "acao"' in linha:
                try:
                    acoes[json.loads(linha)["estrategia"]] += 1
                except Exception:
                    pass
    tot = sum(acoes.values())
    if tot:
        print(f"acoes: {tot} · porta:arena {acoes['porta:arena']} ({100 * acoes['porta:arena'] / tot:.1f}%)")
    if st is None:
        print("sem arena.json: ela nunca foi a arena")
        return
    cam = AR.camera(np.random.default_rng(st["semente"]))
    Zs = []
    for i, m in enumerate(AR.UNIVERSOS):
        for r in range(4):
            ep = AR.episodio(m, np.random.default_rng([st["semente"], 77000 + 10 * i + r]), cam)
            for tr in LB.percebe(ep.quadros, ep.tempos)["trilhas"]:
                Z, _A, ok = LB.estados(tr)
                Zs.append(Z[ok])
    Z = np.concatenate(Zs)
    v = AR.verdadeiras_px(cam, np.concatenate([Z[:, :2] * 64 + 64, Z[:, 2:] * 64], 1))
    print(f"\narena: {st['visitas']} visitas (universo atual: {AR.universo_da_visita(max(st['visitas'] - 1, 0))})")
    print("conceitos inventados (identidade conferida pelo avaliador):")
    for cid, c in lab.conceitos.items():
        q = LB.valor_conceito(c, Z)
        ids = {k: abs(float(np.corrcoef(q, v[k])[0, 1])) for k in ("s2", "h")}
        nome = "= s² (rapidez ao quadrado)" if ids["s2"] >= 0.99 else "= h (momento angular)" if ids["h"] >= 0.99 else ""
        print(f"  {cid}: {LB.formula_conceito(c):50s} |corr s²| {ids['s2']:.3f} |corr h| {ids['h']:.3f} {nome}"
              f"  teorias {sorted(c['teorias'])} revisoes {c.get('revisoes', 0)}")
    print(f"experimentos que ela montou: {lab.contas.get('experimentos', 0)} "
          f"(decisivos {lab.contas.get('experimentos_decisivos', 0)}); cenas guardadas: {len(getattr(lab, 'cenas', {}))}")
    T = lab.teoria_atual("arena")
    cenas = getattr(lab, "cenas", {})
    if T is not None and cenas:
        from experimento_sonho import compara_filmes
        uni = AR.universo_da_visita(max(st["visitas"] - 1, 0))
        rf = np.random.default_rng([st["semente"], 99])
        lei, reto = [], []
        for d in range(10):
            Z0 = lab.estados_imaginaveis(T, 3, rf)
            if Z0 is None:
                break
            c = cenas[sorted(cenas)[int(rf.integers(len(cenas)))]]
            ep = AR.episodio(uni, np.random.default_rng([st["semente"], 99000 + d]), cam, dur=6.0, armadilhas=False,
                             estados0=[AR.de_percebido(cam, u) for u in Z0])
            tt = np.asarray(ep.tempos, float)
            lei.append(compara_filmes(lab.imagina_video(c, Z0, T["lei"], tt)[0], ep.quadros)[1])
            reto.append(compara_filmes(lab.imagina_video(c, Z0, None, tt)[0], ep.quadros)[1])
        if lei:
            print(f"filme imaginado x real no universo atual ({uni}), aos 2 s: lei dela {np.median(lei):.6f} x reto "
                  f"{np.median(reto):.6f}; a lei vence em {sum(a < b for a, b in zip(lei, reto))}/{len(lei)} filmes")
    print("teorias e de onde vieram os episodios delas (universo da arena pela visita):")
    for tid, T in sorted(lab.teorias.items(), key=lambda kv: -len(kv[1]["membros"])):
        uni = collections.Counter()
        for e in T["membros"]:
            u = lab.eps[e]["url"]
            uni[AR.universo_da_visita(int(u.split("//")[1])) if u.startswith("arena://") else "internet"] += 1
        print(f"  teoria {tid}: {len(T['membros'])} episodios {dict(uni)} conceito {T['lei'].get('conceito')}")
        print(f"     {LB.formula_lei(T['lei'])[:160]}")


if __name__ == "__main__":
    main()
