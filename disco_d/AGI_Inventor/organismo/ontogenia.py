"""
organismo.ontogenia — o que mudou DENTRO da mente ao longo da vida
=====================================================================
A regua da ideia E4 ("o que mudou internamente depois de viver isso?") e das
ideias G2-G4 (ler os logs, preferencias emergentes, grafo da ontogenia). Le o
JSONL de uma vida e escreve:

  - curva de acerto por fase (janela de 20 problemas)
  - nascimentos por tipo ao longo do tempo; conceitos com suporte e profundidade
  - o que ficou LATENTE e voltou para uma resposta certa (com o intervalo)
  - coalizoes (habitos) que se formaram
  - episodios de regime (temperatura alta) e o que veio depois
  - crenca nas fontes no fim da vida
  - PREFERENCIA OPERACIONAL: em cada fase, que fracao do computo foi para cada
    operacao (para onde o organismo vai quando tem liberdade e o mesmo custo)
"""
import collections
import json
import sys

import numpy as np


def le(arq):
    return [json.loads(l) for l in open(arq, encoding="utf-8")]


def relatorio(arq, saida=None):
    ev = le(arq)
    out = []
    p = out.append
    probs = [e for e in ev if e["ev"] == "problema"]
    p(f"# Ontogenia de {arq.split('/')[-1].split(chr(92))[-1]}\n")
    p(f"{len(probs)} problemas vividos; {sum(e['ok'] for e in probs)} certos.\n")
    p("## Curva de acerto (janelas de 20 problemas, so percebidos)\n")
    ver = [e for e in probs if e["fonte"] == "percebido"]
    for i in range(0, 120, 20):
        j = [e for e in ver if i <= e["idx"] < i + 20]
        if j:
            p(f"- {i:3d}-{i + 19:3d}: {np.mean([e['ok'] for e in j]):.2f} "
              f"({len(j)} problemas, gasto medio {np.mean([e['gasto'] for e in j]):.0f})")
    p("\n## Nascimentos por tipo e fase\n")
    nasc = [e for e in ev if e["ev"] == "nascimento"]
    tab = collections.defaultdict(lambda: [0, 0, 0])
    for e in nasc:
        tab[e["tipo"]][min(e["t"] // 40, 2)] += 1
    for t, v in sorted(tab.items()):
        p(f"- {t:10s} fase1 {v[0]:3d}  fase2 {v[1]:3d}  fase3 {v[2]:3d}")
    p("\n## Conceitos inventados (sem nome; o organismo nao sabe como os humanos os chamam)\n")
    for e in nasc:
        if e["tipo"] == "conceito":
            d = e.get("desc") or {}
            sup = " + ".join("*".join(m) for m in d.get("suporte", []))
            p(f"- t={e['t']:3d} {e['id']:14s} prof {d.get('prof')}  suporte: {sup}  (fonte {e['fonte']})")
    p("\n## O que estava latente e voltou numa resposta certa\n")
    reat = [e for e in ev if e["ev"] == "reativacao"]
    if not reat:
        p("- nada")
    for e in sorted(reat, key=lambda e: -e["intervalo"])[:15]:
        p(f"- t={e['t']:3d} {e['id']:14s} ({e['tipo']}) voltou depois de {e['intervalo']} problemas")
    p("\n## Habitos (coalizoes)\n")
    co = [e for e in ev if e["ev"] == "coalizao"]
    if not co:
        p("- nenhum")
    for e in co:
        p(f"- t={e['t']:3d} {e['id']}: {' -> '.join(f'{a}({b})' if b else a for a, b in e['passos'])} "
          f"(vista nos problemas {e['problemas']})")
    p("\n## Regime\n")
    reg = [e for e in ev if e["ev"] == "regime"]
    quentes = [e for e in reg if e["T"] >= 0.5]
    p(f"- {len(reg)} mudancas de temperatura de vida; {len(quentes)} acima de 0,5")
    Tq = [e for e in probs if e["T"] >= 0.5]
    if Tq:
        p(f"- problemas que terminaram quentes (T >= 0,5): {len(Tq)}, acerto {np.mean([e['ok'] for e in Tq]):.2f}")
    p("\n## Crenca nas fontes (ultimo valor)\n")
    fontes = {}
    for e in ev:
        if e["ev"] == "fonte":
            fontes[e["fonte"]] = e["cred"]
    for f, c in fontes.items():
        p(f"- {f}: {c:.2f}")
    p("\n## Metaplasticidade (constante de esquecimento por tipo, ultimo valor)\n")
    mp = {}
    for e in ev:
        if e["ev"] == "metaplasticidade":
            mp[e["tipo"]] = e["tau"]
    for t, v in mp.items():
        p(f"- {t}: tau {v:.1f} problemas (comecou em 20)")
    p("\n## Preferencia operacional: fracao de acoes por operacao, por fase\n")
    for fase in range(3):
        c = collections.Counter()
        for e in probs:
            if fase * 40 <= e["idx"] < (fase + 1) * 40:
                for a in e["acoes"]:
                    c[a[0]] += 1
        tot = sum(c.values()) or 1
        top = ", ".join(f"{k} {v / tot:.0%}" for k, v in c.most_common(7))
        p(f"- fase {fase + 1}: {top}")
    import os
    jarq = arq[:-6] + ".json" if arq.endswith(".jsonl") else None
    if jarq and os.path.exists(jarq):
        r = json.load(open(jarq, encoding="utf-8"))
        es = [e for e in r.get("escopos", []) if e["tipo"] in ("conceito", "direcao")
              and e["familias_pagou"]]
        if es:
            p("\n## Alcance aprendido (largura media: 0 = vale em todo lugar, 16 = so no contexto exato)\n")
            for e in sorted(es, key=lambda e: e["beta_medio"]):
                d = e["desc"]
                txt = d.get("expr") if isinstance(d, dict) and "expr" in d else \
                    " + ".join("*".join(m) for m in (d.get("suporte", []) if isinstance(d, dict) else []))
                p(f"- {e['id']:14s} largura {e['beta_medio']:5.2f}  pagou em {', '.join(e['familias_pagou'])}  [{txt}]")
    p("\n## Sonhos\n")
    so = [e for e in ev if e["ev"] == "sonho"]
    p(f"- {len(so)} sonhos; {sum(e['valida'] for e in so)} validaram no dado guardado")
    txt = "\n".join(out)
    if saida:
        open(saida, "w", encoding="utf-8").write(txt)
    return txt


if __name__ == "__main__":
    print(relatorio(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
