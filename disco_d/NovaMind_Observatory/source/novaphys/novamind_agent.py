"""
novaphys.novamind_agent — o ActiveExperimenter do core na bancada
==================================================================

Até aqui a suíte comparava dois cientistas que eu escrevi. Isso mede o
benchmark, não a NovaMind.

Este adaptador põe `core/active_experimenter.py` — código do projeto, sem
modificação — para jogar contra a mesma baseline aleatória, com o mesmo
orçamento e as mesmas sondas. É o teste que ainda não tinha sido feito.

Uma limitação que é do adaptador e não do código: `ActiveExperimenter` é
escrito para exatamente duas dimensões de entrada e uma lei de potência
`t = a·m^b·s^c + d`. Tarefas de uma dimensão recebem uma segunda dimensão
constante. Onde a lei verdadeira não é uma potência (tunelamento é
exponencial, ignição tem limiar), a família de funções não consegue
representá-la — e isso vai aparecer no número. É informação sobre o
alcance da hipótese embutida, não sobre esforço de quem escreveu.
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NOVAMIND = os.path.join(os.path.dirname(_AQUI), "legacy", "novamind")


def _carregar():
    if _NOVAMIND not in sys.path:
        sys.path.insert(0, _NOVAMIND)
    from core.active_experimenter import ActiveExperimenter  # noqa: E402
    return ActiveExperimenter


class NovaMindScientist:
    """
    Envelopa o `ActiveExperimenter` do core como um cientista da suíte.

    Ele escolhe experimentos por ganho de informação esperado — exploração
    (distância à cobertura atual) vezes sensibilidade do modelo naquele ponto.
    A escolha é dele; o adaptador só traduz coordenadas.
    """

    nome = "NovaMind (core)"

    def __init__(self, seed: int = 0, n_cand: int = 60):
        self.rng = random.Random(seed)
        self.n_cand = n_cand
        self._AE = _carregar()
        self.ae = None
        self.dims: List[str] = []
        self.espaco: List[Dict[str, Any]] = []

    # ---- tradução entre o espaço da tarefa e o (m, s) do ActiveExperimenter
    def _para_ms(self, cfg: Dict[str, float]):
        a = float(cfg[self.dims[0]])
        b = float(cfg[self.dims[1]]) if len(self.dims) > 1 else 1.0
        return a, b

    def _de_ms(self, m: float, s: float) -> Dict[str, float]:
        cfg = {self.dims[0]: m}
        if len(self.dims) > 1:
            cfg[self.dims[1]] = s
        return cfg

    def _amostra(self) -> Dict[str, float]:
        cfg = {}
        for d in self.espaco:
            if d.get("log"):
                cfg[d["chave"]] = math.exp(self.rng.uniform(math.log(d["min"]), math.log(d["max"])))
            else:
                cfg[d["chave"]] = self.rng.uniform(d["min"], d["max"])
        return cfg

    def _preparar(self, espaco, historico):
        if self.ae is None or self.espaco != espaco:
            self.espaco = espaco
            self.dims = [d["chave"] for d in espaco]
            d0 = espaco[0]
            d1 = espaco[1] if len(espaco) > 1 else {"min": 0.5, "max": 1.5}
            self.ae = self._AE(dim_mass_range=(d0["min"], d0["max"]),
                               dim_size_range=(d1["min"], d1["max"]))
        # sincroniza as observações que a suíte já coletou
        if len(self.ae.observations) != len(historico):
            self.ae.observations = []
            for h in historico:
                m, s = self._para_ms(h["config"])
                self.ae.observe(m, s, max(1e-9, abs(h["resultado"])))
            self.ae.fit_power_law()

    def escolher(self, espaco, historico, orcamento: int = 20) -> Dict[str, float]:
        self._preparar(espaco, historico)
        candidatos = [self._para_ms(self._amostra()) for _ in range(self.n_cand)]
        try:
            m, s = self.ae.choose_experiment(candidatos)
        except Exception:
            return self._amostra()
        return self._de_ms(m, s)

    # ---- também sabe prever, para o modo "preditor próprio"
    def prever(self, cfgs) -> List[float]:
        out = []
        for c in cfgs:
            m, s = self._para_ms(c)
            try:
                out.append(float(self.ae.predict(m, s)))
            except Exception:
                out.append(1.0)
        return out

    # ---- interface de otimização
    def melhor(self, historico) -> Dict[str, float]:
        h = max(historico, key=lambda x: x["resultado"])
        return dict(h["config"])


def comparar_novamind(w, tarefas: Optional[List[str]] = None,
                      sementes=(7, 23, 91, 144), n_sondas: int = 8):
    """
    Roda o ActiveExperimenter do core contra a baseline aleatória, com o mesmo
    preditor compartilhado nos dois braços — de modo que a única diferença
    continua sendo a escolha dos experimentos.

    Reporta também o resultado com o preditor PRÓPRIO do core (a power-law
    dele), porque as duas coisas respondem a perguntas diferentes:
      · preditor compartilhado → a seleção de experimentos dele é boa?
      · preditor próprio       → o modelo dele representa esta lei?
    """
    import statistics
    from .agents import RandomScientist, HybridPredictor
    from .bench import _rodar_predicao

    disponiveis = [t["id"] for t in w.tarefas()]
    tarefas = [t for t in (tarefas or disponiveis)
               if t in disponiveis and
               next(m for m in w.tarefas() if m["id"] == t)["metrica"] != "otimizacao"]

    print("═" * 78)
    print("ActiveExperimenter do core  vs  escolha aleatória")
    print("mesmo orçamento, mesmas sondas, mesmo preditor compartilhado")
    print("═" * 78)
    linhas = []
    for t in tarefas:
        meta = next(m for m in w.tarefas() if m["id"] == t)
        print(f"\n─── {meta['nome']}  ({t})   orçamento {meta['orcamento']}")
        en, er, ep = [], [], []
        for s in sementes:
            nm = NovaMindScientist(seed=s)
            a = _rodar_predicao(w, t, nm, s, n_sondas, False, hibrido=True)
            b = _rodar_predicao(w, t, RandomScientist(seed=s), s, n_sondas, False, hibrido=True)
            en.append(a["erro"]); er.append(b["erro"])

            # e agora com o preditor do próprio core
            info = w.tarefa_iniciar(t, semente=s, sondas=n_sondas)
            nm2 = NovaMindScientist(seed=s)
            hist = []
            for _ in range(info["orcamento"]):
                cfg = nm2.escolher(info["espaco"], hist, orcamento=info["orcamento"])
                r = w.tarefa_experimento(cfg)
                hist.append({"config": cfg, "resultado": r["resultado"]})
            nm2._preparar(info["espaco"], hist)
            res = w.tarefa_responder(predicoes=nm2.prever(info["sondas"]))
            ep.append(res["erro"])

            print(f"    semente {s:>3}:  core {a['erro']:.4g}   aleatório {b['erro']:.4g}"
                  f"   core c/ modelo próprio {res['erro']:.4g}")
        vant = statistics.median(er) / max(1e-12, statistics.median(en))
        linhas.append({"tarefa": t, "core": statistics.median(en),
                       "aleatorio": statistics.median(er),
                       "core_modelo_proprio": statistics.median(ep), "vantagem": vant})
        print(f"    → mediana: core {statistics.median(en):.4g} · "
              f"aleatório {statistics.median(er):.4g} · vantagem {vant:.2f}×")

    print("\n" + "═" * 78)
    print(f"{'tarefa':<20}{'core':>11}{'aleatório':>12}{'vantagem':>11}{'modelo próprio':>17}")
    for l in linhas:
        print(f"{l['tarefa']:<20}{l['core']:>11.4g}{l['aleatorio']:>12.4g}"
              f"{l['vantagem']:>10.2f}×{l['core_modelo_proprio']:>17.4g}")
    venceu = sum(1 for l in linhas if l["vantagem"] > 1.15)
    print("─" * 78)
    print(f"o ActiveExperimenter do core venceu o acaso em {venceu}/{len(linhas)} tarefas")
    return linhas
