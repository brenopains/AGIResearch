"""
novaphys.bench — o julgamento
==============================

Roda uma tarefa de indução de leis com dois cientistas (ativo e aleatório),
mesmo orçamento, mesmas sondas, mesmo regressor, e compara.

A pergunta que este arquivo responde é única e específica:

    Escolher onde medir vale alguma coisa?

Se a razão erro_aleatório / erro_ativo for ~1, escolher não valeu nada e
não houve agência — só regressão. Se for >1, o agente extraiu mais lei por
experimento do que o acaso, que é o que se espera de um cientista.

Nada aqui mede "inteligência". Mede uma coisa estreita e verificável, que
é o único jeito de um número significar alguma coisa.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from .client import PhysicsWorld
from .agents import (BasisRegressor, HybridPredictor, RandomScientist,
                     ActiveScientist, ActiveScientistV2,
                     ActiveOptimizer, RandomOptimizer)


def _rodar_predicao(w: PhysicsWorld, tarefa: str, cientista, semente: int,
                    n_sondas: int, verboso: bool, hibrido: bool = True) -> Dict[str, Any]:
    info = w.tarefa_iniciar(tarefa, semente=semente, sondas=n_sondas)
    espaco, orcamento, sondas = info["espaco"], info["orcamento"], info["sondas"]
    historico: List[Dict[str, Any]] = []

    for _ in range(orcamento):
        try:
            cfg = cientista.escolher(espaco, historico, orcamento=orcamento)
        except TypeError:
            cfg = cientista.escolher(espaco, historico)
        r = w.tarefa_experimento(cfg)
        historico.append({"config": cfg, "resultado": r["resultado"]})
        if verboso:
            dims = " ".join(f"{k}={v:.4g}" for k, v in cfg.items())
            print(f"      [{cientista.nome:26s}] {dims:38s} → {r['resultado']:.5g}")

    # MESMO preditor nos dois braços — é o que faz a comparação isolar
    # exclusivamente a escolha dos experimentos.
    dims_k = [d["chave"] for d in espaco]
    reg = (HybridPredictor(dims_k, espaco, log_target=True) if hibrido
           else BasisRegressor(dims_k, log_target=True))
    for h in historico:
        reg.add(h["config"], max(1e-12, abs(h["resultado"])))
    reg.fit()
    pred = reg.predict_many(sondas)

    res = w.tarefa_responder(predicoes=pred)
    return {
        "erro": res["erro"], "unidade": res["unidadeDoErro"],
        "verdades": res["verdades"], "predicoes": pred,
        "segredo": res.get("segredo", {}), "historico": historico,
    }


def _rodar_otimizacao(w: PhysicsWorld, tarefa: str, otimizador, semente: int,
                      verboso: bool) -> Dict[str, Any]:
    info = w.tarefa_iniciar(tarefa, semente=semente, sondas=0)
    espaco, orcamento = info["espaco"], info["orcamento"]
    historico: List[Dict[str, Any]] = []

    for _ in range(orcamento):
        cfg = otimizador.escolher(espaco, historico, orcamento=orcamento)
        r = w.tarefa_experimento(cfg)
        historico.append({"config": cfg, "resultado": r["resultado"]})
        if verboso:
            dims = " ".join(f"{k}={v:.5f}" for k, v in cfg.items())
            print(f"      [{otimizador.nome:26s}] {dims:38s} → {r['resultado']:.5g}")

    prop = otimizador.melhor(historico)
    res = w.tarefa_responder(otimo=prop)
    return {"erro": res["erro"], "unidade": "arrependimento (unidades da medição)",
            "proposto": res["proposto"], "valor_obtido": res["valorObtido"],
            "valor_otimo": res["valorOtimo"], "argumento_otimo": res["argumentoOtimo"],
            "distancia_do_argumento": res["distanciaDoArgumento"],
            "historico": historico}


def rodar_tarefa(w: PhysicsWorld, tarefa: str, semente: int = 7,
                 n_sondas: int = 8, verboso: bool = False,
                 versao: str = "v2", hibrido: bool = True) -> Dict[str, Any]:
    """
    Roda uma tarefa com os dois cientistas na MESMA semente (mesmo segredo).
    `versao` escolhe o agente ativo: "v1" (incerteza sobre base suave) ou
    "v2" (desacordo entre modelo suave e local). Comparar as duas é o teste
    direto da correção.
    """
    meta = next(t for t in w.tarefas() if t["id"] == tarefa)
    otimizacao = meta["metrica"] == "otimizacao"

    if otimizacao:
        ativo = _rodar_otimizacao(w, tarefa, ActiveOptimizer(seed=semente), semente, verboso)
        aleat = _rodar_otimizacao(w, tarefa, RandomOptimizer(seed=semente), semente, verboso)
    else:
        Agente = ActiveScientistV2 if versao == "v2" else ActiveScientist
        ativo = _rodar_predicao(w, tarefa, Agente(seed=semente), semente, n_sondas,
                                verboso, hibrido=hibrido)
        aleat = _rodar_predicao(w, tarefa, RandomScientist(seed=semente), semente, n_sondas,
                                verboso, hibrido=hibrido)

    ea, er = ativo["erro"], aleat["erro"]
    vantagem = (er / ea) if (ea and ea > 0 and er < float("inf")) else float("nan")
    return {
        "tarefa": tarefa, "nome": meta["nome"], "semente": semente,
        "orcamento": meta["orcamento"], "metrica": meta["metrica"],
        "erro_ativo": ea, "erro_aleatorio": er, "vantagem": vantagem,
        "unidade": ativo.get("unidade"), "detalhe_ativo": ativo, "detalhe_aleatorio": aleat,
    }


def rodar_suite(w: PhysicsWorld, tarefas: Optional[List[str]] = None,
                sementes: Optional[List[int]] = None, n_sondas: int = 8,
                verboso: bool = False, versao: str = "v2") -> Dict[str, Any]:
    """
    Suite completa.

    O padrão são 15 sementes, e isso é uma correção, não um capricho. A versão
    anterior usava 3 e reportava a mediana — o que parecia razoável e não era:
    numa das tarefas o espalhamento entre sementes chegou a 140×, e com n=3 a
    mediana disso não distingue uma vantagem de 5× de uma desvantagem de 3×.
    Rodando com n=40 na mesma tarefa, conclusões tiradas com n=3 se inverteram.

    Por isso a saída agora mostra p25 e p75 além da mediana: se os intervalos
    de duas configurações se sobrepõem, a comparação entre elas não decide nada,
    e é melhor ver isso do que ler um número único com confiança indevida.
    """
    if sementes is None:
        sementes = [7, 23, 91, 144, 255, 311, 402, 517, 623, 748, 851, 967, 1024, 1177, 1300]
    disponiveis = [t["id"] for t in w.tarefas()]
    tarefas = tarefas or disponiveis
    linhas: List[Dict[str, Any]] = []

    for t in tarefas:
        if t not in disponiveis:
            print(f"  ! tarefa desconhecida ignorada: {t}")
            continue
        meta = next(m for m in w.tarefas() if m["id"] == t)
        print(f"\n─── {meta['nome']}  ({t})")
        print(f"    orçamento: {meta['orcamento']} experimentos · métrica: {meta['metrica']}")
        vs, eas, ers = [], [], []
        for s in sementes:
            r = rodar_tarefa(w, t, semente=s, n_sondas=n_sondas, verboso=verboso,
                             versao=versao)
            vs.append(r["vantagem"]); eas.append(r["erro_ativo"]); ers.append(r["erro_aleatorio"])
            print(f"    semente {s:>3}:  ativo {r['erro_ativo']:.4g}   "
                  f"aleatório {r['erro_aleatorio']:.4g}   vantagem {r['vantagem']:.2f}×")
        vs_ok = sorted(v for v in vs if v == v)
        def _pct(xs, q):
            xs = sorted(xs)
            if not xs:
                return float("nan")
            i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
            return xs[i]
        linha = {
            "tarefa": t, "nome": meta["nome"], "metrica": meta["metrica"],
            "erro_ativo_mediano": statistics.median(eas),
            "erro_aleatorio_mediano": statistics.median(ers),
            "vantagem_mediana": statistics.median(vs_ok) if vs_ok else float("nan"),
            "vantagem_p25": _pct(vs_ok, 0.25), "vantagem_p75": _pct(vs_ok, 0.75),
            "n": len(vs_ok), "sementes": list(sementes),
        }
        linhas.append(linha)
        print(f"    → vantagem mediana {linha['vantagem_mediana']:.2f}×  "
              f"(p25 {linha['vantagem_p25']:.2f}× · p75 {linha['vantagem_p75']:.2f}×, n={linha['n']})")

    print("\n" + "═" * 78)
    print("RESUMO — vantagem = erro do acaso ÷ erro do agente ativo (>1 é melhor)")
    print("═" * 78)
    print(f"{'tarefa':<22}{'ativo':>11}{'aleatório':>12}{'vantagem':>11}{'p25–p75':>18}")
    for l in linhas:
        print(f"{l['tarefa']:<22}{l['erro_ativo_mediano']:>11.4g}"
              f"{l['erro_aleatorio_mediano']:>12.4g}{l['vantagem_mediana']:>10.2f}×"
              f"{l['vantagem_p25']:>10.2f}–{l['vantagem_p75']:.2f}×")
    inconclusivas = [l for l in linhas if l["vantagem_p25"] < 1.0 < l["vantagem_p75"]]
    if inconclusivas:
        print()
        print("INCONCLUSIVAS (o intervalo p25–p75 cruza 1,00× — o sinal não separa do acaso):")
        for l in inconclusivas:
            print(f"  · {l['tarefa']}")
    vs = [l["vantagem_mediana"] for l in linhas if l["vantagem_mediana"] == l["vantagem_mediana"]]
    if vs:
        venceu = [l for l in linhas if l["vantagem_mediana"] > 1.15]
        perdeu = [l for l in linhas if l["vantagem_mediana"] < 0.87]
        empatou = len(linhas) - len(venceu) - len(perdeu)
        print("─" * 78)
        print(f"placar: ativo venceu em {len(venceu)}/{len(linhas)} tarefas, "
              f"perdeu em {len(perdeu)}, empatou em {empatou}")
        print()
        # Uma mediana de razões entre métricas heterogêneas não é um número com
        # sentido físico — reportar tarefa a tarefa evita que um resultado bom
        # numa delas mascare uma falha noutra.
        if perdeu:
            print("ONDE O AGENTE ATIVO PERDEU (é aqui que está a informação útil):")
            for l in perdeu:
                print(f"  · {l['tarefa']}: {l['vantagem_mediana']:.2f}× — a heurística de busca")
                print("    está mal casada com o formato desta paisagem.")
            print()
        if not venceu:
            print("Em nenhuma tarefa escolher onde medir superou o acaso. Nesta configuração")
            print("o que existe é regressão, não descoberta ativa.")
        elif len(venceu) == len(linhas):
            print("Em todas as tarefas o agente extraiu mais lei por experimento que o acaso.")
        else:
            print("Resultado misto: a agência aparece em algumas paisagens e não em outras.")
            print("Isso é mais informativo que um número único — diz QUANDO a estratégia serve.")
    return {"linhas": linhas}
