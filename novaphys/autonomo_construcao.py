"""novaphys.autonomo_construcao — ETAPA 1: o laco autonomo TRANSFERE?

MEDIDA DE TRANSFERENCIA: tamanho do adaptador. Adaptador pequeno = transferiu.
Adaptador grande = eu reescrevi e chamei de transferencia.

RESULTADO DA TENTATIVA DE REUSO DIRETO: 0 linhas de `autonomo.py` importam sem
tocar. `surpresa()` precisa de (1) preditor numerico para qualquer estado,
(2) SIGMA do instrumento, (3) vetor de observacoes. A escada de construcao da UM
VEREDITO BINARIO de um verificador EXATO — nao ha ruido de instrumento, logo
"surpresa contra o ruido" e indefinida.

O QUE TRANSFERE E A FORMA, NAO O CODIGO:
  lei              -> custo previsto de busca por classe: log T(g,n) ~ a_g + b_g n
  ruido            -> variancia entre REINICIOS da busca aleatorizada
  surpresa         -> |log T_observado - log T_previsto| / sigma_reinicio
  observar         -> gastar orcamento tentando uma classe
  reanalise        -> reajustar a crenca sobre TODAS as instancias guardadas,
                      comecando pela que surpreendeu
"""
from __future__ import annotations
import time
import numpy as np

from novaphys.tres_rapido import busca_rapida
from novaphys.tres_em_linha import verifica, GRUPOS

CLASSES = ["rot4", "klein", "rot2", "espelho_h", "espelho_d", "d4", "iden"]


PARIDADE = False        # a lei tem paridade de n como variavel?


def preve_custo(hist, g, n):
    """A 'lei': log do custo da classe g na instancia n, ajustado no historico.
    Com PARIDADE=True a lei so usa instancias da MESMA paridade — e assim
    consegue exprimir 'em n impar vale outra classe'. Sem isso ela e cega ao
    regime que existe no problema."""
    sel = [h for h in hist if (h[1] % 2 == n % 2)] if PARIDADE else hist
    pts = [(m, np.log(max(t, 1e-3))) for (gg, m, t, ok) in sel if gg == g and ok]
    if len(pts) < 2:
        return None, None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (c[0] + c[1] * x)
    sig = max(float(res.std()), 0.35)          # piso: ruido de reinicio
    return float(c[0] + c[1] * n), sig


def surpresa_custo(hist, g, n, t_obs):
    """Analogo exato de `surpresa`: o quanto o custo realizado desmente a lei."""
    pred, sig = preve_custo(hist, g, n)
    if pred is None:
        return 0.0
    return abs(np.log(max(t_obs, 1e-3)) - pred) / sig


def reanalisa_ordem(hist, n, limiar):
    """Analogo de `reanalisa_consenso`: reordena as classes por custo previsto,
    usando TODAS as instancias guardadas; classe sem historico vai para o fim."""
    marc = []
    for g in CLASSES:
        pred, _ = preve_custo(hist, g, n)
        sel = [h for h in hist if (h[1] % 2 == n % 2)] if PARIDADE else hist
        suc = sum(1 for (gg, _m, _t, ok) in sel if gg == g and ok)
        marc.append((pred if pred is not None else 1e9, -suc, g))
    marc.sort()
    return [g for _p, _s, g in marc]


def corrente(braco, ns, prazo_n, semente=0, verbose=False):
    """braco: 'calendario' (ordem fixa por bits) ou 'surpresa' (reordena na crise)."""
    hist, ordem, crises, resolvidas = [], list(CLASSES), [], []
    maior = 0
    for n in ns:
        fatia = prazo_n / max(len(ordem), 1)
        achou = False
        for g in ordem:
            if g in ("rot4", "klein", "d4") and n % 2:
                continue
            t0 = time.time()
            r = busca_rapida(n, g, time.time() + fatia, semente=semente + n)
            dt = time.time() - t0
            ok = r is not None and verifica(r, n) and len(r) == 2 * n
            if braco == "surpresa":
                z = surpresa_custo(hist, g, n, dt)
                if z > 2.0:
                    crises.append((n, g, round(z, 1)))
            hist.append((g, n, dt, ok))
            if ok:
                maior = n
                achou = True
                break
        if braco == "surpresa" and crises:
            ordem = reanalisa_ordem(hist, n + 1, 2.0)
        resolvidas.append((n, achou))
    return maior, crises, hist, resolvidas
