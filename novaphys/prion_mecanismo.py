"""
novaphys.prion_mecanismo — o que mata o neuronio decide qual alvo serve
========================================================================

DE ONDE VEM ESTA PERGUNTA
---------------------------
`prion_validacao.py` reprovou o modelo de carga: ajustando o knockdown no
braco profilatico e prevendo o braco de dose unica tardia, ele errou por
-15,3 pontos, SUBESTIMANDO o beneficio de tratar tarde.

O diagnostico: no modelo de carga so importa quanto agregado se acumulou.
Reduzir o suprimento no dia 120 muda pouco, porque a carga ja esta quase no
limiar. Mas o dado diz que ajuda muito.

A HIPOTESE
------------
O que causa o dano nao e a CARGA de agregado. E o PROCESSO DE CONVERSAO — que
consome PrP normal continuamente. Baixar PrP-C corta o fluxo na hora, mesmo
com agregado ja estabelecido.

Isso nao e invencao: e uma hipotese com apoio na literatura de que PrP-C
medeia a toxicidade, e nao serve so de substrato.

QUATRO DEFINICOES DE DESFECHO, MESMA DINAMICA
-----------------------------------------------
    carga        y cruza um limiar                    <- modelo reprovado
    carga-tempo  integral de y                        exposicao acumulada
    fluxo        integral de 2*beta*x*z               conversao acumulada
    prpc         integral de x*z                      PrP-C x moldes

So o desfecho muda. As EDOs sao identicas. Cada um e calibrado
independentemente para reproduzir o mesmo controle.

O TESTE, SEM PARAMETRO LIVRE
------------------------------
    1. calibra beta para o controle da coorte
    2. ajusta k SO no braco NIH profilatico ASO1 (81%)
    3. PREVE o braco de dose unica no dia 120 (55%) com o mesmo k

Restringido ao braco NIH ASO1 e a dose tardia porque sao a MESMA molecula.
A validacao anterior comparou ASO1 com ASO2 (moleculas diferentes) e NIH com
Broad (cronogramas diferentes) — dois erros meus de desenho que inflaram o
erro medio para 41,9 pp sem medir nada.

O QUE ESTA EM JOGO, E E O PONTO
---------------------------------
    se o desfecho e CARGA   -> remover agregado (depuracao) lidera
    se o desfecho e FLUXO   -> cortar PrP-C (ASO) lidera

A ordem de alvo terapeutico NAO e propriedade da cinetica. E propriedade do
que efetivamente mata a celula. Por isso a tabela de `prion.py` foi retirada:
ela respondeu a pergunta assumindo uma resposta.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  M0  PRE-CONDICAO. Cada desfecho tem que calibrar para o controle.
  M1  algum desfecho preve a dose tardia com erro < 15 pp
      -> ha mecanismo compativel com os dois bracos ao mesmo tempo
  M2  nenhum preve -> a falha nao e do desfecho; falta outra coisa
  M3  com o desfecho vencedor, refazer a ordem de alvos. Se ela inverter em
      relacao a `prion.py`, esta demonstrado que a ordem depende do
      mecanismo de dano, e nao da cinetica.

MINHA PREVISAO: `fluxo` ganha, porque e o unico onde baixar x tem efeito
imediato independente de y. E espero que a ordem de alvos INVERTA, com o ASO
passando a depuracao. Se `carga` ganhar, minha hipotese cai.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from scipy.integrate import solve_ivp

from .prion import BASE
from .prion_validacao import PERSIST, REBOTE, _efeito

CONTROLE_NIH = 143.0        # dias, saline
TARDIO_CTRL = 157.0         # controle da coorte de dose unica
OBS_PROFILATICO = 81.0      # %
OBS_TARDIO = 55.0           # %
DIA_DOSE = 120.0

# taxa de dano por desfecho: g(x, y, z, p) -> dD/dt
DESFECHOS: Dict[str, Callable] = {
    "carga":       lambda x, y, z, p: 0.0,          # tratado a parte (limiar em y)
    "carga-tempo": lambda x, y, z, p: y,
    "fluxo":       lambda x, y, z, p: 2 * p["beta"] * x * z,
    "prpc":        lambda x, y, z, p: x * z,
}


def simular(p: Dict[str, float], desfecho: str, limiar: float,
            k: float = 0.0, inicio: float = 0.0, sustentado: bool = True,
            t_max: float = 2500.0) -> Optional[float]:
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]
    g = DESFECHOS[desfecho]
    usa_carga = desfecho == "carga"

    def campo(t, s):
        x, y, z, _D = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z,
                g(x, y, z, p)]

    def cruza(t, s):
        return (s[1] if usa_carga else s[3]) - limiar
    cruza.terminal = True
    cruza.direction = 1

    sol = solve_ivp(campo, (0.0, t_max), [lam0 / d, n * 1.0, 1.0, 0.0],
                    method="Radau", events=cruza, rtol=1e-9, atol=1e-9)
    if not sol.success:
        return None
    if sol.t_events and len(sol.t_events[0]) > 0:
        return float(sol.t_events[0][0])
    return None


def calibrar(desfecho: str, alvo: float, beta_fixo: float) -> float:
    """
    Fixa beta (a cinetica e a mesma) e calibra o LIMIAR do desfecho para o
    controle. Assim os quatro desfechos partem da mesma dinamica e diferem
    so no que conta como morte.
    """
    p = dict(BASE)
    p["beta"] = beta_fixo
    lo, hi = 1e-6, 1e12
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        t = simular(p, desfecho, mid)
        if t is None or t > alvo:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def calibrar_beta(alvo: float) -> float:
    lo, hi = 1e-5, 1.0
    p = dict(BASE)
    for _ in range(60):
        mid = (lo + hi) / 2
        p["beta"] = mid
        t = simular(p, "carga", 55000.0)
        if t is None or t > alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def relatorio() -> None:
    print("=" * 78)
    print("PRION — qual mecanismo de dano explica os DOIS bracos?")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  M0  cada desfecho calibra para o controle")
    print("  M1  algum preve a dose tardia com erro < 15 pp")
    print("  M2  nenhum preve -> falta outra coisa")
    print("  M3  com o vencedor, a ordem de alvos inverte?")
    print("\nMINHA PREVISAO: `fluxo` ganha e a ordem INVERTE (ASO passa depuracao)")
    print("\nso a MESMA molecula: ajusta em NIH ASO1 profilatico (81%),")
    print(f"preve dose unica no dia {DIA_DOSE:.0f} (observado {OBS_TARDIO:.0f}%)")

    beta_nih = calibrar_beta(CONTROLE_NIH)
    beta_tar = calibrar_beta(TARDIO_CTRL)
    print(f"\nbeta: NIH {beta_nih:.5g} (ctrl {CONTROLE_NIH:.0f} d) · "
          f"tardio {beta_tar:.5g} (ctrl {TARDIO_CTRL:.0f} d)")

    print("\n" + "-" * 78)
    print(f"{'desfecho':<14}{'k ajustado':>12}{'previsto':>11}"
          f"{'observado':>11}{'erro':>9}")
    print("-" * 78)

    resultados = []
    for nome in DESFECHOS:
        lim_n = 55000.0 if nome == "carga" else calibrar(nome, CONTROLE_NIH,
                                                         beta_nih)
        lim_t = 55000.0 if nome == "carga" else calibrar(nome, TARDIO_CTRL,
                                                         beta_tar)
        pn = dict(BASE); pn["beta"] = beta_nih
        pt = dict(BASE); pt["beta"] = beta_tar
        # M0
        t_ctrl = simular(pn, nome, lim_n)
        if t_ctrl is None or abs(t_ctrl - CONTROLE_NIH) / CONTROLE_NIH > 0.02:
            print(f"{nome:<14}  M0 REPROVADA (controle {t_ctrl})")
            continue
        # ajusta k no profilatico
        lo, hi = 0.0, 0.995
        for _ in range(50):
            mid = (lo + hi) / 2
            t = simular(pn, nome, lim_n, k=mid, inicio=-14.0, sustentado=True)
            e = math.inf if t is None else 100 * (t - CONTROLE_NIH) / CONTROLE_NIH
            if e > OBS_PROFILATICO:
                hi = mid
            else:
                lo = mid
        k = (lo + hi) / 2
        # PREVE o tardio
        t = simular(pt, nome, lim_t, k=k, inicio=DIA_DOSE, sustentado=False)
        prev = math.inf if t is None else 100 * (t - TARDIO_CTRL) / TARDIO_CTRL
        err = math.inf if prev == math.inf else prev - OBS_TARDIO
        resultados.append((nome, k, prev, err, lim_n))
        ps = "nao progride" if prev == math.inf else f"{prev:.1f}%"
        es = "—" if err == math.inf else f"{err:+.1f}"
        print(f"{nome:<14}{k:>11.1%}{ps:>11}{OBS_TARDIO:>10.0f}%{es:>9}")

    print("\n" + "=" * 78)
    validos = [r for r in resultados if r[3] != math.inf]
    if not validos:
        print("M2 · nenhum desfecho produz previsao finita.")
        return
    melhor = min(validos, key=lambda r: abs(r[3]))
    nome, k, prev, err, lim = melhor
    if abs(err) < 15.0:
        print(f"M1 · `{nome}` PREVE O BRACO TARDIO — erro {err:+.1f} pp, com k")
        print(f"     ajustado noutro braco e nenhum parametro livre.")
        print(f"     O modelo de carga errava {-15.3:+.1f}; este erra {err:+.1f}.")
    else:
        print(f"M2 · NENHUM desfecho preve. O melhor (`{nome}`) erra {err:+.1f} pp.")
        print("     A falha nao esta na definicao de morte; falta outra coisa.")

    # ── M3: a ordem de alvos depende do mecanismo?
    print("\n" + "-" * 78)
    print(f"M3 · ordem de alvos sob `{nome}` — dose minima que interrompe")
    pn = dict(BASE); pn["beta"] = beta_nih
    ALVOS = {"lam- (ASO)": ("lam", -1), "a+ (depuracao)": ("a", +1),
             "beta- (anti-alonga)": ("beta", -1), "b- (anti-frag)": ("b", -1)}
    ordem = []
    for rot, (par, sinal) in ALVOS.items():
        lo, hi, achou = 0.0, 0.98, False
        for _ in range(40):
            mid = (lo + hi) / 2
            q = dict(pn)
            q[par] = pn[par] * (1 - mid) if sinal < 0 else pn[par] / max(1e-9, 1 - mid)
            if simular(q, nome, lim) is None:
                hi = mid; achou = True
            else:
                lo = mid
        ordem.append((rot, hi if achou else None))
    ordem.sort(key=lambda r: (r[1] is None, r[1] if r[1] else 9))
    for rot, dm in ordem:
        print(f"     {rot:<22}{'nao interrompe' if dm is None else f'{dm:.1%}'}")
    print(f"\n     referencia sob `carga` (retirada): a+ 20,0% · lam- 24,5%")
    if ordem[0][0].startswith("lam-"):
        print("\n     M3 · A ORDEM INVERTEU. Sob este mecanismo de dano, cortar")
        print("     PrP-C lidera — e a liderança da depuracao era artefato de")
        print("     assumir que o dano vem da CARGA.")
    else:
        print(f"\n     M3 · a ordem NAO inverteu; `{ordem[0][0]}` continua na frente.")


if __name__ == "__main__":
    relatorio()
