"""
novaphys.lente_ativa_limpa — escolher o experimento cura a subamostragem?
=========================================================================

DE ONDE VEM
-------------
No mundo LIMPO da lente (particula desvia em direcao a massa adjacente), o sistema
descobriu o termo novo em 3/6 episodios contra 0/6 do controle de ruido. As falhas
tinham padrao: com 5 pares de treino SORTEADOS, as vezes todos os desvios caem para o
mesmo lado, e o caso oposto so aparece no teste. Isso e SUBAMOSTRAGEM.

E subamostragem e exatamente o que AGENCIA deveria curar: em vez de receber cenas
sorteadas, o agente ESCOLHE a cena que mais divide as proprias hipoteses. Nenhuma das
~35 propostas de onze agentes tocou agencia.

O `lente_ativa.py` original rodou sobre o mundo SUJO (colisoes, bloqueio), que estava
confundido — la ativo e passivo deram zero e nao diziam nada. Este refaz sobre o limpo.

O DESENHO
-----------
Orcamento CASADO de 5 observacoes de treino por episodio.

    PASSIVO   5 cenas sorteadas
    ATIVO     3 cenas sorteadas; depois 2 cenas ESCOLHIDAS, cada uma entre 10
              candidatas, a de maior desacordo num comite de 3 hipoteses

Comite por bootstrap (Seung, Opper & Sompolinsky 1992): `montar` em 3 reamostragens
dos pares ja vistos. O braco ativo NAO sabe que a massa importa; so sabe onde as
proprias hipoteses discordam.

Os dois mundos (lente e sorteado) nos dois bracos: a agencia NAO pode manufaturar lei
no mundo sem lei.

PRIOR MEDIDO NESTE PROJETO, E ELE E DURO
------------------------------------------
"desenho de experimento da 30-54x SO com modelo de mundo DADO — com modelo APRENDIDO,
vantagem zero." Aqui o modelo e aprendido. Mas ha agora um mecanismo especifico pelo
qual poderia ajudar: as falhas sao subamostragem de um lado, e o comite deve discordar
justamente no lado nao visto.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. O braco passivo no mundo lente tem que dar descoberta proxima dos
      3/6 do `lente_limpa2` (entre 30% e 60%). Senao a comparacao nao vale.

  A1  ATIVO > PASSIVO em descoberta no mundo lente por >= 2 episodios de 10, e o
      sorteado fica em ~0 nos dois bracos -> escolher o experimento cura
      subamostragem com modelo APRENDIDO, refinando o prior do projeto
  A2  empate (diferenca < 2) -> o prior se mantem
  A3  ativo "descobre" no sorteado -> consultas concentradas manufaturam lei

SEMENTE PRE-REGISTRADA: 20260910

MINHA CALIBRAGEM: sessenta e seis previsoes, VINTE E CINCO certas, e o prior do
projeto sobre este tema exato e negativo.

PREVISAO: A2, empate. O prior e medido e eu nao tenho motivo forte para apostar
contra ele — o mecanismo da subamostragem e plausivel, mas um comite por bootstrap de
3 a 4 pares tende a discordar em ruido tanto quanto em estrutura. Se der A1, e a
primeira evidencia de que agencia paga aqui.
"""

from __future__ import annotations

import argparse
import random
import time

from novaphys.arc import dims
from novaphys import descricao as D

SEMENTE = 20260910
L, P, M = 10, 5, 8
N_INI, N_ESCOLHE, N_CAND, N_COMITE, N_TESTE = 3, 2, 10, 3, 2


def cena(rng, modo):
    g = [[0] * L for _ in range(L)]
    cols = [1, 4, 7]
    rng.shuffle(cols)
    parts = []
    for c in cols[:rng.randint(2, 3)]:
        r = rng.randint(1, 6)
        lado = rng.choice((-1, 0, 1))
        g[r][c] = P
        if lado != 0:
            g[r][c + lado] = M
        parts.append((r, c, lado))
    a = tuple(tuple(l) for l in g)
    b = [[x if x == M else 0 for x in lin] for lin in a]
    for r, c, lado in parts:
        dx = lado if modo == "lente" else rng.choice((-1, 0, 1))
        b[r + 1][c + dx] = P
    return a, tuple(tuple(l) for l in b)


def _le_massa(p):
    return any(not D._marca(x) and str(M) in repr(x[1]) for x in p)


def _comite(pares, rng):
    hips = []
    for _ in range(N_COMITE):
        amostra = [rng.choice(pares) for _ in range(len(pares))]
        try:
            fm, p, _e = D.montar(amostra, refoco=True, feixe=2)
            if p:
                hips.append((fm, p))
        except Exception:
            pass
    return hips


def _desacordo(hips, a):
    preds = []
    for fm, p in hips:
        try:
            q = D.aplicar((fm, p), a)
        except Exception:
            q = None
        if q is not None:
            preds.append(q)
    if len(preds) < 2:
        return 0
    h, w = dims(a)
    return sum(1 for i in range(h) for j in range(w)
               if len({q[i][j] for q in preds}) > 1)


def _episodio(args):
    modo, braco, semente = args
    # tres geradores separados: o comite consome aleatoriedade so no braco ativo, e
    # isso nao pode deslocar as cenas iniciais nem o teste entre os bracos
    rng = random.Random(semente)
    rc = random.Random(semente + 1)
    rt = random.Random(semente + 2)
    pares = [cena(rng, modo) for _ in range(N_INI)]
    for _ in range(N_ESCOLHE):
        if braco == "passivo":
            pares.append(cena(rng, modo))
        else:
            hips = _comite(pares, rc)
            cands = [cena(rng, modo) for _ in range(N_CAND)]
            pares.append(max(cands, key=lambda ab: _desacordo(hips, ab[0])))
    te = [cena(rt, modo) for _ in range(N_TESTE)]
    try:
        fm, p, e = D.montar(pares, refoco=True, feixe=4)
        if not p or e != 0:
            return modo, braco, 0, 0, 0
        teste = all(D.aplicar((fm, p), x) == y for x, y in te)
        return modo, braco, 1, int(teste), int(teste and _le_massa(p))
    except Exception:
        return modo, braco, 0, 0, 0


def _ini():
    D.usar_biblioteca(None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--episodios", type=int, default=10)
    a = ap.parse_args()

    print("=" * 76)
    print("LENTE ATIVA LIMPA — escolher o experimento cura a subamostragem?")
    print("=" * 76)
    print(f"  orcamento casado: {N_INI + N_ESCOLHE} observacoes de treino; "
          f"comite {N_COMITE}; {N_CAND} candidatas por escolha")
    # a MESMA semente por episodio nos dois bracos: as 3 cenas iniciais sao iguais
    trab = [(m, b, SEMENTE + 101 * k + (0 if m == "lente" else 50000))
            for m in ("lente", "sorteado")
            for b in ("passivo", "ativo")
            for k in range(a.episodios)]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_episodio, trab, chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo      braco     fecha   acerta teste   DESCOBERTA")
    print("  --------   -------   -----   ------------   ----------")
    tab = {}
    n = a.episodios
    for m in ("lente", "sorteado"):
        for b in ("passivo", "ativo"):
            rs = [r for r in res if r[0] == m and r[1] == b]
            f, t, d = (sum(r[k] for r in rs) for k in (2, 3, 4))
            tab[(m, b)] = d
            print(f"  {m:9s}  {b:8s}  {f:2d}/{n:<3d}  {t:5d}/{n:<6d}   {d:4d}/{n}")

    print("")
    print("=" * 76)
    lp, la = tab[("lente", "passivo")], tab[("lente", "ativo")]
    sp, sa = tab[("sorteado", "passivo")], tab[("sorteado", "ativo")]
    if not (0.3 * n <= lp <= 0.6 * n):
        print(f"A0 . passivo deu {lp}/{n}, fora da faixa esperada (30-60%).")
    if sa > sp + 1:
        print("A3 . consultas concentradas MANUFATURAM lei no mundo sem lei.")
    if la >= lp + 2 and sa <= 1:
        print("A1 . escolher o experimento cura subamostragem com modelo")
        print("     APRENDIDO — refina o prior do projeto.")
    else:
        print("A2 . empate: o prior 'desenho de experimento so paga com modelo")
        print("     dado' se mantem.")


if __name__ == "__main__":
    main()
