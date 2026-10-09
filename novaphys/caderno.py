"""
novaphys.caderno — o caderno vivo: acumular, esquecer, voltar ao exemplo
==========================================================================

A PERGUNTA DO USUARIO
-----------------------
"Se ele nao acumula aprendizados, erros, experiencias, conceitos, como diabos vai
conseguir evoluir?" E: o sistema deveria ter plasticidade real?

O DADO QUE JA TEMOS
---------------------
No ARC, a biblioteca acumulada deu ZERO (400 mundos diferentes; o que se aprende
num quase nunca volta). Na fisica, o conceito inventado (q = h) PAGOU na
descoberta seguinte. Hipotese: acumular so paga num mundo ESTACIONARIO — um
universo so, onde os conceitos voltam. Humanos acumulam porque vivem num mundo so.

O MUNDO: UMA SEQUENCIA DE 40 TAREFAS NO MESMO UNIVERSO
--------------------------------------------------------
  tarefas de LABORATORIO (36): poucos corpos, aceleracao medida com ruido; o
      agente escreve a lei (sem h no caderno — tem que inventar ou herdar)
  tarefas de TRAJETORIA (4, nas posicoes 5, 15, 25, 35): o agente observa orbitas
      e pode inventar uma grandeza conservada (o mecanismo de `inventa.py`)
  TROCA DE LEI na tarefa 20: o universo passa de einstein (-A h^2/r^4) para
      oblato (-K/r^4). Teste de PLASTICIDADE: o que foi aprendido fica velho.

A MEMORIA (quatro regras, as tres primeiras do usuario)
---------------------------------------------------------
  1. CONCEITOS: grandezas inventadas entram no caderno de variaveis.
  2. CUSTO APRENDIDO PELO USO: cada termo de lei ja aceito vira item da memoria.
     Prior de "restaurante chines": usar o item i custa log2((T + B)/c_i); uma
     ideia NOVA custa log2((T + B)/B) + ~20 bits da gramatica  (T = soma dos usos,
     B = 1). Ideia que ja pagou fica barata de pensar — e a propria regua de bits:
     comprimento de codigo = -log2 probabilidade. (Primeira versao, corrigida antes
     de rodar: frequencia RELATIVA entre itens; com um item so, esquecer nao mudava
     nada.)
  3. ESQUECIMENTO: a cada tarefa, uso *= gama. O que nao paga, some.
  4. VOLTAR AO EXEMPLO: toda tarefa de laboratorio fica guardada; quando um
     conceito NOVO entra, as tarefas guardadas do mundo atual sao REFEITAS.

QUATRO BRACOS, MESMOS DADOS (pareados por tarefa)
---------------------------------------------------
  sem       cada tarefa do zero, sem conceito, sem memoria
  plastica  memoria com gama = 0,7
  rigida    memoria com gama = 1 (nunca esquece)
  placebo   memoria com o MESMO numero de itens e a mesma dinamica de uso, mas o
            conteudo e errado: conceito p = x vx + y vy (nao conservado) e, a
            cada termo aceito, guarda um termo SORTEADO da gramatica

Sucesso numa tarefa = a lei escrita EXTRAPOLA (R2 >= 0,98 em r de 3 a 6) para a
lei verdadeira do mundo daquele momento.

PROTOCOLO EM DUAS ETAPAS
--------------------------
Etapa 1 (calibracao, semente 20260918, declarada): achar o numero de corpos por
tarefa N em que o agente SEM memoria acerta cerca de metade — onde memoria tem
espaco para mostrar algo. Etapa 2 (a medida, semente 20260919): pre-registro abaixo,
escrito DEPOIS da calibracao e ANTES da medida.

CALIBRACAO (feita, semente 20260918, 10 sementes, agente SEM memoria)
----------------------------------------------------------------------
    N    einstein sem conceito   einstein com h dado   oblato
    12          0/10                   1/10             4/10
    15          0/10                   0/10              -
    16          0/10                   3/10             6/10
    20          0/10                   5/10             4/10
    25          0/10                   7/10             9/10
    40          0/10                   9/10              -
    80          0/10                  10/10              -
Sem o conceito, einstein e 0/10 ate 80 corpos: o conceito vale > 10x em dado.
Escolho N = 20: com h fica em 5/10 e o oblato em 4/10 — espaco para a memoria
ajudar e para atrapalhar. Um quinto braco entrou: SO_CONCEITO (conceitos sim,
memoria de leis nao), para separar o efeito do conceito do efeito do custo
aprendido.

PRE-REGISTRO DA ETAPA 2 — semente 20260919, 10 correntes, N = 20
------------------------------------------------------------------
Fases (acerto na PRIMEIRA tentativa, somado nas 10 correntes):
    antes do conceito (0-4) | com conceito (6-19) | logo apos a troca (20-24)
    | adaptado (26-39)
  K1  CONCEITO: so_conceito - sem >= +30 pontos em "com conceito"
  K2  MEMORIA DE LEIS: min(plastica, rigida) - so_conceito >= +15 em "com conceito"
  K3  PLASTICIDADE: plastica - rigida >= +15 em "logo apos a troca"
  K4  PLACEBO: placebo - sem <= +5 em "com conceito"
  K5  VOLTAR AO EXEMPLO: nos bracos com conceito, das falhas em 0-4, >= 40%
      viram acerto quando o conceito chega e a tarefa e refeita

PREVISAO: K1 passa (~+50). K2 passa, com incerteza (+15 a +30). K3 FALHA —
aposto que com 20 corpos o dado decide mais que o custo e plastica ~ rigida.
K4 passa. K5 passa (~50%, que e o acerto com h a N = 20).

RESULTADO DA ETAPA 2 (170 s)
------------------------------
                         sem   so_conceito  plastica  rigida  placebo
  com conceito (6-19)     0%       37%         3%       2%      0%
  logo apos a troca      82%       84%        80%      78%     84%
  adaptado (26-39)       80%       80%        95%      94%     83%
  (a linha 0-4 saiu contaminada pelas revisitas — bug de relatorio; na primeira
   tentativa todos deram 0%. Revisita: so_conceito recuperou 19/50; plastica,
   rigida e placebo 0/50.)
K1 passou (+37). K4 passou. K3 falhou como previsto (+2). K2 FALHOU AO CONTRARIO:
memoria de leis -35 pontos. K5 falhou (19/150; so o so_conceito recuperou).

A LEITURA: ENTRINCHEIRAMENTO. Nas tarefas 0-4, antes do conceito, a memoria
aprendeu a lei ERRADA (o epiciclo v^2 r^-2). Ela ficou barata e passou a vencer
sempre — inclusive depois de q chegar, e inclusive na revisita. Esquecer por
desuso nao ajuda: a ideia errada nunca fica em desuso, porque vence. Efeito
Einstellung (Luchins 1942); paradigma de Kuhn. E depois da troca de lei, onde a
PRIMEIRA lei aprendida estava certa, a memoria ajudou (+15). A memoria amplifica
o que se aprendeu primeiro — certo ou errado.

RODADA 2, PRE-REGISTRO — semente 20260921, mesmo N = 20, 10 correntes
------------------------------------------------------------------------
Braco novo REANALISE (a intuicao do usuario: guardar EXPERIENCIAS, nao so
conclusoes): quando um conceito entra, junta os DADOS de todas as tarefas
guardadas do mundo atual e redescobre a lei com olhos frescos (conceitos sim,
leis velhas nao); a memoria de leis e substituida pelo resultado. E como a ciencia
refaz a analise de todos os dados quando a teoria muda.
  R1  reanalise - so_conceito >= +15 em "com conceito (6-19)"  — a memoria passa
      a ajudar em vez de atrapalhar
  R2  reanalise - so_conceito >= +10 em "adaptado (26-39)"  — mantem o lado bom
  R3  REPLICA do entrincheiramento: max(plastica, rigida) - so_conceito <= -20
      em "com conceito"
  R4  a reanalise recupera >= 60% das falhas de 0-4
PREVISAO: R1 passa (100 medidas juntas com q dao a lei certa; ela vira a barata).
R2 passa. R3 passa (e o mesmo mecanismo, semente nova). R4 passa (~90%).

RESULTADO DA RODADA 2 (199 s):
                      sem  so_conc  plast  rigida  placebo  REANALISE
  com conceito (6-19)  0%    30%     21%    19%      0%       90%
  adaptado (26-39)    79%    79%     96%    95%      80%      96%
  revisita 0-4: reanalise 50/50; so_conceito 22/50; memorias 0/50
R1 +60, R2 +17, R4 50/50: passaram. R3 FALHOU (-9/-11; limiar <= -20): o
entrincheiramento replicou na direcao, com magnitude menor que os -35.

RODADA 3, PRE-REGISTRO — semente 20260926, bracos so_conceito, reanalise,
reanalise_suja
------------------------------------------------------------------------------
A reanalise mistura dois ingredientes: MAIS DADO (100 medidas juntas) e OLHOS
FRESCOS (sem as leis velhas). REANALISE_SUJA junta os dados mas analisa com as
leis velhas ainda baratas na memoria.
  F1  reanalise - reanalise_suja >= +15 em "com conceito"  -> olhos frescos importam
PREVISAO: F1 FALHA — com 100 medidas o ajuste vale centenas de bits e o custo
barato das leis velhas (~poucos bits) nao decide; o ingrediente e juntar os DADOS.

RESULTADO DA RODADA 3 (119 s):
                      so_conceito  reanalise  reanalise_suja
  com conceito (6-19)     37%         88%          52%
  logo apos a troca       80%         94%          94%
  adaptado (26-39)        79%         95%          95%
  revisita 0-4: reanalise 50/50, suja 25/50, so_conceito 16/50
F1 PASSOU (+36) — minha previsao ERROU. Os dois ingredientes contam: juntar os
dados vale +15 (37 -> 52), tirar as leis velhas da analise vale +36 (52 -> 88).
Mesmo com 5x mais dado, a lei velha barata desvia a analise conjunta em metade
das vezes: ela mais uma correcao ajustam os dados dentro da faixa (epiciclo), e a
validacao fora da amostra — que e dentro da faixa — nao pega. Principio de Planck
("a ciencia avanca um funeral de cada vez"), em versao de maquina: quem aceita a
teoria nova e a analise que nao carrega a velha.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random
import time
from math import log2

import numpy as np

from novaphys import orbita as O
from novaphys import orbita2 as O2
from novaphys import orbita3 as O3
from novaphys import inventa as I

SEMENTE_CAL = 20260918
SEMENTE = 20260919
N_TAREFAS = 40
TROCA = 20
TRAJ = (5, 15, 25, 35)
BRACOS = ("sem", "so_conceito", "plastica", "rigida", "placebo", "reanalise")
GAMA = {"so_conceito": 1.0, "plastica": 0.7, "rigida": 1.0, "placebo": 0.7,
        "reanalise": 0.7, "reanalise_suja": 0.7}
N_LAB = 20            # escolhido pela calibracao (ver docstring)
S3 = O2.estados_ext(np.random.default_rng(O3.SEMENTE + 6), 2000)


MUNDOS = ("einstein", "oblato")     # a `generalidade.py` troca o primeiro


def mundo_de(k):
    return MUNDOS[0] if k < TROCA else MUNDOS[1]


# ─── a memoria ───────────────────────────────────────────────────────────────

BETA = 1.0


class Memoria:
    """Prior de 'restaurante chines': usar o item i custa log2((T+B)/c_i); uma
    ideia NOVA custa log2((T+B)/B) + os bits da gramatica. Com o esquecimento,
    T cai: o velho encarece e o novo barateia. (A primeira versao usava a
    frequencia RELATIVA entre itens — com um item so, esquecer nao mudava nada.)"""

    def __init__(self, gama):
        self.gama = gama
        self.conceitos = {}
        self.termos = {}

    def custo(self, chave):
        tot = sum(self.termos.values())
        return log2((tot + BETA) / max(self.termos[chave], 1e-12))

    def custo_novo(self):
        return log2((sum(self.termos.values()) + BETA) / BETA)

    def envelhece(self):
        for c in self.termos:
            self.termos[c] *= self.gama

    def usa(self, chave):
        self.termos[chave] = self.termos.get(chave, 0.0) + 1.0


def _com_extras(mem):
    O.EXTRAS.clear()
    if mem is not None:
        O.EXTRAS.update(mem.conceitos)


def _lib(mem, X, nomes, dirs):
    if mem is None:
        return []
    idx = {n: j for j, n in enumerate(nomes)}
    out = []
    for chave in mem.termos:
        d, vs, es = chave
        if any(v not in idx for v in vs):
            continue
        e = np.zeros(len(nomes))
        for v, x in zip(vs, es):
            e[idx[v]] = x
        out.append((chave, O._F(e[None, :], X)[0][:, None] * dirs[d],
                    mem.custo(chave)))
    return out


def _escolhe(D, bases, idx, S, X, dirs, P, E, lib, novo=0.0):
    """O procedimento de escolha (como orbita3), com a MEMORIA competindo.
    `novo` = custo em bits de uma ideia que nao esta na memoria."""
    n = len(idx) * 2
    R = O3._resid(D, bases, idx, idx)
    sse = (R ** 2).sum()
    Qo = O2._ortonormal([b[idx] for b in bases])
    g, i, d = O2._top_termos(R, Qo, X[:, idx], {k: u[idx] for k, u in dirs.items()},
                             P, n)[0]
    melhor = (g - novo, ("gram", d, i), None)
    for chave, b, custo in lib:
        v = b[idx].copy()
        for q in Qo:
            v = v - (v * q).sum() * q
        den = (v ** 2).sum()
        if not np.isfinite(den) or den <= 1e-12:
            continue
        red = min((R * v).sum() ** 2 / den, sse * (1 - 1e-12))
        gl = 0.5 * n * log2(sse / (sse - red)) - custo - 0.5 * log2(n)
        if gl > melhor[0]:
            melhor = (gl, ("lib", chave), b)
    m, p = O2._fit_obj(R, S[idx])
    red = sse - ((R - m * O._obj_base(S[idx], p)) ** 2).sum()
    go = O.ganho(sse, red, n, O.bits_objeto(n)) - novo
    if go > melhor[0]:
        melhor = (go, ("objeto", p), O._obj_base(S, p))
    if melhor[0] <= 0:
        return None
    if melhor[1][0] == "gram":
        _, d, i = melhor[1]
        return melhor[0], melhor[1], O._F(E[i:i + 1], X)[0][:, None] * dirs[d]
    return melhor


def descobre(S, D, mem, semente):
    _com_extras(mem)
    nomes, X = O.caderno(S, False)
    P = len(nomes)
    E, meta = O.candidatos(P)
    dirs = O.direcoes(S)
    lib = _lib(mem, X, nomes, dirs)
    novo = mem.custo_novo() if mem is not None else 0.0
    neff = D.size
    perm = np.random.default_rng(semente).permutation(len(S))
    A, B = perm[: len(S) // 2], perm[len(S) // 2:]
    tudo = np.arange(len(S))
    estr, bases = [], []
    for _ in range(O.MAX_TERMOS):
        antes = depois = 0.0
        for tr, te in ((A, B), (B, A)):
            r0 = O3._resid(D, bases, tr, te)
            antes += (r0 ** 2).sum()
            esc = _escolhe(D, bases, tr, S, X, dirs, P, E, lib, novo)
            r1 = O3._resid(D, bases + [esc[2]], tr, te) if esc else r0
            depois += (r1 ** 2).sum()
        if 0.5 * neff * log2(antes / max(depois, 1e-300)) <= 0:
            break
        esc = _escolhe(D, bases, tudo, S, X, dirs, P, E, lib, novo)
        if esc is None:
            break
        estr.append(esc[1])
        bases.append(esc[2])
    aceitos, chaves = [], []
    if estr:
        coefs = O2._ajusta(bases, D, tudo)
        for e, c in zip(estr, coefs):
            if e[0] == "objeto":
                aceitos.append(("objeto", float(c), e[1][0], e[1][1], 0.0))
                continue
            if e[0] == "gram":
                _k, vs, es = meta[e[2]]
                chave = (e[1], tuple(nomes[j] for j in vs), tuple(int(x) for x in es))
            else:
                chave = e[1]
            aceitos.append(("lei", chave[0], chave[1], chave[2], float(c), 0.0))
            chaves.append(chave)
    return aceitos, chaves


def acerta(ac, mem, mundo):
    _com_extras(mem)
    try:
        return O2._r2(O.anomalia(mundo, S3), O.preve(ac, S3, False)) >= 0.98
    finally:
        O.EXTRAS.clear()


def _termo_sorteado(rng, nomes):
    k = rng.randint(1, 2)
    vs = tuple(sorted(rng.sample(nomes, k), key=nomes.index))
    return (rng.choice(("r^", "t^", "v^")), vs,
            tuple(rng.choice(O.EXPS) for _ in range(k)))


# ─── uma corrente de 40 tarefas, um braco ───────────────────────────────────

def corrente(args):
    braco, s, n_lab, semente = args
    O.N_OBS = n_lab
    mem = None if braco == "sem" else Memoria(GAMA[braco])
    rsort = random.Random(semente + 99 + 1000 * s)
    guardadas = []          # (k, S, D, semente_dobra)
    acertos = {}            # estado atual (muda com revisitas)
    primeiro = {}           # a PRIMEIRA tentativa, que nao muda
    revisita = []           # (k, antes, depois)
    for k in range(N_TAREFAS):
        mundo = mundo_de(k)
        base = semente + 1000 * s + k
        if k in TRAJ:
            if mem is None:
                continue
            rng = np.random.default_rng(base + 500)
            Ttr = I.trajetorias(mundo, rng, I.N_TRAJ)
            Tho = I.trajetorias(mundo, rng, I.N_TRAJ)
            aceito, top, _n = I.inventa(Ttr, Tho)
            novo = False
            if braco == "placebo":
                if "p" not in mem.conceitos:
                    mem.conceitos["p"] = lambda S: S[..., 0] * S[..., 2] + S[..., 1] * S[..., 3]
                    novo = True
            elif aceito:
                _g, s1, w1, *_ = top[0]
                Q = I.grandeza(s1, w1, Ttr)
                Sh = O.estados(np.random.default_rng(base + 7), 500)
                dup = any(abs(np.corrcoef(Q(Sh), f(Sh))[0, 1]) > 0.99
                          for f in mem.conceitos.values())
                if not dup:
                    mem.conceitos[f"q{len(mem.conceitos)}"] = Q
                    novo = True
            if novo and braco in ("reanalise", "reanalise_suja"):
                # REANALISE: junta os DADOS de todas as tarefas guardadas do mundo
                # atual e redescobre a lei com olhos frescos (conceitos sim, leis
                # velhas nao); a memoria de leis e SUBSTITUIDA pelo resultado.
                # REANALISE_SUJA (rodada 3): junta os dados MAS analisa com as leis
                # velhas ainda baratas na memoria — separa "mais dado" de "olhos
                # frescos".
                doms = [g for g in guardadas if mundo_de(g[0]) == mundo]
                if doms:
                    Sa = np.concatenate([g[1] for g in doms])
                    Da = np.concatenate([g[2] for g in doms])
                    limpa = Memoria(mem.gama)
                    limpa.conceitos = mem.conceitos
                    olhos = limpa if braco == "reanalise" else mem
                    ac, chs = descobre(Sa, Da, olhos, base + 11)
                    mem.termos = {ch: float(len(doms)) for ch in chs}
                    ok = acerta(ac, mem, mundo)
                    for kk, *_r in doms:
                        revisita.append((kk, acertos[kk], ok))
                        acertos[kk] = ok
            elif novo:      # VOLTAR AO EXEMPLO: refaz as tarefas do mundo atual
                for kk, S, D, sd in guardadas:
                    if mundo_de(kk) != mundo:
                        continue
                    ac, _ch = descobre(S, D, mem, sd)
                    novo_ok = acerta(ac, mem, mundo)
                    revisita.append((kk, acertos[kk], novo_ok))
                    acertos[kk] = novo_ok
            continue
        rng = np.random.default_rng(base)
        S = O.estados(rng, n_lab)
        D = O.mede(mundo, S, rng)
        ac, chaves = descobre(S, D, mem, base + 3)
        acertos[k] = acerta(ac, mem, mundo)
        primeiro[k] = acertos[k]
        guardadas.append((k, S, D, base + 3))
        if mem is not None and braco != "so_conceito":   # so_conceito: sem leis
            mem.envelhece()
            _com_extras(mem)
            nomes, _X = O.caderno(S, False)
            O.EXTRAS.clear()
            for ch in chaves:
                mem.usa(_termo_sorteado(rsort, nomes) if braco == "placebo" else ch)
    # bug da rodada 1: devolvia `acertos`, que as revisitas sobrescrevem — a linha
    # "antes do conceito (0-4)" mostrava o resultado DEPOIS da revisita.
    return braco, s, primeiro, revisita


# ─── calibracao ──────────────────────────────────────────────────────────────

def calibra(n_sementes, ns):
    # rodada 1 (15/25/40/80): einstein sem conceito 0/10 em TODOS os N; com h dado
    # 0, 7, 9, 10/10. O conceito vale mais de 10x em dado. Rodada 2 refina N e
    # calibra o mundo oblato (o da segunda metade da corrente).
    print("CALIBRACAO (semente 20260918): acerto SEM memoria")
    for n in ns:
        O.N_OBS = n
        ok_sem = ok_h = ok_ob = 0
        for k in range(n_sementes):
            rng = np.random.default_rng(SEMENTE_CAL + 1000 * k + n)
            S = O.estados(rng, n)
            D = O.mede("einstein", S, rng)
            ac, _ = descobre(S, D, None, SEMENTE_CAL + k)
            ok_sem += acerta(ac, None, "einstein")
            ac2 = O3.descobre(S, D, True, np.random.default_rng(SEMENTE_CAL + k), "gauss")
            ok_h += O2._r2(O.anomalia("einstein", S3), O.preve(ac2, S3, True)) >= 0.98
            rng = np.random.default_rng(SEMENTE_CAL + 1000 * k + n + 1)
            S = O.estados(rng, n)
            D = O.mede("oblato", S, rng)
            ac, _ = descobre(S, D, None, SEMENTE_CAL + k)
            ok_ob += acerta(ac, None, "oblato")
        print(f"  N={n:3d}   einstein sem conceito {ok_sem:2d}/{n_sementes}   "
              f"einstein com h dado {ok_h:2d}/{n_sementes}   "
              f"oblato {ok_ob:2d}/{n_sementes}")


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibra", action="store_true")
    ap.add_argument("--correntes", type=int, default=10)
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--n", type=int, default=N_LAB)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    ap.add_argument("--bracos", default=",".join(BRACOS))
    a = ap.parse_args()
    bracos = a.bracos.split(",")
    if a.calibra:
        calibra(10, (12, 16, 20, 25))
        return
    print("=" * 76)
    print(f"CADERNO VIVO — {a.correntes} correntes x {N_TAREFAS} tarefas, N={a.n}")
    print("=" * 76)
    t0 = time.time()
    print(f"  semente {a.semente}")
    trab = [(b, s, a.n, a.semente) for s in range(a.correntes) for b in bracos]
    with Pool(a.nucleos) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")

    def taxa(b, ks):
        v = [r[2][k] for r in res if r[0] == b for k in ks if k in r[2]]
        return sum(v), len(v)

    fases = {"antes do conceito (0-4)": range(0, 5),
             "com conceito (6-19)": [k for k in range(6, 20) if k not in TRAJ],
             "logo apos a troca (20-24)": range(20, 25),
             "adaptado (26-39)": [k for k in range(26, 40) if k not in TRAJ]}
    print("  acerto na PRIMEIRA tentativa (revisitas contam separado)")
    print("  " + " " * 28 + "  ".join(f"{b:>10s}" for b in bracos))
    tab = {}
    for nome, ks in fases.items():
        linha = []
        for b in bracos:
            x, t = taxa(b, ks)
            tab[(nome, b)] = x / max(t, 1)
            linha.append(f"{x:3d}/{t:<3d} {100*x/max(t,1):3.0f}%")
        print(f"  {nome:28s}" + "  ".join(f"{c:>10s}" for c in linha))
    print("\n  VOLTAR AO EXEMPLO (tarefas refeitas quando um conceito novo entrou):")
    for b in [x for x in bracos if x != "sem"]:
        rv = [x for r in res if r[0] == b for x in r[3]]
        ganhou = sum(1 for _k, a0, a1 in rv if not a0 and a1)
        perdeu = sum(1 for _k, a0, a1 in rv if a0 and not a1)
        falhas = sum(1 for _k, a0, _a1 in rv if not a0)
        print(f"    {b:11s} refeitas {len(rv):3d}   falha->acerto {ganhou:3d} de {falhas:3d}"
              f"   acerto->falha {perdeu:3d}")

    print("")
    print("=" * 76)
    cc, tr_ = "com conceito (6-19)", "logo apos a troca (20-24)"
    if "reanalise_suja" in bracos and "reanalise" in bracos:
        f1 = tab[(cc, "reanalise")] - tab[(cc, "reanalise_suja")]
        print(f"F1 {'.' if f1 >= 0.15 else 'FALHOU .'} olhos frescos: reanalise - "
              f"reanalise_suja em 'com conceito' = {100*f1:+.0f} (limiar +15)")
        for b in ("reanalise", "reanalise_suja"):
            rv = [x for r in res if r[0] == b for x in r[3] if x[0] < 5]
            f_ = sum(1 for _k, a0, _a1 in rv if not a0)
            g_ = sum(1 for _k, a0, a1 in rv if not a0 and a1)
            print(f"   {b}: recupera {g_}/{f_} falhas de 0-4")
    if not all(b in bracos for b in BRACOS):
        return
    k1 = tab[(cc, "so_conceito")] - tab[(cc, "sem")]
    print(f"K1 {'.' if k1 >= 0.30 else 'FALHOU .'} conceito: so_conceito - sem = "
          f"{100*k1:+.0f} pontos (limiar +30)")
    k2 = min(tab[(cc, "plastica")], tab[(cc, "rigida")]) - tab[(cc, "so_conceito")]
    print(f"K2 {'.' if k2 >= 0.15 else 'FALHOU .'} memoria de leis: min(plastica, "
          f"rigida) - so_conceito = {100*k2:+.0f} (limiar +15)")
    k3 = tab[(tr_, "plastica")] - tab[(tr_, "rigida")]
    print(f"K3 {'.' if k3 >= 0.15 else 'FALHOU .'} plasticidade: plastica - rigida "
          f"logo apos a troca = {100*k3:+.0f} (limiar +15)")
    k4 = tab[(cc, "placebo")] - tab[(cc, "sem")]
    print(f"K4 {'.' if k4 <= 0.05 else 'FALHOU .'} placebo - sem = {100*k4:+.0f} "
          f"(limiar <= +5)")
    rv = [x for r in res if r[0] in ("so_conceito", "plastica", "rigida")
          for x in r[3] if x[0] < 5]
    f5 = sum(1 for _k, a0, _a1 in rv if not a0)
    g5 = sum(1 for _k, a0, a1 in rv if not a0 and a1)
    print(f"K5 {'.' if f5 and g5 / f5 >= 0.40 else 'FALHOU .'} voltar ao exemplo: "
          f"{g5}/{f5} falhas de 0-4 viraram acerto (limiar 40%)")

    # rodada 2: a reanalise
    ad = "adaptado (26-39)"
    print("")
    r1 = tab[(cc, "reanalise")] - tab[(cc, "so_conceito")]
    print(f"R1 {'.' if r1 >= 0.15 else 'FALHOU .'} reanalise - so_conceito em 'com "
          f"conceito' = {100*r1:+.0f} (limiar +15)")
    r2 = tab[(ad, "reanalise")] - tab[(ad, "so_conceito")]
    print(f"R2 {'.' if r2 >= 0.10 else 'FALHOU .'} reanalise - so_conceito em "
          f"'adaptado' = {100*r2:+.0f} (limiar +10)")
    r3 = max(tab[(cc, "plastica")], tab[(cc, "rigida")]) - tab[(cc, "so_conceito")]
    print(f"R3 {'.' if r3 <= -0.20 else 'FALHOU .'} entrincheiramento replica: "
          f"max(plastica, rigida) - so_conceito = {100*r3:+.0f} (limiar <= -20)")
    rv = [x for r in res if r[0] == "reanalise" for x in r[3] if x[0] < 5]
    f6 = sum(1 for _k, a0, _a1 in rv if not a0)
    g6 = sum(1 for _k, a0, a1 in rv if not a0 and a1)
    print(f"R4 {'.' if f6 and g6 / f6 >= 0.60 else 'FALHOU .'} reanalise recupera "
          f"{g6}/{f6} falhas de 0-4 (limiar 60%)")


if __name__ == "__main__":
    main()
