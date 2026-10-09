"""
novaphys.gradiente — dar GRADIENTE a admissao de operador
============================================================

A PAREDE, MEDIDA DOS DOIS LADOS
---------------------------------
[[o-criterio-nao-seleciona]]: os vocabularios gerados nunca entram em pilha
vencedora, e perdem em BITS. [[paisagem-e-penhasco]]: trocar UM marcador de
pilha exata salta o erro de 0 para mediana 46 em 96% dos casos, sem valores
intermediarios. E [[segunda-via-nulo]] fechou o outro lado: trocar a FONTE DE
EVIDENCIA (corroboracao entre mundos em vez de bits) nao ajuda, rho parcial
+0,15. O problema nao e evidencia. E a paisagem.

O PRECEDENTE, E E O UNICO QUE ESTE PROJETO TEM CONTRA ESSA PAREDE
-------------------------------------------------------------------
[[partir-adivinhado-resolvido]]: no Strassen o verificador era BINARIO (o
algoritmo calcula o produto exato? sim/nao) e nao havia por onde subir. A saida
nao foi mais busca nem mais evidencia — foi REESCREVER o verificador como
residuo de minimos quadrados, `R(s) = min_θ Σ L(P_{s,θ}(x), y)`, continuo nos
parametros. O recozimento passou a ter ladeira.

DE ONDE VEM O PENHASCO AQUI, EXATAMENTE
-----------------------------------------
[[primeiro-a-escrever-bloqueia]] mediu a causa: a reivindicacao de celula e
WINNER-TAKE-ALL. Cada celula pertence a UM codec, por `setdefault`, e 100% das
celulas erradas ja estao reivindicadas (0,0% livres). Entao trocar um codec
muda quem reivindica o que, cascateia pela pilha inteira, e o erro salta. A
descontinuidade nao esta no criterio: esta na ATRIBUICAO.

A REFORMULACAO
----------------
Amolecer a reivindicacao. Cada codec vota em cada celula com um peso `w_c`; a
celula vira uma DISTRIBUICAO sobre cores por softmax dos votos; a perda e
log-verossimilhanca. Entao

    R(conjunto) = min_w  L(w)

e continua e derivavel em w, e a mesma forma de projecao variavel da bilinear.
Admitir um operador deixa de ser "ele entra na pilha vencedora ou nao" e passa
a ser "a derivada da perda em relacao ao peso dele, em w_c = 0, e negativa?".

A derivada sai fechada. Com S[i][j][v] = Σ_c w_c·1[c preve v em (i,j)] e
p = softmax(S):

    dL/dw_c = - Σ_celulas ( 1[c acerta] - p[o que c previu] )

Isto e o gradiente que hoje nao existe: em w_c = 0 ele ja diz se vale admitir,
sem o operador precisar antes vencer uma pilha inteira.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  G0  PRE-CONDICAO. A forma mole REPRODUZ a dura no limite: nas tarefas que o
      nucleo resolve, o argmax da distribuicao ajustada bate a saida correta em
      >= 90% das celulas. Se a versao mole nao reproduz o que ja funciona, ela e
      outro sistema e nada que ela diga transfere para o nucleo.
  G1  O GRADIENTE EXISTE E DISCRIMINA: a derivada em w_c = 0 e mais negativa
      para codecs que ESTAO na pilha vencedora da tarefa do que para o PLACEBO,
      com separacao >= 2 desvios do erro pareado
      -> ha ladeira, e ela aponta para os uteis. Admissao por descida passa a
         ser possivel, e o canal de importacao deixa de depender de mim.
  G2  gradiente existe mas NAO discrimina (separacao < 2 desvios) -> amolecer
      criou ladeira que nao carrega o sinal; pior que o penhasco, porque agora
      ha uma direcao enganosa para descer.
  G3  gradiente ~ 0 para tudo -> amolecer nao consertou a descontinuidade.

  PLACEBO, e ele e o controle que decide: um codec com a MESMA cobertura de
  celulas do codec real, mas com os VALORES embaralhados. Isola "preve as
  celulas certas" de "preve os valores certos". Sem ele, qualquer codec que
  cubra muita celula teria gradiente grande e eu leria isso como sinal.

PREVISAO: G1 a 0,55. Mecanismo: a derivada e literalmente "quantas celulas esse
codec acerta que a mistura atual erra", e um codec da pilha vencedora acerta por
definicao. Risco nomeado: `copia` cobre a grade inteira e domina o gradiente em
toda tarefa, entao a separacao pode aparecer por COBERTURA e nao por acerto — e
por isso o placebo tem a mesma cobertura. Se mesmo assim `copia` dominar, o
numero sai por tarefa normalizado pela cobertura. G2 a 0,25, G3 a 0,20.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, dims, montar
from novaphys.promessa import PASTA

N_CORES = 10
N_TAREFAS = 40
PASSOS = 60
ETA = 0.5


def _previsoes(treino, max_modelos: int = 4
               ) -> List[Tuple[str, List[Dict[Tuple[int, int], int]]]]:
    """
    Para cada codec, a previsao por celula em cada par de treino.

    Um codec entra UMA vez, com o melhor modelo que ele propoe — o mesmo
    `propor` do nucleo, sem adaptacao.
    """
    saida = []
    for c in CODECS:
        try:
            ms = list(c.propor(treino) or [])[:max_modelos]
        except Exception:
            continue
        melhor, npred = None, -1
        for m in ms:
            try:
                ps = [c.prever(m, a) for a, _b in treino]
            except Exception:
                continue
            n = sum(1 for p, (_a, b) in zip(ps, treino)
                    for (i, j), v in p.items()
                    if i < dims(b)[0] and j < dims(b)[1] and v == b[i][j])
            if n > npred:
                melhor, npred = ps, n
        if melhor is not None:
            saida.append((c.nome, melhor))
    return saida


def _placebo(ps, rnd) -> List[Dict[Tuple[int, int], int]]:
    """MESMA cobertura de celulas, VALORES embaralhados."""
    out = []
    for p in ps:
        vs = list(p.values())
        rnd.shuffle(vs)
        out.append({k: v for k, v in zip(p.keys(), vs)})
    return out


def _monta_tensores(nomes_ps, treino):
    """
    voto[c][t][i][j] = cor que o codec c preve, ou -1 se ele nao preve ali.
    alvo[t][i][j]    = cor certa.
    """
    alvos, votos = [], []
    for _a, b in treino:
        h, w = dims(b)
        alvos.append(np.array(b, dtype=int))
    for _nome, ps in nomes_ps:
        vc = []
        for p, (_a, b) in zip(ps, treino):
            h, w = dims(b)
            m = -np.ones((h, w), dtype=int)
            for (i, j), v in p.items():
                if 0 <= i < h and 0 <= j < w and 0 <= v < N_CORES:
                    m[i, j] = v
            vc.append(m)
        votos.append(vc)
    return votos, alvos


def _perda_e_grad(votos, alvos, w):
    """
    Perda de log-verossimilhanca da mistura mole, e o gradiente em cada w_c.

        S[v] = Σ_c w_c · 1[c preve v]      p = softmax(S)
        L    = - media de log p[correto]
        dL/dw_c = - Σ ( 1[c acerta] - p[previsto por c] ) / N
    """
    C = len(votos)
    L, N = 0.0, 0
    g = np.zeros(C)
    for t, alvo in enumerate(alvos):
        h, w_ = alvo.shape
        S = np.zeros((h, w_, N_CORES))
        for c in range(C):
            m = votos[c][t]
            ok = m >= 0
            if ok.any():
                ii, jj = np.nonzero(ok)
                S[ii, jj, m[ii, jj]] += w[c]
        S -= S.max(axis=2, keepdims=True)
        E = np.exp(S)
        P = E / E.sum(axis=2, keepdims=True)
        ii, jj = np.meshgrid(np.arange(h), np.arange(w_), indexing="ij")
        pc = P[ii, jj, alvo]
        L += -np.log(np.clip(pc, 1e-12, None)).sum()
        N += h * w_
        for c in range(C):
            m = votos[c][t]
            ok = m >= 0
            if not ok.any():
                continue
            a, b_ = np.nonzero(ok)
            prev = m[a, b_]
            acerta = (prev == alvo[a, b_]).astype(float)
            g[c] -= (acerta - P[a, b_, prev]).sum()
    return L / max(N, 1), g / max(N, 1)


def ajusta(votos, alvos, passos: int = PASSOS, eta: float = ETA,
           fixos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Descida simples nos pesos. `fixos` marca pesos presos em zero."""
    w = np.zeros(len(votos))
    L = None
    for _ in range(passos):
        L, g = _perda_e_grad(votos, alvos, w)
        if fixos is not None:
            g = g * (~fixos)
        w = np.maximum(0.0, w - eta * g)   # peso nao-negativo: votar ou calar
    return w, L


def experimento(n_tarefas: int = N_TAREFAS, semente: int = 20260923) -> Dict:
    rnd = random.Random(semente)
    tarefas = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    resolvidas = (json.load(open("resultados/oficial.json"))
                  if os.path.exists("resultados/oficial.json") else {})
    pilhas = {}
    if os.path.exists("resultados/loja_fragmentos.json"):
        for it in json.load(open("resultados/loja_fragmentos.json")):
            pilhas[it["procedencia"]] = set(
                e for e in it["estrutura"] if isinstance(e, str))
    nomes = [n for n in sorted(pilhas) if n in tarefas][:n_tarefas]

    print("=" * 78)
    print("GRADIENTE — a admissao de operador passa a ter ladeira?")
    print("=" * 78)
    print("  %d tarefas resolvidas, com a pilha vencedora conhecida em cache\n"
          % len(nomes))

    g0_ok, g0_tot = 0, 0
    grad_na_pilha, grad_fora, grad_placebo = [], [], []
    por_tarefa = []
    # RISCO PRE-REGISTRADO: `copia` cobre a grade inteira e pode dominar o
    # gradiente por COBERTURA e nao por acerto. Guardo por codec e por
    # cobertura para poder checar em vez de torcer.
    por_codec = defaultdict(list)
    cobertura = defaultdict(list)

    for k, nome in enumerate(nomes):
        tr, _te = tarefas[nome]
        try:
            fms = forma.propor(tr)
            fm = fms[0] if fms else None
            reb = [(forma.base(fm, a), b) for a, b in tr] if fm else tr
            if any(g is None or dims(g) != dims(b) for g, b in reb):
                reb = tr
        except Exception:
            reb = tr
        try:
            nps = _previsoes(reb)
        except Exception:
            continue
        if len(nps) < 3:
            continue
        nps = nps + [("PLACEBO", _placebo(nps[0][1], rnd))]
        votos, alvos = _monta_tensores(nps, reb)

        # ---- G0: a forma mole reproduz a dura?
        w, _L = ajusta(votos, alvos)
        for t, alvo in enumerate(alvos):
            h, w_ = alvo.shape
            S = np.zeros((h, w_, N_CORES))
            for c in range(len(votos)):
                m = votos[c][t]
                ok = m >= 0
                if ok.any():
                    ii, jj = np.nonzero(ok)
                    S[ii, jj, m[ii, jj]] += w[c]
            g0_ok += int((S.argmax(axis=2) == alvo).sum())
            g0_tot += h * w_

        # ---- G1: derivada em w_c = 0, com os outros no otimo
        venc = pilhas.get(nome, set())
        for ci, (cn, _ps) in enumerate(nps):
            fix = np.zeros(len(nps), dtype=bool)
            fix[ci] = True
            w2, _ = ajusta(votos, alvos, fixos=fix)
            w2[ci] = 0.0
            _L2, g2 = _perda_e_grad(votos, alvos, w2)
            d = float(g2[ci])
            cob = sum(int((m >= 0).sum()) for m in votos[ci]) / max(
                sum(int(a.size) for a in alvos), 1)
            por_codec[cn].append(d)
            cobertura[cn].append(cob)
            if cn == "PLACEBO":
                grad_placebo.append(d)
                por_tarefa.append([nome, None, d])
            elif cn in venc:
                grad_na_pilha.append(d)
                if por_tarefa and por_tarefa[-1][0] == nome:
                    pass
                por_tarefa.append([nome, d, None])
            else:
                grad_fora.append(d)
        if (k + 1) % 10 == 0:
            print("    %2d/%d tarefas" % (k + 1, len(nomes)), flush=True)

    rep = g0_ok / max(g0_tot, 1)
    print("\n  G0 a forma MOLE reproduz a dura: %.1f%% das celulas"
          % (100 * rep))
    if rep < 0.90:
        print("  G0 FALHOU: a versao mole nao reproduz o que ja funciona.")
        print("     Ela e outro sistema; nada que ela diga transfere.")
        return {"veredito": "G0 falhou", "reproducao": rep}
    print("     . reproduz; segue.\n")

    def _r(v):
        return (statistics.mean(v), statistics.pstdev(v) / (len(v) ** 0.5)) \
            if v else (float("nan"), float("nan"))
    mp, sp = _r(grad_na_pilha)
    mf, sf = _r(grad_fora)
    mb, sb = _r(grad_placebo)
    print("  derivada dL/dw em w=0 (mais NEGATIVA = mais vale admitir)")
    print("    na pilha vencedora  %+.5f +- %.5f  (n=%d)" % (mp, sp, len(grad_na_pilha)))
    print("    fora da pilha       %+.5f +- %.5f  (n=%d)" % (mf, sf, len(grad_fora)))
    print("    PLACEBO             %+.5f +- %.5f  (n=%d)" % (mb, sb, len(grad_placebo)))

    # separacao pareada por tarefa entre 'na pilha' e placebo
    porn = defaultdict(lambda: {"p": [], "b": []})
    for nome, dp, db in por_tarefa:
        if dp is not None:
            porn[nome]["p"].append(dp)
        if db is not None:
            porn[nome]["b"].append(db)
    ds = [statistics.mean(v["b"]) - statistics.mean(v["p"])
          for v in porn.values() if v["p"] and v["b"]]
    md = statistics.mean(ds) if ds else float("nan")
    sd = (statistics.pstdev(ds) / (len(ds) ** 0.5)) if len(ds) > 1 else float("nan")
    sep = md / sd if sd and sd == sd and sd > 0 else 0.0
    print("\n  separacao PAREADA por tarefa (placebo - na pilha): %+.5f +- %.5f"
          % (md, sd))
    print("  = %.1f desvios  (n=%d tarefas)" % (sep, len(ds)))

    print("\n" + "=" * 78)
    todos = grad_na_pilha + grad_fora + grad_placebo
    if todos and max(abs(x) for x in todos) < 1e-6:
        ver = "G3"
        print("G3 . gradiente ~ 0 para tudo: amolecer nao consertou a")
        print("     descontinuidade.")
    elif sep >= 2.0:
        ver = "G1"
        print("G1 . HA LADEIRA E ELA APONTA PARA OS UTEIS (%.1f desvios)." % sep)
        print("     A derivada em w=0 ja diz se vale admitir, sem o operador")
        print("     precisar antes vencer uma pilha inteira. O canal de")
        print("     importacao deixa de depender so de mim escrever codec.")
    else:
        ver = "G2"
        print("G2 . ha ladeira mas ela NAO carrega o sinal (%.1f desvios)." % sep)
        print("     Pior que o penhasco: agora existe direcao enganosa.")
    return {"veredito": ver, "reproducao": rep, "separacao_desvios": sep,
            "na_pilha": [mp, sp], "fora": [mf, sf], "placebo": [mb, sb],
            "n_tarefas": len(ds)}


if __name__ == "__main__":
    r = experimento()
    json.dump(r, open("resultados/gradiente.json", "w"), indent=1,
              ensure_ascii=False, default=str)
