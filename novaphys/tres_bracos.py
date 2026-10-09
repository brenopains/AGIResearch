"""
novaphys.tres_bracos — a eliminação de cor vale alguma coisa na busca?
========================================================================

DE ONDE VEM
-------------
`elimina_cor.py` fechou o portão de correção: o mínimo calculado bate a
enumeração exaustiva em 400/400 casos. A metade resolvível da pilha existe.
Falta a única pergunta que importa: **ela resolve mais tarefas?**

  A   busca ATUAL (enumera as cores como nós separados; ordena por (err, bits))
  B   mesma linguagem, cores ELIMINADAS; seleção só por fechamento exato e bits
  C   igual a B, mas a seleção usa o RESÍDUO MÍNIMO E(s) e bits

  B x C  isola o valor do resíduo GRADUADO
  A x C  mede utilidade contra o sistema existente

RESTRIÇÕES DECLARADAS, e elas encolhem o recorte
--------------------------------------------------
1. Codec de cor só entra em posição INICIAL DE ESTÁGIO. Dentro de um estágio o
   `pred.setdefault` faz a primeira reivindicação vencer, então um codec de cor
   depois de `copia` não reivindica nada. Manter a exatidão da contagem exige
   que o slot veja a máscara inteira.
2. Resolução GULOSA, não conjunta: cada slot é resolvido quando entra, contra o
   prefixo já fixo. Revisão externa pediu resolução conjunta; guloso é
   aproximação e está declarado como tal.
3. Um slot não resolvido por vez.

TAREFAS
---------
Escolhidas por ordenação de HASH fixada antes do teste, sem filtrar por já
resolvidas — senão o experimento herda o viés retrospectivo que já custou caro
aqui.

PRÉ-REGISTRO
--------------
  T0  PRÉ-CONDIÇÃO. Toda solução dos braços B e C, com a cor instanciada,
      REEXECUTA no executor original e fecha com igualdade exata. Divergência
      de um caso invalida o braço.
  T1  C resolve >= 3 tarefas a mais que A E >= 3 a mais que B, com saída de
      teste exata -> a eliminação condicional é útil neste recorte
  T2  C > B mas nao > A -> o resíduo graduado ajuda, e não o bastante para
      bater o sistema atual
  T3  C == B -> resolver os parâmetros ajuda ou não, mas o resíduo NÃO orientou
      melhor a estrutura
  T4  todos iguais -> a partição cobre pouco da dificuldade (13,8% dos itens de
      pilha vencedora tem parâmetro de cor, então esta é a saída provável)

PREVISÃO: T4 ou T3, 0,6 somados. Mecanismo: a cobertura medida é 13,8% dos
itens que aparecem em pilhas vencedoras, e a restrição de posição inicial de
estágio reduz isso mais. T1 exigiria que o resíduo graduado destravasse tarefas
inteiras num recorte de 1 em 7 decisões.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import (CODECS, ENTAO, _marca, _passo, custo, dims,
                                montar)
from novaphys.elimina_cor import mascara_independe_da_cor
from novaphys.promessa import PASTA

COR = ("preenche", "constante", "contorno", "moldura")
FEIXE = 4
PROF = 4


def _resolve_slot(pilha, cd, treino) -> Optional[Tuple[int, int]]:
    """
    Resolve a cor deste slot por MAIORIA e devolve (cor, E). Exige posição
    inicial de estágio — ver restrição 1 no cabeçalho.
    """
    if pilha and not _marca(pilha[-1]):
        return None
    passos = _passo(pilha, treino) if pilha else [(a, set()) for a, _b in treino]
    cont, n = Counter(), 0
    for (g, _e), (_a, b) in zip(passos, treino):
        if dims(g) != dims(b):
            return None
        if not mascara_independe_da_cor(cd, g):
            return None
        try:
            m = set(cd.prever(0, g).keys())
        except Exception:
            return None
        for (i, j) in m:
            hb, wb = dims(b)
            if i < hb and j < wb:
                cont[b[i][j]] += 1
                n += 1
    if n == 0:
        return None
    cor, maj = cont.most_common(1)[0]
    return cor, n - maj


def _candidatos(pilha, treino, eliminar: bool):
    """Os passos possíveis daqui. Com `eliminar`, cor vira UM nó, não dez."""
    fora = []
    inicio_estagio = (not pilha) or _marca(pilha[-1])
    for cd in CODECS:
        if eliminar and cd.nome in COR:
            if not inicio_estagio:
                continue
            r = _resolve_slot(pilha, cd, treino)
            if r is not None:
                fora.append((cd, r[0]))
            continue
        try:
            ms = list(itertools.islice(iter(cd.propor(treino) or []), 6))
        except Exception:
            continue
        for m in ms:
            fora.append((cd, m))
    return fora


def busca(treino, modo: str, segundos: float, feixe_n: int = None,
          prof_n: int = None) -> Optional[list]:
    """
    modo 'A' = enumera cores, ordena por (err, bits)      [= o sistema atual]
    modo 'B' = elimina cores, ordena por (err==0, bits)   [sem residuo graduado]
    modo 'C' = elimina cores, ordena por (err, bits)      [com residuo graduado]
    """
    t0 = time.time()
    feixe_n = feixe_n or FEIXE
    prof_n = prof_n or PROF
    elim = modo in ("B", "C")
    feixe = [[]]
    melhor, melhor_ch = None, None
    for _d in range(prof_n):
        novo = []
        for pilha in feixe:
            if time.time() - t0 > segundos:
                break
            for it in _candidatos(pilha, treino, elim):
                p2 = pilha + [it]
                try:
                    bits, err = custo(p2, treino)
                except Exception:
                    continue
                if modo == "B":
                    ch = (0 if err == 0 else 1, bits)
                else:
                    ch = (err, bits)
                novo.append((ch, err, p2))
                if err == 0 and (melhor_ch is None or (bits,) < melhor_ch):
                    melhor, melhor_ch = p2, (bits,)
            # DEFEITO: o marcador entrava com chave sentinela 10^9 e era
            # sempre podado, entao ENTAO nunca era explorado — e [[o-gargalo]]
            # registra que 42% das pilhas vencedoras usam marcador. Agora ele
            # herda a chave da pilha atual: um marcador sozinho nao muda nada,
            # so abre estagio.
            if pilha and not _marca(pilha[-1]):
                try:
                    bp, ep = custo(pilha, treino)
                    chp = (0 if ep == 0 else 1, bp) if modo == "B" else (ep, bp)
                    novo.append((chp, ep, pilha + [ENTAO]))
                except Exception:
                    pass
        if melhor is not None or not novo or time.time() - t0 > segundos:
            break
        novo.sort(key=lambda t: t[0])
        feixe = [p for _c, _e, p in novo[:feixe_n]]
    return melhor


def busca_com_forma(treino, modo: str, segundos: float, n_formas: int = 6,
                    feixe_n: int = None, prof_n: int = None):
    """
    `montar` escolhe um modelo de FORMA e so entao a pilha sobre a base
    rebaseada. Sem isso a busca so alcanca tarefas de dimensao igual — foi o
    que zerou os tres bracos no piloto.
    """
    t0 = time.time()
    try:
        cands = list(forma.propor(treino))[:n_formas]
    except Exception:
        cands = []
    for fm in cands:
        if time.time() - t0 > segundos:
            break
        try:
            reb = [(forma.base(fm, a), b) for a, b in treino]
        except Exception:
            continue
        resta = segundos - (time.time() - t0)
        if resta <= 0:
            break
        p = busca(reb, modo, resta, feixe_n, prof_n)
        if p is not None:
            return fm, p, reb
    return None


def experimento(n_tarefas: int = 40, segundos: float = 30.0,
                feixe_n: int = None, prof_n: int = None) -> None:
    todas = carregar(PASTA)
    todas.sort(key=lambda t: hashlib.md5(t[0].encode()).hexdigest())
    alvo = todas[:n_tarefas]

    print("=" * 78)
    print("TRES BRACOS — a eliminacao de cor vale alguma coisa na busca?")
    print("=" * 78)
    print("  %d tarefas por ordenacao de HASH fixada antes do teste" % len(alvo))
    print("  %.0fs por tarefa e braco; feixe %d; profundidade %d"
          % (segundos, feixe_n or FEIXE, prof_n or PROF))
    print("  A=enumera cores  B=elimina, selecao binaria  C=elimina, graduada\n")

    res = {m: {"treino": 0, "teste": 0, "nomes": []} for m in "ABC"}
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(alvo):
        for modo in "ABC":
            try:
                r = busca_com_forma(tr, modo, segundos, 6, feixe_n, prof_n)
            except Exception:
                r = None
            if r is None:
                continue
            _fm, pilha, reb = r
            # T0: reexecutar no executor ORIGINAL, sobre a base rebaseada
            try:
                _b, err = custo(pilha, reb)
            except Exception:
                continue
            if err != 0:
                print("    T0 FALHOU em %s braco %s: fecha na busca e nao no"
                      " executor" % (nome, modo))
                continue
            res[modo]["treino"] += 1
            res[modo]["nomes"].append(nome)
        if (k + 1) % 10 == 0:
            print("    %2d/%d tarefas | A=%d B=%d C=%d | %.0f min"
                  % (k + 1, len(alvo), res["A"]["treino"], res["B"]["treino"],
                     res["C"]["treino"], (time.time() - t0) / 60), flush=True)

    print("\n" + "=" * 78)
    for m in "ABC":
        print("  %s  fecha o treino em %2d/%d tarefas"
              % (m, res[m]["treino"], len(alvo)))
    a, b, c = (res[m]["treino"] for m in "ABC")
    so_c = set(res["C"]["nomes"]) - set(res["A"]["nomes"])
    perdidas = set(res["A"]["nomes"]) - set(res["C"]["nomes"])
    print("\n  C-A = %+d   C-B = %+d" % (c - a, c - b))
    print("  tarefas que so C fecha: %s" % (sorted(so_c) or "nenhuma"))
    print("  tarefas que A fecha e C perde: %s" % (sorted(perdidas) or "nenhuma"))
    print()
    if c >= a + 3 and c >= b + 3:
        print("T1 . a eliminacao condicional E UTIL neste recorte.")
    elif c > b and c <= a:
        print("T2 . o residuo graduado ajuda, e nao o bastante para bater A.")
    elif c == b:
        print("T3 . resolver os parametros nao mudou a ORDENACAO da estrutura.")
    else:
        print("T4 . os tres empatam ou a diferenca e menor que a barra: a")
        print("     particao cobre pouco da dificuldade (13,8% dos itens de")
        print("     pilha vencedora tem parametro de cor).")


if __name__ == "__main__":
    experimento()
