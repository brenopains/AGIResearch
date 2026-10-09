"""
novaphys.fragmento — o sistema sabe quando a ferramenta NÃO se aplica?
========================================================================

POR QUE ESTA PERGUNTA
-----------------------
Toda a série de nulos de biblioteca deste projeto é compatível com uma única
explicação que nunca foi testada diretamente: **o sistema aplica a ferramenta
onde ela não cabe**. [[sala-de-operacoes-nulo]] (4 portões falhados, placebo
melhor que promovidos), [[vocabulario-grande-piora]] (mais vocabulário PERDE
tarefas), [[o-criterio-nao-seleciona]] (os vocabulários novos nunca entram em
pilha vencedora) — nenhum mediu se a peça é aplicada fora do domínio dela.

E revisão externa apontou que meu teste de procedência transferiu **vocabulário
de segmentação**, não **procedimento**. A unidade estava errada. Aqui a unidade
é um FRAGMENTO EXECUTÁVEL: uma pilha vencedora com os parâmetros abertos.

O DESENHO, E O CONTROLE QUE DECIDE
------------------------------------
De cada tarefa resolvida extrai-se a pilha `S`. Constroem-se duas variantes:

  ISO       permutação de cores CONSISTENTE em toda a tarefa. A apresentação
            muda, **as relações são as mesmas**. O benefício DEVE sobreviver.
  QUEBRADO  permutação de cores da SAÍDA, DIFERENTE em cada par. A superfície
            é parecida, **a relação foi destruída**. E, crucialmente, ela foi
            destruída de um jeito que **reajustar parâmetro não conserta** —
            nenhuma atribuição única de cor serve para todos os pares.

E dois modos de uso:

  RIGIDO    aplica `S` exatamente como está
  ADAPTADO  mantém a ESTRUTURA de `S` e reajusta os parâmetros na tarefa nova

O par ISO/QUEBRADO é o que separa transferência de superajuste. Adaptar com
parâmetro livre pode encaixar qualquer coisa; se o ADAPTADO fechar no QUEBRADO,
a "transferência" não vale nada.

PRÉ-REGISTRO
--------------
  F0  PRÉ-CONDIÇÃO. `S` fecha na tarefa ORIGINAL (é de onde ela veio) e o
      QUEBRADO não é resolúvel por reajuste — conferido gerando-o com
      permutações inconsistentes entre pares.
  F1  ADAPTADO fecha no ISO em >= 2/3 E falha no QUEBRADO em >= 2/3
      -> transfere estrutura E sabe quando não se aplica
  F2  ADAPTADO fecha nos DOIS -> o parâmetro livre encaixa qualquer coisa; a
      transferência medida é superajuste, e isso explicaria os nulos de
      biblioteca de uma vez
  F3  ADAPTADO falha nos dois -> não transfere; a estrutura sozinha não carrega

  PLACEBO: fragmento de OUTRA tarefa, de mesmo comprimento, adaptado do mesmo
  jeito. Isola "esta estrutura serve" de "qualquer estrutura com parâmetro
  livre serve".

PREVISÃO: F2 é o risco real, 0,45. Mecanismo: os parâmetros das pilhas são
majoritariamente COR, e reajustar cor é exatamente o que
[[elimina-cor-nulo]] mostrou ser exato e barato — um ajuste que encaixa bem
pode encaixar bem demais. F1 sairia 0,35, F3 0,20.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, _marca, aplicar, custo, dims, montar
from novaphys.promessa import PASTA


def permuta_grade(g, sig: Dict[int, int]):
    return tuple(tuple(sig.get(c, c) for c in l) for l in g)


def variante_iso(tr, te, rnd):
    """Mesma tarefa, cores trocadas CONSISTENTEMENTE. Relacoes intactas."""
    cores = list(range(10))
    emb = cores[:]
    rnd.shuffle(emb)
    sig = dict(zip(cores, emb))
    return ([(permuta_grade(a, sig), permuta_grade(b, sig)) for a, b in tr],
            [(permuta_grade(x, sig), permuta_grade(y, sig)) for x, y in te])


def variante_quebrada(tr, te, rnd):
    """
    Superficie parecida, RELACAO destruida — e destruida de modo que reajustar
    parametro NAO conserta: cada par recebe uma permutacao DIFERENTE, so na
    saida. Nenhuma atribuicao unica de cor serve para todos.
    """
    def perm(rnd):
        c = list(range(10)); e = c[:]; rnd.shuffle(e)
        return dict(zip(c, e))
    return ([(a, permuta_grade(b, perm(rnd))) for a, b in tr],
            [(x, permuta_grade(y, perm(rnd))) for x, y in te])


def estrutura(fm, pilha) -> Tuple[str, List]:
    """
    A ESTRUTURA do fragmento: o TIPO do modelo de forma mais a sequencia de
    codecs e marcadores — tudo sem parametros.

    DEFEITO CORRIGIDO NO PILOTO: eu tratava so a pilha como fragmento e reusava
    o `fm` da tarefa original. Mas o modelo de forma TAMBEM carrega parametros de
    cor (`('ladrilho', (3,3,(0,...)))`), entao sob permutacao ele fica errado e o
    ISO dava 12% — impossivel, ja que o ISO e a mesma tarefa recolorida. O
    fragmento e forma + pilha; eu estava adaptando metade dele.
    """
    return fm[0], [it if _marca(it) else it[0] for it in pilha]


def reajusta_forma(nome_forma: str, tr):
    """Reajusta o modelo de FORMA: mantem o tipo, reescolhe os parametros."""
    try:
        cands = [f for f in forma.propor(tr) if f[0] == nome_forma]
    except Exception:
        cands = []
    for fm in cands:
        try:
            reb = [(forma.base(fm, a), b) for a, b in tr]
            if all(dims(g) == dims(b) for g, b in reb):
                return fm, reb
        except Exception:
            continue
    return None, None


def reajusta(est, tr, limite: int = 4) -> Optional[list]:
    """
    Mantem a ESTRUTURA e reajusta os PARAMETROS na tarefa nova.
    Busca gulosa slot a slot — o parametro e escolhido pelo que mais reduz erro.
    """
    pilha = []
    for passo in est:
        if _marca(passo):
            pilha.append(passo)
            continue
        try:
            ms = list(itertools.islice(iter(passo.propor(tr) or []), limite))
        except Exception:
            ms = []
        if not ms:
            return None
        # COLISAO DE SENTINELA, o defeito que zerava a adaptacao:
        # `Copia.propor` devolve `[None]` — o parametro LEGITIMO dela e `None`.
        # `if arg is None: return None` confundia "nenhum parametro escolhido"
        # com "o parametro e None", e toda pilha com `copia` (80 das 105)
        # morria aqui. Agora o "nao achei" e um sentinela proprio.
        SEM = object()
        melhor, arg = None, SEM
        for m in ms:
            try:
                b, e = custo(pilha + [(passo, m)], tr)
            except Exception:
                continue
            if melhor is None or (e, b) < melhor:
                melhor, arg = (e, b), m
        if arg is SEM:
            return None
        pilha.append((passo, arg))
    return pilha


def fecha(fm, pilha, tr, te) -> bool:
    try:
        reb = [(forma.base(fm, a), b) for a, b in tr]
        _b, err = custo(pilha, reb)
        if err != 0:
            return False
        return all(aplicar((fm, pilha), x) == y for x, y in te)
    except Exception:
        return False


def experimento(n_max: int = 40, teto: int = 20000, semente: int = 20260922
                ) -> None:
    rnd = random.Random(semente)
    alvo = json.load(open("resultados/oficial.json")) if __import__("os").path \
        .exists("resultados/oficial.json") else {}
    tarefas = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    nomes = [n for n in sorted(alvo) if n in tarefas][:n_max]

    print("=" * 78)
    print("FRAGMENTO — o sistema sabe quando a ferramenta NAO se aplica?")
    print("=" * 78)
    print("  %d tarefas resolvidas; de cada uma sai a ESTRUTURA da pilha\n"
          % len(nomes))
    print("  %-10s %6s %8s %9s %9s %9s"
          % ("tarefa", "|S|", "F0 orig", "ISO rig", "ISO adap", "QUEBR adap"))
    print("  " + "-" * 60)

    c = {"f0": 0, "iso_rig": 0, "iso_ad": 0, "queb_ad": 0, "plc_iso": 0,
         "plc_queb": 0, "n": 0}
    guardadas = []
    for nome in nomes:
        tr, te = tarefas[nome]
        try:
            fm, pilha, err = montar(tr, teto=teto)
        except Exception:
            continue
        if err != 0 or not pilha:
            continue
        est_f, est = estrutura(fm, pilha)
        f0 = fecha(fm, pilha, tr, te)
        if not f0:
            continue
        guardadas.append((nome, fm, est_f, est))
        c["n"] += 1
        c["f0"] += 1

        tri, tei = variante_iso(tr, te, rnd)
        trq, teq = variante_quebrada(tr, te, rnd)

        ri = fecha(fm, pilha, tri, tei)                    # rigido no ISO
        # ADAPTADO reajusta o fragmento INTEIRO: forma E pilha
        fmi, rebi = reajusta_forma(est_f, tri)
        pi = reajusta(est, rebi) if rebi else None
        ai = bool(pi) and fecha(fmi, pi, tri, tei)
        fmq, rebq = reajusta_forma(est_f, trq)
        pq = reajusta(est, rebq) if rebq else None
        aq = bool(pq) and fecha(fmq, pq, trq, teq)

        c["iso_rig"] += ri; c["iso_ad"] += ai; c["queb_ad"] += aq
        print("  %-10s %6d %8s %9s %9s %9s"
              % (nome, len([x for x in est if not _marca(x)]),
                 "ok" if f0 else "--", "ok" if ri else "--",
                 "ok" if ai else "--", "FECHA" if aq else "--"), flush=True)

    # PLACEBO: estrutura de OUTRA tarefa, mesmo tratamento
    print("\n  placebo: estrutura de OUTRA tarefa, adaptada do mesmo jeito")
    for i, (nome, fm, _ef, _e) in enumerate(guardadas):
        if len(guardadas) < 2:
            break
        _o, _f, ef_o, est_o = guardadas[(i + 1) % len(guardadas)]
        tr, te = tarefas[nome]
        tri, tei = variante_iso(tr, te, rnd)
        fmp, rebp = reajusta_forma(ef_o, tri)
        p = reajusta(est_o, rebp) if rebp else None
        c["plc_iso"] += bool(p) and fecha(fmp, p, tri, tei)

    n = max(c["n"], 1)
    print("\n" + "=" * 78)
    print("  F0  original fecha                  %d/%d" % (c["f0"], n))
    print("  ISO  rigido                         %d/%d  (%.0f%%)"
          % (c["iso_rig"], n, 100 * c["iso_rig"] / n))
    print("  ISO  ADAPTADO                       %d/%d  (%.0f%%)"
          % (c["iso_ad"], n, 100 * c["iso_ad"] / n))
    print("  ISO  placebo (estrutura de outra)   %d/%d  (%.0f%%)"
          % (c["plc_iso"], n, 100 * c["plc_iso"] / n))
    print("  QUEBRADO adaptado — DEVE falhar     %d/%d  (%.0f%%)"
          % (c["queb_ad"], n, 100 * c["queb_ad"] / n))
    print()
    ok_iso = c["iso_ad"] >= 2 * n / 3
    ok_queb = c["queb_ad"] <= n / 3
    if ok_iso and ok_queb:
        print("F1 . TRANSFERE ESTRUTURA E SABE QUANDO NAO SE APLICA.")
    elif ok_iso and not ok_queb:
        print("F2 . SUPERAJUSTE: o adaptado fecha no QUEBRADO em %.0f%%."
              % (100 * c["queb_ad"] / n))
        print("     O parametro livre encaixa qualquer coisa — e isso")
        print("     explicaria a serie de nulos de biblioteca de uma vez.")
    else:
        print("F3 . nao transfere: ISO adaptado em %.0f%%." % (100 * c["iso_ad"] / n))


if __name__ == "__main__":
    experimento()
