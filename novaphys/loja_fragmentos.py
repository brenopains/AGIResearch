"""
novaphys.loja_fragmentos — experiência acumulada resolve tarefa nova?
=======================================================================

A UNIDADE CERTA, MEDIDA HOJE
------------------------------
[[procedencia-atravessa-recombinando]] mediu que **vocabulário não atravessa**:
reuso rígido de léxico entre fontes dá 1 de 12, e o único cruzamento era
sobreposição de superfície.

[[fragmento-transfere]] mediu que **estrutura executável atravessa**: adaptado
73/105 contra rígido 52/105 e placebo 4/105, com zero falsas aplicações em 105.

Então a loja não guarda texto nem léxico. Guarda **fragmento**: tipo do modelo de
forma + sequência de codecs, com os parâmetros ABERTOS, mais a procedência de
onde veio e a evidência do que já fechou.

A PERGUNTA, QUE É A DO ASTRA
------------------------------
Três coisas diferentes que estavam coladas:

    mais busca resolve mais tarefas          -> benefício de computação
    melhorias no código ajudam vários domínios -> generalidade da implementação
    experiência anterior resolve tarefa nova   -> APRENDIZADO ACUMULATIVO

Só a terceira justifica investir em escala de experiência, e ela nunca foi
testada aqui. Este arquivo a testa: **as 105 soluções guardadas ajudam a
resolver alguma das 295 não resolvidas?**

O DESENHO
-----------
  ZERO      busca normal, orçamento O
  LOJA      recupera os K fragmentos mais promissores para ESTA tarefa,
            tenta adaptar cada um; o que sobrar do orçamento vai para a busca
  PLACEBO   mesma mecânica, K fragmentos SORTEADOS da loja

O placebo é obrigatório: sem ele, um ganho da LOJA pode ser só "tentar K pilhas
prontas antes de buscar", que qualquer K pilhas dariam.

**Orçamento casado**: recuperação e adaptação são contabilizadas. Se a loja
gasta metade do orçamento adaptando, ela busca com metade.

PRÉ-REGISTRO
--------------
  G0  PRÉ-CONDIÇÃO. Nenhuma tarefa de avaliação está na loja. As 105 são a
      experiência; as 295 restantes são o alvo. Sem sobreposição.
  G1  LOJA resolve >= 3 tarefas a mais que ZERO **e** >= 3 a mais que PLACEBO
      -> experiência acumulada paga na unidade certa; escalar coleta tem
         fundamento medido
  G2  LOJA ~ PLACEBO -> tentar pilhas prontas ajuda, a RECUPERAÇÃO não
  G3  LOJA ~ ZERO -> a experiência não paga nem com a unidade certa

PREVISÃO: G2, 0,45. Mecanismo: a recuperação aqui é por similaridade de
assinatura da tarefa, e [[assinatura-nao-e-mecanismo]] mediu que assinatura do
par entrada/saída não vira motor (3 builds, 0/400). G1 sairia 0,3 e seria o
primeiro positivo de acumulação deste projeto.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, _marca, aplicar, custo, dims, montar
from novaphys.fragmento import estrutura, fecha, reajusta, reajusta_forma
from novaphys.promessa import PASTA


def assinatura(tr) -> Tuple:
    """
    Descrição barata da tarefa, para recuperação. NÃO é mecanismo — é índice.
    [[assinatura-nao-e-mecanismo]] mediu que censo de assinatura não vira motor;
    aqui ela só ORDENA candidatos, e o verificador exato decide.
    """
    mesma_dim = all(dims(a) == dims(b) for a, b in tr)
    cres = sum(1 for a, b in tr if dims(b)[0] * dims(b)[1] >
               dims(a)[0] * dims(a)[1])
    cores_a = len({c for a, _b in tr for l in a for c in l})
    cores_b = len({c for _a, b in tr for l in b for c in l})
    h, w = dims(tr[0][0])
    return (int(mesma_dim), cres > 0, cores_a, cores_b, min(h, 30) // 6,
            min(w, 30) // 6, len(tr))


def _dist(a: Tuple, b: Tuple) -> float:
    return sum(0.0 if x == y else 1.0 for x, y in zip(a, b))


class Loja:
    """Guarda FRAGMENTO — estrutura com parâmetros abertos — e procedência."""

    def __init__(self) -> None:
        self.itens: List[dict] = []

    def guarda(self, procedencia: str, fm, pilha, tr) -> None:
        ef, est = estrutura(fm, pilha)
        self.itens.append({
            "procedencia": procedencia,
            "forma": ef,
            "estrutura": est,
            "assinatura": assinatura(tr),
            "n_passos": len([x for x in est if not _marca(x)]),
        })

    # A loja e CARA de construir (105 `montar`) e imutavel depois. Cachear nao
    # e otimizacao: e o que torna o experimento repetivel em minutos em vez de
    # dezenas de minutos, e o que permite rodar os bracos varias vezes.
    def salva(self, caminho: str) -> None:
        json.dump([{
            "procedencia": it["procedencia"], "forma": it["forma"],
            "estrutura": [x if _marca(x) else x.nome for x in it["estrutura"]],
            "assinatura": list(it["assinatura"]), "n_passos": it["n_passos"],
        } for it in self.itens], open(caminho, "w"))

    def carrega(self, caminho: str) -> bool:
        if not os.path.exists(caminho):
            return False
        porn = {cd.nome: cd for cd in CODECS}
        self.itens = []
        for d in json.load(open(caminho)):
            est = [x if _marca(x) else porn.get(x) for x in d["estrutura"]]
            if any(e is None for e in est):
                continue
            self.itens.append({
                "procedencia": d["procedencia"], "forma": d["forma"],
                "estrutura": est, "assinatura": tuple(d["assinatura"]),
                "n_passos": d["n_passos"]})
        return bool(self.itens)

    def recupera(self, tr, k: int) -> List[dict]:
        a = assinatura(tr)
        return sorted(self.itens, key=lambda it: _dist(a, it["assinatura"]))[:k]

    def sorteia(self, k: int, rnd) -> List[dict]:
        return rnd.sample(self.itens, min(k, len(self.itens)))


def tenta_fragmentos(frs: Sequence[dict], tr, te, limite_s: float
                     ) -> Tuple[bool, float]:
    """Adapta cada fragmento na tarefa nova. Devolve (fechou, segundos gastos)."""
    t0 = time.time()
    for fr in frs:
        if time.time() - t0 > limite_s:
            break
        fmn, reb = reajusta_forma(fr["forma"], tr)
        if reb is None:
            continue
        p = reajusta(fr["estrutura"], reb)
        if p and fecha(fmn, p, tr, te):
            return True, time.time() - t0
    return False, time.time() - t0


def experimento(k: int = 8, orcamento: float = 25.0, n_aval: int = 60,
                semente: int = 20260922) -> None:
    rnd = random.Random(semente)
    tarefas = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    resolvidas = set(json.load(open("resultados/oficial.json"))) \
        if os.path.exists("resultados/oficial.json") else set()

    print("=" * 78)
    print("LOJA DE FRAGMENTOS — experiencia acumulada resolve tarefa nova?")
    print("=" * 78)

    CACHE = "resultados/loja_fragmentos.json"
    loja = Loja()
    if loja.carrega(CACHE):
        print("  loja carregada do cache (%s)" % CACHE)
    else:
      for nome in sorted(resolvidas):
          if nome not in tarefas:
              continue
          tr, _te = tarefas[nome]
          try:
              fm, pilha, err = montar(tr, teto=20000)
          except Exception:
              continue
          if err == 0 and pilha:
              loja.guarda(nome, fm, pilha, tr)
      loja.salva(CACHE)
      print("  loja construida e salva em %s" % CACHE)
    print("  loja: %d fragmentos, de %d tarefas resolvidas"
          % (len(loja.itens), len(resolvidas)))
    prof = Counter(it["n_passos"] for it in loja.itens)
    print("  profundidade dos fragmentos: %s" % dict(sorted(prof.items())))

    # G0: alvo DISJUNTO da loja
    alvo = [n for n in sorted(tarefas) if n not in resolvidas][:n_aval]
    print("  alvo: %d tarefas NAO resolvidas, disjuntas da loja" % len(alvo))
    print("  orcamento %ds por tarefa e braco; K=%d\n" % (orcamento, k))

    res = {m: [] for m in ("ZERO", "LOJA", "PLACEBO")}
    for i, nome in enumerate(alvo):
        tr, te = tarefas[nome]

        # ZERO: so busca
        try:
            fm, pilha, err = montar(tr, teto=int(orcamento * 800))
            if err == 0 and pilha and fecha(fm, pilha, tr, te):
                res["ZERO"].append(nome)
        except Exception:
            pass

        # LOJA e PLACEBO: adapta K fragmentos, e o que sobrar vai para a busca
        for modo, frs in (("LOJA", loja.recupera(tr, k)),
                          ("PLACEBO", loja.sorteia(k, rnd))):
            ok, gasto = tenta_fragmentos(frs, tr, te, orcamento * 0.5)
            if not ok:
                resta = max(orcamento - gasto, 1.0)
                try:
                    fm, pilha, err = montar(tr, teto=int(resta * 800))
                    ok = err == 0 and bool(pilha) and fecha(fm, pilha, tr, te)
                except Exception:
                    ok = False
            if ok:
                res[modo].append(nome)

        if (i + 1) % 15 == 0:
            print("    %2d/%d | ZERO=%d LOJA=%d PLACEBO=%d"
                  % (i + 1, len(alvo), len(res["ZERO"]), len(res["LOJA"]),
                     len(res["PLACEBO"])), flush=True)

    print("\n" + "=" * 78)
    for m in ("ZERO", "LOJA", "PLACEBO"):
        print("  %-8s resolve %2d/%d" % (m, len(res[m]), len(alvo)))
    z, l, p = (len(res[m]) for m in ("ZERO", "LOJA", "PLACEBO"))
    print("\n  LOJA - ZERO = %+d   LOJA - PLACEBO = %+d" % (l - z, l - p))
    so = sorted(set(res["LOJA"]) - set(res["ZERO"]))
    print("  so a LOJA resolve: %s" % (so or "nenhuma"))
    print()
    if l >= z + 3 and l >= p + 3:
        print("G1 . EXPERIENCIA ACUMULADA PAGA na unidade certa. Primeiro")
        print("     positivo de acumulacao deste projeto.")
    elif abs(l - p) <= 1:
        print("G2 . tentar pilhas prontas ajuda ou nao, mas a RECUPERACAO nao:")
        print("     a loja empata com o sorteio.")
    else:
        print("G3 . a experiencia nao paga nem com a unidade certa (%+d)." % (l - z))


if __name__ == "__main__":
    experimento()
