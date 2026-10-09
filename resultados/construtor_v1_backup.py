"""
novaphys.construtor — a metade geradora: construir a regra comparando os exemplos
====================================================================================

A PAREDE, NAS PALAVRAS QUE O PROJETO JA TINHA ESCRITO
-------------------------------------------------------
`analogia.py`, semanas atras: "toda maquina que funciona neste projeto e
SUBTRATIVA — deducao elimina, eliminacao remove, residuo remove". Prova por forca
bruta em 200.000 casos de que eliminar nao cria informacao. Cinco tentativas de
construir a metade GERADORA falharam, e nenhuma gerava de verdade.

Em 24/09 vieram mais tres faces da mesma parede: o Sistema 2 BUSCA numa lingua
fixa; o Sistema 1 AJUSTA e decora ([[sistema1-transducao]]); o conceito tirado de
uma tarefa so e solucao, nao conceito ([[destila-nao-compoe]]). Buscar e ajustar
sao "escolher dentro de um espaco". Nenhuma peca CONSTROI.

O QUE HUMANOS FAZEM E O PROJETO SO FEZ NA FISICA
--------------------------------------------------
Humanos abstraem COMPARANDO: poem os exemplos lado a lado e perguntam o que e
igual entre eles. O que se conserva vira a regra; o que varia vira a variavel
que o teste preenche (Gentner: dois exemplos comparados ja produzem o conceito).
As unicas vezes que este sistema CRIOU foram assim: inventou h = x vy - y vx das
trajetorias ([[inventar-a-grandeza]]) — "ache o que se conserva entre os
exemplos". E a operacao ja atravessou para o ARC: invariantes em 342/400
tarefas, 97,4% valendo no teste ([[operacao-transfere-arc]]). So que ela foi usada
como CRITICO, nunca como CONSTRUTOR.

A FABRICA, PLUGADA — o usuario cobrou, e com razao
----------------------------------------------------
"como eu posso ter uma mente de nivel humano se eu fico tirando pedacinhos
dela?" Em 24/09 a mente usava so nucleo + Sistema 1 + sonho; digestora, HDC,
busca sob demanda, atlas e linguagem estavam FORA do ciclo (zero importacoes).
Este modulo nasce dentro do ciclo, com as pecas no papel que o usuario descreveu:

  DIGESTORA  grade crua -> NUTRIENTES: objetos e, para cada um, propriedades e
             relacoes de cena (o maior, o menor, o de cor unica, o que toca a
             borda, o posto de tamanho). A `digestor.py` de texto fazia o mesmo
             papel com lexico; aqui o nutriente certo e objeto e relacao.
  HDC        cada objeto vira um vetor papel-valor (`cena.monta_cena`); o HDC
             ALINHA objetos entre entrada e saida quando eles se movem, e GUARDA
             cada regra construida para ser puxada sob demanda na proxima tarefa
             parecida ([[sob-demanda-funciona]]).
  CONSTROI   para cada objeto de cada exemplo, o destino (mantem, apaga,
             recolore, move). A regra e a funcao nutriente -> destino que vale
             para TODOS os objetos de TODOS os exemplos. O destino pode ser
             absoluto ("vira azul") ou RELACIONAL ("vira a cor do maior") — e o
             relacional e o que generaliza quando o absoluto muda entre exemplos.
  SIMULA     antes de responder, a regra roda em cada entrada de treino e tem de
             reproduzir cada saida EXATAMENTE. Sem isso, nao ha resposta.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  K0  PRECISAO. Quando o construtor responde, acerta em >= 80% das vezes (o
      portao de simulacao existe para isso). Se ele chuta errado com frequencia,
      nao pode entrar no ciclo como fonte de resposta.
  K1  CAPACIDADE NOVA: no ARC-AGI-1 de treino resolve >= 5 tarefas que NEM o
      nucleo NEM o Sistema 1 resolviam -> a metade geradora existe e acrescenta
  K2  so resolve o que o nucleo ja resolvia -> reencontra os codecs que eu
      escrevi (cor_por_prop, apaga_prop), e o ganho e zero
  PLACEBO: pares com saidas DESARRANJADAS (cada entrada com a saida de outro
      par, mesma forma, diferente da verdadeira). Tem de construir <= 1 regra que
      passe na simulacao — senao o construtor encaixa qualquer coisa.

PREVISAO: K2 a 0,45, K1 a 0,35, K0 falhar a 0,20. Mecanismo de K2: recolorir e
apagar por propriedade o nucleo ja faz. O que pode dar K1: destinos RELACIONAIS
(a cor de outro objeto), movimento, e varios destinos combinados numa regra so
— coisas que nenhum codec escrito faz junto.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys.descricao import objetos as _objetos

Grid = Tuple[Tuple[int, ...], ...]


# =========================================================== 1. DIGESTORA
def fundo(g) -> int:
    return Counter(v for l in g for v in l).most_common(1)[0][0]


def _forma(cel) -> Tuple:
    i0 = min(i for i, _ in cel)
    j0 = min(j for _, j in cel)
    return tuple(sorted((i - i0, j - j0) for i, j in cel))


def digere(g) -> Tuple[int, List[dict]]:
    """
    Grade crua -> (fundo, objetos com nutrientes). Os nutrientes de CENA
    (posto, maior, menor, unico...) so existem comparando os objetos entre si —
    sao relacoes, nao atributos, e e isso que o humano ve de relance.
    """
    bg = fundo(g)
    h, w = len(g), len(g[0])
    obs = []
    for cel, cor in _objetos(g, bg):
        ii = [i for i, _ in cel]
        jj = [j for _, j in cel]
        obs.append({"cel": cel, "cor": cor, "tam": len(cel),
                    "alt": max(ii) - min(ii) + 1, "larg": max(jj) - min(jj) + 1,
                    "topo": min(ii), "esq": min(jj), "forma": _forma(cel),
                    "borda": int(min(ii) == 0 or min(jj) == 0
                                 or max(ii) == h - 1 or max(jj) == w - 1)})
    if not obs:
        return bg, obs
    tams = sorted({o["tam"] for o in obs}, reverse=True)
    cores = Counter(o["cor"] for o in obs)
    formas = Counter(o["forma"] for o in obs)
    for o in obs:
        o["posto"] = tams.index(o["tam"])
        o["maior"] = int(o["tam"] == tams[0])
        o["menor"] = int(o["tam"] == tams[-1])
        o["cor_unica"] = int(cores[o["cor"]] == 1)
        o["forma_unica"] = int(formas[o["forma"]] == 1)
        o["n_forma"] = formas[o["forma"]]
        o["par_tam"] = o["tam"] % 2
        o["tudo"] = 1
    return bg, obs


NUTRIENTES = ("tudo", "cor", "tam", "alt", "larg", "forma", "borda", "posto", "maior",
              "menor", "cor_unica", "forma_unica", "n_forma", "par_tam")


def _referencias(obs) -> Dict[str, int]:
    """Cores que se nomeiam por RELACAO na cena: 'a cor do maior' etc."""
    ref = {}
    for nome, filtro in (("maior", "maior"), ("menor", "menor"),
                         ("cor_unica", "cor_unica"), ("forma_unica", "forma_unica")):
        cs = {o["cor"] for o in obs if o[filtro]}
        if len(cs) == 1:
            ref["ref:" + nome] = cs.pop()
    return ref


# ================================================================= 2. HDC
_E = None


def _espaco():
    global _E
    if _E is None:
        from novaphys.hdc_indice import Espaco
        _E = Espaco()
    return _E


def vetor(o: dict) -> np.ndarray:
    """Objeto -> vetor HDC papel-valor. Forma, cor e tamanho: o que o olho reconhece."""
    from novaphys.cena import monta_cena
    return monta_cena(_espaco(), [("forma", str(o["forma"])), ("cor", str(o["cor"])),
                                  ("tam", str(o["tam"]))])


def alinha_movido(o: dict, b, bg) -> Optional[Tuple[int, int]]:
    """
    Para onde foi o objeto `o`? Procura, na SAIDA, um objeto de mesma forma e
    cor — o HDC ranqueia os candidatos pela similaridade papel-valor, e a
    correspondencia exata de celulas decide. Devolve o deslocamento ou None.
    """
    from novaphys.hdc_indice import sim
    bgb, obs_b = digere(b)
    if not obs_b:
        return None
    vo = vetor(o)
    cands = sorted(obs_b, key=lambda p: -sim(vo, vetor(p)))[:4]
    for p in cands:
        if p["forma"] == o["forma"] and p["cor"] == o["cor"]:
            return (p["topo"] - o["topo"], p["esq"] - o["esq"])
    return None


class Memoria:
    """
    O que o construtor aprendeu, guardado em HDC para ser puxado SOB DEMANDA:
    cada regra que passou na simulacao fica indexada pelo vetor da cena de
    entrada. Numa tarefa nova, os nutrientes das regras de cenas parecidas sao
    tentados primeiro — a fabrica puxando nutriente, nao recomecando do zero.
    """

    def __init__(self) -> None:
        self.itens: List[Tuple[np.ndarray, str]] = []
        self.consultas = 0
        self.acertos_ordem = 0

    def cena(self, obs) -> Optional[np.ndarray]:
        from novaphys.hdc_indice import feixe
        if not obs:
            return None
        return feixe([vetor(o) for o in obs], _espaco().dim)

    def guarda(self, obs, nutriente: str) -> None:
        v = self.cena(obs)
        if v is not None:
            self.itens.append((v, nutriente))

    def ordem(self, obs) -> List[str]:
        from novaphys.hdc_indice import sim
        v = self.cena(obs)
        if v is None or not self.itens:
            return []
        self.consultas += 1
        ranq = sorted(self.itens, key=lambda it: -sim(v, it[0]))
        vistos = []
        for _v, n in ranq:
            if n not in vistos:
                vistos.append(n)
        return vistos[:5]


# ======================================================== 3. CONSTROI
def _destinos(o: dict, a, b, bg, refs) -> Optional[set]:
    """
    O que aconteceu com o objeto `o` entre a entrada `a` e a saida `b` — como
    CONJUNTO de descricoes aceitaveis. O destino relacional entra junto do
    absoluto; a comparacao entre exemplos e que decide qual deles sobrevive.
    """
    vals = {b[i][j] for i, j in o["cel"]}
    if vals == {o["cor"]}:
        return {"mantem"}
    if vals == {bg}:
        d = alinha_movido(o, b, bg)
        out = {"apaga"}
        if d is not None and d != (0, 0):
            out = {"move:%d,%d" % d}
        return out
    if len(vals) == 1:
        c = vals.pop()
        return {"cor:%d" % c} | {k for k, v in refs.items() if v == c}
    d = alinha_movido(o, b, bg)
    if d is not None and d != (0, 0):
        return {"move:%d,%d" % d}
    return None


def _aplica_destino(out, o, dest, bg, refs) -> bool:
    if dest == "mantem":
        return True
    if dest == "apaga":
        for i, j in o["cel"]:
            out[i][j] = bg
        return True
    if dest.startswith("cor:"):
        c = int(dest[4:])
    elif dest.startswith("ref:"):
        if dest not in refs:
            return False
        c = refs[dest]
    elif dest.startswith("move:"):
        return True                       # movimentos sao pintados depois
    else:
        return False
    for i, j in o["cel"]:
        out[i][j] = c
    return True


def executa(regra, g) -> Optional[Grid]:
    """A regra construida aplicada a uma grade: e a SIMULACAO."""
    nutr, tabela = regra
    bg, obs = digere(g)
    refs = _referencias(obs)
    out = [list(l) for l in g]
    h, w = len(g), len(g[0])
    movs = []
    for o in obs:
        v = o[nutr]
        if v not in tabela:
            return None                    # valor nunca visto: a regra nao sabe
        dest = tabela[v]
        if dest.startswith("move:"):
            dy, dx = (int(x) for x in dest[5:].split(","))
            movs.append((o, dy, dx))
            for i, j in o["cel"]:
                out[i][j] = bg
        elif not _aplica_destino(out, o, dest, bg, refs):
            return None
    for o, dy, dx in movs:
        for i, j in o["cel"]:
            y, x = i + dy, j + dx
            if not (0 <= y < h and 0 <= x < w):
                return None
            out[y][x] = o["cor"]
    return tuple(tuple(l) for l in out)


def constroi(treino, entradas=(), memoria: Optional[Memoria] = None):
    """
    COMPARA os exemplos e constroi a regra. Para cada nutriente N, a regra
    candidata e uma tabela valor-de-N -> destino que tem de valer para TODO
    objeto de TODO exemplo (a intersecao das descricoes aceitaveis). O que
    sobra depois de comparar todos os exemplos e o que se conserva.
    Devolve (regra, diagnostico).
    """
    if not treino:
        return None, "sem exemplos"
    if any(len(a) != len(b) or len(a[0]) != len(b[0]) for a, b in treino):
        return None, "muda de tamanho"
    cenas = []
    for a, b in treino:
        bg, obs = digere(a)
        if not obs or len(obs) > 40:
            return None, "sem objetos" if not obs else "objetos demais"
        refs = _referencias(obs)
        ds = []
        for o in obs:
            d = _destinos(o, a, b, bg, refs)
            if d is None:
                return None, "destino fora do repertorio"
            ds.append(d)
        cenas.append((a, b, bg, obs, ds))
    ordem = list(NUTRIENTES)
    if memoria is not None:
        primeiro = memoria.ordem(cenas[0][3])
        ordem = [n for n in primeiro if n in NUTRIENTES] + \
            [n for n in ordem if n not in primeiro]
    teste_obs = [digere(x)[1] for x in entradas]
    cands = []
    for nutr in ordem:
        aceita: Dict = {}
        ok = True
        for _a, _b, _bg, obs, ds in cenas:
            for o, d in zip(obs, ds):
                v = o[nutr]
                aceita[v] = (aceita[v] & d) if v in aceita else set(d)
                if not aceita[v]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # o teste tem de cair DENTRO do que foi visto
        if any(o[nutr] not in aceita for obs in teste_obs for o in obs):
            continue
        # prefere o destino RELACIONAL (generaliza) ao absoluto, e o fixo a ambos
        def escolhe(s):
            for pref in ("mantem", "apaga"):
                if pref in s:
                    return pref
            rel = sorted(x for x in s if x.startswith("ref:"))
            return rel[0] if rel else sorted(s)[0]
        tabela = {v: escolhe(s) for v, s in aceita.items()}
        # MDL simples: tabela menor primeiro, com empate pela ordem
        cands.append((len(tabela), len(cands), nutr, tabela))
    for _n, _k, nutr, tabela in sorted(cands, key=lambda c: (c[0], c[1])):
        regra = (nutr, tabela)
        if all(executa(regra, a) == tuple(tuple(l) for l in b) for a, b in treino):
            if memoria is not None:
                memoria.guarda(cenas[0][3], nutr)
            return regra, "construida por %s (%d valores)" % (nutr, len(tabela))
    return None, "nenhuma regra reproduz os exemplos" if cands else "nada se conserva"


def resolve(treino, entradas, memoria: Optional[Memoria] = None):
    """Interface da mente: resposta para as entradas, ou None. Nunca olha gabarito."""
    tr = [(tuple(tuple(l) for l in a), tuple(tuple(l) for l in b)) for a, b in treino]
    ent = [tuple(tuple(l) for l in x) for x in entradas]
    regra, diag = constroi(tr, ent, memoria)
    if regra is None:
        return None, diag
    resp = [executa(regra, x) for x in ent]
    if any(r is None for r in resp):
        return None, "regra nao se aplica ao teste"
    return [list(map(list, r)) for r in resp], diag


# ==================================================== 4. o experimento
def _desarranja(tr, rnd):
    idx = list(range(len(tr)))
    for _ in range(300):
        rnd.shuffle(idx)
        if all(idx[i] != i and np.array(tr[idx[i]][1]).shape == np.array(tr[i][0]).shape
               and tr[idx[i]][1] != tr[i][1] for i in range(len(tr))):
            return [(tr[i][0], tr[idx[i]][1]) for i in range(len(tr))]
    return None


def experimento(pasta: str, etq: str, conhecidos=None) -> Dict:
    from novaphys.arc import carregar
    conhecidos = conhecidos or set()
    T = list(carregar(pasta))
    mem = Memoria()
    rnd = random.Random(20260925)
    res, diag, plc_ok, plc_n = {}, Counter(), 0, 0
    t0 = time.time()
    for nome, tr, te in T:
        r, d = resolve(tr, [x for x, _ in te], mem)
        diag[d.split(" por ")[0] if d.startswith("construida") else d] += 1
        if r is not None:
            res[nome] = all(np.array_equal(np.array(ri), np.array(y))
                            for ri, (_x, y) in zip(r, te))
        tp = _desarranja(tr, rnd)
        if tp is not None:
            plc_n += 1
            rp, _dp = constroi(
                [(tuple(tuple(l) for l in a), tuple(tuple(l) for l in b)) for a, b in tp])
            plc_ok += int(rp is not None)
    resp = len(res)
    certas = sorted(n for n, ok in res.items() if ok)
    novas = sorted(set(certas) - conhecidos)
    print("=" * 74)
    print("CONSTRUTOR — %s: %d tarefas | %.0f s" % (etq, len(T), time.time() - t0))
    print("=" * 74)
    print("  respondeu em %d, acertou %d  (precisao %.0f%%)"
          % (resp, len(certas), 100 * len(certas) / max(resp, 1)))
    print("  novas (fora do que nucleo/Sistema 1 ja resolviam): %d %s"
          % (len(novas), novas))
    print("  PLACEBO (saidas desarranjadas): %d regras passaram a simulacao em %d"
          % (plc_ok, plc_n))
    print("  memoria HDC: %d regras guardadas, %d consultas sob demanda"
          % (len(mem.itens), mem.consultas))
    print("  por que nao construiu:")
    for k, v in diag.most_common():
        print("    %-40s %d" % (k, v))
    return {"etq": etq, "respondeu": resp, "certas": certas, "novas": novas,
            "placebo": [plc_ok, plc_n], "diag": dict(diag)}


if __name__ == "__main__":
    import glob
    from novaphys.promessa import PASTA
    from novaphys.eter import PASTA_AVAL
    conhecidos = set(json.load(open("resultados/oficial.json")))
    if os.path.exists("resultados/sistema1.json"):
        conhecidos |= set(json.load(open("resultados/sistema1.json"))["resolve"])
    saida = {}
    saida["arc1_treino"] = experimento(PASTA, "ARC-AGI-1 treino", conhecidos)
    saida["arc1_aval"] = experimento(PASTA_AVAL, "ARC-AGI-1 avaliacao")
    saida["arc2_aval"] = experimento(r"D:\agi_dados\arc_agi_2\data\evaluation",
                                     "ARC-AGI-2 avaliacao")
    os.makedirs("resultados", exist_ok=True)
    json.dump(saida, open("resultados/construtor.json", "w"), indent=1)
