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

v3 — O CARIMBO LIDO DO DADO (24/09, escrito ANTES de rodar)
-------------------------------------------------------------
O diagnostico da v2 localizou a parede: das tarefas de mesma dimensao em que a
tabela nutriente -> destino NAO tem conflito nenhum e mesmo assim nada se
constroi, a maioria falha porque a saida CRIA conteudo que nenhum raio nem
movimento explica (ARC-2 aval: 22 de 27; ARC-1 treino: 119 de 164). E, quando o
destino sai do repertorio, e quase sempre RECOLORIR PARTE do objeto (114 de 156
objetos no ARC-2). A maquina transforma o que existe; nao desenha.

`celobj.py` atacou o mesmo buraco com doze testes celula-objeto escritos por
mim — vocabulario a mao, e o projeto ja sabe quanto isso rende. Aqui o desenho e
LIDO: cada celula que mudou e descrita pela posicao em relacao ao objeto mais
proximo — em cada eixo, ANTES da caixa, DENTRO ou DEPOIS, e a que distancia — em
quatro niveis de abstracao (distancia exata ou "qualquer" em cada eixo). A
comparacao entre exemplos guarda a descricao MAIS GERAL que nenhum objeto de
mesmo nutriente contradiz. "Moldura em volta", "carimbo em volta do ponto",
"encher a caixa", "pintar a faixa de cima" e "recolorir a linha de cima do
objeto" saem do mesmo mecanismo, sem nome nenhum. O carimbo pode cair no fundo
e nas celulas do proprio objeto — recolorir parte e so carimbo sobre si.

  E0  PRE-CONDICAO: carimbo em >= 5 regras construidas no ARC-1 de treino.
  E1  ARC-1 treino >= 45 certas (+8 sobre as 37 da v2), precisao >= 70% e
      placebo <= 3 regras -> ler o desenho do dado funciona
  E2  +3 a +7 -> ajuda
  E3  <= +2 -> nulo: a posicao relativa a caixa nao e o que decide onde desenhar
  ARC-2 aval: relatado, sem portao. Previsao 0 a 2.

PREVISAO: E2 a 0,45, E1 a 0,30, E3 a 0,25. Minha calibragem: 17 de 27
previsoes erradas, todas otimistas.
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
                    "caixa": (min(ii), max(ii), min(jj), max(jj)),
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
DIRS = {"N": (-1, 0), "S": (1, 0), "L": (0, 1), "O": (0, -1),
        "NE": (-1, 1), "NO": (-1, -1), "SE": (1, 1), "SO": (1, -1)}


def _raio(o: dict, d: str, parada: str, a, bg) -> set:
    """
    Celulas que um RAIO do objeto pinta. Parte de cada celula do objeto; so pinta
    fundo. 'borda': atravessa o que encontrar ate a borda; 'obstaculo': para no
    primeiro nao-fundo que nao e do proprio objeto. E a relacao ENTRE objetos
    mais comum do ARC: estender ate bater, ligar dois do mesmo lado.
    """
    dy, dx = DIRS[d]
    h, w = len(a), len(a[0])
    fora = set()
    for i, j in o["cel"]:
        y, x = i + dy, j + dx
        while 0 <= y < h and 0 <= x < w:
            if (y, x) in o["cel"]:
                y, x = y + dy, x + dx
                continue
            if a[y][x] != bg:
                if parada == "obstaculo":
                    break
            else:
                fora.add((y, x))
            y, x = y + dy, x + dx
    return fora


def _emissoes(o: dict, a, b, bg, refs) -> frozenset:
    """
    Quais raios deste objeto cabem INTEIROS nas celulas novas deste exemplo, e
    de que cor. Pintar qualquer um deles nunca contradiz a saida — a simulacao
    depois exige que a UNIAO cubra tudo que apareceu.
    """
    novos = {(i, j) for i in range(len(a)) for j in range(len(a[0]))
             if a[i][j] == bg and b[i][j] != bg}
    if not novos:
        return frozenset()
    out = set()
    for d in DIRS:
        for parada in ("borda", "obstaculo"):
            R = _raio(o, d, parada, a, bg)
            if not R or not R <= novos:
                continue
            cs = {b[y][x] for y, x in R}
            if len(cs) != 1:
                continue
            c = cs.pop()
            if c == o["cor"]:
                out.add((d, parada, "propria"))
            out.add((d, parada, "cor:%d" % c))
            for k, v in refs.items():
                if v == c:
                    out.add((d, parada, k))
    return frozenset(out)


# ------------------------------------------------ o carimbo lido do dado (v3)
def _zona(i: int, lo: int, hi: int):
    if i < lo:
        return "A", lo - i
    if i > hi:
        return "D", i - hi
    return "M", i - lo


def _faixa(z: str, d, lo: int, hi: int, n: int):
    if z == "A":
        return range(0, lo) if d == "*" else [lo - d]
    if z == "D":
        return range(hi + 1, n) if d == "*" else [hi + d]
    if d == "*":
        return range(lo, hi + 1)
    return [lo + d] if lo + d <= hi else []


def _estampa(o: dict, desc, g, bg) -> frozenset:
    """
    Celulas que o carimbo `desc` cobre em volta de `o`. O ALVO faz parte da
    descricao: pinta o FUNDO em volta, ou o PROPRIO objeto — nunca os dois.
    """
    t, b, l, r = o["caixa"]
    h, w = len(g), len(g[0])
    zy, dy, zx, dx, alvo = desc
    cel = o["cel"]
    if alvo == "si":
        return frozenset((i, j) for i in _faixa(zy, dy, t, b, h) if 0 <= i < h
                         for j in _faixa(zx, dx, l, r, w) if (i, j) in cel)
    return frozenset((i, j) for i in _faixa(zy, dy, t, b, h) if 0 <= i < h
                     for j in _faixa(zx, dx, l, r, w)
                     if 0 <= j < w and g[i][j] == bg)


def _rotulos(c: int, o: dict, refs, bg) -> set:
    out = {"cor:%d" % c}
    if c == o["cor"]:
        out.add("propria")
    if c == bg:
        out.add("fundo")
    out |= {k for k, v in refs.items() if v == c}
    return out


def _dono(a, b, bg, obs):
    """
    Cada celula que MUDOU e de quem? Do proprio objeto, se e dele; senao do objeto
    de caixa mais proxima (proximidade de Gestalt). Devolve {indice: [celulas]}.
    """
    h, w = len(a), len(a[0])
    de = {}
    for k, o in enumerate(obs):
        for c in o["cel"]:
            de[c] = k
    donos = defaultdict(list)
    for i in range(h):
        for j in range(w):
            if a[i][j] == b[i][j]:
                continue
            if (i, j) in de:
                donos[de[(i, j)]].append((i, j))
                continue
            if a[i][j] != bg:
                continue
            dist = []
            for k, o in enumerate(obs):
                t, bb, l, r = o["caixa"]
                dist.append((max(t - i, i - bb, 0, l - j, j - r), k))
            m = min(dist)[0]
            for d, k in dist:
                if d == m:
                    donos[k].append((i, j))
    return donos


def _candidatos_carimbo(o: dict, celulas, b, bg, refs) -> set:
    """O que as celulas mudadas dizem de si mesmas, nos quatro niveis de abstracao."""
    t, bb, l, r = o["caixa"]
    out = set()
    for i, j in celulas:
        zy, dy = _zona(i, t, bb)
        zx, dx = _zona(j, l, r)
        al = "si" if (i, j) in o["cel"] else "fundo"
        labs = _rotulos(b[i][j], o, refs, bg)
        for d in ((zy, dy, zx, dx, al), (zy, "*", zx, dx, al), (zy, dy, zx, "*", al),
                  (zy, "*", zx, "*", al)):
            for lab in labs:
                out.add((d, lab))
    return out


def _carimbo_vale(o: dict, s, a, b, bg, refs) -> bool:
    """Todo lugar que o carimbo cobre tem, na saida, exatamente a cor dele?"""
    desc, lab = s
    cel = _estampa(o, desc, a, bg)
    if not cel:
        return True
    c = bg if lab == "fundo" else _cor_de(lab, o, refs)
    return c is not None and all(b[i][j] == c for i, j in cel)


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
    return {"mantem"}                      # mudou EM PARTE: quem explica e o carimbo


def _cor_de(rotulo: str, o: dict, refs) -> Optional[int]:
    if rotulo == "propria":
        return o["cor"]
    if rotulo.startswith("cor:"):
        return int(rotulo[4:])
    return refs.get(rotulo)


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


def _chave(o, nutr):
    return tuple(o[n] for n in nutr) if isinstance(nutr, tuple) else o[nutr]


def executa(regra, g) -> Optional[Grid]:
    """A regra construida aplicada a uma grade: e a SIMULACAO."""
    nutr, tabela = regra
    bg, obs = digere(g)
    refs = _referencias(obs)
    out = [list(l) for l in g]
    h, w = len(g), len(g[0])
    movs, emite, carimbos = [], [], []
    for o in obs:
        v = _chave(o, nutr)
        if v not in tabela:
            return None                    # valor nunca visto: a regra nao sabe
        dest, emis = tabela[v][0], tabela[v][1]
        for s_ in (tabela[v][2] if len(tabela[v]) > 2 else ()):
            carimbos.append((o, s_))
        if dest.startswith("move:"):
            dy, dx = (int(x) for x in dest[5:].split(","))
            movs.append((o, dy, dx))
            for i, j in o["cel"]:
                out[i][j] = bg
        elif not _aplica_destino(out, o, dest, bg, refs):
            return None
        for e in emis:
            emite.append((o, e))
    for o, dy, dx in movs:
        for i, j in o["cel"]:
            y, x = i + dy, j + dx
            if not (0 <= y < h and 0 <= x < w):
                return None
            out[y][x] = o["cor"]
    for o, (d, parada, rot) in emite:       # raios: calculados na ENTRADA
        c = _cor_de(rot, o, refs)
        if c is None:
            return None
        for y, x in _raio(o, d, parada, g, bg):
            out[y][x] = c
    for o, (desc, lab) in carimbos:          # carimbos: lidos na ENTRADA tambem
        c = bg if lab == "fundo" else _cor_de(lab, o, refs)
        if c is None:
            return None
        for y, x in _estampa(o, desc, g, bg):
            out[y][x] = c
    return tuple(tuple(l) for l in out)


def _escolhe(s):
    for pref in ("mantem", "apaga"):
        if pref in s:
            return pref
    rel = sorted(x for x in s if x.startswith("ref:"))
    return rel[0] if rel else sorted(s)[0]


def constroi(treino, entradas=(), memoria: Optional[Memoria] = None):
    """
    COMPARA os exemplos e constroi a regra. Para cada nutriente (ou PAR de
    nutrientes, se nenhum sozinho se conserva), a regra e uma tabela
    valor -> (destino, emissoes) que vale para TODO objeto de TODO exemplo. O que
    sobra depois de comparar todos os exemplos e o que se conserva. Emissoes sao
    podadas ao MINIMO que ainda reproduz os exemplos (Occam, nao enumeracao).
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
        donos = _dono(a, b, bg, obs)
        ds = []
        for k, o in enumerate(obs):
            d = _destinos(o, a, b, bg, refs)
            if d is None:
                return None, "destino fora do repertorio"
            ds.append((d, _emissoes(o, a, b, bg, refs),
                       _candidatos_carimbo(o, donos.get(k, ()), b, bg, refs)))
        cenas.append((a, b, bg, obs, ds, refs))
    ordem = list(NUTRIENTES)
    if memoria is not None:
        primeiro = memoria.ordem(cenas[0][3])
        ordem = [n for n in primeiro if n in NUTRIENTES] + \
            [n for n in ordem if n not in primeiro]
    teste_obs = [digere(x)[1] for x in entradas]
    alvo = [tuple(tuple(l) for l in b) for _a, b in treino]

    vale_memo: Dict = {}

    def vale(e, k, st):
        chave = (e, k, st)
        if chave not in vale_memo:
            a, b, bg, obs, _ds, refs = cenas[e]
            vale_memo[chave] = _carimbo_vale(obs[k], st, a, b, bg, refs)
        return vale_memo[chave]

    def cobre(e, k, st):
        """As celulas que MUDAM e que este carimbo explica."""
        a, b, bg, obs, _ds, _refs = cenas[e]
        return {(e, i, j) for i, j in _estampa(obs[k], st[0], a, bg) if a[i][j] != b[i][j]}

    def ordem_poda(st):
        (_zy, dy, _zx, dx, _al), lab = st
        n_geral = (dy == "*") + (dx == "*")
        rank = 0 if lab.startswith("cor:") else 1 if lab == "fundo" else             2 if lab.startswith("ref:") else 3
        return (n_geral, rank, str(st))

    def carimbos_de(membros, cands):
        """
        Os carimbos que valem para TODO objeto deste valor, podados ao minimo:
        some primeiro o mais especifico e o de cor absoluta, fica o mais geral e
        o relacional — Occam sobre o que a comparacao deixou.
        """
        validos = [st for st in cands if all(vale(e, k, st) for e, k in membros)]
        cob = {st: set().union(*(cobre(e, k, st) for e, k in membros)) for st in validos}
        # so e REGULARIDADE o que se repete: o carimbo tem de explicar mudanca em
        # pelo menos DOIS pares de treino. Visto num par so, e copia daquele par
        # (a v3 crua deixou 31/134 regras de placebo passarem por isso).
        validos = [st for st in validos if len({c[0] for c in cob[st]}) >= 2]
        conta = Counter(c for st in validos for c in cob[st])
        fica = set(validos)
        for st in sorted(validos, key=ordem_poda):
            if all(conta[c] >= 2 for c in cob[st]):
                fica.discard(st)
                for c in cob[st]:
                    conta[c] -= 1
        return frozenset(fica)

    def tenta(nutr):
        aceita: Dict = {}
        for e, (_a, _b, _bg, obs, ds, _r) in enumerate(cenas):
            for k, (o, (d, em, cc)) in enumerate(zip(obs, ds)):
                v = _chave(o, nutr)
                if v in aceita:
                    t0_, e0_, c0_, m0_ = aceita[v]
                    aceita[v] = (t0_ & d, e0_ & em, c0_ | cc, m0_ + [(e, k)])
                else:
                    aceita[v] = (set(d), set(em), set(cc), [(e, k)])
                if not aceita[v][0]:
                    return None
        if any(_chave(o, nutr) not in aceita for obs in teste_obs for o in obs):
            return None
        tabela = {v: (_escolhe(t), frozenset(em), carimbos_de(m, cc))
                  for v, (t, em, cc, m) in aceita.items()}

        def roda(tab):
            return all(executa((nutr, tab), a) == b_ for (a, _b), b_ in zip(treino, alvo))
        if not roda(tabela):
            return None
        for v in sorted(tabela, key=str):      # poda: o carimbo minimo que basta
            if tabela[v][2]:
                t2 = dict(tabela)
                t2[v] = (tabela[v][0], tabela[v][1], frozenset())
                if roda(t2):
                    tabela = t2
        for v in sorted(tabela, key=str):      # poda: a emissao minima que basta
            if tabela[v][1]:
                t2 = dict(tabela)
                t2[v] = (tabela[v][0], frozenset(), tabela[v][2])
                if roda(t2):
                    tabela = t2
        return (nutr, tabela)

    achadas = []
    for nutr in ordem:
        r = tenta(nutr)
        if r is not None:                      # mais simples: menos valores, menos carimbos
            achadas.append((len(r[1]) + sum(len(t[2]) for t in r[1].values()) / 100.,
                            len(achadas), r))
    if not achadas:
        base = [n for n in NUTRIENTES if n != "tudo"]
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                r = tenta((base[i], base[j]))
                if r is not None:
                    achadas.append((len(r[1]) + 100, len(achadas), r))
            if achadas:
                break
    if not achadas:
        return None, "nada se conserva"
    _n, _k, regra = min(achadas, key=lambda x: (x[0], x[1]))
    n_carimbo = sum(len(t[2]) for t in regra[1].values())
    if memoria is not None and not isinstance(regra[0], tuple):
        memoria.guarda(cenas[0][3], regra[0])
    return regra, "construida por %s (%d valores%s)" % (
        regra[0], len(regra[1]), ", %d carimbos" % n_carimbo if n_carimbo else "")


def resolve(treino, entradas, memoria: Optional[Memoria] = None):
    """
    Interface da mente: resposta para as entradas, ou None. Nunca olha gabarito.
    Se a saida muda de tamanho, pede a GEOMETRIA ao nucleo (`forma`) e constroi
    sobre a grade rebaseada — os dois sistemas cooperando.
    """
    tr = [(tuple(tuple(l) for l in a), tuple(tuple(l) for l in b)) for a, b in treino]
    ent = [tuple(tuple(l) for l in x) for x in entradas]
    regra, diag = constroi(tr, ent, memoria)
    if regra is not None:
        resp = [executa(regra, x) for x in ent]
        if all(r is not None for r in resp):
            return [list(map(list, r)) for r in resp], diag
        return None, "regra nao se aplica ao teste"
    if diag != "muda de tamanho":
        return None, diag
    try:
        from novaphys import forma
        fms = list(forma.propor(tr))[:3]
    except Exception:
        return None, diag
    for fm in fms:
        try:
            reb = [(forma.base(fm, a), b) for a, b in tr]
            rent = [forma.base(fm, x) for x in ent]
        except Exception:
            continue
        if any(g is None for g, _b in reb) or any(g is None for g in rent):
            continue
        if any(len(g) != len(b) or len(g[0]) != len(b[0]) for g, b in reb):
            continue
        reb = [(tuple(tuple(l) for l in g), b) for g, b in reb]
        rent = [tuple(tuple(l) for l in g) for g in rent]
        regra, d2 = constroi(reb, rent, memoria)
        if regra is None:
            continue
        resp = [executa(regra, x) for x in rent]
        if all(r is not None for r in resp):
            return [list(map(list, r)) for r in resp], "forma %s + %s" % (fm[0], d2)
    return None, "muda de tamanho, nenhuma forma serve"


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
