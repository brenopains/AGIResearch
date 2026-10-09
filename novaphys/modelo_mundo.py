"""
novaphys.modelo_mundo — acumular entre mundos SEM envenenar
=============================================================

A PECA QUE O USUARIO NOMEOU, E QUE EU JA TINHA ADMITIDO NAO EXISTIR
--------------------------------------------------------------------
Existe `fisica.py`, `caderno.py` e as leis aprendidas nos mundos de orbita, e
elas funcionam DENTRO daqueles mundos. Nao existe modelo de mundo por dominio
que seja consultavel, combinavel entre dominios, ou que gere material para a
digestora. Isto e essa peca.

O DESENHO NAO E INVENTADO — ELE SAI DE QUATRO MEDIDAS DESTE PROJETO
---------------------------------------------------------------------
  [[integrar-exige-dose-e-afinidade]]  ruido nao dilui, ENVENENA (-0,127, o
        lexico colapsa ao alfabeto). Entao FUNDIR tudo num acervo so e
        exatamente o que nao se pode fazer.
  [[sob-demanda-funciona]]  guardar separado e ordenar NA HORA DA TAREFA bate
        fundir-tudo por 12-20 desvios. Entao a combinacao e por CONSULTA.
  [[fragmento-transfere]]  estrutura executavel atravessa (73/105 adaptado
        x 4/105 placebo); [[procedencia-atravessa-recombinando]] mediu que
        VOCABULARIO nao atravessa (1/12). Entao a unidade guardada e o
        OPERADOR aplicado, nao o lexico do dominio.
  [[variedade-paga]]  mais fontes MELHORA o ranking (0,456 -> 0,738, nulo
        parado em 0,25). Entao acumular entre mundos tem a chance de pagar.

A MOEDA COMUM ENTRE MUNDOS E O OPERADOR
-----------------------------------------
Para "acumular entre mundos" ser mensuravel, os mundos precisam partilhar
ALGUMA coisa. Aqui a coisa partilhada e o operador: `espelha`, `transpoe`,
`desloca`, `inverte`, `borra`, `limiar` aplicam-se a QUALQUER estado 2-D
quantizado — grade do ARC, imagem do CIFAR, janela de sensor do HAR, texto.

Cada dominio entrega estados crus; todos sao normalizados ao MESMO substrato
(lado x lado, `niveis` niveis, por quantil do proprio estado) e sofrem os
MESMOS operadores. Um fragmento e (operador, dominio de procedencia).

E O VENENO E A FICCAO DO USUARIO, TORNADA MEDIVEL
---------------------------------------------------
"isso tudo depende da AGI conseguir separar o que e ficcao do que e real".
O dominio `ficcao` tem episodios com a MESMA forma dos outros — estado antes,
estado depois, rotulo de operador — mas o rotulo anunciado NAO e o operador
executado. Ele nao e ruido: e estrutura que mente sobre a propria relacao.
Essa e a unica forma de veneno que diluicao nao resolve.

TRES BRACOS, E O TERCEIRO E O QUE INTERESSA
---------------------------------------------
  FUNDIDO   um acervo so, sem rotulo de dominio, topo-K por similaridade.
            E o que [[integrar-exige-dose-e-afinidade]] preve que envenena.
  COTA      cota igual por dominio. Protege por DILUICAO: o veneno so pode
            tomar 1/D dos lugares. Protecao barata e burra — esta aqui
            justamente para nao deixar eu creditar a diluicao ao credito.
  CREDITO   peso por dominio GANHO num split de calibracao: um dominio cujos
            fragmentos nao pagam e rebaixado. E a unica das tres que aprende,
            e a unica que generalizaria para filme e serie.

PRE-REGISTRO — escrito ANTES de rodar
---------------------------------------
  W0  PRE-CONDICAO, duas partes, as duas obrigatorias:
      (a) os dominios continuam DISTINGUIVEIS depois da normalizacao (um
          classificador de dominio fica acima do acaso). Se nao, "entre
          mundos" e trivial porque virou um mundo so.
      (b) sem veneno, a precisao@K CRUZADA (excluindo o proprio dominio)
          fica >= 2x o acaso. Se operador nao atravessa, nao ha o que acumular
          e nada depois significa coisa alguma.
  W1  com veneno: FUNDIDO cai >= 2 desvios E CREDITO segura (queda <= 1
      desvio) -> o credito protege, e acumular entre mundos sem envenenar
      tem MECANISMO, nao so arquitetura. Primeiro positivo desse tipo aqui.
  W2  CREDITO cai junto com FUNDIDO -> separar e creditar nao protege; a
      procedencia e contabilidade, nao defesa.
  W3  FUNDIDO nao cai -> o instrumento nao tem sensibilidade a veneno e a
      rodada nao decide nada (nulo do instrumento, nao do mundo).

PREVISAO: W1 a 0,50. Mecanismo: o credito e medido diretamente sobre o desfecho
que interessa (o fragmento daquele dominio carregava o rotulo certo?), entao o
dominio que mente ganha credito de acaso e desce. Risco nomeado e maior que o
usual: W0(a) pode falhar por normalizacao demais ou W0(b) por normalizacao de
menos — a mesma escolha empurra os dois em direcoes opostas, e eu nao sei de
antemao onde ela cai. W0 falho a 0,30, W2 a 0,15, W3 a 0,05.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import statistics
import struct
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DIM = 4096
SEMENTE = 20260922
LADO = 8          # substrato comum: todo estado vira LADO x LADO
NIVEIS = 4        # ... com NIVEIS niveis, por quantil do PROPRIO estado

RAIZ_ARC = "legacy/novamind/data/ARC/data/training"
RAIZ_CIFAR = "legacy/novamind/data/cifar-10-batches-py"
RAIZ_MODA = "legacy/novamind/data/FashionMNIST/raw"
RAIZ_HAR = r"D:\agi_dados\har\UCI HAR Dataset"
TEXTO = "legacy/novamind/data/train/alice.txt"
FACTORIO = "resultados/episodios_livre.json"


# ---------------------------------------------- substrato comum entre mundos
def normaliza(a: np.ndarray, lado: int = LADO, niveis: int = NIVEIS
              ) -> Optional[np.ndarray]:
    """
    Qualquer estado 2-D -> lado x lado com `niveis` niveis.

    A quantizacao e por QUANTIL DO PROPRIO ESTADO, nao por escala global: e o
    que tira a assinatura de escala do dominio (o CIFAR vai a 255, a grade do
    ARC a 9, o giroscopio e fracionario) sem tirar a estrutura interna.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        return None
    h, w = a.shape
    li = np.linspace(0, h, lado + 1).astype(int)
    lj = np.linspace(0, w, lado + 1).astype(int)
    b = np.zeros((lado, lado))
    for i in range(lado):
        for j in range(lado):
            bl = a[li[i]:max(li[i + 1], li[i] + 1),
                   lj[j]:max(lj[j + 1], lj[j] + 1)]
            b[i, j] = bl.mean() if bl.size else 0.0
    v = b.ravel()
    if np.allclose(v, v[0]):
        return None                      # estado constante nao carrega relacao
    qs = np.quantile(v, np.linspace(0, 1, niveis + 1)[1:-1])
    return np.digitize(b, qs).astype(int)


# ------------------------------------------------------ operadores partilhados
def _espelha(a):
    return a[:, ::-1].copy()


def _transpoe(a):
    return a.T.copy()


def _desloca(a):
    return np.roll(a, 2, axis=1)


def _inverte(a):
    return (NIVEIS - 1 - a)


def _borra(a):
    b = a.astype(float)
    p = np.pad(b, 1, mode="edge")
    m = sum(p[i:i + b.shape[0], j:j + b.shape[1]]
            for i in range(3) for j in range(3)) / 9.0
    return np.clip(np.rint(m), 0, NIVEIS - 1).astype(int)


def _limiar(a):
    return ((a >= (NIVEIS - 1) / 2.0) * (NIVEIS - 1)).astype(int)


OPERADORES: Dict[str, Callable] = {
    "espelha": _espelha, "transpoe": _transpoe, "desloca": _desloca,
    "inverte": _inverte, "borra": _borra, "limiar": _limiar,
}


# ------------------------------------------------------ codificador partilhado
class Espaco:
    """Espaco bipolar por hash do nome — o mesmo para todos os dominios."""

    def __init__(self, dim: int = DIM, semente: int = SEMENTE) -> None:
        self.dim, self.semente = dim, semente
        self._c: Dict[str, np.ndarray] = {}

    def atomo(self, nome: str) -> np.ndarray:
        v = self._c.get(nome)
        if v is None:
            h = hashlib.blake2b(("%d:%s" % (self.semente, nome)).encode(),
                                digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(h, "big"))
            v = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
            self._c[nome] = v
        return v


def codifica(e: Espaco, antes: np.ndarray, depois: np.ndarray) -> np.ndarray:
    """
    Codifica a MUDANCA, nao o conteudo.

    [[nuvem-matriz-encoder]] mediu o preco de errar isto: o encoder que so
    amarrava NIVEL dava teto 62,5 em regras de MUDANCA e elas nao compunham.
    Aqui cada termo amarra POSICAO x nivel-DE x nivel-PARA, entao o que fica no
    vetor e a relacao entrada->saida, que e o que pode atravessar dominio.
    """
    ts = [e.atomo("p:%d,%d" % (i, j)) * e.atomo("de:%d" % antes[i, j])
          * e.atomo("para:%d" % depois[i, j])
          for i in range(antes.shape[0]) for j in range(antes.shape[1])]
    s = np.sum(ts, axis=0)
    return np.where(s >= 0, 1, -1).astype(np.int8)


def sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / a.size


# ------------------------------------------------------------ os mundos, do disco
def _arc(n: int) -> List[np.ndarray]:
    fs = sorted(os.listdir(RAIZ_ARC))[:n]
    out = []
    for f in fs:
        try:
            d = json.load(open(os.path.join(RAIZ_ARC, f)))
        except Exception:
            continue
        for par in d.get("train", [])[:1]:
            out.append(np.array(par["input"]))
    return out


def _cifar(n: int) -> List[np.ndarray]:
    p = os.path.join(RAIZ_CIFAR, "data_batch_1")
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    X = d[b"data"][:n].reshape(-1, 3, 32, 32).mean(axis=1)
    return [x for x in X]


def _idx(caminho: str, n: int) -> List[np.ndarray]:
    with open(caminho, "rb") as fh:
        _m, cnt, lin, col = struct.unpack(">IIII", fh.read(16))
        buf = fh.read(min(n, cnt) * lin * col)
    X = np.frombuffer(buf, dtype=np.uint8).reshape(-1, lin, col)
    return [x for x in X]


def _har(n: int) -> List[np.ndarray]:
    """Janela de sensor: 9 canais x 128 amostras — estado 2-D de verdade."""
    base = os.path.join(RAIZ_HAR, "train", "Inertial Signals")
    canais = sorted(f for f in os.listdir(base) if f.endswith("_train.txt"))
    cs = [np.loadtxt(os.path.join(base, c), max_rows=n) for c in canais]
    return [np.stack([c[i] for c in cs]) for i in range(len(cs[0]))]


def _texto(n: int) -> List[np.ndarray]:
    s = open(TEXTO, encoding="utf-8", errors="ignore").read()
    cod = np.frombuffer(s.encode("utf-8", "ignore")[:n * 256], dtype=np.uint8)
    k = (len(cod) // 256) * 256
    return [x.reshape(16, 16) for x in cod[:k].reshape(-1, 256)]


def _factorio(n: int) -> List[np.ndarray]:
    """Episodios reais do jogo: cada um vira matriz numero-a-numero."""
    if not os.path.exists(FACTORIO):
        return []
    eps = json.load(open(FACTORIO))
    out = []
    for ep in eps:
        vs: List[float] = []

        def anda(o):
            if isinstance(o, dict):
                for v in o.values():
                    anda(v)
            elif isinstance(o, list):
                for v in o:
                    anda(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                vs.append(float(o))
        anda(ep)
        if len(vs) >= 16:
            k = (len(vs) // 4) * 4
            out.append(np.array(vs[:k]).reshape(4, -1))
    return out[:n]


MUNDOS: Dict[str, Callable[[int], List[np.ndarray]]] = {
    "arc": _arc,
    "cifar": _cifar,
    "moda": lambda n: _idx(os.path.join(RAIZ_MODA,
                                        "train-images-idx3-ubyte"), n),
    "har": _har,
    "texto": _texto,
    "factorio": _factorio,
}


# ------------------------------------------------------------------- o atlas
class Dominio:
    """Modelo de mundo de UM dominio: fragmentos com procedencia e credito."""

    def __init__(self, nome: str) -> None:
        self.nome = nome
        self.vetores: List[np.ndarray] = []
        self.operadores: List[str] = []
        self.credito: float = 1.0

    def guarda(self, v: np.ndarray, op: str) -> None:
        self.vetores.append(v)
        self.operadores.append(op)

    def matriz(self) -> np.ndarray:
        return np.array(self.vetores, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.vetores)


class Atlas:
    """
    A colecao de modelos de mundo. Nao funde: guarda separado com procedencia
    e combina NA CONSULTA — que e o desenho que [[sob-demanda-funciona]] mediu.
    """

    def __init__(self, e: Optional["Espaco"] = None) -> None:
        self.dominios: Dict[str, Dominio] = {}
        self._e = e or Espaco()

    def dominio(self, nome: str) -> Dominio:
        return self.dominios.setdefault(nome, Dominio(nome))

    def guarda(self, dominio: str, v: np.ndarray, op: str) -> None:
        self.dominio(dominio).guarda(v, op)

    def _scores(self, sonda: np.ndarray, exclui: str
                ) -> List[Tuple[np.ndarray, Dominio]]:
        """Similaridade em bloco por dominio — o proprio dominio sai (CRUZADO)."""
        out = []
        for nd, d in self.dominios.items():
            if nd == exclui or not len(d):
                continue
            out.append((d.matriz() @ sonda.astype(np.int32) / sonda.size, d))
        return out

    # --- os tres bracos de consulta -----------------------------------------
    def fundido(self, sonda: np.ndarray, k: int, exclui: str) -> List[str]:
        c: List[Tuple[float, str]] = []
        for s, d in self._scores(sonda, exclui):
            c += list(zip(s.tolist(), d.operadores))
        c.sort(key=lambda x: -x[0])
        return [op for _s, op in c[:k]]

    def cota(self, sonda: np.ndarray, k: int, exclui: str) -> List[str]:
        """Protecao por DILUICAO: cada dominio contribui no maximo k/D."""
        blocos = self._scores(sonda, exclui)
        if not blocos:
            return []
        por = max(1, k // len(blocos))
        c: List[Tuple[float, str]] = []
        for s, d in blocos:
            par = sorted(zip(s.tolist(), d.operadores), key=lambda x: -x[0])
            c += par[:por]
        c.sort(key=lambda x: -x[0])
        return [op for _s, op in c[:k]]

    def credito(self, sonda: np.ndarray, k: int, exclui: str) -> List[str]:
        """
        Peso GANHO por dominio multiplica a similaridade. O dominio que mente
        ganhou credito de acaso na calibracao e desce sozinho — sem eu precisar
        dizer qual e o mentiroso.
        """
        c: List[Tuple[float, str]] = []
        for s, d in self._scores(sonda, exclui):
            c += list(zip((s * d.credito).tolist(), d.operadores))
        c.sort(key=lambda x: -x[0])
        return [op for _s, op in c[:k]]

    def calibra_surpresa(self, sondas: Sequence[Tuple[np.ndarray, np.ndarray, str]],
                         k: int) -> None:
        """
        CREDITO SEM ROTULO: o desfecho e a SURPRESA QUE O FRAGMENTO REMOVE.

        De onde vem. [[modelo-mundo-credito]] fechou com um limite que travava
        filme e video: o credito exigia sonda rotulada, e um filme cru nao tem
        rotulo. O usuario apontou a saida na propria divagacao — "se uma celula
        se comporta de X maneira do nada, quebrando o padrao, e mais facil de
        lembrar... esses 1% de surpresa". [[laco-autonomo]] ja tinha MEDIDO esse
        mecanismo noutro dominio (SURPRESA contra o ruido pegou a crise 10/10 e
        a troca de lei 9/10); ele nunca tinha sido ligado ao credito.

        Como fica. A sonda e so (antes, depois) — dois quadros, nenhum rotulo.
        Cada dominio propoe seus K fragmentos mais parecidos; o operador de cada
        um e APLICADO em `antes` e comparado com `depois`. O ganho e quanto de
        erro isso tira em relacao a prever que nada muda:

            ganho = (erro_parado - erro_do_fragmento) / erro_parado

        Um dominio que mente propoe o operador errado, a previsao piora, o ganho
        vai a zero e ele desce — sem ninguem dizer que ele mente, e sem rotulo.
        """
        ganhos: Dict[str, List[float]] = defaultdict(list)
        for antes, depois, dom_sonda in sondas:
            sv = codifica(self._e, antes, depois)
            parado = float(np.mean(antes != depois))
            if parado <= 0:
                continue          # nada mudou: nao ha surpresa a remover
            for sc, d in self._scores(sv, dom_sonda):
                top = sorted(zip(sc.tolist(), d.operadores),
                             key=lambda x: -x[0])[:k]
                if not top:
                    continue
                es = []
                for _s, o in top:
                    try:
                        prev = OPERADORES[o](antes)
                    except Exception:
                        continue
                    if prev.shape == depois.shape:
                        es.append(float(np.mean(prev != depois)))
                if es:
                    ganhos[d.nome].append((parado - statistics.mean(es)) / parado)
        for nd, d in self.dominios.items():
            g = statistics.mean(ganhos[nd]) if ganhos.get(nd) else 0.0
            d.credito = max(0.0, g)

    def calibra(self, sondas: Sequence[Tuple[np.ndarray, str, str]],
                k: int) -> None:
        """
        Credito medido, nao declarado: para cada sonda de calibracao, cada
        dominio propoe seus K melhores e ganha a fracao que acertou o operador.
        Nada aqui sabe qual dominio e o veneno.
        """
        acertos: Dict[str, List[float]] = defaultdict(list)
        for sv, op_certo, dom_sonda in sondas:
            for s, d in self._scores(sv, dom_sonda):
                top = sorted(zip(s.tolist(), d.operadores),
                             key=lambda x: -x[0])[:k]
                if top:
                    acertos[d.nome].append(
                        sum(1 for _s, o in top if o == op_certo) / len(top))
        acaso = 1.0 / len(OPERADORES)
        for nd, d in self.dominios.items():
            m = statistics.mean(acertos[nd]) if acertos.get(nd) else acaso
            # credito relativo ao acaso, piso em 0: quem nao supera o acaso nao
            # vale voto. Sem teto arbitrario — o dado define.
            d.credito = max(0.0, (m - acaso) / max(1e-9, 1.0 - acaso))


# --------------------------------------------------------------- o experimento
N_CRUS = 60        # estados crus por dominio
N_MIN = 10         # dominio com menos que isso nao participa (e fica dito)
K = 8
FRAC_CAL = 0.4     # fracao das sondas que vira calibracao de credito


def _episodios(nome: str, crus: Sequence[np.ndarray], e: Espaco
               ) -> List[Tuple[np.ndarray, str]]:
    out = []
    for a in crus:
        n = normaliza(a)
        if n is None:
            continue
        for op, fn in OPERADORES.items():
            d = fn(n)
            if d.shape != n.shape:
                continue
            out.append((codifica(e, n, d), op))
    return out


def _distinguiveis(estados: Dict[str, List[np.ndarray]]) -> Tuple[float, float]:
    """
    W0(a): depois da normalizacao, da ainda para dizer de que mundo veio?
    Centroide mais proximo, deixando-um-de-fora. Se isto der acaso, os mundos
    viraram um mundo so e 'entre mundos' nao quer dizer nada.
    """
    nomes = sorted(estados)
    X, y = [], []
    for i, nm in enumerate(nomes):
        for a in estados[nm]:
            X.append(a.ravel().astype(float))
            y.append(i)
    X, y = np.array(X), np.array(y)
    cert = 0
    for i in range(len(X)):
        m = np.ones(len(X), dtype=bool)
        m[i] = False
        cs = np.array([X[m][y[m] == c].mean(axis=0) for c in range(len(nomes))])
        cert += int(np.argmin(((cs - X[i]) ** 2).sum(axis=1)) == y[i])
    return cert / len(X), 1.0 / len(nomes)


def _precisao(ops: Sequence[str], certo: str) -> float:
    return sum(1 for o in ops if o == certo) / max(len(ops), 1) if ops else 0.0


def experimento(semente: int = SEMENTE) -> None:
    rnd = random.Random(semente)
    e = Espaco()
    print("=" * 78)
    print("MODELO DE MUNDO — acumular entre mundos SEM envenenar")
    print("=" * 78)

    # ---- os mundos, do disco
    estados: Dict[str, List[np.ndarray]] = {}
    for nome, fn in MUNDOS.items():
        try:
            crus = fn(N_CRUS)
        except Exception as ex:
            print("  %-9s FORA: %s" % (nome, str(ex)[:50]))
            continue
        ns = [x for x in (normaliza(a) for a in crus) if x is not None]
        if len(ns) < N_MIN:
            print("  %-9s FORA: so %d estados (minimo %d)" % (nome, len(ns), N_MIN))
            continue
        estados[nome] = ns
    print("  mundos: %s" % ", ".join("%s(%d)" % (n, len(v))
                                     for n, v in sorted(estados.items())))

    # ---- W0(a): os mundos continuam distinguiveis?
    ac, acaso_d = _distinguiveis(estados)
    print("\n  W0(a) dominio reconhecivel apos normalizar: %.3f (acaso %.3f)"
          % (ac, acaso_d))
    if ac < 2 * acaso_d:
        print("  W0(a) FALHOU: a normalizacao apagou os mundos. 'Entre mundos'")
        print("        seria trivial — rodada invalida.")
        return
    print("        . os mundos seguem distintos; cruzar e nao-trivial.")

    # ---- atlas limpo
    atlas = Atlas()
    sondas: List[Tuple[np.ndarray, str, str]] = []
    for nome, ns in sorted(estados.items()):
        corte = int(len(ns) * 0.7)
        for v, op in _episodios(nome, ns[:corte], e):
            atlas.guarda(nome, v, op)
        for v, op in _episodios(nome, ns[corte:], e):
            sondas.append((v, op, nome))
    rnd.shuffle(sondas)
    ncal = int(len(sondas) * FRAC_CAL)
    cal, aval = sondas[:ncal], sondas[ncal:]
    print("  atlas: %d fragmentos; sondas %d (calibra %d, avalia %d)"
          % (sum(len(d) for d in atlas.dominios.values()), len(sondas),
             len(cal), len(aval)))

    acaso = 1.0 / len(OPERADORES)
    print("  %d operadores -> acaso = %.3f; precisao@%d CRUZADA (o proprio"
          % (len(OPERADORES), acaso, K))
    print("  dominio sai do ranking, senao mede memoria e nao travessia)\n")

    # ---- W0(b): sem veneno
    atlas.calibra(cal, K)
    limpo = {m: [] for m in ("FUNDIDO", "COTA", "CREDITO")}
    for sv, op, dom in aval:
        limpo["FUNDIDO"].append(_precisao(atlas.fundido(sv, K, dom), op))
        limpo["COTA"].append(_precisao(atlas.cota(sv, K, dom), op))
        limpo["CREDITO"].append(_precisao(atlas.credito(sv, K, dom), op))
    p0 = statistics.mean(limpo["FUNDIDO"])
    print("  SEM VENENO")
    for m in ("FUNDIDO", "COTA", "CREDITO"):
        print("    %-8s %.3f +- %.3f  (%.1fx o acaso)"
              % (m, statistics.mean(limpo[m]), statistics.pstdev(limpo[m]),
                 statistics.mean(limpo[m]) / acaso))
    if p0 < 2 * acaso:
        print("\n  W0(b) FALHOU: %.3f < 2x acaso. Operador NAO atravessa" % p0)
        print("        dominio — nao ha o que acumular entre mundos.")
        return
    print("  W0(b) . operador atravessa: %.1fx o acaso." % (p0 / acaso))

    # ---- o veneno: ficcao — mesma forma, rotulo que MENTE
    ficcao = []
    todos = [a for ns in estados.values() for a in ns]
    rnd.shuffle(todos)
    tam = max(len(d) for d in atlas.dominios.values())
    ops = list(OPERADORES)
    i = 0
    while len(ficcao) < tam and i < len(todos):
        n = todos[i % len(todos)]
        i += 1
        executado = rnd.choice(ops)
        d = OPERADORES[executado](n)
        if d.shape != n.shape:
            continue
        anunciado = rnd.choice([o for o in ops if o != executado])
        ficcao.append((codifica(e, n, d), anunciado))
    for v, op in ficcao:
        atlas.guarda("ficcao", v, op)
    print("\n  veneno: dominio 'ficcao' com %d fragmentos — estrutura real," % len(ficcao))
    print("          rotulo que MENTE sobre o operador executado.")

    # ---- recalibra COM o veneno dentro (nada sabe qual e o mentiroso)
    atlas.calibra(cal, K)
    print("  credito ganho por dominio: %s"
          % {n: round(d.credito, 3) for n, d in sorted(atlas.dominios.items())})

    sujo = {m: [] for m in ("FUNDIDO", "COTA", "CREDITO")}
    for sv, op, dom in aval:
        sujo["FUNDIDO"].append(_precisao(atlas.fundido(sv, K, dom), op))
        sujo["COTA"].append(_precisao(atlas.cota(sv, K, dom), op))
        sujo["CREDITO"].append(_precisao(atlas.credito(sv, K, dom), op))

    print("\n  COM VENENO (queda PAREADA por sonda — [[piso-de-ruido-e-pareamento]])")
    quedas = {}
    for m in ("FUNDIDO", "COTA", "CREDITO"):
        ds = [a - b for a, b in zip(limpo[m], sujo[m])]
        md, sd = statistics.mean(ds), statistics.pstdev(ds) / (len(ds) ** 0.5)
        quedas[m] = (md, sd)
        print("    %-8s %.3f -> %.3f   queda %+.3f +- %.3f  (%.1f desvios)"
              % (m, statistics.mean(limpo[m]), statistics.mean(sujo[m]),
                 md, sd, md / sd if sd > 0 else 0.0))

    print("\n" + "=" * 78)
    qf = quedas["FUNDIDO"][0] / (quedas["FUNDIDO"][1] or 1e-9)
    qc = quedas["CREDITO"][0] / (quedas["CREDITO"][1] or 1e-9)
    qq = quedas["COTA"][0] / (quedas["COTA"][1] or 1e-9)
    if qf < 2:
        print("W3 . o FUNDIDO nao caiu (%.1f desvios): o instrumento nao tem" % qf)
        print("     sensibilidade a este veneno. A rodada nao decide nada.")
    elif qc <= 1:
        print("W1 . O CREDITO PROTEGE. Fundido cai %.1f desvios, credito %.1f."
              % (qf, qc))
        print("     Acumular entre mundos sem envenenar tem MECANISMO:")
        print("     credito medido no desfecho, nao arquitetura declarada.")
        print("     (cota, a protecao por diluicao, caiu %.1f desvios)" % qq)
    else:
        print("W2 . separar e creditar NAO protege: fundido cai %.1f desvios," % qf)
        print("     credito cai %.1f. A procedencia e contabilidade, nao defesa." % qc)


if __name__ == "__main__":
    experimento()


# ------------------------------------------------- a dose: ficcao NAO mente 100%
def dose(semente: int = SEMENTE,
         fracoes=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> None:
    """
    O W1 saiu com credito EXATAMENTE 0 para a ficcao, e isso e suspeito: ali o
    veneno mentia em 100% dos rotulos, entao era perfeitamente identificavel e a
    protecao ficou trivial. Ficcao de verdade nao e assim.

    O portal do Rick Sanchez e falso; que ele atravessa e chega em outro lugar e
    estrutura causal verdadeira. Um filme mente em PARTE. A pergunta que vale:

      D1  o credito acompanha a dose — um dominio que mente em p ganha peso que
          cai com p, e a precisao segura em toda a faixa (queda <= 1 desvio)
          -> a defesa e GRADUADA, e material parcialmente ficcional pode entrar
             sem ser barrado nem envenenar. E o que filme e serie exigem.
      D2  existe uma dose intermediaria onde o credito falha (queda > 2 desvios)
          -> a defesa so funciona no extremo, e ficcao parcial passa
      D3  o credito zera cedo (mente pouco e ja e barrado) -> defesa boa demais:
          descarta fonte util, que e o erro oposto e igualmente caro
    """
    rnd = random.Random(semente)
    e = Espaco()
    estados: Dict[str, List[np.ndarray]] = {}
    for nome, fn in MUNDOS.items():
        try:
            ns = [x for x in (normaliza(a) for a in fn(N_CRUS)) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            estados[nome] = ns

    base = Atlas()
    sondas: List[Tuple[np.ndarray, str, str]] = []
    for nome, ns in sorted(estados.items()):
        corte = int(len(ns) * 0.7)
        for v, op in _episodios(nome, ns[:corte], e):
            base.guarda(nome, v, op)
        for v, op in _episodios(nome, ns[corte:], e):
            sondas.append((v, op, nome))
    rnd.shuffle(sondas)
    ncal = int(len(sondas) * FRAC_CAL)
    cal, aval = sondas[:ncal], sondas[ncal:]
    todos = [a for ns in estados.values() for a in ns]
    tam = max(len(d) for d in base.dominios.values())
    ops = list(OPERADORES)
    acaso = 1.0 / len(OPERADORES)

    print("=" * 78)
    print("A DOSE — e quando a ficcao mente so em PARTE?")
    print("=" * 78)
    print("  um dominio 'ficcao' que mente numa fracao p dos rotulos.")
    print("  p=0 e uma fonte honesta a mais; p=1 e o veneno do W1.\n")
    print("  %6s  %8s  %26s  %26s"
          % ("mente", "credito", "FUNDIDO", "CREDITO"))
    print("  " + "-" * 72)

    limpo = None
    linhas = []
    for p in fracoes:
        at = Atlas()
        for nm, d in base.dominios.items():
            for v, op in zip(d.vetores, d.operadores):
                at.guarda(nm, v, op)
        r2 = random.Random(semente + int(p * 1000))
        i, n_f = 0, 0
        while n_f < tam and i < 10 * len(todos):
            n = todos[i % len(todos)]
            i += 1
            exe = r2.choice(ops)
            d = OPERADORES[exe](n)
            if d.shape != n.shape:
                continue
            anunciado = (r2.choice([o for o in ops if o != exe])
                         if r2.random() < p else exe)
            at.guarda("ficcao", codifica(e, n, d), anunciado)
            n_f += 1
        at.calibra(cal, K)
        pf = [_precisao(at.fundido(sv, K, dm), op) for sv, op, dm in aval]
        pc = [_precisao(at.credito(sv, K, dm), op) for sv, op, dm in aval]
        if limpo is None:
            limpo = (pf[:], pc[:])       # p=0: a referencia de comparacao
        df = [a - b for a, b in zip(limpo[0], pf)]
        dc = [a - b for a, b in zip(limpo[1], pc)]
        sf = statistics.pstdev(df) / (len(df) ** 0.5) or 1e-9
        sc = statistics.pstdev(dc) / (len(dc) ** 0.5) or 1e-9
        cr = at.dominios["ficcao"].credito
        linhas.append((p, cr, statistics.mean(df) / sf,
                       statistics.mean(dc) / sc, statistics.mean(pc)))
        print("  %5.0f%%  %8.3f  %.3f (queda %+.3f, %4.1f sd)  "
              "%.3f (queda %+.3f, %4.1f sd)"
              % (100 * p, cr, statistics.mean(pf), statistics.mean(df),
                 statistics.mean(df) / sf, statistics.mean(pc),
                 statistics.mean(dc), statistics.mean(dc) / sc), flush=True)

    print("\n" + "=" * 78)
    cred = [(p, c) for p, c, _a, _b, _e in linhas]
    monot = all(cred[i][1] >= cred[i + 1][1] - 0.05 for i in range(len(cred) - 1))
    pior = max(abs(d) for _p, _c, _f, d, _e in linhas)
    zera_cedo = any(c <= 0.01 for p, c in cred if p <= 0.4)
    print("  credito por dose: %s"
          % ", ".join("p=%.0f%%:%.2f" % (100 * p, c) for p, c in cred))
    if zera_cedo:
        print("D3 . DEFESA BOA DEMAIS: a ficcao ja e zerada mentindo pouco.")
        print("     Fonte parcialmente util e descartada inteira — o erro")
        print("     oposto, e igualmente caro para filme e serie.")
    elif pior <= 1.0 and monot:
        print("D1 . DEFESA GRADUADA: o credito cai com a dose (%s) e a" % 
              ("monotonico" if monot else "nao-monotonico"))
        print("     precisao segura em toda a faixa (pior queda %.1f desvios)." % pior)
        print("     Material parcialmente ficcional ENTRA sem envenenar —")
        print("     que e exatamente o que filme e serie exigem.")
    else:
        print("D2 . a defesa falha em dose intermediaria: pior queda %.1f" % pior)
        print("     desvios. Ficcao parcial passa pelo credito.")


# ------------------------------------------- o credito sem rotulo, pre-registro
def surpresa(semente: int = SEMENTE, mente: float = 1.0) -> Dict:
    """
    O credito pode vir de PREVISAO em vez de rotulo?

    Isto decide se filme, serie e gameplay cru entram no atlas. Com rotulo, o
    mecanismo ja foi medido ([[modelo-mundo-credito]]: fundido cai 17,4 desvios,
    credito segura em 0,0). Sem rotulo, nada foi medido ainda — e filme nao tem
    rotulo.

    PRE-REGISTRO, escrito antes de rodar
    --------------------------------------
      S0  PRE-CONDICAO. No atlas LIMPO, o credito por surpresa concorda com o
          credito por rotulo (correlacao de ordem >= 0,5 entre os dominios). Se
          discordarem, a surpresa esta medindo outra coisa e o resto nao vale.
      S1  com o dominio que mente: o credito por SURPRESA da a ele credito
          menor que a mediana dos honestos, E a queda de precisao nao passa do
          dobro da queda do credito por ROTULO
          -> o desfecho pode vir de previsao; filme entra sem rotulo
      S2  surpresa nao protege -> nao substitui rotulo, e video cru continua
          travado
      S3  surpresa da credito ALTO ao mentiroso -> pior que nao ter: o sinal
          premia justamente quem engana

    PREVISAO: S1 a 0,65. Mecanismo: quem mente propoe o operador errado, a
    previsao piora e o ganho vai a zero por construcao. Risco nomeado: depois de
    quantizar em 4 niveis, `borra` e `limiar` ficam perto da identidade, entao o
    "erro parado" ja e baixo nessas e o sinal fica fraco — se a maioria das
    sondas cair nessas duas, S0 falha por piso. S2 a 0,25, S3 a 0,05.
    """
    rnd = random.Random(semente)
    e = Espaco()
    estados: Dict[str, List[np.ndarray]] = {}
    for nome, fn in MUNDOS.items():
        try:
            ns = [x for x in (normaliza(a) for a in fn(N_CRUS)) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            estados[nome] = ns

    at = Atlas(e)
    rot: List[Tuple[np.ndarray, str, str]] = []
    cru: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for nome, ns in sorted(estados.items()):
        c = int(len(ns) * 0.7)
        for v, op in _episodios(nome, ns[:c], e):
            at.guarda(nome, v, op)
        for a in ns[c:]:
            for op, fn in OPERADORES.items():
                d = fn(a)
                if d.shape != a.shape:
                    continue
                rot.append((codifica(e, a, d), op, nome))
                cru.append((a, d, nome))
    idx = list(range(len(rot)))
    rnd.shuffle(idx)
    ncal = int(len(rot) * FRAC_CAL)
    cal_r = [rot[i] for i in idx[:ncal]]
    cal_c = [cru[i] for i in idx[:ncal]]
    aval = [rot[i] for i in idx[ncal:]]

    print("=" * 78)
    print("SURPRESA — o credito pode vir de PREVISAO em vez de ROTULO?")
    print("=" * 78)
    print("  a sonda sem rotulo e so (antes, depois): dois quadros.")
    print("  %d fragmentos, %d sondas de calibracao, %d de avaliacao\n"
          % (sum(len(d) for d in at.dominios.values()), len(cal_r), len(aval)))

    at.calibra(cal_r, K)
    c_rot = {n: d.credito for n, d in at.dominios.items()}
    at.calibra_surpresa(cal_c, K)
    c_sur = {n: d.credito for n, d in at.dominios.items()}
    nomes = sorted(c_rot)
    print("  %-10s %10s %10s" % ("dominio", "rotulo", "surpresa"))
    for n in nomes:
        print("  %-10s %10.3f %10.3f" % (n, c_rot[n], c_sur[n]))
    pr = [sorted(nomes, key=lambda x: c_rot[x]).index(n) for n in nomes]
    ps = [sorted(nomes, key=lambda x: c_sur[x]).index(n) for n in nomes]
    mr, ms = statistics.mean(pr), statistics.mean(ps)
    num = sum((a - mr) * (b - ms) for a, b in zip(pr, ps))
    den = (sum((a - mr) ** 2 for a in pr) * sum((b - ms) ** 2 for b in ps)) ** 0.5
    rho = num / den if den else 0.0
    print("\n  S0 correlacao de ordem entre os dois creditos: %.2f" % rho)
    if rho < 0.5:
        print("  S0 FALHOU: a surpresa mede outra coisa. Rodada invalida.")
        return {"veredito": "S0 falhou", "rho": rho}
    print("     . concordam; a surpresa esta medindo o mesmo eixo.")

    base = {}
    at.calibra(cal_r, K)
    base["rotulo"] = [_precisao(at.credito(v, K, d), o) for v, o, d in aval]
    at.calibra_surpresa(cal_c, K)
    base["surpresa"] = [_precisao(at.credito(v, K, d), o) for v, o, d in aval]
    base["fundido"] = [_precisao(at.fundido(v, K, d), o) for v, o, d in aval]

    todos = [a for ns in estados.values() for a in ns]
    tam = max(len(d) for d in at.dominios.values())
    ops = list(OPERADORES)
    r2 = random.Random(semente + 7)
    n_f, posto = 0, 0
    while n_f < tam and posto < 20 * len(todos):
        n = todos[posto % len(todos)]
        posto += 1
        exe = r2.choice(ops)
        d = OPERADORES[exe](n)
        if d.shape != n.shape:
            continue
        anunciado = (r2.choice([o for o in ops if o != exe])
                     if r2.random() < mente else exe)
        at.guarda("ficcao", codifica(e, n, d), anunciado)
        n_f += 1
    print("\n  entra 'ficcao': %d fragmentos, mente em %.0f%% dos rotulos"
          % (n_f, 100 * mente))

    saida = {}
    at.calibra(cal_r, K)
    saida["rotulo"] = ([_precisao(at.credito(v, K, d), o) for v, o, d in aval],
                       {n: d.credito for n, d in at.dominios.items()})
    at.calibra_surpresa(cal_c, K)
    saida["surpresa"] = ([_precisao(at.credito(v, K, d), o) for v, o, d in aval],
                         {n: d.credito for n, d in at.dominios.items()})
    suj_f = [_precisao(at.fundido(v, K, d), o) for v, o, d in aval]

    def _q(a, b):
        ds = [x - y for x, y in zip(a, b)]
        sd = statistics.pstdev(ds) / (len(ds) ** 0.5) or 1e-9
        return statistics.mean(ds), statistics.mean(ds) / sd

    print("\n  %-14s %9s %9s %18s" % ("braco", "limpo", "sujo", "queda pareada"))
    qf = _q(base["fundido"], suj_f)
    print("  %-14s %9.3f %9.3f   %+.3f (%4.1f sd)"
          % ("FUNDIDO", statistics.mean(base["fundido"]),
             statistics.mean(suj_f), qf[0], qf[1]))
    qs = {}
    for nome in ("rotulo", "surpresa"):
        p, cr = saida[nome]
        qs[nome] = _q(base[nome], p)
        print("  %-14s %9.3f %9.3f   %+.3f (%4.1f sd)   ficcao=%.3f"
              % ("CRED/" + nome, statistics.mean(base[nome]),
                 statistics.mean(p), qs[nome][0], qs[nome][1], cr["ficcao"]))

    cr_s = saida["surpresa"][1]
    honestos = [v for k2, v in cr_s.items() if k2 != "ficcao"]
    med_h = statistics.median(honestos)
    print("\n  credito por surpresa: ficcao=%.3f, mediana dos honestos=%.3f"
          % (cr_s["ficcao"], med_h))
    print("=" * 78)
    if cr_s["ficcao"] > med_h:
        ver = "S3"
        print("S3 . PIOR QUE NADA: a surpresa PREMIA o mentiroso.")
    elif qs["surpresa"][0] <= 2 * abs(qs["rotulo"][0]) + 0.02:
        ver = "S1"
        print("S1 . O DESFECHO PODE VIR DE PREVISAO. A ficcao ganha credito")
        print("     %.3f contra mediana %.3f dos honestos, SEM ROTULO NENHUM."
              % (cr_s["ficcao"], med_h))
        print("     Filme, serie e gameplay cru entram no atlas.")
    else:
        ver = "S2"
        print("S2 . surpresa nao substitui rotulo: queda %+.3f contra %+.3f."
              % (qs["surpresa"][0], qs["rotulo"][0]))
    return {"veredito": ver, "rho": rho, "credito_surpresa": cr_s,
            "queda": {k: v[0] for k, v in qs.items()},
            "queda_fundido": qf[0]}


# ---------------------------------------------- o penhasco, e o piso graduado
def calibra_fragmento(self, sondas, k: int, forca: float = 64.0) -> None:
    """
    CREDITO POR FRAGMENTO, com encolhimento para a media do dominio.

    O DEFEITO QUE ISTO CONSERTA. [[credito-por-surpresa]] mediu um penhasco
    entre 20% e 40% de mentira: o credito e UM escalar por dominio com piso
    `max(0, ganho)`, entao uma fonte que mente 40% e descartada INTEIRA, embora
    60% dela seja verdadeiro. Era o modo D3 que eu mesmo tinha pre-registrado —
    defesa boa demais, que descarta fonte util. Filme e o caso obvio: um filme
    e em boa parte fisicamente plausivel e em parte fantasia, e barrar o filme
    inteiro por causa do portal do Rick e o erro caro.

    COMO FICA. Cada FRAGMENTO ganha o proprio peso pela surpresa que removeu
    quando foi recuperado. Fragmento pouco visto tem evidencia rala, entao o
    peso encolhe para a media do dominio na proporcao da evidencia:

        peso = (n*g_proprio + forca*g_dominio) / (n + forca)

    `forca` NAO foi chutada. Com o chute inicial de 4,0 o P0 FALHOU: no atlas
    limpo o peso por fragmento custava 0,034, porque sem veneno todo fragmento
    e honesto e pesar cada um so acrescenta ruido de amostragem. Varri `forca`
    SO NO ATLAS LIMPO, onde o criterio e "nao machucar", e congelei antes de
    olhar o veneno — 4:-0,034 | 16:-0,017 | 64:0,000 | 256:+0,002. Fica 64, o
    menor com custo <= 0,005.

    Com n=0 o fragmento herda o dominio (nenhuma informacao propria); com n
    grande ele passa a valer pelo que fez. E o estimador hierarquico usual, e
    aqui ele e o que remove o penhasco: o dominio que mente 40% perde os
    fragmentos que mentem e MANTEM os que nao mentem.

    Nada aqui usa rotulo: a sonda continua sendo so (antes, depois).
    """
    somas: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    por_dom: Dict[str, List[float]] = defaultdict(list)
    for antes, depois, dom_sonda in sondas:
        sv = codifica(self._e, antes, depois)
        parado = float(np.mean(antes != depois))
        if parado <= 0:
            continue
        for sc, d in self._scores(sv, dom_sonda):
            ordem = np.argsort(-sc)[:k]
            for i in ordem:
                o = d.operadores[int(i)]
                try:
                    prev = OPERADORES[o](antes)
                except Exception:
                    continue
                if prev.shape != depois.shape:
                    continue
                g = (parado - float(np.mean(prev != depois))) / parado
                somas[(d.nome, int(i))].append(g)
                por_dom[d.nome].append(g)

    for nd, d in self.dominios.items():
        gd = statistics.mean(por_dom[nd]) if por_dom.get(nd) else 0.0
        d.credito = max(0.0, gd)
        d.pesos = np.zeros(len(d), dtype=float)
        for i in range(len(d)):
            gs = somas.get((nd, i))
            n = len(gs) if gs else 0
            gp = statistics.mean(gs) if gs else 0.0
            d.pesos[i] = max(0.0, (n * gp + forca * gd) / (n + forca))


def fragmento(self, sonda: np.ndarray, k: int, exclui: str) -> List[str]:
    """Consulta pesando cada fragmento pelo proprio credito."""
    c: List[Tuple[float, str]] = []
    for s, d in self._scores(sonda, exclui):
        p = getattr(d, "pesos", None)
        if p is None:
            p = np.full(len(d), d.credito)
        c += list(zip((s * p).tolist(), d.operadores))
    c.sort(key=lambda x: -x[0])
    return [op for _s, op in c[:k]]


def fragmento_com_origem(self, sonda: np.ndarray, k: int, exclui: str):
    """Igual a `fragmento`, mas devolve (operador, dominio, indice)."""
    c = []
    for s, d in self._scores(sonda, exclui):
        p = getattr(d, "pesos", None)
        if p is None:
            p = np.full(len(d), d.credito)
        c += [(sc, op, d.nome, i)
              for i, (sc, op) in enumerate(zip((s * p).tolist(), d.operadores))]
    c.sort(key=lambda x: -x[0])
    return [(op, nd, i) for _s, op, nd, i in c[:k]]


Atlas.calibra_fragmento = calibra_fragmento
Atlas.fragmento = fragmento
Atlas.fragmento_com_origem = fragmento_com_origem


def penhasco(semente: int = SEMENTE, doses=(0.2, 0.4, 0.6)) -> Dict:
    """
    O piso graduado por fragmento remove o penhasco?

    PRE-REGISTRO
    --------------
      P0  PRE-CONDICAO. No atlas LIMPO o credito por fragmento nao custa: a
          precisao fica >= a do credito por dominio menos 0,02. Se o
          encolhimento ja machuca sem veneno, nao adianta olhar o resto.
      P1  na dose de 40% (onde o dominio zerava): o credito por FRAGMENTO
          recupera fragmentos da ficcao E a maioria deles (>= 70%) sao os
          VERDADEIROS, com precisao >= a do credito por dominio
          -> o penhasco sumiu; fonte parcialmente falsa entra pela parte
             verdadeira, que e o que filme exige
      P2  recupera fragmentos da ficcao mas a maioria e MENTIRA -> o piso
          graduado abriu a porta para o veneno; pior que o penhasco
      P3  nao recupera nenhum -> o encolhimento e forte demais e o penhasco
          continua, so que mais devagar

    PREVISAO: P1 a 0,60. Mecanismo: o fragmento que mente preve mal por
    construcao e o que nao mente preve bem, entao a evidencia separa os dois
    DENTRO do dominio. Risco: com K=8 e poucas sondas, muitos fragmentos sao
    vistos zero vez e herdam o dominio inteiro — se a media do dominio for 0,
    eles herdam 0 e da P3. P3 a 0,25, P2 a 0,15.
    """
    rnd = random.Random(semente)
    e = Espaco()
    estados: Dict[str, List[np.ndarray]] = {}
    for nome, fn in MUNDOS.items():
        try:
            ns = [x for x in (normaliza(a) for a in fn(N_CRUS)) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            estados[nome] = ns

    base = Atlas(e)
    rot, cru = [], []
    for nome, ns in sorted(estados.items()):
        c = int(len(ns) * 0.7)
        for v, op in _episodios(nome, ns[:c], e):
            base.guarda(nome, v, op)
        for a in ns[c:]:
            for op, fn in OPERADORES.items():
                d = fn(a)
                if d.shape == a.shape:
                    rot.append((codifica(e, a, d), op, nome))
                    cru.append((a, d, nome))
    idx = list(range(len(rot)))
    rnd.shuffle(idx)
    ncal = int(len(rot) * FRAC_CAL)
    cal_c = [cru[i] for i in idx[:ncal]]
    aval = [rot[i] for i in idx[ncal:]]
    todos = [a for ns in estados.values() for a in ns]
    tam = max(len(d) for d in base.dominios.values())
    ops = list(OPERADORES)

    print("=" * 78)
    print("O PENHASCO — piso graduado por FRAGMENTO em vez de binario por DOMINIO")
    print("=" * 78)

    # ---- P0: custa alguma coisa no atlas limpo?
    base.calibra_surpresa(cal_c, K)
    p_dom = statistics.mean([_precisao(base.credito(v, K, d), o)
                             for v, o, d in aval])
    base.calibra_fragmento(cal_c, K)
    p_fra = statistics.mean([_precisao(base.fragmento(v, K, d), o)
                             for v, o, d in aval])
    print("  P0 atlas LIMPO: por dominio %.3f | por fragmento %.3f (%+.3f)"
          % (p_dom, p_fra, p_fra - p_dom))
    if p_fra < p_dom - 0.02:
        print("  P0 FALHOU: o encolhimento ja machuca sem veneno.")
        return {"veredito": "P0 falhou"}
    print("     . nao custa; segue.\n")

    print("  %6s %24s %24s %16s"
          % ("mente", "CRED/dominio", "CRED/fragmento", "ficcao no topo"))
    print("  " + "-" * 74)
    linhas = []
    for p in doses:
        at = Atlas(e)
        for nm, d in base.dominios.items():
            for v, op in zip(d.vetores, d.operadores):
                at.guarda(nm, v, op)
        r2 = random.Random(semente + int(p * 1000))
        verdade: Dict[int, bool] = {}
        n_f, posto = 0, 0
        while n_f < tam and posto < 20 * len(todos):
            n = todos[posto % len(todos)]
            posto += 1
            exe = r2.choice(ops)
            dd = OPERADORES[exe](n)
            if dd.shape != n.shape:
                continue
            mentiu = r2.random() < p
            anunciado = (r2.choice([o for o in ops if o != exe])
                         if mentiu else exe)
            verdade[n_f] = not mentiu
            at.guarda("ficcao", codifica(e, n, dd), anunciado)
            n_f += 1

        at.calibra_surpresa(cal_c, K)
        pd_ = statistics.mean([_precisao(at.credito(v, K, d), o)
                               for v, o, d in aval])
        cred_dom = at.dominios["ficcao"].credito
        at.calibra_fragmento(cal_c, K)
        pf_, vindos, honestos = [], 0, 0
        for v, o, d in aval:
            tops = at.fragmento_com_origem(v, K, d)
            pf_.append(_precisao([t[0] for t in tops], o))
            for _op, nd, i in tops:
                if nd == "ficcao":
                    vindos += 1
                    honestos += int(verdade.get(i, False))
        pf_ = statistics.mean(pf_)
        pureza = honestos / vindos if vindos else float("nan")
        linhas.append((p, pd_, cred_dom, pf_, vindos, pureza))
        print("  %5.0f%% %9.3f (cred %.3f) %13.3f %19s"
              % (100 * p, pd_, cred_dom, pf_,
                 ("%d, %.0f%% verdade" % (vindos, 100 * pureza))
                 if vindos else "nenhum"), flush=True)

    print("\n" + "=" * 78)
    alvo = [l for l in linhas if abs(l[0] - 0.4) < 1e-9]
    l = alvo[0] if alvo else linhas[-1]
    _p, pd_, _cd, pf_, vindos, pureza = l
    if vindos == 0:
        ver = "P3"
        print("P3 . o encolhimento e forte demais: em %.0f%% de mentira nenhum"
              % (100 * _p))
        print("     fragmento da ficcao volta. O penhasco continua.")
    elif pureza >= 0.70 and pf_ >= pd_ - 0.005:
        ver = "P1"
        print("P1 . PENHASCO REMOVIDO. Em %.0f%% de mentira o dominio zerava"
              % (100 * _p))
        print("     (credito %.3f) e agora %d fragmentos voltam, %.0f%% deles os"
              % (_cd, vindos, 100 * pureza))
        print("     VERDADEIROS, com precisao %.3f contra %.3f." % (pf_, pd_))
        print("     Fonte parcialmente falsa entra pela parte verdadeira.")
    elif pureza < 0.70:
        ver = "P2"
        print("P2 . o piso graduado ABRIU A PORTA: %d fragmentos da ficcao"
              % vindos)
        print("     voltam e so %.0f%% sao verdade. Pior que o penhasco."
              % (100 * pureza))
    else:
        ver = "P3"
        print("P3 . recupera pouco e nao melhora: %.3f contra %.3f."
              % (pf_, pd_))
    return {"veredito": ver, "linhas": linhas}


# ------------------------------------------------ corroboracao entre dominios
def consenso(self, sonda: np.ndarray, k: int, exclui: str,
             por_dominio: int = 3) -> List[str]:
    """
    Nao "quem e mais parecido", e sim "em quantos MUNDOS INDEPENDENTES isso
    aparece".

    De onde vem. O usuario: "o grounding da crianca sao os pais, a familia, a
    sociedade... nos comparamos com fontes confiaveis pra diferenciar o que tem
    respaldo". Ate aqui cada dominio ganhava credito ISOLADO. Isto e o que ele
    descreveu: dominios se corroborando.

    E resolve uma coisa que o credito nao resolve: **nao precisa de calibracao
    nenhuma.** Sem sonda, sem rotulo, sem desfecho. Um dominio que mente propoe
    operadores que mais ninguem propoe, tira um voto so, e nunca chega ao topo.

    Nao e a COTA disfarcada. A cota limita QUANTAS vagas cada dominio ocupa; o
    consenso exige CONCORDANCIA. Um dominio honesto sozinho numa regiao que so
    ele conhece perde no consenso e ganha na cota — sao mecanismos diferentes, e
    por isso os dois ficam no quadro.
    """
    votos: Dict[str, List[float]] = defaultdict(list)
    for s, d in self._scores(sonda, exclui):
        for i in np.argsort(-s)[:por_dominio]:
            votos[d.operadores[int(i)]].append(float(s[int(i)]))
    # ordena por NUMERO DE MUNDOS que propuseram, e so desempata por forca
    ordem = sorted(votos.items(), key=lambda kv: (-len(kv[1]),
                                                  -statistics.mean(kv[1])))
    saida: List[str] = []
    for op, vs in ordem:
        saida += [op] * min(len(vs), k - len(saida))
        if len(saida) >= k:
            break
    return saida[:k]


Atlas.consenso = consenso


def corrobora(semente: int = SEMENTE, doses=(0.0, 0.2, 0.4, 0.6, 1.0)) -> Dict:
    """
    Corroboracao entre mundos substitui credito calibrado?

    PRE-REGISTRO
    --------------
      C0  PRE-CONDICAO. No atlas LIMPO o consenso nao fica abaixo do credito
          por dominio em mais de 0,02. Se concordancia entre mundos ja custa
          sem veneno, nao serve de defesa.
      C1  em TODAS as doses o consenso segura (queda <= 1 desvio em relacao ao
          proprio limpo) E em 40% e 60% ele nao perde para o credito por
          fragmento, que e onde o penhasco ainda estava
          -> corroboracao e defesa SEM CALIBRACAO: entra fonte nova sem sonda,
             sem rotulo e sem desfecho
      C2  consenso segura so nas doses baixas -> mesmo teto do credito, e o
          penhasco e do instrumento e nao do estimador
      C3  consenso pior que credito -> exigir concordancia descarta o dominio
          que sabe algo sozinho, e o preco nao compensa

    PREVISAO: C1 a 0,55. Mecanismo: quem mente propoe operador que ninguem mais
    propoe, entao fica com um voto num quadro de 5 mundos. Risco nomeado e
    concreto: com 5 dominios e `por_dominio=3`, os operadores faceis (`inverte`,
    `limiar`) vao ser propostos por todo mundo sempre, e o consenso pode virar
    um voto de popularidade que ignora a sonda — nesse caso C0 passa e C1 falha
    por razao errada. Por isso o limpo tambem e reportado por operador.
    """
    rnd = random.Random(semente)
    e = Espaco()
    estados: Dict[str, List[np.ndarray]] = {}
    for nome, fn in MUNDOS.items():
        try:
            ns = [x for x in (normaliza(a) for a in fn(N_CRUS)) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            estados[nome] = ns

    base = Atlas(e)
    rot, cru = [], []
    for nome, ns in sorted(estados.items()):
        c = int(len(ns) * 0.7)
        for v, op in _episodios(nome, ns[:c], e):
            base.guarda(nome, v, op)
        for a in ns[c:]:
            for op, fn in OPERADORES.items():
                d = fn(a)
                if d.shape == a.shape:
                    rot.append((codifica(e, a, d), op, nome))
                    cru.append((a, d, nome))
    idx = list(range(len(rot)))
    rnd.shuffle(idx)
    ncal = int(len(rot) * FRAC_CAL)
    cal_c = [cru[i] for i in idx[:ncal]]
    aval = [rot[i] for i in idx[ncal:]]
    todos = [a for ns in estados.values() for a in ns]
    tam = max(len(d) for d in base.dominios.values())
    ops = list(OPERADORES)

    print("=" * 78)
    print("CORROBORACAO — concordancia entre mundos dispensa calibracao?")
    print("=" * 78)
    print("  o consenso NAO usa sonda, rotulo nem desfecho: so conta em quantos")
    print("  mundos independentes o operador aparece.\n")

    base.calibra_surpresa(cal_c, K)
    p_dom = statistics.mean([_precisao(base.credito(v, K, d), o)
                             for v, o, d in aval])
    p_con = statistics.mean([_precisao(base.consenso(v, K, d), o)
                             for v, o, d in aval])
    print("  C0 atlas LIMPO: credito %.3f | consenso %.3f (%+.3f)"
          % (p_dom, p_con, p_con - p_dom))
    # o risco nomeado: o consenso virou voto de popularidade?
    porop = defaultdict(list)
    for v, o, d in aval:
        porop[o].append(_precisao(base.consenso(v, K, d), o))
    print("     por operador: %s"
          % ", ".join("%s %.2f" % (o, statistics.mean(x))
                      for o, x in sorted(porop.items())))
    if p_con < p_dom - 0.02:
        print("  C0 FALHOU: concordancia ja custa sem veneno.")
        return {"veredito": "C0 falhou", "limpo": (p_dom, p_con)}
    print("     . nao custa; segue.\n")

    print("  %6s %14s %16s %14s"
          % ("mente", "CRED/dominio", "CRED/fragmento", "CONSENSO"))
    print("  " + "-" * 60)
    linhas = []
    lim = {}
    for p in doses:
        at = Atlas(e)
        for nm, d in base.dominios.items():
            for v, op in zip(d.vetores, d.operadores):
                at.guarda(nm, v, op)
        if p > 0:
            r2 = random.Random(semente + int(p * 1000))
            n_f, posto = 0, 0
            while n_f < tam and posto < 20 * len(todos):
                n = todos[posto % len(todos)]
                posto += 1
                exe = r2.choice(ops)
                dd = OPERADORES[exe](n)
                if dd.shape != n.shape:
                    continue
                anunciado = (r2.choice([o for o in ops if o != exe])
                             if r2.random() < p else exe)
                at.guarda("ficcao", codifica(e, n, dd), anunciado)
                n_f += 1
        at.calibra_surpresa(cal_c, K)
        a1 = [_precisao(at.credito(v, K, d), o) for v, o, d in aval]
        at.calibra_fragmento(cal_c, K)
        a2 = [_precisao(at.fragmento(v, K, d), o) for v, o, d in aval]
        a3 = [_precisao(at.consenso(v, K, d), o) for v, o, d in aval]
        if p == 0.0:
            lim = {"dom": a1, "fra": a2, "con": a3}
        def _sd(cur, ref):
            ds = [x - y for x, y in zip(ref, cur)]
            s = statistics.pstdev(ds) / (len(ds) ** 0.5) or 1e-9
            return statistics.mean(ds) / s
        linhas.append((p, statistics.mean(a1), statistics.mean(a2),
                       statistics.mean(a3), _sd(a3, lim["con"])))
        print("  %5.0f%% %14.3f %16.3f %10.3f (%4.1f sd)"
              % (100 * p, statistics.mean(a1), statistics.mean(a2),
                 statistics.mean(a3), _sd(a3, lim["con"])), flush=True)

    print("\n" + "=" * 78)
    # DEFEITO CORRIGIDO: eu usava `abs(l[4])` numa quantidade COM SINAL,
    # entao uma MELHORA de 3,8 desvios entrava como 'pior queda 3,8' e o
    # veredito saiu C3 num resultado que era o oposto. Queda e so o lado
    # positivo; melhora nao e falha.
    pior = max(max(l[4], 0.0) for l in linhas)
    dificeis = [l for l in linhas if l[0] in (0.4, 0.6)]
    ganha = all(l[3] >= l[2] - 0.005 for l in dificeis) if dificeis else False
    if pior <= 1.0 and ganha:
        ver = "C1"
        print("C1 . CORROBORACAO E DEFESA SEM CALIBRACAO. Pior queda %.1f" % pior)
        print("     desvio, e nas doses onde o penhasco estava o consenso nao")
        print("     perde para o credito por fragmento.")
        print("     Fonte nova entra sem sonda, sem rotulo e sem desfecho.")
    elif pior <= 1.0:
        ver = "C2"
        print("C2 . consenso segura (%.1f sd) mas nao bate o credito por" % pior)
        print("     fragmento nas doses dificeis: mesmo teto, e o penhasco e")
        print("     do instrumento, nao do estimador.")
    else:
        ver = "C3"
        print("C3 . exigir concordancia custa caro: pior queda %.1f desvios."
              % pior)
    return {"veredito": ver, "linhas": linhas, "limpo": (p_dom, p_con)}
