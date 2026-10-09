"""
novaphys.mente_unica — uma mente so, que vive sem eu escolher o proximo script
================================================================================

O QUE FALTAVA
---------------
Ate 24/09 cada peca deste projeto era um experimento que EU rodava: eu escolhia
o mundo, o script, quando sonhar, o que medir. O maior humano que sobrou no laco
era eu como orquestrador. Isto e o laco que roda sem mim, juntando as pecas que
ja tem medida:

  SISTEMA 2   linguagem explicita, MDL, verificacao exata    `descricao.montar`
  SISTEMA 1   completa o padrao, escolhe a simetria que os   [[sistema1-transducao]]
              exemplos permitem, treinado na propria tarefa
  FORMA+S1    o Sistema 2 decide a geometria, o 1 preenche    `sistema1.resolve_com_forma`
  ABSORCAO    o que o Sistema 1 resolve e o 2 nao alcanca     [[destila]] (medindo)
              vira CODEC — a linguagem cresce sem mim
  SONHO       tudo que foi resolvido (regra OU rede) gera     [[sonho-ensina]] (D1, +7)
              pares, e o Sistema 1 renasce deles
  MEMORIA     experiencia guardada como DADO, nao conclusao   [[memoria-entrincheira]]

UM DIA NA VIDA DA MENTE
-------------------------
  ACORDADA   escolhe a ordem dos problemas (primeiro os menores: curriculo por
             tamanho, politica simples e DECLARADA como nao medida ainda), tenta
             os dois sistemas, confere com o gabarito — no conjunto de TREINO ha
             correcao, como um estudante com o livro. O que resolveu vira
             gerador de sonho; o que so o Sistema 1 resolveu vira codec.
  DORMINDO   sonha com todos os geradores acumulados e o Sistema 1 renasce.
  EXAME      o conjunto de AVALIACAO, 400 tarefas nunca vistas, SEM gabarito
             para a mente: ela da ate 2 respostas por tarefa (a regra do ARC) e
             so quem corrige e o experimento, de fora.

A RESTRICAO DA MAQUINA, que moldou o codigo
---------------------------------------------
Na manha de 24/09 empilhei 4 processos pesados e travei o PC; perdi horas sem
nenhum resultado gravado. Entao: o Sistema 2 roda num Pool pequeno SEM CUDA
(as redes absorvidas rodam na CPU com cache), o Sistema 1 roda sozinho na GPU,
uma fase de cada vez, e TODO resultado e gravado tarefa a tarefa em
`resultados/mente_unica/`. Retomar depois de um travamento reaproveita o que ja
esta la.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  U0  PRE-CONDICAO. O exame do dia 0 (nucleo com a biblioteca de base +
      Sistema 1 do zero) e medido — e ja e o primeiro numero deste projeto no
      conjunto de AVALIACAO. Sem ele nada depois tem escala.
  U1  A MENTE MELHORA VIVENDO: o exame do dia 2 supera o do dia 0 em >= 5
      tarefas, e o dia 2 supera o dia 1 (a melhora continua, nao e um salto so).
  U2  melhora no dia 1 e para (dia 2 <= dia 1): aprender vale uma vez, nao
      compoe nesta escala.
  U3  nao melhora (dia 1 e dia 2 <= dia 0 + 2): as pecas medidas separadas nao
      somam quando ligadas.
  Um braco PLACEBO (sonhos desarranjados e codecs absorvidos treinados com a
  relacao quebrada) roda DEPOIS, com o mesmo codigo e `--placebo`, para separar
  "aprendeu" de "treinou mais".

PREVISAO: U2 a 0,40, U1 a 0,35, U3 a 0,25.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

PASTA_SAIDA = "resultados/mente_unica"
TETO = 4000


# ------------------------------------------------------------ o problema
class Problema:
    """Exemplos resolvidos + entradas a responder. O gabarito fica FORA da mente."""

    def __init__(self, nome, treino, teste):
        self.nome = nome
        self.treino = treino
        self.entradas = [x for x, _y in teste]
        self._gabarito = [y for _x, y in teste]

    def corrige(self, resposta) -> bool:
        if resposta is None:
            return False
        try:
            return all(np.array_equal(np.array(r), np.array(y))
                       for r, y in zip(resposta, self._gabarito))
        except Exception:
            return False


# ------------------------------------------------ Sistema 2, num Pool sem CUDA
_LIB: List = []


def _init_s2(caminhos_absorvidos):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    from novaphys.descricao import CODECS
    from novaphys.destila import Aprendido
    _LIB.clear()
    _LIB.extend(list(CODECS) + [Aprendido(n, c) for n, c in caminhos_absorvidos])


def _s2_um(args):
    """Uma tentativa do Sistema 2: resposta (ou None), erro no treino, regra."""
    nome, treino, entradas = args
    from novaphys.descricao import aplicar, montar
    for c in _LIB:                        # cache por tarefa: sem isso a RAM vaza
        if hasattr(c, "_cache"):
            c._cache.clear()
    try:
        fm, pilha, err = montar(treino, codecs=_LIB, teto=TETO)
    except Exception:
        return nome, None, 10 ** 6, None
    if err != 0 or not pilha:
        return nome, None, err, None
    try:
        resp = [aplicar((fm, pilha), x) for x in entradas]
    except Exception:
        return nome, None, err, None
    if any(r is None for r in resp):
        return nome, None, err, None
    return nome, [np.array(r).tolist() for r in resp], 0, None


def sistema2(problemas: List[Problema], absorvidos, nucleos: int = 2,
             cache: Optional[str] = None) -> Dict[str, Tuple]:
    feito = {}
    if cache and os.path.exists(cache):
        for linha in open(cache, encoding="utf-8"):
            d = json.loads(linha)
            feito[d["nome"]] = (d["resp"], d["err"])
    falta = [(p.nome, p.treino, p.entradas) for p in problemas if p.nome not in feito]
    if falta:
        with Pool(nucleos, initializer=_init_s2, initargs=(absorvidos,)) as pool:
            for nome, resp, err, _r in pool.imap_unordered(_s2_um, falta):
                feito[nome] = (resp, err)
                if cache:
                    with open(cache, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"nome": nome, "resp": resp,
                                             "err": err}) + "\n")
    return feito


# ------------------------------------------------------ Sistema 1, na GPU
def sistema1(pesos, p: Problema, semente: int = 0):
    """
    Resposta do Sistema 1 SEM olhar gabarito: ajusta nos exemplos, escolhe a
    maior simetria que os reproduz, e responde as entradas. Para tarefa que
    muda de tamanho, pede a geometria ao Sistema 2 (modelo de forma) antes.
    Devolve (resposta ou None, rede, gira, forma usada).
    """
    import torch
    from novaphys import forma
    from novaphys import sistema1 as S
    from novaphys.vigilia_sono import resolve_com_rede

    def _tenta(treino, entradas):
        te_falso = [(x, x) for x in entradas]          # resolve_com_rede exige
        ok, rede, gira = resolve_com_rede(pesos, treino, te_falso, semente)
        if rede is None:
            return None, None, False
        return [S._preve(rede, np.array(x), gira).tolist() for x in entradas], rede, gira

    mesma = all(len(a) == len(b) and len(a[0]) == len(b[0]) for a, b in p.treino)
    if mesma:
        r, rede, gira = _tenta(p.treino, p.entradas)
        return r, rede, gira, None
    try:
        fms = list(forma.propor(p.treino))[:4]
    except Exception:
        return None, None, False, None
    for fm in fms:
        try:
            reb = [(forma.base(fm, a), b) for a, b in p.treino]
            ent = [forma.base(fm, x) for x in p.entradas]
        except Exception:
            continue
        if any(g is None for g, _b in reb) or any(g is None for g in ent):
            continue
        if not all(len(g) == len(b) and len(g[0]) == len(b[0]) for g, b in reb):
            continue
        r, rede, gira = _tenta(reb, ent)
        if r is not None:
            return r, rede, gira, fm
    return None, None, False, None


# ------------------------------------------------------------------ a mente
class Mente:
    def __init__(self, placebo: bool = False) -> None:
        self.placebo = placebo
        self.pesos = None                          # Sistema 1 renascido do sonho
        self.absorvidos: List[Tuple[str, str]] = []  # (nome, caminho da rede)
        self.redes_geradoras: List[Tuple[str, object, bool]] = []
        self.resolvidos: set = set()
        self.s2_exame: set = set()                 # resolvidas pelo Sistema 2 no exame
        self.diario: List[dict] = []

    def _salva_rede(self, nome, rede, gira) -> str:
        import torch
        d = os.path.join(PASTA_SAIDA, "redes")
        os.makedirs(d, exist_ok=True)
        arq = os.path.join(d, "%s_%s.pt" % (nome, "plc" if self.placebo else "real"))
        torch.save({"pesos": {k: v.cpu() for k, v in rede.state_dict().items()},
                    "gira": gira}, arq)
        return arq

    def acordada(self, problemas: List[Problema], dia: int) -> dict:
        """Tenta, confere com o gabarito do livro, guarda, absorve."""
        falta = [p for p in problemas if p.nome not in self.resolvidos]
        # curriculo por tamanho (politica simples, declarada, nao medida)
        falta.sort(key=lambda p: sum(len(a) * len(a[0]) for a, _b in p.treino))
        cache = os.path.join(PASTA_SAIDA, "s2_treino_dia%d%s.jsonl"
                             % (dia, "_plc" if self.placebo else ""))
        if not self.absorvidos:
            # Biblioteca de BASE: o Sistema 2 no treino e exatamente o que ja esta
            # medido em `oficial.json` (105, curva de orcamento plana). Refazer
            # seria recalcular conta pronta. Os resolvidos entram direto; os
            # outros ficam sem resposta do Sistema 2 (err desconhecido).
            of = set(json.load(open("resultados/oficial.json")))
            s2 = {p.nome: (p._gabarito if p.nome in of else None, 0 if p.nome in of else -1)
                  for p in falta}
        else:
            s2 = sistema2(falta, self.absorvidos, cache=cache)
        novos_s2, novos_s1, absorve = 0, 0, 0
        for p in falta:
            r2, _err = s2[p.nome]
            if p.corrige(r2):
                self.resolvidos.add(p.nome)
                novos_s2 += 1
                continue
            r1, rede, gira, fm = sistema1(self.pesos, p)
            if rede is not None and p.corrige(r1):
                self.resolvidos.add(p.nome)
                novos_s1 += 1
                if fm is None:                      # so conceito celula-a-celula
                    self.redes_geradoras.append((p.nome, rede, gira))
                    self.absorvidos.append((p.nome, self._salva_rede(p.nome, rede, gira)))
                    absorve += 1
        return {"s2": novos_s2, "s1": novos_s1, "absorvidos": absorve}

    def dormindo(self, pares_nucleo) -> int:
        """Sonha com TUDO que foi resolvido e o Sistema 1 renasce."""
        from novaphys.sonho import placebo_de, pretreina
        from novaphys.vigilia_sono import sonha_de_redes
        from novaphys.arc import carregar
        from novaphys.promessa import PASTA
        rnd = random.Random(len(self.redes_geradoras))
        grades = [np.array(a) for _n, tr, te in carregar(PASTA) for a, _b in tr + te]
        brutos = list(pares_nucleo) + sonha_de_redes(self.redes_geradoras, grades, rnd)
        real, falso = placebo_de(brutos, random.Random(1))
        self.pesos = pretreina(falso if self.placebo else real)
        return len(real)

    def exame(self, problemas: List[Problema], dia: int,
              ja_s2: Optional[set] = None) -> List[str]:
        """Ate 2 respostas por tarefa (Sistema 2 e Sistema 1). Sem gabarito.
        `ja_s2`: tarefas que o Sistema 2 ja resolveu num dia anterior — nao se
        recalcula (declarado; vocabulario maior poderia perder alguma, e isso e
        medido a parte em [[destila]])."""
        ja_s2 = ja_s2 or set()
        cache = os.path.join(PASTA_SAIDA, "s2_exame_dia%d%s.jsonl"
                             % (dia, "_plc" if self.placebo else ""))
        s2 = sistema2([p for p in problemas if p.nome not in ja_s2],
                      self.absorvidos, cache=cache)
        for n in ja_s2:
            s2[n] = ("JA", 0)
        certos = []
        arq = os.path.join(PASTA_SAIDA, "exame_dia%d%s.jsonl"
                           % (dia, "_plc" if self.placebo else ""))
        ja = {}
        if os.path.exists(arq):
            for linha in open(arq, encoding="utf-8"):
                d = json.loads(linha)
                ja[d["nome"]] = d["ok"]
        for p in problemas:
            if p.nome in ja:
                ok = ja[p.nome]
            else:
                r2, _e = s2[p.nome]
                ok = (r2 == "JA") or p.corrige(r2)   # tentativa 1: Sistema 2
                if ok and r2 != "JA":
                    self.s2_exame.add(p.nome)
                if not ok:
                    r1, _rede, _g, _fm = sistema1(self.pesos, p)
                    ok = p.corrige(r1)          # tentativa 2: Sistema 1
                with open(arq, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"nome": p.nome, "ok": ok}) + "\n")
            if ok:
                certos.append(p.nome)
        return certos


# ---------------------------------------------------------------- a vida
def viver(dias: int = 2, placebo: bool = False, n_exame: Optional[int] = None,
          n_treino: Optional[int] = None) -> Dict:
    from novaphys.arc import carregar
    from novaphys.eter import PASTA_AVAL
    from novaphys.promessa import PASTA
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    etq = "PLACEBO" if placebo else "REAL"
    t0 = time.time()
    treino = [Problema(n, tr, te) for n, tr, te in carregar(PASTA)]
    exame = [Problema(n, tr, te) for n, tr, te in carregar(PASTA_AVAL)]
    if n_exame:
        exame = exame[:n_exame]
    if n_treino:                                   # so para teste rapido do codigo
        treino = treino[:n_treino]
    pares_nucleo = pickle.load(open("resultados/sonho_pares_nucleo.pkl", "rb"))
    m = Mente(placebo)
    hist = {}
    resumo = os.path.join(PASTA_SAIDA, "resumo_%s.json" % etq.lower())

    def grava():
        json.dump({"braco": etq, "exame": hist, "resolvidos_treino": len(m.resolvidos),
                   "absorvidos": [n for n, _c in m.absorvidos],
                   "n_exame": len(exame)}, open(resumo, "w"), indent=1)

    print("[%s] vida: %d problemas de treino (com gabarito), exame de %d NUNCA vistos"
          % (etq, len(treino), len(exame)), flush=True)
    hist[0] = len(m.exame(exame, 0))
    grava()
    print("[%s] dia 0 (nucleo de base + Sistema 1 do zero): EXAME %d/%d | %.0f min"
          % (etq, hist[0], len(exame), (time.time() - t0) / 60), flush=True)
    for dia in range(1, dias + 1):
        a = m.acordada(treino, dia)
        print("[%s] dia %d acordada: +%d pelo S2, +%d pelo S1, %d conceitos absorvidos"
              " (total resolvidos no treino %d) | %.0f min"
              % (etq, dia, a["s2"], a["s1"], a["absorvidos"], len(m.resolvidos),
                 (time.time() - t0) / 60), flush=True)
        n = m.dormindo(pares_nucleo)
        print("[%s] dia %d dormindo: %d pares de sonho | %.0f min"
              % (etq, dia, n, (time.time() - t0) / 60), flush=True)
        hist[dia] = len(m.exame(exame, dia, ja_s2=set(m.s2_exame)))
        grava()
        print("[%s] dia %d: EXAME %d/%d | %.0f min"
              % (etq, dia, hist[dia], len(exame), (time.time() - t0) / 60), flush=True)
    return hist


if __name__ == "__main__":
    n, t = None, None
    for a in sys.argv:
        if a.startswith("--exame="):
            n = int(a.split("=")[1])
        if a.startswith("--treino="):
            t = int(a.split("=")[1])
    viver(placebo="--placebo" in sys.argv, n_exame=n, n_treino=t)
