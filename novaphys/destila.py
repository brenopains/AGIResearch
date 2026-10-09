"""
novaphys.destila — o Sistema 1 ensina o Sistema 2: a linguagem cresce sozinha
===============================================================================

A PAREDE MAIS ANTIGA DO PROJETO
---------------------------------
[[canal-de-importacao]]: o unico canal de estrutura nova sou eu escrevendo
primitivas. A auditoria de 23/09 confirmou: 23 codecs no nucleo, 23 meus, 0 da
geracao ([[admitir-nao-resolve]]). O usuario perguntou "como tu ainda ta
escrevendo primitiva? ja nao conseguimos extrair, sintetizar e combinar?".

O QUE MUDOU
-------------
Agora existe um Sistema 1 ([[sistema1-transducao]]) que resolve tarefas que a
linguagem explicita NAO alcanca. O diagnostico dessas 13 tarefas:

    INEXPRIMIVEL 10  nenhuma pilha reproduz nem o treino (erro de 4 a 23 celulas)
    AMBIGUO       3  uma pilha reproduz o treino e nao generaliza

Entao "mais exemplos" (um oraculo) resolveria no maximo 3. Para as outras 10 o
conceito nao existe na linguagem. A ligacao certa e mais forte:

    a FUNCAO que o Sistema 1 aprendeu vira um CODEC da linguagem do Sistema 2

Um codec sem nome, com as mesmas propriedades de qualquer outro: paga bits,
e verificavel exatamente, e o `montar` pode COMPO-LO com os codecs explicitos
e com os modelos de forma. Humano faz isso: um conceito aprendido por
intuicao vira peca de raciocinio, e entra em combinacoes que a intuicao
sozinha nao faria. E a literatura chama de "decompilar conhecimento amortizado"
(arXiv 2306.07856).

O TESTE E A COMPOSICAO
------------------------
Nao vale contar a tarefa de onde o codec veio — ele E a solucao dela. Vale
contar tarefas que NEM o nucleo NEM o Sistema 1 sozinhos resolvem e que o
nucleo resolve depois de absorver os conceitos aprendidos. Isso so pode vir de
COMBINAR: conceito aprendido + codec explicito, ou conceito aprendido + forma.

  BASE       os 23 codecs do nucleo
  ABSORVIDO  23 + 13 codecs aprendidos (as redes das 13 tarefas novas)
  PLACEBO    23 + 13 codecs do MESMO tipo, treinados nas MESMAS tarefas com a
             relacao quebrada (desarranjo) — mesma quantidade de codec, mesma
             arquitetura, nenhum conceito

REGUA, declarada: o custo de identidade por slot no `custo` usa
log2(|CODECS global|), nao o tamanho da biblioteca passada — os codecs
absorvidos ganham ~0,65 bit por slot de desconto. ABSORVIDO e PLACEBO tem a
mesma biblioteca (36), entao o vies cancela na comparacao que decide.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  A0  PRE-CONDICAO. Cada codec aprendido reproduz a PROPRIA tarefa de origem
      quando o `montar` o recebe (a traducao rede -> codec nao quebrou nada).
      Exigido em >= 10 das 13.
  A1  A LINGUAGEM CRESCE E COMPOE: ABSORVIDO resolve >= 3 tarefas que nem a
      BASE nem o Sistema 1 sozinho resolviam, fora das 13 de origem, E o
      PLACEBO resolve <= 1 dessas
      -> conceito aprendido por intuicao virou peca de raciocinio e entrou em
         combinacao. A primeira primitiva que eu nao escrevi.
  A2  ABSORVIDO so reproduz as de origem (< 3 novas por composicao)
      -> os conceitos sao especificos demais para compor
  A3  ABSORVIDO PERDE tarefas da BASE (>= 3) -> vocabulario maior piora de
      novo ([[vocabulario-grande-piora]])

PREVISAO: A2 a 0,50, A1 a 0,30, A3 a 0,20. Mecanismo de A2, e por isso ele e o
favorito: cada rede aprendeu UMA tarefa com 3 exemplos; o que ela faz fora da
distribuicao daquela tarefa e arbitrario, e composicao exige que o conceito
funcione em grade alheia. Mecanismo de A1: o `montar` so aceita a pilha se ela
reproduz o treino EXATO — o verificador filtra o arbitrario, e basta um conceito
generalizar para uma tarefa vizinha.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import torch

from novaphys import sistema1 as S
from novaphys.arc import carregar
from novaphys.descricao import CODECS, aplicar, montar
from novaphys.promessa import PASTA

PASTA_REDES = "resultados/redes_destila"
TETO = 4000
PRAZO_S = 60          # por tarefa, IGUAL em todos os bracos

# PRAZO POR TAREFA. O `montar` tem tarefas patologicas (o R4 mostrou 10 que
# passaram de 1 h com teto 4000): no Pool elas rodavam em paralelo, no modo
# sequencial uma trava a fila inteira. `custo` e chamado dentro de try/except no
# `montar`, entao um prazo embutido nele encerra a busca sem matar processo:
# passado o prazo toda avaliacao nova falha, a busca se esgota e o `montar`
# devolve o melhor que tinha. Estourou = nao resolvida, em todos os bracos.
import novaphys.descricao as _D
_CUSTO_ORIG = _D.custo
_PRAZO = [float("inf")]


def _custo_com_prazo(pilha, treino):
    if time.time() > _PRAZO[0]:
        raise TimeoutError("prazo")
    return _CUSTO_ORIG(pilha, treino)


_D.custo = _custo_com_prazo


# ------------------------------------------------ a rede como codec explicito
class Aprendido:
    """
    Um conceito aprendido pelo Sistema 1, com a interface de qualquer codec.
    Roda na CPU (rede pequena) com cache por grade: o `montar` chama `prever`
    muitas vezes na mesma entrada.
    """

    def __init__(self, nome: str, caminho: str, dispositivo: str = "cpu") -> None:
        self.nome = "apr:" + nome
        self.caminho = caminho
        self.disp = dispositivo
        self._rede = None
        self._gira = True
        self._cache: Dict[bytes, dict] = {}

    def _carrega(self):
        if self._rede is None:
            d = torch.load(self.caminho, map_location="cpu", weights_only=False)
            r = S.Rede()
            r.load_state_dict(d["pesos"])
            r.to(self.disp)
            r.eval()
            self._rede, self._gira = r, d["gira"]
        return self._rede

    @torch.no_grad()
    def _grade(self, a) -> np.ndarray:
        g = np.array(a)
        chave = g.tobytes() + bytes(g.shape)
        if chave in self._cache:
            return self._cache[chave]
        rede = self._carrega()
        # As 8 orientacoes numa UNICA passada em lote: o mesmo calculo, e na CPU
        # o lote aproveita muito melhor o BLAS do que 8 passadas separadas.
        ks = list(range(8)) if self._gira else [0]
        gks = [S._d4(g, k) for k in ks]
        lote = torch.tensor(np.stack([S._um_quente(gk) for gk in gks]),
                            device=self.disp)
        lgs = rede(lote).cpu()
        soma = np.zeros((10,) + g.shape, np.float32)
        for k, gk, lg in zip(ks, gks, lgs):
            lg = lg[:, :gk.shape[0], :gk.shape[1]].softmax(0).numpy()
            soma += np.stack([S._d4_inv(lg[c], k) for c in range(10)])
        out = soma.argmax(0)
        self._cache[chave] = out
        return out

    def propor(self, treino, alvo=None):
        try:
            for a, _b in treino:
                g = np.array(a)
                if g.shape[0] > S.LADO or g.shape[1] > S.LADO:
                    return []
            return [None]
        except Exception:
            return []

    def bits(self, m):
        return 1.0                     # o modelo e fixo: nao ha parametro a pagar

    def prever(self, m, a):
        s = self._grade(a)
        return {(i, j): int(s[i, j]) for i in range(s.shape[0])
                for j in range(s.shape[1])}


# ------------------------------------------------------- treinar e guardar
def treina_e_guarda(novas: List[str], placebo: bool) -> Dict[str, str]:
    """Uma rede por tarefa nova (GPU), salva em disco. O placebo usa desarranjo."""
    from novaphys.vigilia_sono import resolve_com_rede
    os.makedirs(PASTA_REDES, exist_ok=True)
    T = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    caminhos = {}
    for n in novas:
        arq = os.path.join(PASTA_REDES, "%s_%s.pt" % (n, "plc" if placebo else "real"))
        if not os.path.exists(arq):
            tr, te = T[n]
            if placebo:
                rnd = random.Random(0)
                pares = [(np.array(a), np.array(b)) for a, b in tr]
                idx = list(range(len(pares)))
                feito = False
                for _ in range(500):
                    rnd.shuffle(idx)
                    if all(idx[i] != i and pares[idx[i]][1].shape == pares[i][0].shape
                           and not np.array_equal(pares[idx[i]][1], pares[i][1])
                           for i in range(len(pares))):
                        feito = True
                        break
                if not feito:
                    continue
                tr = [(pares[i][0].tolist(), pares[idx[i]][1].tolist())
                      for i in range(len(pares))]
                rede = S._treina([(np.array(a), np.array(b)) for a, b in tr],
                                 rnd, True, False)
                gira = True
            else:
                ok, rede, gira = resolve_com_rede(None, tr, te)
                if rede is None:
                    continue
            torch.save({"pesos": {k: v.cpu() for k, v in rede.state_dict().items()},
                        "gira": gira}, arq)
        caminhos[n] = arq
    return caminhos


# --------------------------------------------------- o nucleo, com e sem
_CODS: Dict[str, list] = {}


def _init(extras: Dict[str, List], disp: str = "cpu"):
    torch.set_num_threads(3)
    for braco, pares in extras.items():
        _CODS[braco] = list(CODECS) + [Aprendido(n, c, disp) for n, c in pares]


def _uma(args):
    braco, nome, tr, te = args
    _PRAZO[0] = time.time() + PRAZO_S
    cods = _CODS["ABSORVIDO"] if braco == "BASE" else _CODS[braco]
    # VAZAMENTO CORRIGIDO: o cache de previsao dos codecs aprendidos crescia sem
    # limite (uma tarefa nunca reusa as grades de outra) e levou a RAM livre a
    # 1,3 GB — o mesmo caminho que travou o PC em 24/09. Limpo por tarefa.
    for c in cods:
        if hasattr(c, "_cache"):
            c._cache.clear()
    try:
        fm, pilha, err = montar(tr, codecs=cods, teto=TETO)
        if err != 0 or not pilha:
            return braco, nome, False, []
        ok = all(aplicar((fm, pilha), x) == y for x, y in te)
        usa = [it[0].nome for it in pilha if isinstance(it, tuple)
               and getattr(it[0], "nome", "").startswith("apr:")]
        return braco, nome, ok, usa
    except Exception:
        return braco, nome, False, []


def experimento(nucleos: int = 3, fatia: int = -1, total: int = 1) -> Dict:
    """`fatia`/`total`: roda so as tarefas cujo hash cai nesta fatia, gravando
    num parcial proprio. As duas versoes (real e placebo) de uma mesma tarefa
    ficam SEMPRE na mesma fatia — a comparacao continua pareada."""
    t0 = time.time()
    s1 = json.load(open("resultados/sistema1.json"))
    novas = s1["novas"]
    nucleo = set(json.load(open("resultados/oficial.json")))
    s1_resolve = set(s1["resolve"])
    print("=" * 78)
    print("DESTILA — o Sistema 1 ensina o Sistema 2: a linguagem cresce sozinha?")
    print("=" * 78, flush=True)

    real = treina_e_guarda(novas, placebo=False)
    plc = treina_e_guarda(novas, placebo=True)
    print("  redes: %d conceitos reais, %d placebo | %.0f min"
          % (len(real), len(plc), (time.time() - t0) / 60), flush=True)

    T = list(carregar(PASTA))
    alvo = [(n, tr, te) for n, tr, te in T if n not in nucleo]
    extras = {"ABSORVIDO": sorted(real.items()), "PLACEBO": sorted(plc.items())}
    # BASE no alvo e ZERO por definicao (alvo = o que o nucleo nao resolve), entao
    # o teste de "vocabulario maior piora" (A3) roda o ABSORVIDO nas 105 que a
    # BASE resolve: "BASE" aqui marca esse conjunto, com a biblioteca ABSORVIDA.
    resolvidas = [(n, tr, te) for n, tr, te in T if n in nucleo]
    # A3 (vocabulario maior perde das 105) SAIU desta corrida, declarado: com
    # prazo por tarefa, perder uma tarefa lenta confundiria "o prazo cortou" com
    # "a biblioteca piorou". Passa a ser medido em `mente_unica`.
    # INTERCALADO: cada tarefa roda nos dois bracos em seguida, para que um
    # travamento no meio deixe os dois com o mesmo numero de tarefas feitas.
    tarefas = [t for n, tr, te in alvo
               for t in (("ABSORVIDO", n, tr, te), ("PLACEBO", n, tr, te))]
    print("  alvo: %d tarefas que o nucleo NAO resolve (inclui as 13 de origem)"
          % len(alvo))
    print("  %d processos na CPU, teto %d\n" % (nucleos, TETO), flush=True)

    res = {"ABSORVIDO": {}, "PLACEBO": {}, "BASE": {}}
    parcial = ("resultados/destila_parcial.json" if fatia < 0
               else "resultados/destila_parcial_f%d.json" % fatia)
    if fatia >= 0:
        import zlib
        tarefas = [t for t in tarefas if zlib.crc32(t[1].encode()) % total == fatia]
        base_ja = json.load(open("resultados/destila_parcial.json"))             if os.path.exists("resultados/destila_parcial.json") else {}
        tarefas = [t for t in tarefas if t[1] not in base_ja.get(t[0], {})]
    if os.path.exists(parcial):                    # retoma depois de interrupcao
        res.update(json.load(open(parcial)))
        tarefas = [t for t in tarefas if t[1] not in res[t[0]]]
        print("  retomando: %d ja feitas, %d faltam"
              % (sum(len(v) for v in res.values()), len(tarefas)), flush=True)
    # nucleos=0: UM processo, sequencial, redes na GPU. Mais rapido que o Pool
    # na CPU (centenas de passadas de rede por tarefa a ~0,5 s cada) e com menos
    # RAM — a maquina travou em 24/09 por excesso de processos.
    if nucleos == 0:
        _init(extras, S.DEV)
        fonte = (_uma(t) for t in tarefas)
        pool = None
    else:
        pool = Pool(nucleos, initializer=_init, initargs=(extras,))
        fonte = pool.imap_unordered(_uma, tarefas)
    if True:
        for k, (b, n, ok, usa) in enumerate(fonte):
            res[b][n] = {"ok": ok, "usa": usa}
            if (k + 1) % 10 == 0:
                json.dump(res, open(parcial, "w"))
            if (k + 1) % 50 == 0:
                print("    %4d/%d | ABSORVIDO %d | PLACEBO %d | BASE %d | %.0f min"
                      % (k + 1, len(tarefas),
                         sum(v["ok"] for v in res["ABSORVIDO"].values()),
                         sum(v["ok"] for v in res["PLACEBO"].values()),
                         sum(v["ok"] for v in res["BASE"].values()),
                         (time.time() - t0) / 60), flush=True)
    if pool is not None:
        pool.close()
    json.dump(res, open(parcial, "w"))
    if fatia >= 0:
        print("  fatia %d/%d terminada | %.0f min" % (fatia, total, (time.time() - t0) / 60),
              flush=True)
        return {"fatia": fatia}

    ok = {b: {n for n, v in d.items() if v["ok"]} for b, d in res.items()}
    origem_ok = ok["ABSORVIDO"] & set(novas)
    fora = set(n for n, _, _ in alvo) - set(novas)
    nova_comp = sorted((ok["ABSORVIDO"] & fora) - ok["BASE"] - s1_resolve)
    nova_plc = sorted((ok["PLACEBO"] & fora) - ok["BASE"] - s1_resolve)
    perdeu = []                              # A3 medido em mente_unica
    print("\n" + "=" * 78)
    print("  A0 o codec aprendido reproduz a PROPRIA origem: %d de %d"
          % (len(origem_ok), len(real)))
    print("  fora da origem, que nem o nucleo nem o Sistema 1 resolviam:")
    print("    ABSORVIDO %d %s" % (len(nova_comp), nova_comp))
    print("    PLACEBO   %d %s" % (len(nova_plc), nova_plc))
    for n in nova_comp:
        print("      %s usa %s" % (n, res["ABSORVIDO"][n]["usa"]))
    print("  (A3 — perda nas 105 do nucleo — medido em mente_unica, nao aqui)")
    print("=" * 78)
    if len(origem_ok) < 10:
        ver = "A0 falhou"
        print("A0 FALHOU: a traducao rede -> codec quebrou (%d de %d)."
              % (len(origem_ok), len(real)))
    elif len(perdeu) >= 3:
        ver = "A3"
        print("A3 . vocabulario maior PIORA de novo: perdeu %d da base." % len(perdeu))
    elif len(nova_comp) >= 3 and len(nova_plc) <= 1:
        ver = "A1"
        print("A1 . A LINGUAGEM CRESCEU E COMPOS: %d tarefas novas por combinacao,"
              % len(nova_comp))
        print("     placebo %d. Primeiras primitivas que eu nao escrevi." % len(nova_plc))
    else:
        ver = "A2"
        print("A2 . os conceitos reproduzem a origem mas nao compoem (%d novas,"
              % len(nova_comp))
        print("     placebo %d)." % len(nova_plc))
    r = {"veredito": ver, "origem_ok": sorted(origem_ok), "composicao": nova_comp,
         "placebo": nova_plc, "perdeu": perdeu,
         "contagem": {b: len(v) for b, v in ok.items()}}
    json.dump(r, open("resultados/destila.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()


def junta(total: int = 4) -> Dict:
    """Junta o parcial principal e as fatias e calcula o veredito uma vez so."""
    res = {"ABSORVIDO": {}, "PLACEBO": {}, "BASE": {}}
    arqs = ["resultados/destila_parcial.json"] +         ["resultados/destila_parcial_f%d.json" % k for k in range(total)]
    for a in arqs:
        if os.path.exists(a):
            for b, d in json.load(open(a)).items():
                res.setdefault(b, {}).update(d)
    json.dump(res, open("resultados/destila_parcial.json", "w"))
    for k in range(total):
        a = "resultados/destila_parcial_f%d.json" % k
        if os.path.exists(a):
            os.replace(a, a + ".juntado")
    return experimento(nucleos=0)       # tudo ja feito: so calcula o veredito
