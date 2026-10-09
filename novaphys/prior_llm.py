"""
novaphys.prior_llm — a AGI extrai BITS da LLM, nao texto
==========================================================

A IDEIA DO USUARIO
--------------------
"E se a LLM conseguir ver e interagir com tudo o que acontece dentro da mente da
AGI, enquanto a AGI consegue ler e extrair informacao da LLM?"

A segunda metade e a nova. A AGI nao deve LER TEXTO da LLM — texto e opiniao, e o
usuario ja disse que nao confia num modelo pequeno como autoridade. Ela deve ler
BITS:

    -log2 p_LLM(hipotese | contexto)   E   um comprimento de descricao.

Nao e analogia. Se a LLM da probabilidade alta a uma hipotese, essa hipotese e
BARATA DE ESCREVER no codigo da LLM. O que a humanidade escreveu vira um LIVRO DE
CODIGOS, e o criterio MDL do nucleo consome isso na moeda dele.

E a desconfianca do usuario fica embutida na aritmetica: um prior errado nao
corrompe nada. Ele torna barata uma descricao que paga caro no residuo, e perde.
Nao e preciso confiar na LLM — basta cobrar dela.

O CUSTO, QUE E O QUE TORNA ISTO VIAVEL EM TEMPO REAL
------------------------------------------------------
Para pontuar os 23 codecs eu gero UM TOKEN (a letra da escolha) e leio a
distribuicao de probabilidade sobre ele. Medido: 1,35 s sem cache de prompt,
e o prompt e quase todo prefixo fixo, entao com `cache_prompt` cai muito. Isso e
rapido o bastante para viver DENTRO do laco.

O QUE JA SABEMOS QUE PESA CONTRA
----------------------------------
`proponente.py` deu um nulo limpo: quatro bracos, 40/400 cada, ordenacao nao
move nada. A licao foi "o gargalo e o CONJUNTO de candidatos, nao a ordem". Um
prior sobre os 23 codecs que ja existem mexe no CRITERIO (um codec plausivel fica
genuinamente mais barato e pode ganhar uma disputa apertada), o que e diferente
de reordenar — mas ainda assim opera num conjunto fechado.

Entao o portao que interessa aqui NAO e "ganha tarefas". E B1: o julgamento da
LLM CORRELACIONA com o que o portao de bits acaba escolhendo? Se correlacionar, o
prior vale para codecs NOVOS, onde o conjunto e aberto e o nulo do `proponente`
nao pode se repetir por construcao.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B1  INFORMATIVO: o codec que a busca acaba escolhendo custa, em media, pelo
      menos 1,0 bit A MENOS no prior da LLM que a media sobre os 23 codecs
  B2  GANHA: resolve >= +2 tarefas sobre o nucleo sem prior, teto casado
  B3  PLACEBO: prior EMBARALHADO (os mesmos valores de bits atribuidos a codecs
      ao acaso) nao ganha nada (<= 0)
  B4  NAO QUEBRA: no maximo 1 tarefa perdida
PREVISAO: B1 passa (aposto 1,5 a 3 bits de vantagem). B2 FALHA — aposto +0 ou +1,
pelo mesmo motivo do nulo do `proponente`. B3 passa. B4 passa. Se B1 passar e B2
falhar, a conclusao pre-registrada e: o prior da LLM tem informacao real, mas
gasta-la no alfabeto fechado e desperdicio — ela deve ser aplicada as propostas
NOVAS do `laco_proponente`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

URL = "http://127.0.0.1:8099/completion"
SEMENTE = 20261022

DESCRICAO = {
    "copia": "a saida repete a entrada",
    "mapa_cor": "trocar uma cor por outra, globalmente",
    "preenche": "pintar regiao de fundo cercada",
    "d4": "girar ou espelhar a grade",
    "simetriza": "completar a figura por simetria",
    "atracao": "mover objetos ate encostar",
    "constante": "escrever sempre o mesmo desenho",
    "cruz": "pintar a linha e a coluna de um ponto",
    "cor_por_prop": "escolher a cor pela propriedade do objeto (tamanho, forma)",
    "sovaco": "pintar o canto interno entre dois bracos",
    "buraco_dono": "pintar o buraco com a cor de quem o cerca",
    "periodo": "a saida se repete com um periodo",
    "gravidade": "deixar os objetos cairem",
    "conecta": "ligar pontos da mesma cor por uma linha",
    "contorno": "pintar a borda do objeto",
    "apaga_prop": "apagar objetos com certa propriedade",
    "moldura": "desenhar uma moldura em volta",
    "vizinhanca": "a cor depende das celulas vizinhas",
    "condicao": "regra aprendida com condicao (se isto entao aquilo)",
    "programa": "um programa curto pintado sobre a grade",
    "viz_cor": "decorar cor a partir da vizinhanca",
    "aritmetica": "contar e escrever um numero de celulas",
    "relacao": "relacao binaria entre dois objetos",
}


# ─── o canal de bits ─────────────────────────────────────────────────────────

class Prior:
    """Le a distribuicao da LLM sobre os codecs e devolve BITS."""

    def __init__(self, nomes, embaralha=False, semente=SEMENTE):
        self.nomes = list(nomes)
        self.letra = {n: chr(65 + k) for k, n in enumerate(self.nomes)}
        self.embaralha = embaralha
        self.rng = random.Random(semente)
        self.cache = {}
        self.atual = {}
        self.chamadas = 0
        self.relogio = 0.0

    def _menu(self):
        return "\n".join(f"{self.letra[n]}) {DESCRICAO.get(n, n)}"
                         for n in self.nomes)

    def pergunta(self, contexto):
        if contexto in self.cache:
            self.atual = self.cache[contexto]
            return self.atual
        p = ("Um sistema descreve uma grade colorida a partir de outra, por "
             "compressao. Estas sao as operacoes disponiveis:\n\n"
             + self._menu()
             + f"\n\nSituacao atual:\n{contexto}\n\n"
               "Qual operacao tem mais chance de explicar o que falta?\n"
               "Responda so a letra.\nResposta:")
        d = json.dumps({"prompt": p, "n_predict": 1, "n_probs": 25,
                        "temperature": 1.0, "cache_prompt": True}).encode()
        r = urllib.request.Request(URL, d, {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            o = json.loads(urllib.request.urlopen(r, timeout=120).read())
        except Exception:
            self.atual = {}
            return self.atual
        self.chamadas += 1
        self.relogio += time.time() - t0
        cp = (o.get("completion_probabilities") or [{}])[0]
        bruto = {}
        for e in cp.get("top_logprobs", []):
            t = (e.get("token") or "").strip().upper()
            if len(t) == 1 and "A" <= t <= chr(64 + len(self.nomes)):
                bruto[t] = max(bruto.get(t, 0.0), math.exp(e.get("logprob", -99)))
        # suavizacao de Laplace: nenhum codec pode custar infinito, senao o prior
        # PROIBE em vez de precificar — e proibir e exatamente o que nao se quer
        # de uma fonte em que nao se confia.
        piso = 0.01 / max(len(self.nomes), 1)
        p_ = {n: bruto.get(self.letra[n], 0.0) + piso for n in self.nomes}
        s = sum(p_.values())
        bits = {n: -math.log2(v / s) for n, v in p_.items()}
        if self.embaralha:
            # PLACEBO PAREADO: os mesmos valores de bits, atribuidos ao acaso.
            vals = list(bits.values())
            self.rng.shuffle(vals)
            bits = dict(zip(self.nomes, vals))
        self.cache[contexto] = bits
        self.atual = bits
        return bits

    def contexto(self, treino, pilha, alvo):
        return contexto_de(treino, pilha, alvo)

    def bits_de(self, nome):
        return self.atual.get(nome, math.log2(max(len(self.nomes), 2)))

    def total(self, pilha):
        s = 0.0
        for it in pilha:
            if not D._marca(it):
                s += self.bits_de(it[0].nome)
        return s


def contexto_de(treino, pilha, alvo):
    """O que a LLM ve da 'mente' da AGI, comprimido: dimensoes, cores, o que a
    pilha ja fez, e QUANTO falta. Curto de proposito — a janela e 8k e o estado
    e grande; mostrar tudo nao cabe, entao mostra-se o que surpreende."""
    a0, b0 = treino[0]
    ha, wa = dims(a0)
    hb, wb = dims(b0)
    cores = sorted({c for l in a0 for c in l})
    falta = sum(len(s) for s in alvo) if alvo else hb * wb
    tot = sum(dims(b)[0] * dims(b)[1] for _a, b in treino)
    nomes = [it[0].nome for it in pilha if not D._marca(it)]
    return (f"entrada {ha}x{wa}, saida {hb}x{wb}, cores {cores}\n"
            f"ja aplicado: {' -> '.join(nomes) if nomes else '(nada)'}\n"
            f"faltam {falta} de {tot} celulas por explicar")


# ─── a corrida ───────────────────────────────────────────────────────────────

def roda(tarefas, prior, teto, etiqueta, escolhidos=None):
    D.PRIOR = prior
    ok, t0 = set(), time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y
                                          for x, y in te):
                ok.add(tarefa)
            if escolhidos is not None and prior is not None:
                for it in pilha:
                    if not D._marca(it):
                        escolhidos.append((it[0].nome, prior.bits_de(it[0].nome),
                                           sum(prior.atual.values())
                                           / max(len(prior.atual), 1)))
        except Exception:
            pass
        if (k + 1) % 25 == 0:
            print(f"    [{etiqueta}] {k+1} tarefas, {len(ok)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    D.PRIOR = None
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=100)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]
    nomes = [c.nome for c in D.CODECS]

    print("=" * 78)
    print(f"PRIOR DA LLM EM BITS — {len(tarefas)} tarefas, teto {a.teto}, "
          f"{len(nomes)} codecs")
    print("=" * 78)

    base = roda(tarefas, None, a.teto, "sem prior")
    pr = Prior(nomes)
    esc = []
    com = roda(tarefas, pr, a.teto, "com prior", esc)
    pl = Prior(nomes, embaralha=True)
    plac = roda(tarefas, pl, a.teto, "placebo")

    vant = [media - b for _n, b, media in esc]
    m = sum(vant) / len(vant) if vant else 0.0
    print("")
    print("=" * 78)
    print(f"  sem prior  {len(base):3d}/{len(tarefas)}")
    print(f"  com prior  {len(com):3d}/{len(tarefas)}   "
          f"({pr.chamadas} chamadas, {pr.relogio:.0f}s de LLM)")
    print(f"  placebo    {len(plac):3d}/{len(tarefas)}")
    print(f"\n  codecs escolhidos pela busca: {len(esc)}")
    print(f"  vantagem media do escolhido no prior: {m:+.2f} bits")
    print("")
    print(f"B1 {'.' if m >= 1.0 else 'FALHOU .'} prior informativo: {m:+.2f} "
          f"bits (>= +1,0)")
    print(f"B2 {'.' if len(com)-len(base) >= 2 else 'FALHOU .'} ganha "
          f"{len(com)-len(base):+d} (>= +2)")
    print(f"B3 {'.' if len(plac)-len(base) <= 0 else 'FALHOU .'} placebo "
          f"{len(plac)-len(base):+d} (<= 0)")
    print(f"B4 {'.' if len(base-com) <= 1 else 'FALHOU .'} perdeu "
          f"{len(base-com)} (<= 1)")


if __name__ == "__main__":
    main()
