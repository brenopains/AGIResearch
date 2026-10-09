"""
novaphys.critico — o modelo acha defeito no nosso codigo, ou e so alarmista?
=============================================================================

O PEDIDO, LITERAL
-------------------
"Eu havia dito do sistema ler o proprio codigo nao pra modifica-lo ... mas para
IDENTIFICAR FALHAS, rodar suas proprias simulacoes, criar seu proprio modelo
talvez, aprender a programar, a entender contexto mesmo escrito, entender
logica."

`juiz_modelo.py` mostrou que um 4B local reconhece defeitos estruturais deste
projeto quando eles vem DESCRITOS e com cinco alternativas. Isto aqui e o teste
duro: o defeito vem em CODIGO, sem alternativa, e ele tem de dizer se existe.

O DADO — OITO PARES, CADA UM COM PLACEBO EMBUTIDO
---------------------------------------------------
Oito defeitos que este projeto REALMENTE teve, cada um em duas versoes:

    DEFEITUOSA   o codigo como estava quando o defeito estava vivo
    LIMPA        o mesmo codigo depois do conserto

As 16 amostras sao embaralhadas e apresentadas uma a uma, sem dizer qual e qual.
A versao LIMPA e o placebo: um modelo que grita "tem bug" em tudo acerta todas as
defeituosas e ZERA no pareado. E este o modo de falha que interessa, porque um
critico alarmista e pior que nenhum — ele gasta o orcamento de medicao em
fantasma.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C1  NAO E ALARMISTA: marca defeito em no maximo 2 das 8 versoes LIMPAS
  C2  ACHA: marca defeito em pelo menos 5 das 8 versoes DEFEITUOSAS
  C3  PAREADO: acerta o par inteiro (marca a defeituosa E passa a limpa) em
      pelo menos 4 dos 8 pares
  C4  NOMEIA: entre as defeituosas que marcou, pelo menos 3 descricoes nomeiam o
      mecanismo certo (julgado por mim, com as respostas cruas impressas para o
      usuario poder me conferir)
PREVISAO: C1 FALHA — modelo de codigo e treinado a sempre achar o que melhorar, e
aposto que ele marca defeito em 5 a 8 das limpas. C2 passa. C3 FALHA por causa do
C1. C4 passa com 3 a 5. Se C1 passar, o modelo tem CALIBRACAO, que e a
propriedade cara, e o critico automatico vira real.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"
SEMENTE = 20261021

# ─── os oito pares ───────────────────────────────────────────────────────────
# (apelido, mecanismo do defeito, codigo DEFEITUOSO, codigo LIMPO)

PARES = [
 ("teto zero",
  "TETO_CEL = 0 deveria significar SEM TETO, mas 0 // x = 0 e max(0, 40) = 40, "
  "o teto mais restritivo possivel",
  '''
TETO_CEL = 0   # 0 = sem teto

def limite(treino):
    """Quantas avaliacoes esta tarefa pode gastar."""
    por_aval = max(1, sum(len(b) * len(b[0]) for _a, b in treino))
    return max(TETO_CEL // por_aval, 40)
''',
  '''
TETO_CEL = 0   # 0 = sem teto

def limite(treino):
    """Quantas avaliacoes esta tarefa pode gastar."""
    por_aval = max(1, sum(len(b) * len(b[0]) for _a, b in treino))
    if not TETO_CEL:
        return 0                       # sem teto, de verdade
    return max(TETO_CEL // por_aval, 40)
'''),

 ("melhor None",
  "mc e calculado a partir de melhor ANTES de saber que melhor existe; o "
  "TypeError e engolido pelo except e todo candidato e descartado em silencio",
  '''
def escolhe(candidatos, avalia):
    melhor = None
    for c in candidatos:
        try:
            err, bits = avalia(c)
            mc = (melhor[2], melhor[3])
            if melhor is None or (err, bits) < mc:
                melhor = (c, None, err, bits)
        except Exception:
            continue
    return melhor
''',
  '''
def escolhe(candidatos, avalia):
    melhor = None
    for c in candidatos:
        try:
            err, bits = avalia(c)
            if melhor is None:
                melhor = (c, None, err, bits)
            else:
                mc = (melhor[2], melhor[3])
                if (err, bits) < mc:
                    melhor = (c, None, err, bits)
        except Exception:
            continue
    return melhor
'''),

 ("memoria vaza",
  "a memoria registra a tarefa atual ANTES de ser usada para decidir nela: "
  "vazamento, o resultado nao vale",
  '''
def percorre(tarefas, memoria, resolve):
    """Percorre as tarefas usando uma memoria que aprende com o que ja viu."""
    ok = []
    for t in tarefas:
        memoria.registra(t)
        ordem = memoria.ordena(t.candidatos)
        if resolve(t, ordem):
            ok.append(t)
    return ok
''',
  '''
def percorre(tarefas, memoria, resolve):
    """Percorre as tarefas usando uma memoria que aprende com o que ja viu."""
    ok = []
    for t in tarefas:
        ordem = memoria.ordena(t.candidatos)
        bom = resolve(t, ordem)
        memoria.registra(t)            # so DEPOIS de decidir
        if bom:
            ok.append(t)
    return ok
'''),

 ("orcamento torto",
  "os dois bracos comparados recebem tetos de avaliacao diferentes; a diferenca "
  "mede compute, nao qualidade",
  '''
def compara(tarefas, com_biblioteca, sem_biblioteca):
    a = sum(1 for t in tarefas if com_biblioteca(t, teto=1000))
    b = sum(1 for t in tarefas if sem_biblioteca(t, teto=400))
    print(f"com biblioteca {a}, sem {b} -> ganho {a - b}")
    return a - b
''',
  '''
def compara(tarefas, com_biblioteca, sem_biblioteca, teto=400):
    a = sum(1 for t in tarefas if com_biblioteca(t, teto=teto))
    b = sum(1 for t in tarefas if sem_biblioteca(t, teto=teto))
    print(f"com biblioteca {a}, sem {b} -> ganho {a - b}")
    return a - b
'''),

 ("nulo falso",
  "o nulo embaralha os rotulos MAS mantem o pareamento entrada->rotulo, entao "
  "nao destroi a associacao que deveria destruir",
  '''
import random

def nulo(entradas, rotulos, avalia, semente=0):
    """Controle: quanto se consegue quando a associacao e destruida?"""
    r = random.Random(semente)
    pares = list(zip(entradas, rotulos))
    r.shuffle(pares)
    return avalia([x for x, _y in pares], [y for _x, y in pares])
''',
  '''
import random

def nulo(entradas, rotulos, avalia, semente=0):
    """Controle: quanto se consegue quando a associacao e destruida?"""
    r = random.Random(semente)
    emb = list(rotulos)
    r.shuffle(emb)                     # rotulos embaralhados CONTRA as entradas
    return avalia(list(entradas), emb)
'''),

 ("sobrescreve",
  "quem escreve primeiro deveria ganhar, mas o dict.update deixa o codec "
  "seguinte sobrescrever celula ja explicada",
  '''
def aplica(pilha, entrada):
    """Monta a saida celula a celula: o PRIMEIRO codec que reivindica ganha."""
    saida = {}
    for codec, marcador in pilha:
        saida.update(codec.prever(marcador, entrada))
    return saida
''',
  '''
def aplica(pilha, entrada):
    """Monta a saida celula a celula: o PRIMEIRO codec que reivindica ganha."""
    saida = {}
    for codec, marcador in pilha:
        for p, cor in codec.prever(marcador, entrada).items():
            if p not in saida:
                saida[p] = cor
    return saida
'''),

 ("medida tarde",
  "o conceito e comparado com o gabarito do mundo NOVO depois que o mundo "
  "trocou no meio do fluxo; cobra do agente um conceito que ja nao e o certo",
  '''
def avalia_invencao(agente, fluxo, conceito_certo_por_mundo):
    """O agente inventa conceitos ao longo do fluxo; o mundo troca no meio."""
    for obs in fluxo:
        agente.observa(obs)
    certo = conceito_certo_por_mundo[fluxo[-1].mundo]
    return agente.conceito_atual() == certo
''',
  '''
def avalia_invencao(agente, fluxo, conceito_certo_por_mundo):
    """O agente inventa conceitos ao longo do fluxo; o mundo troca no meio."""
    acertos = []
    mundo = fluxo[0].mundo
    for obs in fluxo:
        if obs.mundo != mundo:         # mede ANTES da troca
            acertos.append(agente.conceito_atual()
                           == conceito_certo_por_mundo[mundo])
            mundo = obs.mundo
        agente.observa(obs)
    acertos.append(agente.conceito_atual() == conceito_certo_por_mundo[mundo])
    return acertos
'''),

 ("ganho imediato",
  "a busca so aceita candidato que reduz o residuo AGORA, entao um operador "
  "cujo ganho so aparece no passo seguinte e invisivel",
  '''
def monta(candidatos, avalia, erro0):
    """Busca gulosa: empilha operadores enquanto o residuo cair."""
    pilha, erro = [], erro0
    while True:
        melhor = None
        for c in candidatos:
            e, b = avalia(pilha + [c])
            if e < erro and (melhor is None or (e, b) < melhor[:2]):
                melhor = (e, b, c)
        if melhor is None:
            return pilha, erro
        pilha.append(melhor[2])
        erro = melhor[0]
''',
  '''
def monta(candidatos, avalia, erro0, fechar):
    """Busca gulosa: empilha operadores enquanto o residuo cair.
    Tambem testa cada candidato JA COMPLETADO com o operador de fechamento,
    para enxergar quem so paga no passo seguinte."""
    pilha, erro = [], erro0
    while True:
        melhor = None
        for c in candidatos:
            e, b = avalia(pilha + [c])
            if e > 0:
                e2, b2 = avalia(pilha + [c, fechar])
                if (e2, b2) < (e, b):
                    e, b, c = e2, b2, (c, fechar)
            if e < erro and (melhor is None or (e, b) < melhor[:2]):
                melhor = (e, b, c)
        if melhor is None:
            return pilha, erro
        pilha.extend(melhor[2] if isinstance(melhor[2], tuple) else [melhor[2]])
        erro = melhor[0]
'''),
]

PERGUNTA = """Voce e revisor de um sistema de busca simbolica que descreve dados por
compressao (MDL). Leia a funcao abaixo.

A pergunta NAO e sobre estilo, nomes, tipos, desempenho, ou o que faltou tratar.
A pergunta e: existe aqui um DEFEITO DE LOGICA que faca o sistema produzir um
resultado SISTEMATICAMENTE ERRADO ou uma MEDIDA QUE NAO VALE?

Muitas destas funcoes estao CORRETAS. Dizer que uma funcao correta tem defeito e
tao grave quanto deixar passar uma defeituosa.

```python
{codigo}
```

Responda em no maximo 4 linhas, e TERMINE com uma destas duas palavras sozinha
na ultima linha:
DEFEITO   (se existe defeito de logica)
LIMPO     (se nao existe)"""


def pergunta_modelo(msg, n=3000, temp=0.0):
    d = json.dumps({"messages": [{"role": "user", "content": msg}],
                    "max_tokens": n, "temperature": temp}).encode()
    r = urllib.request.Request(URL, d, {"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(r, timeout=900).read())
    m = o["choices"][0]["message"]
    t = m.get("content") or ""
    if not t.strip():
        t = m.get("reasoning_content") or ""
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def _veredito(txt):
    """Ultima ocorrencia de DEFEITO/LIMPO manda."""
    achados = re.findall(r"\b(DEFEITO|LIMPO)\b", txt.upper())
    return achados[-1] if achados else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--nome", default="modelo")
    a = ap.parse_args()

    amostras = []
    for k, (apelido, mecanismo, ruim, bom) in enumerate(PARES):
        amostras.append((k, apelido, "defeituosa", ruim, mecanismo))
        amostras.append((k, apelido, "limpa", bom, mecanismo))
    random.Random(SEMENTE).shuffle(amostras)

    print("=" * 78)
    print(f"CRITICO — {a.nome}, {len(PARES)} pares (defeituosa + limpa), "
          f"ordem embaralhada")
    print("=" * 78)
    t0, resp = time.time(), {}
    for n, (k, apelido, tipo, codigo, mecanismo) in enumerate(amostras):
        try:
            r = pergunta_modelo(PERGUNTA.format(codigo=codigo.strip()), a.tokens)
        except Exception as e:
            r = f"erro de rede {e}"
        v = _veredito(r)
        resp[(k, tipo)] = (v, r)
        certo = (v == "DEFEITO") if tipo == "defeituosa" else (v == "LIMPO")
        print(f"  [{n+1:2d}/16] {apelido:16s} {tipo:11s} -> {v:8s} "
              f"{'.' if certo else 'X'}", flush=True)

    marcou_ruim = sum(1 for k in range(len(PARES))
                      if resp[(k, "defeituosa")][0] == "DEFEITO")
    marcou_limpa = sum(1 for k in range(len(PARES))
                       if resp[(k, "limpa")][0] == "DEFEITO")
    pareado = sum(1 for k in range(len(PARES))
                  if resp[(k, "defeituosa")][0] == "DEFEITO"
                  and resp[(k, "limpa")][0] == "LIMPO")

    print("")
    print("=" * 78)
    print(f"  marcou defeito em {marcou_ruim}/8 das DEFEITUOSAS")
    print(f"  marcou defeito em {marcou_limpa}/8 das LIMPAS  (alarme falso)")
    print(f"  acertou o PAR inteiro em {pareado}/8   ({time.time()-t0:.0f}s)")
    print("")
    print(f"C1 {'.' if marcou_limpa <= 2 else 'FALHOU .'} nao e alarmista: "
          f"{marcou_limpa} alarmes falsos (<= 2)")
    print(f"C2 {'.' if marcou_ruim >= 5 else 'FALHOU .'} acha: {marcou_ruim}/8 (>= 5)")
    print(f"C3 {'.' if pareado >= 4 else 'FALHOU .'} pareado: {pareado}/8 (>= 4)")
    print("\nC4 e julgado por mim, com as respostas cruas abaixo — o usuario pode")
    print("conferir se a descricao nomeia o mecanismo certo:\n")
    for k, (apelido, mecanismo, _r, _b) in enumerate(PARES):
        v, txt = resp[(k, "defeituosa")]
        print("-" * 78)
        print(f"  {apelido}  (mecanismo real: {mecanismo})")
        print(f"  veredito: {v}")
        print("  " + txt.replace("\n", "\n  ")[:900])


if __name__ == "__main__":
    main()
