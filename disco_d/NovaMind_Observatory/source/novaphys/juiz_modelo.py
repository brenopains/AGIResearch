"""
novaphys.juiz_modelo — um modelo local serve de PROPONENTE, ou so de escriba?
==============================================================================

A PERGUNTA
------------
O usuario autorizou baixar um modelo local, e desconfia dele: "uma mente parada
no tempo com dados da humanidade e poucos parametros". A desconfianca e uma
hipotese testavel, nao uma opiniao — entao ela vira portao.

O papel que o modelo teria e ler o codigo do sistema, APONTAR FALHAS, entender
logica e escrever simulacoes. Nunca escrever no nucleo. Logo o que precisa ser
medido nao e "ele sabe Python" (sabe), e sim: ELE RACIOCINA SOBRE UM SISTEMA QUE
NAO ESTAVA NOS DADOS DE TREINO DELE?

O DADO
--------
12 perguntas de multipla escolha (5 alternativas, acaso = 20%) sobre defeitos
REAIS que este projeto encontrou e mediu. Elas nao estao na internet: sao a
historia deste repositorio. Os distratores nao sao absurdos — varios deles sao
exatamente os erros que EU cometi antes de medir (duvida bootstrap como gatilho,
esquecer por desuso, "basta reportar os dois numeros").

6 tarefas de programacao corrigidas por EXECUCAO contra casos escondidos. Esta
familia e o controle de sanidade: se um modelo de codigo falha aqui, o
arcabouco esta errado, nao o modelo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  L0  SANIDADE: o melhor modelo acerta >= 4/6 em PROGRAMACAO
  L1  SERVE DE PROPONENTE: algum modelo acerta >= 8/12 em LOGICA (>= 66%)
  L2  ACIMA DO ACASO: o melhor em LOGICA tem p < 0,01 no binomial contra 20%
      (isto e, >= 6/12)
  L3  RACIOCINIO > CODIGO: o modelo de raciocinio supera o modelo de codigo em
      LOGICA por >= +3 questoes
PREVISAO: L0 passa. L1 FALHA — aposto que o melhor fica entre 4 e 7 de 12. L2
passa. L3 passa. Se L1 falhar, a conclusao pre-registrada e: modelo local NAO e
proponente de arquitetura, e so fonte de informacao e escriba — e o proponente
tem que vir de outro lugar. Se L1 passar, vale construir o laco de proposta.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from math import comb

URL = "http://127.0.0.1:8099/v1/chat/completions"
FATOR = 1.0   # multiplica o orcamento de tokens (modelo que "pensa" precisa de mais)

# ─── familia LOGICA: defeitos reais deste projeto ────────────────────────────
# (enunciado, [alternativas], indice da certa)

LOGICA = [
 ("Uma busca gulosa avalia todos os candidatos e so aceita o que REDUZ o residuo "
  "agora; se nenhum reduz, ela para. Que tipo de operador esta busca nunca vai "
  "escolher, mesmo que ele esteja na lista de candidatos?",
  ["Um que reduz o residuo mais que todos os outros",
   "Um que nao muda o residuo agora, mas faz um candidato seguinte reduzir muito",
   "Um que custa muitos bits mas zera o residuo",
   "Um que aparece no fim da lista de candidatos",
   "Um que so se aplica quando entrada e saida tem a mesma dimensao"], 1),

 ("O teto de busca de cada tarefa e calculado como lim = ORCAMENTO // (total de "
  "celulas das saidas de treino). Comparando duas tarefas de mesma dificuldade, "
  "qual delas recebe MENOS busca?",
  ["A que tem menos pares de treino",
   "A que tem mais celulas no total nas saidas de treino",
   "A que usa mais cores distintas",
   "A que tem entrada e saida de mesma dimensao",
   "Nenhuma: o teto e o mesmo para todas"], 1),

 ("Um sistema so aceita uma descricao se ela reproduz a saida EXATAMENTE (residuo "
  "zero). Agora 3% das celulas do dado estao corrompidas ao acaso. O que acontece?",
  ["A busca fica mais lenta mas ainda encontra a regra verdadeira",
   "A busca nao aceita descricao nenhuma, nem a regra verdadeira",
   "A busca encontra uma regra mais simples que a verdadeira",
   "O ruido e absorvido pelos bits de excecao sem custo",
   "A busca encontra a regra certa se o ruido estiver agrupado no espaco"], 1),

 ("Numa pilha de codecs cada celula e escrita pelo PRIMEIRO codec da pilha que a "
  "reivindica, e os seguintes nao podem sobrescrever. Qual e a consequencia?",
  ["A ordem da pilha e irrelevante porque o total de bits e o mesmo",
   "Um codec geral posto cedo pode inutilizar codecs especificos postos depois",
   "Codecs postos depois sempre custam mais bits",
   "A ordem so importa quando dois codecs propoem a mesma mascara",
   "O ultimo codec da pilha determina o residuo final"], 1),

 ("Um agente percorre um fluxo de observacoes e no meio do fluxo o mundo troca de "
  "lei. No FIM do fluxo, o experimento mede qual conceito o agente esta usando e "
  "compara com o conceito certo. Da 0/10, e o veredito diz que inventar conceito "
  "falhou. Qual e o defeito do EXPERIMENTO?",
  ["Faltaram observacoes",
   "A medida e feita depois da troca, entao cobra um conceito que ja nao e o certo",
   "O agente deveria ser reiniciado a cada observacao",
   "Faltou um nulo permutado",
   "Conceito certo nao e uma nocao bem definida"], 1),

 ("O custo de cada operador e aprendido pelo uso: quanto mais usado, mais barato. "
  "Uma teoria errada foi aprendida ANTES de o conceito certo existir. O que "
  "acontece quando o conceito certo finalmente aparece?",
  ["Ele e adotado na hora, porque comprime mais",
   "A teoria errada ficou barata pelo uso e continua ganhando a comparacao de bits",
   "O custo aprendido nao entra na comparacao",
   "Esquecer operadores por desuso resolve",
   "O problema some sozinho com mais dados"], 1),

 ("Uma biblioteca de abstracoes aprendidas leva o sistema de 51 para 55 tarefas. "
  "Qual controle decide se o ganho vem das ABSTRACOES e nao de simplesmente haver "
  "mais candidatos?",
  ["Repetir com outra semente",
   "Repetir com o mesmo numero de candidatos, mas gerados ao acaso (placebo pareado)",
   "Aumentar o numero de tarefas",
   "Medir bits em vez de tarefas resolvidas",
   "Comparar o tempo de execucao dos dois"], 1),

 ("O conceito A comprime os dados melhor que o conceito B, mas a lei do mundo "
  "precisa de B. O que se conclui?",
  ["A e o conceito certo, porque MDL e o criterio correto",
   "Comprimir mais nao implica ser o termo de que a proxima lei precisa",
   "B esta errado justamente porque comprime pior",
   "A e B sao equivalentes a menos de uma constante",
   "O criterio MDL deve ser trocado por maxima verossimilhanca"], 1),

 ("Um sistema aprendeu, de uma amostra facil, uma regra proposicional que confunde "
  "ganhar com empatar, e satura em 95%. Qual mudanca conserta — de LINGUAGEM, nao "
  "de quantidade de dado?",
  ["Mais features proposicionais",
   "Um quantificador (existe lance tal que para todo lance do outro ...)",
   "Peso maior para as posicoes dificeis",
   "Uma rede mais profunda",
   "Normalizar as features"], 1),

 ("Como um agente decide SOZINHO que sua lei atual quebrou, sem calendario e sem "
  "ninguem avisar?",
  ["Quando o residuo e diferente de zero",
   "Quando o residuo passa do que o ruido conhecido do instrumento explicaria",
   "Quando um numero fixo de observacoes se acumulou",
   "Quando a reamostragem dos dados mostra variancia alta",
   "Quando duas leis rivais discordam entre si"], 1),

 ("Depois de inventar um conceito novo, o que recupera as falhas ANTIGAS que o "
  "sistema ja tinha cometido?",
  ["Guardar os dados antigos e reanalisa-los sem as leis velhas",
   "Guardar as leis antigas e reencaixa-las nos dados novos",
   "Descartar os dados antigos, que foram lidos com a lei errada",
   "Dar peso maior as observacoes recentes",
   "Nada: falha antiga e irrecuperavel"], 0),

 ("O braco A resolve mais tarefas que o braco B, mas A avaliou 3x mais candidatos. "
  "O que essa comparacao mostra sobre a qualidade de A?",
  ["Que A e melhor que B",
   "Nada: o orcamento nao esta casado",
   "Que B e melhor, por gastar menos compute",
   "Que a diferenca deve ser dividida por 3",
   "Que basta aumentar o numero de tarefas para decidir"], 1),
]

# ─── familia PROGRAMACAO: corrigida por execucao ─────────────────────────────

CODIGO = [
 ("permuta_cores(g, mapa)",
  "Escreva `def permuta_cores(g, mapa):` que recebe uma grade (lista de listas de "
  "inteiros) e um dicionario cor->cor, e devolve uma NOVA grade com cada cor "
  "substituida pelo valor do mapa. Cor ausente do mapa fica como esta.",
  [("permuta_cores([[1,2],[2,3]], {1:5,2:0})", [[5, 0], [0, 3]]),
   ("permuta_cores([[0]], {})", [[0]]),
   ("permuta_cores([[1,1,1]], {1:2})", [[2, 2, 2]]),
   ("permuta_cores([], {1:2})", [])]),

 ("componentes(g, cor)",
  "Escreva `def componentes(g, cor):` que devolve quantos componentes conexos "
  "(vizinhanca de 4: cima, baixo, esquerda, direita) de celulas com valor `cor` "
  "existem na grade `g`.",
  [("componentes([[1,0],[0,1]], 1)", 2),
   ("componentes([[1,1],[1,1]], 1)", 1),
   ("componentes([[0,0],[0,0]], 1)", 0),
   ("componentes([[1,0,1],[1,0,1],[0,0,0]], 1)", 2)]),

 ("periodo_h(g)",
  "Escreva `def periodo_h(g):` que devolve o MENOR p >= 1 tal que, para toda linha "
  "i e toda coluna j, g[i][j] == g[i][j % p]. Se nenhum p menor servir, devolve o "
  "numero de colunas.",
  [("periodo_h([[1,2,1,2]])", 2),
   ("periodo_h([[1,1,1]])", 1),
   ("periodo_h([[1,2,3]])", 3),
   ("periodo_h([[1,2,1,2],[3,4,3,4]])", 2)]),

 ("desloca(g, di, dj, fundo)",
  "Escreva `def desloca(g, di, dj, fundo):` que desloca a grade `di` linhas para "
  "baixo e `dj` colunas para a direita, mantendo o mesmo tamanho e preenchendo o "
  "que ficou vazio com `fundo`. O que sai da grade e descartado.",
  [("desloca([[1,2],[3,4]], 1, 0, 0)", [[0, 0], [1, 2]]),
   ("desloca([[1,2],[3,4]], 0, 1, 9)", [[9, 1], [9, 3]]),
   ("desloca([[1,2],[3,4]], 0, 0, 0)", [[1, 2], [3, 4]]),
   ("desloca([[1,2],[3,4]], -1, 0, 7)", [[3, 4], [7, 7]])]),

 ("mascara_zera(a, b)",
  "Escreva `def mascara_zera(a, b):` que recebe duas grades de mesmo tamanho e "
  "devolve uma nova grade onde, se a[i][j] == b[i][j], o valor vira 0, e se for "
  "diferente, fica o valor de a[i][j].",
  [("mascara_zera([[1,2]],[[1,3]])", [[0, 2]]),
   ("mascara_zera([[5]],[[5]])", [[0]]),
   ("mascara_zera([[1,1],[2,2]],[[0,1],[2,0]])", [[1, 0], [0, 2]])]),

 ("bits_excecao(a, b, ncores)",
  "Escreva `def bits_excecao(a, b, ncores):` que devolve o custo em bits de "
  "corrigir a grade `a` para virar `b`: para cada celula diferente, paga "
  "log2(numero de celulas da grade) para dizer ONDE, mais log2(ncores) para dizer "
  "O QUE. Se as grades forem iguais devolve 0.0. Use math.log2, devolva float.",
  [("round(bits_excecao([[1,2]],[[1,2]],10),3)", 0.0),
   ("round(bits_excecao([[1,2]],[[1,3]],10),3)", 4.322),
   ("round(bits_excecao([[1,1],[1,1]],[[2,2],[2,2]],4),3)", 16.0)]),
]


# ─── conversa com o servidor ─────────────────────────────────────────────────

def pergunta(msg, n=900, temp=0.0):
    d = json.dumps({"messages": [{"role": "user", "content": msg}],
                    "max_tokens": int(n * FATOR), "temperature": temp}).encode()
    r = urllib.request.Request(URL, d, {"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(r, timeout=900).read())
    t = o["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def _letra(txt):
    """Ultima letra A-E isolada da resposta."""
    m = re.findall(r"\b([A-Ea-e])\b", txt.replace("(", " ").replace(")", " "))
    return m[-1].upper() if m else "?"


def roda_logica(verboso=False):
    """DEFEITO CORRIGIDO: escrevi as 12 questoes com a certa quase sempre em B, e
    o primeiro modelo tirou 6/12 podendo ter tirado tudo so por viés de posicao.
    Agora as alternativas sao EMBARALHADAS por questao com semente fixa (mesma
    ordem para todos os modelos, entao a comparacao segue pareada) e o veredito
    imprime a distribuicao das letras dadas, que denuncia o viés."""
    rng = random.Random(20261020)
    acertos, detalhe, dadas = 0, [], []
    for i, (enun, alts0, certa0) in enumerate(LOGICA):
        ordem = list(range(len(alts0)))
        rng.shuffle(ordem)
        alts = [alts0[k] for k in ordem]
        certa = ordem.index(certa0)
        op = "\n".join(f"{chr(65+k)}) {a}" for k, a in enumerate(alts))
        p = (f"{enun}\n\n{op}\n\nResponda com UMA letra so (A, B, C, D ou E). "
             f"Sua resposta deve TERMINAR com a letra sozinha.")
        try:
            r = pergunta(p, 1200)
        except Exception as e:
            r = f"erro {e}"
        dada = _letra(r)
        if dada == "?":
            # DEFEITO DE INSTRUMENTO, NAO DO MODELO: um modelo que "pensa" pode
            # gastar todo o orcamento de tokens no raciocinio e devolver vazio.
            # Contar isso como erro mede o meu teto, nao a cabeca dele. Segunda
            # chance com o dobro de tokens e ordem de responder direto.
            try:
                r = pergunta(p + "\n\nResponda IMEDIATAMENTE com a letra, sem "
                                 "raciocinio longo.", 2600)
            except Exception as e:
                r = f"erro {e}"
            dada = _letra(r)
        ok = dada == chr(65 + certa)
        acertos += ok
        dadas.append(dada)
        detalhe.append((i + 1, dada, chr(65 + certa), bool(ok)))
        if verboso:
            print(f"    Q{i+1:02d} respondeu {dada} (certa {chr(65+certa)}) "
                  f"{'.' if ok else 'X'}", flush=True)
    viés = {c: dadas.count(c) for c in "ABCDE?" if dadas.count(c)}
    if verboso:
        print(f"    letras dadas: {viés}   (certas: "
              f"{ {c: [d[2] for d in detalhe].count(c) for c in 'ABCDE'} })",
              flush=True)
    return acertos, detalhe


def _extrai(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
    return (m.group(1) if m else txt).strip()


def roda_codigo(verboso=False):
    acertos, detalhe = 0, []
    for nome, enun, casos in CODIGO:
        p = enun + "\n\nResponda so com o codigo Python, num bloco ```python."
        try:
            fonte = _extrai(pergunta(p, 900))
        except Exception:
            fonte = ""
        teste = "import math\n" + fonte + "\n\n_ok = True\n"
        for exp, esp in casos:
            teste += f"_ok = _ok and ({exp} == {esp!r})\n"
        teste += "print('PASSOU' if _ok else 'FALHOU')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(teste)
            cam = f.name
        try:
            s = subprocess.run([sys.executable, cam], capture_output=True,
                               text=True, timeout=20)
            ok = "PASSOU" in s.stdout
        except Exception:
            ok = False
        acertos += ok
        detalhe.append((nome, bool(ok)))
        if verboso:
            print(f"    {nome:28s} {'.' if ok else 'X'}", flush=True)
    return acertos, detalhe


def binomial_p(k, n, p0):
    """P(X >= k) sob acaso."""
    return sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", default="modelo")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--fator", type=float, default=1.0)
    a = ap.parse_args()
    global FATOR
    FATOR = a.fator
    print("=" * 78)
    print(f"JUIZ DO MODELO LOCAL — {a.nome}")
    print("=" * 78)
    t0 = time.time()
    print("  LOGICA (12 questoes, acaso 20%):", flush=True)
    nl, dl = roda_logica(True)
    print("  PROGRAMACAO (6 tarefas, corrigidas por execucao):", flush=True)
    nc, dc = roda_codigo(True)
    dt = time.time() - t0
    print(f"\n  {a.nome}:  logica {nl}/12   programacao {nc}/6   ({dt:.0f}s)")
    print(f"  p(acaso >= {nl}/12) = {binomial_p(nl, 12, 0.2):.4f}")
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as f:
            json.dump({"nome": a.nome, "logica": nl, "codigo": nc,
                       "detalhe_logica": dl, "segundos": dt}, f)


if __name__ == "__main__":
    main()
