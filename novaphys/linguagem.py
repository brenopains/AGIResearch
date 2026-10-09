"""
novaphys.linguagem — ler uma frase que nunca viu e saber o que ela FAZ
========================================================================

A LACUNA
----------
O sistema aprende lexico de fala ([[mdl-descobre-palavras]], F1 0,557), mas nao
le uma frase e extrai uma afirmacao que possa usar. E o usuario foi explicito
sobre o caminho: "nao digo modelos tradicionais de geracao de texto de IA, mas
algo mais alinhado com o que fazemos aqui". Entao nada de LLM como faculdade de
linguagem. O sistema tem de aprender o que as palavras significam do mesmo jeito
que aprende o resto: pela consequencia no mundo.

O PRIMEIRO DEGRAU: ATERRAMENTO E COMPOSICAO
---------------------------------------------
Ler um livro exige, antes, que simbolo aponte para alguma coisa. Isto testa
exatamente isso, e so isso:

  O MUNDO    os estados 2-D do atlas de [[modelo-mundo-credito]] — grades do
             ARC, imagens do CIFAR, janelas do HAR — e os 6 operadores que
             atravessam esses mundos.
  A FALA     um narrador descreve o que aconteceu. Palavras de conteudo nomeiam
             operadores; "depois" e "antes" nomeiam ORDEM (nao sao operador
             nenhum); "agora", "entao" e "isso" sao vazias.
  O SISTEMA  nao recebe dicionario. Recebe frase + estado antes + estado depois,
             e so. Tem de descobrir que "gira" e transpor, que "depois" e ordem
             direta, que "antes" inverte a ordem, e que "entao" nao faz nada.

O TESTE E COMPOSICIONAL (o estilo do SCAN, mas aterrado)
----------------------------------------------------------
Treino: cada operador sozinho, e ALGUMAS combinacoes. Teste: combinacoes que
NUNCA apareceram, em estados de mundos que nunca apareceram. Entender = dada a
frase nova e o estado antes, prever o estado depois exatamente.

TRES BRACOS
-------------
  ATERRADO   aprende palavra -> operador por qual operador EXPLICA a transicao
             quando a palavra aparece (surpresa que a hipotese remove), e
             aprende a ordem testando as duas ordens contra o dado
  SACOLA     mesmo aterramento, mas ignora a ordem das palavras — o controle
             que mostra se "antes" foi entendido ou so tolerado
  PLACEBO    mesmo aprendiz, frases pareadas com transicoes de OUTRA frase —
             palavra sem mundo por tras

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  L0  PRE-CONDICAO. Nas frases de uma palavra so, o ATERRADO acerta o operador
      em >= 0,95 dos casos de teste. Se nao aprende a palavra isolada, nao ha o
      que compor.
  L1  ENTENDE FRASE NOVA: nas combinacoes NUNCA vistas, o ATERRADO preve o
      estado exato em >= 0,80; o PLACEBO fica <= 0,20; e nos itens em que a
      ordem importa (operadores que nao comutam naquele estado) a SACOLA fica
      <= 0,60 enquanto o ATERRADO passa de 0,80
      -> o sistema aprendeu o que as palavras FAZEM, inclusive a que so carrega
         ordem, e compoe o que nunca viu.
  L2  acerta palavra isolada mas falha a composicao (< 0,80) -> aterra, nao
      compoe
  L3  falha L0 -> nem aterra

PREVISAO: L1 a 0,60. Mecanismo: com os operadores ja existindo no atlas, o
aterramento reduz-se a qual hipotese explica a transicao, e isso e o que o
projeto faz bem. Risco nomeado: operadores que comutam (ex.: `espelha` e
`inverte`) nao ensinam ordem nenhuma, e se o treino so tiver pares comutativos
com "antes", a ordem fica indeterminada — por isso o treino inclui pares nao
comutativos e o teste separa os itens sensiveis a ordem. L2 a 0,25, L3 a 0,15.
"""

from __future__ import annotations

import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from novaphys.modelo_mundo import MUNDOS, N_MIN, OPERADORES, normaliza

# o narrador conhece isto; o SISTEMA nao recebe esta tabela
PALAVRA = {"espelha": "espelha", "gira": "transpoe", "empurra": "desloca",
           "troca": "inverte", "borra": "borra", "corta": "limiar"}
VAZIAS = ("agora", "entao", "isso")
SEMENTE = 20260923


def _estados(n_por: int = 40) -> Dict[str, List[np.ndarray]]:
    out = {}
    for nome, fn in MUNDOS.items():
        try:
            ns = [x for x in (normaliza(a) for a in fn(n_por)) if x is not None]
        except Exception:
            continue
        if len(ns) >= N_MIN:
            out[nome] = ns
    return out


def _aplica(seq: Sequence[str], s: np.ndarray) -> np.ndarray:
    for op in seq:
        s = OPERADORES[op](s)
    return s


def narra(ops: Tuple[str, ...], ordem: str, rnd) -> List[str]:
    """Frase para aplicar `ops` em sequencia. 'antes' inverte a ordem dita."""
    inv = {v: k for k, v in PALAVRA.items()}
    ws = [inv[o] for o in ops]
    if len(ws) == 2:
        # ops = (A, B): A e aplicado PRIMEIRO. As duas construcoes do narrador:
        #   "A depois B"      -> ordem das palavras = ordem dos fatos (direta)
        #   "B, antes A"      -> "B, mas antes disso A": palavras invertidas
        if ordem == "antes":
            ws = [ws[1], "antes", ws[0]]
        else:
            ws = [ws[0], "depois", ws[1]]
    if rnd.random() < 0.5:
        ws.insert(rnd.randrange(len(ws) + 1), rnd.choice(VAZIAS))
    return ws


def gera(estados, rnd, combos, n_por: int) -> List[Tuple[List[str], np.ndarray,
                                                        np.ndarray, Tuple]]:
    """(frase, antes, depois, ops) — `ops` e o gabarito, o aprendiz nao o ve."""
    todos = [s for ns in estados.values() for s in ns]
    out = []
    for ops, ordem in combos:
        for _ in range(n_por):
            s = rnd.choice(todos)
            out.append((narra(ops, ordem, rnd), s, _aplica(ops, s), ops))
    return out


class Aprendiz:
    """
    Aprende palavra -> operador e o sentido das palavras de ordem, SO por quais
    hipoteses explicam a transicao observada. Nao recebe o dicionario.
    """

    def __init__(self, sacola: bool = False) -> None:
        self.sacola = sacola
        self.sig: Dict[str, Optional[str]] = {}
        self.ordem: Dict[str, str] = {}      # palavra de ordem -> 'direta'/'inversa'

    @staticmethod
    def _explica(seq, a, b) -> bool:
        try:
            return np.array_equal(_aplica(seq, a), b)
        except Exception:
            return False

    def aprende(self, dados) -> None:
        ops = list(OPERADORES)
        cont: Dict[str, Counter] = defaultdict(Counter)
        freq: Counter = Counter()
        # 1. co-ocorrencia palavra x operador que explica, nas frases de 1 conteudo
        #    (o aprendiz nao sabe o que e conteudo: testa se UM operador so explica)
        for ws, a, b, _g in dados:
            for w in set(ws):
                freq[w] += 1
            um = [o for o in ops if self._explica([o], a, b)]
            if len(um) >= 1 and len(um) < len(ops):
                for w in set(ws):
                    for o in um:
                        cont[w][o] += 1.0 / len(um)
        # palavra significa o operador se o operador explica QUASE SEMPRE que ela
        # aparece nesses casos; senao ela e vazia (ou de ordem)
        # REGRA TROCADA DEPOIS DE VER AS CONTAGENS DO TREINO — declarado. A
        # primeira regra (fracao >= 0,8) ignorava o ruido de COINCIDENCIA: em
        # estados degenerados uma composicao parece um operador so, e isso poe
        # ~50% de contagem espuria em toda palavra. O argmax ja estava certo
        # para todas as palavras de conteudo; a fracao e que reprovava. Para nao
        # escolher limiar a olho (eu conheco o gabarito), o criterio novo e o
        # convencional de associacao: INFORMACAO MUTUA PONTUAL >= 1 bit — a
        # palavra torna o operador >= 2x mais provavel que a taxa de base. O
        # TESTE nao foi olhado para fixar isto. Suporte absoluto (>= 10) mantido.
        total = sum(sum(c.values()) for c in cont.values())
        base: Counter = Counter()
        for c in cont.values():
            for o, v in c.items():
                base[o] += v / total
        import math
        for w, c in cont.items():
            o, n = c.most_common(1)[0]
            tw = sum(c.values())
            pmi = math.log2((n / tw) / base[o]) if base[o] > 0 else 0.0
            self.sig[w] = o if n >= 10 and pmi >= 1.0 else None
        # 2. palavras que sobram: testar se elas carregam ORDEM
        for ws, a, b, _g in dados:
            cont_ws = [w for w in ws if self.sig.get(w)]
            if len(cont_ws) != 2:
                continue
            resto = [w for w in ws if not self.sig.get(w)]
            x, y = self.sig[cont_ws[0]], self.sig[cont_ws[1]]
            direta = self._explica([x, y], a, b)
            inversa = self._explica([y, x], a, b)
            if direta == inversa:
                continue                    # comutam aqui: nao ensina ordem
            for w in resto:
                self.ordem.setdefault(w, Counter())
                self.ordem[w]["direta" if direta else "inversa"] += 1
        self.ordem = {w: c.most_common(1)[0][0] for w, c in self.ordem.items()
                      if sum(c.values()) >= 3 and
                      c.most_common(1)[0][1] / sum(c.values()) >= 0.8}

    def entende(self, ws: List[str]) -> Optional[List[str]]:
        cont = [self.sig[w] for w in ws if self.sig.get(w)]
        if not cont:
            return None
        if self.sacola:
            return sorted(cont)             # ignora a ordem das palavras
        inversa = any(self.ordem.get(w) == "inversa" for w in ws)
        return list(reversed(cont)) if inversa else cont


def experimento(semente: int = SEMENTE) -> Dict:
    rnd = random.Random(semente)
    est = _estados()
    nomes = sorted(est)
    # mundos de TESTE nunca aparecem no treino
    mundos_teste = set(nomes[-2:])
    tr_est = {k: v for k, v in est.items() if k not in mundos_teste}
    te_est = {k: v for k, v in est.items() if k in mundos_teste}
    ops = list(OPERADORES)

    pares = [p for p in itertools.permutations(ops, 2)]
    rnd.shuffle(pares)
    vistos, novos = pares[:12], pares[12:]
    combos_tr = [((o,), "-") for o in ops]
    combos_tr += [(p, rnd.choice(["depois", "antes"])) for p in vistos]
    combos_te1 = [((o,), "-") for o in ops]
    combos_te2 = [(p, rnd.choice(["depois", "antes"])) for p in novos]

    treino = gera(tr_est, rnd, combos_tr, 25)
    teste1 = gera(te_est, rnd, combos_te1, 20)
    teste2 = gera(te_est, rnd, combos_te2, 8)

    print("=" * 78)
    print("LINGUAGEM — ler uma frase que nunca viu e saber o que ela FAZ")
    print("=" * 78)
    print("  treino: %d frases em mundos %s" % (len(treino), sorted(tr_est)))
    print("  teste : mundos NUNCA vistos %s; %d combinacoes NUNCA vistas"
          % (sorted(te_est), len(novos)))
    print("  exemplos de frase: %s | %s\n"
          % (" ".join(treino[0][0]), " ".join(treino[-1][0])))

    # placebo: cada frase recebe a transicao de OUTRA frase
    emb = list(range(len(treino)))
    rnd.shuffle(emb)
    placebo_tr = [(treino[i][0], treino[j][1], treino[j][2], treino[j][3])
                  for i, j in zip(range(len(treino)), emb)]

    bracos = {"ATERRADO": (Aprendiz(), treino),
              "SACOLA": (Aprendiz(sacola=True), treino),
              "PLACEBO": (Aprendiz(), placebo_tr)}
    for _n, (ap, dados) in bracos.items():
        ap.aprende(dados)

    ap = bracos["ATERRADO"][0]
    print("  o que o ATERRADO descobriu (o dicionario ele nunca recebeu):")
    for w in sorted(ap.sig):
        print("    %-9s -> %s" % (w, ap.sig[w] or "(sem operador)"))
    print("    palavras de ORDEM: %s\n" % (ap.ordem or "nenhuma"))

    def acerto(aprendiz, itens, so_ordem=False):
        ok = n = 0
        for ws, a, b, g in itens:
            if so_ordem:
                if len(g) != 2 or np.array_equal(_aplica(g, a),
                                                 _aplica(tuple(reversed(g)), a)):
                    continue
            n += 1
            seq = aprendiz.entende(ws)
            ok += int(seq is not None and np.array_equal(_aplica(seq, a), b))
        return ok / n if n else float("nan"), n

    print("  %-10s %16s %18s %18s" % ("braco", "1 palavra", "frase NOVA",
                                       "so onde ORDEM importa"))
    tab = {}
    for nome, (apr, _d) in bracos.items():
        a1, n1 = acerto(apr, teste1)
        a2, n2 = acerto(apr, teste2)
        a3, n3 = acerto(apr, teste2, so_ordem=True)
        tab[nome] = (a1, a2, a3)
        print("  %-10s %10.3f (n=%d) %11.3f (n=%d) %11.3f (n=%d)"
              % (nome, a1, n1, a2, n2, a3, n3))

    a1, a2, a3 = tab["ATERRADO"]
    print("\n" + "=" * 78)
    if a1 < 0.95:
        ver = "L3"
        print("L3 . nem aterra: palavra isolada %.3f." % a1)
    elif (a2 >= 0.80 and tab["PLACEBO"][1] <= 0.20 and tab["SACOLA"][2] <= 0.60
          and a3 >= 0.80):
        ver = "L1"
        print("L1 . ENTENDE FRASE QUE NUNCA VIU: %.3f nas combinacoes novas," % a2)
        print("     %.3f onde a ordem importa (sacola %.3f), placebo %.3f."
              % (a3, tab["SACOLA"][2], tab["PLACEBO"][1]))
        print("     Aprendeu o que as palavras FAZEM — inclusive as que so")
        print("     carregam ordem — sem nunca receber o dicionario.")
    else:
        ver = "L2"
        print("L2 . aterra (%.3f) mas nao compoe o bastante: frase nova %.3f," % (a1, a2))
        print("     ordem %.3f, sacola %.3f, placebo %.3f."
              % (a3, tab["SACOLA"][2], tab["PLACEBO"][1]))
    r = {"veredito": ver, "tabela": tab, "dicionario": ap.sig, "ordem": ap.ordem}
    json.dump(r, open("resultados/linguagem.json", "w"), indent=1,
              ensure_ascii=False)
    return r


if __name__ == "__main__":
    experimento()
