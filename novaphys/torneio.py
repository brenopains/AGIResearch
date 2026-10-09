"""
novaphys.torneio — mérito absoluto contra comparação na hora da tarefa
========================================================================

A HIPÓTESE, E ELA ENCAIXA DOZE MEDIDAS
----------------------------------------
Todo nulo desta série tentou produzir um juízo ABSOLUTO sobre uma peça —
"esta operação é boa?", "esta pilha vale os bits?", "vale guardar este
operador?", "qual o resíduo mínimo desta estrutura?". Todos falharam:
[[sala-de-operacoes-nulo]], [[o-criterio-nao-seleciona]],
[[vocabulario-grande-piora]], [[promessa-do-buraco]], [[elimina-cor-nulo]].

E o que funcionou foi sempre RELATIVO: [[sob-demanda-funciona]] ordenou a MESMA
loja na hora da tarefa e ganhou por 12-20 desvios (a stdlib saiu da posição 9
para a 1 quando a tarefa mudou); [[confianca-e-margem]] calibrou pela margem
para o 2º e saturou pela nota bruta; [[piso-de-ruido-e-pareamento]] mediu nível
com σ 0,040 e diferença com σ 0,008.

Hipótese: **"útil" nunca foi propriedade da peça.** Não existe "este operador é
bom"; existe "este ganhou daquele, nesta tarefa, agora". Se for verdade,
curadoria de biblioteca por mérito é uma operação sem referente, e é por isso
que ela falhou três vezes.

RESSALVA: isso encaixa onze nulos e três positivos RETROSPECTIVAMENTE. Tem
cheiro de narrativa pós-hoc — o mesmo defeito que a tabela adivinhado/resolvido
teve hoje. Este arquivo é o teste PROSPECTIVO.

O CONTRASTE
-------------
O feixe só consegue considerar K codecs por nó. A pergunta é como escolher os K:

  LIMIAR   os MESMOS K para toda tarefa, escolhidos por mérito global medido
           num conjunto de DESENVOLVIMENTO disjunto
  TORNEIO  K DIFERENTES por tarefa, escolhidos por comparação na própria tarefa
  SORTEIO  K aleatórios por tarefa — controle que isola "variar o conjunto" de
           "ordenar o conjunto"

O SORTEIO é obrigatório: sem ele, um ganho do torneio poderia ser só o efeito de
o conjunto mudar, e não de ele ser ordenado. (A lição de 18/09: casar a
liberdade de escolha, não só o orçamento.)

PRÉ-REGISTRO
--------------
  U0  PRÉ-CONDIÇÃO. Com K = todos os codecs, os três braços coincidem. Se não
      coincidirem, a implementação difere em outra coisa e nada é interpretável.
  U1  TORNEIO >= LIMIAR + 3 E >= SORTEIO + 3, em 40 tarefas de avaliação
      -> comparação na hora bate mérito absoluto, e curadoria por mérito sai
         da arquitetura
  U2  TORNEIO ~ LIMIAR -> mérito absoluto funciona neste recorte e a hipótese
      das doze medidas não se sustenta prospectivamente
  U3  TORNEIO ~ SORTEIO -> variar o conjunto ajuda, a ORDENAÇÃO não

PREVISÃO: U1, 0,55. Mecanismo: [[sob-demanda-funciona]] é o positivo mais forte
do projeto e é exatamente este desenho, num domínio diferente. Risco nomeado:
lá a sonda vinha do alvo; aqui a sonda é o próprio dado de treino da tarefa, o
que é mais fraco e pode não discriminar.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys import forma
from novaphys.arc import carregar
from novaphys.descricao import CODECS, ENTAO, _marca, custo, montar
from novaphys.promessa import PASTA
from novaphys.tres_bracos import busca_com_forma

K = 8
FEIXE, PROF = 12, 6


def merito_global(tarefas, teto: int = 20000) -> List:
    """
    LIMIAR: quantas vezes cada codec aparece em pilha VENCEDORA no conjunto de
    desenvolvimento. É mérito absoluto, medido — a forma mais forte de limiar
    que dá para montar aqui.
    """
    c = Counter()
    for nome, tr, _te in tarefas:
        try:
            _fm, pilha, err = montar(tr, teto=teto)
        except Exception:
            continue
        if err != 0 or not pilha:
            continue
        for it in pilha:
            if not _marca(it):
                c[it[0].nome] += 1
    ordem = [cd for cd in CODECS]
    ordem.sort(key=lambda cd: -c[cd.nome])
    return ordem, c


def ordena_na_tarefa(treino) -> List:
    """
    TORNEIO: ordena os codecs pela redução de erro que cada um dá NESTA tarefa,
    como passo único. Nenhum mérito guardado; tudo decidido agora.
    """
    _b0, e0 = custo([], treino)
    notas = []
    for cd in CODECS:
        melhor = e0
        try:
            ms = list(itertools.islice(iter(cd.propor(treino) or []), 4))
        except Exception:
            ms = []
        for m in ms:
            try:
                _b, e = custo([(cd, m)], treino)
            except Exception:
                continue
            if e < melhor:
                melhor = e
        notas.append((melhor, cd))
    notas.sort(key=lambda t: t[0])
    return [cd for _n, cd in notas]


def com_subconjunto(treino, sub, segundos: float):
    """Roda a busca do braço A restrita a `sub`, salvando/restaurando CODECS."""
    import novaphys.descricao as D
    orig = D.CODECS
    try:
        D.CODECS = list(sub)
        return busca_com_forma(treino, "A", segundos, 6, FEIXE, PROF)
    finally:
        D.CODECS = orig


def experimento(n_dev: int = 40, n_aval: int = 40, segundos: float = 45.0,
                k: int = K) -> None:
    todas = carregar(PASTA)
    todas.sort(key=lambda t: hashlib.md5(t[0].encode()).hexdigest())
    dev, aval = todas[:n_dev], todas[n_dev:n_dev + n_aval]

    print("=" * 78)
    print("TORNEIO x LIMIAR — merito absoluto contra comparacao na hora")
    print("=" * 78)
    print("  dev %d tarefas (so para o LIMIAR) | aval %d tarefas | K=%d de %d"
          % (len(dev), len(aval), k, len(CODECS)))
    print("  %.0fs por tarefa e braco; feixe %d; profundidade %d\n"
          % (segundos, FEIXE, PROF))

    print("  medindo merito global no conjunto de desenvolvimento...", flush=True)
    ordem, cont = merito_global(dev)
    fixo = ordem[:k]
    print("  LIMIAR fixa: %s" % ", ".join(cd.nome for cd in fixo))
    print("  (contagem: %s)\n" % dict(cont.most_common(6)), flush=True)

    res = {m: [] for m in ("LIMIAR", "TORNEIO", "SORTEIO")}
    rnd = random.Random(20260921)
    t0 = time.time()
    for i, (nome, tr, te) in enumerate(aval):
        subs = {
            "LIMIAR": fixo,
            "TORNEIO": ordena_na_tarefa(tr)[:k],
            "SORTEIO": rnd.sample(list(CODECS), k),
        }
        for m, sub in subs.items():
            try:
                r = com_subconjunto(tr, sub, segundos)
            except Exception:
                r = None
            if r is None:
                continue
            _fm, pilha, reb = r
            try:
                _b, err = custo(pilha, reb)
            except Exception:
                continue
            if err == 0:
                res[m].append(nome)
        if (i + 1) % 10 == 0:
            print("    %2d/%d | LIMIAR=%d TORNEIO=%d SORTEIO=%d | %.0f min"
                  % (i + 1, len(aval), len(res["LIMIAR"]), len(res["TORNEIO"]),
                     len(res["SORTEIO"]), (time.time() - t0) / 60), flush=True)

    print("\n" + "=" * 78)
    for m in ("LIMIAR", "TORNEIO", "SORTEIO"):
        print("  %-8s fecha %2d/%d" % (m, len(res[m]), len(aval)))
    t, l, s = (len(res[m]) for m in ("TORNEIO", "LIMIAR", "SORTEIO"))
    print("\n  TORNEIO - LIMIAR = %+d   TORNEIO - SORTEIO = %+d" % (t - l, t - s))
    print("  so o TORNEIO fecha: %s"
          % (sorted(set(res["TORNEIO"]) - set(res["LIMIAR"])) or "nenhuma"))
    print()
    if t >= l + 3 and t >= s + 3:
        print("U1 . COMPARAR NA HORA BATE MERITO ABSOLUTO. Curadoria de")
        print("     biblioteca por merito sai da arquitetura.")
    elif abs(t - l) <= 1:
        print("U2 . merito absoluto funciona neste recorte; a hipotese das doze")
        print("     medidas nao se sustenta prospectivamente aqui.")
    elif abs(t - s) <= 1:
        print("U3 . variar o conjunto ajuda, a ORDENACAO nao.")
    else:
        print("inconclusivo pela barra pre-registrada: %+d e %+d" % (t - l, t - s))


if __name__ == "__main__":
    experimento()
