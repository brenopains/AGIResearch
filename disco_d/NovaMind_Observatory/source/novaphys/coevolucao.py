"""
novaphys.coevolucao — o que separa platô de crescimento sustentado
====================================================================

A PERGUNTA QUE DECIDE ANOS DE TRABALHO
----------------------------------------
A visao e: um mundo procedural rico, com energia custando, inflexibilidade
punida, gerações, linguagem, cultura — e complexidade cognitiva crescendo
sozinha.

Trinta anos de vida artificial ja testaram a parte central disso. Tierra e
Avida tinham mundos procedurais, custo de energia, mutacao, gerações. E
ESTAGNARAM. O diagnostico na literatura e especifico:

    regras fixas + ambiente estatico -> os organismos "jogam todos os jogos"
    disponiveis, adaptam-se completamente, e a evolucao para

Ou seja: nao foi falta de riqueza. Mundo mais completo com regras fixas
continua sendo finito.

E isso converge com tudo que medimos nesta sessao:

    orcamento de busca x3,5     ->  0 tarefas novas
    +7 primitivas               ->  3x o resultado
    consertos marginais         ->  falham
    espaco conjunto             ->  resolve

Mais do mesmo nao paga. Expandir o alcancavel paga.

A HIPOTESE TESTAVEL
---------------------
Se o platô vem de o espaco de problemas ser finito, entao a diferenca deveria
aparecer quando os agentes SAO o ambiente uns dos outros: cada adaptacao de um
cria um problema novo para o outro, e o espaco nunca fecha.

    braco FIXO    recurso distribuido pelo mundo, independente dos agentes
    braco COEVO   recurso vem de OUTROS AGENTES (predacao), entao o ambiente
                  muda quando os agentes mudam

Mesmo custo de energia, mesma mutacao, mesma populacao, mesmas geracoes. A
UNICA diferenca e de onde vem o alimento.

A MEDIDA
----------
Complexidade nao e tamanho do genoma — codigo morto infla isso de graca. E o
numero de REGRAS EFETIVAMENTE DISPARADAS ao longo da vida, media da populacao.
Regra que nunca dispara nao conta, entao inchar o programa nao pontua.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. Nos dois bracos a populacao tem que sobreviver e a
      complexidade tem que subir NO INICIO. Se nem sobe no comeco, o mundo e
      pobre demais e nada abaixo se le.

  C1  FIXO estabiliza e COEVO continua subindo
      -> coevolucao e o ingrediente; a visao tem base e o eixo certo e
         interacao entre agentes, nao riqueza do mundo

  C2  os dois estabilizam
      -> coevolucao tambem nao basta nesta escala; "mundo mais rico" e
         "mais agentes" nao sao o caminho, e a visao precisa de outra coisa

  C3  os dois crescem igual
      -> o crescimento vem do meu medidor ou da mutacao, nao do ambiente;
         instrumento suspeito antes de conclusao

MINHA PREVISAO: C2. Espero que os dois estabilizem, porque mesmo a predacao
aqui tem regras fixas — o jogo entre predador e presa tambem se esgota. Se
der C1, a visao ganha um argumento empirico de verdade.

SEMENTE PRE-REGISTRADA: 20260826
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

SEMENTE = 20260826
LADO = 24
N_INICIAL = 120
MAX_POP = 400
GERACOES = 400
PASSOS_POR_GER = 40
# CALIBRADO, nao chutado. A primeira versao extinguia tudo na geracao 1: o
# agente comecava com 8,0 de energia e gastava ~7,6 so existindo, entao
# precisava comer 2x para empatar e 4x para reproduzir — com regras
# sorteadas, quase ninguem executa `comer` no lugar certo. O portao C0 pegou.
# A varredura mostrou janela ESTREITA dos dois lados: mundo facil demais
# tambem colapsa (populacao estoura, recurso acaba).
CUSTO_REGRA = 0.005       # pensar gasta energia: por regra AVALIADA
CUSTO_VIVER = 0.03
ENERGIA_INICIAL = 8.0
ENERGIA_REPRODUZ = 10.0
MAX_REGRAS = 24
N_COMIDA = 400
# Atacar nao pode ser DE GRACA. Sem custo, predar e estritamente mais facil
# que forragear — comer exige estar em cima da planta, atacar so exige um
# vizinho. Diagnostico: a energia vinda de comida caiu 96% em 3 geracoes e a
# populacao morreu de fome num campo cheio. Suicidio evolutivo por assimetria
# da minha implementacao, nao por biologia.
CUSTO_ATAQUE = 1.0

# sensores: 0 comida perto, 1 agente perto, 2 energia baixa, 3 parede perto,
#           4 agente MAIOR perto, 5 agente MENOR perto, 6 aleatorio, 7 sempre
N_SENS = 8
# acoes: 0..3 mover, 4 comer, 5 atacar, 6 parado
N_ACOES = 7


class Bicho:
    __slots__ = ("regras", "x", "y", "e", "disparos", "idade")

    def __init__(self, regras, x, y):
        self.regras = regras            # lista de (sensor, negado, acao)
        self.x, self.y = x, y
        self.e = ENERGIA_INICIAL
        self.disparos = set()
        self.idade = 0

    def agir(self, sens: List[bool]) -> int:
        for i, (s, neg, a) in enumerate(self.regras):
            v = sens[s] != neg
            self.e -= CUSTO_REGRA          # avaliar custa, disparando ou nao
            if v:
                self.disparos.add(i)
                return a
        return N_ACOES - 1


def nascer(rng: random.Random) -> List[Tuple[int, int, int]]:
    n = rng.randint(1, 3)
    return [(rng.randrange(N_SENS), rng.random() < 0.5, rng.randrange(N_ACOES))
            for _ in range(n)]


def mutar(regras, rng: random.Random):
    r = [list(x) for x in regras]
    op = rng.random()
    if op < 0.35 and len(r) < MAX_REGRAS:          # crescer
        r.insert(rng.randrange(len(r) + 1),
                 [rng.randrange(N_SENS), rng.random() < 0.5,
                  rng.randrange(N_ACOES)])
    elif op < 0.55 and len(r) > 1:                 # encolher
        r.pop(rng.randrange(len(r)))
    elif op < 0.8:                                  # alterar
        i = rng.randrange(len(r))
        j = rng.randrange(3)
        r[i][j] = (rng.randrange(N_SENS) if j == 0 else
                   (rng.random() < 0.5 if j == 1 else rng.randrange(N_ACOES)))
    elif len(r) > 1:                                # reordenar
        i, j = rng.randrange(len(r)), rng.randrange(len(r))
        r[i], r[j] = r[j], r[i]
    return [tuple(x) for x in r]


def rodar(coevo: bool, semente: int, geracoes: int = GERACOES
          ) -> List[Tuple[int, float, float, int]]:
    rng = random.Random(semente)
    pop = [Bicho(nascer(rng), rng.randrange(LADO), rng.randrange(LADO))
           for _ in range(N_INICIAL)]
    comida = {(rng.randrange(LADO), rng.randrange(LADO)) for _ in range(N_COMIDA)}
    hist = []

    for g in range(geracoes):
        for _ in range(PASSOS_POR_GER):
            ocup: Dict[Tuple[int, int], List[Bicho]] = {}
            for b in pop:
                ocup.setdefault((b.x, b.y), []).append(b)
            for b in pop:
                viz = [(b.x + dx, b.y + dy)
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                outros = [o for p in viz for o in ocup.get(p, []) if o is not b]
                sens = [
                    any(p in comida for p in viz),
                    bool(outros),
                    b.e < ENERGIA_INICIAL * 0.5,
                    b.x <= 1 or b.y <= 1 or b.x >= LADO - 2 or b.y >= LADO - 2,
                    any(len(o.regras) > len(b.regras) for o in outros),
                    any(len(o.regras) < len(b.regras) for o in outros),
                    rng.random() < 0.5,
                    True,
                ]
                a = b.agir(sens)
                if a == 0:
                    b.x = (b.x + 1) % LADO
                elif a == 1:
                    b.x = (b.x - 1) % LADO
                elif a == 2:
                    b.y = (b.y + 1) % LADO
                elif a == 3:
                    b.y = (b.y - 1) % LADO
                elif a == 4:                       # comer planta
                    # DESENHO CORRIGIDO: a planta rende o MESMO nos dois
                    # bracos. A primeira versao dava um terco da comida ao
                    # COEVO e planta valendo 1,0 — isso extinguiu o braco na
                    # geracao 25 e confundiu duas variaveis, comparando
                    # "mundo farto" contra "mundo escasso COM predacao".
                    # Agora COEVO = FIXO + a opcao de predar. Uma variavel.
                    if (b.x, b.y) in comida:
                        comida.discard((b.x, b.y))
                        b.e += 4.0
                elif a == 5 and coevo:             # predar outro agente
                    b.e -= CUSTO_ATAQUE            # esforco, com ou sem presa
                    alvo = [o for o in outros if o.e > 0]
                    if alvo:
                        v = min(alvo, key=lambda o: o.e)
                        ganho = min(4.0, v.e)
                        v.e -= ganho
                        b.e += ganho * 0.8
                b.e -= CUSTO_VIVER
                b.idade += 1
            # plantas repõem
            alvo_c = N_COMIDA          # identico nos dois bracos
            while len(comida) < alvo_c:
                comida.add((rng.randrange(LADO), rng.randrange(LADO)))
            pop = [b for b in pop if b.e > 0]
            if not pop:
                break
        if not pop:
            hist.append((g, 0.0, 0.0, 0))
            break
        # reproducao
        filhos = []
        for b in pop:
            if b.e >= ENERGIA_REPRODUZ and len(pop) + len(filhos) < MAX_POP:
                b.e /= 2
                f = Bicho(mutar(b.regras, rng), b.x, b.y)
                f.e = b.e
                filhos.append(f)
        pop.extend(filhos)
        if len(pop) > MAX_POP:
            pop.sort(key=lambda b: -b.e)
            pop = pop[:MAX_POP]
        efet = st.mean(len(b.disparos) for b in pop)
        tam = st.mean(len(b.regras) for b in pop)
        hist.append((g, efet, tam, len(pop)))
        for b in pop:
            b.disparos = set()
    return hist


def inclinacao(h: Sequence[Tuple[int, float, float, int]],
               i0: int, i1: int) -> float:
    seg = [x[1] for x in h[i0:i1]]
    if len(seg) < 10:
        return 0.0
    meio = len(seg) // 2
    return st.mean(seg[meio:]) - st.mean(seg[:meio])


def relatorio(sementes: Sequence[int] = (0, 1, 2)) -> None:
    print("=" * 78)
    print("COEVOLUCAO — o que separa plato de crescimento sustentado")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  C0  os dois bracos sobrevivem e sobem no inicio")
    print("  C1  FIXO estabiliza e COEVO continua subindo -> coevolucao e a peca")
    print("  C2  os dois estabilizam -> nem coevolucao basta nesta escala")
    print("  C3  os dois crescem igual -> o medidor e suspeito")
    print("\nMINHA PREVISAO: C2. O jogo predador-presa tambem tem regras fixas")
    print("e deve se esgotar. Se der C1, a visao ganha argumento empirico.")
    print(f"\nmedida: REGRAS EFETIVAMENTE DISPARADAS (codigo morto nao conta)")
    print(f"{GERACOES} geracoes · {len(sementes)} sementes por braco")

    res: Dict[str, List] = {}
    for coevo, nome in ((False, "FIXO"), (True, "COEVO")):
        hs = [rodar(coevo, SEMENTE + s) for s in sementes]
        res[nome] = hs
        print(f"\n--- {nome}")
        print(f"{'ger':>6}{'efetivas':>11}{'tamanho':>10}{'pop':>7}")
        for g in (0, 25, 50, 100, 200, 300, GERACOES - 1):
            vals = [h[g] for h in hs if g < len(h)]
            if not vals:
                print(f"{g:>6}   extinto")
                continue
            print(f"{g:>6}{st.mean(v[1] for v in vals):>10.2f}"
                  f"{st.mean(v[2] for v in vals):>10.2f}"
                  f"{st.mean(v[3] for v in vals):>7.0f}")

    print("\n" + "=" * 78)
    print("INCLINACAO da complexidade efetiva (2a metade menos 1a metade)")
    print(f"{'braco':<8}{'inicio (0-100)':>18}{'fim (200-400)':>18}")
    incs = {}
    for nome in ("FIXO", "COEVO"):
        i0 = st.mean(inclinacao(h, 0, 100) for h in res[nome])
        i1 = st.mean(inclinacao(h, 200, GERACOES) for h in res[nome])
        incs[nome] = (i0, i1)
        print(f"{nome:<8}{i0:>+17.3f}{i1:>+17.3f}")

    print()
    f0, f1 = incs["FIXO"]
    c0, c1 = incs["COEVO"]
    if f0 <= 0.02 or c0 <= 0.02:
        print("C0 · PRE-CONDICAO REPROVADA. Algum braco nao sobe nem no inicio;")
        print("     o mundo e pobre demais e o resto nao se le.")
        return
    print("C0 · aprovada — os dois sobem no inicio")
    print()
    if c1 > 0.02 and f1 <= 0.02:
        print("C1 · COEVOLUCAO SUSTENTA. O braco fixo estabiliza e o coevolutivo")
        print("     continua subindo. O eixo certo e interacao entre agentes,")
        print("     nao riqueza do mundo.")
    elif c1 <= 0.02 and f1 <= 0.02:
        print("C2 · OS DOIS ESTABILIZAM. Nem coevolucao sustenta crescimento")
        print("     nesta escala. 'Mundo mais rico' e 'mais agentes' nao sao o")
        print("     caminho — e o mesmo plato que Tierra e Avida encontraram.")
    else:
        print(f"C3 · os dois continuam subindo (fixo {f1:+.3f}, coevo {c1:+.3f}).")
        print("     Suspeitar do medidor antes de concluir qualquer coisa.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()
    relatorio(sementes=tuple(range(a.sementes)))
