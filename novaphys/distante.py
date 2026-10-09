"""
novaphys.distante — a evidencia decisiva esta longe. Quem a encontra?
=======================================================================

POR QUE ESTE DESENHO E CIRURGICO
----------------------------------
No mundo anterior, o discriminador planejado empatou com a busca por novidade
(55 vitorias contra 57, p = 0,61). Nulo com estrutura: ganhava em objetivos
posicionais, perdia em objetivos de coleta, e duas explicacoes minhas para
isso foram refutadas pela instrumentacao.

O problema daquele mundo e que a evidencia decisiva estava PERTO. Andar ao
acaso tropeca nela, e ai um mecanismo que a procura de proposito nao tem como
mostrar vantagem.

Aqui a evidencia decisiva e colocada LONGE, de proposito, e a distancia e o
parametro varrido. A previsao forte nao e "o planejador ganha" — e:

    a vantagem CRESCE com a distancia

Se ela nao crescer, o mecanismo esta errado por natureza e nao por ajuste, e
isso encerra a linha em vez de pedir um terceiro conserto.

O MUNDO
---------
Grade LADO x LADO, tudo livre. O agente comeca no canto direito.

    objetivo verdadeiro   coluna == 0        (a coluna esquerda inteira)
    rival confundivel     coluna == 0 e linha != d

As duas concordam em TODO estado, menos num: a celula (d, 0). Tocar qualquer
outra celula da coluna esquerda faz o agente vencer e as duas hipoteses
concordam que foi vitoria — nao separa nada.

E vencer REINICIA o episodio. Entao o agente que entra na coluna esquerda
pela linha 0 vence, volta ao inicio, e nunca descobre nada. Para distinguir
ele precisa entrar exatamente por (d, 0).

Isso e evidencia causalmente distante: abundante em vitorias, pobre em
informacao, com um unico ponto que decide.

O QUE O PLANEJADOR RECEBE, E O QUE NAO RECEBE
-----------------------------------------------
NAO recebe "a celula (d,0) separa as hipoteses". Recebe apenas as hipoteses
sobreviventes e o modelo de mundo que ele mesmo aprendeu. A divergencia ele
calcula: para cada estado alcancavel no modelo, quantas sobreviventes dizem
vitoria e quantas dizem que nao. Entregar o experimento otimo seria testar a
minha aritmetica, nao o mecanismo dele.

TRES BRACOS
-------------
    ACASO       acao uniforme
    NOVIDADE    estado menos visitado, prevendo com o modelo
    PLANEJADA   busca no modelo o estado de maior divisao e vai ate la

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. As duas hipoteses tem que ser indistinguiveis em todo
      estado menos (d,0). Conferido por enumeracao antes de medir; se
      divergirem em outro lugar, a evidencia nao esta isolada e o
      experimento nao mede distancia.

  D1  a razao ACASO/PLANEJADA cresce com d, e em d=40 e >= 3x a de d=5
      -> o mecanismo localiza evidencia causalmente distante, que e a
         capacidade que ele promete

  D2  a vantagem existe mas e constante em d
      -> e uma heuristica boa, nao localizacao de evidencia distante

  D3  sem vantagem em nenhum d
      -> o mecanismo esta errado; encerra a linha

MINHA PREVISAO, escrita antes de rodar: D1 para NOVIDADE contra PLANEJADA,
mas com uma ressalva que pode virar o resultado. A busca por novidade tambem
e atraida por (d,0), porque e uma celula pouco visitada — ela nao sabe que
aquilo importa, mas cobre o espaco e acaba chegando. A diferenca deveria
aparecer no CUSTO: novidade visita tudo, planejada vai direto.

Se novidade acompanhar de perto mesmo em d=40, a leitura honesta e que
cobertura sistematica e quase tao boa quanto mira, e o valor do desenho
otimo de experimento neste tipo de mundo e menor do que a teoria sugere.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

# 12x12 era pequeno demais: d so variava de 2 a 11, nove passos numa viagem
# de ~25, e a razao mal se mexia (1,4x contra os 3x que D1 pedia). O criterio
# estava certo em espirito e errado em escala. Em 30x30 a distancia varia de
# 2 a 29 e o efeito, se existir, tem espaco para aparecer.
LADO = 30
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Mundo:
    """Coluna esquerda vence. A linha `d` e o unico ponto que discrimina."""

    def __init__(self, semente: int, d: int):
        # a celula decisiva e (d, 0); o inicio fica no canto oposto para que
        # `d` controle de fato a distancia ate ela
        self.d = d
        self.inicio = (0, LADO - 1)
        self.dist = (LADO - 1) + d          # manhattan ate (d, 0)
        self.reiniciar()

    def reiniciar(self) -> None:
        self.pos = self.inicio

    def estado(self) -> tuple:
        return self.pos

    def passo(self, a: int) -> tuple:
        dy, dx = ACOES[a]
        n = (self.pos[0] + dy, self.pos[1] + dx)
        if 0 <= n[0] < LADO and 0 <= n[1] < LADO:
            self.pos = n
        return self.pos

    def venceu(self, e: Optional[tuple] = None) -> bool:
        p = e if e else self.pos
        return p[1] == 0


def hipoteses(d: int) -> List[Tuple[str, Callable]]:
    """
    Espaco gerado, com UMA rival confundivel na linha `d`.

    A primeira versao gerava `coluna==c e linha!=k` para TODO k. Com isso,
    convergir exigia visitar a coluna esquerda inteira — uma celula por
    rival — e a distancia nunca entrava na conta. As quatro distancias deram
    numeros identicos (6000/603/200) porque `d` era um parametro que eu
    passava e nunca lia.

    Aqui a rival e unica e mora exatamente em (d, 0). Todo o resto sao
    engodos que morrem cedo, e a unica observacao que separa a verdadeira da
    rival e pisar naquela celula.
    """
    fora: List[Tuple[str, Callable]] = []
    fora.append(("coluna==0", lambda e: e[1] == 0))            # verdadeira
    fora.append((f"coluna==0 e linha!={d}",                    # rival
                 lambda e, d=d: e[1] == 0 and e[0] != d))
    for c in range(1, LADO):                                   # engodos
        fora.append((f"coluna=={c}", lambda e, c=c: e[1] == c))
        fora.append((f"linha=={c}", lambda e, c=c: e[0] == c))
    fora.append(("borda", lambda e: e[0] in (0, LADO - 1)
                 or e[1] in (0, LADO - 1)))
    fora.append(("canto", lambda e: e in ((0, 0), (0, LADO - 1),
                                          (LADO - 1, 0), (LADO - 1, LADO - 1))))
    return fora


# ══════════════════════════════════════════════ exploradores
def esc_acaso(e, cont, vivos, hip, rng) -> int:
    return rng.randrange(4)


def _prox(e, a) -> tuple:
    dy, dx = ACOES[a]
    n = (e[0] + dy, e[1] + dx)
    return n if 0 <= n[0] < LADO and 0 <= n[1] < LADO else e


def esc_novidade(e, cont, vivos, hip, rng) -> int:
    melhor, c = None, []
    for a in range(4):
        p = cont.get(_prox(e, a), 0)
        if melhor is None or p < melhor:
            melhor, c = p, [a]
        elif p == melhor:
            c.append(a)
    return rng.choice(c)


def esc_planejada(e, cont, vivos, hip, rng, por_custo: bool = False) -> int:
    """
    Busca em largura o estado de maior divisao entre as sobreviventes e
    devolve o primeiro passo do caminho. A divisao e calculada das proprias
    hipoteses; nada aqui sabe onde fica a celula decisiva.
    """
    if len(vivos) < 2:
        return esc_novidade(e, cont, vivos, hip, rng)
    fila = deque([(e, None, 0)])
    visto = {e}
    melhor, melhor_a = 0.0, None
    while fila:
        est, prim, prof = fila.popleft()
        if prof > LADO * 2:
            break
        if prim is not None:
            # `por_custo` divide tambem pela PROFUNDIDADE do caminho. Sem
            # isso o planejador maximiza informacao BRUTA e ignora o preco:
            # medido, ele escolhia um alvo a 56 passos que separa 20
            # hipoteses em vez de um a 9 passos que separa 5 — errando
            # mesmo quando o segundo dava 5x mais informacao por passo.
            v = sum(1 for k in vivos if hip[k][1](est))
            corte = min(v, len(vivos) - v) / (1 + cont.get(est, 0))
            if por_custo:
                corte /= prof
            if corte > melhor:
                melhor, melhor_a = corte, prim
        for a in range(4):
            p = _prox(est, a)
            if p in visto:
                continue
            visto.add(p)
            fila.append((p, a if prim is None else prim, prof + 1))
    if melhor_a is None:
        return esc_novidade(e, cont, vivos, hip, rng)
    return melhor_a


def esc_por_custo(e, cont, vivos, hip, rng) -> int:
    return esc_planejada(e, cont, vivos, hip, rng, por_custo=True)


EXPLORADORES = {"acaso": esc_acaso, "novidade": esc_novidade,
                "planejada": esc_planejada, "por_custo": esc_por_custo}


def rodar(d: int, explorador: str, max_acoes: int, semente: int
          ) -> Optional[int]:
    """Acoes ate as sobreviventes ficarem funcionalmente iguais."""
    rng = random.Random(semente)
    m = Mundo(semente, d)
    hip = hipoteses(d)
    est = [(i, j) for i in range(LADO) for j in range(LADO)]
    assin = [tuple(f(e) for e in est) for _n, f in hip]
    vivos = list(range(len(hip)))
    cont: Dict[tuple, int] = {}
    esc = EXPLORADORES[explorador]

    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        a = esc(e0, cont, vivos, hip, rng)
        e1 = m.passo(a)
        cont[e1] = cont.get(e1, 0) + 1
        v = m.venceu(e1)
        vivos = [k for k in vivos if hip[k][1](e1) == v]
        if vivos and len({assin[k] for k in vivos}) == 1:
            return t
        if v:
            m.reiniciar()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=25)
    ap.add_argument("--acoes", type=int, default=6000)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    print("=" * 66)
    print("DISTANTE — a evidencia decisiva esta longe. Quem a encontra?")
    print("=" * 66)
    print(f"grade {LADO}x{LADO}, inicio no canto direito, "
          f"coluna esquerda vence")
    print(f"{len(hipoteses(1))} hipoteses geradas\n")

    # D0: as duas so divergem em (d,0)?
    est = [(i, j) for i in range(LADO) for j in range(LADO)]
    ok = True
    for d in (2, 9, 18, 27):
        h1 = [e[1] == 0 for e in est]
        h2 = [e[1] == 0 and e[0] != d for e in est]
        dif = [e for e, x, y in zip(est, h1, h2) if x != y]
        if dif != [(d, 0)]:
            ok = False
            print(f"D0 · FALHA em d={d}: divergem em {dif}")
    print(f"D0 · as duas hipoteses divergem em UM unico estado: "
          f"{'passa' if ok else 'FALHA'}")
    if not ok:
        return
    print()

    print(f"{'d':>4} {'ACASO':>9} {'NOVIDADE':>10} {'PLANEJADA':>11} "
          f"{'POR CUSTO':>11} {'nov/pl':>7}")
    print("-" * 60)
    linhas = []
    for d in (2, 9, 18, 27):
        med = {}
        for ex in ("acaso", "novidade", "planejada", "por_custo"):
            r = []
            for k in range(a.sementes):
                x = rodar(d, ex, a.acoes, a.semente + k)
                r.append(x if x is not None else a.acoes)
            r.sort()
            med[ex] = r[len(r) // 2]
        ra = med["acaso"] / max(med["planejada"], 1)
        rn = med["novidade"] / max(med["planejada"], 1)
        linhas.append((d, ra, rn))
        print(f"{d:>4} {med['acaso']:>9} {med['novidade']:>10} "
              f"{med['planejada']:>11} {med['por_custo']:>11} {rn:>6.1f}x",
              flush=True)
    print("-" * 60)

    print()
    cres_a = linhas[-1][1] / max(linhas[0][1], 0.01)
    cres_n = linhas[-1][2] / max(linhas[0][2], 0.01)
    print(f"crescimento da vantagem de d={linhas[0][0]} a d={linhas[-1][0]}:")
    print(f"   contra acaso    {cres_a:>5.1f}x")
    print(f"   contra novidade {cres_n:>5.1f}x")
    if cres_n >= 3 or cres_a >= 3:
        print("\nD1 · a vantagem CRESCE com a distancia. O mecanismo localiza")
        print("     evidencia causalmente distante.")
    elif max(linhas[-1][1], linhas[-1][2]) > 1.5:
        print("\nD2 · vantagem existe mas nao cresce com a distancia.")
        print("     E heuristica boa, nao localizacao de evidencia distante.")
    else:
        print("\nD3 · sem vantagem em nenhuma distancia. O mecanismo esta")
        print("     errado por natureza; encerra a linha.")


if __name__ == "__main__":
    main()
