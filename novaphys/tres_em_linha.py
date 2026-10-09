"""
novaphys.tres_em_linha — o principio de bits sai da inducao e vai para a CONSTRUCAO
=====================================================================================

POR QUE ESTE PROBLEMA
-----------------------
O usuario: gaps so aparecem em problemas EM ABERTO, com criterio rigido, onde o
proprio sistema tem que ver que nao funcionou. Mas problema do milenio, para nos,
so responde "falhou" — sinal zero, como o ARC a 1%. O meio-termo: problema em
aberto com VERIFICADOR AUTOMATICO e PLACAR CONTINUO.

"Nenhum trio em linha" (Dudeney, 1917): pôr 2n pontos numa grade n x n sem tres
colineares. 2n e o maximo (dois por linha). Sabe-se fazer para n <= 68 (recorde de
marco de 2026, Prellberg, classe de simetria rot4); a conjectura (Guy-Kelly) diz que
para n grande NAO da. O verificador tem 20 linhas. Qualquer n novo e matematica
nova. E mesmo falhando, ha placar: ate que n o sistema chega.

A PERGUNTA
------------
Todo o projeto usa um principio: DESCRICAO CURTA PRIMEIRO (MDL) — para EXPLICAR
dados. Aqui nao ha dado para explicar; ha um objeto para CONSTRUIR. O principio
atravessa da inducao para a construcao? Construcoes simetricas tem descricao curta
(so as orbitas), e foi por simetria que os recordes vieram. Um construtor que
tenta primeiro as descricoes curtas deveria chegar mais longe que um que procura
sem ordem.

Honestidade antes de rodar: os recordes vieram de SAT solvers e busca exaustiva
com poda (Flammenkamp, Heule, Prellberg), em codigo otimizado. Isto e Python num
PC. Nao vai chegar perto de 68. O que mede e se o principio de bits vira
estrategia de construcao — e onde a distancia entre DESCOBRIR e CONSTRUIR aparece.

O CONSTRUTOR
--------------
Grupos de simetria do quadrado (subgrupos de D4): iden, rot2, rot4, espelho_h,
espelho_d, klein (rot2 + dois espelhos), d4. Uma configuracao invariante sob G e
uma uniao de ORBITAS. Busca com retrocesso sobre as orbitas (ordenadas pela
primeira linha que tocam), dois pontos por linha e por coluna, colinearidade
checada a cada ponto, desempate aleatorio e reinicios.

  SEM ORDEM   so iden (a busca crua), orcamento inteiro
  MDL         os grupos em ordem de COMPRIMENTO DE DESCRICAO de uma configuracao
              invariante — log2 C(#orbitas, pontos / tamanho da orbita) — cada um
              com a mesma fatia do orcamento, o mais curto primeiro
  So n par (as classes rot4 e klein pedem n par para nao ter celula central).

PRE-REGISTRO, ANTES DE RODAR — semente 20260922, 60 s por n, n = 6, 8, 10, ...
(para no segundo n seguido sem solucao)
--------------------------------------------------------------------------------
  T0  todo "achado" passa num verificador INDEPENDENTE (O(n^3), todos os trios)
  T1  MDL chega a um n maximo MAIOR que SEM ORDEM
  (descritivo) quais classes de simetria deram solucao, contra a literatura

PREVISAO: SEM ORDEM para por volta de n = 14-18; MDL chega a n = 20-30. Nenhum
perto de 68. O gap que aparece e de CONSTRUCAO: velocidade, poda, e nao aprender
nada de um n para o seguinte.

RESULTADO 1: sem_ordem ate n = 10; MDL ate n = 18 (T1 passou). As duas previsoes de
alcance erraram OTIMISTAS. A classe vencedora foi rot4 em 6 de 7 n resolvidos — a
mesma do recorde de 68 (Prellberg). O gap, visivel na saida: a cada n o MDL gasta
a primeira fatia em d4 (curta, quase nunca tem solucao), porque NAO LEMBRA que
rot4 venceu no n anterior.

RODADA 2, PRE-REGISTRO — semente 20260923, 60 s por n, mdl x mdl_memoria alternados
-------------------------------------------------------------------------------------
Liga com o `caderno.py`: la a memoria ENTRINCHEIROU um erro porque o credito vinha
da propria avaliacao do agente. Aqui o credito vem do VERIFICADOR RIGIDO — so conta
como sucesso o que passou no teste de todos os trios. Hipotese: verificador rigido
torna a memoria SEGURA.
  MDL_MEMORIA  mesmo "restaurante chines" do caderno (B = 1): a fatia do grupo g e
               proporcional a (sucessos verificados de g + B/7) / (total + B); a
               ordem, por peso e depois por bits
  M1  mdl_memoria chega a um n maximo MAIOR que mdl
PREVISAO: M1 passa; mdl_memoria chega a n = 20-22 (rot4 passa a receber quase todo o
tempo). Se nao passar, o limite nao e alocacao — e o proprio construtor.

RESULTADO 2: M1 PASSOU — mdl 18, mdl_memoria 20 (previsao certa). A memoria deu
todo o credito a rot4 (8 de 8 sucessos verificados) e ficou ~5x mais rapida nos
mesmos n (n = 18: 2,8 s x 14,2 s). Mas com os 60 s inteiros em rot4, n = 22 falhou:
de 20 (1,3 s) para 22 (> 60 s) e a parede exponencial do retrocesso. A alocacao
estava resolvida; o limite agora e o CONSTRUTOR. Proximo degrau com mecanismo:
extensao — as solucoes de n e n+2 compartilham estrutura (ha literatura sobre o
problema "extensivel"); induzir o padrao das solucoes pequenas e usa-lo como
semente para a grande e o laco DESCOBRIR -> CONSTRUIR.

RODADA 3, PRE-REGISTRO — semente 20260924, 60 s por n, mdl_memoria x mdl_analogia
------------------------------------------------------------------------------------
MDL_ANALOGIA = mdl_memoria + a ultima solucao VERIFICADA (n - 2) centrada na grade
n (deslocada 1 em x e y — o mapa que preserva o centro e as simetrias) vira
PREFERENCIA de ordem na busca: orbita que toca uma celula sugerida e tentada
primeiro como "incluir". E analogia entre instancias, no sentido mais literal: "o
problema grande se parece com o pequeno com uma moldura". O mapa (centrar) e meu.
Contagem de sucessos agora SEPARADA por metodo (na rodada 2 era um dicionario so,
mas so um metodo o usava — sem contaminacao la).
  A1  mdl_analogia chega a um n maximo MAIOR que mdl_memoria
PREVISAO: incerta, e o palpite honesto e FALHA — as solucoes de n e n+2 na
literatura nao sao, em geral, uma a moldura da outra; a preferencia pode mais
atrapalhar que ajudar. Se passar, chega a 22-24.

RESULTADO 3: A1 passou PELA LETRA (analogia 22, memoria 20), mas fraco: a analogia
foi mais lenta em n = 16 e 18, falhou n = 20, e o sucesso em 22 veio com a solucao
de 18 deslocada 1 (bug: apos uma falha o deslocamento centrado e 2). E o MESMO
metodo resolveu n = 20 em 1,3 s numa rodada e 31 s noutra: maximo-n de corrida
unica e ruidoso demais para aceitar um "passou".

RODADA 4, PRE-REGISTRO — semente 20260925, 5 sementes, 60 s por n
--------------------------------------------------------------------
Taxa de sucesso por n em {18, 20, 22, 24}, com a memoria aquecida em 6..16 em cada
semente; deslocamento centrado corrigido ((n - n_anterior)/2); mdl_memoria x
mdl_analogia alternados.
  A2  soma em n = 20, 22, 24 de (analogia - memoria) >= +3 (de 15)
PREVISAO: FALHA (|diferenca| <= 2). Se falhar, o A1 da rodada 3 era ruido.

RESULTADO 4 (5 sementes):
                 n=18  n=20  n=22  n=24
  mdl_memoria      5     1     0     0
  mdl_analogia     5     3     0     0
A2 FALHOU (+2; previsao certa). O "22" da rodada 3 era ruido: nenhum metodo passa
de 20 em nenhuma das 5 sementes. A analogia ajuda um pouco NA BORDA (n = 20: 3 x 1),
nao empurra a fronteira. A parede em ~22 e do construtor (retrocesso em Python),
nao da alocacao nem da semente.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from math import comb, gcd, log2

sys.setrecursionlimit(20000)     # iden tem n^2 orbitas: 1600 niveis em n = 40
SEMENTE = 20260922

GRUPOS = {
    "iden": [lambda x, y, n: (x, y)],
    "rot2": [lambda x, y, n: (x, y), lambda x, y, n: (n - 1 - x, n - 1 - y)],
    "rot4": [lambda x, y, n: (x, y), lambda x, y, n: (n - 1 - y, x),
             lambda x, y, n: (n - 1 - x, n - 1 - y), lambda x, y, n: (y, n - 1 - x)],
    "espelho_h": [lambda x, y, n: (x, y), lambda x, y, n: (n - 1 - x, y)],
    "espelho_d": [lambda x, y, n: (x, y), lambda x, y, n: (y, x)],
    "klein": [lambda x, y, n: (x, y), lambda x, y, n: (n - 1 - x, y),
              lambda x, y, n: (x, n - 1 - y), lambda x, y, n: (n - 1 - x, n - 1 - y)],
}
GRUPOS["d4"] = GRUPOS["rot4"] + [lambda x, y, n: (y, x),
                                 lambda x, y, n: (n - 1 - x, y),
                                 lambda x, y, n: (x, n - 1 - y),
                                 lambda x, y, n: (n - 1 - y, n - 1 - x)]


def orbitas(n, g):
    visto, out = set(), []
    for y in range(n):
        for x in range(n):
            if (x, y) in visto:
                continue
            o = sorted({f(x, y, n) for f in GRUPOS[g]}, key=lambda p: (p[1], p[0]))
            visto.update(o)
            out.append(o)
    out.sort(key=lambda o: (o[0][1], o[0][0]))
    return out


def bits_descricao(n, g):
    """Comprimento de descricao de uma configuracao de 2n pontos invariante sob g."""
    obs = orbitas(n, g)
    tam = sum(len(o) for o in obs) / len(obs)
    k = max(1, round(2 * n / tam))
    return log2(comb(len(obs), min(k, len(obs)))) + log2(len(GRUPOS))


def _linha(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    g = gcd(dx, dy)
    dx, dy = dx // g, dy // g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy, dx * a[1] - dy * a[0])


def verifica(pts, n):
    """Independente: 2n pontos distintos na grade e nenhum trio colinear."""
    pts = list(pts)
    if len(pts) != 2 * n or len(set(pts)) != len(pts):
        return False
    if any(not (0 <= x < n and 0 <= y < n) for x, y in pts):
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                (ax, ay), (bx, by), (cx, cy) = pts[i], pts[j], pts[k]
                if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) == 0:
                    return False
    return True


def busca(n, g, prazo, rng, pref=None):
    """Retrocesso sobre orbitas com reinicios ate o prazo (tempo absoluto).
    `pref`: celulas sugeridas por ANALOGIA com a solucao menor — a orbita que toca
    uma delas e tentada primeiro como 'incluir' (com 10% de desvio por reinicio)."""
    obs = orbitas(n, g)
    # uma orbita que ja tem 3 pontos colineares, ou 3 na mesma linha/coluna, nao serve
    boas = []
    for o in obs:
        lin = {}
        ok = True
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                L = _linha(o[i], o[j])
                lin.setdefault(L, set()).update((o[i], o[j]))
                if len(lin[L]) > 2:
                    ok = False
        if ok:
            boas.append(o)
    ultima = {}
    for i, o in enumerate(boas):
        for (_x, y) in o:
            ultima[y] = max(ultima.get(y, -1), i)
    alvo_linha = {y: ultima.get(y, -1) for y in range(n)}
    if any(v < 0 for v in alvo_linha.values()):
        return None

    while time.time() < prazo:
        pts, linhas = [], {}
        cont_l, cont_c = [0] * n, [0] * n
        pilha = []

        def pode(o):
            if any(cont_l[y] >= 2 or cont_c[x] >= 2 for x, y in o):
                return False
            ll = {}
            for (x, y) in o:
                if cont_l[y] + sum(1 for p in o if p[1] == y) > 2:
                    return False
                if cont_c[x] + sum(1 for p in o if p[0] == x) > 2:
                    return False
            novos = []
            for p in o:
                for a in pts + novos:
                    L = _linha(a, p)
                    s = linhas.get(L, set()) | ll.get(L, set())
                    if len(s | {a, p}) > 2:
                        return False
                    ll.setdefault(L, set()).update((a, p))
                novos.append(p)
            return True

        def poe(o):
            mud = []
            for p in o:
                for a in pts:
                    L = _linha(a, p)
                    if L not in linhas:
                        linhas[L] = set()
                        mud.append((L, None))
                    else:
                        mud.append((L, set(linhas[L])))
                    linhas[L].update((a, p))
                pts.append(p)
                cont_l[p[1]] += 1
                cont_c[p[0]] += 1
            return mud

        def tira(o, mud):
            for p in reversed(o):
                pts.pop()
                cont_l[p[1]] -= 1
                cont_c[p[0]] -= 1
            for L, antes in reversed(mud):
                if antes is None:
                    del linhas[L]
                else:
                    linhas[L] = antes

        passos = [0]
        limite = 20000 * n

        def rec(i):
            passos[0] += 1
            if passos[0] > limite or time.time() > prazo:
                return None
            # toda linha cuja ultima orbita ja passou tem que ter exatamente 2
            for y, u in alvo_linha.items():
                if u < i and cont_l[y] != 2:
                    return False
            if i == len(boas):
                return len(pts) == 2 * n
            o = boas[i]
            if pref is not None:
                sim = any(p in pref for p in o)
                if rng.random() < 0.1:
                    sim = not sim
                escolhas = [True, False] if sim else [False, True]
            else:
                escolhas = [True, False] if rng.random() < 0.5 else [False, True]
            for pega in escolhas:
                if pega:
                    if not pode(o):
                        continue
                    mud = poe(o)
                    r = rec(i + 1)
                    if r:
                        return True
                    tira(o, mud)
                    if r is None:
                        return None
                else:
                    r = rec(i + 1)
                    if r:
                        return True
                    if r is None:
                        return None
            return False

        r = rec(0)
        if r:
            return list(pts)
    return None


def passo(metodo, n, prazo, rng, usos, ultima):
    """Um metodo em um n. `ultima` = (solucao, n dela) ou None."""
    t0 = time.time()
    achado = classe = None
    bits = {g: bits_descricao(n, g) for g in GRUPOS}
    if metodo == "sem_ordem":
        achado = busca(n, "iden", t0 + prazo, rng)
        classe = "iden" if achado else None
    elif metodo in ("mdl_memoria", "mdl_analogia"):
        tot = sum(usos.values())
        peso = {g: (usos[g] + 1.0 / len(GRUPOS)) / (tot + 1.0) for g in GRUPOS}
        ordem = sorted(GRUPOS, key=lambda g: (-peso[g], bits[g]))
        pref = None
        if metodo == "mdl_analogia" and ultima is not None:
            sol, nprev = ultima
            d = (n - nprev) // 2          # CENTRADA (rodada 3 usava sempre 1)
            pref = {(x + d, y + d) for x, y in sol}
        for g in ordem:
            achado = busca(n, g, time.time() + prazo * peso[g], rng, pref)
            if achado:
                classe = g
                break
    else:
        for g in sorted(GRUPOS, key=lambda g: bits[g]):
            achado = busca(n, g, time.time() + prazo / len(GRUPOS), rng)
            if achado:
                classe = g
                break
    ok = achado is not None and verifica(achado, n)
    if achado is not None and not ok:
        print(f"    n={n:2d}  {metodo} ACHADO INVALIDO — bug no construtor")
    return ok, classe, time.time() - t0, achado


def repete(metodos, prazo, semente, k):
    """Rodada 4: taxa de sucesso por n, varias sementes, memoria aquecida em 6..16."""
    ns = (18, 20, 22, 24)
    conta = {m: {n: 0 for n in ns} for m in metodos}
    for s in range(k):
        usos = {m: {g: 0 for g in GRUPOS} for m in metodos}
        ultima = {m: None for m in metodos}
        rngs = {m: random.Random(semente + 1000 * s) for m in metodos}
        print(f"\n  --- semente {s} ---")
        for n in range(6, max(ns) + 1, 2):
            for m in metodos:
                ok, classe, dt, achado = passo(m, n, prazo, rngs[m], usos[m], ultima[m])
                if ok:
                    usos[m][classe] += 1
                    ultima[m] = (achado, n)
                if n in ns:
                    conta[m][n] += ok
                    print(f"    n={n:2d}  {m:13s} {'SOLUCAO' if ok else 'nada':8s} "
                          f"{classe or '':6s} {dt:5.1f}s")
    print("")
    print("=" * 76)
    print("  sucessos por n (de " + str(k) + "):  " + "   ".join(f"n={n}" for n in ns))
    for m in metodos:
        print(f"  {m:13s} " + "   ".join(f"{conta[m][n]:4d}" for n in ns))
    if "mdl_analogia" in metodos and "mdl_memoria" in metodos:
        dif = sum(conta["mdl_analogia"][n] - conta["mdl_memoria"][n] for n in (20, 22, 24))
        print(f"A2 {'.' if dif >= 3 else 'FALHOU .'} analogia - memoria em n = 20, 22, 24: "
              f"{dif:+d} (limiar +3 de {3 * k})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repete", type=int, default=0)
    ap.add_argument("--prazo", type=float, default=60.0)
    ap.add_argument("--max_n", type=int, default=40)
    ap.add_argument("--metodos", default="sem_ordem,mdl")
    ap.add_argument("--semente", type=int, default=SEMENTE)
    a = ap.parse_args()
    metodos = a.metodos.split(",")
    if a.repete:
        print("=" * 76)
        print(f"NENHUM TRIO EM LINHA — REPETICAO, {a.repete} sementes, {a.prazo:.0f} s por n")
        print("=" * 76)
        repete(metodos, a.prazo, a.semente, a.repete)
        return
    # sucessos VERIFICADOS por grupo, SEPARADOS por metodo (nao contaminar)
    usos_m = {m: {g: 0 for g in GRUPOS} for m in metodos}
    ultima = {m: None for m in metodos}  # ultima solucao de cada metodo
    print("=" * 76)
    print(f"NENHUM TRIO EM LINHA — {a.prazo:.0f} s por n")
    print("=" * 76)
    # os dois metodos ALTERNADOS a cada n: se a carga da maquina mudar no meio,
    # muda para os dois ao mesmo tempo
    resultado = {m: 0 for m in metodos}
    falhas = {m: 0 for m in metodos}
    rngs = {m: random.Random(a.semente) for m in metodos}
    for n in range(6, a.max_n + 1, 2):
        for metodo in metodos:
            if falhas[metodo] >= 2:
                continue
            rng = rngs[metodo]
            t0 = time.time()
            achado, classe = None, None
            if metodo == "sem_ordem":
                achado = busca(n, "iden", t0 + a.prazo, rng)
                classe = "iden" if achado else None
            elif metodo in ("mdl_memoria", "mdl_analogia"):
                usos = usos_m[metodo]
                bits = {g: bits_descricao(n, g) for g in GRUPOS}
                tot = sum(usos.values())
                peso = {g: (usos[g] + 1.0 / len(GRUPOS)) / (tot + 1.0) for g in GRUPOS}
                ordem = sorted(GRUPOS, key=lambda g: (-peso[g], bits[g]))
                pref = None
                if metodo == "mdl_analogia" and ultima[metodo] is not None:
                    # a solucao de n-2 CENTRADA em n: desloca 1 em x e y (preserva
                    # o centro, logo as simetrias rot2/rot4/klein/d4)
                    pref = {(x + 1, y + 1) for x, y in ultima[metodo]}
                for g in ordem:
                    achado = busca(n, g, time.time() + a.prazo * peso[g], rng, pref)
                    if achado:
                        classe = g
                        break
            else:
                # descricao curta PRIMEIRO, fatias iguais: sem constante de
                # temperatura minha (Levin puro daria todo o tempo a d4, que nao
                # tem solucao para n grande)
                bits = {g: bits_descricao(n, g) for g in GRUPOS}
                ordem = sorted(GRUPOS, key=lambda g: bits[g])
                for g in ordem:
                    fatia = a.prazo / len(ordem)
                    achado = busca(n, g, time.time() + fatia, rng)
                    if achado:
                        classe = g
                        break
            dt = time.time() - t0
            ok = achado is not None and verifica(achado, n)
            if achado is not None and not ok:
                print(f"    n={n:2d}  {metodo:9s} ACHADO INVALIDO — bug no construtor")
            print(f"    n={n:2d}  {metodo:9s} {'SOLUCAO' if ok else 'nada':8s} "
                  f"{classe or '':10s} {dt:5.1f}s")
            if ok:
                resultado[metodo], falhas[metodo] = n, 0
                usos_m[metodo][classe] += 1    # credito so pelo VERIFICADOR
                ultima[metodo] = achado
            else:
                falhas[metodo] += 1
        if all(f >= 2 for f in falhas.values()):
            break
    print("")
    print("=" * 76)
    print("  maior n resolvido:  " + "   ".join(f"{m} {v}" for m, v in resultado.items())
          + "   (recorde humano+maquina: 68)")
    if "sem_ordem" in resultado and "mdl" in resultado:
        print(f"T1 {'.' if resultado['mdl'] > resultado['sem_ordem'] else 'FALHOU .'} "
              f"MDL chega mais longe que a busca sem ordem")
    if "mdl_memoria" in resultado and "mdl" in resultado:
        print(f"M1 {'.' if resultado['mdl_memoria'] > resultado['mdl'] else 'FALHOU .'} "
              f"memoria verificada chega mais longe que MDL sem memoria")
    if "mdl_analogia" in resultado and "mdl_memoria" in resultado:
        print(f"A1 {'.' if resultado['mdl_analogia'] > resultado['mdl_memoria'] else 'FALHOU .'} "
              f"analogia com a solucao menor chega mais longe que so a memoria")
    for m in metodos:
        print(f"   usos verificados ({m}): "
              + ", ".join(f"{g} {v}" for g, v in usos_m[m].items() if v))


if __name__ == "__main__":
    main()
