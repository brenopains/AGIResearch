"""
novaphys.modific — modificadores de operacao, com sistema imunologico
=======================================================================

DOIS DEGRAUS DE UMA VEZ, E OS DOIS SAO PEDIDOS DO USUARIO
-----------------------------------------------------------
1. MODIFICADORES. Ate agora a gramatica produz `X` e nunca `inverso(X)`. Nenhum
   operador pode ser modificado depois de criado. O bebe que imagina laser nos
   olhos do ursinho faz duas coisas: destaca a propriedade da cena onde a viu, e
   a MODIFICA ao reaplicar. Nos temos a primeira metade (analogia) e zero da
   segunda.

2. SISTEMA IMUNOLOGICO. Medido quatro vezes hoje que geracao paralela sem
   portao duro produz lixo confiante:

       diversidade no feixe            8 -> 5, piorou
       novidade sobre o imprevisivel   0% vs 100% de convergencia
       credito parcial dirigindo acao  11 de 12 falhas
       oito lentes em paralelo         21x de sobreajuste, 127 ajustam / 6 passam

   O antidoto medido foi validacao cruzada: 4,7% -> 100% de precisao. Ampliar o
   espaco SEM ele e garantir alucinacao. Modificadores ampliam o espaco. Entao
   os dois vem juntos ou nenhum vem.

POR QUE MODIFICADOR E BARATO AQUI
-----------------------------------
O delta de um objeto ja e lido como tupla: (dy, dx, mesma_forma, cores, forma).
Modificador vira ARITMETICA na tupla:

    ident      (dy, dx)          nada muda
    inverso    (-dy, -dx)        mesma magnitude, sentido oposto
    dobro      (2dy, 2dx)
    metade     (dy//2, dx//2)
    esp_h      (dy, -dx)         espelha o eixo horizontal
    esp_v      (-dy, dx)
    transp     (dx, dy)          troca os eixos
    parado     (0, 0)            anula o deslocamento

Nao e arquitetura nova: e operar sobre o que ja esta sendo lido e jogado fora.

O QUE ISSO PASSA A EXPRESSAR, e era inexpressivel
---------------------------------------------------
Hoje a regra exige: todo objeto com a mesma propriedade sofre o MESMO delta.
Isso nao cobre "cada objeto se move em direcao ao centro" — cada um vai para um
lado, entao os deltas sao todos diferentes e nenhuma propriedade os prevê.

Com modificador: existe UM delta base e cada valor de propriedade escolhe um
modificador dele. "Os de cima descem, os de baixo sobem" vira
`(base, {topo: ident, fundo: inverso})`. E uma familia inteira do ARC.

E E LITERALMENTE SUBIR UMA ORDEM: de operadores para operadores-sobre-
operadores. Foi o unico tipo de mudanca que deu salto qualitativo hoje —
proposicional->quantificado no objetivo rendeu 49,6% -> 99,3%.

O SISTEMA IMUNOLOGICO, e ele nao e opcional
---------------------------------------------
Tres camadas, todas sem tocar o teste:

  1. VALIDACAO CRUZADA deixe-um-de-fora nos pares de treino. Uma regra que so
     existe porque viu o par i nao sobrevive a ser deduzida sem ele. Exige >= 3
     pares.
  2. OCKHAM: entre as sobreviventes, prefere-se `ident` (sem modificador) e a
     lente mais simples. Ordem declarada antes de rodar.
  3. RELATORIO DE PRECISAO: ajustou / sobreviveu / passou. Se precisao cair,
     o numero de tarefas NAO deve ser lido como progresso.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  M0  PRE-CONDICAO. As 4 tarefas de `ockham.py` precisam sobreviver.
      Acrescentar modificador so pode ADICIONAR familia.

  M1  >= 8/400 com precisao >= 70%
      -> modificadores sao o degrau, e o sistema imunologico segura
  M2  5 a 7 com precisao >= 70%
      -> ajuda, na ordem dos outros mecanismos
  M3  precisao < 70%
      -> ampliar com modificador furou o imunologico; o espaco cresceu mais do
         que a validacao cruzada consegue filtrar
  M4  <= 4 tarefas
      -> modificador nao acrescenta familia nenhuma

MINHA CALIBRAGEM: treze previsoes na sessao, CINCO certas. As oito erradas
todas otimistas — e as certas foram justamente as que eu fiz DEPOIS de comecar
a corrigir para o proprio vies.

PREVISAO: M2, 5 a 7 tarefas, precisao alta. Aposto que a precisao SEGURA (a
validacao cruzada foi perfeita em 127 -> 4 sem falso positivo) e que o ganho e
modesto, porque a familia "delta base modificado por posicao" e real no ARC mas
nao e das maiores. E aposto que `inverso` e `esp_h` rendem tudo e que `dobro`,
`metade` e `transp` rendem zero.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.segmenta import (LENTES, PROPS, Ce, Obj, deltas, fundo_de,
                               props, render)

# ══════════════════════════════════════════════ os modificadores
MODS: List[Tuple[str, Callable]] = [
    ("ident", lambda d: d),
    ("inverso", lambda d: (-d[0], -d[1]) + d[2:]),
    ("esp_h", lambda d: (d[0], -d[1]) + d[2:]),
    ("esp_v", lambda d: (-d[0], d[1]) + d[2:]),
    ("transp", lambda d: (d[1], d[0]) + d[2:]),
    ("dobro", lambda d: (2 * d[0], 2 * d[1]) + d[2:]),
    ("metade", lambda d: (d[0] // 2, d[1] // 2) + d[2:]),
    ("parado", lambda d: (0, 0) + d[2:]),
]
ORDEM_MOD = {n: i for i, (n, _f) in enumerate(MODS)}
ORDEM_LENTE = {n: i for i, (n, _f) in enumerate(LENTES)}
ORDEM_PROP = {p: i for i, p in enumerate(PROPS)}


def relativizar(d, ci):
    """
    A COR precisa ser RELATIVA para o modificador funcionar.

    `segmenta.deltas` guarda a cor ABSOLUTA da saida. Entao dois objetos de
    cores diferentes que sofrem o MESMO movimento tem deltas diferentes, e
    nenhum modificador (que so mexe em dy,dx) consegue levar um no outro. A
    familia "os de cima descem, os de baixo sobem" morria por causa da cor.

    E o mesmo erro ja corrigido em `cenadeduz2` e reintroduzido aqui: guardar
    valor absoluto onde a regra e sobre MUDANCA.
    """
    porid = {o.id: tuple(sorted({c for _y, _x, c in o.pix})) for o in ci.objs}
    fora = {}
    for oid, lst in d.items():
        cor_in = porid.get(oid, ())
        fora[oid] = tuple(sorted(
            ((dy, dx, mesma, (None if cores == cor_in else cores), forma)
             for (dy, dx, mesma, cores, forma) in lst), key=_chave))
    return fora


def _chave(t):
    """
    Ordenacao segura. `relativizar` poe None no campo de cor quando ela nao
    muda, entao `sorted` acabava comparando None com uma tupla e levantando
    TypeError — engolido pelo `except`, fazendo TODA tarefa devolver nada.
    Resultado: 0/400 inclusive no ajuste ao treino, que e impossivel como
    medida (o motor base ajusta 127). O portao M0 pegou.
    """
    return (t[0], t[1], bool(t[2]), t[3] is None, t[3] or (), str(t[4]))


def aplica_mod(nome: str, lst) -> tuple:
    f = dict(MODS)[nome]
    return tuple(sorted((f(d) for d in lst), key=_chave))


def deduzir_mod(treino, lente):
    """
    Deduz (propriedade, delta_base, {valor: modificador}, tela).

    A regra antiga exigia que todo objeto com a mesma propriedade sofresse o
    MESMO delta. Aqui existe UM delta base e cada valor de propriedade escolhe
    um MODIFICADOR dele — o que cobre "os de cima descem, os de baixo sobem",
    familia que a regra antiga nao podia expressar.
    """
    obs: Dict[str, Dict[Any, Set]] = {p: {} for p in PROPS}
    telas: Set[str] = set()
    for a, b in treino:
        bg = fundo_de(a)
        ci, co = lente(a, bg), lente(b, bg)
        if not ci.objs:
            return None
        ha, wa = dims(a)
        hb, wb = dims(b)
        telas.add("igual" if (hb, wb) == (ha, wa) else "corte")
        d = deltas(ci, co)
        if d is None:
            return None
        d = relativizar(d, ci)
        pi = props(ci)
        for oid, lst in d.items():
            obs_p = obs
            for p in PROPS:
                obs_p[p].setdefault(pi[oid][p], set()).add(lst)
    if len(telas) != 1:
        return None
    tela = next(iter(telas))

    # ERRO DE DESENHO CORRIGIDO, e ele zerou o motor (0/400 inclusive no
    # ajuste): eu fiz o modificador SUBSTITUIR o mecanismo antigo em vez de
    # ESTENDE-LO. O esquema novo exige que TODOS os valores sejam modificacoes
    # de UMA base — mas `n_buracos` mapeia cada contagem para uma COR
    # diferente, e modificador so mexe em dy,dx. A regra que funcionava deixou
    # de ser expressavel.
    #
    # Levantar guarda so pode ADICIONAR familia. Entao: mapa DIRETO primeiro
    # (o comportamento antigo, e o mais simples por Ockham), e base+modificador
    # so quando nenhuma propriedade da o mapa direto.
    for p in PROPS:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        mapa = {k: next(iter(v)) for k, v in obs[p].items()}
        if mapa:
            return (p, None, mapa, tela, 0)

    for p in PROPS:
        if any(len(v) > 1 for v in obs[p].values()):
            continue
        mapa = {k: next(iter(v)) for k, v in obs[p].items()}
        if not mapa:
            continue
        for base_k in sorted(mapa, key=lambda z: str(z)):
            base = mapa[base_k]
            escolha: Dict[Any, str] = {}
            ok = True
            for k, alvo in mapa.items():
                m = None
                for nome, _f in MODS:
                    if aplica_mod(nome, base) == alvo:
                        m = nome
                        break
                if m is None:
                    ok = False
                    break
                escolha[k] = m
            if ok:
                return (p, base, escolha, tela, len(set(escolha.values())))
    return None


def aplicar_mod(regra, g, lente):
    p, base, escolha, tela, _u = regra
    bg = fundo_de(g)
    c = lente(g, bg)
    if not c.objs:
        return None
    pr = props(c)
    novos, k = [], 0
    for o in c.objs:
        chave = pr[o.id][p]
        if chave not in escolha:
            return None
        # base=None e o mapa DIRETO (comportamento antigo); com base, cada
        # valor escolhe um modificador dela.
        saidas = (escolha[chave] if base is None
                  else aplica_mod(escolha[chave], base))
        for (dy, dx, mesma, cores, forma) in saidas:
            y0, x0, _a, _b = o.caixa()
            if cores is None:
                cores = tuple(sorted({c2 for _y, _x, c2 in o.pix}))
            if mesma:
                if len(cores) == 1:
                    pix = frozenset((y + dy, x + dx, cores[0])
                                    for y, x, _c in o.pix)
                else:
                    pix = frozenset((y + dy, x + dx, c2) for y, x, c2 in o.pix)
            else:
                if forma is None:
                    return None
                pix = frozenset((y + y0 + dy, x + x0 + dx, c2)
                                for y, x, c2 in forma)
            novos.append(Obj(k, pix))
            k += 1
    if not novos:
        return None
    if tela == "igual":
        if any(not (0 <= y < c.h and 0 <= x < c.w)
               for o in novos for y, x, _c in o.pix):
            return None
        return render(Ce(c.h, c.w, bg, novos))
    ys = [y for o in novos for y, _x, _c in o.pix]
    xs = [x for o in novos for _y, x, _c in o.pix]
    y0, x0 = min(ys), min(xs)
    des = [Obj(o.id, frozenset((y - y0, x - x0, c2) for y, x, c2 in o.pix))
           for o in novos]
    return render(Ce(max(ys) - y0 + 1, max(xs) - x0 + 1, bg, des))


# ══════════════════════════════════════════════ sistema imunologico
def imune(treino, lente) -> bool:
    """
    Deixe-um-de-fora nos proprios pares de TREINO. O teste nunca e tocado.
    Uma regra que so existe porque viu o par i nao sobrevive a ser deduzida
    sem ele. Foi o que levou a precisao de 4,7% para 100% em `ockham.py`.
    """
    if len(treino) < 3:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir_mod(resto, lente)
        if r is None:
            return False
        try:
            if aplicar_mod(r, treino[i][0], lente) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te = args
    cands = []
    ajustou = False
    for ln, lente in LENTES:
        try:
            r = deduzir_mod(tr, lente)
            if r is None:
                continue
            if any(aplicar_mod(r, a, lente) != b for a, b in tr):
                continue
            ajustou = True
            if not imune(tr, lente):
                continue
            # OCKHAM: menos modificadores distintos, lente e propriedade mais
            # simples. Ordem declarada antes de rodar.
            cands.append((r[4], ORDEM_LENTE[ln], ORDEM_PROP.get(r[0], 99),
                          ln, r, lente))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None
    cands.sort(key=lambda z: (z[0], z[1], z[2]))
    _u, _a, _b, ln, r, lente = cands[0]
    try:
        ok = all(aplicar_mod(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    # base=None e o mapa direto: r[2] guarda deltas, nao nomes de modificador.
    mods = ("direto" if r[1] is None
            else "+".join(sorted(set(r[2].values()))))
    return nome, ajustou, True, (f"{ln}|{r[0]}|{mods}" if ok else None)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 78)
    print("MODIFIC — modificadores de operacao + sistema imunologico")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)
    ajust = [n for n, aj, _v, _o in res if aj]
    sobrev = [n for n, _aj, v, _o in res if v]
    ok = {n: o for n, _aj, _v, o in res if o}

    print(f"\n  {len(MODS)} modificadores x {len(LENTES)} lentes x "
          f"{len(PROPS)} propriedades")
    print(f"\n  ajustou o TREINO       : {len(ajust):3d}/400")
    print(f"  sobreviveu ao IMUNE    : {len(sobrev):3d}/400")
    print(f"  passou no TESTE        : {len(ok):3d}/400   "
          f"({time.time()-t0:.0f}s)")
    prec = len(ok) / max(len(sobrev), 1)
    print(f"\n  precisao (teste/sobreviveu): {prec:.1%}   "
          f"(`ockham` sem modificador: 100%, 4 tarefas)")
    antes = {"37d3e8b2", "72a961c9", "b1fc8b8e", "f45f5ca7"}
    print(f"\nM0 · das 4 de `ockham`, mantidas: {sorted(antes & set(ok))}")
    print(f"     perdidas: {sorted(antes - set(ok))}")
    for k, v in Counter(o.split("|")[2] for o in ok.values()).most_common():
        print(f"    {v:2d}  modificadores: {k}")
    print(f"\n  tarefas: {sorted(ok)}")

    print("\n" + "=" * 78)
    if not antes <= set(ok):
        print(f"M0 FALHOU: perdi {sorted(antes - set(ok))}.")
    if prec < 0.70:
        print(f"M3 · precisao {prec:.1%}: modificador furou o imunologico. O")
        print("     espaco cresceu mais do que a validacao cruzada filtra.")
    elif len(ok) >= 8:
        print("M1 · modificadores sao o degrau, e o imunologico segura.")
    elif len(ok) >= 5:
        print("M2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("M4 · modificador nao acrescenta familia nenhuma.")


if __name__ == "__main__":
    main()
