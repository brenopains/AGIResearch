"""
novaphys.elimina_cor — a metade RESOLVÍVEL da pilha do ARC
============================================================

DE ONDE VEM
-------------
[[partir-adivinhado-resolvido]] mediu que abdução funciona onde existe uma parte
RESOLVÍVEL cujo resíduo pontua continuamente a parte adivinhada, e que dez nulos
se acumularam justamente onde a pilha inteira é adivinhada contra verificador
binário.

Revisão externa deu a partição exata para o ARC, e ela é por TIPO DE INCÓGNITA —
estrutura contra valores — e não por prefixo contra sufixo:

    ADIVINHAR   a estrutura que determina QUAIS células recebem QUAL fonte
    RESOLVER    os VALORES de pintura, compartilhados entre os exemplos

E a receita geral tem nome na literatura: `variable projection` / mínimos
quadrados separáveis. R(s) = min_θ Σ_i L(P_{s,θ}(x_i), y_i) é a melhor adequação
POSSÍVEL daquela estrutura depois de resolver os parâmetros — não uma previsão
de utilidade futura.

O RECORTE ONDE VALE
---------------------
`Preenche.prever(m, a)` devolve `{cercadas(a): m}` e `Constante.prever(m, a)`
devolve `{tudo: m}`. Em ambos **a máscara não depende de `m`**. Esse é o recorte
de "cobertura independente das cores", e é onde a eliminação por contagem é
exata.

E hoje a busca ENUMERA essas cores como itens de pilha separados —
`Preenche.propor` devolve uma lista de cores candidatas, cada uma virando um nó
de busca. Isto é literalmente adivinhar o que dá para resolver.

A CONTA
---------
Com a atribuição de células fixa, para cada slot j:

    n[j][c] = #{células do slot j cujo alvo tem cor c}
    n[j]    = Σ_c n[j][c]
    θ*[j]   = argmax_c n[j][c]                       (cor majoritária)

    E(s) = E_fixo + Σ_j ( n[j] − max_c n[j][c] )

`E_fixo` conta erros das saídas já determinadas E as células não reivindicadas,
porque `custo` deste projeto cobra célula não explicada como errada.

A EXIGÊNCIA ANTES DE QUALQUER ALEGAÇÃO
----------------------------------------
Qualquer divergência entre o mínimo CALCULADO e a ENUMERAÇÃO EXAUSTIVA invalida
a implementação deste recorte. Nada de utilidade é medido antes disso fechar.

  N0  PRÉ-CONDIÇÃO. Em N casos pequenos, `E(s)` calculado == min sobre todas as
      10 cores rodadas no executor ORIGINAL, e a cor escolhida reproduz esse
      mínimo quando reexecutada. Divergência de 1 caso já reprova.
  N1  fecha em 100% dos casos -> a eliminação está correta e pode ser medida
      quanto a utilidade
  N2  qualquer divergência -> a implementação está errada; diagnosticar antes
      de seguir

E uma honestidade sobre novidade: `arc.ajustar_mapa` já resolve um mapa global
de cores a partir dos pares — mas devolve `None` no PRIMEIRO conflito. A
generalização que importa é **conservar o mínimo de conflitos em vez de
desistir**, e resolver vários parâmetros compartilhados conjuntamente. "Resolver
cor a partir dos exemplos" não é descoberta nova.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys.descricao import (CODECS, ENTAO, _marca, _passo, bits_excecao,
                                custo, dims)

# codecs cuja MASCARA nao depende da cor: o recorte onde a conta e exata
SEGUROS = ("preenche", "constante")


def _mascara(cd, g) -> set:
    """As células que o codec reivindica. Só é válido chamar com codec SEGURO."""
    try:
        return set(cd.prever(0, g).keys())
    except Exception:
        return set()


def mascara_independe_da_cor(cd, g, cores=range(10)) -> bool:
    """
    CONFERE em vez de confiar no nome: a máscara é a mesma para toda cor?
    O critério é dependência semântica, não a lista de nomes seguros.
    """
    base = None
    for c in cores:
        try:
            m = set(cd.prever(c, g).keys())
        except Exception:
            return False
        if base is None:
            base = m
        elif m != base:
            return False
    return base is not None


def residuo_minimo(prefixo, cd, treino) -> Optional[Tuple[int, int, int]]:
    """
    E(s), a cor ótima, e o total de células do slot — SEM rodar as 10 cores.

    Exige que `prefixo` termine em marcador (ou seja vazio), para que `cd` abra
    um estágio novo e reivindique a máscara inteira. Assim a atribuição de
    células ao slot é fixa e a contagem vale.
    """
    if prefixo and not _marca(prefixo[-1]):
        return None
    passos = _passo(prefixo, treino) if prefixo else \
        [(a, set()) for a, _b in treino]

    e_fixo = 0
    cont = Counter()
    n_slot = 0
    for (g, expl), (_a, b) in zip(passos, treino):
        hb, wb = dims(b)
        if dims(g) != (hb, wb):
            return None
        if not mascara_independe_da_cor(cd, g):
            return None
        m = _mascara(cd, g)
        for i in range(hb):
            for j in range(wb):
                p = (i, j)
                if p in m:
                    cont[b[i][j]] += 1
                    n_slot += 1
                elif p not in expl or g[i][j] != b[i][j]:
                    e_fixo += 1
    if n_slot == 0:
        return e_fixo, 0, 0
    cor, maj = cont.most_common(1)[0]
    return e_fixo + (n_slot - maj), cor, n_slot


def por_enumeracao(prefixo, cd, treino) -> Optional[Tuple[int, int]]:
    """O mesmo mínimo, rodando as 10 cores no executor ORIGINAL."""
    melhor, arg = None, None
    for c in range(10):
        try:
            _b, err = custo(list(prefixo) + [(cd, c)], treino)
        except Exception:
            continue
        if melhor is None or err < melhor:
            melhor, arg = err, c
    return (melhor, arg) if melhor is not None else None


def experimento(n_tarefas: int = 120) -> None:
    from novaphys.arc import carregar
    from novaphys.promessa import PASTA

    cds = [c for c in CODECS if c.nome in SEGUROS]
    print("=" * 78)
    print("N0 — o mínimo CALCULADO bate a ENUMERAÇÃO EXAUSTIVA?")
    print("=" * 78)
    print("  codecs do recorte: %s" % ", ".join(c.nome for c in cds))
    print("  prefixos: vazio, e um codec + ENTAO\n")

    casos, iguais, div = 0, 0, []
    tarefas = carregar(PASTA)[:n_tarefas]
    for nome, tr, _te in tarefas:
        prefixos = [[]]
        for outro in CODECS[:8]:
            try:
                ms = list(outro.propor(tr) or [])[:1]
            except Exception:
                continue
            for m in ms:
                prefixos.append([(outro, m), ENTAO])
        for pre in prefixos:
            for cd in cds:
                calc = residuo_minimo(pre, cd, tr)
                if calc is None:
                    continue
                enu = por_enumeracao(pre, cd, tr)
                if enu is None:
                    continue
                casos += 1
                if calc[0] == enu[0]:
                    iguais += 1
                else:
                    div.append((nome, cd.nome, len(pre), calc[0], enu[0]))
        if casos >= 400:
            break

    print("  casos comparados: %d" % casos)
    print("  batem exatamente: %d (%.1f%%)" % (iguais, 100 * iguais / max(casos, 1)))
    if div:
        print("\n  DIVERGENCIAS (primeiras 8):")
        for d in div[:8]:
            print("    %s  codec=%s  |prefixo|=%d  calculado=%d  enumerado=%d"
                  % d)
    print("\n" + "=" * 78)
    if casos and iguais == casos:
        print("N1 . A ELIMINACAO ESTA CORRETA em %d casos. A metade resolvivel" % casos)
        print("     da pilha existe e e computavel sem rodar as cores.")
    else:
        print("N2 . DIVERGE em %d de %d. A implementacao esta errada; nada de"
              % (casos - iguais, casos))
        print("     utilidade pode ser medido antes de fechar isso.")


if __name__ == "__main__":
    experimento()
