"""
novaphys.generalizacao — a composição é capacidade ou regra codificada?
=======================================================================

A bancada de emergência mediu 0% → 100% na família "herança" depois que eu
implementei a composição `Reach(A→D) ∘ Attribute(D=V)` em `_forward_chain`.

Esse número, sozinho, não distingue duas coisas muito diferentes:

    (a) o sistema ganhou a capacidade de compor mecanismos;
    (b) o sistema ganhou UMA regra, escrita à mão, para UM formato de item —
        justamente o formato que a bancada cobrava.

(b) é o modo normal de um benchmark ser vencido sem que nada tenha melhorado,
e quem escreveu a correção é a última pessoa qualificada para afirmar que foi
(a). Então esta bancada não pergunta a ele: pergunta a **composições que eu
não implementei**.

O ORÁCULO É UM SÓ, E É ANTERIOR À IMPLEMENTAÇÃO
------------------------------------------------
Isto é o que impede a bancada de ser desenhada em torno da resposta. Não há
um oráculo por padrão. Há um princípio único:

    Trate `causal_edges` e as regras de atributo como arcos do MESMO grafo,
    e feche por transitividade.

    reach(A,B)  é um arco A→B
    has(C,P)    é um arco C→P

Tudo o mais sai daí. `A→B→C` mais `C tem P` dá `A ⇝ P` porque é um caminho.
`C tem P` mais `P tem Q` dá `C ⇝ Q` pelo mesmo motivo, e nenhuma linha nova
foi escrita para esse caso. Os cinco padrões abaixo são caminhos diferentes
no mesmo grafo — o oráculo não sabe distingui-los, e é isso que o torna um
teste de generalização e não um gabarito.

OS PADRÕES
-----------
    P1  reach ∘ attr          A→B→C, C tem P              ⟹  A ⇝ P
    P2  attr ∘ attr           C tem P, P tem Q            ⟹  C ⇝ Q
    P3  attr ∘ reach          A tem B, B→C                ⟹  A ⇝ C
    P4  reach ∘ attr ∘ attr   A→B→C, C tem P, P tem Q     ⟹  A ⇝ Q
    P5  chain ∘ attr          mesma composição de P1, mas o alcance vem de
                              `causal_chains` e não de `causal_edges`

P1 é o padrão implementado — está aqui como referência, não como resultado.
P5 é o controle POSITIVO do arnês: é a mesma composição num campo diferente,
então se nem ele transferir, a bancada está medindo formato de entrada e não
capacidade, e nenhuma das outras notas vale. Um arnês que só sabe reprovar é
tão inútil quanto um que só sabe aprovar.

P6 — O PADRÃO QUE NÃO DEVE SER RESOLVIDO
-----------------------------------------
    P6  exceção              A→B→C, C tem P, mas "B bloqueia P"  ⟹  NADA

O fecho sobre arcos vai concluir `A ⇝ P` aqui, e vai estar errado: há uma
premissa que derruba a conclusão. P6 existe porque a generalização tem um
preço, e uma bancada que só mede o que a mudança melhora esconde o preço.
Raciocínio real é derrotável; fecho transitivo não é.
"""

from __future__ import annotations

import os
import random
import re
import sys
from typing import Any, Callable, Dict, List, Set, Tuple

from .emergencia import (_carregar_reasoner, _E, _regra_atributo, _tok,
                         _fecho, _menciona, alegacoes)


# ══════════════════════════════════════════════ o oráculo, um só
def _arcos(kb: Dict) -> Set[Tuple[str, str]]:
    """
    O grafo unificado. `reach` e `has` viram arcos do mesmo tipo — é essa
    unificação, e só ela, que gera as respostas dos seis padrões.
    """
    arcos = {(e["source"], e["target"]) for e in kb.get("causal_edges", [])}
    for ch in kb.get("causal_chains", []):
        for i in range(len(ch) - 1):
            arcos.add((ch[i], ch[i + 1]))
    for r in kb.get("semantic_rules", []):
        c = r.get("content", {})
        ab = c.get("abstraction", {})
        if c.get("type") == "attribute" and ab.get("frequency", 0) > 0.5:
            arcos.add((ab.get("attribute", ""), ab.get("value", "")))
    return {(s, t) for (s, t) in arcos if s and t}


def _so_alcance(kb: Dict) -> Set[Tuple[str, str]]:
    """Mecanismo 1 isolado: fecho só sobre as arestas causais."""
    ar = {(e["source"], e["target"]) for e in kb.get("causal_edges", [])}
    for ch in kb.get("causal_chains", []):
        for i in range(len(ch) - 1):
            ar.add((ch[i], ch[i + 1]))
    return _fecho(ar)


def _so_atributos(kb: Dict) -> Set[Tuple[str, str]]:
    """Mecanismo 2 isolado: os pares das regras, sem encadeamento."""
    out = set()
    for r in kb.get("semantic_rules", []):
        c = r.get("content", {})
        ab = c.get("abstraction", {})
        if c.get("type") == "attribute" and ab.get("frequency", 0) > 0.5:
            out.add((ab.get("attribute", ""), ab.get("value", "")))
    return out


def oraculo(kb: Dict) -> Set[Tuple[str, str]]:
    """Fecho sobre o grafo unificado. Vale para os seis padrões, sem ramo."""
    return _fecho(_arcos(kb))


# Marcadores de retratação. Isto é uma CONVENÇÃO sobre o vocabulário de saída
# do motor, e é a parte frágil desta bancada: um sistema que prefixasse
# "Defeated:" em tudo passaria em P6 sem raciocinar. A versão robusta lê saída
# estruturada — a conclusão com um campo de polaridade — em vez de prosa. Fica
# registrado como limite conhecido, não como detalhe de implementação.
_NEGATIVAS = ("defeated", "does not hold", "blocked", "retracted", "não vale")


def _negativa(texto: str) -> bool:
    t = texto.lower()
    return any(m in t for m in _NEGATIVAS)


# ══════════════════════════════════════════════ os padrões
def _conc(label: str, alvo_conexao: str) -> List[Dict]:
    return [{"label": label, "position": [0.1] * 8,
             "connections": {"observado_em": [alvo_conexao]}}]


def p1_reach_attr(rng, controle=False) -> Dict:
    a, b, c, z = _tok(rng, "g", 4)
    p = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [_E(a, b), _E(b, c)],
          "causal_chains": [], "semantic_rules": [_regra_atributo(z if controle else c, p)]}
    return {"padrao": "P1 reach∘attr", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (a, p), "controle": controle}


def p2_attr_attr(rng, controle=False) -> Dict:
    c, z = _tok(rng, "g", 2)
    p = f"prop{rng.randint(100, 999)}"
    q = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(c, "amostra"), "causal_edges": [], "causal_chains": [],
          "semantic_rules": [_regra_atributo(c, p),
                             _regra_atributo(z if controle else p, q)]}
    return {"padrao": "P2 attr∘attr", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (c, q), "controle": controle}


def p3_attr_reach(rng, controle=False) -> Dict:
    a, b, c, w = _tok(rng, "g", 4)
    origem = w if controle else b
    kb = {"concepts": _conc(a, "amostra"), "causal_edges": [_E(origem, c)],
          "causal_chains": [], "semantic_rules": [_regra_atributo(a, b)]}
    return {"padrao": "P3 attr∘reach", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (a, c), "controle": controle}


def p4_tripla(rng, controle=False) -> Dict:
    a, b, c, z = _tok(rng, "g", 4)
    p = f"prop{rng.randint(100, 999)}"
    q = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [_E(a, b), _E(b, c)],
          "causal_chains": [],
          "semantic_rules": [_regra_atributo(c, p),
                             _regra_atributo(z if controle else p, q)]}
    return {"padrao": "P4 reach∘attr∘attr", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (a, q), "controle": controle}


def p5_chain_attr(rng, controle=False) -> Dict:
    a, b, c, z = _tok(rng, "g", 4)
    p = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [], "causal_chains": [[a, b, c]],
          "semantic_rules": [_regra_atributo(z if controle else c, p)]}
    return {"padrao": "P5 chain∘attr", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (a, p), "controle": controle}


PADROES: List[Tuple[str, Callable]] = [
    ("P1 reach∘attr", p1_reach_attr),
    ("P2 attr∘attr", p2_attr_attr),
    ("P3 attr∘reach", p3_attr_reach),
    ("P4 reach∘attr∘attr", p4_tripla),
    ("P5 chain∘attr", p5_chain_attr),
]


# ══════════════════════════════════════════════ P6 — derrotabilidade
def p6_excecao(rng) -> Dict:
    """
    A→B→C, C tem P, e uma regra explícita diz que B BLOQUEIA P.

    A conclusão `A ⇝ P` deixa de valer. Não porque falte informação: porque
    chegou informação que a derruba. Nenhum fecho transitivo representa isso —
    fecho é monotônico por construção, e monotonicidade é a hipótese de que
    saber mais nunca faz você concluir menos.
    """
    a, b, c = _tok(rng, "x", 3)
    p = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [_E(a, b), _E(b, c)],
          "causal_chains": [],
          "semantic_rules": [_regra_atributo(c, p),
                             {"content": {"type": "blocking",
                                          "abstraction": {"blocker": b, "blocked": p,
                                                          "frequency": 1.0}}}],
          "bloqueios": [{"blocker": b, "blocked": p}]}
    return {"padrao": "P6 exceção", "kb": kb, "conceitos": kb["concepts"],
            "proibido": (a, p)}


# ══════════════════════════════════════════════ P7 e P8 — os retidos
#
# P1–P5 deixaram de ser teste de generalização no instante em que o fecho
# unificado foi implementado: eles agora testam se a implementação bate com o
# oráculo, e bate por construção. Passar neles não é mais evidência de nada.
#
# P7 e P8 existem para manter a auditoria viva. Nenhum dos dois é um caminho
# no grafo de relações, que é a única coisa que o mecanismo sabe fazer — e a
# graça é justamente essa: eles marcam a FRONTEIRA da capacidade, que é uma
# informação melhor que uma nota alta.

def p7_conjuncao(rng) -> Dict:
    """
    "X e Y JUNTOS produzem Z", com X e Y ambos estabelecidos.

    Não é um caminho. Nenhuma sequência de arcos leva de X a Z, porque a
    premissa exige as DUAS coisas ao mesmo tempo — é uma inferência de aridade
    2, e fecho transitivo é de aridade 1 por definição. Um sistema que compõe
    caminhos perfeitamente pode ser incapaz disto, e a distinção importa:
    quase toda regra física real ("se há gradiente E há condutividade, há
    corrente") tem essa forma.
    """
    a, x, y = _tok(rng, "c", 3)
    z = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, x), "causal_edges": [_E(a, x), _E(a, y)],
          "causal_chains": [], "semantic_rules": [
              {"content": {"type": "conjunction",
                           "abstraction": {"requires": [x, y], "produces": z,
                                           "frequency": 0.95}}}]}
    return {"padrao": "P7 conjunção", "kb": kb, "conceitos": kb["concepts"],
            "alvo": (a, z)}


def p8_tipo_de_relacao(rng) -> Dict:
    """
    A --causa--> B,  B --correlaciona_com--> C.  NÃO se segue que A causa C.

    Causalidade compõe com causalidade. Correlação não compõe com nada — é o
    erro elementar da inferência causal, e o motor está estruturalmente exposto
    a ele porque `_arcos` DESCARTA o campo `type` da aresta. Unificar os arcos
    foi o que fez P2–P4 funcionarem; jogar fora o tipo da relação foi o preço,
    e é aqui que ele aparece.
    """
    a, b, c = _tok(rng, "r", 3)
    kb = {"concepts": _conc(a, b),
          "causal_edges": [_E(a, b, "causa"), _E(b, c, "correlaciona_com")],
          "causal_chains": [], "semantic_rules": []}
    return {"padrao": "P8 tipo de relação", "kb": kb, "conceitos": kb["concepts"],
            "proibido": (a, c)}


# ══════════════════════════════════════════════ crivo
def crivar(item: Dict) -> Tuple[bool, str]:
    kb = item["kb"]
    if item.get("controle"):
        proib = item["alvo"]
        if proib in oraculo(kb):
            return False, "controle inválido: o oráculo produz o par"
        return True, "ok"
    alvo = item["alvo"]
    if alvo in _so_alcance(kb):
        return False, "não composicional: o alcance sozinho basta"
    if alvo in _so_atributos(kb):
        return False, "não composicional: os atributos sozinhos bastam"
    if alvo not in oraculo(kb):
        return False, "insolúvel: nem o oráculo unificado produz"
    return True, "ok"


# ══════════════════════════════════════════════ medição
def avaliar(n: int = 30, semente: int = 0, verboso: bool = False) -> Dict[str, Any]:
    SR = _carregar_reasoner()
    sr = SR()
    rng = random.Random(semente)
    res: Dict[str, Any] = {}
    descartados: List[str] = []

    for nome, gerar in PADROES:
        ok_n = tot = 0
        fab = tot_c = 0
        for i in range(n):
            it = gerar(rng, controle=False)
            valido, motivo = crivar(it)
            if not valido:
                descartados.append(f"{nome}: {motivo}")
            else:
                tot += 1
                al = alegacoes(sr.reason(it["conceitos"], it["kb"]))
                acertou = any(_menciona(x, it["alvo"]) for x in al)
                ok_n += int(acertou)
                if verboso and i == 0:
                    print(f"  [{nome}] alvo {it['alvo']} → {'ACERTOU' if acertou else 'falhou'}")
                    for x in al[:5]:
                        print(f"      {x[:92]}")

            ic = gerar(rng, controle=True)
            valido_c, motivo_c = crivar(ic)
            if not valido_c:
                descartados.append(f"{nome}/controle: {motivo_c}")
                continue
            tot_c += 1
            al_c = alegacoes(sr.reason(ic["conceitos"], ic["kb"]))
            if any(_menciona(x, ic["alvo"]) for x in al_c):
                fab += 1

        res[nome] = {"acerto": ok_n / max(1, tot), "n": tot,
                     "fabricacao": fab / max(1, tot_c), "n_controle": tot_c}

    # P6: aqui acertar é NÃO AFIRMAR — e afirmar não é a mesma coisa que
    # mencionar. A primeira versão desta contagem marcou 100% de violação
    # contra um motor que estava fazendo a coisa certa: ele emitia
    #     "Defeated: x964 ⇝ prop530 does not hold"
    # e o teste, que só procurava os dois tokens na mesma linha, leu a
    # RETRATAÇÃO como se fosse a afirmação. Um arnês que não distingue
    # "P" de "P não vale" não mede raciocínio derrotável, mede ortografia.
    viola = 0
    for _ in range(n):
        it = p6_excecao(rng)
        al = alegacoes(sr.reason(it["conceitos"], it["kb"]))
        afirma = any(_menciona(x, it["proibido"]) and not _negativa(x) for x in al)
        retrata = any(_menciona(x, it["proibido"]) and _negativa(x) for x in al)
        if afirma and not retrata:
            viola += 1
    res["P6 exceção"] = {"violacao": viola / max(1, n), "n": n}

    # P7 — retido: acertar exige inferência de aridade 2
    ok7 = 0
    for _ in range(n):
        it = p7_conjuncao(rng)
        al = alegacoes(sr.reason(it["conceitos"], it["kb"]))
        if any(_menciona(x, it["alvo"]) and not _negativa(x) for x in al):
            ok7 += 1
    res["P7 conjunção"] = {"acerto": ok7 / max(1, n), "n": n}

    # P8 — retido: acertar é NÃO compor por cima de uma relação não-transitiva
    viola8 = 0
    for _ in range(n):
        it = p8_tipo_de_relacao(rng)
        al = alegacoes(sr.reason(it["conceitos"], it["kb"]))
        if any(_menciona(x, it["proibido"]) and not _negativa(x) for x in al):
            viola8 += 1
    res["P8 tipo de relação"] = {"violacao": viola8 / max(1, n), "n": n}
    res["_descartados"] = descartados
    return res


def relatorio(n: int = 30, sementes: Tuple[int, ...] = (0, 1, 2),
              verboso: bool = False) -> Dict[str, Any]:
    linhas = [avaliar(n, s, verboso=(verboso and s == sementes[0])) for s in sementes]

    print("═" * 78)
    print("BANCADA DE GENERALIZAÇÃO COMPOSICIONAL")
    print("A composição implementada é capacidade, ou é uma regra para um formato?")
    print(f"{n} itens + {n} controles por padrão × {len(sementes)} sementes")
    print("═" * 78)
    print(f"{'padrão':<24}{'implementado?':>15}{'acerto':>10}{'fabricação':>13}")

    agreg = {}
    for nome, _ in PADROES:
        ac = sum(l[nome]["acerto"] for l in linhas) / len(linhas)
        fb = sum(l[nome]["fabricacao"] for l in linhas) / len(linhas)
        agreg[nome] = ac
        marca = "sim (referência)" if nome.startswith("P1") else "não"
        print(f"{nome:<24}{marca:>15}{ac:>9.0%}{fb:>12.0%}")

    v6 = sum(l["P6 exceção"]["violacao"] for l in linhas) / len(linhas)
    a7 = sum(l["P7 conjunção"]["acerto"] for l in linhas) / len(linhas)
    v8 = sum(l["P8 tipo de relação"]["violacao"] for l in linhas) / len(linhas)
    print("─" * 78)
    print("RETIDOS — não são caminhos no grafo; é aqui que a fronteira aparece")
    print(f"{'P6 exceção (bloqueio)':<24}{'—':>15}{'':>9}{v6:>12.0%}  ← violação; ideal 0%")
    print(f"{'P7 conjunção (aridade 2)':<24}{'não':>15}{a7:>9.0%}{'':>12}  ← acerto; ideal 100%")
    print(f"{'P8 tipo de relação':<24}{'—':>15}{'':>9}{v8:>12.0%}  ← violação; ideal 0%")

    desc = [d for l in linhas for d in l["_descartados"]]
    total = len(PADROES) * n * len(sementes) * 2
    print(f"\nitens descartados pelo crivo: {len(desc)}/{total}")
    for d in sorted(set(desc))[:4]:
        print(f"  · {d}")

    naoimpl = [k for k in agreg if not k.startswith("P1")]
    transferiu = [k for k in naoimpl if agreg[k] >= 0.9]
    p5 = agreg.get("P5 chain∘attr", 0.0)

    print("─" * 78)
    if p5 < 0.9:
        print("ARNÊS INVÁLIDO. P5 é a MESMA composição de P1 num campo diferente")
        print("(`causal_chains` em vez de `causal_edges`). Se nem ela transfere, a")
        print("bancada está medindo formato de entrada, não capacidade, e nenhuma")
        print("das outras notas pode ser lida.")
    elif not transferiu:
        print("REGRA CODIFICADA, NÃO CAPACIDADE. O que foi implementado resolve o")
        print("padrão para o qual foi escrito e nenhum outro. Os 100% da bancada de")
        print("emergência são reais e são estreitos: mudou o formato da composição,")
        print("a nota vai a zero. Isso não é generalização — é uma regra a mais.")
    elif len(transferiu) < len(naoimpl):
        print(f"TRANSFERÊNCIA PARCIAL: {len(transferiu)}/{len(naoimpl)} padrões não")
        print("implementados foram resolvidos. Vale olhar o que os que transferiram")
        print("têm em comum — é aí que está o mecanismo real, e não no que eu escrevi")
        print("achando que estava escrevendo.")
    else:
        print("P1–P5: TRANSFERÊNCIA TOTAL — um mecanismo, cinco formatos de composição.")
        print("Mas isto deixou de medir generalização no instante em que o fecho")
        print("unificado foi implementado: agora testa se a implementação bate com o")
        print("oráculo, e bate por construção. Quem ainda audita são P7 e P8.")

    print()
    print("A FRONTEIRA, em uma frase:")
    if a7 < 0.5 and v8 > 0.5:
        print("  o sistema compõe QUALQUER caminho num grafo de relações, e só isso.")
        print("  Não faz inferência de aridade 2 (P7): 'X e Y juntos produzem Z' não é")
        print("  um caminho, e quase toda regra física real tem essa forma.")
        print("  E compõe por cima de relações que não compõem (P8): unificar os arcos")
        print("  fez P2–P4 funcionarem e jogou fora o tipo da aresta, então 'A causa B'")
        print("  mais 'B correlaciona com C' vira 'A ⇝ C'. Isso é o erro elementar da")
        print("  inferência causal, agora estrutural.")
    elif a7 < 0.5:
        print("  compõe caminhos; não faz inferência de aridade 2 (P7).")
    elif v8 > 0.5:
        print("  compõe caminhos, inclusive por cima de relações não-transitivas (P8).")
    else:
        print("  P7 e P8 passaram. Vale desconfiar antes de comemorar: verifique se os")
        print("  itens ainda são retidos ou se o mecanismo passou a cobri-los.")
    return {"agregado": agreg, "p6_violacao": v6, "p7_acerto": a7,
            "p8_violacao": v8, "descartados": desc}


if __name__ == "__main__":
    relatorio(verboso=True)
