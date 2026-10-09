"""
novaphys.emergencia — bancada de conclusão inédita
===================================================

A bancada de dedução (`deducao.py`) mede se o sistema deriva o que se segue
de premissas. Esta mede coisa diferente e mais difícil:

    o sistema chega a uma conclusão que NÃO está na base e que NENHUM
    mecanismo isolado produz — só a composição de dois mecanismos distintos?

Essa é a separação entre um mecanismo de recuperação-com-dedução e um
sistema que constrói explicação nova a partir do que tem.

O PROBLEMA DE MEDIR ISSO, E COMO ESTA BANCADA RESOLVE
------------------------------------------------------
"Emergente" é fácil de alegar e difícil de provar. Nesta mesma sessão a
bancada de dedução mediu 100% num item que, olhado de perto, entregava a
composição já feita numa lista — o sistema só ecoava. A nota era real; o
que ela media, não.

Então aqui a propriedade "emergente" não é afirmada, é VERIFICADA item a
item, antes de o sistema ver qualquer coisa. Todo item passa por três
solucionadores programáticos:

    so_fecho     aplica só fecho transitivo sobre as arestas
    so_regras    aplica só as regras de atributo
    composto     aplica os dois, juntando o resultado de um no outro

Um item só entra na bancada se `so_fecho` FALHA, `so_regras` FALHA e
`composto` ACERTA. Itens que não passam nesse crivo são descartados e
contados — se muitos caírem, o gerador está produzindo recuperação
disfarçada e o número sai marcado como suspeito.

Isso torna a alegação verificável: quando o sistema acerta um item desta
bancada, está provado que nem o fecho nem as regras sozinhos bastavam.

AS TRÊS FAMÍLIAS
-----------------
1. HERANÇA        A→B→C na base, e uma regra diz que C tem a propriedade P.
                  Inédito: A leva a P. O motor já produz "Transitive: A → C"
                  e "Common attribute: C = P" em linhas separadas — e nunca
                  as junta. É o caso mais limpo: os dois ingredientes estão
                  na saída, o produto não.

2. PONTE          A→B e C→D, desconexos, mais o fato "B equivale a C" — que
                  NÃO é uma aresta causal (se fosse, o fecho sozinho pegava,
                  e aí não seria emergência, seria multi-salto). Inédito: A→D.
                  Exige substituição por identidade antes do fecho.

3. ANALOGIA       Domínio-fonte s1→s2→s3. Domínio-alvo t1→t2 e nós soltos.
                  O mapeamento dado cobre só (s1,t1) e (s2,t2). O par (s3,·)
                  tem de ser encontrado pelo tipo. Inédito: t2→t3.
                  Transferência estrutural, não dedução.

CONTROLE NEGATIVO
------------------
Cada família tem um gêmeo em que o ingrediente-ponte foi removido: a regra
fala de um nó fora da cadeia, a identidade aponta para um beco sem saída,
nenhum nó solto tem o tipo certo. Nesses itens a resposta certa é NÃO
CONCLUIR NADA. Sem eles, um sistema que sempre "conclui" alguma coisa
marcaria alto nas três famílias — que foi exatamente o que a prova do
controle vazio impediu na bancada de dedução.

TETO DE MECANISMO ÚNICO
------------------------
O relatório imprime, ao lado da nota do sistema, a nota dos dois
solucionadores isolados. Por construção elas são 0%. Se a nota do sistema
também for 0%, ele está no teto do mecanismo único: deduz e recupera, não
compõe.
"""

from __future__ import annotations

import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

_AQUI = os.path.dirname(os.path.abspath(__file__))
_NOVAMIND = os.path.join(os.path.dirname(_AQUI), "legacy", "novamind")


def _carregar_reasoner():
    if _NOVAMIND not in sys.path:
        sys.path.insert(0, _NOVAMIND)
    from core.reasoning import SymbolicReasoner  # noqa: E402
    return SymbolicReasoner


# ══════════════════════════════════════════════ vocabulário
# Tokens com prefixo e sufixo numérico próprios: "no3" nunca é subcadeia de
# "no31" porque a checagem é por fronteira de palavra, mas o prefixo evita
# também colisão acidental com palavras do texto gerado pelo motor.
def _tok(rng: random.Random, prefixo: str, n: int) -> List[str]:
    base = rng.sample(range(100, 999), n)
    return [f"{prefixo}{v}" for v in base]


TIPOS = ["catalisador", "barreira", "reservatorio", "gatilho", "amortecedor"]


def _regra_atributo(attr: str, valor: str, freq: float = 0.9) -> Dict:
    return {"content": {"type": "attribute",
                        "abstraction": {"attribute": attr, "value": valor,
                                        "frequency": freq}}}


def _E(s: str, t: str, tipo: str = "leva_a") -> Dict:
    return {"source": s, "type": tipo, "target": t}


# ══════════════════════════════════════════════ família 1 — herança
def item_heranca(rng: random.Random, controle: bool = False) -> Dict[str, Any]:
    """
    A→B→C  +  "C tem a propriedade P"   ⟹   A leva a P.

    No controle a regra fala de um nó Z que não está na cadeia: nada liga A a P.
    """
    a, b, c, z = _tok(rng, "h", 4)
    p = f"prop{rng.randint(100, 999)}"
    dono = z if controle else c
    kb = {
        "concepts": [{"label": a, "position": [0.1] * 8, "connections": {"leva_a": [b]}}],
        "causal_edges": [_E(a, b), _E(b, c)],
        "causal_chains": [],
        "semantic_rules": [_regra_atributo(dono, p)],
    }
    return {"familia": "heranca", "controle": controle,
            "conceitos": kb["concepts"], "kb": kb,
            "alvo": None if controle else (a, p),
            "proibido": (a, p) if controle else None}


# ══════════════════════════════════════════════ família 2 — ponte de identidade
def item_ponte(rng: random.Random, controle: bool = False) -> Dict[str, Any]:
    """
    A→B, C→D desconexos, mais "B equivale a C"  ⟹  A→D.

    A identidade vai em `identities` e como regra de tipo `equivalence` —
    legível, nomeada, presente. O que ela NÃO é: uma aresta causal. Se fosse,
    o fecho transitivo sozinho resolveria e o item viraria multi-salto comum.

    No controle a identidade aponta para um nó sem saída: A→D não se segue.
    """
    a, b, c, d, morto = _tok(rng, "p", 5)
    par_direito = morto if controle else c
    kb = {
        "concepts": [{"label": a, "position": [0.1] * 8, "connections": {"leva_a": [b]}}],
        "causal_edges": [_E(a, b), _E(c, d)],
        "causal_chains": [],
        "identities": [{"left": b, "right": par_direito}],
        "semantic_rules": [{"content": {"type": "equivalence",
                                        "abstraction": {"left": b, "right": par_direito,
                                                        "frequency": 1.0}}}],
    }
    return {"familia": "ponte", "controle": controle,
            "conceitos": kb["concepts"], "kb": kb,
            "alvo": None if controle else (a, d),
            "proibido": (a, d) if controle else None}


# ══════════════════════════════════════════════ família 3 — analogia estrutural
def item_analogia(rng: random.Random, controle: bool = False) -> Dict[str, Any]:
    """
    Fonte s1→s2→s3, alvo t1→t2 e nós soltos. O mapeamento dado cobre (s1,t1)
    e (s2,t2) — o par de s3 falta e tem de sair do TIPO: entre os nós soltos
    do alvo, exatamente um tem o mesmo tipo de s3.

    Inédito: t2→t3, uma aresta que não existe em domínio nenhum.

    No controle nenhum nó solto tem o tipo de s3: a transferência não fecha
    e a resposta certa é não afirmar aresta nova.
    """
    s1, s2, s3 = _tok(rng, "s", 3)
    t1, t2, t3, t4, t5 = _tok(rng, "t", 5)
    tipo_alvo, *outros = rng.sample(TIPOS, 3)
    tipos = {s1: outros[0], s2: outros[1], s3: tipo_alvo,
             t1: outros[0], t2: outros[1],
             t3: (outros[0] if controle else tipo_alvo),
             t4: outros[1], t5: outros[1]}
    kb = {
        "concepts": [{"label": t2, "position": [0.1] * 8, "connections": {"leva_a": []}}],
        "causal_edges": [_E(s1, s2), _E(s2, s3), _E(t1, t2)],
        "causal_chains": [],
        "semantic_rules": [],
        "tipos": tipos,
        "mapeamento": [{"fonte": s1, "alvo": t1}, {"fonte": s2, "alvo": t2}],
        "nos_soltos": [t3, t4, t5],
    }
    return {"familia": "analogia", "controle": controle,
            "conceitos": kb["concepts"], "kb": kb,
            "alvo": None if controle else (t2, t3),
            "proibido": (t2, t3) if controle else None}


# ══════════════════════════════════════════════ solucionadores de referência
def _arestas(kb: Dict) -> Set[Tuple[str, str]]:
    ar = {(e["source"], e["target"]) for e in kb.get("causal_edges", [])}
    for ch in kb.get("causal_chains", []):
        for i in range(len(ch) - 1):
            ar.add((ch[i], ch[i + 1]))
    return ar


def _fecho(arestas: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    alcanca = set(arestas)
    mudou = True
    while mudou:
        mudou = False
        for (a, b) in list(alcanca):
            for (c, d) in list(alcanca):
                if b == c and a != d and (a, d) not in alcanca:
                    alcanca.add((a, d)); mudou = True
    return alcanca


def so_fecho(item: Dict) -> List[Tuple[str, str]]:
    """Mecanismo 1 isolado: fecho transitivo, nada mais."""
    ar = _arestas(item["kb"])
    return sorted(_fecho(ar) - ar)


def so_regras(item: Dict) -> List[Tuple[str, str]]:
    """Mecanismo 2 isolado: regras de atributo, nada mais."""
    out = []
    for r in item["kb"].get("semantic_rules", []):
        ab = r.get("content", {}).get("abstraction", {})
        if r.get("content", {}).get("type") == "attribute":
            out.append((ab.get("attribute", ""), ab.get("value", "")))
    return out


def composto(item: Dict) -> List[Tuple[str, str]]:
    """
    O oráculo: o que sai de JUNTAR os mecanismos. É o alvo da bancada e a
    prova de que o item é solúvel — se nem ele acerta, o item é impossível
    e o crivo o descarta.
    """
    kb = item["kb"]
    fam = item["familia"]
    if fam == "heranca":
        ar = _arestas(kb)
        alcanca = _fecho(ar)
        props = {(a["content"]["abstraction"]["attribute"],
                  a["content"]["abstraction"]["value"])
                 for a in kb.get("semantic_rules", [])
                 if a["content"]["type"] == "attribute"}
        return sorted({(a, p) for (a, b) in alcanca for (dono, p) in props if b == dono})

    if fam == "ponte":
        # substitui pela identidade, depois fecha
        subst = {}
        for idt in kb.get("identities", []):
            subst[idt["right"]] = idt["left"]
        ar = {(subst.get(s, s), subst.get(t, t)) for (s, t) in _arestas(kb)}
        return sorted(_fecho(ar) - _arestas(kb))

    if fam == "analogia":
        tipos = kb.get("tipos", {})
        mapa = {m["fonte"]: m["alvo"] for m in kb.get("mapeamento", [])}
        ar = _arestas(kb)
        novas = []
        for (s, t) in ar:
            if s in mapa and t not in mapa:
                # t é o nó-fonte sem par; procura no alvo um nó solto do mesmo tipo
                cands = [n for n in kb.get("nos_soltos", []) if tipos.get(n) == tipos.get(t)]
                if len(cands) == 1:
                    novas.append((mapa[s], cands[0]))
        return sorted(novas)

    return []


# ══════════════════════════════════════════════ crivo de emergência
def crivar(item: Dict) -> Tuple[bool, str]:
    """
    Um item vale se e somente se:
        o alvo NÃO sai de `so_fecho`, NÃO sai de `so_regras`, e SAI de `composto`.

    Nos controles a exigência é a simétrica: o proibido não sai de NENHUM
    dos três — se o oráculo composto o produzisse, o controle não seria
    controle.
    """
    if item["controle"]:
        proib = item["proibido"]
        for nome, fn in (("fecho", so_fecho), ("regras", so_regras), ("composto", composto)):
            if proib in fn(item):
                return False, f"controle inválido: {nome} produz o par proibido"
        return True, "ok"

    alvo = item["alvo"]
    if alvo in so_fecho(item):
        return False, "não é emergente: o fecho sozinho já produz o alvo"
    if alvo in so_regras(item):
        return False, "não é emergente: as regras sozinhas já produzem o alvo"
    if alvo not in composto(item):
        return False, "insolúvel: nem o oráculo composto produz o alvo"
    return True, "ok"


# ══════════════════════════════════════════════ leitura das alegações
_RE_INFER = re.compile(r"Inferred:\s*(.+?)\s*\(from rule", re.UNICODE)


def alegacoes(resultado) -> List[str]:
    """
    Uma alegação por derivação, mais a conclusão final.

    A pontuação é POR ALEGAÇÃO: a conclusão inédita tem de aparecer inteira
    numa linha só. Somar menções espalhadas por linhas diferentes daria o
    ponto a um sistema que produziu "A → C" numa linha e "C = P" noutra —
    que é precisamente o que ele já faz e precisamente o que esta bancada
    existe para não confundir com composição.
    """
    out = []
    for linha in getattr(resultado, "trace", []):
        m = _RE_INFER.search(str(linha))
        if m:
            out.append(m.group(1).strip())
    conc = getattr(resultado, "conclusion", "") or ""
    if conc:
        out.append(conc)
    return out


def _menciona(texto: str, par: Tuple[str, str]) -> bool:
    t = texto.lower()
    return all(re.search(rf"\b{re.escape(x.lower())}\b", t) for x in par)


# ══════════════════════════════════════════════ avaliação
FAMILIAS = {"heranca": item_heranca, "ponte": item_ponte, "analogia": item_analogia}


def avaliar(n_por_familia: int = 30, semente: int = 0,
            verboso: bool = False) -> Dict[str, Any]:
    SR = _carregar_reasoner()
    sr = SR()
    rng = random.Random(semente)
    res: Dict[str, Any] = {}
    descartados: List[str] = []

    for nome, gerar in FAMILIAS.items():
        acertos = tot = 0
        fabricou = tot_c = 0
        for i in range(n_por_familia):
            it = gerar(rng, controle=False)
            ok, motivo = crivar(it)
            if not ok:
                descartados.append(f"{nome}: {motivo}")
            else:
                tot += 1
                al = alegacoes(sr.reason(it["conceitos"], it["kb"]))
                acertou = any(_menciona(a, it["alvo"]) for a in al)
                acertos += int(acertou)
                if verboso and i == 0:
                    print(f"  [{nome}] alvo inédito {it['alvo']}")
                    for a in al[:6]:
                        print(f"      alegou: {a[:96]}")

            ic = gerar(rng, controle=True)
            ok_c, motivo_c = crivar(ic)
            if not ok_c:
                descartados.append(f"{nome}/controle: {motivo_c}")
                continue
            tot_c += 1
            al_c = alegacoes(sr.reason(ic["conceitos"], ic["kb"]))
            if any(_menciona(a, ic["proibido"]) for a in al_c):
                fabricou += 1

        res[nome] = {
            "acerto": acertos / max(1, tot), "n": tot,
            "fabricacao": fabricou / max(1, tot_c), "n_controle": tot_c,
        }

    res["_descartados"] = descartados
    return res


def relatorio(n_por_familia: int = 30, sementes: Tuple[int, ...] = (0, 1, 2),
              verboso: bool = False) -> Dict[str, Any]:
    linhas = [avaliar(n_por_familia, s, verboso=(verboso and s == sementes[0]))
              for s in sementes]

    print("═" * 78)
    print("BANCADA DE EMERGÊNCIA — conclusão inédita por composição de mecanismos")
    print(f"{n_por_familia} itens + {n_por_familia} controles por família "
          f"× {len(sementes)} sementes")
    print("═" * 78)
    print(f"{'família':<14}{'só fecho':>10}{'só regras':>11}{'oráculo':>10}"
          f"{'SISTEMA':>10}{'fabricação':>13}")

    agregado = {}
    for fam in FAMILIAS:
        ac = sum(l[fam]["acerto"] for l in linhas) / len(linhas)
        fb = sum(l[fam]["fabricacao"] for l in linhas) / len(linhas)
        agregado[fam] = {"acerto": ac, "fabricacao": fb}
        # "só fecho" e "só regras" são 0% POR CONSTRUÇÃO — o crivo descartou
        # qualquer item em que não fossem. Imprimir mesmo assim é o ponto:
        # é a linha de base contra a qual a nota do sistema significa algo.
        print(f"{fam:<14}{'0%':>10}{'0%':>11}{'100%':>10}"
              f"{ac:>9.0%}{fb:>12.0%}")

    desc = [d for l in linhas for d in l["_descartados"]]
    print("─" * 78)
    total_itens = len(FAMILIAS) * n_por_familia * len(sementes) * 2
    print(f"itens descartados pelo crivo: {len(desc)}/{total_itens}")
    if desc:
        vistos = {}
        for d in desc:
            vistos[d] = vistos.get(d, 0) + 1
        for d, k in sorted(vistos.items(), key=lambda x: -x[1])[:5]:
            print(f"  · {k}× {d}")
        if len(desc) > total_itens * 0.2:
            print("  ! mais de 20% descartado — o gerador está mal calibrado e as")
            print("    notas acima não devem ser lidas como medida de emergência.")

    media = sum(a["acerto"] for a in agregado.values()) / len(agregado)
    fab = sum(a["fabricacao"] for a in agregado.values()) / len(agregado)
    print("─" * 78)
    if media <= 0.0:
        print("TETO DE MECANISMO ÚNICO. O sistema não produziu nenhuma conclusão")
        print("inédita: ele deduz (fecho) e recupera (regras), mas não junta o")
        print("resultado de um mecanismo à entrada do outro. Os dois ingredientes")
        print("aparecem na saída dele em linhas separadas; o produto, nunca.")
    elif fab > 0.1:
        print(f"ATENÇÃO: acerto {media:.0%} mas fabricação {fab:.0%} nos controles.")
        print("Parte do acerto pode ser afirmação indiscriminada, não composição.")
        print("Um sistema que sempre conclui algo acerta por acidente.")
    else:
        print(f"COMPOSIÇÃO MEDIDA: {media:.0%} de conclusões inéditas com "
              f"{fab:.0%} de fabricação.")
        print("Cada acerto é um par que nem o fecho nem as regras produzem sozinhos —")
        print("isso está verificado item a item pelo crivo, não presumido.")
    return {"agregado": agregado, "descartados": desc}


if __name__ == "__main__":
    relatorio(verboso=True)
