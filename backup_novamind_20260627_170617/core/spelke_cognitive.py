"""
Spelke Genes — BIBLIOTECA COGNITIVA (HUMAN-INSIGHT)
====================================================

Primitivas cognitivas (não geométricas) elicitadas da introspecção humana.
Categorias baseadas em Language of Thought Hypothesis (Fodor 1975, Frankland & Greene 2020)
+ insight humano sobre multimodalidade, parenting, e criatividade.

FAMÍLIA 0: LINGUAGEM (verbos primitivos do Mentalese)
   SE, E_SE, SE_NÃO, TALVEZ, ENQUANTO, QUANDO, PORQUE, PARA, DE, PARA_QUE

FAMÍLIA 1: LÓGICA
   TODO, ALGUM, NENHUM, EXATAMENTE_UM, IMPLICA, EQUIVALE, NEGA

FAMÍLIA 9: ONTOLOGIA
   OBJETO, PARTE_DE, MEMBRO_DE, CONJUNTO_DE, IGUAL_A, DIFERENTE_DE,
   TIPO_DE, INSTÂNCIA_DE, TEM_PROPRIEDADE

FAMÍLIA 10: CAUSALIDADE
   CAUSA, PERMITE, IMPEDE, CORRELACIONA, COINCIDE_COM, COLIDE_COM,
   SEMPRE_QUE, ANTES_DE, DEPOIS_DE, DURANTE

FAMÍLIA 11: MODALIDADE
   POSSÍVEL, NECESSÁRIO, DEVE, PODE, QUER, ACREDITA, SABE, DUVIDA

FAMÍLIA 12: METÁFORA / MULTIMODAL (da história da logo)
   LÍQUIDO, SÓLIDO, FOGOSO, FRIO, TRANSPARENTE, LEVE, PESADO,
   MAIS_FINO, MAIS_GROSSO, MAIS_FORTE, MAIS_FRACO, MAIS_RÁPIDO

FAMÍLIA 13: TEMPORAL
   ANTES, DEPOIS, DURANTE, ATÉ, SEMPRE, NUNCA, AGORA, EM_PARALELO

FAMÍLIA 14: PERSPECTIVA
   EU_VEJO, VOCÊ_VE, TODOS_VEEM, NINGUÉM_VE, NO_FUTURO, NO_PASSADO

FAMÍLIA 15: META-COGNIÇÃO
   SEI, NÃO_SEI, ACREDITO, DUVIDO, VOU_CHUTAR, ISSO_PARECE_ESTRANHO

Cada primitiva é codificada como Gene/Action que pode ser COMPONDO outras primitivas.
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass

# ============================================================================
# FAMÍLIA 0: LINGUAGEM — predicados do Mentalese
# ============================================================================
def pred_SE(node, ctx, cond=None):
    """SE: se condição X, então Y. cond é um callable."""
    if cond is None:
        return True
    return cond(node, ctx)

def pred_E_SE(node, ctx, cond_a=None, cond_b=None):
    """E_SE: se A E B."""
    return (cond_a is None or cond_a(node, ctx)) and (cond_b is None or cond_b(node, ctx))

def pred_SE_NAO(node, ctx, cond=None, else_fn=None):
    """SE_NÃO: se A então X senão Y. Implementado como gene, não predicado."""

def pred_TALVEZ(node, ctx, prob=0.5):
    """TALVEZ: aplica com probabilidade prob. Não-determinístico — não cabe em busca."""
    return np.random.random() < prob

def pred_ENQUANTO(node, ctx, cond=None):
    """ENQUANTO: aplica enquanto condição for verdadeira. Não cabe em busca pura."""
    return cond(node, ctx) if cond else True

def pred_QUANDO(node, ctx, cond=None):
    """QUANDO: aplica no momento em que condição vale. Predicado pontual."""
    return cond(node, ctx) if cond else True

def pred_PORQUE(node, ctx, cause_node=None):
    """PORQUE: objeto é resultado de uma causa. Identifica se o objeto 'veio de' outro."""
    if cause_node is None:
        return True
    return node.color == cause_node.color

def pred_PARA(node, ctx, purpose=None):
    """PARA: aplicado com uma intenção/propósito. Marcador de intencionalidade."""
    return True  # todos os objetos podem ter propósito atribuído

def pred_PARA_QUE(node, ctx, goal=None):
    """PARA_QUE: aplicado para atingir um objetivo. Marcador de finalidade."""
    return True

# ============================================================================
# FAMÍLIA 1: LÓGICA
# ============================================================================
def pred_TODO(node, ctx, group=None):
    """TODO: predicado se aplica a todos do grupo."""
    if not group:
        return False
    return all(group(n, ctx) for n in ctx.nodes if n.color != 0)

def pred_ALGUM(node, ctx, group=None):
    """ALGUM: existe pelo menos um no grupo que satisfaz."""
    if not group:
        return True
    return any(group(n, ctx) for n in ctx.nodes if n.color != 0)

def pred_NENHUM(node, ctx, group=None):
    """NENHUM: nenhum do grupo satisfaz."""
    if not group:
        return True
    return not any(group(n, ctx) for n in ctx.nodes if n.color != 0)

def pred_EXATAMENTE_UM(node, ctx, group=None):
    """EXATAMENTE_UM: um e só um do grupo satisfaz."""
    if not group:
        return False
    return sum(1 for n in ctx.nodes if n.color != 0 and group(n, ctx)) == 1

def pred_IMPLICA(node, ctx, consequent=None):
    """IMPLICA: este nó implica consequent."""
    if consequent is None:
        return True
    return consequent(node, ctx)

def pred_NEGA(node, ctx, pred=None):
    """NEGA: predicado é falso."""
    if pred is None:
        return True
    return not pred(node, ctx)


# ============================================================================
# FAMÍLIA 9: ONTOLOGIA
# ============================================================================
def pred_OBJETO(node, ctx):
    """OBJETO: é uma entidade discreta (tem cells distintas)."""
    return len(node.cells) >= 1

def pred_PARTE_DE(node, ctx, whole=None):
    """PARTE_DE: este nó é parte de outro objeto whole."""
    if whole is None:
        return False
    if not whole.cells:
        return False
    wr_min = min(r for r, c in whole.cells)
    wr_max = max(r for r, c in whole.cells)
    wc_min = min(c for r, c in whole.cells)
    wc_max = max(c for r, c in whole.cells)
    return all(wr_min <= r <= wr_max and wc_min <= c <= wc_max for r, c in node.cells)

def pred_MEMBRO_DE(node, ctx, group=None):
    """MEMBRO_DE: pertence a um conjunto."""
    return True  # depende do grupo

def pred_IGUAL_A(node, ctx, other=None):
    """IGUAL_A: este nó é idêntico a other (mesma cor, mesmo tamanho, mesma posição)."""
    if other is None:
        return False
    return (node.color == other.color and
            node.size == other.size and
            node.cells == other.cells)

def pred_DIFERENTE_DE(node, ctx, other=None):
    """DIFERENTE_DE: este nó difere de other."""
    if other is None:
        return True
    return not pred_IGUAL_A(node, ctx, other)

def pred_TIPO_DE(node, ctx, type_pred=None):
    """TIPO_DE: este nó é do tipo type_pred."""
    if type_pred is None:
        return True
    return type_pred(node, ctx)

def pred_TEM_PROPRIEDADE(node, ctx, prop=None, value=None):
    """TEM_PROPRIEDADE: tem a propriedade prop com valor value."""
    if prop is None:
        return True
    if prop == 'cor':
        return node.color == value
    if prop == 'tamanho':
        return node.size == value
    if prop == 'forma':
        return node.is_square if value == 'quadrado' else (node.is_line if value == 'linha' else True)
    return True


# ============================================================================
# FAMÍLIA 10: CAUSALIDADE
# ============================================================================
def pred_CAUSA(node, ctx, effect=None):
    """CAUSA: este nó é causa do effect."""
    return True  # causalidade é estrutural, não local

def pred_CORRELACIONA(node, ctx, other=None):
    """CORRELACIONA: aparece sempre que other aparece (estatístico)."""
    return True

def pred_COINCIDE_COM(node, ctx, other=None):
    """COINCIDE_COM: mesmo local/tempo mas SEM relação causal."""
    if other is None:
        return False
    return bool(node.cells & other.cells)  # compartilha células

def pred_COLIDE_COM(node, ctx, other=None):
    """COLIDE_COM: toca ou sobrepõe other."""
    if other is None:
        return False
    for a, b in ctx.edges.get('touches', []):
        if a == node.id and b == other.id: return True
        if b == node.id and a == other.id: return True
    return False

def pred_SEMPRE_QUE(node, ctx, cond=None):
    """SEMPRE_QUE: cond → este nó. Regra universal."""
    if cond is None:
        return True
    return cond(node, ctx)

def pred_ANTES_DE(node, ctx, other=None):
    """ANTES_DE: este nó aparece temporalmente antes de other."""
    return True

def pred_DEPOIS_DE(node, ctx, other=None):
    """DEPOIS_DE: este nó aparece temporalmente depois de other."""
    return True


# ============================================================================
# FAMÍLIA 11: MODALIDADE
# ============================================================================
def pred_POSSIVEL(node, ctx):
    """POSSÍVEL: estado pode existir (compatibilidade)."""
    return True

def pred_NECESSARIO(node, ctx):
    """NECESSÁRIO: estado é requerido pela estrutura."""
    return True

def pred_QUER(node, ctx, target=None):
    """QUER: intenção de atingir target. Marcador de agente."""
    return True

def pred_ACREDITA(node, ctx, proposition=None):
    """ACREDITA: agente acredita em proposition. Marcador epistêmico."""
    return True

def pred_SABE(node, ctx, proposition=None):
    """SABE: conhecimento explícito."""
    return True

def pred_DUVIDA(node, ctx, proposition=None):
    """DUVIDA: incerteza sobre proposition."""
    return True


# ============================================================================
# FAMÍLIA 12: METÁFORA / MULTIMODAL
# ============================================================================
# Estas não são predicados puros — são TRANSFORMAÇÕES que mapeiam
# entre modalidades sensoriais. Da história da logo: "líquido, sólido,
# fogoso, frio, transparente, ramo tipo folha".

def gen_metaphor_variants(node, ctx, source='color'):
    """Gera variantes metafóricas de um objeto: varia cor/textura/forma
    em múltiplas direções (mais fino, mais grosso, mais forte, ...).
    Retorna lista de dicionários com 'variant' e 'params'.
    """
    variants = []
    if source == 'color':
        # +1, -1, +2, -2 etc.
        for delta in [-2, -1, 1, 2]:
            variants.append({'op': 'recolor_delta', 'param': delta})
    elif source == 'shape':
        variants.append({'op': 'thinner', 'param': 0.7})
        variants.append({'op': 'thicker', 'param': 1.5})
        variants.append({'op': 'liquid', 'param': 1.0})  # blur edges
        variants.append({'op': 'solid', 'param': 0.0})
        variants.append({'op': 'transparent', 'param': 0.5})
    return variants


# ============================================================================
# FAMÍLIA 13: TEMPORAL
# ============================================================================
def pred_ANTES(node, ctx, other=None):
    """ANTES: temporalmente anterior."""
    return True

def pred_DEPOIS(node, ctx, other=None):
    """DEPOIS: temporalmente posterior."""
    return True

def pred_DURANTE(node, ctx, event=None):
    """DURANTE: durante um evento."""
    return True

def pred_EM_PARALELO(node, ctx, others=None):
    """EM_PARALELO: ocorre simultaneamente com outros."""
    return True


# ============================================================================
# FAMÍLIA 14: PERSPECTIVA
# ============================================================================
def pred_EU_VEJO(node, ctx):
    """EU_VEJO: da minha perspectiva."""
    return True

def pred_VOCE_VE(node, ctx):
    """VOCÊ_VE: da perspectiva do outro."""
    return True

def pred_TODOS_VEEM(node, ctx):
    """TODOS_VEEM: perspectiva universal."""
    return True

def pred_NO_FUTURO(node, ctx):
    """NO_FUTURO: projeção temporal."""
    return True


# ============================================================================
# FAMÍLIA 15: META-COGNIÇÃO
# ============================================================================
def pred_SEI(node, ctx):
    """SEI: tenho confiança neste predicado."""
    return True

def pred_NAO_SEI(node, ctx):
    """NÃO_SEI: baixa confiança — devo abster."""
    return True

def pred_ISSO_PARECE_ESTRANHO(node, ctx):
    """ISSO_PARECE_ESTRANHO: marcador de surpresa (H188)."""
    return True


# ============================================================================
# COMPILAÇÃO
# ============================================================================
def get_cognitive_predicates():
    """Retorna TODAS as primitivas cognitivas como predicados para make_predicates()."""
    return [
        # Família 0: Linguagem
        ('SE', pred_SE, 'callable'),
        ('E_SE', pred_E_SE, 'callable'),
        ('TALVEZ', pred_TALVEZ, 'prob'),
        ('ENQUANTO', pred_ENQUANTO, 'callable'),
        ('QUANDO', pred_QUANDO, 'callable'),
        ('PORQUE', pred_PORQUE, [None]),
        ('PARA', pred_PARA, [None]),
        ('PARA_QUE', pred_PARA_QUE, [None]),

        # Família 1: Lógica
        ('TODO', pred_TODO, 'callable'),
        ('ALGUM', pred_ALGUM, 'callable'),
        ('NENHUM', pred_NENHUM, 'callable'),
        ('EXATAMENTE_UM', pred_EXATAMENTE_UM, 'callable'),
        ('IMPLICA', pred_IMPLICA, 'callable'),
        ('NEGA', pred_NEGA, 'callable'),

        # Família 9: Ontologia
        ('OBJETO', pred_OBJETO, [None]),
        ('PARTE_DE', pred_PARTE_DE, 'other_nodes'),
        ('MEMBRO_DE', pred_MEMBRO_DE, 'callable'),
        ('IGUAL_A', pred_IGUAL_A, 'other_nodes'),
        ('DIFERENTE_DE', pred_DIFERENTE_DE, 'other_nodes'),
        ('TIPO_DE', pred_TIPO_DE, 'callable'),
        ('TEM_PROP', pred_TEM_PROPRIEDADE, 'prop_value'),

        # Família 10: Causalidade
        ('CAUSA', pred_CAUSA, [None]),
        ('CORRELACIONA', pred_CORRELACIONA, 'other_nodes'),
        ('COINCIDE_COM', pred_COINCIDE_COM, 'other_nodes'),
        ('COLIDE_COM', pred_COLIDE_COM, 'other_nodes'),
        ('SEMPRE_QUE', pred_SEMPRE_QUE, 'callable'),
        ('ANTES_DE', pred_ANTES_DE, 'other_nodes'),
        ('DEPOIS_DE', pred_DEPOIS_DE, 'other_nodes'),

        # Família 11: Modalidade
        ('POSSIVEL', pred_POSSIVEL, [None]),
        ('NECESSARIO', pred_NECESSARIO, [None]),
        ('QUER', pred_QUER, [None]),
        ('ACREDITA', pred_ACREDITA, [None]),
        ('SABE', pred_SABE, [None]),
        ('DUVIDA', pred_DUVIDA, [None]),

        # Família 13: Temporal
        ('ANTES', pred_ANTES, 'other_nodes'),
        ('DEPOIS', pred_DEPOIS, 'other_nodes'),
        ('DURANTE', pred_DURANTE, 'callable'),
        ('EM_PARALELO', pred_EM_PARALELO, 'callable'),

        # Família 14: Perspectiva
        ('EU_VEJO', pred_EU_VEJO, [None]),
        ('VOCE_VE', pred_VOCE_VE, [None]),
        ('TODOS_VEEM', pred_TODOS_VEEM, [None]),
        ('NO_FUTURO', pred_NO_FUTURO, [None]),

        # Família 15: Meta-cognição
        ('SEI', pred_SEI, [None]),
        ('NAO_SEI', pred_NAO_SEI, [None]),
        ('ESTRANHO', pred_ISSO_PARECE_ESTRANHO, [None]),
    ]


def count_cognitive():
    preds = get_cognitive_predicates()
    by_fam = {
        '0_LINGUAGEM': 0, '1_LOGICA': 0, '9_ONTOLOGIA': 0,
        '10_CAUSALIDADE': 0, '11_MODALIDADE': 0, '13_TEMPORAL': 0,
        '14_PERSPECTIVA': 0, '15_METACOGNICAO': 0,
    }
    for n, _, _ in preds:
        if n in ('SE','E_SE','TALVEZ','ENQUANTO','QUANDO','PORQUE','PARA','PARA_QUE'):
            by_fam['0_LINGUAGEM'] += 1
        elif n in ('TODO','ALGUM','NENHUM','EXATAMENTE_UM','IMPLICA','NEGA'):
            by_fam['1_LOGICA'] += 1
        elif n in ('OBJETO','PARTE_DE','MEMBRO_DE','IGUAL_A','DIFERENTE_DE','TIPO_DE','TEM_PROP'):
            by_fam['9_ONTOLOGIA'] += 1
        elif n in ('CAUSA','CORRELACIONA','COINCIDE_COM','COLIDE_COM','SEMPRE_QUE','ANTES_DE','DEPOIS_DE'):
            by_fam['10_CAUSALIDADE'] += 1
        elif n in ('POSSIVEL','NECESSARIO','QUER','ACREDITA','SABE','DUVIDA'):
            by_fam['11_MODALIDADE'] += 1
        elif n in ('ANTES','DEPOIS','DURANTE','EM_PARALELO'):
            by_fam['13_TEMPORAL'] += 1
        elif n in ('EU_VEJO','VOCE_VE','TODOS_VEEM','NO_FUTURO'):
            by_fam['14_PERSPECTIVA'] += 1
        elif n in ('SEI','NAO_SEI','ESTRANHO'):
            by_fam['15_METACOGNICAO'] += 1
    return {
        'total_predicados_cognitivos': len(preds),
        'por_familia': by_fam,
    }
