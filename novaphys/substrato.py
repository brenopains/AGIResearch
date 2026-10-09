"""
novaphys.substrato — o degrau seguinte: representação a partir de fluxo cru
============================================================================

A trajetória das bancadas, e o que cada uma deixou em pé:

    operadores.py   escolher entre operadores dados            resolvido
    hipoteses.py    compor UM operador recursivamente          resolvido
    abstracao.py    a primitiva emerge de peças genéricas      resolvido
    ESTA            as peças emergem de PERCEPÇÃO?             aqui

E o regresso desce mais um andar, sem sumir:

    antes de abstracao.py   eu escolhi o operador (PARTIR)
    depois                  eu escolhi as primitivas (SE, MENOR, VAR, NUM)
    aqui                    eu escolho os ÁTOMOS DE SINAL (ler canal, atraso,
                            diferença, média de janela, comparação)

Isso é progresso e é honesto chamar de progresso: os átomos são operações de
sinal, agnósticas a tarefa e a domínio — nenhum deles menciona regime, causa,
limiar ou objeto. Mas não é "representação do nada", e quem disser que é está
vendendo o degrau como se fosse a escada.

O QUE MUDA EM RELAÇÃO A `abstracao.py`
---------------------------------------
Lá o contexto vinha em campos que EU nomeei — `x`, `z`, `w` — e cada exemplo
era independente. Aqui não há nome nenhum: o mundo é uma matriz de números,
`T` instantes × `K` canais, e o tempo é contínuo. O sistema tem de construir
sozinho o que vai olhar: qual canal, com que atraso, comparado a quê.

O fluxo importa. Numa lista de exemplos independentes não existe "a previsão
falhou e por isso eu reparei"; num fluxo, existe.

AS QUATRO MEDIDAS, SEPARADAS DE PROPÓSITO
------------------------------------------
  1. FORMAÇÃO      programas sobre o fluxo cru resolvem as tarefas?
  2. MINERAÇÃO     o que se repete vira primitiva, por compressão?
  3. TRANSFERÊNCIA a primitiva de uns canais serve em canais nunca vistos?
  4. ABERTURA      diante de estrutura que a biblioteca não cobre, o sistema
                   CONSTRÓI algo novo, ou força a primitiva e erra?

A quarta é a que separa "tenho uma biblioteca que aprendi" de "consigo
aumentar minha biblioteca quando o mundo exige". As três primeiras podem
passar num sistema que só sabe reusar.

OS CONTROLES, QUE JÁ SALVARAM ESTA SESSÃO VÁRIAS VEZES
--------------------------------------------------------
  · canal SEM estrutura — nada pode ser promovido, nada pode ajustar bem
  · tarefa monotônica    — a primitiva de regime não pode ser usada
  · esqueletos CONCORRENTES na fase de mineração — ela tem de escolher, e
    pode escolher errado

Sem o primeiro, "formou representação" é compatível com ajustar ruído. Sem o
segundo, "transferiu" é compatível com enfiar a primitiva em tudo.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics as st
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# O mundo: uma matriz de números. Sem nomes, sem unidades, sem semântica.
Fluxo = List[List[float]]


# ══════════════════════════════════════════════ o mundo como fluxo
def gerar_fluxo(rng: random.Random, T: int = 260, K: int = 6,
                estrutura: str = 'regime') -> Tuple[Fluxo, List[int]]:
    """
    `T` instantes × `K` canais, e um alvo binário por instante.

    O processo gerador é escondido. Os canais 0–2 e 3–5 codificam o estado
    latente de maneiras DIFERENTES — escalas, sinais e ruídos distintos — para
    que "usar o canal 4" não seja a mesma coisa que "usar o canal 1" na
    superfície. A transferência, se acontecer, terá de ser de FORMA.

    `estrutura`:
      'regime'  o alvo depende de o latente estar acima ou abaixo de um limiar
      'deriva'  o alvo depende do latente estar SUBINDO (precisa de atraso)
      'ruido'   controle: o alvo é sorteado, não há o que representar
    """
    lim = rng.uniform(-0.3, 0.3)
    s = 0.0
    fluxo: Fluxo = []
    alvo: List[int] = []
    anterior = 0.0
    for t in range(T):
        s = 0.86 * s + rng.gauss(0, 0.45)
        linha = [
            1.7 * s + rng.gauss(0, 0.05),          # canal 0
            -0.6 * s + rng.gauss(0, 0.05),         # canal 1
            0.9 * s * s + rng.gauss(0, 0.05),      # canal 2
            -2.4 * s + 3.0 + rng.gauss(0, 0.05),   # canal 3  (outra escala)
            0.4 * s - 1.5 + rng.gauss(0, 0.05),    # canal 4
            rng.gauss(0, 1.0),                     # canal 5  (só ruído)
        ][:K]
        fluxo.append(linha)
        if estrutura == 'regime':
            alvo.append(1 if s > lim else -1)
        elif estrutura == 'deriva':
            alvo.append(1 if s > anterior else -1)
        else:
            alvo.append(rng.choice([1, -1]))
        anterior = s
    return fluxo, alvo


# ══════════════════════════════════════════════ os átomos de sinal
@dataclass(frozen=True)
class No:
    """
    Átomos deliberadamente do nível de SINAL, não de tarefa:

        LER(k, d)     canal k, `d` instantes atrás
        MEDIA(k, w)   média do canal k na janela w
        NUM(v)        constante
        MENOS(a, b)   diferença
        MAIOR(a, b)   comparação
        SE(p, A, B)   ramificação
        SINAL(s)      devolve ±1

    Nenhum menciona regime, limiar, causa ou objeto. `SE` é controle de fluxo:
    sozinho não sabe em que canal olhar, com que atraso, nem contra o quê
    comparar. É a combinação que significa alguma coisa, e é ela que precisa
    emergir.
    """
    op: str
    args: Tuple[Any, ...] = ()

    def aval(self, f: Fluxo, t: int) -> Any:
        o = self.op
        if o == 'LER':
            k, d = self.args
            return f[max(0, t - d)][k]
        if o == 'MEDIA':
            k, w = self.args
            i0 = max(0, t - w + 1)
            janela = [f[i][k] for i in range(i0, t + 1)]
            return sum(janela) / len(janela)
        if o == 'NUM':
            return self.args[0]
        if o == 'SINAL':
            return self.args[0]
        if o == 'MENOS':
            return self.args[0].aval(f, t) - self.args[1].aval(f, t)
        if o == 'MAIOR':
            return self.args[0].aval(f, t) > self.args[1].aval(f, t)
        if o == 'SE':
            return (self.args[1] if self.args[0].aval(f, t) else self.args[2]).aval(f, t)
        if o == 'LIB':
            return self.args[0].instanciar(self.args[1:]).aval(f, t)
        raise ValueError(o)

    def tamanho(self) -> int:
        if self.op == 'LIB':
            return 1 + sum(a.tamanho() for a in self.args[1:] if isinstance(a, No))
        return 1 + sum(a.tamanho() for a in self.args if isinstance(a, No))

    def esqueleto(self) -> str:
        o = self.op
        if o in ('LER', 'MEDIA', 'NUM'):
            return '□'
        if o == 'SINAL':
            return 's'
        if o == 'LIB':
            return f"@{self.args[0].nome}"
        return f"{o}({','.join(a.esqueleto() for a in self.args if isinstance(a, No))})"

    def tipos_dos_buracos(self) -> List[str]:
        o = self.op
        if o == 'LER':
            return ['leitura']
        if o == 'MEDIA':
            return ['media']
        if o == 'NUM':
            return ['numero']
        if o == 'SINAL':
            return []
        out: List[str] = []
        for a in self.args:
            if isinstance(a, No):
                out.extend(a.tipos_dos_buracos())
        return out

    def __str__(self) -> str:
        o = self.op
        if o == 'LER':
            return f"c{self.args[0]}" + (f"[-{self.args[1]}]" if self.args[1] else "")
        if o == 'MEDIA':
            return f"méd(c{self.args[0]},{self.args[1]})"
        if o == 'NUM':
            return f"{self.args[0]:.3g}"
        if o == 'SINAL':
            return '+' if self.args[0] > 0 else '−'
        if o == 'MENOS':
            return f"({self.args[0]}−{self.args[1]})"
        if o == 'MAIOR':
            return f"{self.args[0]}>{self.args[1]}"
        if o == 'SE':
            return f"se({self.args[0]}){{{self.args[1]}}}senão{{{self.args[2]}}}"
        if o == 'LIB':
            return f"{self.args[0].nome}({','.join(str(a) for a in self.args[1:])})"
        return o


@dataclass
class Primitiva:
    nome: str
    esqueleto: str
    modelo: No
    vezes_vista: int = 0

    def instanciar(self, valores: Sequence[Any]) -> No:
        it = iter(valores)

        def rec(n: No) -> No:
            if n.op in ('LER', 'MEDIA', 'NUM'):
                return next(it)
            if n.op == 'SINAL':
                return n
            return No(n.op, tuple(rec(a) if isinstance(a, No) else a for a in n.args))

        return rec(self.modelo)


MAIS, MENOS_S = No('SINAL', (1,)), No('SINAL', (-1,))


# ══════════════════════════════════════════════ busca
def _leituras(K: int, atrasos: Sequence[int]) -> List[No]:
    return [No('LER', (k, d)) for k in range(K) for d in atrasos]


def _medias(K: int, janelas: Sequence[int]) -> List[No]:
    return [No('MEDIA', (k, w)) for k in range(K) for w in janelas]


def enumerar(K: int, constantes: Sequence[float],
             biblioteca: Sequence[Primitiva]) -> List[No]:
    leituras = _leituras(K, (0, 1))
    medias = _medias(K, (4,))
    numeros = [No('NUM', (c,)) for c in constantes]
    quant = leituras + medias

    testes: List[No] = []
    for a in quant:
        for b in numeros:
            testes.append(No('MAIOR', (a, b)))
    # comparação entre duas leituras do MESMO canal em instantes diferentes:
    # é o que 'deriva' exige, e é uma forma diferente de `MAIOR(leitura, NUM)`
    for k in range(K):
        testes.append(No('MAIOR', (No('LER', (k, 0)), No('LER', (k, 1)))))
        testes.append(No('MAIOR', (No('LER', (k, 0)), No('MEDIA', (k, 4)))))

    progs: List[No] = [MAIS, MENOS_S]
    for t in testes:
        progs.append(No('SE', (t, MAIS, MENOS_S)))
        progs.append(No('SE', (t, MENOS_S, MAIS)))

    for p in biblioteca:
        tipos = p.modelo.tipos_dos_buracos()
        if len(tipos) > 2:
            continue
        opcoes = []
        for tp in tipos:
            if tp == 'leitura':
                opcoes.append(leituras)
            elif tp == 'media':
                opcoes.append(medias)
            else:
                opcoes.append(numeros)
        for combo in itertools.product(*opcoes):
            progs.append(No('LIB', (p,) + combo))
    return progs


def resolver(fluxo: Fluxo, alvo: Sequence[int], K: int,
             constantes: Sequence[float], biblioteca: Sequence[Primitiva],
             lam: float = 2.0, t0: int = 5) -> Dict[str, Any]:
    progs = enumerar(K, constantes, biblioteca)
    idx = range(t0, len(alvo))
    melhor, melhor_c, melhor_ac = None, math.inf, 0.0
    for p in progs:
        erros = 0
        for t in idx:
            if p.aval(fluxo, t) != alvo[t]:
                erros += 1
        c = erros + lam * p.tamanho()
        if c < melhor_c:
            melhor, melhor_c, melhor_ac = p, c, 1 - erros / max(1, len(idx))
    return {'programa': melhor, 'custo': melhor_c, 'ajuste': melhor_ac,
            'tamanho': melhor.tamanho(), 'n_avaliados': len(progs)}


# ══════════════════════════════════════════════ mineração
def _subarvores(n: No) -> List[No]:
    out = [n]
    for a in n.args:
        if isinstance(a, No):
            out.extend(_subarvores(a))
    return out


def minerar(solucoes: Sequence[No], min_vezes: int = 3,
            lam: float = 2.0, max_prim: int = 2) -> List[Primitiva]:
    freq: Dict[str, List[No]] = {}
    for s in solucoes:
        for sub in _subarvores(s):
            if sub.op in ('SINAL', 'LER', 'NUM', 'MEDIA'):
                continue
            freq.setdefault(sub.esqueleto(), []).append(sub)
    out: List[Primitiva] = []
    for esq, ex in sorted(freq.items(), key=lambda kv: -len(kv[1])):
        if len(ex) < min_vezes:
            continue
        modelo = ex[0]
        if modelo.op not in ('SE', 'SINAL', 'LIB'):
            continue           # tem de poder ocupar o lugar de um programa
        if (modelo.tamanho() - 1) * len(ex) * lam <= lam * modelo.tamanho():
            continue           # não paga o próprio custo
        out.append(Primitiva(nome=f"Q{len(out)+1}", esqueleto=esq, modelo=modelo,
                             vezes_vista=len(ex)))
        if len(out) >= max_prim:
            break
    return out


# ══════════════════════════════════════════════ o experimento
CANAIS_A = 3          # fase 1 enxerga só os canais 0–2
CANAIS_TODOS = 6


def experimento(n_fase1: int = 10, n_teste: int = 8, semente: int = 0,
                verboso: bool = False) -> Dict[str, Any]:
    rng = random.Random(semente)
    consts = [-0.6, -0.2, 0.0, 0.2, 0.6]

    # ── 1. formação: resolver com átomos crus, só canais 0–2 ────────────
    solucoes: List[No] = []
    aj_form: List[float] = []
    for i in range(n_fase1):
        est = 'regime' if i % 2 == 0 else 'deriva'
        f, a = gerar_fluxo(rng, K=CANAIS_A, estrutura=est)
        r = resolver(f, a, CANAIS_A, consts, [])
        solucoes.append(r['programa']); aj_form.append(r['ajuste'])

    # controle de formação: fluxo sem estrutura
    aj_ruido = []
    for _ in range(4):
        f, a = gerar_fluxo(rng, K=CANAIS_A, estrutura='ruido')
        aj_ruido.append(resolver(f, a, CANAIS_A, consts, [])['ajuste'])

    # ── 2. mineração ────────────────────────────────────────────────────
    lib = minerar(solucoes)
    if verboso:
        print("primitivas promovidas a partir do fluxo cru:")
        for p in lib:
            print(f"  {p.nome}  {p.esqueleto}  vista {p.vezes_vista}×  "
                  f"exemplar: {p.modelo}")
        if not lib:
            print("  nenhuma")

    # ── 3. transferência: canais 3–5, nunca vistos na fase 1 ────────────
    def so_novos(f: Fluxo) -> Fluxo:
        return [linha[CANAIS_A:] for linha in f]

    tr_sem, tr_com, tr_uso = [], [], 0
    for _ in range(n_teste):
        f, a = gerar_fluxo(rng, K=CANAIS_TODOS, estrutura='regime')
        fn = so_novos(f)
        s_ = resolver(fn, a, CANAIS_TODOS - CANAIS_A, consts, [])
        c_ = resolver(fn, a, CANAIS_TODOS - CANAIS_A, consts, lib)
        tr_sem.append(s_['custo']); tr_com.append(c_['custo'])
        tr_uso += int(any(n.op == 'LIB' for n in _subarvores(c_['programa'])))

    # ── 4. abertura: estrutura que a biblioteca não cobre ───────────────
    #     alvo depende de DUAS condições (regime E deriva) — nenhum esqueleto
    #     de profundidade 1 alcança. O sistema força a primitiva, ou constrói?
    ab_aj, ab_uso = [], 0
    for _ in range(n_teste):
        f, a = gerar_fluxo(rng, K=CANAIS_A, estrutura='regime')
        f2, a2 = gerar_fluxo(rng, K=CANAIS_A, estrutura='deriva')
        combinado = [1 if (x > 0 and y > 0) else -1 for x, y in zip(a, a2)]
        r = resolver(f, combinado, CANAIS_A, consts, lib)
        ab_aj.append(r['ajuste'])
        ab_uso += int(any(n.op == 'LIB' for n in _subarvores(r['programa'])))

    return {
        'lib': lib,
        'formacao': st.mean(aj_form), 'formacao_ruido': st.mean(aj_ruido),
        'transf_sem': st.mean(tr_sem), 'transf_com': st.mean(tr_com),
        'transf_uso': tr_uso / n_teste,
        'abertura_ajuste': st.mean(ab_aj), 'abertura_uso': ab_uso / n_teste,
    }


def relatorio(sementes: Tuple[int, ...] = (0, 1, 2)) -> Dict[str, Any]:
    print("═" * 78)
    print("SUBSTRATO — representação a partir de fluxo temporal cru")
    print("o mundo é T×K números. Sem nomes de campo, sem semântica.")
    print("átomos: ler canal · atraso · média de janela · constante · comparar ·")
    print("        ramificar · sinal    (nenhum menciona regime, causa ou limiar)")
    print("═" * 78)

    ls = [experimento(semente=s, verboso=(i == 0)) for i, s in enumerate(sementes)]
    m = lambda k: st.mean([l[k] for l in ls])
    print()

    print(f"1. FORMAÇÃO      ajuste em fluxo com estrutura ..... {m('formacao'):.0%}")
    print(f"                 ajuste em fluxo de RUÍDO (controle) {m('formacao_ruido'):.0%}")
    print(f"2. MINERAÇÃO     primitivas promovidas por rodada ... "
          f"{st.mean([len(l['lib']) for l in ls]):.1f}")
    print(f"3. TRANSFERÊNCIA custo sem biblioteca .............. {m('transf_sem'):.1f}")
    print(f"                 custo com biblioteca .............. {m('transf_com'):.1f}")
    print(f"                 usou a primitiva em canais novos .. {m('transf_uso'):.0%}")
    print(f"4. ABERTURA      ajuste em estrutura não coberta .... {m('abertura_ajuste'):.0%}")
    print(f"                 forçou a primitiva assim mesmo .... {m('abertura_uso'):.0%}")

    print("─" * 78)
    form, ruido = m('formacao'), m('formacao_ruido')
    uso, ganho = m('transf_uso'), m('transf_sem') - m('transf_com')
    ab_aj = m('abertura_ajuste')

    if form - ruido < 0.15:
        print("FORMAÇÃO FALHOU: o ajuste em fluxo estruturado mal se separa do ajuste")
        print("em ruído. Sem isso as outras três medidas não querem dizer nada — o")
        print("sistema estaria ajustando qualquer coisa.")
        return {'linhas': ls}

    print(f"Formação separa estrutura de ruído ({form:.0%} contra {ruido:.0%}): os")
    print("programas encontrados descrevem o mundo, e não a amostra.")
    if uso > 0.8 and ganho > 0:
        print()
        print("TRANSFERÊNCIA ENTRE CANAIS: a primitiva minerada nos canais 0–2 é")
        print("reusada nos canais 3–5, que têm escala, sinal e ruído diferentes e")
        print("nunca participaram da mineração. O que transferiu foi a FORMA.")
    elif uso <= 0.8:
        print()
        print(f"Transferência {uso:.0%}: a primitiva não está sendo reusada nos canais")
        print("novos de forma consistente.")

    print()
    print(f"4. A PERGUNTA QUE INTERESSA — ajuste {ab_aj:.0%} onde a biblioteca não cobre")
    if ab_aj < form - 0.12:
        print("  O ajuste CAI. Diante de estrutura que precisa de duas condições, o")
        print("  sistema entrega o melhor programa disponível e erra mais — em vez de")
        print("  construir a composição que faltava.")
        print()
        print("  É o mesmo achado de sempre, um andar abaixo: ele reusa a biblioteca")
        print("  que aprendeu; não a AUMENTA quando o mundo exige. A diferença entre")
        print("  'tenho ferramentas' e 'faço ferramentas' continua de pé.")
    else:
        print("  O ajuste NÃO cai — vale conferir se a tarefa de abertura está mesmo")
        print("  fora do alcance da enumeração antes de tratar como descoberta.")
    return {'linhas': ls}


if __name__ == "__main__":
    relatorio()
