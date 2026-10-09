"""
novaphys.geracao — a experiência seleciona a OPERAÇÃO DE EXPANSÃO?
===================================================================

`invencao.py` fechou a seta que faltava: o sistema levantou uma constante a
parâmetro, produzindo `MENOR(VAR,VAR)` — um átomo fora do span do substrato,
verificado por enumeração, usado onde serve e não usado onde não serve.

E deixou exatamente a mesma limitação que `abstracao.py` tinha antes da
auditoria: **havia uma só operação de expansão, e fui eu que a escolhi.** Com
uma alternativa só, "o sistema aplicou a operação certa" não informa nada — não
havia outra.

Este módulo repete o desenho de `descoberta.py` um andar acima:

    descoberta.py   I(experiência ; primitiva promovida)      1,585 bits
    ESTE            I(experiência ; operação de EXPANSÃO)     ?

TRÊS OPERAÇÕES DE EXPANSÃO
---------------------------
Cada uma produz um átomo FORA do span de `MENOR(VAR,NUM)`, e nenhuma é
composição das outras:

    PARAMETRIZAR   MENOR(v_i, v_j)         constante vira variável
    SOMAR          MENOR(v_i + v_j, k)     grandeza derivada aditiva
    MULTIPLICAR    MENOR(v_i · v_j, k)     grandeza derivada multiplicativa

E três famílias de tarefa, cada uma solúvel exatamente por uma delas. O
substrato base só aproxima qualquer das três por escadaria.

A REGRA DO GERADOR — UMA SÓ, PARA TODAS
----------------------------------------
Esta é a parte que decide se o experimento vale. Se o gerador tivesse uma
heurística por operação — "para parametrizar olhe covariância, para somar olhe
outra coisa" — eu teria escolhido a resposta três vezes em vez de uma, e a
informação mútua mediria o meu código.

Então a regra é única e agnóstica:

    olhe ONDE o programa atual erra, e pergunte qual grandeza derivada
    SEPARA esses erros

Cada operação candidata propõe uma grandeza; a que melhor separa os erros
vence. Nenhuma preferência embutida, nenhum ramo por tipo de operação.

O VALIDADOR É SEPARADO DO GERADOR
----------------------------------
O gerador pode propor lixo à vontade — é barato e é o ponto. Quem decide é o
validador, e ele exige o mesmo de sempre:

    · o átomo está FORA do span?      (enumeração, não alegação)
    · reduz o comprimento de descrição?
    · é usado onde serve?
    · NÃO é usado no controle alinhado, onde não ajuda?

DIVERSIDADE: A HIPÓTESE DO PONTO DOCE
--------------------------------------
Com 1, 2 e 3 operações disponíveis, mede-se acerto e lixo. Mais diversidade
deveria achar mais coisa fora do espaço — e também produzir mais candidato
inútil. Se houver ponto ótimo, ele aparece aqui; se não houver, também.

A PRÉ-CONDIÇÃO CONTINUA SENDO PORTÃO
-------------------------------------
Como em `invencao.py`: força bruta primeiro. Se cada família não for
comprovadamente mais barata com a sua operação do que sem nenhuma, o mecanismo
não roda.
"""

from __future__ import annotations

import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 3
LAM = 1.0


# ══════════════════════════════════════════════ grandezas e programas
@dataclass(frozen=True)
class Grandeza:
    """O que um teste compara. `op` identifica a operação que a criou."""
    op: str                       # 'BASE' | 'PARAMETRIZAR' | 'SOMAR' | 'MULTIPLICAR'
    i: int
    j: int = -1

    def valor(self, v: Sequence[float]) -> float:
        if self.op in ('BASE', 'PARAMETRIZAR'):
            return v[self.i]
        if self.op == 'SOMAR':
            return v[self.i] + v[self.j]
        if self.op == 'MULTIPLICAR':
            return v[self.i] * v[self.j]
        raise ValueError(self.op)

    def __str__(self) -> str:
        if self.op in ('BASE', 'PARAMETRIZAR'):
            return f"v{self.i}"
        s = '+' if self.op == 'SOMAR' else '·'
        return f"(v{self.i}{s}v{self.j})"


@dataclass(frozen=True)
class Teste:
    """`grandeza < referência`. A referência é constante ou outra variável."""
    g: Grandeza
    k: Optional[float] = None      # constante
    ref: int = -1                  # ou índice de variável

    @property
    def operacao(self) -> str:
        """Qual operação de expansão produziu este teste. 'BASE' = nenhuma."""
        if self.ref >= 0:
            return 'PARAMETRIZAR'
        return self.g.op if self.g.op != 'PARAMETRIZAR' else 'BASE'

    def aval(self, v: Sequence[float]) -> bool:
        alvo = v[self.ref] if self.ref >= 0 else (self.k or 0.0)
        return self.g.valor(v) < alvo

    def __str__(self) -> str:
        d = f"v{self.ref}" if self.ref >= 0 else f"{self.k:.2f}"
        return f"{self.g}<{d}"


@dataclass(frozen=True)
class Prog:
    """Lista de decisão: se t1 então s1, senão se t2 então s2, senão s."""
    testes: Tuple[Tuple[Teste, int], ...] = ()
    padrao: int = 1

    def aval(self, v: Sequence[float]) -> int:
        for t, s in self.testes:
            if t.aval(v):
                return s
        return self.padrao

    def tamanho(self) -> int:
        return 1 + 3 * len(self.testes)

    def operacoes(self) -> set:
        return {t.operacao for t, _ in self.testes} - {'BASE'}

    def custo(self, obs: Sequence[Obs]) -> float:
        e = sum(1 for (v, s) in obs if self.aval(v) != s)
        return e + LAM * self.tamanho()

    def __str__(self) -> str:
        if not self.testes:
            return '+' if self.padrao > 0 else '−'
        p = " senão ".join(f"se {t} → {'+' if s > 0 else '−'}" for t, s in self.testes)
        return p + f" senão {'+' if self.padrao > 0 else '−'}"


# ══════════════════════════════════════════════ tarefas
FAMILIAS = ('relacional', 'soma', 'produto')


def tarefa(tipo: str, rng: random.Random, n: int = 90,
           par: Tuple[int, int] = (0, 1)) -> List[Obs]:
    i, j = par
    obs: List[Obs] = []
    cs = rng.uniform(0.7, 1.3)
    cp = rng.uniform(0.15, 0.35)
    ca = rng.uniform(0.3, 0.7)
    for _ in range(n):
        v = [rng.random() for _ in range(N_VARS)]
        if tipo == 'relacional':
            s = 1 if v[i] < v[j] else -1
        elif tipo == 'soma':
            s = 1 if (v[i] + v[j]) < cs else -1
        elif tipo == 'produto':
            s = 1 if (v[i] * v[j]) < cp else -1
        elif tipo == 'alinhada':
            s = 1 if v[i] < ca else -1
        else:
            raise ValueError(tipo)
        obs.append((v, s))
    return obs


# ══════════════════════════════════════════════ busca
def _cortes(obs: Sequence[Obs], g: Grandeza, n: int = 5) -> List[float]:
    vals = sorted(g.valor(v) for v, _ in obs)
    return [vals[int((k + 1) * len(vals) / (n + 1))] for k in range(n)]


def _testes_base(obs: Sequence[Obs]) -> List[Teste]:
    out = []
    for i in range(N_VARS):
        g = Grandeza('BASE', i)
        for c in _cortes(obs, g):
            out.append(Teste(g, k=c))
    return out


def _testes_da_operacao(obs: Sequence[Obs], op: str) -> List[Teste]:
    out: List[Teste] = []
    if op == 'PARAMETRIZAR':
        for i in range(N_VARS):
            for j in range(N_VARS):
                if i != j:
                    out.append(Teste(Grandeza('BASE', i), ref=j))
    else:
        for i in range(N_VARS):
            for j in range(i + 1, N_VARS):
                g = Grandeza(op, i, j)
                for c in _cortes(obs, g):
                    out.append(Teste(g, k=c))
    return out


def melhor(obs: Sequence[Obs], testes: Sequence[Teste],
           profundidade: int = 2) -> Prog:
    """
    Lista de decisão gulosa por custo.

    OS DOIS SINAIS DE `padrao` SÃO EXPLORADOS, e isso não é detalhe. A primeira
    versão fixava `padrao` no sinal majoritário e nunca o revisava — então numa
    tarefa equilibrada, onde a resposta certa é `se (v0+v1)<c → + senão −`, com
    `padrao` travado em `+1` nenhum teste melhorava e a busca desistia no
    programa CONSTANTE.

    O estrago foi grande e silencioso: a pré-condição reportou custo 46 sem
    ganho para SOMAR e MULTIPLICAR e ABORTOU o experimento, com a leitura de
    que aquelas operações "não criam pressão". Não era isso. Medido depois à
    mão, a mesma tarefa cai de 21 para 10 com a grandeza derivada.

    Um portão que reprova por bug do próprio solucionador é pior que não ter
    portão: ele produz uma conclusão negativa com cara de rigor.
    """
    melhor_p, melhor_c = None, math.inf
    for padrao in (1, -1):
        atual = Prog(padrao=padrao)
        c_atual = atual.custo(obs)
        for _ in range(profundidade):
            cand, cand_c = atual, c_atual
            for t in testes:
                for s in (1, -1):
                    p = Prog(atual.testes + ((t, s),), padrao)
                    c = p.custo(obs)
                    if c < cand_c:
                        cand, cand_c = p, c
            if cand is atual:
                break
            atual, c_atual = cand, cand_c
        if c_atual < melhor_c:
            melhor_p, melhor_c = atual, c_atual
    return melhor_p


# ══════════════════════════════════════════════ PRÉ-CONDIÇÃO
CERTA = {'relacional': 'PARAMETRIZAR', 'soma': 'SOMAR', 'produto': 'MULTIPLICAR'}


def sanidade_do_solucionador(verboso: bool = True) -> Dict[str, Any]:
    """
    O solucionador acha a solução quando ela está no conjunto de testes?

    Esta checagem existe porque "não há pressão" tem TRÊS causas possíveis e
    elas eram indistinguíveis:

        1. realmente não há pressão
        2. a referência de comparação está errada
        3. o SOLUCIONADOR não encontra uma solução que sabemos existir

    A (3) aconteceu: a busca fixava o sinal padrão no majoritário e nunca o
    revisava, então em tarefa equilibrada ela desistia no programa constante e
    a pré-condição reprovou SOMAR e MULTIPLICAR — com a leitura, que eu
    reportei, de que aquelas operações não criavam pressão. Era o instrumento.

    Um portão que reprova por defeito do próprio solucionador é pior que não
    ter portão: produz conclusão negativa com cara de rigor.
    """
    falhas = []
    for fam in FAMILIAS:
        rng = random.Random(4242)
        obs = tarefa(fam, rng)
        testes = list(_testes_base(obs)) + _testes_da_operacao(obs, CERTA[fam])
        p = melhor(obs, testes)
        aj = 1 - sum(1 for (v, s) in obs if p.aval(v) != s) / len(obs)
        usou = CERTA[fam] in p.operacoes()
        if aj < 0.85 or not usou:
            falhas.append((fam, aj, usou))
        if verboso:
            print(f"  {fam:<12}{CERTA[fam]:<15} ajuste {aj:>5.0%}  "
                  f"usou a operação: {'sim' if usou else 'NÃO'}")
    return {'ok': not falhas, 'falhas': falhas}


def precondicao(sementes: Sequence[int] = tuple(range(8)),
                verboso: bool = True) -> Dict[str, Any]:
    if verboso:
        print("SANIDADE DO SOLUCIONADOR — acha a solução quando ela é dada?")
    san = sanidade_do_solucionador(verboso)
    if not san['ok']:
        if verboso:
            print("  → INSTRUMENTO NÃO CALIBRADO. Qualquer 'não há pressão' daqui")
            print("    para a frente seria defeito do solucionador, não resultado.")
        return {'linhas': [], 'aprovada': False, 'sanidade': san}
    if verboso:
        print()
    linhas = []
    for fam in FAMILIAS:
        ok = 0
        gsem, gcom = [], []
        for s in sementes:
            rng = random.Random(s * 733 + hash(fam) % 91)
            obs = tarefa(fam, rng)
            base = melhor(obs, _testes_base(obs))
            exp = melhor(obs, list(_testes_base(obs))
                         + _testes_da_operacao(obs, CERTA[fam]))
            gsem.append(base.custo(obs)); gcom.append(exp.custo(obs))
            ok += int(exp.custo(obs) < base.custo(obs) - 1.0)
        linhas.append({'familia': fam, 'op': CERTA[fam], 'taxa': ok / len(sementes),
                       'custo_sem': st.mean(gsem), 'custo_com': st.mean(gcom)})
    if verboso:
        print("PRÉ-CONDIÇÃO — cada operação compensa na sua família? (força bruta)")
        print(f"{'família':<12}{'operação':<15}{'custo sem':>11}{'custo com':>11}{'compensa':>11}")
        for l in linhas:
            print(f"{l['familia']:<12}{l['op']:<15}{l['custo_sem']:>11.1f}"
                  f"{l['custo_com']:>11.1f}{l['taxa']:>10.0%}")
    aprovada = all(l['taxa'] >= 0.9 for l in linhas)
    return {'linhas': linhas, 'aprovada': aprovada, 'sanidade': san}


# ══════════════════════════════════════════════ GERADOR — uma regra só
def gerar(obs: Sequence[Obs], base: Prog, disponiveis: Sequence[str],
          k: int = 1, n_perm: int = 40, alpha: float = 0.95) -> List[str]:
    """
    A ÚNICA regra: onde o programa atual erra, qual grandeza derivada separa
    esses erros?

    Para cada operação disponível, monta as grandezas que ela produz e mede
    quanto o melhor corte sobre elas separa acerto de erro do programa atual.
    A operação cuja melhor grandeza separa mais é proposta.

    Nenhum ramo por tipo de operação. Se houvesse — "para parametrizar olhe
    covariância, para somar olhe outra coisa" — eu teria escolhido a resposta
    uma vez por operação, e a informação mútua mediria o meu código em vez da
    experiência do sistema.
    """
    errou = [(v, s) for (v, s) in obs if base.aval(v) != s]
    if len(errou) < 5:
        return []
    marcado = [(v, 1 if base.aval(v) != s else -1) for (v, s) in obs]

    # ── ABSTENÇÃO CALIBRADA ──────────────────────────────────────────────
    # A regra geradora é a mesma, e continua sendo uma só. O que muda é que
    # ela agora precisa PASSAR NUM TESTE antes de propor.
    #
    # O problema que isto resolve: a regra faz max_j Separação(X_j, erro), e
    # com muitas grandezas candidatas algum X_j parece ótimo mesmo quando é
    # independente do erro. Medido: 64% de propostas nas tarefas alinhadas ao
    # eixo, onde nenhuma expansão ajuda — um gerador que propõe em tudo acerta
    # por acidente e torna a informação mútua ilegível.
    #
    # Duas travas, e elas resolvem coisas diferentes:
    #   SELEÇÃO NO TREINO, DECISÃO NO HOLDOUT — a grandeza é escolhida numa
    #     metade e avaliada na outra, então "achei algo que separa" tem de
    #     sobreviver a dados que não participaram da escolha.
    #   H₀ POR PERMUTAÇÃO — embaralhando os rótulos de erro no holdout, mede-se
    #     quanta separação aparece por acaso com aquele mesmo teste. Permutação
    #     em vez de correção analítica porque ela absorve sozinha o número de
    #     candidatos, e como a escolha foi feita sem ver o holdout, não há
    #     comparação múltipla a corrigir.
    meio = len(marcado) // 2
    treino, holdout = marcado[:meio], marcado[meio:]
    if len(treino) < 20 or len(holdout) < 20:
        return []

    def separacao(t: Teste, dados: Sequence[Tuple[List[float], int]]) -> float:
        dentro = [m for (v, m) in dados if t.aval(v)]
        fora = [m for (v, m) in dados if not t.aval(v)]
        if len(dentro) < 5 or len(fora) < 5:
            return 0.0
        p1 = sum(1 for m in dentro if m > 0) / len(dentro)
        p2 = sum(1 for m in fora if m > 0) / len(fora)
        return abs(p1 - p2)

    rng = random.Random(20240)
    escores: List[Tuple[float, str]] = []
    for op in disponiveis:
        # 1. escolhe a grandeza olhando SÓ o treino
        cand, melhor_tr = None, 0.0
        for t in _testes_da_operacao(obs, op):
            s_tr = separacao(t, treino)
            if s_tr > melhor_tr:
                cand, melhor_tr = t, s_tr
        if cand is None:
            continue
        # 2. mede no holdout
        s_ho = separacao(cand, holdout)
        # 3. H₀: a mesma grandeza contra rótulos embaralhados
        nulos = []
        vs = [v for v, _ in holdout]
        ms = [m for _, m in holdout]
        for _ in range(n_perm):
            emb = ms[:]
            rng.shuffle(emb)
            nulos.append(separacao(cand, list(zip(vs, emb))))
        nulos.sort()
        limite = nulos[min(len(nulos) - 1, int(alpha * len(nulos)))]
        if s_ho > limite:
            escores.append((s_ho, op))
    escores.sort(reverse=True)
    return [op for _, op in escores[:k]]


def validar(obs_novas: Sequence[Sequence[Obs]], op: str) -> Dict[str, Any]:
    """Compressão e escopo. O gerador propõe; aqui se decide."""
    ganhos, usou = [], 0
    for ob in obs_novas:
        b = melhor(ob, _testes_base(ob))
        e = melhor(ob, list(_testes_base(ob)) + _testes_da_operacao(ob, op))
        ganhos.append(b.custo(ob) - e.custo(ob))
        usou += int(op in e.operacoes())
    return {'ganho': st.mean(ganhos), 'uso': usou / max(1, len(obs_novas))}


# ══════════════════════════════════════════════ experimento
def info_mutua(pares: Sequence[Tuple[str, str]]) -> float:
    n = len(pares)
    if n == 0:
        return 0.0
    pxy, px, py = Counter(pares), Counter(a for a, _ in pares), Counter(b for _, b in pares)
    return sum((c / n) * math.log2((c / n) / ((px[a] / n) * (py[b] / n)))
               for (a, b), c in pxy.items())


def experimento(diversidade: int, sementes: Sequence[int] = (0, 1, 2, 3, 4),
                n_tarefas: int = 5) -> Dict[str, Any]:
    todas = ['PARAMETRIZAR', 'SOMAR', 'MULTIPLICAR']
    disp = todas[:diversidade]
    pares: List[Tuple[str, str]] = []
    acertos = tot = 0
    lixo = tot_prop = 0
    matriz: Dict[str, str] = {}

    for fam in FAMILIAS:
        escolhas: List[str] = []
        for s in sementes:
            rng = random.Random(s * 449 + hash(fam) % 83)
            props: List[str] = []
            for k in range(n_tarefas):
                par = [(0, 1), (1, 2), (0, 2)][k % 3]
                obs = tarefa(fam, rng, par=par)
                base = melhor(obs, _testes_base(obs))
                props.extend(gerar(obs, base, disp, k=1))
            if not props:
                escolhas.append('nenhuma')
                continue
            op = Counter(props).most_common(1)[0][0]
            escolhas.append(op)
            tot += 1
            acertos += int(op == CERTA[fam])
            # lixo: proposta validada que não compensa
            novas = [tarefa(fam, rng, par=(0, 1)) for _ in range(3)]
            v = validar(novas, op)
            tot_prop += 1
            lixo += int(v['ganho'] <= 0.5)
        for e in escolhas:
            pares.append((fam, e))
        matriz[fam] = Counter(escolhas).most_common(1)[0][0]

    # controle: tarefas alinhadas ao eixo — nenhuma expansão deve ser proposta
    rng = random.Random(999)
    propostas_controle = 0
    for _ in range(len(sementes) * n_tarefas):
        obs = tarefa('alinhada', rng)
        base = melhor(obs, _testes_base(obs))
        propostas_controle += int(bool(gerar(obs, base, disp, k=1)))

    return {'diversidade': diversidade, 'matriz': matriz,
            'im': info_mutua(pares), 'acerto': acertos / max(1, tot),
            'lixo': lixo / max(1, tot_prop),
            'controle': propostas_controle / (len(sementes) * n_tarefas)}


def relatorio() -> Dict[str, Any]:
    print("═" * 78)
    print("SELEÇÃO DE OPERAÇÕES DE EXPANSÃO")
    print("uma regra de geração só, igual para todas: onde erro, e qual grandeza")
    print("derivada separa esses erros")
    print("═" * 78)

    pc = precondicao()
    print("─" * 78)
    if not pc['aprovada']:
        print("PRÉ-CONDIÇÃO REPROVADA — o mecanismo NÃO será executado.")
        return {'precondicao': pc}
    print("Pré-condição APROVADA em todas as famílias. O mecanismo pode rodar.")
    print()

    res = [experimento(d) for d in (1, 2, 3)]
    print(f"{'ops disponíveis':<17}{'relacional':>14}{'soma':>10}{'produto':>12}"
          f"{'I(exp;op)':>12}{'acerto':>9}{'lixo':>7}{'controle':>10}")
    for r in res:
        m = r['matriz']
        print(f"{r['diversidade']:<17}{m['relacional']:>14}{m['soma']:>10}"
              f"{m['produto']:>12}{r['im']:>11.3f}b{r['acerto']:>8.0%}"
              f"{r['lixo']:>7.0%}{r['controle']:>10.0%}")

    print("─" * 78)
    r3 = res[-1]
    if r3['controle'] > 0.2:
        print(f"CONTROLE REPROVADO: propõe expansão em {r3['controle']:.0%} das tarefas")
        print("alinhadas ao eixo, onde nenhuma expansão ajuda. As outras notas ficam")
        print("sem sentido — um gerador que propõe sempre acerta por acidente.")
    elif r3['im'] > 1.0 and r3['acerto'] > 0.8:
        print("A EXPERIÊNCIA SELECIONA A OPERAÇÃO DE EXPANSÃO. Com três operações")
        print("disponíveis e uma regra de geração única, qual delas é aplicada depende")
        print("da distribuição de tarefas — não de uma preferência escrita no código.")
        print()
        print("Isto é um andar acima de `invencao.py`: lá o sistema decidia QUANDO")
        print("expandir e ONDE; aqui decide COM QUE OPERAÇÃO, entre alternativas.")
    elif r3['im'] < 0.3:
        print("A EXPERIÊNCIA NÃO SELECIONA a operação. Com três disponíveis, a escolha")
        print("praticamente não depende do que o sistema viu — o que existe é")
        print("preferência estrutural do gerador, não seleção experiencial.")
    else:
        print(f"Parcial: I = {r3['im']:.3f} bits, acerto {r3['acerto']:.0%}.")

    print()
    print("PONTO DOCE — mais diversidade acha mais, e também produz mais lixo:")
    for r in res:
        print(f"  {r['diversidade']} operação(ões): acerto {r['acerto']:>4.0%} · "
              f"lixo {r['lixo']:>4.0%} · I {r['im']:.3f}b")
    return {'precondicao': pc, 'resultados': res}


if __name__ == "__main__":
    relatorio()
