"""
novaphys.aquisicao — o auto-relato muda a estratégia, e isso vale a pena?
=========================================================================

As bancadas anteriores mediram representação. Esta mede AÇÃO: o campo de
crenças dizer "minha representação é insuficiente" só vale alguma coisa se
mudar o que o agente faz em seguida, e se essa mudança render conhecimento por
experimento.

    representação → incerteza → decisão → experimento → atualização → ↺

O MUNDO
--------
Um grafo causal com tipos de aresta. Cada TIPO é transitivo ou não, e isso é
sorteado e ESCONDIDO. A verdade que o agente persegue é quais pares (a,b) estão
causalmente ligados — o que depende da semântica dos tipos, que ele não recebe.

Duas ações, ambas custando um experimento:

    sondar(a,b)        observação ruidosa sobre o par (a,b) valer
    testar_relacao(t)  revela se o tipo t compõe ou não

A segunda é a que o auto-relato deveria pedir. Ela não responde nenhuma
pergunta sobre pares diretamente — resolve o MECANISMO, e com isso torna
utilizável toda a evidência já coletada por aquele tipo. Um agente que só
sonda acumula observações que a competência mantém amortecidas; um agente que
lê o próprio estado gasta um experimento e destrava as outras.

Se isso for verdade, aparece em eficiência amostral. Se não for, aparece
também — e a bancada de indução já mostrou uma vez que escolher onde medir só
venceu o acaso em 1 de 5 paisagens, então este resultado não é dado.

TRÊS BRAÇOS, UMA VARIÁVEL POR VEZ
----------------------------------
Todos com o MESMO espaço de ações, o MESMO campo de crenças, as MESMAS
sementes e o MESMO orçamento. Só a política de escolha muda:

    aleatorio    escolhe uniformemente (baseline)
    incerteza    máxima entropia sobre as hipóteses — aprendizado ativo clássico,
                 ignora o auto-relato
    autorelato   resolve o que está SUSPENSO primeiro, depois máxima entropia

A comparação que importa é `incerteza` vs `autorelato`: mesma família de
política, mesma informação disponível, diferença única em usar ou não o estado
epistemológico. `aleatorio` está lá porque sem ele não se sabe se qualquer uma
das duas está fazendo algo.

MEDIR AQUISIÇÃO, NÃO ACERTO
----------------------------
"Quantos experimentos até acertar" esconde o que interessa. Um agente pode
gastar 100 e acertar; outro, 8. Então o registro é por experimento:

    entropia do campo · erro de predição · hipóteses eliminadas ·
    mudanças estruturais (tipos de relação resolvidos)
"""

from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .crencas import Hipotese, Observacao, VOCABULARIO


# ══════════════════════════════════════════════ o mundo
@dataclass
class Mundo:
    """
    Grafo causal com semântica de tipo escondida.

    `transitiva` é o segredo: o agente vê o NOME do tipo em cada aresta e não
    vê se ele compõe. É essa lacuna que a ação `testar_relacao` preenche, e é
    ela que separa incerteza sobre os dados de incerteza sobre o mecanismo.
    """
    nos: List[str]
    arestas: List[Tuple[str, str, str]]          # (origem, destino, tipo)
    transitiva: Dict[str, bool]                  # ESCONDIDO do agente
    ruido: float = 0.8                           # P(observação concorda com a verdade)

    def verdade(self) -> Set[Tuple[str, str]]:
        """Fecho usando SÓ as arestas de tipos que compõem."""
        comp = {(o, d) for (o, d, t) in self.arestas if self.transitiva.get(t, False)}
        todas = {(o, d) for (o, d, _) in self.arestas}
        alcanca = set(comp)
        mudou = True
        while mudou:
            mudou = False
            for (a, b) in list(alcanca):
                for (c, d) in list(alcanca):
                    if b == c and a != d and (a, d) not in alcanca:
                        alcanca.add((a, d)); mudou = True
        return alcanca | todas

    def hipoteses(self) -> List[Tuple[str, str]]:
        """Os pares em aberto: os que não são aresta direta."""
        direta = {(o, d) for (o, d, _) in self.arestas}
        return [(a, b) for a in self.nos for b in self.nos
                if a != b and (a, b) not in direta]

    def tipo_do_caminho(self, par: Tuple[str, str]) -> str:
        """O tipo da primeira aresta que sai da origem — o canal da evidência."""
        for (o, d, t) in self.arestas:
            if o == par[0]:
                return t
        return self.arestas[0][2] if self.arestas else 'causa'


def gerar_mundo(rng: random.Random, n_nos: int = 8, n_tipos: int = 3,
                nomes: Optional[Sequence[str]] = None,
                semantica: Optional[Dict[str, bool]] = None) -> Mundo:
    nos = [f"n{i}" for i in range(n_nos)]
    tipos = list(nomes) if nomes else [f"t{i}" for i in range(n_tipos)]
    trans = dict(semantica) if semantica else {t: rng.random() < 0.5 for t in tipos}
    arestas = []
    for i in range(n_nos - 1):
        arestas.append((nos[i], nos[i + 1], rng.choice(tipos)))
    for _ in range(n_nos // 2):
        a, b = rng.sample(nos, 2)
        arestas.append((a, b, rng.choice(tipos)))
    return Mundo(nos=nos, arestas=arestas, transitiva=trans)


# ══════════════════════════════════════════════ o agente
class Agente:
    """
    Campo de crenças sobre os pares, mais um registro do que já se sabe sobre
    os TIPOS. Os três braços diferem só em `escolher`.
    """

    def __init__(self, mundo: Mundo, politica: str, seed: int = 0):
        self.m = mundo
        self.politica = politica
        self.rng = random.Random(seed)
        self.tipos_resolvidos: Dict[str, bool] = {}
        self.campo: Dict[Tuple[str, str], Hipotese] = {}
        for par in mundo.hipoteses():
            self.campo[par] = Hipotese(afirmacao=par, prior=0.5)
        self.tipos = sorted({t for (_, _, t) in mundo.arestas})
        self.historia: List[Dict[str, Any]] = []

    # ── vocabulário efetivo: o que o agente considera "sabido" ────────────
    def _conhecido(self, t: str) -> bool:
        return t in VOCABULARIO or t in self.tipos_resolvidos

    def _obs_para(self, par: Tuple[str, str]) -> Observacao:
        t = self.m.tipo_do_caminho(par)
        verdade = par in self.m.verdade()
        q = self.m.ruido
        concorda = self.rng.random() < q
        aponta = concorda if verdade else (not concorda)
        lr = (q / (1 - q)) if aponta else ((1 - q) / q)
        # o rótulo do canal é o que o agente vê; a semântica, não
        rel = t if self._conhecido(t) else f"__{t}__"
        return Observacao(lr=lr, relacao=rel if self._conhecido(t) else t, rotulo=t)

    # ── ações ─────────────────────────────────────────────────────────────
    def sondar(self, par: Tuple[str, str]) -> None:
        h = self.campo[par]
        t = self.m.tipo_do_caminho(par)
        obs = self._obs_para(par)
        # Se o tipo foi resolvido por experimento, a evidência passa a ser
        # legível: injetamos o tipo como conhecido no cálculo de competência.
        if t in self.tipos_resolvidos:
            obs.relacao = 'causa'
        h.observar(obs)

    def testar_relacao(self, t: str) -> None:
        """
        Resolve a semântica do tipo. Não responde nenhuma pergunta sobre pares
        — mas destrava TODA a evidência já coletada por aquele canal, o que é
        o ponto: o auto-relato pede uma ação cujo retorno é indireto.
        """
        self.tipos_resolvidos[t] = self.m.transitiva.get(t, False)
        # reprocessa: as observações antigas daquele tipo viram legíveis
        for par, h in self.campo.items():
            if self.m.tipo_do_caminho(par) != t:
                continue
            antigas = list(h.evid_favor) + list(h.evid_contra)
            novo = Hipotese(afirmacao=par, prior=h.prior)
            for o in sorted(antigas, key=lambda x: x.experimento):
                novo.observar(Observacao(lr=o.lr, relacao='causa', rotulo=o.rotulo))
            self.campo[par] = novo

    # ── política ──────────────────────────────────────────────────────────
    def escolher(self) -> Tuple[str, Any]:
        pares = list(self.campo.keys())
        acoes: List[Tuple[str, Any]] = [('sondar', p) for p in pares]
        acoes += [('testar', t) for t in self.tipos if t not in self.tipos_resolvidos]

        if self.politica == 'aleatorio':
            return self.rng.choice(acoes)

        if self.politica == 'incerteza':
            # máxima entropia sobre as hipóteses; ignora o estado epistemológico
            alvo = max(pares, key=lambda p: -abs(self.campo[p].posterior - 0.5))
            return ('sondar', alvo)

        # autorelato: resolve o que o campo marca como SUSPENSO primeiro
        suspensos = set()
        for p, h in self.campo.items():
            if h.estado == 'suspensa':
                t = self.m.tipo_do_caminho(p)
                if t not in self.tipos_resolvidos:
                    suspensos.add(t)
        if suspensos:
            return ('testar', sorted(suspensos)[0])
        alvo = max(pares, key=lambda p: -abs(self.campo[p].posterior - 0.5))
        return ('sondar', alvo)

    def agir(self, n: int) -> None:
        for k in range(n):
            tipo, arg = self.escolher()
            if tipo == 'sondar':
                self.sondar(arg)
            else:
                self.testar_relacao(arg)
            self.historia.append({'k': k + 1, 'acao': tipo, 'arg': arg, **self.metricas()})

    # ── métricas de AQUISIÇÃO, não de acerto ──────────────────────────────
    def metricas(self) -> Dict[str, float]:
        verdade = self.m.verdade()
        posts = [h.posterior for h in self.campo.values()]
        ent = sum(-(p * math.log(max(p, 1e-12)) + (1 - p) * math.log(max(1 - p, 1e-12)))
                  for p in posts) / max(1, len(posts))
        erro = st.mean([abs(h.posterior - (1.0 if par in verdade else 0.0))
                        for par, h in self.campo.items()])
        elim = sum(1 for p in posts if p < 0.1 or p > 0.9) / max(1, len(posts))
        return {'entropia': ent, 'erro': erro, 'eliminadas': elim,
                'tipos_resolvidos': len(self.tipos_resolvidos)}


# ══════════════════════════════════════════════ experimento A — circuito conhecido
def experimento_a(n_mundos: int = 40, orcamento: int = 140, alvo_erro: float = 0.20,
                  semente: int = 0, nomes_conhecidos: bool = True,
                  n_nos: int = 6) -> Dict[str, Any]:
    """
    Mesmas sementes, mesmos mundos, mesmo orçamento, mesmo campo. Muda só a
    política. A métrica principal é quantos experimentos até o erro médio
    cruzar o alvo — e não o erro final, porque erro final premia quem gastou
    tudo e esconde quem chegou lá antes.

    DOIS REGIMES, e a distinção é o que torna o experimento informativo:

      nomes_conhecidos=True   nenhuma hipótese fica SUSPENSA, então a política
                              `autorelato` degenera em `incerteza`. O resultado
                              esperado é EMPATE, e ele é controle negativo: se
                              `autorelato` perder aqui, o mecanismo cobra um
                              preço mesmo quando não tem nada a fazer.

      nomes_conhecidos=False  há mecanismo a resolver. É o único regime em que
                              a pergunta "gastar um experimento no mecanismo se
                              paga?" tem conteúdo.

    A primeira rodada desta bancada rodou só o primeiro regime, deu empate, e
    o empate não queria dizer nada — eu tinha tirado do agente exatamente a
    ação que estava sendo testada. Orçamento e tamanho do mundo também foram
    recalibrados: com 8 nós e 60 experimentos NENHUM braço cruzava o alvo, e um
    alvo que ninguém alcança não compara nada.
    """
    braços = ['aleatorio', 'incerteza', 'autorelato']
    res: Dict[str, Dict[str, List[float]]] = {
        b: {'passos': [], 'erro_final': [], 'ganho_por_exp': [], 'eliminadas': []}
        for b in braços}

    for i in range(n_mundos):
        rng = random.Random(semente + i)
        if nomes_conhecidos:
            nomes = ['causa', 'leva_a', 'correlaciona_com']
            sem = {'causa': True, 'leva_a': True, 'correlaciona_com': False}
        else:
            nomes, sem = None, None
        mundo = gerar_mundo(rng, n_nos=n_nos, nomes=nomes, semantica=sem)
        if not mundo.hipoteses():
            continue
        for b in braços:
            ag = Agente(mundo, b, seed=semente + i)
            e0 = ag.metricas()['erro']
            ag.agir(orcamento)
            passos = next((h['k'] for h in ag.historia if h['erro'] <= alvo_erro),
                          orcamento + 1)
            fim = ag.historia[-1]
            res[b]['passos'].append(passos)
            res[b]['erro_final'].append(fim['erro'])
            res[b]['ganho_por_exp'].append((e0 - fim['erro']) / orcamento)
            res[b]['eliminadas'].append(fim['eliminadas'])
    return res


# ══════════════════════════════════════════════ experimento B — circuito aberto
_SIL_A = ['ka', 'ru', 'mi', 'to', 'ze', 'pa', 'lo', 'vi', 'ne', 'su']
_SIL_B = ['dra', 'plo', 'tri', 'gan', 'vex', 'mor', 'lith', 'ras', 'kun', 'fel']


def gerador_externo(seed: int, celula: str) -> Tuple[List[str], Dict[str, bool]]:
    """
    Vocabulário e semântica criados DEPOIS de o sistema estar congelado.

    Três células, e a terceira é a que decide:

      novo       nomes inventados. Um detector por lista vence isto de graça —
                 nome fora da lista, marca desconhecido. Está aqui como
                 controle positivo do arnês, não como desafio.

      conhecido  nomes do vocabulário com a semântica de sempre. Não pode
                 disparar alarme. É o teste de ceticismo excessivo.

      trocado    nomes do vocabulário com a semântica INVERTIDA: `causa` que
                 não compõe, `correlaciona_com` que compõe. A busca no
                 vocabulário devolve competência 1,0 e o sistema segue
                 confiante dentro de um mundo que ele está lendo errado.

    É `trocado` que separa detecção de novidade de consulta a uma lista, e é
    por isso que ela existe: nenhum nome novo aparece, então não há nada para
    a lista estranhar.
    """
    rng = random.Random(seed)
    if celula == 'novo':
        nomes = [rng.choice(_SIL_A) + rng.choice(_SIL_B) for _ in range(3)]
        return nomes, {t: rng.random() < 0.5 for t in nomes}
    if celula == 'conhecido':
        nomes = ['causa', 'leva_a', 'correlaciona_com']
        return nomes, {'causa': True, 'leva_a': True, 'correlaciona_com': False}
    if celula == 'trocado':
        nomes = ['causa', 'leva_a', 'correlaciona_com']
        return nomes, {'causa': False, 'leva_a': False, 'correlaciona_com': True}
    raise ValueError(celula)


def experimento_b(n_mundos: int = 40, orcamento: int = 40,
                  semente: int = 0) -> Dict[str, Any]:
    """
    O sistema percebe SOZINHO que a estrutura não cabe no modelo?

    Sinal medido: fração de hipóteses que o campo marca como SUSPENSA depois
    do orçamento. Em `trocado` o mundo é tão ilegível quanto em `novo` — a
    única diferença é que os nomes são familiares.
    """
    out: Dict[str, Dict[str, float]] = {}
    for celula in ('novo', 'conhecido', 'trocado'):
        sinal, erro = [], []
        for i in range(n_mundos):
            nomes, sem = gerador_externo(semente + i, celula)
            mundo = gerar_mundo(random.Random(semente + i), nomes=nomes, semantica=sem)
            if not mundo.hipoteses():
                continue
            ag = Agente(mundo, 'incerteza', seed=semente + i)
            ag.agir(orcamento)
            susp = sum(1 for h in ag.campo.values() if h.estado == 'suspensa')
            sinal.append(susp / max(1, len(ag.campo)))
            erro.append(ag.metricas()['erro'])
        out[celula] = {'sinal_suspensa': st.mean(sinal) if sinal else float('nan'),
                       'erro': st.mean(erro) if erro else float('nan'),
                       'n': len(sinal)}
    return out


# ══════════════════════════════════════════════ relatório
def _tabela_a(a: Dict[str, Any], orcamento: int) -> Tuple[float, float]:
    print(f"{'política':<14}{'exp. até o alvo':>17}{'erro final':>13}"
          f"{'ganho/exp':>12}{'eliminadas':>13}")
    for b in ('aleatorio', 'incerteza', 'autorelato'):
        d = a[b]
        med = st.median(d['passos'])
        marca = f"{med:.0f}" if med <= orcamento else f">{orcamento}"
        print(f"{b:<14}{marca:>17}{st.mean(d['erro_final']):>13.3f}"
              f"{st.mean(d['ganho_por_exp']):>12.4f}{st.mean(d['eliminadas']):>12.0%}")
    return (st.median(a['incerteza']['passos']), st.median(a['autorelato']['passos']),
            st.median(a['aleatorio']['passos']))


def relatorio(n_mundos: int = 40, orcamento: int = 140) -> Dict[str, Any]:
    print("═" * 78)
    print("AQUISIÇÃO DE CONHECIMENTO — o auto-relato muda a estratégia, e compensa?")
    print("═" * 78)

    a1 = experimento_a(n_mundos=n_mundos, orcamento=orcamento, nomes_conhecidos=True)
    print(f"\nA1 — VOCABULÁRIO CONHECIDO   (controle negativo: nada a resolver,")
    print(f"     o esperado é EMPATE)      {n_mundos} mundos · orçamento {orcamento} · alvo 0,20")
    p1_inc, p1_aut, p1_ale = _tabela_a(a1, orcamento)

    a2 = experimento_a(n_mundos=n_mundos, orcamento=orcamento, nomes_conhecidos=False)
    print(f"\nA2 — VOCABULÁRIO DESCONHECIDO  (há mecanismo a resolver: é aqui que a")
    print(f"     pergunta tem conteúdo)")
    p2_inc, p2_aut, p2_ale = _tabela_a(a2, orcamento)

    print("─" * 78)
    print("LEITURA — e uma ressalva que muda qual número reportar:")
    print()
    print("  A política `incerteza` NUNCA escolhe `testar_relacao`: ela seleciona por")
    print("  entropia sobre as hipóteses, e um tipo de relação não é uma hipótese.")
    print("  Em A2 ela é ESTRUTURALMENTE incapaz de resolver o mecanismo, então o")
    print(f"  contraste {p2_inc:.0f} → {p2_aut:.0f} é em parte artefato de como escrevi a baseline.")
    print("  Isso é um achado sobre aprendizado ativo clássico — ele não tem onde")
    print("  representar 'meu mecanismo está errado' — mas não é uma vitória medida.")
    print()
    print("  A comparação justa é contra `aleatorio`, que tem o MESMO espaço de ações")
    print("  e escolhe uniformemente:")
    print()
    print(f"  A1 (nada a resolver) ..... aleatorio {p1_ale:>4.0f} · incerteza {p1_inc:>4.0f} · "
          f"autorelato {p1_aut:>4.0f}")
    print(f"  A2 (há o que resolver) ... aleatorio {p2_ale:>4.0f} · incerteza {p2_inc:>4.0f} · "
          f"autorelato {p2_aut:>4.0f}")
    print()
    if p1_aut > p1_inc * 1.15:
        print("PROBLEMA no controle: o auto-relato PIORA mesmo sem ter nada a fazer.")
        print("Isso é custo puro, e invalida qualquer ganho medido em A2.")
    elif p2_aut < p2_ale * 0.9:
        ganho = (p2_ale - p2_aut) / p2_ale
        print(f"Contra a baseline de mesmo espaço de ações: {p2_ale:.0f} → {p2_aut:.0f} "
              f"experimentos ({ganho:.0%} a menos).")
        custo = (p1_aut - p1_inc) / max(1.0, p1_inc)
        print(f"E o preço quando não há o que resolver é {custo:+.0%} (A1) — o mecanismo")
        print("não é de graça, mas é barato onde não serve e paga onde serve.")
    elif p2_aut > p2_ale * 1.1:
        print("Contra a baseline de mesmo espaço de ações o auto-relato PIORA: o")
        print("experimento gasto no mecanismo não se paga neste orçamento.")
    else:
        print("Contra a baseline de mesmo espaço de ações, empate. O auto-relato não")
        print(f"muda a aquisição de forma detectável, e um empate com n={n_mundos} não é")
        print("evidência de que nunca mude.")
    a = {'A1': a1, 'A2': a2}

    b = experimento_b(n_mundos=n_mundos)
    print(f"\nB — CIRCUITO ABERTO  (vocabulário e semântica gerados depois do "
          f"sistema congelado)")
    print(f"{'célula':<14}{'sinal SUSPENSA':>17}{'erro':>10}   o que significa")
    nomes_leg = {
        'novo': 'nomes inventados — detector por lista vence de graça',
        'conhecido': 'nomes e semântica usuais — não pode alarmar',
        'trocado': 'nomes usuais, semântica INVERTIDA — a prova real',
    }
    for c in ('novo', 'conhecido', 'trocado'):
        d = b[c]
        print(f"{c:<14}{d['sinal_suspensa']:>16.0%}{d['erro']:>10.3f}   {nomes_leg[c]}")

    print("─" * 78)
    s_novo, s_conh, s_troc = (b['novo']['sinal_suspensa'],
                              b['conhecido']['sinal_suspensa'],
                              b['trocado']['sinal_suspensa'])
    # O critério é COMPARATIVO, não um limiar absoluto. A pergunta é se
    # `trocado` se distingue de `conhecido` — se não se distingue, o detector
    # não viu a troca, independente do valor em que os dois pousaram.
    if s_novo > 0.5 and abs(s_troc - s_conh) < 0.05:
        print("VEREDITO: o detector é uma LISTA. Ele estranha nome desconhecido e")
        print("não estranha mundo desconhecido. Em `trocado` não aparece nenhum nome")
        print("novo — e o mundo é tão ilegível quanto em `novo` — então nada dispara,")
        print(f"e o erro fica em {b['trocado']['erro']:.3f} com o campo se dizendo competente.")
        print()
        print("Isto responde à pergunta que estava em aberto desde crencas.py: o")
        print("mecanismo representa ignorância quando alguém a nomeia, e não a")
        print("DESCOBRE. Detectar `trocado` exigiria comparar predição com observação")
        print("— desajuste de modelo — e não consultar vocabulário.")
        print()
        print("Repare também no erro da célula `novo`: 0,500, ou seja, o agente não")
        print("aprendeu NADA. Recusar-se a atualizar diante de evidência ilegível é")
        print("honesto e é inútil sozinho — detecção sem a ação que resolve o")
        print("mecanismo é só paralisia bem fundamentada. É o experimento A2 que diz")
        print("se a ação existe e se ela se paga.")
    elif s_troc - s_conh > 0.3:
        print("O sinal sobrevive ao gerador externo: dispara onde o mundo é ilegível")
        print("e cala onde não é, INCLUSIVE sem nome novo para servir de pista.")
    elif s_conh > 0.3:
        print("Falso alarme alto na célula de controle: o sinal está disparando onde")
        print("o mundo é perfeitamente legível, o que anula as outras duas leituras.")
    return {'A': a, 'B': b}


if __name__ == "__main__":
    relatorio()
