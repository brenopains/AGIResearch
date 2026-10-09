"""
novaphys.ciencia_real — descoberta de estrutura em física de verdade
=====================================================================

Todas as bancadas até aqui, inclusive a "externa", rodaram em mundos que EU
gerei. A avaliação externa reduziu a dependência da minha palavra (arquitetura
com hash, semente de fora, espaço mais largo), mas não a eliminou: o gerador
continuava sendo meu.

Aqui o mundo é o **Simulador de Realidade** — doze módulos de física resolvidos
por métodos numéricos estabelecidos e validados contra resultados conhecidos,
escritos meses antes de existir campo de crenças ou modelo semântico, e nunca
tocados por eles. Split-step Fourier, Barnes-Hut, Boris pusher, Bosch-Hale,
projeção de pressão de Stam.

Não é um dataset. É um ambiente que RESPONDE quando se mexe nele — que é o
requisito do ciclo, porque um corpus não tem como refutar uma hipótese:

    observação → hipótese → previsão → ERRO → revisão → experimento

A TAREFA: ESTRUTURA, NÃO RESPOSTA
----------------------------------
Para cada par (manopla, medição), o sistema tem de descobrir a FORMA da
dependência, não prever valores:

    H1  a medição cresce quando a manopla cresce
    H2  a medição decresce quando a manopla cresce

Uma observação = rodar a simulação em x₁ < x₂ e ver de que lado a medição foi.
Sob H1 a previsão é "subiu"; sob H2, "desceu". É exatamente o mecanismo de
`semantica.py`, sem uma linha de adaptação — o que é o ponto: se precisasse
adaptar, não estaria sendo testado.

O QUE ESTÁ FORA DO ESPAÇO, E DESTA VEZ NÃO FUI EU QUE PUS
-----------------------------------------------------------
Boa parte da física não é monotônica. Ressonância tem pico. Ignição tem
limiar. Tunelamento tem interferência. Nesses pares **nem H1 nem H2 é
verdadeira**, e o sistema deveria terminar SUSPENSO em vez de escolher.

Na avaliação externa esse regime existia porque eu sorteei `p_composicao=0,4`.
Aqui ele existe porque o modo de Chladni tem nó — e ninguém decidiu isso: é
consequência da equação de onda que já estava implementada.

A previsão, escrita antes de rodar: **a superconfiança vai reaparecer**, na
mesma faixa dos 75% medidos no gerador sintético. Se reaparecer, o achado
"pesa hipóteses, não constrói" deixa de ser propriedade do meu gerador e passa
a ser propriedade da arquitetura.

O LIMIAR DE "PLANA" É MEDIDO, NÃO ESCOLHIDO
--------------------------------------------
Para classificar a verdade preciso decidir quando uma variação é real e quando
é ruído. Escolher esse número a olho seria a porta dos fundos por onde eu
decidiria o resultado. Então ele é MEDIDO: repetições na mesma configuração com
sementes diferentes dão o piso de ruído, e só variação acima dele conta.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
import time
from typing import Any, Dict, List, Optional, Tuple

from .client import PhysicsWorld
from .semantica import ModeloSemantico


# ══════════════════════════════════════════════ leitura do simulador
def _medir(w: PhysicsWorld, modulo: str, manopla: str, valor: float,
           passos: int, semente: int) -> Dict[str, float]:
    """Uma configuração, um protocolo fixo, todas as medições de uma vez."""
    w.reset(modulo=modulo, semente=semente)
    o = w.agir(ajustar={manopla: valor}, passos=passos)
    out = {}
    for k, v in (o.get('escalares') or {}).items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = float(v)
    return out


def piso_de_ruido(w: PhysicsWorld, modulo: str, manopla: str, valor: float,
                  passos: int, n: int = 5) -> Dict[str, float]:
    """
    Quanto uma medição varia SEM mudar a manopla, só trocando a semente.

    Este número é o que separa "a manopla não afeta" de "afeta pouco", e ele é
    medido em vez de estipulado — se eu o escolhesse a olho, estaria escolhendo
    o resultado junto.
    """
    amostras: Dict[str, List[float]] = {}
    for s in range(n):
        for k, v in _medir(w, modulo, manopla, valor, passos, semente=1000 + s).items():
            amostras.setdefault(k, []).append(v)
    return {k: (st.pstdev(v) if len(v) > 1 else 0.0) for k, v in amostras.items()}


# ══════════════════════════════════════════════ verdade por varredura densa
def classificar(xs: List[float], ys: List[float], ruido: float) -> str:
    """
    'crescente' · 'decrescente' · 'plana' · 'nao-monotonica'

    A banda morta é 2× o desvio medido entre repetições: um passo só conta como
    subida ou descida se sair do ruído. Sem isso toda curva vira não-monotônica
    e a bancada mediria a variância do simulador, não a física.
    """
    banda = max(2.0 * ruido, 1e-12)
    passos = []
    for i in range(len(ys) - 1):
        d = ys[i + 1] - ys[i]
        passos.append(0 if abs(d) < banda else (1 if d > 0 else -1))
    subiu = sum(1 for p in passos if p > 0)
    desceu = sum(1 for p in passos if p < 0)
    amplitude = max(ys) - min(ys)
    if amplitude < banda or (subiu == 0 and desceu == 0):
        return 'plana'
    if subiu > 0 and desceu > 0:
        return 'nao-monotonica'
    return 'crescente' if subiu > 0 else 'decrescente'


def classificar_na_escala_do_agente(xs: List[float], ys: List[float], ruido: float,
                                    n_amostras: int = 400,
                                    semente: int = 99) -> str:
    """
    A mesma classificação, mas feita como o AGENTE olha: intervalos largos e
    aleatórios, não pontos vizinhos da grade.

    ISTO CORRIGE UM ERRO GRAVE DE ARNÊS, e o erro merece estar escrito aqui.
    `classificar` compara pontos adjacentes da varredura densa: uma ondulação
    minúscula, acima da banda de ruído, já a torna 'nao-monotonica'. Só que o
    agente nunca vê pontos adjacentes — ele sorteia (x₁,x₂) com pelo menos 15%
    da faixa entre eles, e a direção disso é decidida pelos extremos.

    Resultado: pares rotulados 'nao-monotonica' eram explicados por UMA única
    direção em 100% das observações do agente. Chamar isso de superconfiança
    era medir o desencontro entre duas definições de monotonicidade e atribuir
    a diferença ao sistema.

    Os números de superconfiança em física medidos antes desta função — 89% em
    `fusao`, 87% nos quatro módulos, 84% no subconjunto — estão RETIRADOS por
    esse motivo. É a mesma classe do erro do `degrau` na bancada de hipóteses:
    o arnês media outra coisa.

    Aqui um par só é 'nao-monotonica' se, na escala em que o agente mede, as
    duas direções aparecem de verdade — minoria acima de 15%.
    """
    if len(xs) < 2:
        return 'plana'
    banda = max(2.0 * ruido, 1e-12)
    lo, hi = xs[0], xs[-1]

    def em(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                t = (x - xs[i]) / max(1e-12, xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    rng = random.Random(semente)
    sobe = desce = 0
    for _ in range(n_amostras):
        x1, x2 = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
        if x2 - x1 < 0.15 * (hi - lo):
            x2 = min(hi, x1 + 0.35 * (hi - lo))
        d = em(x2) - em(x1)
        if abs(d) < banda:
            continue
        if d > 0:
            sobe += 1
        else:
            desce += 1
    tot = sobe + desce
    if tot < 0.2 * n_amostras:
        return 'plana'
    minoria = min(sobe, desce) / tot
    if minoria > 0.15:
        return 'nao-monotonica'
    return 'crescente' if sobe > desce else 'decrescente'


def verdade_do_modulo(w: PhysicsWorld, modulo: str, manoplas: List[Dict],
                      passos: int, n_pontos: int = 9) -> Dict[Tuple[str, str], str]:
    verdade: Dict[Tuple[str, str], str] = {}
    for man in manoplas:
        lo, hi = float(man['min']), float(man['max'])
        meio = 0.5 * (lo + hi)
        ruido = piso_de_ruido(w, modulo, man['chave'], meio, passos)
        xs = [lo + (hi - lo) * i / (n_pontos - 1) for i in range(n_pontos)]
        serie: Dict[str, List[float]] = {}
        for x in xs:
            for k, v in _medir(w, modulo, man['chave'], x, passos, semente=7).items():
                serie.setdefault(k, []).append(v)
        for k, ys in serie.items():
            if len(ys) == n_pontos:
                verdade[(man['chave'], k)] = classificar_na_escala_do_agente(
                    xs, ys, ruido.get(k, 0.0))
    return verdade


# ══════════════════════════════════════════════ o sistema, sem adaptação
def confiabilidade(w: PhysicsWorld, modulo: str, manopla: str, lo: float, hi: float,
                   passos: int, n_pares: int = 4,
                   rng: Optional[random.Random] = None) -> Dict[str, float]:
    """
    q — com que frequência uma observação de direção está certa.

    O detector de suficiência precisa do TETO alcançável, e ele não pode ser
    chutado: se eu fixasse q=0,8 e a medição real fosse 0,95, pares monotônicos
    perfeitamente explicados passariam por insuficientes.

    Estimado sem gabarito, por concordância entre réplicas: mede-se o MESMO par
    (x₁,x₂) com duas sementes e conta-se quantas vezes as duas apontam para o
    mesmo lado. Se cada leitura acerta com probabilidade q, duas leituras
    concordam com probabilidade q² + (1−q)², o que se inverte em

        q = ½·(1 + √(2·concordância − 1))

    É a mesma disciplina do piso de ruído: medir em vez de estipular.
    """
    rng = rng or random.Random(1234)
    conc: Dict[str, List[int]] = {}
    for _ in range(n_pares):
        x1, x2 = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
        if x2 - x1 < 0.15 * (hi - lo):
            x2 = min(hi, x1 + 0.35 * (hi - lo))
        leituras = []
        for s in (11, 22):
            a = _medir(w, modulo, manopla, x1, passos, semente=s)
            b = _medir(w, modulo, manopla, x2, passos, semente=s)
            leituras.append({k: b[k] > a[k] for k in set(a) & set(b)})
        for k in set(leituras[0]) & set(leituras[1]):
            conc.setdefault(k, []).append(int(leituras[0][k] == leituras[1][k]))
    out = {}
    for k, v in conc.items():
        c = sum(v) / len(v)
        arg = max(0.0, 2.0 * c - 1.0)
        out[k] = min(0.98, max(0.55, 0.5 * (1.0 + math.sqrt(arg))))
    return out


def suficiencia_do_par(modelo: ModeloSemantico, obs: List[Tuple[float, float, bool]],
                       q: float, z_critico: float = 2.5) -> Dict[str, Any]:
    """
    "A leitura que eu escolhi explica os dados que eu tenho?"

    NÃO altera o modelo de crenças: lê o estado dele e confronta com as
    observações que ele já viu. `estavel` quer dizer que ele decidiu por H1
    ("sobe"); `revisada`, por H2 ("desce"). Se a leitura escolhida acerta muito
    abaixo do teto q, nenhuma das duas está certa — e isso é diferente de
    "faltam dados", que é o diagnóstico que o posterior sozinho dá.

    Esta é a única peça nova. O mecanismo de crenças, o protocolo e as métricas
    anteriores ficam idênticos, para que a comparação com a rodada de 87% seja
    válida.
    """
    if modelo.estado == 'estavel':
        previsao = True
    elif modelo.estado == 'revisada':
        previsao = False
    else:
        return {'aplicavel': False, 'insuficiente': False, 'z': float('nan')}
    if not obs:
        return {'aplicavel': False, 'insuficiente': False, 'z': float('nan')}
    acertos = sum(1 for (_, _, subiu) in obs if subiu == previsao)
    a = acertos / len(obs)
    ep = math.sqrt(max(1e-12, q * (1 - q) / len(obs)))
    z = (a - q) / ep
    return {'aplicavel': True, 'insuficiente': z < -z_critico, 'z': z,
            'ajuste': a, 'teto': q}


def investigar(w: PhysicsWorld, modulo: str, manoplas: List[Dict], passos: int,
               n_obs: int = 14, semente: int = 0) -> Dict[Tuple[str, str], ModeloSemantico]:
    """
    O mecanismo de `semantica.py` aplicado tal como está. Cada par de rodadas
    (x₁ < x₂) devolve uma observação para TODAS as medições ao mesmo tempo, o
    que faz o custo ser por manopla e não por par.
    """
    rng = random.Random(semente)
    modelos: Dict[Tuple[str, str], ModeloSemantico] = {}
    # As MESMAS observações, guardadas para o detector de suficiência conferir
    # depois. Guardar não muda nada do que o modelo de crenças faz com elas —
    # a sequência de `observar` e os parâmetros são idênticos aos da rodada
    # anterior, que é a condição para a comparação valer.
    registro: Dict[Tuple[str, str], List[Tuple[float, float, bool]]] = {}
    for man in manoplas:
        lo, hi = float(man['min']), float(man['max'])
        for _ in range(n_obs):
            x1 = rng.uniform(lo, hi)
            x2 = rng.uniform(lo, hi)
            if x2 < x1:
                x1, x2 = x2, x1
            if x2 - x1 < 0.15 * (hi - lo):
                x2 = min(hi, x1 + 0.35 * (hi - lo))
            s = rng.randint(1, 9999)
            a = _medir(w, modulo, man['chave'], x1, passos, semente=s)
            b = _medir(w, modulo, man['chave'], x2, passos, semente=s)
            for k in set(a) & set(b):
                ch = (man['chave'], k)
                if ch not in modelos:
                    modelos[ch] = ModeloSemantico(tipo=f"{man['chave']}→{k}")
                # H1 prevê "subiu"; H2 prevê "desceu". q=0.8 é a mesma
                # confiabilidade por observação usada nas bancadas sintéticas.
                modelos[ch].observar(True, False, b[k] > a[k], q=0.8)
                registro.setdefault(ch, []).append((x1, x2, b[k] > a[k]))
    investigar.ultimo_registro = registro      # type: ignore[attr-defined]
    return modelos


# ══════════════════════════════════════════════ avaliação
def avaliar_modulo(w: PhysicsWorld, modulo: str, passos: int = 400,
                   max_manoplas: int = 6, n_obs: int = 14,
                   semente: int = 0, suficiencia: bool = False) -> Dict[str, Any]:
    cap = w.capacidades()
    m = next(x for x in cap['modulos'] if x['id'] == modulo)
    manoplas = [k for k in m['manoplas']
                if k.get('tipo') == 'range' and 'min' in k and 'max' in k][:max_manoplas]
    if not manoplas:
        return {'modulo': modulo, 'erro': 'sem manoplas contínuas'}

    verdade = verdade_do_modulo(w, modulo, manoplas, passos)
    modelos = investigar(w, modulo, manoplas, passos, n_obs=n_obs, semente=semente)
    registro = getattr(investigar, 'ultimo_registro', {})

    qs: Dict[str, float] = {}
    if suficiencia:
        rq = random.Random(semente + 555)
        for man in manoplas:
            for k, v in confiabilidade(w, modulo, man['chave'], float(man['min']),
                                       float(man['max']), passos, rng=rq).items():
                qs[f"{man['chave']}|{k}"] = v

    dentro_ok = dentro_tot = 0
    fora_conf = fora_tot = 0
    sinalizou_fora = sinalizou_dentro = 0
    detalhe: List[Tuple[str, str, str, str]] = []
    for ch, mod in modelos.items():
        v = verdade.get(ch)
        if v is None:
            continue
        sinal = ''
        if suficiencia:
            q = qs.get(f"{ch[0]}|{ch[1]}", 0.8)
            s = suficiencia_do_par(mod, registro.get(ch, []), q)
            if s['aplicavel'] and s['insuficiente']:
                sinal = 'INSUFICIENTE'
        if v in ('crescente', 'decrescente'):
            dentro_tot += 1
            certo = (v == 'crescente' and mod.estado == 'estavel') or \
                    (v == 'decrescente' and mod.estado == 'revisada')
            dentro_ok += int(certo)
            sinalizou_dentro += int(bool(sinal))
        else:                                   # plana ou não-monotônica
            fora_tot += 1
            if mod.estado in ('estavel', 'revisada'):
                fora_conf += 1
                sinalizou_fora += int(bool(sinal))
        detalhe.append((f"{ch[0]}→{ch[1]}", v, mod.estado, sinal))

    return {'modulo': modulo, 'n_pares': len(detalhe),
            'estrutura_recuperada': dentro_ok / max(1, dentro_tot),
            'n_dentro': dentro_tot,
            'superconfianca': (fora_conf / fora_tot) if fora_tot else None,
            'n_fora': fora_tot,
            'sinalizou': (sinalizou_fora / fora_conf) if fora_conf else None,
            'falso_alarme': sinalizou_dentro / max(1, dentro_tot),
            'n_conf': fora_conf, 'detalhe': detalhe}


def relatorio(modulos: Optional[List[str]] = None, passos: int = 400,
              n_obs: int = 14, semente: int = 0, verboso: bool = False,
              suficiencia: bool = False) -> Dict[str, Any]:
    w = PhysicsWorld()
    modulos = modulos or ['fusao', 'fluidos', 'ressonancia', 'vacuo']
    print("═" * 78)
    print("DESCOBERTA DE ESTRUTURA EM FÍSICA REAL")
    print("o mundo é o Simulador de Realidade — numérica validada, escrita antes")
    print("de existir campo de crenças, e nunca tocada por ele")
    print("═" * 78)
    cab = f"{'módulo':<14}{'pares':>7}{'estrutura recuperada':>22}{'superconfiança':>17}"
    if suficiencia:
        cab += f"{'sinalizou':>12}{'falso al.':>11}"
    print(cab)

    linhas, t0 = [], time.time()
    for mod in modulos:
        try:
            r = avaliar_modulo(w, mod, passos=passos, n_obs=n_obs, semente=semente,
                               suficiencia=suficiencia)
        except Exception as e:
            print(f"{mod:<14}  erro: {e}")
            continue
        if 'erro' in r:
            print(f"{mod:<14}  {r['erro']}")
            continue
        linhas.append(r)
        sc = r['superconfianca']
        sc_txt = f"{sc:.0%} ({r['n_fora']})" if sc is not None else "—"
        linha = (f"{mod:<14}{r['n_pares']:>7}"
                 f"{r['estrutura_recuperada']:>19.0%} ({r['n_dentro']:>2})"
                 f"{sc_txt:>17}")
        if suficiencia:
            sg = r.get('sinalizou')
            linha += (f"{(f'{sg:.0%}' if sg is not None else '—'):>12}"
                      f"{r.get('falso_alarme', 0):>10.0%}")
        print(linha)
        if verboso:
            for nome, v, est, sinal in r['detalhe'][:8]:
                print(f"      {nome:<26} verdade={v:<16} sistema={est:<11} {sinal}")

    print("─" * 78)
    d_ok = sum(l['estrutura_recuperada'] * l['n_dentro'] for l in linhas)
    d_n = sum(l['n_dentro'] for l in linhas)
    f_c = sum((l['superconfianca'] or 0) * l['n_fora'] for l in linhas)
    f_n = sum(l['n_fora'] for l in linhas)
    print(f"agregado: estrutura recuperada {d_ok/max(1,d_n):.0%} em {d_n} pares monotônicos")
    print(f"          superconfiança {f_c/max(1,f_n):.0%} em {f_n} pares fora do espaço")
    print(f"          {time.time()-t0:.0f} s")
    print()
    print("A PREVISÃO, escrita antes de rodar: a superconfiança reaparece na faixa")
    print("dos 75% medidos no gerador sintético. Se reaparecer aqui, o achado deixa")
    print("de ser propriedade do meu gerador e passa a ser da arquitetura.")
    print()
    sc = f_c / max(1, f_n)
    if f_n == 0:
        print("Nenhum par fora do espaço nestes módulos — sem isso o teste não")
        print("responde à pergunta. Rode com outros módulos.")
    elif sc > 0.5:
        print(f"CONFIRMADO em física real: {sc:.0%}. Ressonância tem pico, ignição tem")
        print("limiar, e ninguém decidiu isso — é consequência das equações que já")
        print("estavam implementadas. Diante de uma dependência que não é nem")
        print("crescente nem decrescente, o sistema escolhe um dos dois extremos e")
        print("confirma o estado.")
        print()
        print("O sistema pesa hipóteses; não as constrói. Agora medido fora do meu")
        print("gerador, contra numérica que existia antes da arquitetura.")
    else:
        print(f"Superconfiança {sc:.0%}, abaixo do previsto. Vale conferir a")
        print("classificação da verdade antes de comemorar: um limiar de 'plana' mal")
        print("calibrado transforma curva com pico em curva monotônica.")
    return {'linhas': linhas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--modulos', nargs='*', default=None)
    ap.add_argument('--passos', type=int, default=400)
    ap.add_argument('--obs', type=int, default=14)
    ap.add_argument('--semente', type=int, default=0)
    ap.add_argument('--verboso', action='store_true')
    ap.add_argument('--suficiencia', action='store_true')
    a = ap.parse_args()
    relatorio(modulos=a.modulos, passos=a.passos, n_obs=a.obs,
              semente=a.semente, verboso=a.verboso, suficiencia=a.suficiencia)
