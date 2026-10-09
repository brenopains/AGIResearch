"""
novaphys.crencas_bench — as seis provas do campo de crenças
============================================================

A ordem não é arbitrária. A prova 1 confere a aritmética contra a forma
fechada, e ela vem primeiro porque, se a conta estiver errada, todas as outras
notas são ruído bem formatado. Foi a lição mais cara desta sessão: consertar
antes de medir custou 18× uma vez, e uma bancada que media eco marcou 100%.

    1. ARITMÉTICA     bate com o produto analítico das razões?
    2. CALIBRAÇÃO     posterior 0,7 quer dizer verdade em 70% dos casos?
    3. CONVERGÊNCIA   com evidência consistente, fecha?
    4. ABERTURA       com evidência ambígua, NÃO fecha?
    5. RECUPERAÇÃO    depois de evidência enganosa, a verdade volta?
    6. INSUFICIÊNCIA  detecta representação furada SEM gritar em tudo?

A prova 6 é a que interessa e é a única de dois lados, porque a nota de um
lado só é fácil de vencer trapaceando: marcar "não sei" em tudo dá 100% de
detecção. O par (detecção, falso alarme) não tem esse atalho, e a terceira
célula — evidência fraca, vocabulário conhecido — separa "preciso de mais
dados" de "meu mecanismo está furado", que são diagnósticos com ações opostas.
"""

from __future__ import annotations

import math
import random
import statistics as st
from typing import Any, Dict, List, Tuple

from .crencas import (CampoDeCrencas, Hipotese, Observacao, VOCABULARIO,
                      _logit, _sig)


# ══════════════════════════════════════════════ 1. aritmética
def prova_aritmetica(n_seq: int = 200, semente: int = 0) -> Dict[str, Any]:
    """
    Com competência 1 (todas as relações conhecidas), o posterior tem de ser
    exatamente

        odds_final = odds_prior · Π LR_i

    Sem tolerância generosa: 1e-9. Um campo de crenças cuja conta desvia disto
    não é bayesiano, é uma heurística com nome bonito — e aí "posterior
    calibrado" deixa de ser uma alegação verificável.
    """
    rng = random.Random(semente)
    piores = []
    for _ in range(n_seq):
        prior = rng.uniform(0.05, 0.95)
        h = Hipotese(afirmacao=('a', 'b'), prior=prior)
        lrs = [math.exp(rng.uniform(-2.5, 2.5)) for _ in range(rng.randint(1, 25))]
        for lr in lrs:
            h.observar(Observacao(lr=lr, relacao='causa'))
        # forma fechada
        odds = (prior / (1 - prior)) * math.prod(lrs)
        esperado = odds / (1 + odds)
        piores.append(abs(h.posterior - esperado))
    return {"prova": "aritmética", "erro_max": max(piores),
            "erro_medio": st.mean(piores), "n": n_seq,
            "passou": max(piores) < 1e-9}


# ══════════════════════════════════════════════ 2. calibração
def prova_calibracao(n_casos: int = 4000, semente: int = 0,
                     n_bins: int = 10) -> Dict[str, Any]:
    """
    Mundo simulado com verdade conhecida: H vale com probabilidade `prior`, e
    cada observação concorda com a verdade com probabilidade q. O posterior
    correto é calculável, então a pergunta é se o CAMPO chega nele e se o
    número resultante quer dizer o que diz.

    ECE sobre o posterior, e não sobre um escore inventado — é a mesma medida
    que a bancada de calibração usa, para que os dois números sejam comparáveis.
    """
    rng = random.Random(semente)
    posts, verdades = [], []
    for _ in range(n_casos):
        prior = rng.choice([0.2, 0.35, 0.5, 0.65, 0.8])
        q = rng.choice([0.6, 0.7, 0.8, 0.9])
        verdade = rng.random() < prior
        h = Hipotese(afirmacao=('a', 'b'), prior=prior)
        for _ in range(rng.randint(1, 12)):
            concorda = rng.random() < q
            aponta_verdadeiro = concorda if verdade else (not concorda)
            lr = (q / (1 - q)) if aponta_verdadeiro else ((1 - q) / q)
            h.observar(Observacao(lr=lr, relacao='causa'))
        posts.append(h.posterior); verdades.append(verdade)

    N = len(posts)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [k for k in range(N) if lo < posts[k] <= hi or (i == 0 and posts[k] == 0)]
        if not idx:
            continue
        pm = sum(posts[k] for k in idx) / len(idx)
        vm = sum(1 for k in idx if verdades[k]) / len(idx)
        ece += (len(idx) / N) * abs(pm - vm)
    return {"prova": "calibração", "ece": ece, "n": N,
            "post_medio": st.mean(posts),
            "verdade_media": sum(verdades) / N, "passou": ece < 0.05}


# ══════════════════════════════════════════════ 3. convergência
def prova_convergencia(n_casos: int = 300, semente: int = 0,
                       q: float = 0.8, teto: int = 60) -> Dict[str, Any]:
    """Evidência consistente a favor: o posterior tem de cruzar 0,95 e ficar."""
    rng = random.Random(semente)
    passos, fechou = [], 0
    for _ in range(n_casos):
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        for k in range(1, teto + 1):
            h.observar(Observacao(lr=q / (1 - q), relacao='causa'))
            if h.posterior > 0.95:
                passos.append(k); fechou += 1
                break
    return {"prova": "convergência", "taxa": fechou / n_casos,
            "passos_mediano": st.median(passos) if passos else float('nan'),
            "passou": fechou / n_casos > 0.99}


# ══════════════════════════════════════════════ 4. abertura
def prova_abertura(n_casos: int = 300, semente: int = 0,
                   n_obs: int = 30) -> Dict[str, Any]:
    """
    Evidência ambígua (metade a favor, metade contra, mesma força): o posterior
    tem de FICAR perto do prior. Um sistema que deriva para uma ponta aqui está
    fabricando certeza a partir de ruído — que é o modo de falha mais perigoso,
    porque é indistinguível de aprendizado quando se olha só o final.
    """
    rng = random.Random(semente)
    desvios, suspensas = [], 0
    for _ in range(n_casos):
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        obs = [2.0] * (n_obs // 2) + [0.5] * (n_obs // 2)
        rng.shuffle(obs)
        for lr in obs:
            h.observar(Observacao(lr=lr, relacao='causa'))
        desvios.append(abs(h.posterior - 0.5))
        if h.estado == 'suspensa':
            suspensas += 1
    return {"prova": "abertura", "desvio_max": max(desvios),
            "desvio_medio": st.mean(desvios),
            "marcou_suspensa": suspensas / n_casos,
            "passou": max(desvios) < 0.02}


# ══════════════════════════════════════════════ 5. recuperação
def prova_recuperacao(n_casos: int = 300, semente: int = 0,
                      enganosas: int = 8, q: float = 0.8,
                      teto: int = 80) -> Dict[str, Any]:
    """
    H é verdadeira, mas as primeiras 8 observações apontam contra (uma amostra
    azarada, que acontece). Depois vem evidência honesta.

    A hipótese certa volta? Um campo que cristaliza cedo e congela falha aqui,
    e falha em silêncio: o posterior fica em 0,01 e nunca mais sobe. É a
    diferença entre um sistema que aprende e um que decidiu.
    """
    rng = random.Random(semente)
    voltou, custo, fundo = 0, [], []
    for _ in range(n_casos):
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        for _ in range(enganosas):
            h.observar(Observacao(lr=(1 - q) / q, relacao='causa'))
        fundo.append(h.posterior)
        for k in range(1, teto + 1):
            h.observar(Observacao(lr=q / (1 - q), relacao='causa'))
            if h.posterior > 0.95:
                voltou += 1; custo.append(k)
                break
    return {"prova": "recuperação", "taxa": voltou / n_casos,
            "fundo_mediano": st.median(fundo),
            "custo_mediano": st.median(custo) if custo else float('nan'),
            "passou": voltou / n_casos > 0.99}


# ══════════════════════════════════════════════ 6. insuficiência (dois lados)
def prova_insuficiencia(n_casos: int = 400, semente: int = 0) -> Dict[str, Any]:
    """
    Três células, e a nota é o conjunto delas. Nenhuma isolada decide nada.

      A) relação DESCONHECIDA + evidência forte
         → deve marcar SUSPENSA. É o caso `varia_junto_com`: evidência não
           resolve, porque o problema é o mecanismo.

      B) vocabulário CONHECIDO + evidência forte
         → NÃO pode marcar suspensa. Este é o teste de ceticismo excessivo, e
           é o que impede a solução trivial de marcar tudo como duvidoso.

      C) vocabulário CONHECIDO + evidência FRACA
         → deve marcar ABERTA, não suspensa. "Preciso de mais dados" e "meu
           mecanismo está furado" pedem ações opostas; um sistema que os
           confunde manda coletar dados quando devia trocar de modelo, ou o
           contrário.
    """
    rng = random.Random(semente)
    desconhecidas = ['varia_junto_com', 'acompanha', 'oscila_com', 'segue_de_perto']
    detectou = falso_alarme = confundiu_C = 0
    n = n_casos // 3

    for _ in range(n):  # A
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        rel = rng.choice(desconhecidas)
        for _ in range(12):
            h.observar(Observacao(lr=4.0, relacao=rel))
        if h.estado == 'suspensa':
            detectou += 1

    for _ in range(n):  # B
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        for _ in range(12):
            h.observar(Observacao(lr=4.0, relacao=rng.choice(['causa', 'leva_a'])))
        if h.estado == 'suspensa':
            falso_alarme += 1

    for _ in range(n):  # C
        h = Hipotese(afirmacao=('a', 'b'), prior=0.5)
        for _ in range(2):
            h.observar(Observacao(lr=1.15, relacao='causa'))
        if h.estado != 'aberta':
            confundiu_C += 1

    return {"prova": "insuficiência", "deteccao": detectou / n,
            "falso_alarme": falso_alarme / n, "confusao_dados_vs_modelo": confundiu_C / n,
            "n_por_celula": n,
            "passou": detectou / n > 0.9 and falso_alarme / n < 0.05
                      and confundiu_C / n < 0.05}


# ══════════════════════════════════════════════ relatório
PROVAS = [prova_aritmetica, prova_calibracao, prova_convergencia,
          prova_abertura, prova_recuperacao, prova_insuficiencia]


def relatorio() -> Dict[str, Any]:
    print("═" * 78)
    print("CAMPO DE CRENÇAS — o sistema representa corretamente a própria incerteza?")
    print("═" * 78)

    r = {p.__name__: p() for p in PROVAS}

    a = r['prova_aritmetica']
    print(f"1. aritmética vs. forma fechada .... erro máx {a['erro_max']:.2e}   "
          f"{'OK' if a['passou'] else 'FALHOU'}    ideal < 1e-9")
    if not a['passou']:
        print("   ↳ a conta está errada. As provas abaixo não significam nada até isto")
        print("     fechar: um posterior que não é posterior não pode ser calibrado.")

    c = r['prova_calibracao']
    print(f"2. calibração do posterior ......... ECE {c['ece']:.3f}          "
          f"{'OK' if c['passou'] else 'FALHOU'}    ideal < 0,05")
    print(f"   posterior médio {c['post_medio']:.3f} · verdade real {c['verdade_media']:.3f}")

    cv = r['prova_convergencia']
    print(f"3. convergência (evid. consistente)  {cv['taxa']:.0%} fecham em "
          f"{cv['passos_mediano']:.0f} obs   {'OK' if cv['passou'] else 'FALHOU'}")

    ab = r['prova_abertura']
    print(f"4. abertura (evid. ambígua) ........ desvio máx {ab['desvio_max']:.4f}  "
          f"{'OK' if ab['passou'] else 'FALHOU'}    ideal < 0,02")

    rc = r['prova_recuperacao']
    print(f"5. recuperação após 8 enganosas .... {rc['taxa']:.0%} voltam "
          f"(fundo {rc['fundo_mediano']:.3f}, custo {rc['custo_mediano']:.0f} obs)  "
          f"{'OK' if rc['passou'] else 'FALHOU'}")

    ins = r['prova_insuficiencia']
    print("─" * 78)
    print("6. INSUFICIÊNCIA DE REPRESENTAÇÃO — a prova de dois lados")
    print(f"   A) relação desconhecida  → detectou .......... {ins['deteccao']:.0%}   ideal 100%")
    print(f"   B) vocabulário conhecido → falso alarme ...... {ins['falso_alarme']:.0%}   ideal 0%")
    print(f"   C) evidência fraca       → confundiu c/ modelo {ins['confusao_dados_vs_modelo']:.0%}   ideal 0%")
    print("─" * 78)

    if not ins['passou']:
        print("A prova 6 não passou — e é a que decide, porque é a única sem atalho.")
    elif ins['deteccao'] > 0.9 and ins['falso_alarme'] < 0.05:
        print("O campo distingue AS TRÊS COISAS: falta de evidência, falta de")
        print("competência e falta de espaço de hipóteses. Detecta o mecanismo furado")
        print("sem virar cético profissional — que era a solução trivial e inútil.")
        print()
        print("O que isto NÃO é: nada aqui prova que a detecção funciona fora do")
        print("vocabulário que eu mesmo escrevi. `varia_junto_com` é desconhecida")
        print("porque não está numa lista que eu montei. Um mecanismo que só sabe")
        print("estranhar o que já foi catalogado como estranho ainda não é novidade")
        print("epistemológica — é uma lista. A prova que falta é evidência de que a")
        print("marcação sobrevive a um vocabulário gerado por outra pessoa.")
    return r


def demonstrar() -> None:
    """O decoder em ação: o campo se concentrando, e o estado sendo lido."""
    print("═" * 78)
    print("DECODIFICAÇÃO DO ESTADO DA REPRESENTAÇÃO")
    print("═" * 78)

    campo = CampoDeCrencas({('plasma', 'fusao'): 0.34,
                            ('campo', 'fusao'): 0.33,
                            ('pressao', 'fusao'): 0.33})
    passos = [(('plasma', 'fusao'), 2.5), (('campo', 'fusao'), 0.6),
              (('plasma', 'fusao'), 3.0), (('campo', 'fusao'), 0.5),
              (('plasma', 'fusao'), 3.0), (('pressao', 'fusao'), 0.4)]
    print("\ncampo inicial:")
    print(campo.desenhar())
    for i, (af, lr) in enumerate(passos, 1):
        campo.observar(af, Observacao(lr=lr, relacao='causa', experimento=380 + i))
        if i % 3 == 0:
            print(f"\napós {i} experimentos:")
            print(campo.desenhar())

    print("\n" + "─" * 78)
    h = campo.hipoteses[('plasma', 'fusao')]
    h.dominio = 'física de plasmas'
    h.dependencias = [('plasma', 'temperatura'), ('temperatura', 'fusao')]
    print(h.decodificar())

    print("\n" + "─" * 78)
    print("O MESMO, com uma relação fora do vocabulário:")
    h2 = Hipotese(afirmacao=('densidade', 'ignicao'), prior=0.5, dominio='física de plasmas')
    for _ in range(12):
        h2.observar(Observacao(lr=4.0, relacao='varia_junto_com'))
    print(h2.decodificar())
    print()
    print("Doze observações, TODAS a favor, e o posterior ficou exatamente no prior.")
    print("É o comportamento certo: evidência que chega por um canal que o sistema")
    print("não sabe ler não é evidência fraca, é evidência ilegível — e a resposta")
    print("honesta a doze dessas é continuar em 0,500, não subir para 0,850.")
    print()
    print("Compare com o motor simbólico no mesmo caso (bancada de calibração):")
    print("  confiança 0,850, acerto 0%. O número escalar não tinha onde guardar a")
    print("  diferença entre 'não sei se é verdade' e 'não sei ler a pergunta'.")


if __name__ == "__main__":
    relatorio()
    print()
    demonstrar()
