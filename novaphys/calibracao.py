"""
novaphys.calibracao — a confiança reportada prevê o acerto?
============================================================

Todas as bancadas anteriores perguntam se o sistema acerta. Esta pergunta uma
coisa diferente e, para quem vai USAR o sistema, mais importante:

    quando ele diz 0,85, acerta mais vezes do que quando diz 0,60?

Se não, o número de confiança é decoração. Pior que decoração: é um convite a
confiar na hora errada, porque o leitor humano trata 0,85 como promessa. Um
sistema que erra e avisa é utilizável; um que erra com a mesma cara de sempre
não é, por melhor que seja a média de acerto.

Isto é a forma medível de "metacognição". Não a versão vaga — a versão que dá
número: o sistema sabe o que ele não sabe?

O QUE É MEDIDO
---------------
Três coisas diferentes, que costumam ser confundidas:

  DISCRIMINAÇÃO (AUC)   a confiança SEPARA acertos de erros? 0,5 = não separa
                        nada, é ruído; 1,0 = separa perfeitamente. É a
                        pergunta principal, e é invariante a escala: um
                        sistema sistematicamente pessimista pode ter AUC 1,0.

  CALIBRAÇÃO (ECE)      quando ele diz 0,75, acerta 75% das vezes? Mede o
                        desencontro entre o número dito e a frequência real.
                        Um sistema pode discriminar bem e ser mal calibrado
                        (a ordem está certa, a escala não).

  RESOLUÇÃO (desvio)    a confiança VARIA? Um sistema que responde sempre 0,75
                        tem ECE baixo se acerta ~75% das vezes — e não está
                        sabendo nada. Sem esta terceira medida, um valor fixo
                        e sortudo passa por calibração.

OS DOIS BALIZADORES
--------------------
Um AUC solto não diz nada, então cada rodada imprime dois números de
referência calculados sobre EXATAMENTE os mesmos itens:

    acaso    confiança sorteada ao acaso        → AUC ≈ 0,50
    oráculo  confiança = 1 se acertou, 0 se não → AUC = 1,00

O sistema fica entre os dois, e a posição dele nesse intervalo é a nota. Sem
os balizadores, "AUC 0,52" parece um número; com eles, é o acaso.

COMO SE OBTÊM ERROS
--------------------
Calibração não é medível sem variância: se o sistema acerta tudo, não há o que
separar. Então a bancada mistura famílias de dificuldade conhecida, incluindo
duas em que ele comprovadamente falha (conjunção, que exige aridade 2, e
cadeia acima do teto de corte). Isso não é armar contra ele — é a única
condição em que a pergunta faz sentido.
"""

from __future__ import annotations

import random
import statistics as st
from typing import Any, Callable, Dict, List, Tuple

from .emergencia import _carregar_reasoner, _E, _regra_atributo, _tok, _menciona
from .generalizacao import _negativa


def _alegacoes_conf(resultado) -> Tuple[List[str], float]:
    """As alegações e a confiança global que o motor anexa à conclusão."""
    import re
    out = []
    for linha in getattr(resultado, "trace", []):
        m = re.search(r"Inferred:\s*(.+?)\s*\(from rule", str(linha))
        if m:
            out.append(m.group(1).strip())
    conc = getattr(resultado, "conclusion", "") or ""
    if conc:
        out.append(conc)
    return out, float(getattr(resultado, "confidence", 0.0))


def _conc(label: str, alvo: str) -> List[Dict]:
    return [{"label": label, "position": [0.1] * 8, "connections": {"observado_em": [alvo]}}]


# ══════════════════════════════════════════════ famílias de dificuldade
# Cada uma devolve (item, esperado_positivo). Em `esperado_positivo=True` o
# acerto é AFIRMAR o par; em False, o acerto é NÃO afirmar.

def f_facil(rng) -> Tuple[Dict, bool]:
    """Composição de um salto. O sistema resolve — medido, 100%."""
    a, b, c = _tok(rng, "f", 3)
    p = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [_E(a, b), _E(b, c)],
          "causal_chains": [], "semantic_rules": [_regra_atributo(c, p)]}
    return {"familia": "fácil", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, p)}, True


def f_conjuncao(rng) -> Tuple[Dict, bool]:
    """Aridade 2. O sistema NÃO resolve — medido, 0%. Erro conhecido."""
    a, x, y = _tok(rng, "j", 3)
    z = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, x), "causal_edges": [_E(a, x), _E(a, y)],
          "causal_chains": [], "semantic_rules": [
              {"content": {"type": "conjunction",
                           "abstraction": {"requires": [x, y], "produces": z,
                                           "frequency": 0.95}}}]}
    return {"familia": "conjunção", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, z)}, True


def f_cadeia_longa(rng) -> Tuple[Dict, bool]:
    """
    Cadeia de 60 nós. A origem alcança 59 destinos e o motor corta em 40 —
    então o alvo mais distante pode cair fora, dependendo da ordem alfabética.
    Erro por truncamento, não por incapacidade: é justamente o tipo de falha
    em que um sistema honesto deveria baixar a confiança, porque ele SABE que
    cortou.
    """
    base = rng.randint(100, 400)
    nos = [f"L{base + i:04d}" for i in range(61)]
    kb = {"concepts": _conc(nos[0], nos[1]),
          "causal_edges": [_E(nos[i], nos[i + 1]) for i in range(60)],
          "causal_chains": [], "semantic_rules": []}
    return {"familia": "cadeia longa", "kb": kb, "conceitos": kb["concepts"],
            "par": (nos[0], nos[60])}, True


def f_bloqueio(rng) -> Tuple[Dict, bool]:
    """Premissa que derruba a conclusão. Acerto = não afirmar."""
    a, b, c = _tok(rng, "q", 3)
    p = f"prop{rng.randint(100, 999)}"
    kb = {"concepts": _conc(a, b), "causal_edges": [_E(a, b), _E(b, c)],
          "causal_chains": [], "semantic_rules": [_regra_atributo(c, p)],
          "bloqueios": [{"blocker": b, "blocked": p}]}
    return {"familia": "bloqueio", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, p)}, False


def f_nao_transitiva(rng) -> Tuple[Dict, bool]:
    """Causa + correlação. Acerto = não afirmar."""
    a, b, c = _tok(rng, "n", 3)
    kb = {"concepts": _conc(a, b),
          "causal_edges": [_E(a, b, "causa"), _E(b, c, "correlaciona_com")],
          "causal_chains": [], "semantic_rules": []}
    return {"familia": "não-transitiva", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, c)}, False


def f_vazio(rng) -> Tuple[Dict, bool]:
    """Nada se segue. Acerto = não afirmar."""
    a, b, c = _tok(rng, "v", 3)
    kb = {"concepts": _conc(a, b), "causal_edges": [], "causal_chains": [],
          "semantic_rules": []}
    return {"familia": "vazio", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, c)}, False


def f_relacao_desconhecida(rng) -> Tuple[Dict, bool]:
    """
    RETIDA — escrita DEPOIS de as constantes de confiança terem sido fixadas.

    As seis famílias acima estavam todas visíveis quando escolhi os valores
    0,95 / 0,70 / 0,45 / 0,10. Elas separam perfeitamente porque eu as vi
    separar — o teste de sensibilidade varia a PROPORÇÃO de cada uma, não
    introduz erro novo, então um AUC 1,000 lá dentro não prova discernimento,
    prova consistência do que eu já tinha ajustado.

    Esta família é o erro que os dois auto-relatos não cobrem. A relação
    `varia_junto_com` é tão não-transitiva quanto `correlaciona_com`, e não
    está na lista `NAO_TRANSITIVAS` do motor — vocabulário é aberto, a lista
    nunca vai estar completa. Então ele compõe, erra, e nada dispara: sem
    truncamento, sem regra ilegível. Confiança alta num erro.

    É exatamente a distância entre "sei que não sei" e "não sei que não sei",
    agora medida em vez de argumentada.
    """
    a, b, c = _tok(rng, "d", 3)
    kb = {"concepts": _conc(a, b),
          "causal_edges": [_E(a, b, "causa"), _E(b, c, "varia_junto_com")],
          "causal_chains": [], "semantic_rules": []}
    return {"familia": "relação desconhecida", "kb": kb, "conceitos": kb["concepts"],
            "par": (a, c)}, False


FAMILIAS: List[Tuple[str, Callable]] = [
    ("fácil", f_facil), ("conjunção", f_conjuncao), ("cadeia longa", f_cadeia_longa),
    ("bloqueio", f_bloqueio), ("não-transitiva", f_nao_transitiva), ("vazio", f_vazio),
    ("relação desconhecida", f_relacao_desconhecida),
]


# ══════════════════════════════════════════════ métricas
def auc(confs: List[float], acertos: List[bool]) -> float:
    """
    Área sob a curva ROC pela contagem de pares concordantes. Empates contam
    meio ponto — o que importa aqui, porque um sistema de confiança constante
    empata em TUDO e tem de sair exatamente 0,50, não 1,00.
    """
    pos = [c for c, a in zip(confs, acertos) if a]
    neg = [c for c, a in zip(confs, acertos) if not a]
    if not pos or not neg:
        return float("nan")
    ganhos = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return ganhos / (len(pos) * len(neg))


def ece(confs: List[float], acertos: List[bool], n_bins: int = 10) -> float:
    """Erro de calibração esperado: |confiança média − acerto real| por faixa."""
    N = len(confs)
    if N == 0:
        return float("nan")
    total = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [k for k in range(N) if (lo < confs[k] <= hi) or (i == 0 and confs[k] == 0)]
        if not idx:
            continue
        cm = sum(confs[k] for k in idx) / len(idx)
        am = sum(1 for k in idx if acertos[k]) / len(idx)
        total += (len(idx) / N) * abs(cm - am)
    return total


# ══════════════════════════════════════════════ medição
def avaliar(n_por_familia: int = 40, semente: int = 0,
            pesos: Dict[str, float] | None = None) -> Dict[str, Any]:
    SR = _carregar_reasoner()
    sr = SR()
    rng = random.Random(semente)
    confs: List[float] = []
    acertos: List[bool] = []
    por_familia: Dict[str, List[Tuple[float, bool]]] = {}

    for nome, gerar in FAMILIAS:
        por_familia[nome] = []
        n_esta = int(round(n_por_familia * (pesos or {}).get(nome, 1.0)))
        for _ in range(n_esta):
            it, positivo = gerar(rng)
            al, conf = _alegacoes_conf(sr.reason(it["conceitos"], it["kb"]))
            afirmou = any(_menciona(x, it["par"]) and not _negativa(x) for x in al)
            acertou = afirmou if positivo else (not afirmou)
            confs.append(conf); acertos.append(acertou)
            por_familia[nome].append((conf, acertou))

    rng_b = random.Random(semente + 7777)
    acaso = [rng_b.random() for _ in confs]
    oraculo = [1.0 if a else 0.0 for a in acertos]

    return {
        "n": len(confs),
        "acerto_global": sum(acertos) / max(1, len(acertos)),
        "conf_media": st.mean(confs) if confs else float("nan"),
        "conf_desvio": st.pstdev(confs) if len(confs) > 1 else 0.0,
        "conf_distintas": sorted(set(round(c, 4) for c in confs)),
        "auc": auc(confs, acertos),
        "auc_acaso": auc(acaso, acertos),
        "auc_oraculo": auc(oraculo, acertos),
        "ece": ece(confs, acertos),
        "conf_quando_certo": st.mean([c for c, a in zip(confs, acertos) if a])
                             if any(acertos) else float("nan"),
        "conf_quando_errado": st.mean([c for c, a in zip(confs, acertos) if not a])
                              if not all(acertos) else float("nan"),
        "por_familia": {k: {"acerto": sum(1 for _, a in v if a) / max(1, len(v)),
                            "conf": st.mean([c for c, _ in v]) if v else float("nan")}
                        for k, v in por_familia.items()},
    }


def relatorio(n_por_familia: int = 40, sementes: Tuple[int, ...] = (0, 1, 2)) -> Dict[str, Any]:
    linhas = [avaliar(n_por_familia, s) for s in sementes]
    m = lambda k: st.mean([l[k] for l in linhas])

    print("═" * 78)
    print("BANCADA DE CALIBRAÇÃO — a confiança reportada prevê o acerto?")
    print(f"{n_por_familia} itens × {len(FAMILIAS)} famílias × {len(sementes)} sementes "
          f"= {linhas[0]['n'] * len(sementes)} decisões")
    print("═" * 78)

    print(f"{'família':<18}{'acerto':>10}{'confiança média':>18}")
    for nome, _ in FAMILIAS:
        ac = st.mean([l["por_familia"][nome]["acerto"] for l in linhas])
        cf = st.mean([l["por_familia"][nome]["conf"] for l in linhas])
        print(f"{nome:<18}{ac:>9.0%}{cf:>18.3f}")

    print("─" * 78)
    print(f"acerto global .................. {m('acerto_global'):.0%}")
    print(f"confiança quando ACERTOU ....... {m('conf_quando_certo'):.3f}")
    print(f"confiança quando ERROU ......... {m('conf_quando_errado'):.3f}")
    print(f"desvio-padrão da confiança ..... {m('conf_desvio'):.3f}")
    vals = sorted(set(v for l in linhas for v in l["conf_distintas"]))
    print(f"valores distintos observados ... {len(vals)}  {vals[:8]}"
          f"{' …' if len(vals) > 8 else ''}")

    print("─" * 78)
    print(f"{'':<22}{'AUC':>8}   (0,50 = ruído · 1,00 = perfeito)")
    print(f"{'  acaso (balizador)':<22}{m('auc_acaso'):>8.3f}")
    print(f"{'  SISTEMA':<22}{m('auc'):>8.3f}")
    print(f"{'  oráculo (balizador)':<22}{m('auc_oraculo'):>8.3f}")
    print(f"\nECE (erro de calibração) ....... {m('ece'):.3f}   ideal 0,000")

    print("─" * 78)
    a, a0 = m("auc"), m("auc_acaso")
    dif = m("conf_quando_certo") - m("conf_quando_errado")
    if abs(a - 0.5) < 0.05:
        print("A CONFIANÇA NÃO CARREGA INFORMAÇÃO. AUC no nível do acaso: o número")
        print("que o sistema anexa à conclusão não separa o que ele acertou do que")
        print("ele errou. Quem ler 0,85 e 0,60 está lendo a mesma coisa.")
        print()
        print("Isto não é um defeito de ajuste — é estrutural. As confianças são")
        print("constantes escritas no código (0,75 para transitividade, a frequência")
        print("da regra para atributo) e nenhuma delas olha para o que aconteceu na")
        print("derivação: quantos saltos, se o fecho foi truncado, se havia ruído,")
        print("se houve caminho único ou vários. O sistema não tem como saber o que")
        print("não sabe porque nada no cálculo mede isso.")
    elif a < a0 + 0.05:
        print(f"AUC {a:.3f} contra {a0:.3f} do acaso: a confiança mal se distingue de")
        print("um número sorteado. Não é utilizável como sinal de quando confiar.")
    elif dif < 0:
        print("SINAL INVERTIDO: a confiança é MAIOR nos erros que nos acertos. Pior")
        print("que ausência de informação — é informação enganosa.")
    else:
        print(f"A confiança discrimina: AUC {a:.3f} contra {a0:.3f} do acaso.")
        print(f"Diferença entre acerto e erro: {dif:+.3f}. Olhe o ECE antes de")
        print("confiar na ESCALA: discriminar a ordem certa não é dizer o número certo.")
    return {"linhas": linhas}


def sensibilidade(n_por_familia: int = 40,
                  sementes: Tuple[int, ...] = (0, 1, 2)) -> Dict[str, Any]:
    """
    O AUC é propriedade do SISTEMA ou da mistura de itens que eu escolhi?

    Esta pergunta não é opcional. A confiança 0,200 é o valor único de "não
    derivei nada", e ela aparece em duas situações opostas: no silêncio CORRETO
    (não há o que derivar) e na FALHA (havia, e ele não conseguiu). Se o AUC
    subir e descer conforme eu mudo a proporção entre essas duas situações,
    então ele mede a minha escolha de itens, não o discernimento do sistema —
    e qualquer nota reportada seria minha, não dele.

    A checagem é direta: repesar as famílias e ver o intervalo. Se o intervalo
    cruza 0,50, o veredito é ausência de informação, e o valor da mistura
    uniforme é coincidência.
    """
    perfis = {
        "uniforme": {},
        "mais silêncio correto": {"não-transitiva": 3.0, "vazio": 3.0},
        "mais silêncio errado": {"conjunção": 3.0, "cadeia longa": 3.0},
        "só famílias que derivam": {"não-transitiva": 0.0, "vazio": 0.0},
        "só famílias de silêncio": {"fácil": 0.0, "bloqueio": 0.0, "cadeia longa": 0.0},
    }
    print("═" * 78)
    print("SENSIBILIDADE — o AUC é do sistema ou da mistura de itens?")
    print("═" * 78)
    print(f"{'perfil de mistura':<28}{'acerto':>9}{'AUC':>9}{'acaso':>9}")
    aucs = []
    for nome, pesos in perfis.items():
        ls = [avaliar(n_por_familia, s, pesos=pesos) for s in sementes]
        a = st.mean([l["auc"] for l in ls if l["auc"] == l["auc"]])
        a0 = st.mean([l["auc_acaso"] for l in ls if l["auc_acaso"] == l["auc_acaso"]])
        ac = st.mean([l["acerto_global"] for l in ls])
        aucs.append(a)
        print(f"{nome:<28}{ac:>8.0%}{a:>9.3f}{a0:>9.3f}")

    lo, hi = min(aucs), max(aucs)
    print("─" * 78)
    print(f"intervalo do AUC entre perfis: {lo:.3f} – {hi:.3f}  (amplitude {hi-lo:.3f})")
    if lo < 0.5 < hi:
        print()
        print("VEREDITO: o AUC CRUZA 0,50 conforme a mistura. Ele não mede o sistema,")
        print("mede quantos itens de cada tipo eu coloquei na prova. O 0,625 do")
        print("relatório uniforme é coincidência da proporção que escolhi, e reportá-lo")
        print("como 'a confiança discrimina' seria eu me dando nota.")
        print()
        print("A causa é concreta: 0,200 é o valor único de 'não derivei nada', e não")
        print("distingue 'não havia o que derivar' de 'havia e eu falhei'. São os dois")
        print("casos que mais importam para quem depende da resposta, e o sistema os")
        print("reporta com o mesmo número.")
    elif hi - lo > 0.15:
        print()
        print("O AUC varia demais entre misturas para ser lido como nota do sistema.")
        print("Reporte o intervalo, nunca o valor de uma mistura só.")
    else:
        print()
        print("O AUC é estável entre misturas — aí sim é propriedade do sistema.")
    return {"aucs": aucs, "intervalo": (lo, hi)}


if __name__ == "__main__":
    relatorio()
    print()
    sensibilidade()
