"""
novaphys.auditor — os três defeitos de 21/09, virados em checagem automática
==============================================================================

POR QUE ISTO EXISTE
---------------------
Em um único dia, três defeitos reais foram achados no meu próprio código por
revisão EXTERNA, nenhum por um portão deste projeto:

  1. `promessa` chamava `_estagio([p], treino)`, que sem MARCADOR na pilha
     devolve o treino INTACTO. A medida comparava o treino consigo mesmo e
     retornava 0,0 para todo candidato, com o experimento já lançado.
  2. O portão de `K1` calculava o nulo permutado, IMPRIMIA, e não o usava na
     condição. O rótulo podia disparar sem vantagem sobre o placebo.
  3. `lzma.compress` padrão carrega ~60 bytes de cabeçalho fixo, que DOMINA
     quando o delta encolhe — produzindo razões de −11,6 que eu quase li como
     sinal.

Pré-registro, controles, nulo e análise estavam todos corretos nos três casos.
**Portão nenhum testa se o INSTRUMENTO responde**; eles testam se o RESULTADO
passa. Estes três viram checagem.

O QUE ISTO É E O QUE NÃO É
----------------------------
É conhecimento de engenharia que EU estou codificando, não descoberta autônoma.
Pegar os três defeitos conhecidos demonstra que a auditoria FUNCIONA; não
demonstra que o sistema descobriu esses princípios. A pergunta interessante é a
quarta: rodar sobre os módulos já existentes e ver se acusa algo que ninguém
achou.

AS TRÊS CHECAGENS
-------------------
  testemunha_execucao(f, entradas)
      A função MEXE na entrada? Distingue "executou uma transformação" de
      "devolveu a entrada intacta". Identidade legítima não é defeito — por isso
      o veredito é INFORMATIVO ("nunca altera") e não acusatório.

  varia(medida, casos)
      A medida VARIA entre candidatos? Uma coluna de constantes não pode
      sustentar conclusão nenhuma, e é exatamente o que o defeito 1 produzia.

  portao_usa_controle(modulo)
      Estática sobre o AST: a condição que imprime um veredito POSITIVO
      referencia alguma variável de nulo/placebo? Esta é a que pode acusar
      módulos antigos.

  estavel_sob_recodificacao(medida, dado, recodifica)
      Medida de compressão sobrevive a uma recodificação REVERSÍVEL? Se o
      número muda muito quando só os rótulos trocam, ela lê serialização e não
      estrutura.

PRÉ-REGISTRO
--------------
  A0  as quatro checagens rodam e não quebram nos módulos do projeto.
  A1  as três versões quebradas de 21/09 são ACUSADAS, e as três corrigidas
      passam limpas -> a auditoria funciona
  A2  qualquer uma escapa -> a checagem é decorativa e não entra no ciclo

  E a pergunta que interessa, sem barra pré-fixada porque é exploratória:
  quantos módulos EXISTENTES o `portao_usa_controle` acusa? Cada acusação
  precisa ser conferida à mão — a estática não sabe se um `if` tem placebo
  embutido noutro nome.
"""

from __future__ import annotations

import ast
import io
import os
import statistics
from typing import Callable, Dict, List, Sequence, Tuple

# nomes que, aparecendo numa condição, contam como "o controle está no portão"
MARCAS_NULO = ("nulo", "null", "placebo", "plc", "perm", "embaralh", "shuffle",
               "controle", "ctrl", "_n", "sd_n", "acaso", "aleator", "sorteio")

# rótulos de veredito POSITIVO usados neste projeto
POSITIVOS = ("K1", "P1", "Q1", "A1", "L1", "E1", "F1", "G1", "H1", "R1", "T1",
             "X1", "Y1", "C1", "D1", "J1", "N1", "S1", "M1", "V1")


# ---------------------------------------------------------------- execução ---

def testemunha_execucao(f: Callable, entradas: Sequence) -> Dict:
    """
    A funcao MEXE na entrada? Este e o defeito 1 em forma de checagem.

    Identidade legitima NAO e defeito: o veredito e informativo.
    """
    iguais, difs, erros = 0, 0, 0
    for x in entradas:
        try:
            y = f(x)
        except Exception:
            erros += 1
            continue
        if y == x:
            iguais += 1
        else:
            difs += 1
    n = iguais + difs
    return {
        "checagem": "testemunha_execucao",
        "n": n, "inalterados": iguais, "alterados": difs, "erros": erros,
        "alerta": (n > 0 and difs == 0),
        "nota": ("NUNCA altera a entrada em %d casos — se deveria transformar, "
                 "o executor esta errado" % n) if (n > 0 and difs == 0)
                else "altera em %d de %d" % (difs, n),
    }


def varia(medida: Callable, casos: Sequence) -> Dict:
    """A medida VARIA entre candidatos? Coluna constante nao sustenta nada."""
    vs = []
    for c in casos:
        try:
            vs.append(float(medida(c)))
        except Exception:
            continue
    if not vs:
        return {"checagem": "varia", "alerta": True, "n": 0,
                "nota": "a medida falhou em TODOS os casos"}
    dp = statistics.pstdev(vs) if len(vs) > 1 else 0.0
    dist = len(set(round(v, 12) for v in vs))
    return {
        "checagem": "varia", "n": len(vs), "distintos": dist,
        "desvio": dp, "faixa": (min(vs), max(vs)),
        "alerta": dist <= 1,
        "nota": ("CONSTANTE em %d casos (valor %.6g) — nao mede nada"
                 % (len(vs), vs[0])) if dist <= 1
                else "%d valores distintos, desvio %.4g" % (dist, dp),
    }


# ------------------------------------------------------------ portao/nulo ---

def _nomes(no: ast.AST) -> List[str]:
    return [n.id for n in ast.walk(no) if isinstance(n, ast.Name)] + \
           [n.attr for n in ast.walk(no) if isinstance(n, ast.Attribute)]


def _imprime_positivo(corpo: Sequence[ast.stmt]) -> List[str]:
    achados = []
    for no in corpo:
        for s in ast.walk(no):
            if not (isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
                    and s.func.id == "print"):
                continue
            for a in s.args:
                txt = None
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    txt = a.value
                elif isinstance(a, ast.JoinedStr):
                    txt = "".join(v.value for v in a.values
                                  if isinstance(v, ast.Constant)
                                  and isinstance(v.value, str))
                elif isinstance(a, ast.BinOp):
                    txt = " ".join(v.value for v in ast.walk(a)
                                   if isinstance(v, ast.Constant)
                                   and isinstance(v.value, str))
                if not txt:
                    continue
                t = txt.strip()
                for rot in POSITIVOS:
                    if t.startswith(rot + " ") or t.startswith(rot + "."):
                        achados.append(rot)
    return achados


def portao_usa_controle(caminho: str) -> List[Dict]:
    """
    ESTATICA: a condicao que imprime um veredito POSITIVO referencia alguma
    variavel de nulo/placebo? Este e o defeito 2 em forma de checagem.

    Heuristica, e ela erra dos dois lados: um `if` pode carregar o placebo
    dentro de outro nome, e um nome com "nulo" pode nao ser o controle. Toda
    acusacao precisa de conferencia a mao.
    """
    try:
        arv = ast.parse(io.open(caminho, encoding="utf-8").read())
    except Exception as e:
        return [{"arquivo": caminho, "alerta": False, "nota": "nao parseou: %s" % e}]
    fora = []
    for no in ast.walk(arv):
        if not isinstance(no, ast.If):
            continue
        rots = _imprime_positivo(no.body)
        if not rots:
            continue
        nomes = " ".join(_nomes(no.test)).lower()
        tem = any(m in nomes for m in MARCAS_NULO)
        fora.append({
            "arquivo": os.path.basename(caminho), "linha": no.lineno,
            "veredito": sorted(set(rots)), "condicao": ast.unparse(no.test)[:110],
            "alerta": not tem,
            "nota": ("o portao do veredito positivo NAO referencia nulo/placebo"
                     if not tem else "controle presente na condicao"),
        })
    return fora


def portao_depende_do_nulo(portao, valores: Dict, chave_nulo: str,
                          faixa=(-1e6, 1e6), n: int = 11) -> Dict:
    """
    RUNTIME. Substitui a checagem estatica, que dava falso positivo porque o
    valor do controle pode entrar por INDICE (`l[4]`) ou por nome que nao parece
    nulo (`md`, `venceN`) — casar nome nao enxerga isso e nunca vai.

    E varre a partir de um ponto VIVO, nao do ponto base. Falso positivo isolado
    em revisao externa: um portao CORRETO como `ganho > 0 and ganho > nulo` era
    acusado quando `ganho` era negativo, porque a outra clausula ja impedia
    qualquer aprovacao e varrer o nulo nao podia mudar nada. Ausencia de
    dependencia e dependencia MASCARADA por outra clausula sao coisas
    diferentes.

    Procedimento: acha um ponto onde o portao APROVA (perturbando as outras
    variaveis); dali varre o controle e ve se o veredito consegue virar. Se
    nenhum ponto aprova, INCONCLUSIVO em vez de acusacao.
    """
    lo, hi = faixa
    grade = [lo + (hi - lo) * k / max(n - 1, 1) for k in range(n)]
    outras = [c for c in valores if c != chave_nulo]

    def ok(v):
        try:
            return bool(portao(v))
        except Exception:
            return None

    # 1. achar um ponto VIVO (portao aprova), comecando pelo proprio base
    vivo = None
    if ok(dict(valores)):
        vivo = dict(valores)
    else:
        import itertools
        # moderado primeiro: um ponto vivo com magnitude enorme torna a
        # varredura do controle inofensiva e reintroduz o falso positivo
        escala = [0.0, 1.0, -1.0, 10.0, -10.0, 1e3, -1e3, 1e9, -1e9]
        for k in range(1, min(len(outras), 2) + 1):
            for chs in itertools.combinations(outras, k):
                for vals in itertools.product(escala, repeat=k):
                    alt = dict(valores)
                    for c, x in zip(chs, vals):
                        alt[c] = x
                    if ok(alt):
                        vivo = alt
                        break
                if vivo:
                    break
            if vivo:
                break
    if vivo is None:
        return {"checagem": "portao_depende_do_nulo", "alerta": False,
                "inconclusivo": True,
                "nota": ("INCONCLUSIVO: o portao nao aprova em ponto algum, "
                         "entao a dependencia de `%s` nao e observavel aqui"
                         % chave_nulo)}

    # 2. dali, varrer o CONTROLE e ver se o veredito consegue virar.
    # A grade tem de cobrir a ESCALA DO PONTO VIVO: se ele tem ganho=1e3 e a
    # varredura vai so ate 50, o veredito nunca vira e o falso positivo volta.
    esc = max([abs(x) for x in vivo.values()
               if isinstance(x, (int, float))] or [1.0]) or 1.0
    grade = sorted(set(grade + [s_ * esc for s_ in
                                (-100, -10, -2, -1, 0, 1, 2, 10, 100)]))
    vistos = {True}
    for x in grade:
        alt = dict(vivo)
        alt[chave_nulo] = x
        r = ok(alt)
        if r is not None:
            vistos.add(r)
    depende = len(vistos) > 1
    return {
        "checagem": "portao_depende_do_nulo", "alerta": not depende,
        "inconclusivo": False,
        "nota": ("o veredito NAO muda varrendo `%s` a partir de um ponto que "
                 "APROVA — o placebo nao esta no portao" % chave_nulo)
                if not depende else "o veredito depende de `%s`" % chave_nulo,
    }


# ------------------------------------------------------------- codificacao ---

def estavel_sob_recodificacao(medida: Callable, dado, recodifica: Callable,
                              n: int = 8, tol: float = 0.10) -> Dict:
    """
    Medida de compressao sobrevive a recodificacao REVERSIVEL? Este e o
    defeito 3 (e o seu primo geral) em forma de checagem.
    """
    try:
        base = float(medida(dado))
    except Exception as e:
        return {"checagem": "recodificacao", "alerta": True,
                "nota": "a medida falhou: %s" % e}
    vs = []
    for k in range(n):
        try:
            vs.append(float(medida(recodifica(dado, k))))
        except Exception:
            continue
    if not vs:
        return {"checagem": "recodificacao", "alerta": True,
                "nota": "toda recodificacao falhou"}
    esc = max(abs(base), 1e-9)
    piora = max(abs(v - base) for v in vs) / esc
    return {
        "checagem": "recodificacao", "base": base, "n": len(vs),
        "desvio_relativo_max": piora, "alerta": piora > tol,
        "nota": ("varia %.1f%% sob recodificacao REVERSIVEL — le serializacao, "
                 "nao estrutura" % (100 * piora)) if piora > tol
                else "estavel (%.1f%%) sob recodificacao" % (100 * piora),
    }


# ------------------------------------------------------------------ varre ---

def varre_projeto(pasta: str = "novaphys") -> None:
    print("=" * 78)
    print("AUDITOR — portao de veredito positivo sem controle na condicao")
    print("=" * 78)
    print("  ESTATICA e HEURISTICA: toda acusacao precisa de conferencia.\n")
    acusados, total = [], 0
    for nome in sorted(os.listdir(pasta)):
        if not nome.endswith(".py"):
            continue
        for r in portao_usa_controle(os.path.join(pasta, nome)):
            if "veredito" not in r:
                continue
            total += 1
            if r["alerta"]:
                acusados.append(r)
    for r in acusados:
        print("  %-24s l.%-5d %-10s  %s"
              % (r["arquivo"], r["linha"], ",".join(r["veredito"]),
                 r["condicao"]))
    print("\n  %d portoes de veredito positivo encontrados, %d SEM controle na"
          % (total, len(acusados)))
    print("  condicao (%.0f%%)." % (100 * len(acusados) / max(total, 1)))


if __name__ == "__main__":
    varre_projeto()
