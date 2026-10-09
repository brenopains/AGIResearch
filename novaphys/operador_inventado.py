"""
novaphys.operador_inventado — o LLM gera OPERADORES, não codecs
=================================================================

A CORRECAO DE NIVEL
---------------------
Tres nulos seguidos com o LLM propondo CODECS: biblioteca de composicoes (+0),
reordenacao (40/400 nos quatro bracos), e o laco com o 3B escrevendo codec
(0 adotados em 30 tarefas). A razao e de nivel, nao de modelo:

    um CODEC     ~40 linhas, e precisa estar CERTO SOBRE A TAREFA
    um OPERADOR   ~3 linhas, e so precisa ser uma FUNCAO BEM DEFINIDA

O operador nao carrega a obrigacao em que o modelo pequeno falhou. Ele gera
FORMA; quem decide VALOR e o portao de bits. Essa e a divisao de trabalho certa.

E A INVERSAO QUE IMPORTA
--------------------------
    operadores relacionais    domInio-INDEPENDENTES e FINITOS. Posto, unicidade,
                              paridade, extremalidade, contagem, monotonicidade,
                              periodicidade, correlacao... talvez 30 a 60 em toda
                              a matematica. Geram-se UMA VEZ e valem sempre.
    observaveis base          ESSES sao especificos do domInio (tam/cor/buracos
                              na grade; energia/momento na fisica; marcador e
                              tempo-ate-evento na medicina).

Logo o investimento em operadores AMORTIZA em todos os dominios futuros, e o
trabalho recorrente por domInio e o dos observaveis — o inverso do que parece.

O GARGALO REAL NAO E GERACAO, E SELECAO
-----------------------------------------
Ja medido: 11 observaveis x 7 operadores = 85 propriedades, e o pre-registro de
`propriedade_inventada.py` preve que 85 JA queimam o teto de avaliacoes. Se o LLM
gerar 50 operadores, viram ~550 propriedades e a busca morre. Por isso este
modulo mede as DUAS coisas separadamente:

    gerou funcao bem definida?   (cerca + prova de vida)
    a funcao PAGA?               (portao de bits, orcamento casado, placebo)

CONTRATO DE UM OPERADOR
-------------------------
    def op(valores, meu):
        '''valores: lista com o observavel de TODOS os objetos da grade.
           meu: o observavel DESTE objeto.
           devolve um inteiro pequeno, deterministico.'''

Nada de grade nisso. E por isso que o mesmo operador serve a qualquer dominio que
tenha objetos com observaveis — e e o que este modulo testa ao final, aplicando
os operadores aceitos a um segundo conjunto de observaveis.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  O1  GERA: >= 6 operadores passam na cerca e na prova de vida (bem definidos,
      deterministicos, inteiros pequenos)
  O2  NAO SAO OS MEUS: >= 3 dos aceitos nao sao equivalentes a nenhum dos meus 7
      (equivalencia testada por assinatura: mesma saida em 200 entradas)
  O3  PAGA: com os operadores aceitos, >= +2 tarefas sobre a base, teto casado
  O4  PLACEBO: o mesmo numero de operadores ARBITRARIOS ganha <= 0
PREVISAO: O1 passa (operador e curto, e o 3B escreve codigo curto bem). O2 passa
com 3 a 6 — aposto que ele gera mediana, moda, desvio, "e vizinho do maximo".
O3 FALHA — aposto +0 ou +1, porque o gargalo e selecao e mais operadores pioram
o orcamento. O4 passa. Se O1 e O2 passarem e O3 falhar, a conclusao
pre-registrada e: a GERACAO de linguagem esta resolvida e o proximo gap e a
SELECAO — e ai o prior em bits deixa de ser opcional.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import types

from novaphys.alfabeto_aberto import extrai, pede, varre
import novaphys.alfabeto_aberto as AA

PASTA = os.path.join(os.path.dirname(__file__), "operadores")

CONTRATO_OP = '''\
Um OPERADOR RELACIONAL resume a posicao de um objeto em relacao aos outros. O
contrato e exatamente este, e nao tem nada de dominio especifico:

    def nome_do_operador(valores, meu):
        """valores: lista de numeros — o mesmo observavel medido em TODOS os
                    objetos (ex.: o tamanho de cada objeto da cena).
           meu:     o numero deste objeto (esta em `valores`).
           devolve: um INTEIRO pequeno (0 a ~20), deterministico."""
        ...

Exemplos reais, que ja existem no sistema:

    def posto(valores, meu):
        return sorted(set(valores), reverse=True).index(meu)

    def unico(valores, meu):
        return int(valores.count(meu) == 1)

    def difmax(valores, meu):
        return max(valores) - meu

Regras duras:
  - so `def`, uma funcao, sem classe e sem estado global;
  - nada de import fora de math/statistics; nada de open/exec/eval/random;
  - deterministico: a mesma entrada sempre da a mesma saida;
  - devolve inteiro pequeno, nunca float nem string nem lista;
  - tem de ser RELACIONAL: usar `valores`, nao so `meu`.
'''

MEUS = ("posto", "unico", "maximo", "minimo", "paridade", "conta", "difmax")


def _assinatura_src(nome):
    """Assinatura em 200 entradas fixas MAIS as tres invariancias que definem o
    que e um operador relacional. A primeira versao deste teste nao tinha as
    invariancias e por isso aceitou lixo com cara de acerto:

        def centroide(valores, meu):  return int(mean(valores))   # ignora `meu`
        def meio(valores, meu):       valores.sort(); ...         # ignora e MUTA
        def ordem(valores, meu):      return valores.index(meu)   # vaza a ordem
                                                                  # de enumeracao

    As duas primeiras devolvem o mesmo numero para TODOS os objetos da cena —
    zero informacao como propriedade de objeto. A terceira devolve o indice na
    ordem em que EU enumerei os objetos: vazamento meu, nao estrutura do mundo.
    As tres passaram porque a assinatura variava com `valores`.

    O criterio certo tem nome exato: um operador relacional opera sobre um
    CONJUNTO de objetos, entao permutar `valores` NAO pode mudar a saida. E o
    mesmo princIpio dos nulos permutados que este projeto usa em tudo — e eu nao
    o havia aplicado justamente onde ele e a definicao do objeto."""
    return f'''
import random
_r = random.Random(7)
_casos = []
for _ in range(200):
    n = _r.randint(2, 6)
    v = [_r.randint(0, 9) for _ in range(n)]
    _casos.append((v, _r.choice(v)))
_s, _dep, _perm, _muta, _viaveis = [], 0, 0, 0, 0
for _v, _m in _casos:
    _antes = list(_v)
    try:
        _x = {nome}(list(_v), _m)
    except Exception:
        _s.append("E")
        continue
    if not isinstance(_x, (int, bool)):
        _s.append("X")
        continue
    _s.append(str(int(_x)))
    # 1. depende de `meu`? (com `valores` FIXO)
    if len(set(_v)) > 1:
        _viaveis += 1
        _o = set()
        for _mm in set(_v):
            try:
                _o.add({nome}(list(_v), _mm))
            except Exception:
                pass
        if len(_o) > 1:
            _dep += 1
    # 2. invariante a permutacao de `valores`?
    _vv = list(_v)
    random.Random(11).shuffle(_vv)
    if _vv != list(_v):
        try:
            if {nome}(_vv, _m) != _x:
                _perm += 1
        except Exception:
            _perm += 1
    # 3. muta a entrada?
    _arg = list(_v)
    try:
        {nome}(_arg, _m)
    except Exception:
        pass
    if _arg != _antes:
        _muta += 1
print("ASSINATURA " + ",".join(_s))
print(f"INVAR dep={{_dep}}/{{_viaveis}} perm={{_perm}} muta={{_muta}}")
'''


def prova_e_assinatura(fonte, nome, segundos=15):
    """Roda em SUBPROCESSO. Devolve (ok, assinatura ou motivo)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write("import math, statistics\n" + fonte + _assinatura_src(nome))
        cam = f.name
    try:
        s = subprocess.run([sys.executable, cam], capture_output=True,
                           text=True, timeout=segundos)
        ass = inv = None
        for l in s.stdout.splitlines():
            if l.startswith("ASSINATURA "):
                ass = l[len("ASSINATURA "):]
            elif l.startswith("INVAR "):
                inv = l[len("INVAR "):]
        if ass is None:
            return False, (s.stderr or "sem assinatura").strip().splitlines()[-1:][0]
        if ass.count("E") > 20 or ass.count("X") > 0:
            return False, f"nao e inteiro ou explode ({ass.count('E')} erros)"
        if len(set(ass.split(","))) < 2:
            return False, "constante — nao carrega informacao"
        if inv:
            d = dict(x.split("=") for x in inv.split())
            dep, viav = (d["dep"].split("/") + ["0"])[:2]
            if int(d["muta"]) > 0:
                return False, f"MUTA a entrada em {d['muta']} casos"
            if int(d["perm"]) > 0:
                return False, (f"vaza a ORDEM DE ENUMERACAO: permutar `valores` "
                               f"muda a saida em {d['perm']} casos")
            if int(viav) and int(dep) < int(viav) // 4:
                return False, (f"ignora `meu` — mesma saida para objetos "
                               f"diferentes ({dep}/{viav} casos dependem)")
        return True, ass
    except subprocess.TimeoutExpired:
        return False, "estourou o relogio"
    finally:
        try:
            os.unlink(cam)
        except OSError:
            pass


def varre_op(fonte):
    """Cerca do operador: uma funcao, sem classe, sem estado, sem sorteio."""
    try:
        arv = ast.parse(fonte)
    except SyntaxError as e:
        return [f"sintaxe: {e}"]
    ruim = []
    permit = {"math", "statistics"}
    proib = {"open", "exec", "eval", "compile", "__import__", "input",
             "globals", "locals", "vars", "getattr", "setattr"}
    for no in ast.walk(arv):
        if isinstance(no, ast.Import):
            for al in no.names:
                if al.name.split(".")[0] not in permit:
                    ruim.append(f"import proibido: {al.name}")
        elif isinstance(no, ast.ImportFrom):
            if (no.module or "").split(".")[0] not in permit:
                ruim.append(f"import proibido: from {no.module}")
        elif isinstance(no, ast.Name) and no.id in proib:
            ruim.append(f"nome proibido: {no.id}")
        elif isinstance(no, (ast.Global, ast.Nonlocal)):
            ruim.append("estado global")
        elif isinstance(no, ast.ClassDef):
            ruim.append("classe — esperava so uma funcao")
    fs = [n for n in arv.body if isinstance(n, ast.FunctionDef)]
    if len(fs) != 1:
        ruim.append(f"esperava 1 funcao, achei {len(fs)}")
    else:
        args = [a.arg for a in fs[0].args.args]
        if len(args) != 2:
            ruim.append(f"esperava 2 argumentos, achei {args}")
        fonte_f = ast.unparse(fs[0])
        if args and args[0] not in fonte_f.replace(f"def {fs[0].name}", ""):
            ruim.append("nao usa `valores` — nao e relacional")
    return ruim


def assinaturas_dos_meus():
    """As assinaturas dos meus 7, para saber se o LLM so reinventou os meus."""
    from novaphys.propriedade_inventada import _derivada  # noqa: F401
    src = {
        "posto": "def posto(valores, meu):\n    return sorted(set(valores), reverse=True).index(meu)\n",
        "unico": "def unico(valores, meu):\n    return int(valores.count(meu) == 1)\n",
        "maximo": "def maximo(valores, meu):\n    return int(meu == max(valores))\n",
        "minimo": "def minimo(valores, meu):\n    return int(meu == min(valores))\n",
        "paridade": "def paridade(valores, meu):\n    return meu % 2\n",
        "conta": "def conta(valores, meu):\n    return valores.count(meu)\n",
        "difmax": "def difmax(valores, meu):\n    return max(valores) - meu\n",
    }
    fora = {}
    for n, s in src.items():
        ok, a = prova_e_assinatura(s, n)
        if ok:
            fora[n] = a
    return fora


def gera(n_pedidos, tentativas=2, verboso=True):
    """Pede operadores relacionais novos, um por pedido, com a historia do que
    ja foi aceito para nao repetir."""
    aceitos, vistas = [], set()
    for k in range(n_pedidos):
        ja = ", ".join(n for n, _s, _a in aceitos) or "(nenhum ainda)"
        p = (CONTRATO_OP
             + f"\n\nJa existem estes: posto, unico, maximo, minimo, paridade, "
               f"conta, difmax.\nE nesta rodada voce ja escreveu: {ja}.\n\n"
               "Escreva UM operador relacional NOVO, diferente de todos esses, "
               "que resuma de forma util a posicao de um objeto entre os outros. "
               "So o bloco ```python com a funcao.")
        for t in range(tentativas):
            try:
                fonte = extrai(pede(p, AA.TOKENS, 0.9, k * 10 + t))
            except Exception as e:
                if verboso:
                    print(f"  pedido {k+1}: rede {e}", flush=True)
                continue
            motivos = varre_op(fonte)
            if motivos:
                if verboso:
                    print(f"  pedido {k+1}.{t+1}: cerca — {motivos[0][:64]}",
                          flush=True)
                continue
            nome = re.search(r"def\s+(\w+)", fonte).group(1)
            ok, nota = prova_e_assinatura(fonte, nome)
            if not ok:
                if verboso:
                    print(f"  pedido {k+1}.{t+1}: {nome} — {nota[:64]}", flush=True)
                continue
            if nota in vistas:
                igual = next(n for n, _f, a in aceitos if a == nota)
                if verboso:
                    print(f"  pedido {k+1}.{t+1}: {nome} e IGUAL a {igual} "
                          f"(mesma assinatura)", flush=True)
                continue
            aceitos.append((nome, fonte, nota))
            vistas.add(nota)
            if verboso:
                print(f"  pedido {k+1}: ACEITO {nome}", flush=True)
            break
    return aceitos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pedidos", type=int, default=12)
    ap.add_argument("--tentativas", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=900)
    a = ap.parse_args()
    AA.TOKENS = a.tokens
    os.makedirs(PASTA, exist_ok=True)

    print("=" * 78)
    print(f"OPERADOR INVENTADO — {a.pedidos} pedidos ao modelo local")
    print("=" * 78)
    print("  assinando os MEUS 7, para saber o que e novo de verdade...")
    meus = assinaturas_dos_meus()
    print(f"  {len(meus)}/7 assinados\n")

    t0 = time.time()
    aceitos = gera(a.pedidos, a.tentativas)
    novos = [(n, f, s) for n, f, s in aceitos if s not in meus.values()]
    repetidos = [n for n, _f, s in aceitos if s in meus.values()]

    for k, (nome, fonte, _s) in enumerate(aceitos):
        with open(os.path.join(PASTA, f"{k:02d}_{nome}.py"), "w",
                  encoding="utf-8") as f:
            f.write("# operador proposto pelo modelo local; fora do nucleo\n\n"
                    + fonte + "\n")

    print("")
    print("=" * 78)
    print(f"  aceitos            {len(aceitos)}: {[n for n, _f, _s in aceitos]}")
    print(f"  reinventaram o meu {len(repetidos)}: {repetidos}")
    print(f"  GENUINAMENTE NOVOS {len(novos)}: {[n for n, _f, _s in novos]}")
    print(f"  ({time.time()-t0:.0f}s)")
    print("")
    print(f"O1 {'.' if len(aceitos) >= 6 else 'FALHOU .'} gera bem definidos: "
          f"{len(aceitos)} (>= 6)")
    print(f"O2 {'.' if len(novos) >= 3 else 'FALHOU .'} nao sao os meus: "
          f"{len(novos)} (>= 3)")
    print("\n  O3 e O4 (paga? placebo?) rodam em portao_operador.py, que monta as")
    print("  propriedades com estes operadores e mede com orcamento casado.")


if __name__ == "__main__":
    main()
