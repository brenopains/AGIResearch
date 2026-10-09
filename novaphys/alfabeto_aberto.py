"""
novaphys.alfabeto_aberto — o alfabeto deixa de ser meu
=======================================================

O GARGALO, DITO PELO USUARIO
------------------------------
"se a gente continuar desenhando tudo ao inves de resolver o ponto que causa
essa falha/gap, nao vamos evoluir muito". O ponto e este: o unico canal de
estrutura nova neste projeto sou eu escrevendo primitivas a mao. Cada conserto
meu desce o teto um degrau e para.

Aqui um modelo local escreve CODECS NOVOS e o sistema julga. Nao e "IA escreve o
codigo do projeto": e um canal estreito, com contrato minusculo e cerca dura.

A CERCA (a restricao do usuario, literal)
-------------------------------------------
"Eu havia dito do sistema ler o proprio codigo NAO pra modifica-lo, isso iria nos
fazer perder o projeto e seria arriscado, mas para identificar falhas, rodar suas
proprias simulacoes".

Entao, por construcao:
  1. o modelo NUNCA escreve em `descricao.py` nem em arquivo nenhum do nucleo;
  2. o que ele escreve cai em `propostas/*.py`, isolado, e so entra numa rodada
     como item EXTRA da lista de codecs, nunca substituindo nada;
  3. todo codigo gerado passa por uma varredura estatica (sem import, sem open,
     sem exec/eval, sem __dunder__) e roda em SUBPROCESSO com relogio;
  4. o veredito nao e opiniao do modelo nem minha: e o portao de bits.

POR QUE UM CODEC E SEGURO DE TERCEIRIZAR
------------------------------------------
O contrato tem tres metodos e nenhum deles pode estragar o estado do sistema:

    nome                   str
    propor(treino, alvo)   -> lista de marcadores (parametros candidatos)
    bits(m)                -> float, o preco do marcador
    prever(m, a)           -> dict (i,j) -> cor, as celulas que ele reivindica

Quem escreve primeiro na celula ganha (first-writer-wins) e tudo e cobrado em
bits. Um codec mentiroso nao "corrompe" nada: ele paga caro e perde. O unico dano
real que um codec ruim causa e gastar ORCAMENTO DE AVALIACAO — e por isso o
portao compara com orcamento casado e com placebo pareado.

PRE-REGISTRO — ver `portao_alfabeto.py`, escrito antes de qualquer proposta.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"
TOKENS = 4000   # orcamento de geracao; modelo que "pensa" precisa de folga
PASTA = os.path.join(os.path.dirname(__file__), "propostas")

# ─── cerca estatica ──────────────────────────────────────────────────────────

PROIBIDO_NOME = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit",
    "quit", "memoryview",
}
PERMITIDO_IMPORT = {"math", "itertools", "collections"}


def varre(fonte: str):
    """Devolve lista de motivos para recusar. Vazia = aceito."""
    try:
        arv = ast.parse(fonte)
    except SyntaxError as e:
        return [f"sintaxe: {e}"]
    ruim = []
    for no in ast.walk(arv):
        if isinstance(no, ast.Import):
            for al in no.names:
                if al.name.split(".")[0] not in PERMITIDO_IMPORT:
                    ruim.append(f"import proibido: {al.name}")
        elif isinstance(no, ast.ImportFrom):
            if (no.module or "").split(".")[0] not in PERMITIDO_IMPORT:
                ruim.append(f"import proibido: from {no.module}")
        elif isinstance(no, ast.Name) and no.id in PROIBIDO_NOME:
            ruim.append(f"nome proibido: {no.id}")
        elif isinstance(no, ast.Attribute) and no.attr.startswith("__"):
            ruim.append(f"atributo dunder: {no.attr}")
        elif isinstance(no, (ast.Global, ast.Nonlocal)):
            ruim.append("estado global")
    classes = [n for n in arv.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1:
        ruim.append(f"esperava 1 classe, achei {len(classes)}")
    else:
        met = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
        falta = {"propor", "bits", "prever"} - met
        if falta:
            ruim.append(f"faltam metodos: {sorted(falta)}")
    return ruim


# ─── o que o modelo ve ───────────────────────────────────────────────────────

CONTRATO = '''\
Um CODEC descreve parte da saida a partir da entrada. O contrato e exatamente este:

    class MeuCodec:
        nome = "meu_codec"

        def propor(self, treino, alvo=None):
            """treino: lista de pares (a, b) de grades (lista de listas de int).
            alvo[k]: conjunto das celulas (i,j) do par k AINDA NAO explicadas —
            se alvo nao e None, ignore as celulas fora dele.
            Devolve uma LISTA de marcadores: cada marcador e um parametro
            candidato, e precisa ser HASHAVEL (int, str, tupla)."""
            return []

        def bits(self, m):
            """Preco em bits do marcador m. Numero maior = descricao mais cara."""
            return 1.0

        def prever(self, m, a):
            """Dado o marcador m e a grade de entrada a, devolve um dicionario
            {(i, j): cor} com as celulas que este codec REIVINDICA escrever.
            Celula nao reivindicada fica para outro codec."""
            return {}

Regras duras:
  - quem escreve primeiro numa celula ganha; nao tente sobrescrever;
  - reivindique SO o que voce sabe; celula errada custa bits de excecao;
  - nada de import fora de math/itertools/collections, nada de open/exec/eval;
  - so a classe, sem texto fora do bloco de codigo.

Dois exemplos reais do sistema:

    class MapaCor:
        """Substituicao global de cor, deduzida do treino."""
        nome = "mapa_cor"

        def propor(self, treino, alvo=None):
            mp = {}
            for k, (a, b) in enumerate(treino):
                al = None if alvo is None else alvo[k]
                for i in range(len(a)):
                    for j in range(len(a[0])):
                        if al is not None and (i, j) not in al:
                            continue
                        c, d = a[i][j], b[i][j]
                        if c in mp and mp[c] != d:
                            return []
                        mp[c] = d
            mp = {c: d for c, d in mp.items() if c != d}
            return [tuple(sorted(mp.items()))] if mp else []

        def bits(self, m):
            return len(m) * 2 * 3.32

        def prever(self, m, a):
            mp = dict(m)
            return {(i, j): mp[a[i][j]] for i in range(len(a))
                    for j in range(len(a[0])) if a[i][j] in mp}


    class Periodo:
        """A saida repete um bloco da entrada com periodo (ph, pw)."""
        nome = "periodo"

        def propor(self, treino, alvo=None):
            fora = set()
            for a, b in treino:
                h, w = len(b), len(b[0])
                for ph in range(1, h + 1):
                    for pw in range(1, w + 1):
                        if all(b[i][j] == b[i % ph][j % pw]
                               for i in range(h) for j in range(w)):
                            fora.add((ph, pw))
            return sorted(fora)

        def bits(self, m):
            return 10.0

        def prever(self, m, a):
            ph, pw = m
            return {(i, j): a[i % ph][j % pw] for i in range(len(a))
                    for j in range(len(a[0]))}
'''


def desenha(g):
    return "\n".join("".join(str(c) for c in l) for l in g)


def descreve_tarefa(tr, te, limite=3):
    p = []
    for k, (a, b) in enumerate(tr[:limite]):
        p.append(f"  par {k+1}: entrada {len(a)}x{len(a[0])}  ->  saida "
                 f"{len(b)}x{len(b[0])}\n"
                 f"{textwrap.indent(desenha(a), '    ')}\n    ->\n"
                 f"{textwrap.indent(desenha(b), '    ')}")
    return "\n".join(p)


def pede(msg, n=1400, temp=0.7, semente=None):
    corpo = {"messages": [{"role": "user", "content": msg}],
             "max_tokens": n, "temperature": temp}
    if semente is not None:
        corpo["seed"] = semente
    d = json.dumps(corpo).encode()
    r = urllib.request.Request(URL, d, {"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(r, timeout=900).read())
    msg = o["choices"][0]["message"]
    t = msg.get("content") or ""
    # Um modelo que "pensa" pode gastar TODO o orcamento no raciocinio e devolver
    # content vazio (finish_reason = length, sem </think>). Nesse caso o codigo,
    # se existir, esta no raciocinio — e se nem la existir, a culpa e do teto.
    if "```" not in t:
        rac = msg.get("reasoning_content") or ""
        # SO se o raciocinio tiver um bloco FECHADO. Raciocinio cortado no meio
        # deixa codigo pela metade, e o parser engolia isso como proposta — o
        # erro aparecia depois como "leading zeros" ou "string nao terminada",
        # que parecia burrice do modelo e era truncamento meu.
        if rac.count("```") >= 2:
            t = rac + "\n" + t
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return t.strip()


def extrai(txt):
    m = re.findall(r"```(?:python)?\s*(.*?)```", txt, re.S)
    return (m[-1] if m else txt).strip()


# ─── prova de vida: o codec roda e nao explode ───────────────────────────────

PROVA = '''
import json, sys
{fonte}
_c = {classe}()
_tr = json.loads({treino!r})
_tr = [(a, b) for a, b in _tr]
_ms = _c.propor(_tr, None)
assert isinstance(_ms, (list, tuple)), "propor nao devolveu lista"
_n = 0
for _m in list(_ms)[:8]:
    hash(_m)
    float(_c.bits(_m))
    _p = _c.prever(_m, _tr[0][0])
    assert isinstance(_p, dict), "prever nao devolveu dict"
    for _k, _v in list(_p.items())[:50]:
        assert isinstance(_k, tuple) and len(_k) == 2, "chave nao e (i,j)"
        int(_v)
    _n += 1
print("VIVO", len(_ms), _n)
'''


def prova_de_vida(fonte, classe, treino, segundos=20):
    src = PROVA.format(fonte=fonte, classe=classe,
                       treino=json.dumps([[a, b] for a, b in treino[:3]]))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(src)
        cam = f.name
    try:
        s = subprocess.run([sys.executable, cam], capture_output=True,
                           text=True, timeout=segundos)
        if "VIVO" in s.stdout:
            return True, s.stdout.strip()
        return False, (s.stderr or s.stdout).strip().splitlines()[-1:] or ["vazio"]
    except subprocess.TimeoutExpired:
        return False, "estourou o relogio"
    finally:
        try:
            os.unlink(cam)
        except OSError:
            pass


# ─── geracao ─────────────────────────────────────────────────────────────────

def gera(tr, te, tentativas=3, semente=0, verboso=True):
    """Pede um codec novo para uma tarefa que o nucleo nao resolve."""
    p = (CONTRATO + "\n\nEsta tarefa o sistema NAO resolve. Os pares de treino:\n\n"
         + descreve_tarefa(tr, te)
         + "\n\nEscreva UM codec novo, generico (nao decore esta tarefa: ele tem "
           "de valer para qualquer tarefa que siga o mesmo principio), que capture "
           "a regra que liga entrada e saida. So o bloco ```python com a classe.")
    for t in range(tentativas):
        try:
            fonte = extrai(pede(p, TOKENS, 0.8, semente * 100 + t))
        except Exception as e:
            if verboso:
                print(f"      tentativa {t+1}: erro de rede {e}", flush=True)
            continue
        motivos = varre(fonte)
        if motivos:
            if verboso:
                print(f"      tentativa {t+1}: recusado — {motivos[0]}", flush=True)
            continue
        classe = re.search(r"class\s+(\w+)", fonte).group(1)
        vivo, nota = prova_de_vida(fonte, classe, tr)
        if not vivo:
            if verboso:
                print(f"      tentativa {t+1}: nao vive — {nota}", flush=True)
            continue
        if verboso:
            print(f"      tentativa {t+1}: ACEITO {classe} ({nota})", flush=True)
        return fonte, classe
    return None, None


def main() -> None:
    from novaphys.arc import carregar
    from novaphys.eter import PASTA_AVAL
    from novaphys import descricao as D

    global TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=12)
    ap.add_argument("--tentativas", type=int, default=3)
    ap.add_argument("--teto", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    a = ap.parse_args()
    TOKENS = a.tokens
    os.makedirs(PASTA, exist_ok=True)

    todas = carregar(PASTA_AVAL)
    falhas = [(t, tr, te) for (t, tr, te) in todas if t not in D.JA][: a.tarefas]
    print("=" * 78)
    print(f"ALFABETO ABERTO — {len(falhas)} tarefas que o nucleo NAO resolve")
    print("=" * 78)
    t0, aceitos = time.time(), []
    for k, (tarefa, tr, te) in enumerate(falhas):
        print(f"  [{k+1}/{len(falhas)}] {tarefa}", flush=True)
        fonte, classe = gera(tr, te, a.tentativas, semente=k)
        if fonte:
            cam = os.path.join(PASTA, f"{tarefa}_{classe}.py")
            with open(cam, "w", encoding="utf-8") as f:
                f.write(f"# proposto pelo modelo local a partir de {tarefa}\n"
                        f"# NAO faz parte do nucleo; entra so como codec EXTRA\n\n"
                        + fonte + "\n")
            aceitos.append((tarefa, classe, cam))
    print("")
    print("=" * 78)
    print(f"  {len(aceitos)}/{len(falhas)} propostas passaram na cerca e vivem "
          f"({time.time()-t0:.0f}s)")
    for tarefa, classe, cam in aceitos:
        print(f"    {tarefa}  {classe}")
    print("\n  nenhuma delas entrou no nucleo. O portao decide — portao_alfabeto.py")


if __name__ == "__main__":
    main()
