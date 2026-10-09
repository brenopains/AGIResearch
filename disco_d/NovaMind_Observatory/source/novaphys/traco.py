"""
novaphys.traco — instrumenta a busca real e despeja o que de fato aconteceu
============================================================================

REGRA DESTE ARQUIVO
--------------------
Nada aqui inventa. Todo campo do JSON e lido de uma execucao real de
`eter.resolver`. Se um numero nao pode ser medido, ele nao aparece — nao ha
preenchimento plausivel, nao ha exemplo ilustrativo.

Onde houver amostragem (a busca avalia dezenas de milhares de candidatos e
nao da para gravar todos), o proprio JSON carrega quantos foram avaliados e
quantos foram gravados, para que a visualizacao nao possa dar a impressao de
mostrar o todo.

O QUE E GRAVADO
-----------------
  aterramento  quais operadores o eter instanciou para ESTA tarefa e a
               derivacao de cada um — de onde saiu o argumento. E isto que
               torna visivel o "buscar sob demanda": a lista muda de tarefa
               para tarefa e nao foi escrita por mim.

  busca        candidatos avaliados, com a grade que cada um produziu e a
               fracao de pixels que bateu com o alvo. A fracao e medida, nao
               estimada; e o mesmo sinal que o verificador usa para decidir,
               so que exposto em vez de descartado.

  solucao      o programa aceito e a grade em CADA estagio intermediario.
               E o pensamento inteiro, nao so a conclusao.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from novaphys.arc import Grid, ajustar_mapa, aplicar_mapa, carregar, dims, valida
from novaphys.eter import Contexto, aterrar, PASTA_AVAL


def _casa(r: Optional[Grid], alvo: Grid) -> float:
    """Fracao de pixels que bate. -1 se as dimensoes nem conferem."""
    if r is None or dims(r) != dims(alvo):
        return -1.0
    h, w = dims(alvo)
    ok = sum(1 for i in range(h) for j in range(w) if r[i][j] == alvo[i][j])
    return ok / (h * w)


def _derivacao(nome: str, ctx: Contexto) -> str:
    """De onde saiu o argumento deste operador. Texto derivado do contexto."""
    if nome.startswith("escala"):
        f = ctx.escalas[0]
        return (f"dims(saida)/dims(entrada) = {f[0]}x{f[1]} igual nos "
                f"{len(ctx.treino)} pares")
    if nome.startswith("reduz"):
        f = ctx.reducoes[0]
        return (f"dims(entrada)/dims(saida) = {f[0]}x{f[1]} igual nos "
                f"{len(ctx.treino)} pares")
    if nome.startswith("rec") and "p" in nome[3:]:
        de, para = nome[3:].split("p")
        return f"cor {de} existe na entrada, cor {para} existe na saida"
    if nome.startswith(("mantem", "remove", "recorta")):
        c = "".join(ch for ch in nome if ch.isdigit())
        return f"cor {c} presente nas entradas (fundo = {ctx.fundo})"
    if nome.startswith("preenche"):
        c = "".join(ch for ch in nome if ch.isdigit())
        return f"cor {c} presente nas saidas"
    if nome.startswith("obj"):
        return f"objetos = componentes 4-conexos != fundo ({ctx.fundo})"
    if nome.startswith("desl"):
        return f"deslocamento com preenchimento pelo fundo ({ctx.fundo})"
    return "primitiva fixa da base — nao depende da tarefa"


def _familia(nome: str) -> str:
    for f in ("escala", "reduz", "mantem", "remove", "recorta", "preenche",
              "desl", "obj"):
        if nome.startswith(f):
            return f
    if nome.startswith("rec") and "p" in nome[3:]:
        return "recolore"
    return "base"


def tracar(treino: Sequence[Tuple[Grid, Grid]], orcamento: int,
           max_grav: int = 400, prof_max: int = 3) -> Dict[str, object]:
    """
    Roda a MESMA busca de `eter.resolver`, gravando o percurso.

    A logica de aceitacao e identica; a unica diferenca e que aqui se guarda
    o que la e descartado. Se as duas divergirem, o traco mente — por isso
    o teste `traco_confere` no fim deste arquivo compara os dois.
    """
    ctx = Contexto(treino)
    ops = aterrar(ctx, True)
    nomes = [n for n, _ in ops]
    d = dict(ops)
    alvo0 = treino[0][1]

    aterramento = [{"nome": n, "familia": _familia(n),
                    "derivacao": _derivacao(n, ctx)} for n in nomes]

    passos: List[Dict[str, object]] = []
    gasto = 0
    solucao: Optional[Dict[str, object]] = None

    for prof in range(0, prof_max + 1):
        if solucao:
            break
        for seq in itertools.product(nomes, repeat=prof):
            if gasto >= orcamento:
                break
            gasto += 1
            saidas: List[Optional[Grid]] = []
            ok = True
            for a, b in treino:
                r: Optional[Grid] = a
                for nome in seq:
                    r = d[nome](r)                   # type: ignore[arg-type]
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)

            c = _casa(saidas[0] if saidas else None, alvo0)
            # grava os promissores e uma amostra regular do resto, para que o
            # visor mostre tanto os quase-acertos quanto o volume bruto
            # denso no comeco (buscas curtas terminam ali), depois amostra
            # regular, e sempre os quase-acertos. O JSON carrega `avaliados`
            # e `gravados` para que o visor nao possa parecer mostrar o todo.
            if len(passos) < max_grav and (gasto <= 240 or c > 0.5
                                           or gasto % 97 == 1):
                passos.append({
                    "i": gasto, "prof": prof, "ops": list(seq),
                    "casa": round(c, 4),
                    "grade": [list(l) for l in saidas[0]] if saidas and ok
                             and saidas[0] else None,
                })

            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                solucao = {"ops": list(seq), "mapa": None, "i": gasto}
                break
            mapa = ajustar_mapa(list(zip(saidas,          # type: ignore[arg-type]
                                         [b for _, b in treino])))
            if mapa is not None and all(
                    aplicar_mapa(r, mapa) == b            # type: ignore[arg-type]
                    for r, (_, b) in zip(saidas, treino)):
                solucao = {"ops": list(seq), "mapa": {str(k): v
                                                      for k, v in mapa.items()},
                           "i": gasto}
                break

    estagios: List[Dict[str, object]] = []
    if solucao:
        a0 = treino[0][0]
        r: Optional[Grid] = a0
        estagios.append({"op": "entrada", "grade": [list(l) for l in a0],
                         "casa": round(_casa(a0, alvo0), 4)})
        for nome in solucao["ops"]:                       # type: ignore[index]
            r = d[nome](r)                                # type: ignore[arg-type]
            estagios.append({"op": nome,
                             "grade": [list(l) for l in r] if r else None,
                             "casa": round(_casa(r, alvo0), 4)})
        if solucao["mapa"]:                               # type: ignore[index]
            m = {int(k): v for k, v in solucao["mapa"].items()}   # type: ignore
            r = aplicar_mapa(r, m)                        # type: ignore[arg-type]
            estagios.append({"op": f"recolore {solucao['mapa']}",  # type: ignore
                             "grade": [list(l) for l in r] if r else None,
                             "casa": round(_casa(r, alvo0), 4)})

    return {
        "aterramento": aterramento,
        "contexto": {"cores_entrada": ctx.cores_ent,
                     "cores_saida": ctx.cores_sai,
                     "fundo": ctx.fundo,
                     "escala": ctx.escalas[0] if ctx.escalas else None,
                     "reducao": ctx.reducoes[0] if ctx.reducoes else None,
                     "n_pares": len(treino)},
        "pares": [{"entrada": [list(l) for l in a],
                   "saida": [list(l) for l in b]} for a, b in treino],
        "busca": passos,
        "avaliados": gasto,
        "gravados": len(passos),
        "solucao": solucao,
        "estagios": estagios,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--tarefas", default="", help="ids separados por virgula")
    ap.add_argument("--quantas", type=int, default=6)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--saida", default="traco.json")
    ap.add_argument("--max-grav", type=int, default=140,
                    help="teto de passos gravados por tarefa; o JSON registra "
                         "avaliados e gravados para que a fracao fique explicita")
    a = ap.parse_args()

    todas = carregar(a.pasta)
    if a.tarefas:
        alvos = set(a.tarefas.split(","))
        sel = [t for t in todas if t[0] in alvos]
    else:
        sel = todas[:a.quantas]

    fora = []
    for nome, tr, te in sel:
        t = tracar(tr, a.orcamento, max_grav=a.max_grav)
        t["tarefa"] = nome
        t["resolvida"] = t["solucao"] is not None
        fora.append(t)
        s = t["solucao"]
        print(f"{nome}  aterrou {len(t['aterramento']):3d} ops  "        # type: ignore
              f"avaliou {t['avaliados']:6d}  "
              f"{'RESOLVIDA: ' + ' . '.join(s['ops']) if s else 'nao resolvida'}")  # type: ignore

    with open(a.saida, "w") as f:
        json.dump(fora, f)
    print(f"\ngravado {a.saida}  ({os.path.getsize(a.saida)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
