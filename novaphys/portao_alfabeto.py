"""
novaphys.portao_alfabeto — as primitivas propostas pela maquina valem alguma coisa?
====================================================================================

ESCRITO ANTES DE OLHAR QUALQUER PROPOSTA.

O QUE ESTA SENDO TESTADO
--------------------------
`alfabeto_aberto.py` mostra a um modelo local os pares de TREINO de tarefas que o
nucleo nao resolve e pede um codec novo. Aqui se mede se isso paga — e o que
importa nao e o numero bonito, e a separacao entre duas coisas muito diferentes:

  CAPTURA     o codec proposto a partir da tarefa X resolve o TESTE de X.
              Legitimo (o teste ficou escondido), mas e o caso facil.
  TRANSFERE   o codec proposto a partir de X resolve tarefas que o modelo NUNCA
              viu. E isto que decide se o alfabeto cresceu de verdade ou se o
              modelo so escreveu uma tabela para a tarefa que leu.

TRES BRACOS, ORCAMENTO CASADO EM AVALIACOES
---------------------------------------------
  base       CODECS do nucleo
  proposta   CODECS + as propostas aceitas
  placebo    CODECS + as MESMAS propostas com as cores previstas embaralhadas
             por uma permutacao fixa. Mesmo numero de candidatos, mesmo custo de
             avaliacao, mesma forma — e nenhuma informacao. Controla "ganhou
             porque tem mais bilhete de loteria".

Todos com o MESMO teto de avaliacoes por tarefa. Sem isso a comparacao nao vale
nada, e este projeto ja se enganou assim uma vez.

PRE-REGISTRO
--------------
  A1  CAPTURA: >= 3 das tarefas usadas para gerar passam a ser resolvidas
  A2  TRANSFERE: >= 2 tarefas NAO mostradas ao modelo passam a ser resolvidas
      (este e o portao que decide se o canal serve para alguma coisa)
  A3  NAO E LOTERIA: o placebo nao ganha nada sobre a base (ganho <= 0)
  A4  NAO QUEBRA: no maximo 1 tarefa perdida em relacao a base
PREVISAO: A1 passa (aposto 3 a 6 capturas). A2 FALHA — aposto 0 ou 1
transferencia, porque o modelo tende a escrever a regra da tarefa que leu e nao
o principio. A3 passa. A4 passa. Se A2 passar, o gargalo de alfabeto abriu e o
canal vira permanente; se falhar, o que abriu foi so a CAPTURA, e o proximo gap
e "escrever o principio, nao a tarefa".
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import types

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

PASTA = os.path.join(os.path.dirname(__file__), "propostas")
PERM = {c: (c * 7 + 3) % 10 for c in range(10)}   # placebo: bijecao fixa das cores


def carrega_propostas(pasta=PASTA):
    """Importa cada proposta num modulo isolado. Nada disso toca o nucleo."""
    fora = []
    for cam in sorted(glob.glob(os.path.join(pasta, "*.py"))):
        fonte = open(cam, encoding="utf-8").read()
        mod = types.ModuleType("prop_" + os.path.basename(cam)[:-3])
        try:
            exec(compile(fonte, cam, "exec"), mod.__dict__)
        except Exception as e:
            print(f"    {os.path.basename(cam)}: nao importa ({e})")
            continue
        for nome, obj in vars(mod).items():
            if (isinstance(obj, type) and hasattr(obj, "propor")
                    and hasattr(obj, "prever") and hasattr(obj, "bits")):
                try:
                    inst = obj()          # classe que exige argumento nao entra
                except Exception as e:
                    print(f"    {os.path.basename(cam)}: nao instancia ({e})")
                    break
                if not getattr(inst, "nome", None):
                    inst.nome = nome.lower()
                tarefa = os.path.basename(cam).split("_")[0]
                fora.append((tarefa, inst))
                break
    return fora


class Cego:
    """Placebo pareado: mesma proposta, mesma forma, cores destruidas."""

    def __init__(self, dentro):
        self.dentro = dentro
        self.nome = "cego_" + getattr(dentro, "nome", "x")

    def propor(self, treino, alvo=None):
        return self.dentro.propor(treino, alvo)

    def bits(self, m):
        return self.dentro.bits(m)

    def prever(self, m, a):
        return {p: PERM[c % 10] for p, c in self.dentro.prever(m, a).items()}


def roda(tarefas, codecs, teto, etiqueta):
    ok, t0 = set(), time.time()
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
            if pilha and err == 0 and all(D.aplicar((fm, pilha), x) == y for x, y in te):
                ok.add(tarefa)
        except Exception:
            pass
        if (k + 1) % 50 == 0:
            print(f"    [{etiqueta}] {k+1} tarefas, {len(ok)} resolvidas, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=120)
    ap.add_argument("--teto", type=int, default=400)
    a = ap.parse_args()

    props = carrega_propostas()
    vistas = {t for t, _c in props}
    extras = [c for _t, c in props]
    cegos = [Cego(c) for c in extras]
    tarefas = carregar(PASTA_AVAL)[: a.tarefas]

    print("=" * 78)
    print(f"PORTAO DO ALFABETO — {len(extras)} propostas, {len(tarefas)} tarefas, "
          f"teto {a.teto}")
    print(f"  geradas a partir de: {sorted(vistas)}")
    print("=" * 78)

    base = roda(tarefas, D.CODECS, a.teto, "base")
    prop = roda(tarefas, D.CODECS + extras, a.teto, "proposta")
    plac = roda(tarefas, D.CODECS + cegos, a.teto, "placebo")

    nomes = {t for t, _tr, _te in tarefas}
    ganhou = prop - base
    perdeu = base - prop
    captura = ganhou & vistas
    transfere = ganhou - vistas
    print("")
    print("=" * 78)
    print(f"  base     {len(base):3d}/{len(tarefas)}")
    print(f"  proposta {len(prop):3d}/{len(tarefas)}   ganhou {len(ganhou)}, "
          f"perdeu {len(perdeu)}")
    print(f"  placebo  {len(plac):3d}/{len(tarefas)}   ganhou {len(plac-base)}, "
          f"perdeu {len(base-plac)}")
    print(f"\n  CAPTURA   {len(captura)}: {sorted(captura)}")
    print(f"  TRANSFERE {len(transfere)}: {sorted(transfere)}")
    print("")
    print(f"A1 {'.' if len(captura) >= 3 else 'FALHOU .'} captura {len(captura)} (>= 3)")
    print(f"A2 {'.' if len(transfere) >= 2 else 'FALHOU .'} TRANSFERE "
          f"{len(transfere)} (>= 2)")
    print(f"A3 {'.' if len(plac - base) <= 0 else 'FALHOU .'} placebo ganhou "
          f"{len(plac-base)} (<= 0)")
    print(f"A4 {'.' if len(perdeu) <= 1 else 'FALHOU .'} perdeu {len(perdeu)} (<= 1)")
    if vistas - nomes:
        print(f"\n  aviso: {len(vistas - nomes)} tarefas geradoras estao fora "
              f"da fatia medida")


if __name__ == "__main__":
    main()
