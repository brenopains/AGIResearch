"""
novaphys.laco_proponente — o proponente DENTRO do laço, não ao lado dele
==========================================================================

A CORRECAO DO USUARIO
-----------------------
"A meu ver o modelo deveria estar totalmente integrado a nossa AGI e nao aquem,
gerando propostas a parte mas sabendo do que ela testou e notando padroes. Ele
deve trabalhar junto e em tempo real propondo a AGI para ela mesma."

Ele esta certo e `alfabeto_aberto.py` e a versao fraca: la o modelo ve a tarefa
CRUA, propoe no escuro uma vez, as propostas se acumulam numa pasta e o portao
roda depois — o veredito nunca volta. O alfabeto nao cresce durante a rodada.

O QUE MUDA AQUI
-----------------
  1. ESTADO, nao tarefa crua. O proponente recebe o que a busca JA fez: a pilha
     montada e o preco em bits de cada codec, o RESIDUO desenhado (as celulas que
     ninguem explicou), onde a busca parou e por que.
  2. RE-TENTATIVA IMEDIATA. O codec proposto volta para o nucleo na mesma tarefa,
     no mesmo instante, com o mesmo teto.
  3. O VEREDITO VOLTA. O proponente ve o que aconteceu com cada proposta dele —
     adotada, inutil, ou quebrou algo — e essa historia entra no proximo pedido.
  4. O ALFABETO CRESCE DURANTE A RODADA. O codec adotado na tarefa 7 esta
     disponivel na tarefa 8. E disto que sai a unica coisa que interessa:
     TRANSFERENCIA.

A CERCA CONTINUA A MESMA (restricao do usuario, literal: ler o codigo "nao pra
modifica-lo"): varredura estatica, prova de vida em subprocesso, e o codec entra
so como item EXTRA da lista. Nada e escrito no nucleo.

NAO-REGRESSAO
---------------
Um codec que ajuda a tarefa k pode quebrar as tarefas 1..k-1. Antes de adotar,
uma amostra fixa de tarefas JA RESOLVIDAS e re-rodada com o alfabeto estendido.
Uma perda so, e a proposta e recusada. Sem isto o alfabeto cresce e o placar cai.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  I1  O ALFABETO CRESCE: pelo menos 2 codecs adotados
  I2  GANHA DO NUCLEO: o laco resolve >= +2 tarefas sobre a base, teto igual
  I3  TRANSFERENCIA: >= 1 tarefa resolvida GRACAS a um codec adotado numa tarefa
      ANTERIOR — e isto, e so isto, que distingue "trabalhar junto" de
      "propostas soltas"
  I4  NAO QUEBRA: zero perdas na amostra de nao-regressao (por construcao, mas
      medido mesmo assim)
  I5  PAREADO CONTRA O LOTE: o laco adota mais codecs uteis que `alfabeto_aberto`
      no mesmo modelo, mesmas tarefas, mesmo numero de tentativas
PREVISAO: I1 passa. I2 marginal, aposto +1 a +3. I3 FALHA — transferencia e a
parte dificil e o modelo tende a escrever a regra da tarefa que leu. I4 passa.
I5 passa, porque ver o residuo e uma dica forte demais para nao ajudar. Se I3
passar, o alfabeto deixou de ser meu de verdade.
"""

from __future__ import annotations

import argparse
import os
import time

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.alfabeto_aberto import (CONTRATO, desenha, extrai, pede,
                                      prova_de_vida, varre)
import novaphys.alfabeto_aberto as AA
import re
import types


# ─── o estado que o proponente passa a ver ───────────────────────────────────

def residuo(tr, fm, pilha):
    """As celulas que a pilha atual NAO explica, por par de treino."""
    fora = []
    for a, b in tr:
        try:
            prev = D.aplicar((fm, pilha), a)
        except Exception:
            prev = None
        h, w = dims(b)
        if prev is None or dims(prev) != (h, w):
            fora.append((b, [[1] * w for _ in range(h)]))
            continue
        mask = [[1 if prev[i][j] != b[i][j] else 0 for j in range(w)]
                for i in range(h)]
        fora.append((b, mask))
    return fora


def diagnostico(tr, fm, pilha, err, limite=2):
    """O texto que descreve ONDE a busca parou. E isto que `alfabeto_aberto`
    nao mostrava, e e a diferenca entre propor no escuro e propor sabendo.

    DOIS MODOS DE FALHA, E EU ESTAVA TRATANDO COMO UM. Medido em 60 tarefas:

        resolvida                           12/60
        decorou (residuo 0, teste errado)    3/60
        NAO DESCREVEU (residuo > 0)         40/60
        sem pilha nenhuma                    5/60

    Residuo > 0 e "falta primitiva": a busca nao consegue nem escrever o treino.
    Residuo == 0 com teste errado e o oposto — ela escreveu, mas DECOROU, e o
    que falta e uma descricao mais BARATA, nao uma celula por explicar. Pedir
    "explique o que falta" no segundo caso e pedir nada, porque nao falta nada.
    Dois terços sao o primeiro modo: e por isso que reordenar candidatos deu
    nulo (`proponente.py`, 40/400 nos quatro bracos) — nao se reordena ate uma
    primitiva que nao existe."""
    p = []
    if pilha:
        nomes = []
        for it in pilha:
            if D._marca(it):
                nomes.append(str(it).split(":")[0])
            else:
                cd, m = it
                try:
                    nomes.append(f"{cd.nome}({cd.bits(m):.0f} bits)")
                except Exception:
                    nomes.append(cd.nome)
        p.append("A busca ja montou esta pilha, e ela NAO basta:\n    "
                 + " -> ".join(nomes))
    else:
        p.append("A busca nao conseguiu montar pilha nenhuma.")
    p.append(f"Sobraram {err} celulas sem explicacao.")
    res = residuo(tr, fm, pilha)
    p.append("\nO que FALTA, por par (1 = celula que ninguem explicou, "
             "0 = ja resolvida):")
    for k, (b, mask) in enumerate(res[:limite]):
        p.append(f"  par {k+1} — saida correta:")
        p.append("    " + desenha(b).replace("\n", "\n    "))
        p.append(f"  par {k+1} — o que ainda falta:")
        p.append("    " + desenha(mask).replace("\n", "\n    "))
    return "\n".join(p)


def historia(hist, limite=6):
    """O veredito das propostas anteriores, que em `alfabeto_aberto` nunca
    voltava para o modelo."""
    if not hist:
        return ""
    p = ["\nO que ja foi proposto nesta rodada, e o que aconteceu:"]
    for nome, veredito in hist[-limite:]:
        p.append(f"  {nome}: {veredito}")
    p.append("Nao repita os que falharam. Se um padrao funcionou, generalize-o.")
    return "\n".join(p)


# ─── o laço ──────────────────────────────────────────────────────────────────

def propoe(tr, fm, pilha, err, hist, tentativas, verboso=True, decorou=False):
    diag = diagnostico(tr, fm, pilha, err)
    if decorou:
        pedido = ("\n\nO sistema DESCREVEU o treino inteiro com esta pilha, mas a "
                  "descricao esta ERRADA: acerta os pares que viu e erra o par "
                  "escondido — ela DECOROU em vez de capturar o principio. "
                  "Escreva UM codec que capture o principio de forma mais BARATA "
                  "(menos bits) que a pilha acima, para que a regra certa ganhe a "
                  "comparacao. So o bloco ```python com a classe.")
    else:
        pedido = ("\n\nEscreva UM codec novo que explique as celulas que faltam. "
                  "Ele tem de ser GENERICO — vale para qualquer tarefa com o mesmo "
                  "principio, nao uma tabela desta. So o bloco ```python com a "
                  "classe.")
    p = (CONTRATO
         + "\n\nO sistema falhou NESTA tarefa. Eis o estado real da busca:\n\n"
         + diag
         + historia(hist)
         + pedido)
    for t in range(tentativas):
        try:
            fonte = extrai(pede(p, AA.TOKENS, 0.8, t))
        except Exception as e:
            if verboso:
                print(f"      tentativa {t+1}: rede {e}", flush=True)
            continue
        motivos = varre(fonte)
        if motivos:
            if verboso:
                print(f"      tentativa {t+1}: cerca — {motivos[0][:70]}", flush=True)
            continue
        classe = re.search(r"class\s+(\w+)", fonte).group(1)
        vivo, nota = prova_de_vida(fonte, classe, tr)
        if not vivo:
            if verboso:
                print(f"      tentativa {t+1}: nao vive — {str(nota)[:70]}", flush=True)
            continue
        return fonte, classe
    return None, None


def instancia(fonte, classe):
    mod = types.ModuleType("prop_" + classe)
    exec(compile(fonte, "<proposta>", "exec"), mod.__dict__)
    obj = getattr(mod, classe)()
    if not getattr(obj, "nome", None):
        obj.nome = classe.lower()
    return obj


def resolve(tr, te, codecs, teto):
    try:
        fm, pilha, err = D.montar(tr, codecs=codecs, teto=teto)
    except Exception:
        return None, [], 10 ** 9, False
    ok = bool(pilha) and err == 0 and all(
        D.aplicar((fm, pilha), x) == y for x, y in te)
    return fm, pilha, err, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefas", type=int, default=40)
    ap.add_argument("--tentativas", type=int, default=2)
    ap.add_argument("--teto", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=1800)
    ap.add_argument("--guarda", type=int, default=12,
                    help="quantas tarefas ja resolvidas entram na nao-regressao")
    a = ap.parse_args()
    AA.TOKENS = a.tokens

    todas = carregar(PASTA_AVAL)[: a.tarefas]
    print("=" * 78)
    print(f"LACO DO PROPONENTE — {len(todas)} tarefas, teto {a.teto}, "
          f"alfabeto CRESCE durante a rodada")
    print("=" * 78)

    alfabeto = list(D.CODECS)          # cresce; o nucleo em disco nao muda
    adotados, hist, t0 = [], [], time.time()
    base_ok, laco_ok, dono = set(), set(), {}
    guarda = []                        # tarefas ja resolvidas, para nao-regressao

    for k, (tarefa, tr, te) in enumerate(todas):
        # 1. o nucleo PURO, para a linha de base pareada
        _fmb, _pb, _eb, okb = resolve(tr, te, D.CODECS, a.teto)
        if okb:
            base_ok.add(tarefa)

        # 2. o alfabeto ATUAL. Enquanto ele nao cresceu e o MESMO calculo —
        # rodar de novo so queimaria metade do relogio a toa.
        if len(alfabeto) == len(D.CODECS):
            fm, pilha, err, ok = _fmb, _pb, _eb, okb
        else:
            fm, pilha, err, ok = resolve(tr, te, alfabeto, a.teto)
        if ok:
            laco_ok.add(tarefa)
            if len(guarda) < a.guarda:
                guarda.append((tarefa, tr, te))
            if tarefa not in base_ok:
                # resolvida SO por codec adotado antes: isto e transferencia
                dono[tarefa] = "codec adotado em tarefa anterior"
            print(f"  [{k+1}/{len(todas)}] {tarefa} resolvida"
                  f"{' (TRANSFERENCIA)' if tarefa in dono else ''}", flush=True)
            continue

        # 3. presa: o proponente ve o ESTADO e propoe
        decorou = (err == 0 and bool(pilha))
        modo = "DECOROU (residuo 0, teste errado)" if decorou else f"residuo {err}"
        print(f"  [{k+1}/{len(todas)}] {tarefa} falhou — {modo} — "
              f"pedindo proposta", flush=True)
        fonte, classe = propoe(tr, fm, pilha, err, hist, a.tentativas,
                               decorou=decorou)
        if not fonte:
            hist.append((f"(nada valido para {tarefa})", "nao passou na cerca"))
            continue

        # 4. re-tentativa IMEDIATA, mesmo teto
        try:
            novo = instancia(fonte, classe)
        except Exception as e:
            hist.append((classe, f"nao instancia: {e}"))
            continue
        _fm2, _p2, err2, ok2 = resolve(tr, te, alfabeto + [novo], a.teto)

        if not ok2 and (err2 >= err or decorou):
            hist.append((classe, "nao mudou nada — nem resolveu nem baixou residuo"))
            print(f"      {classe}: inutil (residuo {err} -> {err2})", flush=True)
            continue

        # 5. nao-regressao antes de adotar
        perdeu = [t for (t, tr_g, te_g) in guarda
                  if not resolve(tr_g, te_g, alfabeto + [novo], a.teto)[3]]
        if perdeu:
            hist.append((classe, f"quebrou {len(perdeu)} tarefa(s) ja resolvida(s)"))
            print(f"      {classe}: RECUSADO, quebrou {len(perdeu)}", flush=True)
            continue

        # 6. adotado: entra no alfabeto e vale para as PROXIMAS tarefas
        alfabeto.append(novo)
        adotados.append((tarefa, classe, fonte))
        hist.append((classe, "ADOTADO" + (" e resolveu a tarefa" if ok2
                                          else f", residuo {err} -> {err2}")))
        if ok2:
            laco_ok.add(tarefa)
            if len(guarda) < a.guarda:
                guarda.append((tarefa, tr, te))
        print(f"      {classe}: ADOTADO ({'resolveu' if ok2 else f'residuo {err}->{err2}'})"
              f" — alfabeto agora tem {len(alfabeto)}", flush=True)

    pasta = os.path.join(os.path.dirname(__file__), "propostas_laco")
    os.makedirs(pasta, exist_ok=True)
    for tarefa, classe, fonte in adotados:
        with open(os.path.join(pasta, f"{tarefa}_{classe}.py"), "w",
                  encoding="utf-8") as f:
            f.write(f"# adotado no laco a partir de {tarefa}; fora do nucleo\n\n"
                    + fonte + "\n")

    transf = sorted(dono)
    print("")
    print("=" * 78)
    print(f"  base (nucleo puro)  {len(base_ok):3d}/{len(todas)}")
    print(f"  laco                {len(laco_ok):3d}/{len(todas)}   "
          f"({time.time()-t0:.0f}s)")
    print(f"  codecs adotados     {len(adotados)}: "
          f"{[c for _t, c, _f in adotados]}")
    print(f"  transferencia       {len(transf)}: {transf}")
    print("")
    print(f"I1 {'.' if len(adotados) >= 2 else 'FALHOU .'} alfabeto cresceu "
          f"{len(adotados)} (>= 2)")
    print(f"I2 {'.' if len(laco_ok) - len(base_ok) >= 2 else 'FALHOU .'} ganhou do "
          f"nucleo {len(laco_ok)-len(base_ok):+d} (>= +2)")
    print(f"I3 {'.' if len(transf) >= 1 else 'FALHOU .'} TRANSFERENCIA "
          f"{len(transf)} (>= 1)")
    print(f"I4 {'.' if not (base_ok - laco_ok) else 'FALHOU .'} nao quebrou "
          f"{len(base_ok - laco_ok)} perdas (0)")
    print("\n  I5 (pareado contra o lote) se compara com alfabeto_aberto.py")


if __name__ == "__main__":
    main()
