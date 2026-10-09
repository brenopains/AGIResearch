"""
novaphys.nuvem — a nuvem estilhacando
=======================================

O CRITERIO DO USUARIO, LITERAL
--------------------------------
"Uma mente artificial que e incapaz de imaginar vidro estilhacando, virando
metal, depois uma nuvem de algodao e depois a nuvem estilhacando nao e forte o
bastante pra minha ambicao."

O que morde e a ultima clausula: aplicar uma transformacao a um material que
NUNCA foi visto sofrendo aquilo.

  MATERIAL A = canais 0-2   (1,7s ; -0,6s ; 0,9s^2)
  MATERIAL B = canais 3-5   (-2,4s+3 ; 0,4s-1,5 ; ruido puro)
  REGIME  = alvo +1 quando o latente cruza um limiar   ("estilhacar")
  DERIVA  = alvo +1 quando o latente esta subindo      (outra transformacao)

  A loja tem   A|regime, A|deriva, B|deriva.   NAO tem B|regime.
  A tarefa e   prever o alvo de um episodio NOVO de B|regime.

Isso so e tentavel porque [[indice-ve-superficie]] rodada 2 mostrou que o eixo
de transformacao EXISTE (+0,1185 contra nulo +0,0359). Sem isso, um fracasso
aqui seria ambiguo entre "gerador fraco" e "pedido sem endereco".

O OBSTACULO DE DESENHO, E POR QUE A SAIDA NAO E TRAPACA
---------------------------------------------------------
`de_par_inv` orienta cada canal pelo sinal da correlacao dele com o ALVO — e no
teste o alvo e exatamente o que falta. Entao aqui a orientacao e pelo PRIMEIRO
COMPONENTE PRINCIPAL do fluxo, que e essencialmente o latente `s` e nao usa o
alvo nenhuma vez. Orientar cada canal para correlacionar positivamente com ele
torna 1,7s / -0,6s / -2,4s+3 / 0,4s-1,5 todos canonicos.

Sobra UMA ambiguidade: o sinal do proprio PC1 e arbitrario. Resolve-se testando
as duas e ficando com a que casa melhor com a regra recuperada — o que e o
passo de abducao, alinhar antes de comparar, e nao um vazamento do gabarito:
a escolha usa so o fluxo de teste e a regra vinda de OUTRO material.

O MECANISMO DE GERACAO
------------------------
Recupera-se pelo eixo de transformacao; feixa-se os episodios recuperados numa
REGRA R. Para cada instante do fluxo de teste e cada alvo candidato a, montam-se
os termos bind(papel, bind(nivel, a)) deslocados pelo atraso, e soma-se
sim(termo, R). Escolhe-se o `a` de maior escore. E recall associativo puro —
o HDC no papel em que mediu positivo (ordenar/casar), nunca propondo simbolo
novo, que e onde ele mede 2/400.

PRE-REGISTRO
--------------
  L0  PRE-CONDICAO. O TETO (R vinda de B|regime, que a tarefa nao pode ter)
      tem de passar de 60%. Se nem com a regra do proprio material o mecanismo
      expressa a transformacao, nada abaixo e interpretavel.
  L1  COMPOSTO (R de A|regime aplicada a B) > placebo + 5 pontos E > 55%
      -> a nuvem estilhaca: a transformacao atravessou o material
  L2  composto ~ acaso -> nao transfere; a recombinacao falha aqui como
      falhou no ARC (2/400)
  L3  composto > placebo MAS o nulo (R de alvo sorteado) tambem passa do acaso
      -> e duto, nao transferencia

  placebo = R feita de episodios SORTEADOS da loja (mesma quantidade)
  nulo    = R feita de episodios com alvo SORTEADO
  acaso   = 50%, e reporto tambem a classe majoritaria do episodio

PREVISAO: L1, 0,7. Mecanismo: sob 'regime' o alvo e +1 com o latente alto; apos
a orientacao por PC1 os canais lineares dos DOIS materiais viram +s, entao a
associacao "nivel orientado alto <-> +1" vale nos dois. O risco nomeado e a
resolucao do sinal global: se ela errar, a acuracia INVERTE para ~1-p em vez de
cair para o acaso — e isso e assinatura diagnostica, nao ruido.
Placar do dia: 6 previsoes, 2 certas, e as 2 certas foram as feitas DEPOIS de
rastrear o mecanismo no codigo do gerador. Esta e desse tipo.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe, sim
from novaphys.substrato import gerar_fluxo

N_LOJA = 16          # episodios por celula na loja
N_TESTE = 20         # episodios de teste
T = 200
ATRASOS = (0, 1)
NIVEIS = 4           # -4..+4
SEMENTE = 20260917


def episodio(rng: random.Random, material: str, transf: str):
    fluxo, alvo = gerar_fluxo(rng, T=T, K=6, estrutura=transf)
    col = (0, 1, 2) if material == "A" else (3, 4, 5)
    return np.array([[l[c] for c in col] for l in fluxo]), np.array(
        [1 if a > 0 else -1 for a in alvo])


# ────────────────────────────────── orientacao SEM olhar o alvo
def orienta_pc1(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    z-score por canal, orienta cada canal para correlacionar + com o PC1, e
    ordena os PAPEIS por |correlacao com o PC1|. Nenhum uso do alvo.
    """
    mu, sd = A.mean(axis=0), A.std(axis=0)
    Z = (A - mu) / np.where(sd > 1e-9, sd, 1.0)
    u, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    pc1 = u[:, 0] * s[0]
    K = Z.shape[1]
    r = np.zeros(K)
    for k in range(K):
        if np.std(Z[:, k]) > 1e-9 and np.std(pc1) > 1e-9:
            r[k] = np.corrcoef(Z[:, k], pc1)[0, 1]
    r = np.nan_to_num(r)
    # FASE DO PC1. O sinal que o SVD devolve e arbitrario, e sem fixa-lo o
    # TETO saiu bimodal (35,2% +- 34,5) mesmo com a regra do PROPRIO material.
    #
    # A 1a tentativa fixou pelo canal de maior |r| — e NAO resolveu (39,9% +-
    # 36,0). O diagnostico medido explica: em A os canais sao 1,7s e -0,6s,
    # ambos com ruido 0,05, entao |r| da [0,9813 ; 0,9830] — magnitudes
    # praticamente iguais e SINAIS OPOSTOS. O argmax desempatava na terceira
    # casa decimal entre dois canais contrarios, e a orientacao invertia em
    # 30% dos episodios de A e 45% dos de B (medido: papel 0 caiu em ch0 em
    # 14/20 e 9/20).
    #
    # CONSERTO: entre os canais empatados em magnitude, desempatar pelo MENOR
    # INDICE, que e estavel dentro de um material. Agora a fase e
    # deterministica. Limite honesto: se o canal 0 de um material fosse ruido
    # puro, esta convencao escolheria mal — aqui ele e o mais forte nos dois.
    m = float(np.max(np.abs(r)))
    empatados = [k for k in range(K) if abs(r[k]) >= 0.95 * m]
    dominante = min(empatados) if empatados else int(np.argmax(np.abs(r)))
    if r[dominante] < 0:
        pc1 = -pc1
        r = -r
    Zo = Z * np.sign(np.where(np.abs(r) < 1e-9, 1.0, r))
    ordem = np.argsort(-np.abs(r))
    papel = np.empty(K, dtype=int)
    papel[ordem] = np.arange(K)
    return Zo, papel


def _niv(x: float) -> int:
    return int(np.clip(round(x * 2), -NIVEIS, NIVEIS))


def termos_do_instante(e: Espaco, Zo, papel, t: int, a: int) -> List[np.ndarray]:
    """
    Amarra o alvo ao NIVEL e tambem a MUDANCA.

    O CONSERTO QUE A MATRIZ NOMEOU ([[nuvem-matriz-encoder]]): a versao anterior
    so amarrava NIVEL nos atrasos 0 e 1. Uma regra sobre derivada ficava
    inexpressivel, e as duas transformacoes de mudanca deram teto 62,5% e 59,6%
    — o recall nao as expressava nem com o material PROPRIO. As de nivel deram
    teto 81,6% e 79,6% e compuseram (+16,7 e +22,5).

    Agora entram tres termos por canal: nivel, SINAL DA PRIMEIRA DIFERENCA e
    SINAL DA SEGUNDA. Os sinais, e nao as magnitudes, porque magnitude de
    diferenca carrega escala e escala e superficie — o mesmo motivo que fez o
    nivel bruto agrupar por material.

    Isto e pre-requisito para EEG e dinamica neural, que sao quase inteiramente
    sobre mudanca.
    """
    out = []
    va = e.atomo(f"a:{a}")
    K = Zo.shape[1]
    for d in ATRASOS:
        if t - d < 0:
            continue
        for k in range(K):
            v = e.bind(e.atomo(f"p:{papel[k]}"),
                       e.bind(e.atomo(f"n:{_niv(Zo[t - d, k])}"), va))
            out.append(e.desloca(v, d))
    for k in range(K):
        if t >= 1:
            d1 = 1 if Zo[t, k] > Zo[t - 1, k] else -1
            out.append(e.desloca(e.bind(e.atomo(f"p:{papel[k]}"),
                                        e.bind(e.atomo(f"d1:{d1}"), va)), 5))
        if t >= 2:
            a1 = Zo[t, k] - Zo[t - 1, k]
            a0 = Zo[t - 1, k] - Zo[t - 2, k]
            d2 = 1 if a1 > a0 else -1
            out.append(e.desloca(e.bind(e.atomo(f"p:{papel[k]}"),
                                        e.bind(e.atomo(f"d2:{d2}"), va)), 7))
    return out


def codifica(e: Espaco, A: np.ndarray, y: np.ndarray) -> np.ndarray:
    Zo, papel = orienta_pc1(A)
    termos: List[np.ndarray] = []
    for t in range(A.shape[0]):
        termos += termos_do_instante(e, Zo, papel, t, int(y[t]))
    return feixe(termos, e.dim)


# ────────────────────────────────── o gerador: recall associativo
def _tabela(e: Espaco, R: np.ndarray, K: int) -> Dict[tuple, float]:
    """
    Pre-computa sim(termo, R) para as poucas combinacoes possiveis.

    Cobre NIVEL e tambem as DIFERENCAS, em par com `termos_do_instante`: se a
    regra guarda termos que o preditor nunca consulta, o conserto do encoder
    nao chega na predicao.
    """
    tab = {}
    for d in ATRASOS:
        for p in range(K):
            for n in range(-NIVEIS, NIVEIS + 1):
                for a in (-1, 1):
                    v = e.bind(e.atomo(f"p:{p}"),
                               e.bind(e.atomo(f"n:{n}"), e.atomo(f"a:{a}")))
                    tab[(d, p, n, a)] = sim(e.desloca(v, d), R)
    for nome, desl in (("d1", 5), ("d2", 7)):
        for p in range(K):
            for sg in (-1, 1):
                for a in (-1, 1):
                    v = e.bind(e.atomo(f"p:{p}"),
                               e.bind(e.atomo(f"{nome}:{sg}"), e.atomo(f"a:{a}")))
                    tab[(nome, p, sg, a)] = sim(e.desloca(v, desl), R)
    return tab


def preve(e: Espaco, R: np.ndarray, A: np.ndarray,
          flip: float = 1.0) -> np.ndarray:
    """Preve o alvo instante a instante por recall contra R, com fase DADA."""
    K = A.shape[1]
    tab = _tabela(e, R, K)
    Zo, papel = orienta_pc1(A)
    Zo = Zo * flip
    pred = np.zeros(A.shape[0], dtype=int)
    for t in range(A.shape[0]):
        esc = {}
        for a in (-1, 1):
            tot = 0.0
            for d in ATRASOS:
                if t - d < 0:
                    continue
                for k in range(K):
                    tot += tab[(d, int(papel[k]), _niv(Zo[t - d, k]), a)]
            for k in range(K):
                if t >= 1:
                    sg = 1 if Zo[t, k] > Zo[t - 1, k] else -1
                    tot += tab[("d1", int(papel[k]), sg, a)]
                if t >= 2:
                    sg = 1 if (Zo[t, k] - Zo[t - 1, k]) >                               (Zo[t - 1, k] - Zo[t - 2, k]) else -1
                    tot += tab[("d2", int(papel[k]), sg, a)]
            esc[a] = tot
        pred[t] = 1 if esc[1] >= esc[-1] else -1
    return pred


def calibra(e: Espaco, R_ponte: np.ndarray,
            rotulados: Sequence[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Descobre a fase que liga o material de origem ao de destino.

    CONSERTO 2. A primeira versao resolvia a fase pela MARGEM total entre os
    candidatos, e isso e degenerado: virar Zo mapeia nivel n -> -n, e a regra
    contem (n,+1) e (-n,-1) com forca parecida, entao as duas orientacoes dao
    margem equivalente. O criterio nao discriminava, e por isso ate o TETO saiu
    bimodal (35,2% com desvio 34,5).

    A ancora certa ja estava na loja. Ela tem A|regime, A|deriva e B|deriva —
    a quadrupla analogica com UMA celula faltando. O par `deriva`, que existe
    nos DOIS materiais e e ROTULADO, fixa o mapeamento A<->B: escolhe-se a fase
    que faz a regra de A|deriva prever melhor os B|deriva GUARDADOS. Depois a
    mesma fase serve para levar A|regime a B|regime.

    Nada aqui toca o alvo de teste: a calibragem usa so dado ja armazenado.
    E isto e a analogia se completando — A:deriva :: B:deriva diz como A vira
    B, e entao A:regime diz o que B:regime deve ser.
    """
    melhor, melhor_ac = 1.0, -1.0
    for flip in (1.0, -1.0):
        acs = [float(np.mean(preve(e, R_ponte, A, flip) == y))
               for A, y in rotulados]
        m = statistics.mean(acs)
        if m > melhor_ac:
            melhor, melhor_ac = flip, m
    return melhor


def acuracia(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


# ────────────────────────────────────────────────── o experimento
def experimento() -> None:
    e = Espaco()
    rng = random.Random(SEMENTE)

    print("=" * 78)
    print("NUVEM — a transformacao atravessa o material?")
    print("=" * 78)
    print(f"  loja: A|regime, A|deriva, B|deriva ({N_LOJA} cada). SEM B|regime.")
    print(f"  tarefa: prever o alvo de {N_TESTE} episodios NOVOS de B|regime\n")

    def regra(material, transf, n=N_LOJA):
        vs = []
        for _ in range(n):
            A, y = episodio(rng, material, transf)
            vs.append(codifica(e, A, y))
        return feixe(vs, e.dim)

    R_comp = regra("A", "regime")                 # a regra vinda do OUTRO material
    R_teto = regra("B", "regime")                 # teto: o proprio material
    R_nulo = regra("A", "ruido")                  # nulo: alvo sorteado

    # A PONTE: A|deriva + os B|deriva GUARDADOS fixam a fase que liga A a B.
    R_ponte = regra("A", "deriva")
    b_deriva = [episodio(rng, "B", "deriva") for _ in range(N_LOJA)]
    fase = calibra(e, R_ponte, b_deriva)
    print(f"  ponte A:deriva :: B:deriva  ->  fase {fase:+.0f}")
    print("")
    # PLACEBO CONSERTADO. A 1a versao sorteava da loja INTEIRA — que contem
    # A|regime, ou seja, a propria regra em teste. Deu 89,1%, empatando com o
    # teto, e o script concluiu "nao transfere" a partir de um controle
    # contaminado por construcao. O controle que discrimina e a regra montada
    # SO com a transformacao ERRADA: se ela tambem previr ~88%, qualquer regra
    # real serve e nao houve transferencia de transformacao.
    R_plac = regra("A", "deriva")

    testes = [episodio(rng, "B", "regime") for _ in range(N_TESTE)]
    maioria = statistics.mean(max(np.mean(y == 1), np.mean(y == -1))
                              for _A, y in testes)

    res = {}
    for rot, R in (("TETO    B|regime", R_teto),
                   ("COMPOSTO A|regime -> B", R_comp),
                   ("controle A|deriva (transf. errada)", R_plac),
                   ("nulo     alvo sorteado", R_nulo)):
        f = 1.0 if rot.startswith("TETO") else fase
        accs = [acuracia(preve(e, R, A, f), y) for A, y in testes]
        m, sd = statistics.mean(accs), statistics.pstdev(accs)
        res[rot.split()[0]] = m
        print(f"  {rot:24s}  {100*m:5.1f}%  +- {100*sd:4.1f}")
    print(f"  {'classe majoritaria':24s}  {100*maioria:5.1f}%")
    print(f"  {'acaso':24s}   50.0%\n")

    teto, comp, plac, nulo = (res["TETO"], res["COMPOSTO"],
                              res["controle"], res["nulo"])
    print("=" * 78)
    if teto <= 0.60:
        print(f"L0 FALHOU: o teto e {100*teto:.1f}%. Nem com a regra do proprio")
        print("   material o mecanismo expressa a transformacao. Nada abaixo")
        print("   disso e interpretavel — o defeito e do RECALL, nao da")
        print("   composicao.")
        return
    print(f"L0 . teto {100*teto:.1f}% > 60%: o mecanismo expressa a regra.")
    if nulo > 0.60:
        print(f"L3 . o NULO tambem passa ({100*nulo:.1f}%): e DUTO. A predicao")
        print("     vem da estrutura do fluxo, nao da regra transferida.")
        return
    if comp > plac + 0.05 and comp > 0.55:
        print(f"L1 . A NUVEM ESTILHACA. Composto {100*comp:.1f}% contra")
        print(f"     placebo {100*plac:.1f}% e nulo {100*nulo:.1f}%.")
        print("     A transformacao aprendida no material A previu o material")
        print("     B sofrendo o que B nunca foi visto sofrendo.")
    elif comp < 0.45:
        print(f"L1? INVERTIDO: {100*comp:.1f}%, abaixo do acaso. A assinatura")
        print("     pre-registrada de sinal global resolvido ao contrario —")
        print("     ha transferencia, com a orientacao trocada.")
    else:
        print(f"L2 . nao transfere: composto {100*comp:.1f}%, placebo "
              f"{100*plac:.1f}%.")


if __name__ == "__main__":
    experimento()
