"""
novaphys.abstrai_motivo — o conceito ENTRE as pessoas, nao o alinhamento de uma a outra
========================================================================================

DE ONDE VEM
-------------
De [[motivo-e-pessoal]]: congelado em 21 voluntarios e solto em 9 novos, o
sistema acerta 0,545 — transfere —, mas o ganho TEMPORAL desaba de +0,210 para
+0,075. O motivo de passada e assinatura INDIVIDUAL de marcha, nao o conceito
"andar".

E de `poucos_exemplos.py`, que tentou o conserto obvio — ALINHAR a pessoa nova
as antigas por reancoragem — e deu Q2 duas vezes (-0,051, com ponte identica ao
placebo). A conclusao registrada foi que o problema esta no MOTIVO, nao no
mapeamento.

A IDEIA, E ELA E O OPOSTO DA ANTERIOR
---------------------------------------
Alinhar pergunta "como traduzo B para A?". **Abstrair** pergunta "o que A e B
tem em comum que nenhum dos dois e sozinho?".

E aqui ha uma resposta concreta e barata. Um motivo hoje e uma STRING LITERAL de
simbolos, tipo `abcab`. Se a pessoa B anda com `adcad`, os dois atomos sao
ortogonais no HDC e nao se falam — embora sejam **o mesmo padrao de repeticao**:
primeiro simbolo, segundo, terceiro, volta ao primeiro, volta ao segundo.

Renomear por ordem de primeira ocorrencia expoe isso:

    abcab  ->  0,1,2,0,1
    adcad  ->  0,1,2,0,1      MESMO conceito

E ISSO E EXATAMENTE O QUANTIFICADOR
-------------------------------------
[[objetivo-precisa-de-quantificador]] mediu que features proposicionais colidem
e travam em 95,06%, e que UM quantificador leva a 100% e transfere. `mot:abcab`
e proposicional: nomeia constantes. `pad:0,1,2,0,1` diz "existem simbolos x,y,z
tais que o motivo e x y z x y" — e uma variavel ligada.

Se a lei vale, a mesma correcao deve funcionar aqui, num dominio que nao tem
nada a ver com xadrez.

TRES BRACOS, MESMO LEXICO, MESMA ANALISE
------------------------------------------
  LITERAL   atomo `mot:<string>`                     (o sistema de hoje)
  PADRAO    atomo `pad:<forma canonica>`             (so a abstracao)
  AMBOS     os dois atomos no mesmo feixe            (abstrair sem destruir)

Media, desvio, centroides da quantizacao e o lexico sao ajustados UMA VEZ nos 21
voluntarios de treino e compartilhados pelos tres bracos. A analise de Viterbi
tambem e feita uma vez: os bracos diferem SO no rotulo dado a cada motivo achado.
Entao qualquer diferenca e da abstracao e de mais nada.

PRE-REGISTRO
--------------
  A0  PRE-CONDICAO. O braco LITERAL reproduz a base medida (0,545 +- 0,02). E a
      abstracao precisa COLAPSAR de fato: numero de padroes < numero de motivos.
      Se nao colapsa, nao ha o que testar.
  A0b DIAGNOSTICO (nao e portao). Concordancia entre pessoas do perfil de
      motivos por atividade, literal x canonico. Se a abstracao faz o que eu
      digo, essa concordancia SOBE. Mede o mecanismo direto, sem depender do
      desfecho.
  A1  acuracia em pessoas novas >= LITERAL + 0,05 E >= placebo + 2 desvios
      -> abstrair atravessa onde alinhar falhou; o quantificador vale fora do
         xadrez
  A2  sem ganho -> o conceito compartilhado nao esta no padrao de repeticao
  A3  ganho mas o placebo acompanha -> e o COLAPSO que ajuda (menos rotulos,
      feixe menos diluido), nao o CONTEUDO da abstracao

  PLACEBO PAREADO, e ele e exato: pega a particao que o padrao canonico induz no
  lexico e PERMUTA quais motivos recebem qual classe, preservando o multiconjunto
  de tamanhos de classe. Mesmo numero de rotulos, mesmas cardinalidades, zero
  estrutura. 10 permutacoes, desvio reportado.
  (A licao de [[vocabulario-grande-piora]]: o tamanho importa e o conteudo nao —
  ate que se prove o contrario. Este placebo e o que prova ou nao.)

PREVISAO: A1, 0,45 — abaixo de um a um. Mecanismo a favor: a lei do
quantificador ja atravessou uma vez e a colisao aqui e do mesmo tipo. Mecanismo
contra, e e serio: a forma canonica descarta QUAIS centroides, e os centroides
sao compartilhados entre pessoas (foram congelados no treino), entao parte da
informacao descartada era boa. `abab` de marcha e `cdcd` de deitado parado com
tremor de sensor viram o mesmo padrao. Se A2 sair por isso, o braco AMBOS e quem
mostra — ele nao pode perder para o LITERAL, ja que so acrescenta.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.har import NOMES, RAIZ, carrega
from novaphys.har_pessoas import K_GRUPOS, N_TESTE, N_TREINO, PipelineCongelado, SEMENTE
from novaphys.hdc_indice import Espaco, feixe
from novaphys.lingua import _viterbi

N_PERM = 10


def canon(w: str) -> str:
    """Forma canonica por ordem de primeira ocorrencia: a variavel ligada."""
    m: Dict[str, int] = {}
    saida = []
    for ch in w:
        if ch not in m:
            m[ch] = len(m)
        saida.append(m[ch])
    return ",".join(map(str, saida))


def analisa(p: PipelineCongelado, X: np.ndarray) -> List[List[str]]:
    """Viterbi UMA vez. Os bracos so trocam o rotulo, nao a analise."""
    prob = {w: 1.0 / len(p.lex) for w in p.lex}
    return [[w for w in _viterbi(s, p.lex, prob) if len(w) > 1]
            for s in p._cadeias(X)]


def vetores(e: Espaco, peds: Sequence[Sequence[str]],
            rotulo) -> np.ndarray:
    vazio = e.atomo("vazio")
    saida = []
    for ped in peds:
        t = [e.atomo(r) for w in ped for r in rotulo(w)]
        saida.append(feixe(t, e.dim) if t else vazio)
    return np.stack(saida)


def agrupa_e_nomeia(V: np.ndarray, y: np.ndarray, rng, k: int
                    ) -> Tuple[np.ndarray, Dict[int, int]]:
    """Passos (4) e (5) do pipeline congelado, identicos ao original."""
    Vf = V.astype(np.float32)
    idx = rng.choice(len(V), size=k, replace=False)
    atrib = np.zeros(len(V), dtype=int)
    for _ in range(15):
        atrib = np.argmax(Vf @ Vf[idx].T, axis=1)
        novos = []
        for c in range(k):
            m = np.where(atrib == c)[0]
            if len(m) == 0:
                novos.append(rng.integers(len(V)))
                continue
            novos.append(int(m[np.argmax((Vf[m] @ Vf[m].T).sum(1))]))
        novos = np.array(novos)
        if np.array_equal(np.sort(novos), np.sort(idx)):
            break
        idx = novos
    nome = {}
    for c in range(k):
        rot = y[atrib == c]
        nome[c] = Counter(rot.tolist()).most_common(1)[0][0] if len(rot) else 1
    return V[idx], nome


def braco(e: Espaco, ped_tr, ped_te, ytr, yte, rotulo, k: int) -> float:
    rng = np.random.default_rng(SEMENTE)
    Vtr = vetores(e, ped_tr, rotulo)
    med, nome = agrupa_e_nomeia(Vtr, ytr, rng, k)
    Vte = vetores(e, ped_te, rotulo)
    g = np.argmax(Vte.astype(np.float32) @ med.astype(np.float32).T, axis=1)
    return float(np.mean(np.array([nome[int(c)] for c in g]) == yte))


def concordancia(peds, y, suj, rotulo) -> float:
    """
    A0b: o quanto pessoas diferentes usam os MESMOS rotulos na MESMA atividade.
    Por atividade, monta o perfil de rotulos de cada pessoa e tira o cosseno
    medio entre pares de pessoas. Nao depende do desfecho.
    """
    perfis = defaultdict(lambda: defaultdict(Counter))
    for ped, a, s in zip(peds, y.tolist(), suj.tolist()):
        for w in ped:
            for r in rotulo(w):
                perfis[a][s][r] += 1
    cos = []
    for a, porp in perfis.items():
        pessoas = sorted(porp)
        for i in range(len(pessoas)):
            for j in range(i + 1, len(pessoas)):
                u, v = porp[pessoas[i]], porp[pessoas[j]]
                comum = set(u) & set(v)
                num = sum(u[r] * v[r] for r in comum)
                den = (sum(c * c for c in u.values()) ** 0.5 *
                       sum(c * c for c in v.values()) ** 0.5)
                if den > 0:
                    cos.append(num / den)
    return statistics.mean(cos) if cos else 0.0


def experimento() -> None:
    import os

    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr = carrega("train", N_TREINO)
    Xte, yte = carrega("test", N_TESTE)
    str_ = np.loadtxt(os.path.join(RAIZ, "train", "subject_train.txt"),
                      max_rows=N_TREINO, dtype=int)

    print("=" * 78)
    print("ABSTRAI MOTIVO — o conceito ENTRE as pessoas")
    print("=" * 78)

    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    print(f"  congelado nos 21 de treino: {len(p.lex)} entradas de lexico")

    ped_tr = analisa(p, Xtr)
    ped_te = analisa(p, Xte)
    print(f"  Viterbi feito UMA vez: {len(ped_tr)} + {len(ped_te)} janelas\n")

    motivos = sorted({w for ped in ped_tr for w in ped})
    padroes = sorted({canon(w) for w in motivos})
    print(f"  A0 — colapso: {len(motivos)} motivos literais -> "
          f"{len(padroes)} padroes  ({len(padroes)/len(motivos):.2f}x)")
    if len(padroes) >= len(motivos):
        print("       NAO COLAPSA. Nada a testar.")
        return

    c_lit = concordancia(ped_tr, ytr, str_, lambda w: [f"mot:{w}"])
    c_pad = concordancia(ped_tr, ytr, str_, lambda w: [f"pad:{canon(w)}"])
    print(f"  A0b — concordancia entre pessoas na mesma atividade:")
    print(f"        literal {c_lit:.3f}   canonico {c_pad:.3f}   "
          f"({c_pad - c_lit:+.3f})\n")

    ac_lit = braco(e, ped_tr, ped_te, ytr, yte,
                   lambda w: [f"mot:{w}"], K_GRUPOS)
    print(f"  LITERAL (o sistema de hoje)   {ac_lit:.3f}", flush=True)
    if abs(ac_lit - 0.545) > 0.02:
        print(f"       ATENCAO: A0 esperava 0,545 +- 0,02. O instrumento mudou.")

    ac_pad = braco(e, ped_tr, ped_te, ytr, yte,
                   lambda w: [f"pad:{canon(w)}"], K_GRUPOS)
    print(f"  PADRAO  (so a abstracao)      {ac_pad:.3f}", flush=True)

    ac_amb = braco(e, ped_tr, ped_te, ytr, yte,
                   lambda w: [f"mot:{w}", f"pad:{canon(w)}"], K_GRUPOS)
    print(f"  AMBOS   (abstrai sem destruir){ac_amb:.3f}", flush=True)

    # placebo: mesma particao, classes permutadas entre os motivos
    plc = []
    for s in range(N_PERM):
        r2 = np.random.default_rng(7000 + s)
        classes = [canon(w) for w in motivos]
        perm = r2.permutation(len(motivos))
        mapa = {motivos[i]: classes[perm[i]] for i in range(len(motivos))}
        plc.append(braco(e, ped_tr, ped_te, ytr, yte,
                         lambda w: [f"pad:{mapa.get(w, w)}"], K_GRUPOS))
        print(f"    placebo {s+1}/{N_PERM}: {plc[-1]:.3f}", flush=True)
    m_pl, sd_pl = statistics.mean(plc), statistics.pstdev(plc)
    print(f"\n  PLACEBO (mesmo colapso, sem estrutura)  "
          f"{m_pl:.3f} +- {sd_pl:.3f}")

    melhor_nome, melhor = max([("PADRAO", ac_pad), ("AMBOS", ac_amb)],
                              key=lambda t: t[1])
    print("\n" + "=" * 78)
    print(f"  melhor abstracao: {melhor_nome} {melhor:.3f}")
    print(f"  contra literal {ac_lit:.3f} ({melhor - ac_lit:+.3f}) "
          f"e placebo {m_pl:.3f} ({melhor - m_pl:+.3f})")
    barra = m_pl + 2 * max(sd_pl, 1e-9)
    if melhor >= ac_lit + 0.05 and melhor >= barra:
        print(f"\nA1 . ABSTRAIR ATRAVESSA onde alinhar falhou. O rotulo canonico")
        print(f"     e uma variavel ligada, e a lei do quantificador vale fora")
        print(f"     do xadrez: {ac_lit:.3f} -> {melhor:.3f}, placebo {m_pl:.3f}.")
    elif melhor >= ac_lit + 0.05:
        print(f"\nA3 . ganha {melhor - ac_lit:+.3f} mas o placebo acompanha "
              f"({m_pl:.3f}):")
        print("     e o COLAPSO, nao o conteudo. 2a aparicao da lei de")
        print("     vocabulario-grande-piora.")
    else:
        print(f"\nA2 . sem ganho ({melhor - ac_lit:+.3f}). O conceito compartilhado")
        print("     nao esta no padrao de repeticao. Checar o risco nomeado:")
        print("     se AMBOS >= LITERAL, a abstracao nao machuca e o problema e")
        print("     que ela nao acrescenta; se AMBOS < LITERAL, ela dilui.")


if __name__ == "__main__":
    experimento()
