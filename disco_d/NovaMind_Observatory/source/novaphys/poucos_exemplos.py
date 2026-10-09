"""
novaphys.poucos_exemplos — a ponte analogica entre PESSOAS
============================================================

DE ONDE VEM
-------------
De [[motivo-e-pessoal]]: congelado em 21 voluntarios e solto em 9 nunca vistos,
o sistema acerta 0,545 contra 0,187 de maioria — transfere —, mas o ganho
TEMPORAL cai de +0,210 para +0,075. E nao e falta de cobertura: os motivos
disparam igual no teste (92,6% x 91,5%). **O motivo de passada e assinatura
INDIVIDUAL de marcha, nao o conceito "andar".**

E do usuario, que definiu como marca de mente "precisar de poucos exemplos pra
compreender".

A IDEIA, E ELA NAO CONSTROI NADA NOVO
---------------------------------------
Se o motivo e pessoal, o conceito esta um nivel ACIMA dele — no que os motivos
de pessoas diferentes tem em comum. E achar isso e exatamente a ponte que fez a
nuvem estilhacar em [[nuvem-estilhaca]], com "material" trocado por "pessoa":

    pessoa_A:sentar :: pessoa_B:sentar   fixa como o espaco de B mapeia no de A
    pessoa_A:andar                       entao diz o que pessoa_B:andar e

Concretamente: alguns exemplos ROTULADOS da pessoa nova dizem em quais grupos
congelados o corpo dela cai, e isso RE-ANCORA o mapeamento grupo->atividade
para ela. Nada e re-treinado; so o dicionario de nomes muda, por pessoa.

AS FAMILIAS SAO DISJUNTAS DE PROPOSITO
----------------------------------------
  PONTE  atividades estaticas: SITTING, STANDING, LAYING
  ALVO   familia de andar: WALKING, WALK_UP, WALK_DOWN

O alvo e onde a transferencia mais falhou (WALK_UP em 0,292). Se a ponte
vazasse para o alvo, o teste seria treinar no teste.

PRE-REGISTRO
--------------
  Q0  PRE-CONDICAO. Sem adaptacao, a acuracia na familia ALVO reproduz a linha
      de base medida (~0,49 nas tres de andar). Senao o instrumento mudou.
  Q1  com k exemplos de ponte por pessoa, a acuracia no ALVO sobe >= 0,10 E
      supera o placebo por >= 2 desvios
      -> poucos exemplos bastam para o sistema reancorar uma pessoa nova
  Q2  sem ganho -> a especificidade pessoal nao e corrigivel por reancoragem;
      o problema esta no motivo, nao no mapeamento
  Q3  ganho mas o placebo tambem sobe -> o ganho e de ter mais dado qualquer,
      nao da PONTE

  PLACEBO PAREADO: os MESMOS k exemplos, com os ROTULOS EMBARALHADOS. Mesma
  quantidade de dado, mesma pessoa, sem a informacao de correspondencia.
  10 embaralhamentos, desvio reportado. (A licao de 18/09: controle com a mesma
  oportunidade, nao so o mesmo orcamento.)

  Varre-se k = 1, 3, 6, 12 exemplos por atividade de ponte, para ter CURVA e
  nao um ponto — e a curva responde "quao poucos e 'poucos'".

PREVISAO: Q1, 0,6. Mecanismo: as revocacoes por atividade sugerem erro
SISTEMATICO e nao difuso (WALK_UP 0,292 enquanto SITTING da 0,725), e erro
sistematico e exatamente o que reancoragem conserta. Risco nomeado: a ponte usa
atividades ESTATICAS e o alvo e DINAMICO; se a correcao que a ponte revela for
especifica de postura, ela nao atravessa para marcha e Q2 sai por design meu,
nao por limite do sistema.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.har import NOMES, RAIZ, carrega
from novaphys.har_pessoas import K_GRUPOS, N_TREINO, PipelineCongelado, SEMENTE
from novaphys.hdc_indice import Espaco

PONTE = (4, 5, 6)          # SITTING, STANDING, LAYING
ALVO = (1, 2, 3)           # WALKING, WALK_UP, WALK_DOWN
KS = (1, 3, 6, 12)
N_EMB = 10


def sujeitos(split: str, n_max: int) -> np.ndarray:
    return np.loadtxt(os.path.join(RAIZ, split, f"subject_{split}.txt"),
                      max_rows=n_max, dtype=int)


def grupos_de(p: PipelineCongelado, X: np.ndarray) -> np.ndarray:
    """Em qual grupo congelado cada janela cai. Nada e reajustado."""
    V = np.stack([p._vetor(s) for s in p._cadeias(X)])
    M = p.medoides.astype(np.float32)
    return np.argmax(V.astype(np.float32) @ M.T, axis=1)


def experimento(n_teste: int = 1500) -> None:
    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr = carrega("train", N_TREINO)
    Xte, yte = carrega("test", n_teste)
    ste = sujeitos("test", n_teste)

    print("=" * 78)
    print("POUCOS EXEMPLOS — a ponte analogica entre PESSOAS")
    print("=" * 78)
    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    g = grupos_de(p, Xte)
    print(f"  congelado em 21 voluntarios; {len(set(ste.tolist()))} pessoas "
          f"novas, {len(Xte)} janelas")
    print(f"  PONTE  {[NOMES[a] for a in PONTE]}")
    print(f"  ALVO   {[NOMES[a] for a in ALVO]}  (familias disjuntas)\n")

    alvo_m = np.isin(yte, ALVO)
    base = float(np.mean(np.array([p.nome[int(c)] for c in g])[alvo_m]
                         == yte[alvo_m]))
    print(f"  Q0 — sem adaptacao, acuracia no ALVO: {base:.3f}")

    print("\n  k por atividade   com PONTE      com PLACEBO (10x)")
    print("  ---------------   -----------    -----------------")
    linhas = []
    for k in KS:
        acs, plc = [], []
        for suj in sorted(set(ste.tolist())):
            m_s = ste == suj
            # reancoragem: grupo -> atividade, pela MAIORIA nos exemplos de ponte
            def reancora(rot: np.ndarray) -> Dict[int, int]:
                """
                A PONTE DE VERDADE: um MAPEAMENTO entre grupos, nao um
                sobrescritor de nomes.

                CONSERTO. A 1a versao renomeava cada grupo usando SO janelas de
                ponte — entao todo grupo com alguma janela estatica virava
                'sentado/em pe/deitado', e as janelas de andar que caiam nele
                eram rotuladas como estaticas. A acuracia CAIU de 0,502 para
                0,451, e ponte e placebo deram identico porque o resultado nem
                dependia dos rotulos. Era bug meu, nao achado.

                Agora: para cada atividade de ponte, ve-se em que grupos o corpo
                DESTA pessoa cai (`orig`) e em que grupos o TREINO poe a MESMA
                atividade (`ref`). O par (orig -> ref) e a correspondencia. Esse
                mapeamento sigma e aplicado as outras atividades dela; grupos sem
                evidencia ficam como estavam.
                """
                sigma = {}
                for a in PONTE:
                    idx = np.where(m_s & (rot == a))[0][:k]
                    if len(idx) == 0:
                        continue
                    orig = Counter(g[idx].tolist()).most_common(1)[0][0]
                    ref = [c for c in range(K_GRUPOS) if p.nome[c] == a]
                    if ref and orig not in sigma:
                        sigma[orig] = ref[0]
                novo = {}
                for c in range(K_GRUPOS):
                    novo[c] = p.nome[sigma.get(c, c)]
                return novo

            mapa = reancora(yte)
            mm = m_s & alvo_m
            if mm.sum() == 0:
                continue
            acs.append(float(np.mean(
                np.array([mapa[int(c)] for c in g[mm]]) == yte[mm])))
            ps = []
            for s in range(N_EMB):
                r2 = np.random.default_rng(3000 + s)
                emb = yte.copy()
                pm = np.where(m_s & np.isin(yte, PONTE))[0]
                emb[pm] = r2.permutation(emb[pm])
                mapa_e = reancora(emb)
                ps.append(float(np.mean(
                    np.array([mapa_e[int(c)] for c in g[mm]]) == yte[mm])))
            plc.append(statistics.mean(ps))
        m_ac = statistics.mean(acs)
        m_pl, sd_pl = statistics.mean(plc), statistics.pstdev(plc)
        linhas.append((k, m_ac, m_pl, sd_pl))
        print(f"  {k:15d}   {m_ac:.3f}          {m_pl:.3f} +- {sd_pl:.3f}",
              flush=True)

    print("\n" + "=" * 78)
    melhor = max(linhas, key=lambda l: l[1])
    k, ac, pl, sd = melhor
    ganho = ac - base
    print(f"  melhor: k={k}, acuracia {ac:.3f} contra base {base:.3f} "
          f"({ganho:+.3f})")
    print(f"  placebo no mesmo k: {pl:.3f} +- {sd:.3f}")
    if ganho >= 0.10 and ac >= pl + 2 * max(sd, 1e-9):
        print(f"\nQ1 . POUCOS EXEMPLOS BASTAM: {k} por atividade de ponte")
        print(f"     levam o ALVO de {base:.3f} a {ac:.3f}, e o placebo com os")
        print(f"     MESMOS exemplos e rotulos embaralhados fica em {pl:.3f}.")
        print("     A ponte entre PESSOAS funciona como a ponte entre")
        print("     materiais que fez a nuvem estilhacar.")
    elif ganho >= 0.10:
        print(f"\nQ3 . ganha {ganho:+.3f} mas o placebo acompanha: e ter mais")
        print("     dado, nao a correspondencia da ponte.")
    else:
        print(f"\nQ2 . sem ganho ({ganho:+.3f}). A especificidade pessoal nao e")
        print("     corrigivel por reancoragem — o problema esta no MOTIVO,")
        print("     nao no mapeamento. E checar o risco nomeado: a ponte e")
        print("     estatica e o alvo e dinamico.")


if __name__ == "__main__":
    experimento()
