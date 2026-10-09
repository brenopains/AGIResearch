"""
O TRIANGULO — onde mora o sinal, e por que ele e pessoal.

Tres bracos IDENTICOS em tudo (mesma normalizacao, mesmos centroides, mesmo
lexico, MESMA analise de Viterbi, mesmo agrupamento, mesma nomeacao). Diferem so
no que cada um JOGA FORA do motivo encontrado:

  LITERAL    quais simbolos E em que ordem      abcab -> "abcab"
  CANONICO   so a ordem (a variavel ligada)     abcab -> "0,1,2,0,1"
  MULTI      so os simbolos (sem ordem)         abcab -> "a,a,b,b,c"

Se o sinal sobrevive numa das projecoes, aquela e o conceito compartilhado. Se
so sobrevive no produto das duas, o sinal esta no TERMO DE INTERACAO — e o termo
de interacao e justamente o que [[motivo-e-pessoal]] mediu como individual.

PRE-REGISTRO
  R1  alguma projecao >= literal - 0,02  -> ha conceito numa das marginais
  R2  as duas projecoes < literal - 0,02 -> o sinal e a INTERACAO, e transferir
      entre pessoas exige separar interacao de assinatura, nao marginalizar
"""
import numpy as np
from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, braco, canon

def multi(w):
    return ",".join(sorted(w))

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)
mot = sorted({w for pe in ped_tr for w in pe})

print("="*78)
print("O TRIANGULO — o que sobra quando se joga fora metade do motivo")
print("="*78)
print()
print("  braco       joga fora          rotulos   acuracia")
print("  ---------   ----------------   -------   --------")
res = {}
for nome, joga, rot in (
        ("literal ", "nada            ", lambda w: ["mot:"+w]),
        ("canonico", "quais simbolos  ", lambda w: ["pad:"+canon(w)]),
        ("multi   ", "a ordem         ", lambda w: ["mul:"+multi(w)])):
    n = len({rot(w)[0] for w in mot})
    a = braco(e, ped_tr, ped_te, ytr, yte, rot, K_GRUPOS)
    res[nome.strip()] = a
    print("  %s    %s   %4d      %.3f" % (nome, joga, n, a), flush=True)

lit = res["literal"]
print()
print("="*78)
if max(res["canonico"], res["multi"]) >= lit - 0.02:
    print("R1 . uma marginal preserva o sinal — ha conceito compartilhado nela.")
else:
    print("R2 . AS DUAS MARGINAIS PERDEM (canonico %+.3f, multi %+.3f)."
          % (res["canonico"]-lit, res["multi"]-lit))
    print("     O sinal esta no TERMO DE INTERACAO entre QUAIS simbolos e EM QUE")
    print("     ORDEM — e e exatamente esse termo que motivo-e-pessoal mediu")
    print("     como assinatura individual. Marginalizar nao abstrai: apaga.")
