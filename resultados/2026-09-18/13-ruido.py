"""
O PISO DE RUIDO QUE EU NUNCA MEDI.

`multi` rotula 100 motivos com 100 rotulos distintos: e uma BIJECAO do literal.
Bijecao nao pode carregar informacao nova — so troca a string, logo o hash, logo
o vetor aleatorio do atomo. Entao 0,585 x 0,537 e, por construcao, ruido.

Este script mede QUANTO ruido. Dez rotulacoes literais salgadas ("mot1:", ...),
todas bijecoes uma da outra, todas com exatamente a mesma informacao. A dispersao
entre elas e o piso de ruido do pipeline inteiro — e nenhuma diferenca menor que
isso, em NENHUMA medida anterior, significa coisa alguma.

PRE-REGISTRO
  N1  desvio >= 0,03 -> o piso engole as comparacoes de hoje e varias precisam
      ser reavaliadas contra ele
  N2  desvio <  0,03 -> o piso e pequeno e as diferencas medidas sobrevivem
"""
import statistics
import numpy as np
from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, braco, canon

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)

print("="*78)
print("PISO DE RUIDO — dez bijecoes do MESMO conteudo")
print("="*78)
print()
acs = []
for s in range(10):
    rot = (lambda t: lambda w: ["mot%d:%s" % (t, w)])(s)
    a = braco(e, ped_tr, ped_te, ytr, yte, rot, K_GRUPOS)
    acs.append(a)
    print("  sal %2d   %.3f" % (s, a), flush=True)
m, sd = statistics.mean(acs), statistics.pstdev(acs)
print()
print("  media %.3f   desvio %.3f   faixa %.3f a %.3f"
      % (m, sd, min(acs), max(acs)))
print()
print("="*78)
print("  reavaliando o que eu media hoje contra ESTE piso:")
for nome, v in (("literal (sal original)", 0.537), ("canonico", 0.438),
                ("ambos", 0.467), ("multi (bijecao)", 0.585),
                ("placebo medio", 0.479)):
    z = (v - m)/sd if sd > 0 else 0.0
    print("    %-22s %.3f   %+.2f desvios do piso" % (nome, v, z))
print()
if sd >= 0.03:
    print("N1 . O PISO ENGOLE AS COMPARACOES. Trocar SO o nome dos atomos, sem")
    print("     mudar uma unica bit de informacao, move a acuracia +-%.3f." % sd)
else:
    print("N2 . piso pequeno (%.3f); as diferencas medidas sobrevivem." % sd)
