# PRE-REGISTRO — Factorio tem INCLINACAO ou PENHASCO?

17/09/2026, antes de rodar. PRIMEIRO experimento no dominio novo, e ele testa a
PREMISSA em vez de construir em cima dela.

## DE ONDE VEM

O nono experimento mediu a paisagem do ARC: trocar UM marcador de uma pilha
exata faz o erro saltar de 0 para mediana 46, em 96% dos casos, SEM valores
intermediarios. E penhasco. Isso explicou os oito nulos: construcao de operacao
nova nao tem gradiente para seguir.

A saida proposta foi "fazer abducao onde HA gradiente, e transportar". Factorio
foi escolhido por ter metrica continua. **Mas isso e suposicao minha e precisa
ser medida ANTES de qualquer agente rodar.**

## O TESTE (espelho exato do nono, agora em Factorio)

1. montar uma cadeia que produz (perfuratriz -> forno -> placa de ferro)
2. medir producao base em N ticks
3. PERTURBAR uma peca de cada vez: tirar combustivel, girar, mover 1 tile,
   remover um item do meio
4. medir de novo

## PORTOES

  I0  SANIDADE: a cadeia base produz > 0 placas em N ticks. Se nao produzir,
      o meu setup esta errado e nada abaixo vale.
  I1  INCLINACAO: as perturbacoes produzem valores INTERMEDIARIOS. Concretamente:
      >= 30% das perturbacoes dao producao entre 10% e 90% da base.
  I2  PENHASCO (o contrario): >= 80% das perturbacoes vao para 0 ou para a base,
      sem meio-termo — igual ao ARC (96% saltaram).

## PREVISAO HONESTA, antes de rodar: I1 passa, 70%

Razao: producao e taxa, e taxa e continua por construcao — meia perfuratriz
alimentando um forno da metade das placas. Razao da duvida (30%): Factorio tem
MUITA dependencia binaria. Sem combustivel a perfuratriz nao gira: ZERO, nao
metade. Esteira virada para o lado errado: ZERO. Pode ser que a "continuidade"
da producao por minuto esconda um grafo de dependencias que e binario em cada
elo — e ai Factorio seria penhasco com fachada de ladeira.

**Se I2 passar:** minha premissa esta errada, Factorio nao resolve o problema do
gradiente, e eu digo isso na primeira hora em vez da decima. A explicacao dos
nove nulos continua valendo como DESCRICAO do ARC, mas a SAIDA que eu propus
perde o endereco.
