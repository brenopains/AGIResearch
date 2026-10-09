# PRE-REGISTRO — bits no treino preve acerto no teste?

17/09/2026, antes de rodar.

## DE ONDE VEM

Setimo nulo (`residuo_dirigido`): tres vocabularios novos, NENHUM entra em pilha
vencedora — perdem em BITS, nao em orcamento. E residuo de treino ZERO em 40 de
80 tarefas com teste errado: a pilha explica o treino PERFEITAMENTE e erra.
Em metade das falhas a DECORACAO ganha em bits.

## POR QUE NAO E LARGURA DE FEIXE

Alargar o feixe conserta "o criterio ranqueou mal entre candidatas". Nao conserta
ajuste PERFEITO no treino: com residuo 0 nao existe candidata mais barata que
tambem explique. O criterio nao escolhe mal — o otimo do treino esta errado.

## A MEDIDA

DEIXE-UM-DE-FORA nos pares de TREINO. Para cada tarefa com >= 3 pares:
  - esconde o par i, monta a pilha nos demais (mesmo teto), preve o par i
  - `gen_treino` = fracao dos pares escondidos previstos certo

Isso NAO usa a saida de teste. E um sinal que o sistema JA PODERIA calcular.

## PORTOES

  C0  SANIDADE: as 12 que a base resolve tem gen_treino alta (mediana >= 0,6)
  C1  O SINAL SEPARA: AUC de gen_treino prevendo "resolve o teste" >= 0,75
      -> existe sinal interno gratuito de "eu decorei" que o sistema NAO USA,
         e isso e conserto construivel
  C2  ESPECIFICO DA DECORACAO: entre as tarefas de residuo ZERO que falham,
      gen_treino mediana <= 0,34 (elas decoram e o deixe-um-de-fora denuncia)
  C3  NULO: AUC de um sinal SORTEADO com a mesma distribuicao ~ 0,50

## PREVISAO HONESTA, antes de rodar

C1 passa com ~55% de confianca. Razao da esperanca: decoracao e exatamente o que
deixe-um-de-fora detecta, e e o modo de falha de metade das tarefas. Razao da
duvida: com 3 pares de treino, tirar um muda MUITO a evidencia — a pilha pode
falhar no par escondido por subamostragem e nao por decoracao, e ai o sinal e
ruidoso demais para separar.

## SE C1 FALHAR

O sistema nao tem sinal interno de generalizacao acessivel por este caminho, e o
conserto do criterio nao e "usar o que ja da para calcular". Ai a unica saida
medida e mudar o CRITERIO em si (bits no treino -> bits com penalidade de
complexidade que o proprio treino nao paga), e isso e mudanca de nucleo, nao de
camada.
