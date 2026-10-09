# PRE-REGISTRO — construir o termo OLHANDO O RESIDUO

Escrito 17/09/2026, antes de implementar. Desenho do usuario, adotado quase
integral.

## A HIPOTESE (nao lei — 8 pontos a sustentam, e so isso)

Seis nulos pre-registrados ofereceram VOCABULARIO ao sistema (propriedades,
operadores da LLM, classe raio, biblioteca, reordenacao, prior). Dois sucessos
CONSTRUIRAM o termo a partir do dado compondo operacoes dadas (h = x*vy - y*vx;
n^ = rot(v^)). Hipotese: a diferenca esta no ERRO PARTICIPAR da construcao.

## A PERGUNTA, exata

Nao "consigo inventar propriedades?" mas:

    dado um residuo observavel, consigo CONSTRUIR uma expressao que reduza esse
    residuo usando apenas operacoes primitivas disponiveis?

## O DUTO

```
tarefa -> codec atual -> RESIDUO (so nos pares de TREINO)
       -> observaveis das celulas do residuo (so da ENTRADA)
       -> compor operacoes primitivas
       -> candidatos CONSTRUIDOS
       -> MDL / verificador
       -> codec atualizado -> reavaliar residuo
```

**Protecao contra vazamento (exigencia do usuario):** o gerador so enxerga (a) os
pares de TREINO e (b) o residuo do codec atual sobre eles. A saida de TESTE nao
entra em nenhum ponto da geracao. Explicar o treino e o trabalho legitimo;
olhar o teste e trapaca.

## BRACOS, orcamento de composicao e teto de avaliacao CASADOS

  R1  residuo dirigido    observaveis nas celulas do residuo REAL desta tarefa
  R2  residuo embaralhado observaveis nas celulas do residuo de OUTRA tarefa
                          (mesma quantidade de informacao, dissociada)
  R3  sem residuo         mesmas operacoes e mesmo orcamento, celulas sorteadas
                          da grade em vez das do residuo

## METRICA PRIMARIA, brutal e unica

    tarefas NOVAS resolvidas entre as que a base NAO resolvia, com orcamento igual

Nada de metrica secundaria para salvar resultado. Perdas contam contra.

## PORTOES

  D0  SANIDADE: a base reproduz 12/80 com o mesmo teto
  D1  R1 - base >= +2 tarefas NOVAS
  D2  R1 - R2 >= +2   e   R1 - R3 >= +2   (o ganho vem do DIRECIONAMENTO, nao
      de possuir um gerador combinatorio melhor)
  D3  NAO QUEBRA: R1 perde 0
  D4  O resultado interessante e R1 > R2 ~ R3. Se R2 ~ R1, o que paga e ter
      mais candidatos e nao o feedback do erro.

## PREVISAO HONESTA, antes de rodar: 25% de D1+D2 passarem

Razao da duvida: o espaco de codecs do nucleo JA cobre boa parte de "funcao de
observavel de celula para cor"; a busca dirigida pelo residuo pode so
redescobrir o que ja existe. Razao da esperanca: composicao de profundidade 2
sobre coordenadas (x-y mod 3, x+y mod 2) nao esta no vocabulario atual, e padrao
periodico/diagonal e comum no ARC.

## SE DER NULO — conclusao pre-registrada, do usuario

Parar de procurar "mais um mecanismo de abducao". Com 7 nulos, a evidencia passa
a apontar para algo mais basico: a LINGUAGEM DE COMPOSICAO ou a REPRESENTACAO DO
RESIDUO, nao o mecanismo de selecao nem o de geracao. E o proximo experimento
muda de "achar o mecanismo" para "medir a expressividade da linguagem".
