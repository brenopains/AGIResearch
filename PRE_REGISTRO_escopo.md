# PRE-REGISTRO — operacao + ESCOPO

17/09/2026, antes de implementar. Nasce de RASTREIO COMPLETO de um caso, nao de
agregado — a primeira vez em dois dias.

## O QUE O RASTREIO MOSTROU (tarefa 0607ce86)

```
o par ESCREVE                  506 celulas
acerta                         386   (a maioria ja estava certa)
dessas, que estavam ERRADAS:     6   <- o conserto real
logo escreve ERRADO:           120
```
Conserta 6 e quebra 120. **A operacao nova nao e fraca, e INDISCRIMINADA.**
Falta ESCOPO: a condicao que diz "aplique aqui e nao ali".

## O QUE JA EXISTE E O QUE NAO

`CondicaoAprendida` (no nucleo) aprende a condicao para UMA operacao: pintar de
uma cor um conjunto de celulas de fundo. Ja deu zero duas vezes antes
(`celobj.py`, `condicao.py`) num regime mais duro (explicar a tarefa inteira);
esta versao so precisa explicar o residuo.

**O que NUNCA foi feito:** aprender escopo para uma operacao QUALQUER. E o que
este modulo faz: `ComEscopo(codec)` aprende, do treino, ONDE aquele codec deve
escrever — e so escreve la.

## PORTOES

  Y0  SANIDADE: base 12/80 com o mesmo teto
  Y1  O ESCOPO CONTEM: a versao com escopo escreve MUITO menos celulas e erra
      muito menos (medido no rastreio: 506 -> esperado dezenas)
  Y2  REDUZ ERRO onde a operacao sozinha NAO reduzia (o teste que falhou 3x hoje)
  Y3  PAGA: >= +2 tarefas NOVAS
  Y4  PLACEBO PAREADO: escopo SORTEADO com a mesma seletividade. Se empatar, o
      ganho e de escrever menos, nao de escrever no lugar certo.

## PREVISAO HONESTA: Y2 40%, Y3 20%

Maior que os oito nulos porque nasce de rastreio e tem mecanismo. Menor que 50%
porque a mesma ideia ja deu zero DUAS vezes noutro regime, e porque conter uma
operacao indiscriminada pode simplesmente deixa-la sem nada para fazer.

**Se Y1 e Y2 passarem e Y3 falhar:** o escopo conserta o mecanismo e ainda assim
nao resolve tarefa — porque consertar 6 de 23 celulas nao zera o erro, e o
criterio e exato. Isso seria o NONO nulo com diagnostico completo, e a conclusao
seria que o gargalo e a COBERTURA do conserto, nao a sua precisao.
