# PRE-REGISTRO — composicao mais PROFUNDA paga?

17/09/2026, antes de rodar.

## DE ONDE VEM

A checagem de hoje mostrou: o nucleo distingue operacao (codec) de termo
(marcador), TEM composicao sequencial (`ENTAO`), ponto fixo (`ATE`) e iteracao
contada (`VEZES:k`), e a busca ALCANCA a composicao — 5 das 12 pilhas vencedoras
usam `ENTAO` (42%). Logo "a linguagem e fechada" estava errado.

O unico fechamento REAL que sobrou e um numero: `montar()` chama
`_montar_base(..., estagios=2, ...)` — cravado na linha 2702. A composicao esta
capada em DOIS estagios: da para A-entao-B, nao A-entao-B-entao-C.

## O TESTE

Trocar 2 por 4, TETO IGUAL (mesmas avaliacoes por tarefa). Nao e mais orcamento
— e a MESMA verba gasta com composicao mais profunda permitida.

## PORTOES

  E0  SANIDADE: estagios=2 reproduz 12/80
  E1  PAGA: estagios=4 resolve >= +2 tarefas NOVAS
  E2  NAO QUEBRA: perde 0
  E3  O GANHO E DE PROFUNDIDADE: as tarefas ganhas usam >= 3 estagios (senao o
      ganho e ruido de busca, nao composicao profunda)

## PREVISAO HONESTA, antes de rodar: E1 FALHA, ~65% de confianca

Razao: `ENTAO` custa 2 bits e nao explica NENHUMA celula no instante em que
entra — ganho atrasado, que o feixe ordenado por bits mal enxerga (ja precisou
de "vaga reservada" para valer +3). Permitir MAIS estagios multiplica essa
dificuldade e gasta verba em ramos que o criterio miope vai podar. Espero
11 a 13 de 80, ou seja empate ou leve perda.

Se E1 passar com E3, e o ganho mais barato disponivel e eu estava errado sobre
onde a profundidade morre. Se E1 falhar, fecha a ultima duvida estrutural sobre
a linguagem e sobra so o desenho de sala de espera para OPERACOES.
