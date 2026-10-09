# PRE-REGISTRO — excecao ganha de mecanismo?

17/09/2026, antes de rodar.

## A AFIRMACAO A TESTAR

Do oitavo nulo: 183 de 506 pares EXPLICAM celula que nenhuma metade explica, e
mesmo assim nenhum entra em pilha vencedora. Leitura proposta: **as celulas do
residuo sao mais baratas de LISTAR como excecao do que de EXPLICAR com operacao
nova.** Se for verdade, o gargalo e o CRITERIO, nao o gerador.

## O PRECO ATUAL, e ele NAO e arbitrario

    bits_excecao(n,h,w) = n * (log2(h*w) + log2(10))     # posicao + cor
    grade 20x20 -> ~12 bits por celula de excecao

## O TESTE, causal e de um parametro

Multiplicar `bits_excecao` por f em {0,5 · 1 · 2 · 4 · 8}, TETO IGUAL, e medir:
  (a) tarefas resolvidas
  (b) se codecs novos (pares promovidos) passam a ENTRAR em pilha vencedora

## PORTOES

  X0  SANIDADE: f=1 reproduz 12/80
  X1  A AFIRMACAO: com f>=2, pares promovidos passam a entrar em pilha vencedora
      (hoje: 0). Se entrarem, excecao ESTAVA ganhando de mecanismo por preco.
  X2  E PAGA: com algum f, resolvidas > 12. Se so entrar sem resolver mais, o
      mecanismo vence em bits e erra o teste — decoracao com outra roupa.
  X3  DIRECAO: com f=0,5 (excecao mais barata) resolvidas devem CAIR ou empatar.
      Se subirem, minha leitura esta invertida.

## PREVISAO HONESTA, antes de rodar

X1: 40%. X2: 15%. Razao da duvida: ja medi que so 3 de 61 falhas tem residuo
zero (decoracao de fato), e as outras 58 falham por NAO DESCREVER. Encarecer
excecao nao cria descricao que nao existe — deve quebrar as tarefas que
legitimamente precisam de poucas excecoes e derrubar o numero.

**Se X1 passar e X2 falhar:** o preco estava errado E o mecanismo continua nao
generalizando. Isso e informativo e e meio caminho.
**Se X1 falhar:** minha leitura do oitavo nulo esta errada, excecao NAO ganha de
mecanismo, e os 183 pares perdem por outro motivo que eu ainda nao sei.
