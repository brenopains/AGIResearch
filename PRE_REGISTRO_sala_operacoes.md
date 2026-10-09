# PRE-REGISTRO — sala de espera para OPERACOES

17/09/2026, antes de implementar. Desenho do usuario (3 camadas), aterrado na
checagem do nucleo feita hoje.

## O QUE A CHECAGEM DE HOJE ESTABELECEU

  operacao = codec (23, entidade propria: propor/bits/prever)
  termo    = marcador que o codec propoe
  ENTAO    = composicao sequencial; o proximo estagio le a grade NOVA
  e a busca ALCANCA composicao: 5 das 12 vencedoras usam ENTAO (42%)

Logo "a linguagem e fechada" era FALSO. O gargalo e QUAL operacao acrescentar —
meu placar de palpiteiro e 0/7.

## A IDEIA, e ela sai da checagem

`ENTAO` tem ganho ATRASADO porque as duas metades sao precificadas SEPARADAS: o
primeiro codec custa bits e nao explica celula ate o segundo chegar; o feixe
ordenado por bits o poda antes. **Empacotar (A ENTAO B) como operacao ATOMICA,
com custo unico, faz o ganho deixar de ser atrasado — o par paga junto.**
E o conserto da cegueira-do-ganho-imediato no nivel das OPERACOES (nos tres
niveis anteriores o conserto valeu +3).

## OS TRES OBSTACULOS, e como o desenho ataca cada um

  obst.1 ampliar dilui orcamento   -> candidata NAO entra em D.CODECS; nao gasta
                                      teto. Some por construcao.
  obst.2 ganho atrasado invisivel  -> (a) par precificado junto; (b) promocao
                                      julgada ENTRE TAREFAS, nao dentro de uma
  obst.3 nao paga agora            -> promocao nao exige pagar agora

## O MECANISMO

  1. REGISTRO  candidatas = pares ordenados de codecs existentes (23x22 = 506).
     Enumeravel — e enumeracao substitui gerador, que e 0/7 e 0/30.
  2. SALA      para cada tarefa com residuo, mede-se BARATO se o par reduziria o
     residuo mais que A sozinho e que B sozinho. Sem montar pilha, sem gastar teto.
  3. PROMOCAO  par com evidencia em >= K tarefas entra em D.CODECS. Revisao
     quando a evidencia DOBRA (o calendario do `conceito_util`).

## PORTOES

  W0  SANIDADE: base 12/80 com o mesmo teto
  W1  PAGA: com os promovidos, >= +2 tarefas NOVAS
  W2  PLACEBO PAREADO: promover pares SORTEADOS, mesma quantidade. Se o placebo
      empatar, o criterio de promocao nao vale nada e o ganho e so "mais codecs"
  W3  USO: os promovidos aparecem em pilha vencedora (senao o ganho vem de outro
      lugar — foi o que o `raio` me ensinou: entrou em 1, ganhou 0)
  W4  A SALA IMPORTA: promover por 1 tarefa x promover por K tarefas. Se K=1
      empatar com K alto, a sala de espera e decoracao e o que paga e o par.

## PREVISAO HONESTA, antes de rodar: 30% de W1+W2 passarem

Maior que os 7 nulos anteriores por dois motivos concretos: (a) e COMPOSICAO DE
OPERACOES EXISTENTES, nao vocabulario estrangeiro — a forma dos 2 sucessos
(h, rot(v^)); (b) o conserto de precificacao tem mecanismo, nao e palpite.
Menor que 50% porque meu placar e 0/7 e porque enumerar 506 pares pode
simplesmente reencontrar o que o ENTAO ja alcanca em 42% das vencedoras.

## SE W1 PASSAR E W4 FALHAR

O ganho e do EMPACOTAMENTO (precificar junto), nao da sala de espera. Isso e
resultado, e mais barato de implementar que a sala. Eu reporto assim.
