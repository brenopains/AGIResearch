# Gerador executável, resultado e limites

Objetivo de mente humana permanece não atingido. Estes dados não permitem
estimar proximidade, nem identificar uma única peça suficiente para AGI.
O projeto original em C: permaneceu intacto; trabalho isolado em D:.

## O que foi implementado

`funcoes_contraexemplos.py` constrói funções completas a partir de pares de
histórias observadas em episódios distintos, para a mesma ação anônima.
Prioriza discrepâncias de recompensa que o leitor anterior não explica.
Cada função é uma comparação afim com tanh, cujos coeficientes vêm do par;
também propõe produtos de duas comparações, antes de exigir aprovação das
partes. Guarda parâmetros, árvore executável e origem. O intérprete recusa
operações fora da linguagem; não executa Python aprendido arbitrário.

A família numérica e o orçamento foram escolhidos por nós: 24 comparações,
8 produtos e até 4 coordenadas novas. Isto é geração limitada de funções,
não invenção irrestrita ou mecanismo inédito mundialmente demonstrado.
Há precedentes para adicionar unidades a partir de erros, como
[Cascade-Correlation](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf).

O piloto construiu suportes e operadores somente com episódios 0–3 de 97.
Escolha em 4–5; episódios 6–7 e banco 103 separados desta escolha. Estes bancos
já eram conhecidos de experimentos anteriores: desenvolvimento, não teste
independente. Corpos foram congelados antes de sua avaliação final. Nenhuma
função foi escolhida, nenhum corpo foi promovido, nenhum critério mudou.

## Resultado do piloto

Breakout: os oito pares de validação contêm quatro diferenças reais de
retorno. A base erra dois casos. Todos os 32 candidatos de cada gerador
passam pela exigência de expansão numérica no treino. O primário reduz erro
quadrático em 15 candidatos, melhora decisão em 9 e faz ambos em 5, dos quais
2 são produtos. Nenhum melhora dois casos distintos: o protocolo exigia
essa repetição, além das duas métricas, e rejeitou todos corretamente.

Direções aleatórias também apresentam 5 candidatos que melhoram ambas as
métricas em um caso. Rótulos embaralhados apresentam 2, pares aleatórios 1.
Estes contadores não são testes estatísticos de superioridade do gerador;
candidatos compartilham dados e não são amostras independentes.

Pong: todos os oito pares de validação têm diferença real de retorno zero.
Arrependimento zero resulta da ausência de decisão preferível, não de acerto
inteligente. O seletor declara `inconclusive_no_validation_decision_error`.
Por isso, ausência de função escolhida aqui não testa adequadamente sua
capacidade de gerar distinções úteis.

O portão conjunto em 103/16 blocos permanece `failed_development_gate`.
Sem funções escolhidas, todos os braços de geração fazem exatamente as
mesmas previsões da base, inclusive as ablações que removem essas funções.
As métricas da base não são ganho atribuível ao gerador:

| Banco 103, 16 blocos | Arrependimento médio | Erro quadrático |
|---|---:|---:|
| Breakout, todos os geradores / sem funções novas | 0,15625 | 0,793974 |
| Pong, todos os geradores / sem funções novas | 0,25 | 1,810446 |

Piloto: 6,20 segundos de cálculo principal, 136 ajustes nos seletores,
2 ajustes de base, 2 ajustes adicionais sem funções, 128 soluções de cabeças
para previsões cruzadas usadas na geração. Zero chamadas ALE novas.

## Diagnóstico que distingue rejeição de ausência de efeito

`diagnosticar_pool_funcoes.py` congelou um diagnóstico separado, com os mesmos
32 candidatos e mesmos dados de construção/validação. Não usou 103, não
alterou o portão e não salvou candidatos como soluções promovidas.
Foram 256 ajustes em 3,16 segundos de cálculo principal.

Em Breakout, para **todos** os quatro geradores, algum candidato corrige a
linha 21 do banco (episódio 5), mas nenhum corrige a linha 18 (episódio 4).
Mesmo um oráculo que escolhesse o candidato por resposta verdadeira deixa
arrependimento 0,125 contra 0,25 da base. Esse oráculo não é decisão executável
e não torna o resultado positivo. A conclusão cobre candidatos individuais
do pool; não prova que combinações arbitrárias também falhariam.

O diagnóstico separa erro local de recompensa, transporte das coordenadas
antigas e transporte da coordenada nova, com decomposição telescópica exata
do retorno e termos cruzados. Algumas funções melhoram a leitura local:

| Validação local | Erro de recompensa da base | Melhor candidato primário | Melhor candidato com rótulos embaralhados |
|---|---:|---:|---:|
| Breakout | 0,068741 | 0,066522 | 0,064042 |
| Pong | 0,105851 | 0,089617 | 0,092894 |

São mínimos retrospectivos, não funções selecionadas ou ganhos confirmados.
Em Breakout, a melhor leitura local do controle embaralhado é melhor que a
do primário. Em Pong, a leitura melhora em parte do pool, mas falta contraste
de decisão nessa validação. Não inferir inteligência geral dessas reduções.

Fornecer todas as histórias futuras reais, ou retirar erro de recompensa
com respostas verdadeiras, permanece intervenção de oráculo. Termos cruzados
negativos mostram cancelamento de erros; não atribuir percentuais causais
independentes às parcelas. As origens do primário incluem recompensa não
nula em três episódios de construção em Breakout e quatro em Pong; não é
correto atribuir a falha simplesmente a uma única âncora ou episódio.

## Verificações e reprodução

Cinco testes do gerador e quatro testes de cobertura passaram. A auditoria
do piloto recontou 128 candidatos e 1.920 comparações finais, com diferença
máxima de funções 1,20e-13 e de execução 1,02e-14. A auditoria do diagnóstico
reconstruiu 256 corpos por Cholesky e execução para a frente, recontando
2.048 previsões pareadas; discrepância máxima 8,03e-13.

O auditor original importa NumPy antes de configurar BLAS e falhou na
reprodução estrita da geração. Sua fonte congelada foi preservada. O novo
ponto de entrada configura as mesmas duas threads antes da importação e
executa **o mesmo auditor**, sem modificar critérios ou tolerâncias. Ele
passou. O protocolo congelado descreve a execução do auditor como matrizes
aumentadas; a fonte efetiva usa propagação de estado para a frente. A
comparação independente é com a composição de funções para trás do piloto.

Comandos na pasta D:\AGI_Inventor\diagnostico_estado:

```powershell
py -3.11 -B testes_funcoes_contraexemplos.py
py -3.11 -B testes_pool_funcoes.py
```

Auditorias já produziram registros exclusivos e não sobrescrevem resultados:
`resultados/funcoes_contraexemplos_auditoria.json` e
`resultados/pool_funcoes_auditoria.json`. Fontes, parâmetros, corpos, dados
e eventos estão identificados por hashes nos manifestos. O ponto de entrada
correto do primeiro auditor é `executar_auditoria_funcoes_contraexemplos.py`;
o segundo é `auditar_pool_funcoes.py`. Rodar novamente após seus JSONs já
existirem termina com proteção de preservação, não cria nova confirmação.

## Próxima hipótese, ainda não implementada ou demonstrada

O gerador atual propõe distinções por recompensa imediata. A hipótese a
atacar é construir distinções por **respostas diferentes à mesma sequência
de ações**, aprendendo também como cada distinção evolui após agir. Dois
contextos podem dar a mesma recompensa agora e exigir decisões diferentes
depois; o gerador atual não usa esse contraste prospectivo na proposta.

O próximo teste precisa cruzar as mesmas perguntas entre situações e ter
contrastes de consequência em episódios separados. Essa cobertura deve ser
checável antes de uma avaliação de capacidade. Não afrouxar a regra atual
para aprovar melhora em um único caso, nem aumentar candidatos para obter
um vencedor nesses mesmos dados.

Distinções prospectivas, corpos reutilizáveis entre mundos, variáveis novas
a partir de experiência crua e investigação autônoma continuam hipóteses a
testar. Resolver jogos pequenos seria evidência parcial dessas capacidades,
não substituto para descobertas científicas ou uma mente de nível humano.
