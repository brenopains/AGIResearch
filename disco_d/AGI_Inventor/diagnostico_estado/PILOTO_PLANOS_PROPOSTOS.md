# Planos propostos pelo modelo: piloto de desenvolvimento

Registro antes de gerar e executar planos. O treino permanece 97, com arestas
de oito passos observadas nos trajetos constantes. A semente 103 já foi vista
em outros testes: este piloto é desenvolvimento, não confirmação independente.
Os mesmos corpos e parâmetros serão usados em Breakout e Pong.

`planejador_distribuido.py` recebe apenas bancos numéricos de experiências.
O estado pode ter qualquer dimensão, e o conjunto qualquer número de ações.
Ele compõe palavras de índices de ação, mantendo distribuições conjuntas de
recompensa, sucessor e término. O contrato exato é testado num mundo finito
onde trocar de ação é necessário; busca larga deve coincidir com enumeração.
Palavras constantes devem coincidir com o cálculo anterior confirmado.

Busca proposta: feixe de oito prefixos, 16 blocos, oito passos por bloco.
Todos os índices de ação expandem cada prefixo retido; selecionar pelo retorno
previsto acumulado, com ordem lexicográfica para empates. Não ajustar largura
ou horizonte depois do resultado. A busca é aproximada e pode descartar
prefixos que exigiriam recompensa posterior. Não chamá-la de busca completa.

Gerar para todas as 64 raízes, usando somente dois quadros iniciais e treino97,
antes de executar qualquer plano. Cada raiz recebe cinco planos: distribuição,
colapso ao estado médio, sucessores de treino embaralhados, recompensas de treino
embaralhadas e bancos de ação trocados. Todos recebem a mesma busca. Acrescentar
como referências a ação constante escolhida no treino, a escolha entre quatro
constantes pelo corpo confirmado e regressão direta de preferências.

Registrar planos, valores previstos, sobrevivência prevista, entropia da
distribuição e alternativas do feixe como hipóteses. Registrar resultados
medidos em eventos distintos. O avaliador reconstrói cada história a partir
da mesma trajetória aleatória e verifica contra o banco existente. Certifica
os quatro ramos de oito passos contra o banco existente. Executa cada plano
novo a partir do estado clonado, com pelo menos um replay completo por episódio.
Não alterar um plano depois de ver seu retorno. Teto 65.000 passos ALE por jogo,
CPU de geração com registro de duração, no máximo dois processos em paralelo.

Medida: ganho pareado de retorno real contra cada referência, por oito episódios.
Só recomendar confirmação nova se houver ganho médio e limite inferior de
bootstrap95% acima de zero contra cada referência e controle, e melhora em
dois episódios ao menos contra cada um, sem piora por episódio contra ação
constante. Bootstrap 10.000, RNG9763. Todos os controles entram no portão.
Sem contraste/margem contra constante, inconclusivo. Reportar distribuição
de erros de valor e frequência de planos não constantes, mesmo em falha.

Em caso negativo, manter o resultado e investigar a previsão das ações
compostas; não retocar critério nem afirmar impossibilidade. Um passe ainda
seria um ganho de planejamento nesses jogos. Não demonstraria geração de
variáveis gerais, agência científica, transferência para regras inéditas
ou mente de nível humano.

```powershell
py -3.11 -B testes_planejador_distribuido.py
py -3.11 -B piloto_planos_propostos.py breakout --propor
py -3.11 -B piloto_planos_propostos.py breakout --executar
```

Repetir para `pong`. O arquivo de propostas contém hashes antes de observação
dos resultados. Scripts recusam sobrescrever propostas e pilotos concluídos.
Os corpos previamente confirmados ficam congelados. Artefatos:
`resultados/{jogo}_planos_propostos_103_propostas.json`, `.jsonl`,
`{jogo}_planos_propostos_103.json` e eventos `.jsonl`.
