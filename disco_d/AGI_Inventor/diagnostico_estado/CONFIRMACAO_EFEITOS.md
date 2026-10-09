# Confirmação de efeitos locais condicionados ao histórico

O banco de desenvolvimento é a semente ALE 97, 64 checkpoints por jogo,
quatro ramos de oito ações. O ensaio leave-one-episode-out mostrou que a
regressão kernel dos dois últimos quadros 21×16 prevê contraste visual entre
ações melhor que persistência e um molde de efeito fixo por ação. A proposta
de trilhas explícitas não superou os pixels e não é a hipótese primária.

**Congelado antes de coletar a semente 103:** ajustar um único regressor por
jogo nos 64 checkpoints da semente 97, com entrada `r0_21x16_2` e saída
quatro diferenças entre imagem após oito ações e imagem atual, ambas 21×16.
Não usar recompensa, RAM, nomes de objetos, rótulos de jogo ou resultados da
semente 103 para ajustar entradas, parâmetros ou escolher variantes. Avaliar
64 checkpoints da semente 103 com o mesmo relógio de oito episódios × oito
checkpoints e quatro ramos pareados. A coleta tem teto de 20 mil passos ALE
por jogo, replay de um ramo e JSONL de eventos.

Métrica primária: para os seis pares de ações, erro quadrático do contraste
previsto, com peso por pixel `0.05 + maior contraste absoluto real entre os
pares` no checkpoint. Dividir pelo erro da previsão sem contraste. Reportar
por episódio e média dos oito episódios. O avaliador pode usar o desfecho
verdadeiro somente para calcular a métrica, nunca para produzir previsões.

Controles: (1) persistência/ação cega, razão 1; (2) molde médio de efeito
por ação aprendido apenas no banco 97; (3) 32 modelos treinados após permutar
os quatro rótulos de ação independentemente em cada checkpoint 97; (4) 32
permutações de históricos **dentro de cada episódio 103** com o modelo
original intacto.

Portão positivo *local*: ambos os jogos devem ter contraste não nulo em todos
os episódios; a razão do modelo deve ser menor que 1 em todos os oito
episódios por jogo; superar o molde por pelo menos 0,10 na média e em pelo
menos seis episódios de cada jogo; no máximo 1/32 controles de cada tipo e
jogo pode igualar ou vencer o modelo. Isso provaria apenas previsão
condicionada de efeitos visuais de oito passos em dois jogos, não modelo de
mundo geral, planejamento longo, nem mente artificial de nível humano.

Hashes SHA-256 congelados antes de gerar a semente 103:
`confirmar_efeitos_locais.py` =
`c25ebb00fcabbb4c2adc56c44108e9ad6711d8b2b30a44d209346a8015061333`;
`coletar_efeitos_locais.py` =
`1afbc809e3e5b14ae1d823db236cb6e06eb1171ab95ca9f3c8446c371dda0392`;
`avaliar_efeitos_locais.py` =
`a0644063a7f605095e6fffeaa20e6ce9336e0d773d619607cd435d05a14c8e87`.
