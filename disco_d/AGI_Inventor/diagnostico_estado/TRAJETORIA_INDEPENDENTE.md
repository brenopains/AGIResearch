# Confirmação em trajetórias longas independentes

O ensaio com macros curtas da semente 73 teve poucas oportunidades para a
ação constante errar. Este protocolo é congelado antes de executar a nova
semente 97. Não alterar o regressor treinado somente na semente 59 nem usar
qualquer rótulo da semente 73 para treino/seleção.

Em cada jogo (Breakout e Pong), oito episódios independentes. Após resetar o
ALE, executar ações uniformemente aleatórias, observar oito checkpoints a
cada 128 ações. Se o jogo terminar, resetar e prosseguir com o relógio do
episódio; contar todos os resets. Os checkpoints são determinados pelo tempo,
sem consultar o modelo nem o retorno. Capturar 16 imagens anteriores. Em cada
checkpoint clonar o estado e verificar os 16 prefixos/sufixos de ações de
32+96 passos, assim como no ensaio anterior. Testar um replay de snapshot.
Teto de 170 mil passos ALE por jogo, incluindo trajetória e verificações.

Reportar arrependimento normalizado por raiz e episódio, número de raízes
onde a ação constante não é ótima, diversidade de ações, custo e 32 controles
que permutam históricos dentro do mesmo episódio. A margem de decisão é
adequada somente se houver ≥12 raízes em que a constante falha, distribuídas
por ≥4 episódios. Condicionado a isso, o novo modelo passa este teste local
se reduzir o arrependimento médio contra a constante em ≥0,10, vencê-la em
≥5/8 episódios e no máximo 1/32 permutações tiver arrependimento tão baixo.
Se não houver margem, reportar insuficiência, sem declarar sucesso ou falha de
representação. Este teste ainda não prova variáveis causais gerais, nem AGI.

As trajetórias, estados e retornos são salvos para auditoria, mas os rótulos
da semente 97 ficam reservados e não serão usados para ajustar o modelo que
é avaliado nessa semente.
