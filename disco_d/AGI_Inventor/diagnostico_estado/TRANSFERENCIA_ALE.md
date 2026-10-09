# Transferência visual para estados ALE inéditos

O banco reconstruído da semente 59 já foi inspecionado. Congelar agora:
regressão kernel com entrada dos dois últimos quadros 21×16, treinada apenas
nos episódios 0–5 desse banco; ação constante escolhida nos mesmos episódios;
modelo antigo treinado no banco da semente 29. Nenhum recebe valores da
semente de avaliação.

Nova semente ALE 73. Em Breakout e Pong, oito episódios, cada qual começa com
32 ações aleatórias comuns; gerar oito programas de quatro segmentos de 32
ações, distintos e independentes dos modelos. Em cada estado não terminal,
clonar, executar todos os 16 prefixos/sufixos (32+96 ações) e guardar os
quatro melhores retornos por prefixo. Verificar replay. Teto 170 mil passos
ALE por jogo. Sem selecionar estados por predição ou retorno.

Métrica: arrependimento normalizado de ação em raízes com contraste, reportado
por episódio. Em Breakout, evidência local exige que o novo modelo vença a
constante e o antigo em ≥6/8 episódios informativos, reduza arrependimento
médio por ≥0,10 contra ambos e supere 31/32 controles que permutam históricos
**dentro do episódio de teste**. Em Pong, uma perda >0,05 em relação à
constante já refuta transferência sem dano. Se o modelo antigo/constante
empatam no teto, reportar falta de margem. O portão só fala sobre aprender
decisão visual dentro destes jogos; não prova representação geral.

Toda execução, terminal e custo entra no JSONL. O banco de avaliação fica
separado e nunca atualiza o modelo congelado.
