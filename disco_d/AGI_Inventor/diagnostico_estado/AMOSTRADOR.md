# Sexto ensaio: selecionar estados informativos antes de sondar

Exploratório no banco existente. A pergunta é se o estado visual atual contém sinal para prever quais situações terão um prefixo de ação **unicamente melhor** depois de resolver quatro sufixos. O aprendiz vê pixels do estado, não o valor verdadeiro nas raízes reservadas.

Treinar regressão kernel fixa nos episódios 0–3 com rótulo binário `melhor valor > segundo melhor valor`. Episódios 4–5 mostram somente a estabilidade, sem trocar arquitetura/hiperparâmetros; treinar novamente em 0–5 e escolher as quatro raízes mais promissoras dos 16 estados dos episódios 6–7. Comparar o número de situações com ótimo único com 1.000 subconjuntos aleatórios de quatro raízes do **mesmo** conjunto final. Nenhuma resposta do teste orienta seleção.

Portão: pelo menos três das quatro raízes escolhidas são informativas em cada jogo, e menos de 5% dos subconjuntos aleatórios igualam/superam o resultado, em ambos. Passar indicaria utilidade para aquisição de dados neste banco, não um agente que já alcança essas raízes online. Falha aponta que o seletor/observação não antecipa a raridade ou que a amostra é pequena; não prova impossibilidade.
