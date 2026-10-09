# Compilador de contraexemplos de decisão

Este ensaio é exploratório nos bancos existentes, após ver nulos anteriores. O teste de confirmação, se houver sinal, precisa de sementes/mundos novos. O mecanismo é independente de nomes de jogos: recebe histórias de imagens, valores pareados por ação, grupos de episódio e orçamento.

1. Construir previsões de valor de quatro prefixos deixando de fora o episódio da amostra. Um contraexemplo existe somente se o prefixo escolhido por essa previsão perde valor para outro prefixo realmente executado.
2. Formar o vetor de resíduos de valor **apenas** dos episódios de construção. Uma direção de erro por decomposição SVD define o alvo de síntese; não usar pixels futuros como alvo.
3. Criar operadores candidatos diretamente dos históricos: quadro a 2/4/8/15 passos e diferenças entre pares desses tempos. Para cada operador, correlacionar cada pixel com o resíduo ponderado por erro de decisão; reter os 4/16/64 pesos de maior magnitude. Cada operador vira uma função numérica completamente especificada (`lag`, máscara e pesos), sem rótulo semântico.
4. Treinar o mesmo preditor de valores com `r0` ou `r0+z`. Escolher em episódios 4–5 pelo menor arrependimento de ação, com MSE apenas para desempate. Avaliar sem ajustes nos episódios 6–7.

Controles: a mesma síntese com **apenas o quadro atual**; e 32 históricos embaralhados independentemente dentro de cada partição. Todas as seleções são refeitas em cada nulo. Portão: pelo menos quatro contraexemplos independentes de construção e quatro estados informativos de avaliação; ganho de arrependimento positivo; no máximo um nulo iguala/supera; operador histórico supera operador atual; dois domínios distintos passam. Caso contrário, reportar o elo que falhou. Nenhum resultado aqui equivale a AGI humana.
