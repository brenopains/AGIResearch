# Quarto ensaio: prefixo escolhido, sufixo resolvido

Este experimento foi concebido **após** observar a contagem de estados informativos no banco do ciclo. Resultados nele são exploratórios. Uma confirmação exigirá novo banco/seed com código e limiares congelados.

Os 16 programas são quatro ações por 32 passos seguidas de quatro ações por 96 passos. Para cada prefixo, resolver o sufixo finito por `V(s,p)=max_a R(s,p,a)`. A quantidade é o maior retorno realmente obtido entre os quatro sufixos a partir do mesmo estado; todo o custo de executar os 16 programas foi contabilizado. Não há crédito de solução para estados com valores iguais.

Aprender `V(s,p)` a partir de pixels sem usar retornos da avaliação. Comparar estado atual de dois quadros com a inclusão de uma leitura histórica executável, escolhida por resíduos de construção e validação. Reportar erro da função valor e arrependimento de escolher o prefixo no conjunto de avaliação. O arrependimento usa o melhor sufixo observado e é, portanto, diagnóstico de um decisor **com oráculo de sufixo**, não desempenho jogando sem rollback. Placebo: históricos embaralhados dentro de cada partição, seleção repetida, mesmo banco e custo.

Portão exploratório: a memória reduz o arrependimento em ambos os jogos e supera 31 de 32 placebos em cada um. Qualquer outro resultado indica que o sinal de valor gerado por resolver o sufixo ainda não virou decisão confiável. Não chamar isso invenção de AGI ou algoritmo científico inédito.
