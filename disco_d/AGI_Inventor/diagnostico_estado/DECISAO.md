# Quinto ensaio: variável escolhida pelo erro de decisão

Especificado após o resultado nulo da seleção por MSE de valor. Separado e exploratório no banco existente; confirmação requer novos episódios/seed.

O valor de cada prefixo continua sendo o melhor retorno entre quatro sufixos realmente executados. Para inventar uma variável de memória, usar os resíduos fora do próprio episódio de construção apenas para propor candidatos. A escolha final na validação minimiza **arrependimento normalizado de escolher o prefixo**, com MSE como desempate. Assim uma variável que prevê pixels ou valores médios, mas não muda a ação correta, não recebe crédito. A variável é uma expressão executável de quadro passado, sem nome de objeto.

Mesmas partições e nulos: embaralhar histórias dentro de cada partição, repetir toda a seleção, manter o mesmo banco de ações e resultados. Portão: ganho de arrependimento positivo em ambos os jogos, no máximo um de 32 nulos iguala cada ganho, e validação também melhora. Caso contrário, o mecanismo não está demonstrado.
