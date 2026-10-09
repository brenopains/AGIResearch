# Comparação aprendida entre programas de ações

Pré-registro anterior ao ajuste e à nova coleta. A meta humana permanece
aberta. Semente103 é desenvolvimento já visto; as palavras de ações serão
novas, não uma confirmação de mundo novo. Nenhum artefato anterior muda.

Os ensaios anteriores já aprenderam preferências sobre ações fixas e sondas
de decisão. Este teste não será chamado de primeira invenção desse conceito.
Intervenção: aprender `D(estado, programaA, programaB)` diretamente a partir
de consequências pareadas. O corpo impõe D(A,A)=0, D(A,B)=−D(B,A) e
D(A,B)+D(B,C)=D(A,C); não garante que o sinal esteja correto. Não compor
estados médios a cada passo para calcular a diferença.

Treino:32raízes97, dois programas de16blocos já realmente executados nos
bancos de transições mistas. Seus16prefixos fornecem512pares/1.024retornos
observados, correlacionados por raiz. Não contar prefixos como experiências
independentes. O condicionante comum é o estado64 da memória residual97
congelada, que recebe assinatura inicial medida por sondas e história passada.
É um diagnóstico oracular de comparação, não agente perceptivo pronto.

Programas são palavras sobre quatro ações anônimas, com símbolo de término
para preencher até16posições. Kernel de estado × kernel de palavra. Em cada
par, a característica implícita é diferença entre características de A e B;
resolver regressão regularizada nas diferenças reais de retorno. Registrar
o corpo completo, suportes, escalas e coeficientes. Esse estimador é uma
escolha estatística humana conhecida; não alegar nova teoria ou representação
humana inventada automaticamente.

Cinco braços, mesmas1.024observações e matriz de ajuste1.024×1.024:

- contraste condicional, primário;
- contraste com alvos embaralhados entre pares de mesmo comprimento e mesma
  condição A≠B; pares A=A continuam com diferença zero;
- contraste com estados de treino embaralhados entre raízes, mantendo todos
  os prefixos de cada estado juntos;
- contraste independente do estado, usando somente palavras;
- regressão de retornos absolutos, mesmo kernel/estado/palavras.

Regularização0,05 e escalas de distância aprendidas somente97. Todos usam
o mesmo orçamento de observações, dimensão e tamanho de sistema linear;
tempo/CPU de cada ajuste será registrado, sem fingir operações iguais.
Também comparar com os seis modelos de memória já congelados, nos programas
novos, e com escolher sempre o índice de programa cujo retorno médio no treino
é maior. Essas referências não recebem dados de treino ou avaliação novos.

Salvar todos os corpos/hashes antes de coletar novos resultados. Avaliação:
mesmas32raízes103, dois programas uniformes de16blocos gerados com RNG9811
antes de executar. Certificar64raízes da caminhada,128ramoscurtos,8replays
de um programa inteiro; limite24.000passos ALE/jogo, até dois processos.
Retornos por bloco reais, sem reset depois do término. Verificar que nenhum
programa completo coincide com o treino; se houver, não alterar a semente:
reportar alcance reduzido/inconclusivo para generalização de palavras.

Primário128passos: ganho médio de arrependimento e limite inferior95% acima
de zero contra cada braço e referência; pelo menos dois episódios positivos
por comparação; nenhuma piora por episódio contra escolha constante. Exigir
também menor MSE da diferença de retornos que regressão absoluta e alvos
embaralhados. Bootstrap8episódios/10.000amostras/RNG9817. Sem contraste real,
inconclusivo.32/64passos são secundários, sem mudar portão depois.

Se falhar, não promover nem aumentar coleta para salvar o resultado. Separar
erros de diferença aprendida, informações marginais das palavras e composição
dos modelos antigos. Mesmo positivo, exige confirmação nova, percepção sem
sondas oraculares e transferência entre mundos antes de representar avanço
de capacidade geral. Não é um salto garantido ou suficiente para AGI.
