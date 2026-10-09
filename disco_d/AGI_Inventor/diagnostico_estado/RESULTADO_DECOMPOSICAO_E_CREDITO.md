# O erro é localizável; o remédio ainda não foi demonstrado

A continuação anterior foi progresso: implementou transporte de funções e
produziu evidência auditada que alterou a próxima ação. Nesta continuação,
as fontes, bancos, pesos e relatórios atuais foram inspecionados novamente.
O objetivo continua sendo uma mente de nível humano e investigação geral;
nenhum resultado de jogo substitui esse objetivo. O projeto original foi
apenas lido. Todo código novo e evidência ficaram noD.

## Decomposição do erro

Foi implementada uma identidade telescópica para todos os corpos congelados:
erro de retorno = erro de recompensa + erro de transporte + erro de raiz
inicialmente terminada. A parcela de transporte é ponderada pela função do
sufixo restante. Isso mede o efeito de um erro na pergunta futura relevante,
em vez de usar apenas erro médio do vetor. É uma identidade do modelo sobre
os registros, não um novo teorema científico ou prova de suficiência perceptual.

Todas as32raízes, dois ramos, horizontes4/8/16 e cinco corpos foram reexecutados
nos dois jogos. Sem ajuste ou coleta; dados103 já vistos, diagnóstico de
desenvolvimento. Cinco testes passaram. Auditor independente atribuiu erro
pela diferença entre valores adjacentes dos sufixos, em vez da fórmula vetorial
usada pelo diagnosticador:960comparações e15.575transições, erro máximo2,12e-14.
O portão anterior permaneceu intacto. Principal1,32segundos de execução.

Arrependimento do primário em16blocos, menor é melhor:

| Consulta | Breakout | Pong |
|---|---:|---:|
| Corpo executável atual |0,18750|0,15625|
| Recompensa aprendida em todas as histórias reais futuras |0,40625|0,46875|
| Retirada ideal do erro de recompensa |0|0|
| História inicial fixa com términos reais |0,15625|0,59375|
| Previsãozero com desempate uniforme |0,34375|0,40625|

As linhas com histórias futuras, términos reais ou retirada de erro usam
oráculos. Não são políticas aprendidas. Retirar erro de recompensa usa respostas
verdadeiras na decomposição: o arrependimentozero não é um acerto do agente.

MSE do contraste de retorno:0,878043Breakout,1,846340Pong. Parcelas:

| Termo | Breakout | Pong |
|---|---:|---:|
| Quadrado do erro de recompensa |0,984027|2,497732|
| Quadrado do erro de transporte |0,137655|0,226272|
| Termo cruzado recompensa/transporte |−0,243639|−0,877663|
| Termos de raiz inicialmente terminada |0|0|

Há cancelamento de erro. Não converter esses valores em percentuais independentes
de culpa. Mesmo com histórias reais, a leitura atual de recompensa não recupera
decisão; melhorar apenas previsão de estados não é solução demonstrada para
esse corpo. Isso não identifica uma causa única deAGI.

Código e reprodução em`DECOMPOSICAO_TRANSPORTE.md`. Evidências:
`resultados/decomposicao_transporte.json`, eventos e auditoria homônimos.

## Metade fixa, metade resolvida

Depois foi testada uma intervenção que mantém as33funções sensoriais e os
quatro transportes exatamente iguais. A concatenação permite construir um
vetor de créditoX_P para cada programaP, tal que o retorno sejaX_P*w. Resolver
os132pesoswé um subproblema linear. A receita vale para este corpo linear
nos pesos, não para qualquer composição discreta ouARC. Nenhuma medida de
promessa foi usada.

Treino em512pares prefixais do banco97 por jogo, espelhados. Eles provêm das
mesmas32raízes e não são512experiências independentes. Todos os controles
ajustados receberam os mesmos alvos derivados; sem coleta. Código, regularização
0,05, controles e portão congelados antes do ajuste. Novos corpos congelados
antes da leitura103 nesta execução, porém103 já era conhecido: desenvolvimento.

Arrependimento em16blocos:

| Corpo | Breakout | Pong |
|---|---:|---:|
| Pesos novos por sequência |0,21875|0,37500|
| Pesos locais originais |0,18750|0,15625|
| Novos pesos com alvos embaralhados |0,21875|0,43750|
| Novos pesos sem contexto |0,15625|0,28125|
| Novos pesos sem transporte |0,06250|0,50000|
| Novos pesos com histórias embaralhadas |0,18750|0,28125|

O primário piorou frente ao corpo anterior nos dois jogos. SeuMSEde contraste
foi0,882421/2,098752. O portão conjunto falhou nos dois; nenhum corpo promovido.
O controle sem transporte bom emBreakout não substitui o primário depois do
resultado e não é evidência de mente humana.

Além disso, a nova cabeça piorou a leitura local de recompensa: de0,074515
para0,183621 emBreakout e de0,095366 para0,369081 emPong. Por isso os corpos
foram descritos como pesos efetivos de retorno, não como recompensas causais
corretas. Ajustar retorno pode compensar erro de dinâmica em vez de compreendê-la.

No treino, o resíduo numérico mínimo sem regularização,rcond1e-10, foi0,075752
Breakout e0,024362Pong, com posto132. A solução regularizada ficou em0,241443/
0,131967. A família linear de pesos com os operadores fixos não reproduz todas
essas respostas no cálculo feito. Não é prova de impossibilidade matemática
nem distingue perfeitamente representação, transporte, cobertura e regularização.
Histórias embaralhadas também permitem baixo resíduo de treino; as palavras
longas não foram plenamente cruzadas entre contextos neste banco. Isso limita
a interpretação do ajuste como aprendizado da situação.

Três testes passaram. Auditor independente refez o crédito por produtos de
matrizes, validou equações normais, controles, alvos e1.354eventos. Erro máximo
de crédito1,60e-14; resíduo máximo das equações5,28e-13; execução matricial versus
corpo3,55e-14. Principal2,48segundos, dez ajustes, zero chamadasALE. Nenhuma
regularização/escala/critério foi alterado para salvar o resultado.

Código/comandos em`CREDITO_SEQUENCIAS_TRANSPORTADAS.md`; resultados, sistemas
lineares, corpos e auditoria em`resultados/credito_sequencias*`.

## Próxima hipótese, ainda sem resultado

O próximo gerador deve trabalhar com pares de consequências confundidas:
extrair do treino a função futura e a ação que produziram discrepância,
propor uma função nova da história que explique essa discrepância, e testar
se ela conserva a distinção em outro episódio. Seu corpo completo deve ser
armazenado, composto e removido numa ablação. Não confundir uma combinação
linear das funções existentes com expansão do espaço representacional.

Primeiro repetir cruzamentos de perguntas entre contextos: baixo erro de
treino com palavras distintas por raiz pode existir sem representação útil.
Seleção/extrator devem usar apenas dados de desenvolvimento; evidência final
precisa de mundos/configurações ainda não usados para escolha. Uma função
aceita deve ajudar mais de uma pergunta, incluindo questões não usadas na
sua geração. O objetivo amplo requer transferência de capacidades e descoberta
autônoma verificável em vários domínios. Isso permanece não demonstrado.
