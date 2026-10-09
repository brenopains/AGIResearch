# Resultado: operador executável, benefício local, sem validação geral

Não há evidência para dizer que estamos próximos de uma mente humana.
Os ensaios aqui medem leitura visual e composição em dois jogos. Não medem
invenção autônoma de representações, pesquisa de fronteira, linguagem geral,
transferência de um corpo entre mundos ou eficiência comparável à humana.

Foi implementado um corpo que interpreta uma sequência pela execução das
ações, em vez de interpolar a sequência inteira como uma palavra:

`v_vazio = 0; v_aQ = recompensa_a + transporte_a * v_Q`.

Cada transporte foi ajustado para preservar funções da próxima observação,
com recompensa e término registrados. Isso evita transportar uma imagem
média e depois extrair funções dela. São32suportes extraídos das observações,
33funções numéricas e quatro matrizes por corpo. A famíliaRBF foi definida
por nós: não apresentar esses suportes como conceitos inventados ou inovação
mundial. O aprendiz não executaPython arbitrário.

Os dois jogos usaram o mesmo código/configuração, mas parâmetros aprendidos
separadamente. Treino:861transições válidasBreakout97 e1.024Pong97. Avaliação:
32pares de sequências de16blocos por jogo em103, dados de desenvolvimento
conhecidos. As transições e os corpos foram congelados por hashes; ambos os
ajustes terminaram antes da leitura dos alvos103 nesta execução. Não houve
coletaALE. Palavras inteiras não foram alvos de ajuste do novo mecanismo,
mas as trajetórias de treino pertencem à mesma bancada; não afirmar descoberta
de mundo nunca visto nem confirmação independente.

Arrependimento final: menor é melhor. Previsão empatada escolhe uniformemente
no cálculo principal, impedindo o braçozero de ganhar por desempate fixo.

| Braço | Breakout103 | Pong103 |
|---|---:|---:|
| Transporte aprendido |0,187500|0,156250|
| Alvos embaralhados dentro da ação |0,156250|0,562500|
| Contexto inicial médio |0,187500|0,250000|
| Sem transporte, contexto inicial fixo |0,187500|0,593750|
| Identificação de ações permutada |0,312500|0,250000|
| Previsãozero |0,343750|0,406250|

MSE das diferenças de retorno do primário:0,878043Breakout,1,846340Pong.
EmPong, ambos os critérios foram melhores que todos os controles. EmBreakout,
o controle embaralhado teve menor arrependimento; retirar contexto/transporte
empatou em decisão e retirar transporte melhorouMSE. O portão pré-fixado
exigia ganho nos dois jogos; portanto **falhou**. Não promover o corpoPong,
trocar critérios ou interpretar o resultado como generalidade humana.

Resíduo de transporte com predecessor real:0,010876Breakout e0,000904Pong;
resíduo de recompensa local:0,074515/0,095366. Esses erros não substituem
decisão. Normas espectrais dos operadores são maiores que1; isso não prova
instabilidade, pois a base não é ortonormal, nem autoriza extrapolar para
sequências maiores que16. Não alteramos regularização/suportes após os números.

Cinco testes passaram: execução por produto matricial igual à recursão,
ordem das ações não comutativa, recompensa terminal seguida de absorção,
exportação/reexecução, entradas inválidas e controles de contexto/transporte.
Auditor separado reconciliou1.152eventos reais, métricas, controles e portões,
com erro máximo2,75e-14 entre matriz aumentada e recursão. Principal6,78segundos,
excluindo implementação/congelamento/auditoria. Zero chamadasALE, quatro
ajustes (primário/embaralhado por jogo). Projeto original não foi editado.

Código e comandos em`TRANSPORTE_FUNCOES.md`. Fontes congeladas, corpos,
resultados por horizonte e auditoria em`resultados/transporte_funcoes*`.
O resultado autoriza investigar a composição de funções, não dizer que ela
é a única peça que falta. O próximo contraste útil é trocar, somente como
oráculo diagnóstico, as funções transportadas pelas funções da história real
antes de cada ação, preservando o leitor de recompensa. Se isso recuperar
decisão, localizará perda no transporte para esse corpo; se não recuperar,
leitura de consequência/cobertura/representação ainda não estarão separadas.
Nenhum desses casos garante invenção irrestrita ouAGI.
