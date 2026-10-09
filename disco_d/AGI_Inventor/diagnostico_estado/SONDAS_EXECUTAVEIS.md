# Sondas construídas a partir de falhas: implementação e alcance

A meta de mente de nível humano permanece aberta. Esta implementação testa
um passo necessário proposto: produzir novas perguntas executáveis a partir
de experiências, sem escrever objetos ou relações do jogo. O mecanismo atual
enumera palavras curtas de ações anônimas e extrai uma projeção sensorial do
resíduo de um par. **Não inventa ainda a linguagem de leitura**; ela é uma
projeção linear fixa dos pixels. Nenhuma candidata foi promovida.

## Código e reprodução

Em `D:\AGI_Inventor\diagnostico_estado`, execute:

```powershell
py -3.11 -B testes_sondas.py
py -3.11 -B testes_sondas_decisao.py
py -3.11 -B testes_extrator_experiencias.py
py -3.11 -B testes_conjuntos_sondas.py
```

Os controles verificam uma sequência composta que não está cadastrada, ausência
de sinal, orçamento, melhora em episódios separados, separação de treino e um
caso XOR em que nenhuma leitura isolada basta. Não são evidência de AGI.

Primeiro gerador, por divergência visual:

```powershell
py -3.11 -B sondas_contraexemplo.py breakout --seed 97
py -3.11 -B sondas_contraexemplo.py breakout --seed 103
py -3.11 -B avaliar_sondas.py breakout --seed 103
py -3.11 -B diagnosticar_extrator_sondas.py breakout
```

Segundo gerador, por erros de decisão e conjuntos de ações ótimas incompatíveis:

```powershell
py -3.11 -B sondas_decisao.py breakout --seed 97
py -3.11 -B sondas_decisao.py breakout --seed 103
py -3.11 -B selecionar_sondas_decisao.py breakout
py -3.11 -B avaliar_sondas_experiencias.py breakout
py -3.11 -B conjuntos_sondas.py breakout
```

Repetir com `pong`, depois `py -3.11 -B auditar_sondas.py`. Os bancos de
horizontes e estado preditivo existentes são pré-requisitos. Os scripts novos
não modificam os coletores congelados da confirmação anterior.

## Contrato e contabilização

Cada definição armazena sequência de ações, duração do bloco, vetor completo
de leitura, par de origem e controles. As palavras são construídas por produto
do alfabeto, até duas ações: quatro palavras de tamanho um e dezesseis de
tamanho dois. Não há nomes de famílias ou conceitos do jogo.

Todas as reconstruções e intervenções contam no limite de 50.000 passos ALE
por coleta. A versão visual usou 13.928/13.995 passos no treino de
Breakout/Pong e 11.976/12.254 em 103. A versão de decisão usou
26.320/26.360 no treino e 25.124/26.456 em 103. Replays e histórico foram
conferidos. Ainda não existe medição agregada de CPU de todo o pipeline;
estes contadores não são uma prova de orçamento computacional total igual.

Para uma palavra `(a,b)`, o extrator de experiências usa a soma da mudança
medida sob `a` com a mudança medida sob `b` no estado realmente alcançado
após `a`. As 16 composições foram conferidas nas 64 raízes contra execução
direta: diferença máxima 7,45×10⁻⁹ em Breakout e zero em Pong, nos quadros
normalizados 21×16. O treino usa até 512 estados para leituras simples e até
256 por primeira ação para leituras compostas, estratificados apenas nos
episódios permitidos.

Os episódios 0–3 de 97 geram testemunhas e treinam; 4–5 selecionam; 6–7
verificam. A base preditiva usada como conhecimento anterior já fora
aprendida em toda a semente 97: os episódios reservados são reservados para
**o alvo de decisão e seleção destas sondas**, não uma confirmação de mundo
inteiramente independente. A semente 103 também já foi vista em outros
ensaios. Nenhum resultado desta rodada é confirmação independente.

## Resultado e defeito do teste

Sondas isoladas não passaram. A seleção de conjuntos testou 200 propostas
em Breakout e 114 em Pong, incluindo propostas aleatórias e grupos por
programa/testemunha, sem exigir melhora das partes isoladas. Controles
receberam os mesmos conjuntos, dimensões e seletor.

Em Breakout, um conjunto reduziu arrependimento na seleção de 0,375 para
0,250, mas empatou com a base em 0,5625 nos episódios reservados e piorou
de 0,40625 para 0,421875 em 103. Aleatório e embaralhado obtiveram ganhos
de seleção ainda maiores, que também desapareceram. A candidata foi rejeitada.

Em Pong, o episódio 5 tem arrependimento zero na base. O critério de melhora
estrita nos dois episódios é impossível nessa amostra. O diagnóstico é
**inconclusivo por ausência de margem no portão**, não prova de incapacidade
do gerador. Esse defeito é registrado em JSON e não foi corrigido retroativamente
para anunciar sucesso. Uma futura bancada deve verificar a margem antes de
selecionar e usar diferença agregada pareada com controles, congelada antes
de uma semente nova.

Também não há evidência de que mais histórico ou resolução, com este extrator,
resolvam o problema. No mesmo conjunto de perguntas de decisão, reutilizar
estados intermediários mudou a razão média de erro de extração de
0,641 para 0,693 em Breakout e de 0,583 para 0,515 em Pong; melhorar a
predição de perguntas não foi suficiente para melhorar decisão.

## Hipótese que merece o próximo teste

O mecanismo encontrou leituras que separam **instâncias**; não descobriu uma
relação que preserve utilidade em situações novas. A projeção em coordenadas
fixas da tela é uma restrição importante da linguagem, mas ainda não foi
demonstrado que seja a causa única. A próxima comparação deve permitir
leituras executáveis com posição variável e relações entre trechos temporais,
aprendidas dos próprios resíduos, mantendo os mesmos ramos e controles. Isso
testa expressão/regularidade da variável, em vez de só fornecer mais pixels.

Não atribuir o resultado à falta de um dataset universal ou à ausência de uma
"última peça" de AGI. O diagnóstico acima é específico destes dados, dessa
linguagem de perguntas, dos estimadores e desse protocolo de seleção e validação.
