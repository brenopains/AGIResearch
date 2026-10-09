# Resultado do primeiro diagnóstico de estado

Executado em 2026-09-26/27, código isolado em D:. Duas coletas de desenvolvimento, sem jogos lacrados. O projeto original permaneceu intacto.

## O que o teste encontrou

Em cada jogo foram 64 estados, de oito episódios independentes; episódios 0–3 construíram, 4–5 selecionaram, 6–7 avaliaram. As restaurações do mesmo estado deram **0 falhas** em oito checagens por jogo. Breakout usou 28.883 passos ALE; Pong, 34.624. Os desfechos de quatro programas de ação foram observados a partir de cada estado clonado.

| Jogo, horizonte 128 | Dois quadros 21×16: erro de contraste | 16 quadros 21×16 | Dois quadros 42×32 | Placebo de ação | Estados de avaliação com diferença de recompensa |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,00007483 | 0,00012991 | 0,00007255 | 0,00033430 | 5/16 |
| Pong | 0,00003309 | 0,00005912 | 0,00003401 | 0,00010273 | 3/16 |

Menor erro é melhor. O controle embaralhado perde claramente para o modelo que conhece qual programa causou qual imagem. Há sinal de previsão de consequências visuais, mesmo na representação curta. Acrescentar os 16 quadros **piorou** este estimador nos dois jogos; aumentar a resolução produziu diferença pequena e inconsistente. Isso não demonstra que história longa seja inútil: os braços têm dimensões diferentes e só 32 estados de construção. Demonstramos apenas que empilhar quadros diretamente não aproveitou informação adicional sob este orçamento.

O gerador propôs uma leitura de memória (média de uma região passada). Em Breakout ela piorou a validação e a avaliação: ganho de teste **−8,50%**, rejeitada. Em Pong melhorou validação e teste: ganho de teste **+3,17%**; isso é um sinal pequeno em uma semente, insuficiente para afirmar transferência ou descoberta robusta. A definição exata e os erros estão nos JSON de avaliação.

O diagnóstico mais limitante foi a escassez de **consequências relevantes**. No horizonte 8 não houve sequer um estado reservado com retorno diferente entre os quatro programas; em 32 houve 3/16 em Breakout e 1/16 em Pong; em 128, 5/16 e 3/16. Essas contagens são de estados, não de episódios independentes. Assim, a escolha de ação offline tem pouquíssimos exemplos decisivos; um arrependimento médio aparentemente bom em Breakout não demonstra um decisor capaz. Ademais, são quatro programas constantes, não uma busca livre por experimentos.

## Implicação para a hipótese de AGI

Existe uma impossibilidade matemática local: se dois passados distintos geram o mesmo `r0`, nenhuma função `z=f(r0)` distingue esses passados. Os testes sintéticos verificam alias temporal e espacial. **O banco real não demonstrou que essa impossibilidade seja o gargalo dominante hoje.** O gargalo observável nesta bancada é anterior: o conjunto de intervenções raramente cria diferenças na recompensa e a previsão visual média pode favorecer pixels fáceis em vez de consequências para decisão.

A próxima aposta técnica é um **gerador de perguntas contrafactuais**, não apenas um compressor de imagens. Ele deve construir programas de ação compostos a partir das ações disponíveis, procurar onde dois modelos de estado fazem previsões distintas e executar primeiro o programa mais curto capaz de decidir entre eles. Quando duas histórias com a mesma representação atual exigirem respostas diferentes ao mesmo programa, esse par vira contraexemplo que obriga a criação de um estado de memória executável. O programa e a variável devem ser retestados em episódios separados; o custo da busca e de todas as ações entra na conta. O teste crítico é se o gerador, sem nomes de jogo, produz **mais estados com consequências relevantes por passo** e melhora escolha/transferência em jogo reservado, contra programas aleatórios sob o mesmo orçamento. Se gerar contraste sem melhorar decisão, faltará aprendizado de valor; se não gerar contraste, faltará linguagem de intervenção/alcance; se houver contraste mas a memória não generalizar, faltará construção de estado causal.

Essa hipótese se relaciona a estados preditivos e abstração comportamental existentes na literatura; sua eficácia e integração aqui ainda não estão demonstradas. Nenhum resultado deste piloto estabelece nível humano, descoberta científica inédita ou suficiência para AGI.

## Segundo ensaio, exploratório: programas compostos

O protocolo separado está em `PERGUNTAS.md`. Dezesseis programas de duas ações (32 + 96 passos) foram executados a partir de cada um dos 32 estados pareados por jogo. Foram 60.721 passos em Breakout e 67.520 em Pong, com zero falhas de replay. A seleção de quatro programas usou só episódios de construção; os episódios 6–7 ficaram reservados.

| Jogo | Selecionados: estados informativos na avaliação | Quatro constantes | Média de 128 conjuntos aleatórios | Nulos que igualaram o selecionado |
|---|---:|---:|---:|---:|
| Breakout | 8/8 | 7/8 | 6,66/8 | 44/128 |
| Pong | 6/8 | 5/8 | 4,70/8 | 36/128 |

Os programas compostos selecionados ganharam um estado informativo sobre as constantes em cada jogo, mas **não se destacaram dos conjuntos aleatórios**. O teste mostra que a gramática de duas ações consegue produzir mais diferenças; não mostra que o sistema tenha aprendido a escolher perguntas melhores que acaso. Não houve medição de melhoria do agente jogando. A próxima tentativa precisa tornar a escolha de intervenção informada por desacordo entre modelos testáveis, e compará-la com nulo pareado antes de qualquer alegação de progresso geral.

## Reprodutibilidade e proveniência

Comandos no README. Os arquivos `resultados/breakout_bank.npz` e `resultados/pong_bank.npz` têm SHA-256, respectivamente, `8884ee78f4e2bce52cb623e4babbba5115078e66150b7f6f5f01ce8ff5c821ec` e `9d9f13d3b7e6a1999aa6a87a1fe24847e36f99a0744668597442918b68d86adf`. Relatórios completos: `resultados/breakout_diagnostico.json` e `resultados/pong_diagnostico.json`.

Arquivos consultados no sistema anterior, somente leitura, SHA-256: `instrumentos/instrumentos.py` `17B97226FDE6F622E7DF4AAA18D8E2960AC8F1A3C9ECD0C92F60496B04B40ED9`; `instrumentos/ambiente.py` `CB569D81EC0539C401838BF0A226FA711AFAB1B9059CC71450626AB6BD1CE07B`; `instrumentos/agente.py` `9E82275DDF7FFA5381D5C3000C2CBECC22335838A926644E657ADDDFBC5E03DD`.

Segundo ensaio: `resultados/breakout_perguntas.npz` SHA-256 `b3be3b42ae9c5c5ad456634d573189190240069f2b9e16b22fe9dae1486d6007`; `resultados/pong_perguntas.npz` SHA-256 `8511d53b8bf7753bde40c79b2612bac458b00e9d904796be38643107e32fa9dd`. Os JSON correspondentes terminam em `_perguntas_diagnostico.json`.

## Terceiro ensaio: pergunta e memória inventadas juntas

O protocolo está em `CICLO.md`. Construí um gerador que, para cada programa, extrai a direção de consequência visual causada pela ação em relação ao programa neutro, detecta erros fora do próprio episódio e busca uma leitura executável de quadros anteriores que corrija a previsão. Um controle positivo sintético confirmou que o mecanismo recupera uma variável temporal ausente do presente e a pergunta que a revela.

Em jogos reais, foram 64 raízes por jogo, 16 ramos por raiz, 110.311 passos ALE em Breakout e 133.824 em Pong, com 8/8 replays corretos em cada. Os episódios 6–7 não entraram na construção/seleção.

| Jogo | Ganho visual reservado | Nulos embaralhados iguais/melhores | Raízes com diferença de recompensa para a pergunta escolhida | Escolha de recompensa |
|---|---:|---:|---:|---|
| Breakout | +2,84% | 14/32 | 4/16 | Sem melhora: 4/4 antes e depois |
| Pong | +46,43% | 0/32 | 0/16 | Não avaliável |

O portão pré-definido **falhou nos dois**. Em Pong há informação temporal real para antecipar uma consequência visual, mas a pergunta selecionada não muda recompensa. Inspeção apenas diagnóstica do banco mostrou que outra pergunta tinha contraste de recompensa; portanto faltou selecionar a pergunta pelo efeito na decisão, não faltou toda variação de valor no ambiente. Não usamos a avaliação para alterar o seletor deste ensaio.

## Quarto e quinto ensaios: resolver o sufixo e selecionar por decisão

Como análise exploratória separada, tratei cada programa como prefixo de 32 passos mais sufixo de 96. Com os 16 ramos efetivamente executados, `V(prefixo)=máximo retorno entre quatro sufixos`. Isso fez o valor dos prefixos diferir em 13/16 estados reservados de Breakout e 4/16 de Pong. Um controle positivo sintético confirmou que o código de valor reduz arrependimento quando a variável causal só aparece no passado.

Aprender `V` com os pixels atuais produziu arrependimento normalizado de **0,115** em Breakout (uniforme 0,506) e **0,25** em Pong (uniforme 0,719). Adicionar uma memória escolhida por erro quadrático piorou Breakout para 0,167 e não mudou Pong. Uma nova seleção explicitamente orientada por arrependimento piorou Breakout para 0,244 e deixou Pong em 0,25; os nulos igualaram/superaram os ganhos em 22/32 e 14/32. Ambos os portões falharam. Esses testes reutilizaram banco cuja contagem de contrastes já tinha sido vista: **exploratórios, não confirmação**.

O resultado mais específico é que construir uma leitura temporal que prevê melhor pixels, ou até o valor médio, não basta para mudar a ação correta. O alvo precisa ser o **limite entre decisões**: pares de histórias que a representação atual confunde mas cujo melhor prefixo, depois de resolver o sufixo, é diferente. O próximo gerador deve procurar esses pares e sintetizar a menor variável que separa seus resultados em episódios novos. Também deve construir intervenções além da gramática fixa de 16 programas, com custo contabilizado. Sem passar um nulo pareado e sem jogar melhor fora da coleta, isso permanece uma hipótese de engenharia, não uma capacidade geral demonstrada.

Uma inspeção final mostrou por que a busca direta por separadores ainda é mal condicionada: entre 32 raízes de construção, apenas **12** em Breakout e **3** em Pong tinham um prefixo ótimo único. Logo, antes de ampliar a linguagem de memória, o sistema precisa **encontrar estados que exponham decisões diferentes**. Esta é uma hipótese verificável: aprender a selecionar raízes para sondagem a partir de dados anteriores e medir, sob o mesmo custo total, quantos contraexemplos de decisão independentes gera em novos episódios. Se falhar frente a raízes aleatórias, a limitação é alcance/geração de experiências; se passar e ainda não melhorar decisão, a limitação é síntese do estado ou do decisor.

Terceiro banco: `resultados/breakout_ciclo.npz` SHA-256 `364a3084065d62cf046ef51b1100d3562aade906de9fc6783a5b304f731b802d`; `resultados/pong_ciclo.npz` SHA-256 `43e7bf4990ddbfa6c35265044f689230b7551ad70569d0e8ceac951abbf2384f`. Relatórios de avaliação: sufixos `_diagnostico.json`, `_prefixo.json` e `_decisao.json`.

## Sexto ensaio: escolher onde vale sondar

O protocolo exploratório está em `AMOSTRADOR.md`. Um seletor só de pixels treinado para reconhecer estados com prefixo ótimo único escolheu 2/4 raízes reservadas em Breakout e 2/4 em Pong. Subconjuntos aleatórios de quatro raízes igualaram ou superaram o seletor em 471/1.000 e 236/1.000, respectivamente. **Falhou o portão nos dois jogos.** Esse teste apenas escolhe entre raízes já alcançadas por trajetória aleatória; não mede um agente que as alcança deliberadamente. Relatórios: `resultados/breakout_ciclo_amostrador.json` e `resultados/pong_ciclo_amostrador.json`.

## Decisão de engenharia depois dos nulos

Este conjunto de testes não autoriza afirmar que identificamos “a única peça” da AGI. Encontramos uma cadeia com gargalos distinguíveis: (1) chegar a estados com escolhas verdadeiramente diferentes; (2) construir perguntas que exponham a diferença; (3) formar variável histórica que antecipe **qual ação vence**, não só pixels ou valor médio; (4) usar isso em interação contínua sem oráculo de rollback. O código atual resolve casos sintéticos em que o passado contém diretamente a variável, mas não transportou essa capacidade para decisão nos dois jogos.

A próxima implementação deve trabalhar nesses elos em ordem, com portões locais e nulos. Um candidato concreto é busca bidirecional por contraexemplos de decisão: de um estado, sintetizar sufixos que maximizem consequências; de dois passados visualmente próximos com vencedores diferentes, localizar o menor trecho histórico que separa a preferência; então buscar um novo estado em que as duas hipóteses discordem e executar a intervenção. Promover uma variável somente após mudar decisões corretas em novos episódios e novos mundos, contabilizando toda execução e busca. Isso é uma hipótese de pesquisa agora especificada; ainda não passou nos dados atuais.

## Sétimo ensaio: compilador de contraexemplos decisórios

O módulo `contraexemplos.py`, definido em `CONTRAEXEMPLOS.md`, implementa parte concreta da busca bidirecional: identifica perdas reais de ação fora do próprio episódio, deriva uma direção de erro dos valores, sintetiza operadores esparsos de quadros antigos e os valida por arrependimento. Toda definição produzida inclui origem temporal, índices de pixels, pesos e centros; é executável e inspecionável. Um controle positivo com estado oculto no passado passou, enquanto um operador com somente quadro atual não recuperou a decisão. Isso valida a implementação da capacidade local, não sua generalização.

| Jogo | Contraexemplos de construção | Arrependimento base reservado | Com variável histórica | Nulos tão bons/melhores | Portão |
|---|---:|---:|---:|---:|---|
| Breakout | 6 | 0,115 | 0,192 | 13/32 | Falhou |
| Pong | 0 | 0,25 | Não há variável admissível | Não aplicável | Falhou por dados insuficientes |

O ensaio reutilizou os bancos cujo comportamento já era parcialmente conhecido e é exploratório. Em Breakout, o ganho de validação não se sustentou na avaliação. Em Pong, recusar a síntese foi a resposta correta do mecanismo: não há contraexemplos decisórios de construção para orientar uma variável. Isso não prova que o passado seja dispensável em Pong; prova que **este experimento não obteve a evidência necessária**. Não vou ajustar a gramática usando a avaliação reservada.

O próximo trabalho justificável é aquisição de contraexemplos sob orçamento pareado, incluindo estados alcançados por programas novos e teste em mundos além desses dois jogos. A condição de entrada para a síntese será haver erros decisórios independentes; se nem um buscador de estados superar amostragem aleatória no custo por contraexemplo, a aposta de co-invenção de pergunta/estado terá falhado no seu primeiro elo.

## Oitavo ensaio: alcançar contraexemplos com trajetórias novas

O protocolo congelado está em `AQUISICAO.md`; código em `aquisicao.py`. O decisor foi treinado somente com episódios 0–5 do banco anterior. Em nova semente 43, cada episódio oferece o **mesmo estado inicial** aos braços ativo e aleatório. Cada braço gera oito trajetórias de quatro segmentos; depois de cada trajetória paga por 16 ramos para verificar exatamente se o prefixo escolhido pelo decisor perdeu valor. Um teste de replay passou antes da rodada.

| Jogo | Episódios com falha, ativo | Aleatório | Passos ativo | Passos aleatório | Portão |
|---|---:|---:|---:|---:|---|
| Breakout | 7/8 | 8/8 | 131.888 | 132.110 | Falhou |
| Pong | 0/8 | 0/8 | 135.168 | 135.168 | Falhou |

Breakout teve 26 sondas ativas com erro versus 18 aleatórias, mas elas se concentraram em episódios que já tinham erro; a métrica primária conta cada episódio uma vez, e o nulo venceu. Em Pong, 64/64 sondas por braço tiveram diferenças de valor entre prefixos, e aproximadamente metade teve vencedor único, porém **todos os vencedores únicos eram o prefixo 3**, que o decisor congelado sempre escolheu. Assim o zero não decorre de ausência de recompensa ou de falha de replay: o subproblema definido pela gramática e pelos estados iniciais é degenerado para decisão. Expandir memória não pode melhorar uma escolha que já está correta nesses estados.

Este piloto não demonstrou aquisição superior a acaso. A conclusão prática é parar de usar apenas essa janela inicial de Pong como prova da capacidade de inventar decisões. A próxima avaliação deve diversificar profundidade temporal, tipos de mundo e gramática de ações **antes** de testar um novo aprendiz; caso contrário continuaremos confundindo teto do teste com teto da arquitetura. Não existe base para declarar que isso é a última peça para nível humano.

Arquivos de auditoria: `resultados/breakout_aquisicao.json` e `.jsonl`; `resultados/pong_aquisicao.json` e `.jsonl`. Os JSON incluem SHA-256 dos eventos, passos e resultados por episódio.

## Nono ensaio: contrato comum e limite da percepção visual

O protocolo em `GERAL.md` converteu Breakout, Pong e um mundo físico de três
corpos em histórias `N×16×D` e quatro valores de decisão, sem mudar o compilador
de variáveis de `contraexemplos.py`. Não há transferência demonstrada: em
órbitas, a variável histórica elevou arrependimento reservado de 0,405 para
0,510 (32/32 nulos tão bons ou melhores); em Breakout, de 0,115 para 0,192
(13/32); em Pong não havia contraexemplo de construção. Ampliar a memória para
até 16 componentes não corrigiu esse quadro: órbitas 0,405→0,389, igualado pelo
controle só do presente e por 20/32 nulos; Breakout piorou para 0,269; Pong
ficou em 0,25. Ver `CAPACIDADE.md`, `geral.py` e `capacidade.py`.

Como auditoria separada, reproduzi exatamente os 64 alvos do mundo orbital e
dei **somente ao avaliador** as posições e velocidades atuais do simulador.
Com o mesmo estimador e divisão, arrependimento caiu de 0,405 nos pixels para
0,146 no estado oracular, com 0/32 oráculos embaralhados tão bons. Posições
oraculares sozinhas também obtiveram 0,146; velocidades sozinhas, 0,279.
Isso aponta informação geométrica utilizável no primeiro mundo, mas é um
resultado de apenas dois episódios de avaliação, já inspecionados. Não mede
aprendizagem autônoma de representação.

Implementei então extração sem verdade oculta: componentes de primeiro plano,
centros, correspondência temporal e propagação sob oclusão. Quando os três
componentes estão separados, os centros diretos erram em média 0,18 pixel na
avaliação da semente 53; isso ocorre em apenas 7/16 casos. O rastreador de
história integral mantém três trilhas em 16/16, mas seu erro de posição sobe
para 2,00 pixels e seu arrependimento permanece 0,354. O ganho frente a pixels
0,405 é insuficiente para concluir solução do elo visual.

Depois de fixar esse código, `SUITE_FISICA.md` registrou quatro mundos com
câmera, trajetórias e oclusões novas. A tabela mostra arrependimento normalizado
reservado; menor é melhor. Nulos são 32 permutações do estado oracular ou das
trilhas dentro da divisão de ajuste/avaliação, e a coluna registra quantos
igualaram ou venceram o observado.

| Semente | Pixels | Oráculo | Trilhas | Nulos oráculo | Nulos trilhas | Erro posição trilhas |
|---:|---:|---:|---:|---:|---:|---:|
| 97 | 0,872 | 0,671 | 0,626 | 14/32 | 10/32 | 0,26 px |
| 101 | 0,071 | 0,010 | 0,115 | 0/32 | 11/32 | 12,65 px |
| 107 | 0,421 | 0,186 | 0,186 | 3/32 | 1/32 | 0,23 px |
| 113 | 0,413 | 0,415 | 0,551 | 3/32 | 15/32 | 0,43 px |

O oráculo melhorou 3/4 mundos, mas o controle forte foi superado claramente
em apenas um; no quarto não ajudou. O rastreador melhorou 2/4, capturou ao
menos metade do ganho oracular em 2/4, e só em uma semente ficou à frente de
quase todos os nulos. O caso 101 evidencia troca incorreta de identidade ou
propagação equivocada apesar de três trilhas declaradas. O caso 113 lembra que
mesmo coordenadas exatas não garantem previsão/decisão com 48 exemplos de
ajuste. Portanto **não há evidência de transferência robusta nem de um salto
geral**. O próximo ensaio deve testar hipóteses de identidade e dinâmica como
programas de estado com incerteza explícita e intervenções, e exigir ganho
pareado em mundos inéditos. Adicionar datasets indiscriminadamente ou chamar
mais RAM de representação não é sustentado por estes dados.

Artefatos: `resultados/orbitas_geral.npz`, `*_geral_diagnostico.json`,
`*_geral_capacidade.json`, `orbitas_geral_oraculo.json`,
`orbitas_geral_percepcao_pixels.json` e `resultados/suite_fisica.json`.
As quatro sementes novas têm bancos e resultados próprios no mesmo diretório.
Todos foram criados em `D:\AGI_Inventor`; o projeto original ficou intacto.

## Décimo ensaio: manter hipóteses de identidade em vez de uma trilha única

`hipoteses_estado.py` mantém até 16 atribuições de componentes a entidades.
Cada uma acumula custo de erro de previsão observável, pode deixar uma trilha
temporariamente sem observação e emite posições, velocidades e incerteza. O
filtro lê só quadros anteriores ao instante de decisão; a verdade física e os
valores futuros são usados exclusivamente no avaliador. A aposta e seu limite
estão em `HIPOTESES_ESTADO.md`.

No conjunto **já conhecido**, a média das hipóteses melhorou o arrependimento
frente aos pixels em 4/5 sementes. Na semente 101, que derrotara o rastreador
guloso, passou de 0,071 para 0,017, com 0/32 nulos embaralhados tão bons.
Esse sucesso não era confirmação; foi usado para verificar o instrumento.

`CONFIRMACAO_HIPOTESES.md` congelou fonte e critério antes de seis mundos
novos. O script `suite_hipoteses.py` registrou hashes SHA-256 dos códigos antes
e depois, hashes dos bancos, tempos e resultados por mundo em
`resultados/suite_hipoteses.json`. Valores abaixo são arrependimento normalizado
reservado (menor é melhor). A coluna de nulos refere-se à **média** de hipóteses.

| Semente | Pixels | Guloso | Melhor hipótese | Média de hipóteses | Nulos da média |
|---:|---:|---:|---:|---:|---:|
| 127 | 0,794 | 0,660 | 0,660 | 0,688 | 15/32 |
| 131 | 0,048 | 0,026 | 0,026 | 0,026 | 2/32 |
| 137 | 0,450 | 0,424 | 0,455 | 0,225 | 2/32 |
| 139 | 0,548 | 0,303 | 0,285 | 0,251 | 7/32 |
| 149 | 0,374 | 0,007 | 0,007 | 0,007 | 0/32 |
| 151 | 0,000 | 0,000 | 0,000 | 0,000 | 11/32 |

O critério exigia vencer **pixels e guloso** em ≥5/6 mundos, e no máximo um
nulo tão bom em ≥4/6. A melhor hipótese fez 1/6 e 1/6; a média fez 2/6 e
1/6. **Ambas falharam.** O caso 149 tem nulo forte, mas empata com o guloso;
o caso 151 não oferece margem de melhoria por arrependimento. A execução
completa custou 132 segundos de tempo de parede acumulado, incluindo
geração e auditorias.

O resultado distingue um ganho local real em 101 de uma capacidade robusta.
Mais largura de feixe não é o próximo passo justificado. O elo ainda aberto é
**vincular uma hipótese de estado a previsões e intervenções que alterem a
escolha**, aprender isso em mais de um mundo e mostrar vantagem em episódios
que oferecem margem. Nesta bancada passiva e com o mesmo regressor, a maioria
das alternativas de identidade não mudou a ação corretamente.

## Diagnóstico adicional: a crença muda a decisão?

`decisao_crencas.py` testa a diferença matemática entre estimar valor do estado
médio `Q(E[z])` e tirar média dos valores de cada hipótese `E[Q(z)]`. É uma análise
**exploratória em mundos já abertos**, sem novo portão de confirmação. Nos quatro
mundos examinados, o segundo cálculo não venceu consistentemente o primeiro:

| Semente | Melhor estado | Estado médio | Valor médio | Escolhas alteradas pelo valor médio | Raízes com >10% de peso em ação concorrente |
|---:|---:|---:|---:|---:|---:|
| 101 | 0,022 | 0,017 | 0,043 | 2/16 | 2/16 |
| 113 | 0,496 | 0,586 | 0,527 | 1/16 | 1/16 |
| 137 | 0,455 | 0,225 | 0,455 | 0/16 | 2/16 |
| 139 | 0,285 | 0,251 | 0,285 | 0/16 | 0/16 |

A incerteza de identidade raramente era **incerteza de escolha**. Uma visão
adicional que só distingue identidades, sem separar previsões de valor, tem
baixo valor neste banco. O próximo mecanismo precisa gerar e alcançar
**situações onde modelos de estado/dinâmica plausíveis recomendam testes ou
ações diferentes**, então usar a resposta observada para eliminar modelos.
Contar tais situações e compará-las com busca aleatória deve preceder outro
treinamento de representação. Esta conclusão é sobre o instrumento atual, não
uma prova de que a identidade de objetos é irrelevante para inteligência geral.

## Décimo primeiro ensaio: escolher experiências por desacordo entre modelos

`DESACORDO_ATIVO.md` definiu o teste antes da execução. O mesmo pool de 24
trajetórias de quatro segmentos de 32 ações foi gerado para ambos os braços em
cada episódio. Doze modelos treinados em episódios antigos votaram no prefixo
de ação; o braço ativo selecionou quatro estados com maior desacordo e o nulo
selecionou quatro ao acaso. **Somente depois da seleção** os 16 ramos
prefixo/sufixo foram executados a partir de cada snapshot. O decisor, as ações
e o orçamento foram idênticos; replay de snapshot passou. Fumaça independente
passou em `testes_desacordo.py`.

| Jogo | Episódios com erro ativo/aleatório | Sondas com erro ativo/aleatório | Vencedor único ativo/aleatório | Candidatos com votos não unânimes | Passos totais |
|---|---:|---:|---:|---:|---:|
| Breakout | 8/8 · 8/8 | 18/32 · 10/32 | 22/32 · 18/32 | 117/190 | 146.699 |
| Pong | 7/8 · 8/8 | 27/32 · 23/32 | 27/32 · 13/32 | 114/192 | 156.160 |

**Portão pré-definido falhou nos dois.** Breakout saturou a métrica primária:
o aleatório já achou erro em todos os episódios. Em Pong o braço ativo perdeu
um episódio. As contagens secundárias sugerem enriquecimento de sondas com
erro/contraste, especialmente de vencedor único no Pong, mas não autorizam
declarar superioridade: foram quatro sondas correlatas por episódio e uma
única semente. O braço ativo em Breakout gastou 62.153 passos de verificação
contra 59.496 do aleatório, também falhando a condição de custo.

Uma falha explicativa ficou explícita em Pong, episódio 0: os modelos votaram
por ações diferentes (2, 6, 0, 4 votos), porém os quatro valores **reais** de
prefixo após resolver o sufixo foram todos −4 nas quatro sondas ativas. Desacordo
de preditores pode refletir incerteza fora da distribuição sem haver uma
diferença causal de decisão. Esse é o nulo que o experimento precisava expor.
O mecanismo seguinte deve obter uma *testemunha de contraste* entre duas
previsões de consequência, executá-la com custo contabilizado e só então
investir em ramificações mais caras. Não promover pontuação de desacordo como
"promessa" geral. Comparar com seleção aleatória no mesmo pool e exigir ganho
em episódios independentes continua obrigatório.

O piloto atual alterou simultaneamente profundidade das trajetórias, tamanho
do pool e semente em relação à aquisição anterior; a mudança de 0/8 para
8/8 de erros do braço aleatório em Pong **não isola** qual fator causou a
diferença. Arquivos: `desacordo_ativo.py`,
`resultados/breakout_desacordo_ativo.json` e `.jsonl`, e correspondentes do
Pong. Os relatórios incluem SHA-256 dos códigos, banco de origem e eventos.

## Décimo segundo ensaio: a pergunta precisa de uma testemunha verificável

Depois do nulo do comitê, medi a condição anterior a qualquer busca ativa.
`calibracao_modelos.py` usa somente os episódios 0–5 para estimar o quantil 90%
do erro fora do próprio episódio. Nos episódios 6–7, os intervalos da previsão
média cobriram todos os quatro valores em 13/16 raízes de Breakout e 9/16 de
Pong. Mas o conjunto de ações ainda potencialmente ótimas tinha tamanho médio
**4,0/4** e **3,875/4**: cobertura comprada com intervalos largos não escolhe
experimento nem decisão. Em Pong, 11/13 raízes com votos não unânimes não
tinham contraste real de retorno.

Implementei `testemunha.py`: duas hipóteses só autorizam uma consulta se
predizem intervalos **disjuntos para a mesma ação no mesmo estado**. O módulo
registra as duas previsões e o custo, elimina modelos incompatíveis após a
observação e retorna `modelos_inadequados` se nenhum sobreviver. Testes exatos
verificam discriminação, ausência de testemunha e refutação de todos; passaram.
Para previsões estatísticas, "testemunha" significa separação sob os
intervalos calibrados, nunca prova exata.

`auditar_testemunhas.py` aplicou o contrato aos 12 membros do comitê, com raio
calibrado **por membro** em predições fora do próprio episódio (mais conservador
que o raio da média usado no diagnóstico anterior):

| Jogo | Raízes reservadas com consulta discriminante | Com algum modelo compatível com os quatro valores | Consultas executadas que refutaram todos os modelos |
|---|---:|---:|---:|
| Breakout | 0/16 | 16/16 | 0 |
| Pong | 3/16 | 10/16 | **3/3** |

O resultado é diagnóstico, pois os bancos já tinham sido examinados. Ainda
assim, ele determina a próxima dependência de engenharia: um novo gerador de
modelos de consequência deve produzir **previsões testáveis com cobertura e
separação**, não apenas votos diferentes. Seu portão futuro precisa medir, em
episódios e domínios novos, (1) cobertura dos desfechos por pelo menos uma
hipótese, (2) consultas cujos resultados previstos se excluem e (3) atualização
que conserva uma hipótese após a execução. Se houver cobertura sem separação,
o modelo é vago; se houver separação e todos forem refutados, é falso. Nenhum
seletor de perguntas conserta essas duas falhas.

Artefatos: `CALIBRACAO_MODELOS.md`, `TESTEMUNHAS.md`,
`resultados/{breakout,pong}_calibracao_modelos.json` e
`resultados/{breakout,pong}_testemunhas.json`, com resultados por raiz e hashes.

## Décimo terceiro ensaio: transferir a previsão visual para estados inéditos

Reconstruí evento por evento os estados ALE do ensaio interativo da semente 59.
Os 64 ramos de cada jogo, os retornos e os contadores de passos coincidiram
com os eventos originais. Isso gerou 55 estados únicos em Breakout e 62 em
Pong, com históricos de imagem e todos os quatro valores de prefixo. Os
programas `reconstruir_desacordo.py` e `converter_ativos.py` mantêm os hashes
dos bancos, do código e dos eventos. Nada foi incorporado ao projeto original.

No banco reconstruído, a regressão kernel temporal treinada nos episódios
0–5 de Breakout obteve arrependimento normalizado 0,042 nos episódios 6–7,
contra 0,292 da melhor ação constante escolhida no treino e 0,458 do modelo
anterior. Nenhum dos 32 controles com rótulos embaralhados igualou esse ganho.
Em Pong, porém, o modelo novo e a ação constante empataram em zero, assim como
os 32 controles: o banco não demonstrou compreensão visual nesse jogo.

Antes de ver a semente 73, `TRANSFERENCIA_ALE.md` congelou a representação,
o treino e o protocolo. Oito episódios por jogo usaram oito programas de
ações aleatórios distintos cada, independentes dos modelos; todos os 16
ramos foram verificados em cada estado não terminal, com replay checado.

| Jogo | Raízes verificadas | Com contraste | Novo | Modelo antigo | Constante | Controles com histórico permutado tão bons quanto o novo |
|---|---:|---:|---:|---:|---:|---:|
| Breakout | 61 | 48 | **0,000** | 0,371 | 0,101 | 0/32 |
| Pong | 64 | 51 | 0,050 | 0,950 | 0,050 | 32/32 |

As médias são médias por episódio informativo; todos os oito episódios foram
informativos em cada jogo. Em Breakout, o novo modelo escolheu uma ação
ótima nas 48 raízes com contraste e a ação constante só em 45/48. Isso é
evidência local de informação visual útil transferida para novos estados da
mesma tarefa. **O portão pré-registrado falhou**: a vitória *estrita* sobre
ambos os comparadores ocorreu em 3/8 episódios, pois a constante empatou
nos outros cinco; o requisito era ≥6/8. Não reinterpretar 0,000 como êxito
desse portão. Em Pong, a constante foi ótima em 49/51 raízes informativas;
o novo modelo escolheu a mesma ação em todos os 64 estados e os controles
permutados não pioraram. Não há evidência de transferência representacional
para Pong; a perda >0,05 em relação à constante também não ocorreu.

Este resultado localiza o limite do instrumento: Breakout demonstrou
generalização de uma associação temporal de pixels para estados inéditos,
mas a bancada tem pouca margem sobre uma ação fixa e não testa invenção
autônoma de objetos, relações ou variáveis causais. O próximo ensaio deve
amostrar estados com contraste de decisão **sem usar o modelo avaliado para
selecioná-los**, e exigir que uma representação construída da história crua
preveja consequências separáveis e sobreviva à intervenção em mais de um
mundo. As respostas da semente 73 ficam fora de qualquer treino posterior
destinado a confirmação nessa semente.

Dados auditáveis: `resultados/{breakout,pong}_transferencia_ale.json` e
`.jsonl`. O código de avaliação congelado tem SHA-256
`46540d8b3d020a0f78de8d22dbb66156f0f60bfc8df4b74339d72595f9b7282c`.

## Décimo quarto ensaio: a trajetória revela o atalho da ação fixa

`TRAJETORIA_INDEPENDENTE.md` congelou antes da nova semente 97 uma mudança
apenas na **distribuição de estados**: em vez de partir do início e executar
macros de 128 passos, visitar oito checkpoints a intervalos de 128 ações
aleatórias sequenciais por episódio. Modelos, treino e verificador de 16 ramos
continuaram congelados. Se o jogo terminasse, a trajetória reiniciaria,
registrando o reset. Checkpoints são escolhidos pelo relógio, sem olhar a
previsão. O mesmo código rodou Breakout e Pong.

| Jogo | Raízes com contraste | Constante falha | Novo | Antigo | Constante | Episódios em que novo vence constante | Nulos tão bons |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breakout | 58/64 | 29, em 8 episódios | 0,462 | **0,188** | 0,491 | 1/8 | 7/32 |
| Pong | 21/64 | 20, em 7 episódios | 0,875 | **0,721** | 0,875 | 0/8 | 32/32 |

Métricas são arrependimento normalizado médio por episódio informativo. Ambos
os jogos atingiram o critério pré-definido de **margem adequada** (≥12 falhas
da constante em ≥4 episódios). O portão local **falhou nos dois**. Em
Breakout, o modelo novo ainda previu ação 1 em 55/64 raízes. Em Pong, previu
ação 2 nas 64/64: não houve decisão condicionada aos pixels. A vantagem
aparente do ensaio anterior em Breakout dependia de estados em que a constante
já era quase sempre ótima. Esta nova distribuição a reverteu. O modelo antigo
não é bom em termos absolutos; seu menor arrependimento apenas mostra que
maior quantidade de exemplos do novo banco não resolve esse desvio.

Foram consumidos 114.979 passos ALE em Breakout e 130.170 em Pong, incluindo
trajetórias e ramificações. Houve 41 e 4 resets, respectivamente. O primeiro
snapshot de cada jogo passou no teste de replay; os 64 históricos e todos os
resultados pareados foram salvos por jogo. SHA-256 do código congelado:
`5d1cc0aa61ab6e6422b3ef2e20d3f5e0d716ea010e7ed33a1d56835f481b8dee`.
Os hashes do código, bancos e JSONL conferiram após a execução.

O limite agora é específico: o regressor dos dois últimos quadros não aprende
uma representação que mantenha a distinção de ação sob mudanças de regime e
posição na trajetória. Antes de buscar mais dados ou mais seletor, o sistema
precisa propor variáveis temporais persistentes, atribuir-lhes efeitos
intervencionais e testar se elas mantêm calibração e discriminação em mundos
novos. Os dados da semente 97 podem ser usados para análise de falhas, mas
não como confirmação futura de um modelo ajustado a eles.

Como diagnóstico **pós-hoc** (não um novo teste confirmatório),
`diagnosticar_vistas_transferencia.py` treinou o mesmo regressor nos episódios
0–5 da semente 59 e avaliou cinco visões nos estados da semente 97: dois
quadros 21×16, 16 quadros 21×16, dois quadros 42×32, 16 quadros 42×32 e
dois quadros nativos. O arrependimento médio por episódio foi, na mesma
ordem, **0,462 / 0,488 / 0,480 / 0,465 / 0,462** em Breakout e **0,875**
em todas as visões de Pong. Nenhuma visão resolveu mais que 3 das 29 raízes
de Breakout em que a constante falhou; em Pong, 0 das 20. Portanto, ampliar
janela e resolução *sem mudar como estado é construído* não explica nem
remove esta falha do estimador. Os números, bancos de origem e código estão
em `resultados/{breakout,pong}_vistas_transferencia.json`.

## Décimo quinto ensaio: o desvio não é só tamanho da imagem

`diagnosticar_cobertura.py` comparou, exploratoriamente, treino e avaliação
**dentro** da distribuição de trajetórias longas da semente 97 (episódios
0–5 versus 6–7). Em Breakout, todos os regressores de pixels/16 quadros
ficaram em 0,339–0,411 de arrependimento, contra 0,201 da ação constante
escolhida no treino; 18/32 controles com rótulos embaralhados foram tão bons
quanto a visão básica. Em Pong, 16 quadros 21×16 deram 0,167 contra 0,417 da
constante, mas havia só quatro raízes informativas no teste e 31/32 nulos
da visão básica igualaram-na ou venceram-na. Não há promoção confiável.

Inspecionei folhas de contato dos 16 quadros em estados em que a constante
falha. Os objetos que se movem ocupam poucos pixels frente ao cenário
estático. `variaveis_movimento.py` implementou uma proposta genérica, sem
nomes dos jogos: fundo mediano temporal, componentes de mudança positiva e
negativa, continuidade entre quadros, posição, velocidade e relações entre
trilhas. Seus testes de movimento, polaridade e entrada vazia passaram. Na
avaliação exploratória leave-one-episode-out da semente 97, porém, essa
representação perdeu para a constante: **Breakout 0,344 vs. 0,209** e
**Pong 0,544 vs. 0,306**. Só venceu pixels e constante em 1/8 episódios de
Breakout e 0/8 de Pong; 16/32 e 25/32 históricos de trilha permutados foram
tão bons. A proposta não foi promovida nem levada a uma semente de confirmação.

Para testar a hipótese de que o obstáculo fosse exclusivamente perceptivo,
`oraculo_ram_ale.py` refez os 64 checkpoints da semente 97 em cada jogo e
comparou **todos os 16 quadros** de cada história com o banco salvo; 64/64
coincidiram em ambos. Os 128 bytes internos do ALE foram então usados apenas
em um regressor diagnóstico leave-one-episode-out. Também não superaram a
constante: **Breakout 0,221 vs. 0,209; Pong 0,442 vs. 0,306**. RAM crua
mistura fatores relevantes e irrelevantes; esse nulo não prova que uma
representação ideal seja impossível. Ele refuta, porém, a afirmação simples
de que basta trocar pixels por estado privilegiado mantendo este estimador,
este tamanho de banco e estes alvos longos.

Próxima dependência experimental: medir previsões de **consequências visuais
de curto prazo sob ações pareadas**, com persistência, ação cega e ação
embaralhada como controles. Se o estado construído não prever efeitos locais,
o erro está antes do planejador; se prever efeitos locais mas não retornos
longos, atacar composição temporal e crédito. As análises desta seção são
exploratórias sobre banco já inspecionado e não constituem confirmação.

Artefatos: `resultados/{breakout,pong}_diagnostico_cobertura.json`,
`_avaliacao_movimento.json` e `_oraculo_ram_ale.json`, além dos códigos e
testes correspondentes.

## Décimo sexto ensaio: consequências locais são previsíveis

O nulo da escolha de retorno a 128 passos não permitia concluir que o mundo
fosse opaco aos pixels. `EFEITOS_LOCAIS.md` mudou o alvo para o quadro após
oito ações de um mesmo comando. `coletar_efeitos_locais.py` clonou cada
checkpoint, executou quatro ramos pareados e guardou quadros, recompensas,
terminações e passos. Na semente de desenvolvimento 97, os 64 históricos
por jogo coincidiram com os bancos anteriores; a semente nova 103 teve replay
positivo. Nenhum estado interno do ALE entrou no aprendiz.

No desenvolvimento 97, a regressão kernel dos dois últimos quadros 21×16
previu os **contrastes entre ações** com razão de erro 0,154 em Breakout e
0,465 em Pong, tendo como 1 a previsão cega à ação. Uma trilha explícita de
componentes móveis foi pior (0,497 e 0,744). O molde de efeito médio fixo
por ação marcou 0,460 e 0,742. Histórico de teste embaralhado e rótulos de
ação embaralhados foram controles. A proposta de trilhas não foi promovida.

`CONFIRMACAO_EFEITOS.md` congelou entrada, saída, métrica, controles,
limiares e hashes dos códigos **antes** da coleta 103. O modelo foi treinado
somente nos 64 checkpoints de 97 por jogo; 103 não o atualizou. A avaliação
mediu erro quadrático dos seis pares de contrastes visuais, com peso na região
onde houve mudança real e divisão pelo erro da previsão sem contraste.

| Jogo | Modelo visual, razão | Molde fixo por ação | Episódios em que o visual venceu o molde | Nulos de ação tão bons | Nulos de histórico tão bons | Portão |
|---|---:|---:|---:|---:|---:|---|
| Breakout | **0,225** | 0,512 | 8/8 | 0/32 | 0/32 | Passou |
| Pong | **0,493** | 0,746 | 8/8 | 0/32 | 0/32 | Passou |

Todos os 16 episódios tiveram contraste não nulo, e em cada um a razão do
modelo ficou abaixo de 1. O ganho médio sobre o molde foi 0,287 em Breakout
e 0,253 em Pong, acima do limiar pré-definido 0,10. Breakout consumiu
10.144 passos ALE na coleta de confirmação; Pong, 10.248. Os hashes do
código congelado, bancos e eventos coincidiram após a execução. Relatórios e
eventos estão em `resultados/{breakout,pong}_confirmacao_efeitos.json` e
`resultados/{breakout,pong}_efeitos_locais_103.{npz,json,jsonl}`.

**Interpretação limitada:** há sinal visual condicionado ao estado e à ação
que transfere entre sementes para prever efeitos de oito passos em dois jogos.
Isto enfraquece a hipótese de que toda falha de retorno longo decorra de
percepção impossível. O experimento usa o mesmo algoritmo treinado
separadamente em cada jogo, não demonstra transferência de conceito entre
jogos, invenção autônoma de representação, planejamento, descoberta científica
ou inteligência humana. O próximo problema é compor efeitos locais, manter
incerteza e prever como decisões alteram consequências distantes. Uma melhora
nessa composição precisa vencer ação constante e controles em episódios
com margem, sem reusar a semente de confirmação 103 para ajuste.

Uma ponte exploratória com `curto_para_longo.py` usou, como oráculo, os
**quadros finais verdadeiros** das quatro intervenções de oito passos na
semente 97 para tentar prever os valores de prefixo/sufixo de 128 passos.
Os bancos de curto e longo prazo coincidiram quadro a quadro antes do teste.
Mesmo com esse privilégio impossível para um agente comum, o mesmo regressor
obteve arrependimento 0,264 em Breakout e 0,317 em Pong, contra 0,209 e
0,306 de uma ação constante escolhida no treino. Recompensa real nos oito
passos deu 0,209 e 0,252. Este resultado não prova que os efeitos curtos
sejam inúteis: testa apenas uma leitura direta com 64 estados e um regressor.
Ele reforça que passar do efeito local ao valor distante requer aprender a
**compor transições e eventos**, não apenas usar o quadro curto como mais um
vetor de entrada.

## Décimo sétimo ensaio: por que a dinâmica local não se encadeia

`COMPOSICAO_TEMPORAL.md` especificou uma coleta pareada a cada oito passos
até 128. `coletar_horizontes.py` reproduziu as histórias de 97 e 103;
**256/256 ramos de oito passos em cada jogo e semente** coincidiram em imagem,
recompensa, terminação e duração com os bancos anteriores. Foram 34.898 e
32.417 passos ALE para Breakout (97/103), 38.610 e 38.168 para Pong.
Os arquivos `resultados/{breakout,pong}_horizontes_{97,103}.{npz,json,jsonl}`
guardam estados, ramos, eventos e hashes.

Treinei em 97 uma transição de oito passos dos dois quadros 21×16 e a
realimentei para cada comando constante. Também treinei, separadamente, uma
regressão **direta** para cada horizonte. A razão de erro do contraste visual
entre ações usa previsão cega como 1; menor é melhor. A semente 103 já fora
inspecionada no teste local, portanto a curva é **diagnóstica**, não nova
confirmação.

| Jogo | Passos | Direta | Composta sem correção | Molde fixo por ação |
|---|---:|---:|---:|---:|
| Breakout | 8 | 0,232 | 0,232 | 0,510 |
| Breakout | 32 | 0,333 | 0,595 | 0,562 |
| Breakout | 64 | 0,399 | 1,542 | 0,607 |
| Breakout | 128 | 0,377 | **10,633** | 0,596 |
| Pong | 8 | 0,493 | 0,493 | 0,746 |
| Pong | 32 | 0,604 | 1,175 | 0,780 |
| Pong | 64 | 0,634 | 3,034 | 0,795 |
| Pong | 128 | 0,666 | **11,714** | 0,812 |

Em Pong, se a cada oito passos o modelo recebesse o **estado real** em vez
do seu próprio estado previsto, o erro local ficaria aproximadamente entre
0,44 e 0,63 nos dezesseis passos. Assim, grande parte da explosão vem da
realimentação de estados produzidos pelo próprio modelo. Em Breakout, esse
erro local com estado real cresce depois das primeiras transições: há também
deslocamento da distribuição de treino. Terminações não explicam tudo:
mesmo excluindo raízes em que algum dos quatro ramos terminou, a composição
sem correção marcou 9,698 em Breakout e 10,899 em Pong aos 128 passos.

`estabilizar_composicao.py` testou exploratoriamente treino adicional com
128 transições intermediárias por comando e duas formas de projeção sobre
estados de 97: vizinho real mais próximo e correção contínua pela metade da
componente fora de um subespaço aprendido de 32 dimensões. O treino
intermediário melhorou a previsão local ao longo da trajetória, mas os
quatro braços originais e os dois de projeção contínua ainda não venceram
a previsão direta em 128 passos nos dois jogos. Encaixe duro no vizinho
conteve o erro numérico (Breakout 0,590; Pong 1,350 com modelo de raiz),
mas destruiu contraste causal em Pong desde oito passos (1,299 contra 0,493
sem encaixe). Correção contínua preservou melhor o curto prazo, mas em 128
passos ainda ficou em 4,231/6,645 com modelo de raiz e 0,659/2,076 com
treino intermediário (Breakout/Pong). Relatórios por episódio e sem terminação
estão em `resultados/{breakout,pong}_avaliacao_composicao.json` e
`_estabilizar_composicao.json`.

O requisito técnico agora é mais específico que "mais memória": um estado
interno precisa permanecer coerente sob iteração **e** preservar assinaturas
de consequências para ações alternativas. Similaridade visual de um único
estado não basta. A próxima hipótese a testar é representar o estado por
previsões de pequenas intervenções, aprender sua atualização sob ações e
comparar com a composição em pixels. Isso é uma hipótese de pesquisa, não
uma solução já demonstrada para planejamento ou AGI.

## Décimo oitavo ensaio: quais previsões curtas conservam o futuro?

`ponte_assinatura.py` testou um caminho sem realimentar pixels: aprender em
97 a relação da assinatura visual de oito passos para os contrastes visuais
de 16–128 passos e avaliar em 103. Para impedir que o segundo estágio veja
assinaturas de treino perfeitas enquanto recebe assinaturas previstas no
teste, as assinaturas usadas para treiná-lo foram previstas fora do próprio
episódio de 97.

| Jogo | Horizonte | Pixels diretos | Assinatura curta real (oráculo) | Assinatura curta prevista |
|---|---:|---:|---:|---:|
| Breakout | 16 | 0,224 | **0,153** | 0,262 |
| Breakout | 128 | 0,377 | **0,322** | 0,449 |
| Pong | 16 | 0,501 | **0,386** | 0,583 |
| Pong | 128 | 0,666 | **0,580** | 0,783 |

Assim, uma assinatura curta **medida** carrega informação futura útil nos
dois jogos, mas a assinatura **estimada** perde esse ganho. Isto identifica
um erro de interface entre previsão local e uso posterior: vencer erro médio
de pixel no curto prazo não assegura preservar os componentes que importam
para efeitos distantes. `relevancia_futura.py` tentou selecionar oito
direções da assinatura de treino pela covariância com contrastes futuros;
a versão prevista ainda perdeu para pixels diretos em todos os horizontes
e jogos. Não há promoção dessa tentativa. Os arquivos
`resultados/{breakout,pong}_{ponte_assinatura,relevancia_futura}.json` contêm
resultados por horizonte; a segunda tentativa guarda também a base
inspecionável em NPZ.

`aliasing_assinatura.py` procurou pares de estados entre os 128 disponíveis
por jogo. Houve **79 pares em Breakout e 108 em Pong** com exatamente o
mesmo contraste visual entre ações em oito passos, mas contraste diferente
em 128; nenhum dos contrastes curtos desses pares era zero. Isto é um
contraexemplo finito à suficiência de **somente o contraste entre ações de
oito passos** nesta resolução. Ao incluir as quatro mudanças visuais
completas (também a dinâmica comum às ações), restou apenas 1 par em
Breakout e 0 em Pong. Ao incluir ainda os dois quadros iniciais, restou 0
em ambos. Igualdade aqui é exata nos quadros reduzidos 21×16; não é uma
afirmação sobre pixels nativos ou sobre todos os mundos. Os pares, hashes e
distâncias estão em `resultados/{breakout,pong}_aliasing_assinatura.json`.

O primeiro processo de coleta foi interrompido após dois episódios devido
a uma descompressão repetida do mesmo banco NPZ no código. Depois de carregar
os arrays uma única vez, `coletar_estado_preditivo.py` completou os quatro
bancos. Nenhum banco parcial entrou na avaliação. A hipótese que sobrevive é
precisa: um estado reutilizável deve carregar tanto a dinâmica comum do mundo
quanto as diferenças causadas por ações, e o treinamento do extrator deve
valorizar a informação que permanece útil em horizontes longos.

## Décimo nono ensaio: compor estados definidos por consequências

Em cada estado intermediário, o novo banco registra o efeito visual das
quatro ações durante oito passos. As 64 histórias, 256 testes iniciais e
2.048 pontos intermediários por jogo e semente coincidiram com os bancos
anteriores; foram 1.889–2.207 estados preditivos válidos por banco. Os
arquivos `resultados/{breakout,pong}_estado_preditivo_{97,103}.{npz,json,jsonl}`
guardam arrays, registros e hashes.

`avaliar_estado_preditivo.py` comprime a assinatura das quatro consequências
em 32 dimensões e aprende atualizações distintas para cada uma das quatro
ações, somente com trajetórias da semente 97. Avalia as trajetórias da
semente 103, já inspecionada anteriormente; é **diagnóstico exploratório**.
O controle direto recebe as mesmas raízes em pixels e é treinado apenas nos
estados válidos de cada ação/horizonte.

| Jogo | Passos | Composição desde estado previsto | Desde estado inicial real | Pixels diretos |
|---|---:|---:|---:|---:|
| Breakout | 8 | 0,148 | 0,137 | **0,134** |
| Breakout | 64 | 0,209 | 0,138 | **0,109** |
| Pong | 8 | 0,336 | **0,276** | 0,327 |
| Pong | 64 | 0,317 | **0,199** | 0,284 |

Menor razão de erro do contraste causal é melhor; a previsão cega vale 1.
O estado por consequências compõe sem explosão e supera a persistência do
estado inicial, mas **não supera a previsão direta** em nenhum dos dois jogos
aos 64 passos. O início com assinatura real melhora a curva, mostrando que
a extração da assinatura a partir dos pixels continua sendo um gargalo.
Isso não estabelece uma representação suficiente para planejamento: as ações
avaliadas até aqui são constantes. O próximo teste usa planos mistos nunca
vistos no treino, com controles de ação permutada e persistência.
Resultados por episódio, hashes e erros com estado real a cada atualização:
`resultados/{breakout,pong}_avaliacao_estado_preditivo.json`.

## Vigésimo ensaio: composição de planos novos, com confirmação congelada

O modelo de transição do ensaio anterior foi treinado somente em trajetórias
de **ação constante** da semente 97. Fixei três planos alternados de oito
blocos antes de medi-los. Na semente 103, já usada em desenvolvimento, o
estado previsto obteve razão de erro do contraste de consequências de
0,381 em Breakout e 0,471 em Pong aos 64 passos; ação ciclicamente permutada
deu 0,585 e 0,950; persistência deu 1,030 e 0,917. O modelo não recebeu os
pixels intermediários reais. Isto motivou o pré-registro
`PRE_REGISTRO_PLANOS_109.md`, com scripts e SHA-256 congelados antes de
coletar a semente 109.

Na semente 109, as 64 raízes e os 256 testes curtos por jogo foram certificados
contra bancos independentes. Os planos produziram 126 estados válidos de 192
possíveis em Breakout e 174 em Pong ao fim dos 64 passos; a máscara remove
estados após terminação igualmente para todos os braços.

| Jogo | Estado previsto | Ações permutadas | Estado persistente | Vitórias pareadas |
|---|---:|---:|---:|---:|
| Breakout | **0,407** | 0,609 | 0,979 | 8/8 contra ambos |
| Pong | **0,454** | 0,956 | 0,975 | 8/8 contra ambos |

As diferenças pareadas médias controle menos proposta foram, para ação
permutada, 0,202 em Breakout (IC bootstrap de 95%: 0,167–0,236) e 0,502 em
Pong (0,458–0,535). Contra persistência foram 0,572 (0,516–0,632) e 0,522
(0,488–0,562). Os quatro limites inferiores ficaram acima de zero. O script
`veredito_planos_mistos.py` verificou os hashes e aplicou o limiar fixo de
0,75, as vitórias em pelo menos seis episódios e os quatro intervalos.
**O portão pré-registrado passou nos dois jogos.** Detalhes e arrays estão em
`resultados/{breakout,pong}_planos_mistos_109.{npz,json}`,
`resultados/{breakout,pong}_avaliacao_planos_mistos_109.json` e
`resultados/veredito_planos_mistos_109.json`.

O resultado estabelece somente que um estado de 32 dimensões definido por
consequências curtas pode ser atualizado ao longo de sequências discretas de
ações inéditas e preservar parte do contraste causal visual em dois jogos
específicos. **Não demonstra** escolha de ações por valor, ganho de recompensa,
invenção autônoma da representação, transferência entre jogos, modelo de
mundo geral ou inteligência humana. A assinatura de quatro sondas, a janela
de oito passos, o posto 32 e o conjunto de ações foram escolhidos por nós.
Nos planos constantes, a previsão direta a partir dos pixels ainda venceu
a composição. O próximo gargalo verificável é deixar o sistema escolher
suas próprias sondas e a granularidade temporal, e exigir melhora de
decisão em mundos não vistos; sem isso, o mecanismo segue uma estrutura
projetada manualmente.

## Vigésimo primeiro ensaio: gerar uma sonda a partir de um resíduo

`sondas_contraexemplo.py` implementa uma sonda com dois corpos inspecionáveis:
uma sequência de ações anônimas e uma projeção dos pixels extraída da diferença
de respostas de um par. O gerador enumera as 20 palavras de uma ou duas ações,
nunca objetos ou conceitos do jogo. Seleciona quatro pares próximos no estado
atual com desfechos visuais diferentes, somente em 97, e produz oito candidatas
por jogo. `testes_sondas.py` exige uma sequência nova de duas ações num controle
finito; também verifica ausência de sinal e orçamento. 64 histórias e 16 replays
das candidatas por jogo passaram. Isso valida o gerador mecânico, não utilidade.

`avaliar_sondas.py` estimou as respostas a partir dos dois quadros iniciais e
comparou a previsão de 128 passos com base, leituras aleatórias, mais dimensões
e respostas embaralhadas. Em Breakout a razão visual foi 0,448 com sondas
contra 0,456 sem elas; o arrependimento de decisão piorou de 0,344 para 0,359.
Em Pong foi 0,793 contra 0,788; o arrependimento permaneceu 0,281, pior que
0,156 da ação constante. Com respostas reais como oráculo, o erro visual de
Pong caiu de 0,719 para 0,617, mas a decisão não superou a ação constante.
Nenhuma promoção. `diagnosticar_extrator_sondas.py` comparou cinco vistas do
mesmo histórico; aumentar resolução/histórico não recuperou ganho geral de
decisão nem vantagem sobre os controles. Todos esses resultados são exploratórios.

## Vigésimo segundo ensaio: perguntas orientadas por erro de decisão

`sondas_decisao.py` usa apenas episódios 0–3 de 97 para selecionar pares de
estados próximos cujos conjuntos de ações ótimas são disjuntos e em que há
erro de preferência previsto fora do episódio. Produziu quatro pares/80
candidatas em Breakout e um par/20 em Pong. Os 20 programas foram observados
nas 64 raízes; a leitura de cada candidata continua sendo extraída do par,
não cadastrada. Os corpos e respostas estão em
`resultados/{breakout,pong}_sondas_decisao_{97,103}.npz` e
`_sondas_decisao_definicoes.json`.

`selecionar_sondas_decisao.py` exige melhora em ambos os episódios 4 e 5;
6 e 7 recebem as escolhas congeladas. Nenhuma leitura isolada passou.
Não interpretar isso como impossibilidade: o braço com estado e resposta
reais já tem erro zero na seleção de Pong, e o resultado posterior depende
também do extrator e do estimador de preferência. O controle positivo
`testes_sondas_decisao.py` verifica aceitação de uma variável útil em dois
episódios e rejeição de ganho em só um deles.

## Vigésimo terceiro ensaio: compor medições e combinar perguntas sem ganho isolado

`extrator_experiencias.py` reaproveita as consequências de estados
intermediários, em vez de só 64 raízes. Para uma palavra `(a,b)`, soma a
mudança observada sob `a` com a mudança observada sob `b` no estado real
alcançado. Todas as 16 palavras nas 64 raízes coincidiram com execução direta:
erro máximo 7,45×10⁻⁹ em Breakout e zero em Pong nos quadros normalizados.
Usa até 512 experiências para estado/sondas simples e 256 por primeira ação
para as compostas, sem misturar episódios de treino e avaliação. O controle
`testes_extrator_experiencias.py` verifica estratificação, separação e soma.
Um defeito de implementação (passar um conjunto Python diretamente a
`np.isin`) foi detectado antes de obter resultado e corrigido; a checagem
de separação faz parte do controle positivo.

Na mesma coleção de perguntas de decisão, a razão média de erro do extrator
contra resposta constante passou de 0,641 para 0,693 em Breakout e de 0,583
para 0,515 em Pong. Reutilizar experiências pode melhorar algumas leituras,
mas não trouxe ganho de decisão nessa seleção. O conjunto previsto de
treino/teste fica em `resultados/{breakout,pong}_predicoes_sondas_experiencias.npz`.

`conjuntos_sondas.py` testa também grupos por programa/testemunha e conjuntos
diversos de 2/4/8 leituras, para não descartar composição porque partes
isoladas não melhoram. O controle XOR em `testes_conjuntos_sondas.py` exige
precisamente esse caso. Foram 200 conjuntos em Breakout e 114 em Pong;
leituras aleatórias e linhas embaralhadas recebem os mesmos conjuntos/seletor.

| Breakout: arrependimento | Base | Sondas por testemunha | Leituras aleatórias | Linhas embaralhadas |
|---|---:|---:|---:|---:|
| Seleção, 97 episódios 4–5 | 0,375 | 0,250 | 0,188 | 0,188 |
| Reservados, 97 episódios 6–7 | 0,563 | 0,563 | 0,563 | 0,563 |
| Semente 103, retrospectiva | 0,406 | 0,422 | 0,406 | 0,406 |

O ganho de seleção **não transferiu**, e os controles também foram escolhidos
por ganho aparente. Nenhuma variável foi promovida. Em Pong, a base no episódio
5 de seleção tem arrependimento zero; exigir melhora estrita nos dois episódios
é impossível. O relatório marca **inconclusivo por portão sem margem**, sem
mudar retroativamente o critério nem anunciar incapacidade do gerador.

`auditar_sondas.py` verifica a utilidade incremental nos dados reservados e
na outra semente contra base e controles, e registra promoção zero em
`resultados/auditoria_sondas.json`. Não foi iniciada confirmação em semente
nova, pois não houve candidato com evidência positiva suficiente de decisão.
A definição atual é uma projeção linear fixa dos pixels: separar instâncias
não foi suficiente para descobrir uma relação útil em situações novas.
O próximo teste deve investigar leituras com posição e relações temporais
variáveis, mantendo ramos e controles. Isso é uma hipótese específica de
expressividade do gerador, não o diagnóstico de uma peça única suficiente
para inteligência humana. Detalhes, custos e comandos: `SONDAS_EXECUTAVEIS.md`.

## Vigésimo quarto ensaio: fragmentos com localização variável

`leituras_variaveis.py` extrai dois fragmentos de cada resíduo de testemunha,
um por sinal, e procura sua correspondência em posições variáveis. As leituras
são forças e deslocamento relativo entre fragmentos, sem declarar identidade
de objetos. O corpo é armazenado e executável. O extrator reaproveita os mesmos
ramos e amostras; rótulos derivados foram confrontados com execução real.
Os controles de translação, deslocamento relativo e vazio passaram.

| 103, decisão em 128 passos | Base | Fragmentos variáveis | Fragmentos aleatórios | Associação embaralhada | Ação constante |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,406250 | 0,265625 | 0,265625 | 0,406250 | 0,437500 |
| Pong | 0,250000 | 0,140625 | 0,156250 | 0,171875 | 0,156250 |

Breakout empatou com controle aleatório: piloto negativo. Pong teve ganho de
apenas 1/64 de unidade sobre a ação constante. O pré-registro congelou 23
fontes/artefatos antes de abrir a semente 113. A confirmação de Pong **falhou**:
fragmentos 0,312500; base/aleatório/embaralhado 0,359375; ação constante
0,203125. A vantagem sobre base ocorreu em um só episódio, com limite inferior
bootstrap zero. Contra ação constante o ganho foi −0,109375, IC95%
[−0,203125; −0,031250]. Foram 48.932 passos ALE, sem alterar o portão.

`diagnosticar_oraculo_leituras.py` mediu respostas reais dos mesmos programas
em 113 depois da falha. Com os mesmos fragmentos congelados e respostas reais,
arrependimento 0,343750, ainda pior que constante 0,203125. A seleção oracular
prévia também deu 0,343750. O extrator tem erro, mas substituí-lo por respostas
reais não bastou para essa leitura/decisão. Custo adicional 26.912 passos,
64 históricos certificados e 20 replays de programa. Nenhuma promoção.

Corpos, protocolo e controles: `LEITURAS_VARIAVEIS.md`,
`CONFIRMACAO_LEITURAS_113.md`, `testes_leituras_variaveis.py`,
`testes_portao_leituras.py`, `testes_confirmacao_leituras.py`. Os relatórios
`pong_confirmacao_leituras_113.json` e
`pong_leituras_variaveis_oraculo_113.json` preservam escolhas e evidências.

## Vigésimo quinto ensaio: ligar transições a crédito local

`valor_composto.py` usa os estados preditivos de 97 para aprender recompensa
e término em oito passos, e compõe até 128. Os incrementos reconstruíram
exatamente os retornos acumulados; terminação absorve os blocos posteriores.
Treinou 1.859 experiências válidas em Breakout, 1.970 em Pong. Não houve nova
intervenção ALE; CPU de avaliação 21,7/23,4 segundos, respectivamente.

Com o estado real, o erro local de recompensa ficou entre 0,133 e 0,299 da
média por ação em Breakout, e entre 0,196 e 0,223 em Pong. Isso não garante
decisão: a composição prevista perdeu para ação constante. Aos 128 passos,
Breakout 0,562500 contra 0,437500; Pong 0,312500 contra 0,156250. O controle
embaralhado chegou a vencer o proposto em Breakout, impedindo promoção.

`diagnosticar_credito.py` substitui a trajetória prevista pela real, sem
permitir seu uso pelo agente. Compara estado comprimido, assinatura inteira
e pixels. Um controle positivo soma recompensas verdadeiras e exige reconstruir
exatamente o retorno; todos os horizontes tiveram arrependimento zero nesse
controle. Nenhuma coleta adicional.

| 103, 64 passos | Trajetória prevista + crédito aprendido | Trajetória real + crédito do estado 32 | Trajetória real + assinatura completa | Trajetória real + pixels | Ação constante |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,250000 | 0,000000 | 0,000000 | 0,109375 | 0,218750 |
| Pong | 0,265625 | 0,062500 | 0,062500 | 0,156250 | 0,156250 |

Nesse banco, há perda grande no caminho de imaginar estados futuros. A
compressão 32 não explica sozinha a perda de decisão, pois a assinatura completa
não melhora esse oráculo. Isso não prova suficiência da representação para
qualquer mundo, nem dá ao agente acesso legítimo às trajetórias reais.
Detalhes, eventos raros e resultados por episódio nos JSON de crédito.

## Vigésimo sexto ensaio: preservar distribuições de futuros

Hipótese registrada em `VALOR_COMPOSTO.md` antes de executar: aplicar leitura
não linear ao estado médio pode apagar futuros úteis. `futuros_distribuidos.py`
conserva estados observados, sucessores, recompensas e términos da mesma
experiência. O kernel distribui peso entre oito vizinhos, com largura local,
sem escolher hiperparâmetro no teste. Um cálculo de Bellman compõe valores
nessa distribuição; compara com colapsar ao estado médio usando o mesmo kernel.
Não há cadastro de objetos ou regra do jogo. A composição empírica e o cálculo
de Bellman não são apresentados como uma invenção científica nova.

| 103, 128 passos | Distribuído | Colapso à média | Sucessor embaralhado | Recompensa embaralhada | Ações trocadas | Ação constante |
|---|---:|---:|---:|---:|---:|---:|
| Breakout | 0,140625 | 0,437500 | 0,421875 | 0,421875 | 0,828125 | 0,437500 |
| Pong | 0,171875 | 0,296875 | 0,265625 | 0,234375 | 0,421875 | 0,156250 |

Breakout deu sinal de desenvolvimento; Pong continuou perdendo para ação
constante. Os controles positivos verificaram soma, pesos normalizados,
absorção e um caso onde colapsar possibilidades destrói a previsão de valor.
Custo de avaliação 15,7/14,9 segundos de CPU, sem nova coleta.

## Vigésimo sétimo ensaio: confirmação congelada de decisão em Breakout

`CONFIRMACAO_FUTUROS_127.md` foi escrito e 23 fontes/artefatos congelados antes
de expor Breakout 127. Treino somente 97, mesmos oito vizinhos, sem ajuste na
confirmação. A política recebe só dois quadros iniciais; futuros e retornos
de teste são acessados pelo avaliador depois de concluir as escolhas.
Os quatro programas de ação constante são os mesmos de antes.

| 127, horizonte primário 128 | Distribuído | Estado médio | Sucessor embaralhado | Recompensa embaralhada | Ações trocadas | Ação constante | Regressão direta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arrependimento | 0,234375 | 0,531250 | 0,531250 | 0,500000 | 0,671875 | 0,468750 | 0,421875 |

O portão **passou**: cada controle teve vantagem média, limite inferior
bootstrap de 95% acima de zero e pelo menos dois episódios melhores. Contra
ação constante foram cinco episódios melhores, três empatados e nenhum pior;
ganho 0,234375, IC95% [0,093750; 0,375000]. Contra regressão direta, ganho
0,187500, IC95% [0,078125; 0,312500]. Todos os seis controles entram na
condição de aceite. O controle positivo do portão rejeita empate com placebo,
ganho em um episódio apenas e empate com regressão direta.

Resultado secundário de 64 passos: distribuído 0,171875, constante 0,265625,
regressão direta 0,312500. O veredito deriva somente dos 128 passos.
Coleta 42.759 passos ALE, 64 históricos reconstruídos nas duas coletas e
256 ramos curtos certificados entre elas; replay repetido também certificado.
Confirmação 29,9 segundos de parede, 31,5 de CPU. O custo prévio de treino
nos bancos de 97 não é zero e não está incluído nesses 42.759 passos de teste.

O alcance confirmado é **redução de erro ao escolher entre quatro programas
constantes em episódios novos do mesmo jogo**. Não houve jogo reservado com
regras desconhecidas, transferência de representação entre domínios, busca
livre por planos, aprendizado humano por minuto nem descoberta científica.
Pong segue sem ganho sobre constante no desenvolvimento. A máquina ainda
não inventa autonomamente todas as variáveis de um mundo novo. O objetivo
de mente de nível humano permanece não demonstrado e ativo.

Código: `confirmar_futuros_distribuidos.py`. Manifesto, previsões reais por
raiz, escolhas, controles, custos e veredito:
`resultados/breakout_confirmacao_futuros_127_manifesto.json`,
`breakout_confirmacao_futuros_127.json` e `.jsonl`. Não sobrescrever a
confirmação nem alterar suas fontes congeladas para interpretar o ganho.

## Vigésimo oitavo ensaio: propor sequências sem lista de planos

`planejador_distribuido.py` generaliza o cálculo confirmado para palavras de
ações diferentes. A interface admite qualquer número de ações e dimensão
numérica de estado; os adaptadores desta rodada continuam usando os estados
visuais de 32 dimensões dos dois jogos. O corpo não inventa essas dimensões
nem dispensa treino. A busca guarda oito prefixos, expande todas as ações,
e compõe 16 blocos de oito passos. É uma busca aproximada, com poda pelo
retorno previsto acumulado, não enumeração completa de 4¹⁶ planos.

O contrato positivo encontrou a troca de ação necessária e coincidiu com
enumeração exata de todas as palavras de quatro ações num mundo de duas
ações. Nas 64 raízes de cada jogo, quatro modelos (normal e três controles)
tiveram concordância entre valor da busca e composição da palavra: erro
máximo 1,33×10⁻¹⁵. Isso certifica implementação, não valor do mundo previsto.

`PILOTO_PLANOS_PROPOSTOS.md` foi registrado antes da geração. Todos os planos
e hashes foram armazenados antes de observar seus resultados. O treino foi
somente 97; 103 já era desenvolvimento. Os cinco braços receberam a mesma
busca e largura. As referências foram ação fixa escolhida em 97, escolha
entre constantes pelo corpo confirmado e regressão direta.

| 103, retorno real em 128 passos, maior é melhor | Busca distribuída | Busca no estado médio | Sucessor embaralhado | Recompensa embaralhada | Ações trocadas | Constante | Distribuição entre constantes | Regressão direta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breakout | 0,937500 | 0,250000 | 0,375000 | 1,156250 | 0,406250 | 0,656250 | 0,953125 | 0,687500 |
| Pong | −3,203125 | −3,000000 | −3,296875 | −3,171875 | −3,171875 | −2,984375 | −3,000000 | −3,078125 |

**Ambos os portões falharam.** Em Breakout, a busca venceu a ação fixa em
seis episódios sem piorar nos demais, mas perdeu da recompensa embaralhada
e não superou o corpo anterior. Em Pong, também perdeu de ação constante.
Não abrir confirmação em semente nova nem declarar salto geral.

As buscas normais produziram 55/64 planos não constantes em Breakout e
64/64 em Pong. Todos os 64 históricos e 256 ramos curtos por jogo foram
conferidos; houve oito replays completos de plano por jogo. Coleta nova
39.658/48.787 passos ALE. Planos idênticos na mesma raiz compartilham uma
medida física, contando passos uma vez; os eventos de políticas referem-se
a essa mesma medida. Referências constantes usam ramos previamente medidos,
cujo custo não está incluído na nova coleta. Geração: 15,0/10,0 segundos de
CPU; tempos de execução separados estão nos JSON. Nenhum arquivo original
ou corpo de confirmação foi alterado.

## Vigésimo nono ensaio: repertório, ordenação e oráculo de trajetória

`diagnosticar_planos_propostos.py` recalculou a previsão normal para todos
os oito planos/políticas já medidos. Um seletor que visse respostas reais
teria retorno 1,515625 em Breakout, contra 0,937500 escolhido; em Pong,
−2,640625 contra −3,203125. Há alternativa estritamente melhor em 26/64 e
21/64 raízes, respectivamente. Isso demonstra um limite de seleção nesse
repertório, não entrega ao agente o seletor oracular. Reordenar pelo mesmo
modelo normal dá 1,125000/−3,046875; o modelo também erra a comparação.

O modelo normal previa vantagem média de 0,432456 sobre os planos gerados
pelo controle de recompensa de Breakout; a vantagem real foi −0,218750.
O viés de valor do plano escolhido foi +0,480301 em Breakout e +0,682979
em Pong. A busca favoreceu resultados previstos que não se realizaram.

`ORACULO_PLANOS_PROPOSTOS.md` registra o diagnóstico posterior, sem resgatar
o piloto: medir os primeiros quatro checkpoints de cada um dos oito episódios,
dois planos já congelados por raiz (normal e recompensa embaralhada). O
coletor reproduziu todos os 64 históricos por jogo, conferiu 128 sondas
iniciais e 64 retornos de programa. As assinaturas intermediárias foram
medidas por intervenções reais; não são observações legítimas na raiz.

| Plano normal, 32 raízes: MSE do retorno | Previsão inicial congelada | Início real, composição normal | Trajetória real, crédito de oito vizinhos | Trajetória real, crédito KernelRidge | Recompensa verdadeira |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,865966 | 0,851237 | 0,416994 | 0,118720 | 0 |
| Pong | 1,390115 | 1,218549 | 1,081210 | 0,348575 | 0 |

Trocar somente o início pelo estado real não recuperou a previsão. A
trajetória real e o estimador de crédito já existente reduziram muito o
erro, mas ainda houve erro residual. No plano do controle, KernelRidge
teve MSE 0,230843 em Breakout e 0,160254 em Pong, contra 0,391325/0,317244
dos oito vizinhos. Portanto a falha não é só perceber a raiz, nem só a
trajetória imaginada: a geometria/leitura de crédito usada pelo planejador
também perde informação relevante nesses estados.

Coleta adicional 32.000/50.176 passos ALE, parede 21,2/26,5 segundos. Todos
os retornos coincidiram com o piloto congelado; recompensas reais somadas
tiveram MSE zero como controle positivo. Os corpos e resultados anteriores
ficaram intactos. Dados, custos e eventos em
`resultados/{jogo}_oraculo_planos_propostos_103.*`. Nenhuma promoção.

## Trigésimo ensaio: a poda também perde planos no próprio modelo

`auditar_poda_planos.py` verificou se outras palavras já disponíveis tinham
valor maior no **mesmo modelo normal**, antes de discutir sua verdade. Em
12/64 raízes de Breakout e 34/64 de Pong havia plano conhecido com previsão
melhor que a busca de largura oito; perdas médias previstas 0,132138 e
0,039726. Logo, parte da falha também está na seleção de prefixos.

Uma busca retrospectiva de largura 64 elevou a previsão média de Breakout
de 1,417801 para 1,795661, melhorando em 56 raízes; em Pong, de −2,520146
para −2,463195, melhorando em 55 e piorando em três. **Esses planos não
foram executados.** Maior valor no modelo pode explorar ainda mais o erro
já medido. Não apresentar isso como ganho real, não reajustar o piloto.

O próximo mecanismo precisa enfrentar conjuntamente uma leitura de
consequências relevante e calibrada, composição de trajetórias fora dos
trajetos de treino e retenção de prefixos com ganho posterior. Os dados
não provam uma peça única suficiente para mente humana, nem identificam
com perfeição todos os motivos. O progresso desta rodada é corpo genérico
executável de planos, medidas reais e localização de perdas específicas.
O objetivo de inteligência humana e descoberta de fronteira permanece
não alcançado e ativo. A confirmação anterior de Breakout entre quatro
constantes continua válida no seu escopo; não se estendeu automaticamente
à busca livre por sequências.

## Trigésimo primeiro ensaio: crédito aprendido e coordenadas de consequências

`coordenadas_consequencias.py` aprende funções executáveis de recompensa e
continuação por ação, somente em 97. Os corpos completos de kernel, parâmetros,
suportes e coeficientes ficam em NPZ com definição JSON; salvar/recarregar
reproduz a leitura. Não há objeto, tipo de entidade ou regra de jogo.
O corpo é uma regressão numérica conhecida; não reivindicamos invenção
científica da técnica nem geração universal de representações.

O protocolo `PILOTO_COORDENADAS_CONSEQUENCIAS.md` foi escrito antes de gerar
planos. Todos os planos e hashes foram armazenados antes de executar. A
intervenção combina a representação atual com respostas previstas das ações,
normalizando por distâncias de treino com peso igual. Compara geometria atual,
coordenadas reais, coordenadas de funções treinadas com respostas embaralhadas
e projeção aleatória de igual dimensão. Todos usam o mesmo crédito aprendido,
busca de oito prefixos e 16 blocos. Término continua associado ao sucessor real.

| 103, retorno real em 128 passos | Crédito aprendido, geometria atual | Coordenadas reais | Coordenadas embaralhadas | Coordenadas aleatórias |
|---|---:|---:|---:|---:|
| Breakout | 1,328125 | 1,281250 | 1,328125 | 1,453125 |
| Pong | −3,218750 | −3,078125 | −3,062500 | −3,062500 |

**Ambos os portões falharam.** A geometria aleatória foi melhor em Breakout;
as duas geometrias de controle foram ligeiramente melhores em Pong. O
crédito aprendido na geometria atual elevou o retorno de Breakout em relação
à busca antiga (0,937500), mas não transferiu ganho para Pong, nem demonstra
que a nova coordenada seja útil. Esse braço era comparação, não um novo
critério escolhido depois para resgatar o piloto.

A previsão de retorno continua ruim: MSE das coordenadas reais 1,298905 em
Breakout e 1,158090 em Pong. Erro menor em alguns braços não implica melhor
decisão. Planos não constantes: 59/64 e 49/64 no braço primário. Os controles
receberam a mesma busca. Não abrir confirmação nova nem promover coordenadas.

Coleta nova 31.739/40.644 passos ALE, 64 históricos e 256 ramos curtos
conferidos por jogo, oito replays completos por jogo. Geração 27,8/26,0
segundos de CPU; execução 16,8/16,0. Eventos distinguem medida física nova
de reuso de plano idêntico, com custo físico zero no reuso. Fontes/treino,
parâmetros e propostas têm hashes antes de executar; confirmações anteriores
permaneceram intactas. Relatórios e corpos em
`resultados/{jogo}_coordenadas_consequencias_103*`.

## Trigésimo segundo ensaio: existe futuro adequado no suporte?

`diagnosticar_transporte.py` usa as trajetórias reais já coletadas dos planos
anteriores, somente para diagnóstico. Foram 579 transições mistas válidas
em Breakout e 1.024 em Pong. Filtra transições sobreviventes para comparar
estados numéricos; não avalia término nem retorno distante nessa medida.

| Razão de erro sobre persistência, por ação | Vizinhos de origem | Geometria de consequências | Kernel de atualização média | Oráculo: sucessor mais próximo do real |
|---|---:|---:|---:|---:|
| Breakout 0 | 0,589 | 0,591 | 0,535 | 0,203 |
| Breakout 1 | 0,594 | 0,674 | 0,606 | 0,166 |
| Breakout 2 | 0,228 | 0,257 | 0,470 | 0,024 |
| Breakout 3 | 0,659 | 0,616 | 0,549 | 0,053 |
| Pong 0 | 0,129 | 0,163 | 0,095 | 0,068 |
| Pong 1 | 0,139 | 0,162 | 0,111 | 0,079 |
| Pong 2 | 0,083 | 0,111 | 0,088 | 0,023 |
| Pong 3 | 0,124 | 0,165 | 0,121 | 0,049 |

Escolher no suporte usando o próximo estado verdadeiro tem muito menos
erro, especialmente em Breakout. Isso mostra que o banco contém estados
próximos dos resultados reais desses casos. **O oráculo não é uma previsão:**
ele olha o futuro e não torna a escolha utilizável. A leitura de consequência
não melhorou consistentemente a correspondência de transições. Sem nova
coleta, sem ajustar modelos em 103 e sem promoção.

## Trigésimo terceiro ensaio: aplicar mudança em vez de copiar estado

`operadores_locais.py` extrai um corpo `x -> x + delta` de cada experiência,
com definição completa e origem. Consulta mantém alternativas de mudança
inteiras; a média é um método separado. O contrato verifica reprodução da
experiência e aplicação a contexto novo num caso de translação. A linguagem
de soma é fornecida pelo programador; não é uma transformação científica
inédita nem uma abstração universal inventada pela máquina.

`avaliar_operadores_locais.py` compara a atualização contextual com copiar
sucessores da galeria, mudança fixa, persistência e associação de mudanças
embaralhada. Usa exatamente as mesmas transições mistas reais do diagnóstico
anterior, sem novas intervenções e com treino apenas97.

| MSE local, menor é melhor | Mudança aplicada ao contexto | Sucessor da galeria | Mudança embaralhada | Mudança fixa | Persistência |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,000909 | 0,000708 | 0,001496 | 0,001355 | 0,001369 |
| Pong | 0,001218 | 0,000901 | 0,008840 | 0,007479 | 0,007486 |

A nova atualização perdeu da galeria **nos oito episódios de ambos os jogos**,
embora vença mudança fixa, persistência e embaralhamento. Não fazer coleta
longa para promover um corpo que já perdeu nesse diagnóstico; isso não prova
que qualquer transformação contextual falhe. Os corpos de translação testados
não corrigiram a previsão em estados novos.

Nesta rodada, funções de consequência tornaram-se exportáveis/reexecutáveis,
mas a geometria baseada nelas não trouxe ganho contra controles. O suporte
contém futuros aproximados que a correspondência atual não seleciona bem;
trocar cópia por translação local também falhou. Há perdas mensuráveis em
composição e crédito, sem evidência de uma peça única ou garantia de AGI.
O objetivo de mente humana permanece não demonstrado e ativo.

## Trigésimo quarto ensaio: probabilidades de desfechos conjuntos

`desfechos_aprendidos.py` extrai categorias diretamente dos sucessores de
treino97. Mantém recompensa e continuação junto de cada próximo estado real.
Divide apenas pelos valores observados dessas duas grandezas; escolhe até16
sucessores por cobertura em cada grupo. Um kernel aprende a probabilidade da
categoria dado o estado anterior. Não há entidade, regra de jogo ou objeto
fornecido, mas kernel, cobertura e linguagem de composição são escolhas do
programador. Não reivindicar representação humana nem descoberta inédita.

O protocolo `PILOTO_DESFECHOS_APRENDIDOS.md` foi escrito antes da geração.
Planos, parâmetros completos e hashes foram congelados antes de medir qualquer
plano novo. Busca igual16blocos/largura8 nos três braços: condicionado aprendido,
associação origem/categoria embaralhada e frequência independente do estado.
O teste103 é desenvolvimento extensamente visto, sem confirmação nova.

| Retorno real128passos, maior é melhor | Condicional aprendido | Associação embaralhada | Frequência sem contexto |
|---|---:|---:|---:|
| Breakout | 0,812500 | 1,109375 | 0,671875 |
| Pong | −3,093750 | −3,203125 | −2,984375 |

**Ambos os portões falharam.** O primário perde do embaralhado em Breakout
e da frequência em Pong; também não vence todas as versões anteriores. Não
promover nem escolher um braço de comparação como sucesso depois do resultado.
Erro quadrático das previsões longas do primário:8,222621/5,434014; planos
não constantes33/64 e14/64. A nova composição é especialmente otimista.

Contratos passaram: eventos mantidos juntos, probabilidades normalizadas,
corpo salvo/recarregado, composição contra enumeração exata num universo
pequeno e absorção após término. Nos64estados reais de cada jogo, valor da
busca e composição independente das palavras concordaram até5,33e−15;
exportação reproduziu as probabilidades sem diferença observada. Isso
verifica implementação numérica, sem garantir modelo correto do ambiente.

Coleta nova:28.863passos ALE em Breakout e29.230em Pong;64históricos,
256ramoscurtos e8replays completos certificados por jogo.192eventos por jogo,
reuso explícito de medidas idênticas com custo físico zero. Geração28,75/25,45
segundos CPU; execução15,27/12,64. Custos das referências anteriores registrados
separadamente. `auditar_desfechos_aprendidos.py` reconciliou eventos, custos,
referências, hashes e43condições do portão por jogo. Ambas as confirmações
anteriores permanecem intactas, sem mudanças no projeto original.

## Trigésimo quinto diagnóstico: local versus composto

`diagnosticar_desfechos_aprendidos.py` reaproveita623transições válidas de
Breakout e1.024de Pong das trajetórias reais mistas medidas anteriormente.
Nenhuma interação nova, nenhum ajuste em103. Em todas as ações nos dois
jogos, a previsão local de recompensa do condicionado tem menos erro que
frequência/associação embaralhada. Isso mede informação local utilizável,
nunca qualidade de decisão nem invenção de conceitos.

| MSE de retorno nos32planos antigos `distributed_search` | Crédito nas trajetórias reais | Composição desde estado inicial real |
|---|---:|---:|
| Breakout | 0,106245 | 0,552132 |
| Pong | 0,376686 | 2,862954 |

O erro cresce quando o modelo passa a usar os próprios sucessores. Reduzir
o banco a categorias também removeu estados adequados: em Pong, o erro do
sucessor mais próximo do futuro real no suporte de categorias é3,58a10,76
vezes o do suporte completo, dependendo da ação. Essa escolha olha o futuro
e é só diagnóstico de cobertura. Nenhum evento de recompensa/término dos
casos medidos ficou ausente; a perda detectada é também nos detalhes de estado.
Recompensas locais boas não asseguram transições que preservem decisões futuras.

## Trigésimo sexto diagnóstico: corrigir só a leitura inicial

`diagnosticar_raiz_desfechos.py` conserva exatamente os corpos, planos e
retornos do ensaio34. Substitui apenas a leitura inferida dos pixels por
estado inicial medido por quatro intervenções já presentes no banco103.
Esse estado é oracular para a política original; não apresentar como agente.

| Primário, MSE do mesmo retorno longo | Leitura inicial original | Estado inicial real |
|---|---:|---:|
| Breakout | 8,222621 | 9,093212 |
| Pong | 5,434014 | 5,334533 |

Em Breakout, os planos escolhidos valeram0,812500, enquanto o modelo com
estado inicial real previu2,692431. Em Pong, valeram−3,093750 e a previsão
foi−1,039330. Já o primeiro bloco com esse mesmo estado tem MSE0,024699 e
0,041905. Melhorar só a leitura inicial não recuperou a previsão longa desta
proposta. O otimismo nos planos escolhidos é consistente com a busca priorizar
erros de previsão; os dados não isolam uma causa universal dessa discrepância.

Não há medida de distância até inteligência humana, prova de uma peça única
restante ou demonstração de descoberta de fronteira. A rodada implementou
corpos aprendidos e rejeitou sua utilidade pelo teste fixo; localizou perdas
na compressão/composição que persistem mesmo com leitura inicial real. O
próximo diagnóstico deve separar informação perdida do estado de erro do
modelo de transição, antes de escalar datasets ou repetir busca longa.

## Trigésimo sétimo ensaio: memória extraída dos erros em transições mistas

`coletar_transicoes_mistas.py` coletou32raízes, duas sequências aleatórias de
16blocos por raiz, nos dois jogos e sementes97/103. Os programas foram escritos
antes de medir resultados e são idênticos entre jogos/sementes. Guarda16quadros
passados por estado, quatro sondas custeadas, recompensa e término. Cada banco
conferiu64raízes da caminhada,128sondas iniciais e8replays completos. Sondas
não entram na história da trajetória. Agora o treino contém trocas de ação;
os resultados anteriores treinavam essas transições sobre programas constantes.

Uma falha do instrumento parou as primeiras tentativas97, antes de banco válido:
a tela da última sonda persistia após restaurar a dinâmica do simulador. O teste
separado confirmou a diferença em160ações de Pong. O replay passou a comparar
com o quadro real já guardado na história da ação principal. Preservados código
e planos da primeira tentativa; mesma sequência, sem afrouxar tolerância ou
portão. Detalhes em `CORRECAO_COLETA_MISTA.md`. Custos das tentativas interrompidas
são parcialmente desconhecidos, portanto não alegar custo agregado exato de
todo o trabalho incluindo essas tentativas.

`memoria_residual.py` gera32variáveis numéricas da história: correlaciona quadros
passados com resíduos de próximo estado/recompensa/término previstos fora do
ajuste, separando episódios97 em duas metades. Extrai direções por SVD e salva
média, matriz, normalização e todos os parâmetros da dinâmica. Não fornece
objetos/regras de cada jogo. Não reivindicar novidade científica: projeção
linear, correlação e kernel foram escolhidos pelo programador.

Seis braços têm64dimensões e exatamente as mesmas observações: estado32+zeros;
resíduos com dois quadros; resíduos com16quadros (primário); resíduos embaralhados;
ordem passada embaralhada preservando os dois últimos quadros; PCA sem alvo.
Corpos foram congelados antes da avaliação103. A assinatura inicial e a história
inicial são medidas reais, oraculares para percepção. Rollout usa somente seus
próprios estados previstos. Esta é avaliação diagnóstica, não agente promovível.

| Erro local de recompensa, menor é melhor | Estado sozinho | Memória residual16quadros | Resíduos embaralhados |
|---|---:|---:|---:|
| Breakout | 0,016453 | 0,014167 | 0,014670 |
| Pong | 0,042465 | 0,039521 | 0,041274 |

| MSE do retorno128passos | Estado sozinho | Memória residual16quadros | PCA16quadros |
|---|---:|---:|---:|
| Breakout | 0,666508 | 0,668247 | 0,629523 |
| Pong | 1,823209 | 0,863397 | 0,850754 |

| Arrependimento da escolha entre duas sequências, menor é melhor | Estado sozinho | Dois quadros | Memória residual16quadros | Resíduos embaralhados |
|---|---:|---:|---:|---:|
| Breakout | 0,250000 | 0,406250 | 0,312500 | 0,312500 |
| Pong | 0,218750 | 0,312500 | 0,375000 | 0,187500 |

**Os dois portões diagnósticos falharam.** Em Pong, o MSE do retorno primário
caiu bastante, mas o contraste entre sequências não venceu os resíduos
embaralhados (1,845814 contra1,789679), e a escolha piorou. Em Breakout, o
contraste piorou contra estado sozinho (1,154924 contra1,080285). Não rebatizar
melhora de média/local como memória útil para decisão nem promover o braço PCA
depois do resultado. Não abrir confirmação nova.

Treino válido:861transições Breakout (191/209/234/227 por ação),1.024Pong
(228/256/278/262); avaliação696/1.024. Cada braço tem as mesmas contagens.
Custos exatos dos quatro bancos válidos:42.664/35.415passos ALE de Breakout97/103,
51.200/51.200de Pong97/103, total180.479. Coletas válidas26,1/21,8/26,1/25,2
segundos de parede; ajuste dos seis braços25,4/26,5segundos de parede e136,9/147,5
CPU. Avaliação não fez ações novas. Até dois processos; nenhum processo deixado
rodando após os resultados. Tentativas interrompidas e teste de instrumento
são custos adicionais separados, conforme documentação da correção.

Contratos passaram: passado não contaminado por permutação, dois últimos quadros
preservados no controle, isolamento de episódios no resíduo, variável gerada
que distingue história antiga num caso pequeno, corpo salvo/reexecutado e
absorção após término. O caso pequeno verifica implementação; não prova que
variáveis humanas tenham sido inventadas nem que memória resolveu os jogos.

`auditar_memoria_residual.py` conferiu416eventos de coleta,3.605sobreposições
de história e384eventos de avaliação. Reexecutou os12corpos salvos com os
mesmos estados iniciais e reproduziu as previsões sem diferença observada.
Recalculou23condições por jogo e preservou ambos os vereditos negativos;
planos/código das tentativas interrompidas e confirmações anteriores intactos.
A leitura dos arquivos comprimidos foi depois armazenada uma única vez em
memória para tornar essa auditoria mais rápida, mantendo os mesmos controles.

## Trigésimo oitavo diagnóstico: o critério deve preservar diferenças

A rodada mostra que melhorar recompensa local e retorno médio pode coexistir
com piorar escolhas. Em Pong, a história tem informação extraível pelo modelo,
mas a composição não preserva suficientemente **qual sequência é melhor**.
Como não houve ganho no portão e os dados103 já foram vistos, não concluir
suficiência do estado, irrelevância de memória ou único gargalo da arquitetura.

A próxima intervenção deve partir de pares de consequências que o modelo
confunde e medir se variáveis aprendidas conservam suas diferenças sob novas
sequências. Treinar só para reduzir erro médio do vetor/retorno não satisfez
essa exigência aqui. Antes de aumentar dataset ou busca, separar contraste de
ação, composição e percepção: os dois primeiros continuam falhando mesmo
quando a percepção inicial é substituída por medidas reais. Nenhum desses
ensaios verifica pesquisa de fronteira, transferência geral ou nível humano.

## Trigésimo nono ensaio: comparar programas sem acumular estados médios

`contrastes_programas.py` aprende uma função que recebe estado e duas palavras
de ações e estima a diferença entre seus retornos. Não reconstrói toda a
trajetória para esse cálculo. Diferenças entre características de programas
impõem comparação consigo mesmo zero, inversão de sinal ao trocar os programas
e aditividade das diferenças. Essas propriedades não garantem sinal correto.
Cada corpo salva suportes, palavras codificadas, coeficientes, escalas, alvo
e linguagem. Alfabeto/horizonte são parâmetros; quatro ações/16blocos é a
configuração da bancada. Não reivindicar novo método científico nem gerador
universal de representações: kernel e linguagem foram fornecidos por nós.

O treino97 fornece512pares de prefixos/1.024retornos dos32pares de trajetórias
reais anteriores. Prefixos são correlacionados; não contá-los como512experiências
independentes. O condicionante64 vem do corpo de memória residual97 congelado,
com assinatura inicial e história reais. Continua sendo diagnóstico oracular.
Cinco braços usam o mesmo orçamento de dados e sistema linear1.024×1.024:
contraste condicional (primário), alvos embaralhados por comprimento/identidade,
estados embaralhados entre raízes mantendo prefixos juntos, somente palavras
e regressão de retornos absolutos. Regularização0,05, sem ajuste em103.

`CONTRASTES_PROGRAMAS.md` precedeu ajuste/coleta. Corpos/hashes precederam
as propostas físicas de palavras reservadas. RNG9811 gerou dois programas
de16blocos nas mesmas32raízes103; nenhum dos64programas completos de avaliação
coincide com um programa de treino. Mundo103 é conhecido/desenvolvimento,
portanto isso não é transferência para mundo novo nem confirmação independente.
Também reexecutamos os seis modelos de memória antigos nas palavras novas.

| Arrependimento128passos, menor é melhor | Contraste condicional | Retornos absolutos | Alvos embaralhados | Estados embaralhados | Só palavras | Escolha constante |
|---|---:|---:|---:|---:|---:|---:|
| Breakout | 0,187500 | 0,281250 | 0,312500 | 0,156250 | 0,125000 | 0,250000 |
| Pong | 0,250000 | 0,343750 | 0,406250 | 0,312500 | 0,312500 | 0,250000 |

**Ambos os portões falharam.** Há melhora média contra regressão absoluta nos
dois jogos, mas o ganho0,093750 tem IC95%[0;0,218750] em Breakout e
[−0,125000;0,312500] em Pong. Não está confirmado. Em Breakout, o primário
perde do estado embaralhado e do modelo que ignora estado. Em Pong, vantagem
contra ambos vem de um único episódio; empata com escolha constante e com
memória residual antiga. Nada foi promovido nem houve confirmação nova.

MSE de diferença de retorno:0,627991/1,306108 no primário,0,648290/1,522671
na regressão absoluta e0,778762/1,617938 nos alvos embaralhados. Erro menor
novamente não assegurou benefício de decisão que pudesse ser atribuído à
leitura do estado. A informação marginal das palavras precisa continuar como
controle, para não chamar viés de programa de compreensão da situação.

Nova coleta15.006passos ALE Breakout e18.432Pong, total33.438, com64raízes,
128sondas iniciais e8replays certificados por jogo. Coleta10,0/9,4segundos
de parede; ajuste83,1/82,0segundos CPU dos cinco braços. Cada braço registra
CPU/parede e tamanho do sistema; não alegar quantidade idêntica de operações.
A avaliação não fez interações novas. Até dois processos, todos concluídos.

Na primeira gravação dos resultados, booleanos NumPy impediram o JSON.
O avaliador congelado e seus hashes foram preservados. Uma entrada separada,
`avaliar_contrastes_serializacao.py`, converte somente escalares NumPy a
escalares Python no namespace de serialização do avaliador. Não alterou dados,
modelos, portões ou consultas; não repetiu coleta nem ajuste. Valores não
finitos e tipos desconhecidos continuam rejeitados. Registro próprio liga
hash do adaptador, fonte congelada e relatório sem mudanças.

Contratos passaram: identidade, antissimetria, aditividade, contexto numérico,
exportação/reexecução e diagnóstico de par determinístico contraditório.
`auditar_contrastes_programas.py` reconciliou208eventos físicos,2.304eventos
de avaliação, corpos com1.024linhas, marginais dos nulos, alvos reais, palavras
novas e37condições por jogo. Reexecução dos corpos reproduziu as diferenças
sem erro observado. Confirmações anteriores e projeto original preservados.

## Quadragésima leitura: condicionar a previsão não basta

Esta intervenção respondeu uma pergunta útil: evitar acumulação de estados
médios e aprender diretamente diferenças pode melhorar alguns resultados,
mas ainda não há vantagem robusta da situação percebida contra ignorá-la.
Não atribuir isso a um único gap de AGI: há poucos pares, representação
comprimida, linguagem/kernel limitados e mundos já vistos.

Uma próxima intervenção deve testar se a própria **leitura da observação**
pode ser construída em função da comparação pedida, usando experiências de
treino, em vez de acrescentar uma história fixa e esperar que uma cabeça de
valor aproveite. Ainda precisa vencer condicionamento embaralhado e informação
marginal do programa, em palavras reservadas e mundos diferentes. Nenhuma
dessas ideias será incorporada apenas por ser plausível ou por vencer o
controle mais fraco. Mente humana e descobertas de fronteira seguem não
demonstradas; a função de comparação é inspecionável, não uma prova de AGI.

## Quadragésimo primeiro ensaio: construir a leitura para o programa

Implementado `leituras_por_programa.py`: para cada palavra de ações, construir
uma função sensorial completa de uma história visual. Salvar média, escala,
viés, pesos lineares, templates observados e pesos de projeções quadráticas.
A previsão executa essa função; exportação reproduz seu resultado. A linguagem
e a família de funções são fornecidas por nós. Não são invenção irrestrita de
conceitos, nem método inédito demonstrado. A equivalência com características
fixas foi conferida explicitamente; não esconder essa limitação sob "atenção".

O primário recebe16observações visuais passadas e palavras, sem assinatura
obtida por sondas, estado oculto ou retorno reservado. Observações são cinza,
reduzidas por médias10×10 a21×16, normalizadas por255; coletaALE tem frame_skip4.
Não testar isso como se fosse toda a informação de imagens RGB originais.
Treino97:32raízes,512pares correlacionados de prefixos e1.024retornos usados
no ajuste espelhado1.024×1.024. Sete braços novos têm as mesmas observações/
tamanho de sistema, com custosCPU próprios; não afirmar operações idênticas.
Corpos/protocolo/fontes foram congelados antes da avaliação. Nenhuma coleta
nova: banco103/RNG9811 conhecido e já avaliado, portanto desenvolvimento.

| Arrependimento128passos, menor é melhor | Quadrática16 | Linear16 | Quadrática2 | Alvos embaralhados | Histórias embaralhadas | Só palavras | RBFpixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breakout | 0,125000 | 0,125000 | 0,125000 | 0,218750 | 0,156250 | 0,125000 | 0,125000 |
| Pong | 0,156250 | 0,156250 | 0,250000 | 0,218750 | 0,281250 | 0,312500 | 0,250000 |

Comparados também os12braços anteriores no mesmo banco. **Ambos os portões
primários falharam.** No horizonte128, quadrática e linear tiveram exatamente
as mesmas32escolhas em cada jogo. Breakout empatou com ignorar os pixels;
em Pong houve ganho médio0,156250 contra só palavras, em dois episódios,
IC95%[0;0,375000]. Contra história embaralhada, ganho0,125000, IC[0;0,312500].
Assim, o efeito positivo médio não tem o limite inferior estritamente positivo
exigido. Não promover a versão linear, nem mudar critério para salvar resultado.

Ganho frente à escolha constante:0,125000 em Breakout, IC[−0,093750;0,312500],
com um episódio pior;0,093750 em Pong, IC[−0,156250;0,312500], com dois piores.
História completa versus dois quadros em Pong melhora0,093750, mas vem de
um único episódio, IC[0;0,281250]. Não confundir com memória temporal validada.

MSE de diferença quadrática/linear:0,605989/0,639233 em Breakout,
1,283038/1,272919 em Pong. Os produtos propostos não melhoraram a decisão e
pioraram esse erro médio em Pong. Não alegar que aumentar riqueza algébrica
da família resolveu o gerador de representações a partir de experiência.

`auditar_leituras_por_programa.py` passou: origem crua de treino, normalização,
templates, palavras, marginais dos nulos, permutação de raízes preservando
prefixos, coeficientes satisfazendo ajuste,1.388eventos de avaliação e58condições
por jogo. Reexecução dos corpos reproduziu previsões sem erro observado;
diferença máxima entre execução e cálculo dual inferior7×10⁻¹⁵. Equivalência
com espaço fixo inferior6×10⁻¹⁶.64programas completos reservados por jogo não
coincidem com treino; isso não é transferência para mundo novo. Confirmações
anteriores e original preservados. Nenhum modelo promovido.

Ajustes concluídos em dois processos:203,53/212,59segundosCPU,
40,61segundos de parede por jogo. Avaliações4,11/4,34segundosCPU,
1,03/1,02segundos de parede. Zero interaçõesALE adicionais. O tamanho das
matrizes é registrado; não houve instrumentação do pico de memória do processo.
Contratos passaram para produto expressivo, exportação, equivalência, identidade,
sinal, aditividade, parâmetros inválidos e experiência contraditória.

## Quadragésimo segundo diagnóstico: pergunta e situação não estão cruzadas

O diagnóstico **posterior** `diagnosticar_identificabilidade_leituras.py` não
modifica portão. Em ambos os jogos, treino97 e desenvolvimento103, cada par
de programas de4/8/16blocos aparece em uma única raiz. Nos grafos finitos de
comparações, existem valores por palavra que ajustam todas essas diferenças
sem contexto: no comprimento16,64palavras/32arestas/32componentes desconexas.
Isso não significa prever novas palavras, nem que contexto seja inútil.
É uma limitação de identificação da amostra, não a raiz única da falha.

Em Pong, prefixos de1/2blocos já contêm inconsistências contextuais: em103,
um par repetido exige escolhas opostas em situações diferentes. Em Breakout,
esses prefixos têm zero diferenças em103. No horizonte128 há contraste real
em14das32raízes de cada jogo, embora todos os oito episódios tenham algum
contraste. Não contar32raízes como32episódios independentes nem512prefixos
como512experiências distintas. Os testes do diagnóstico verificam árvore,
ciclo consistente/contraditório, orientação do par e empate.

Antes de outra mudança de representação, usar o mesmo conjunto de programas
em várias situações e separar programas conhecidos/contextos novos de
programas novos/contextos novos. Medir inversões de preferência sem selecionar
situações pelas respostas reservadas. `EXPERIENCIA_CRUZADA.md` especifica
essa próxima bancada, ainda não executada. Ausência de contraste deve causar
inconclusivo, não anúncio de impossibilidade ou aumento indefinido de busca.

O objetivo humano segue não demonstrado. As leituras executáveis são uma
implementação verificável, mas não trouxeram o salto. Novos mundos, invenção
de mecanismos úteis e investigação autônoma de fronteira continuam questões
abertas do projeto, sem evidência de proximidade quantificável.

## Quadragésimo terceiro ensaio: perguntas repetidas em situações diferentes

Executada a bancada pré-especificada em `EXPERIENCIA_CRUZADA.md`. Quatro
programas de16blocos são comuns a todas as32raízes131; em137, esses quatro
e quatro novos são executados em todas as32raízes. Oito episódios por semente;
checkpoints0/2/4/6, não selecionados por retornos. Ajuste somente131,512pares
derivados de128trajetórias,1.024valores disponíveis ao ajustador; reportados
também os2.048retornos fisicamente medidos em todos os prefixos. Sete braços
iguais em observações, regularização0,05 e sistema1.024². Os mesmos corpos
congelados antes da coleta137 respondem às duas partições de avaliação.

Há nova evidência de uso de contexto, com limites precisos. Em **Pong,
programas conhecidos em contextos novos**, os dois pares exigem preferências
opostas entre situações; há contraste em todos os oito episódios. O primário
tem arrependimento0,125000 versus0,187500 só palavras: ganho0,062500,
IC95%[0,015625;0,109375], em quatro episódios. Contra história embaralhada,
ganho0,046875, IC[0,015625;0,093750], em três episódios. Entrada é somente
história visual reduzida e palavras; sem assinatura medida por sondas,
variáveis nomeadas ou estado interno no aprendiz.

**O portão completo dessa partição falhou.** Leitura de dois quadros é melhor
que o primário, e o ganho contra a versão linear não tem limite inferior
positivo. RBFpixels empata em arrependimento. Não promover um controle nem
transformar a evidência localizada em invenção irrestrita de representações.

| Arrependimento128, menor é melhor | Quadrática16 | Linear16 | Quadrática2 | Alvos embaralhados | História embaralhada | Só palavras | RBFpixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Breakout, conhecidos | 0,031250 | 0,046875 | 0,093750 | 0,281250 | 0,062500 | 0,015625 | 0,031250 |
| Breakout, novos | 0,375000 | 0,375000 | 0,421875 | 0,234375 | 0,375000 | 0,312500 | 0,359375 |
| Pong, conhecidos | 0,125000 | 0,140625 | 0,046875 | 0,296875 | 0,171875 | 0,187500 | 0,125000 |
| Pong, novos | 0,125000 | 0,093750 | 0,156250 | 0,187500 | 0,125000 | 0,078125 | 0,109375 |

As **outras três partições são inconclusivas** pelo critério de identificação
contextual congelado: Breakout conhecidos só um par inverte preferência;
Breakout novos nenhum par; Pong novos só um par. Todas têm diferenças reais,
mas exigíamos inversões em ambos os pares. Não relaxar essa exigência após
resultados. Também não apresentar esses números como transferência validada
de composição. Os doisROMs são conhecidos;137é uma nova semente, não um novo
domínio. Mecanismo comum foi reexecutado com corpos ajustados separadamente
por jogo; não houve transferência de um mesmo corpo aprendido entre jogos.

Auditoria `auditar_retomada_cruzada_2.py` passou: quatro bancos completos,
1.056eventos físicos,32replays certificados,5.380eventos de avaliação,
origem131dos alvos, marginais dos nulos, normalização, corpos e quatro
vereditos. Reexecução reproduziu previsões sem erro observado. Coeficientes
satisfazem os sistemas declarados, com resíduo máximo inferior3×10⁻¹³.
Contratos de término absorvente, contador físico, pares, portão, placebo,
colisão e interrupção porRSS/prazo passaram. O original foi preservado.

## Reparos de instrumento e custos desse ensaio

A primeira tentativa parou na primeira raizBreakout131 com384chamadas,
zero raízes completas e nenhum ajuste. Exigir que a tela imediatamente após
restaurar fosse a da raiz era um contrato incorreto. Na calibração29, estado
restaurado igual ao original e replay completo igual coexistem com tela em
cache igual ao final do ramo anterior. Foram378chamadas de calibração. V2
preserva fonte/proposta/checagensV1 e troca apenas esse contrato por igualdade
do estado clonado. Estado interno certifica o instrumento e nunca é entrada
do aprendiz. Retornos, término, imagens após ações e comprimentos são reais.

O monitor também interrompeu importações que criavamcmd.exe/conhost.exe para
consultas auxiliares de sistema. Um teste cobriuNumPy.testing, mas nãoALE
em importação fresca; uma segunda calibração cobriu ambos. Novos lançadores
consultam os metadados reais Windows/arquitetura diretamente, só nos próprios
processos, sem editar Python/NumPy/ALE instalados. Fontes, modelos, sementes,
primário e portõesV2 permaneceram intactos. Os registros interrompidos com
pico de quatro processos foram preservados; não afirmar limite de dois
processos para essas tentativas abortadas. Depois da correção, a retomada
efetiva ficou em dois processos incluindo monitor e picoRSSamostrado309,63MiB.

Bancos concluídos:19.668/30.683chamadasBreakout131/137 e24.484/41.024Pong,
total115.859. Incluindo tentativa falha e calibração:116.621. Os384passos
anteriores foram debitados do teto50.000Breakout131. Cada banco e suas
tentativas ficam abaixo do teto por jogo/semente. Piloto79,64segundos de
parede incluindo execuções interrompidas; implementação, calibrações e
auditoria são custos separados. Os modelos têmCPU/parede registrados por
braço. Monitor100ms, limite6GiB/1800s; o pico é amostrado, não limite rígido
do sistema operacional. Coletas131válidas foram reutilizadas sem repeti-las;
137não existia quando os corpos foram congelados. Nenhum modelo promovido.

## Quadragésimo quarto diagnóstico: limite de ignorar a situação

Com programas repetidos, agora podemos calcular um limite empírico: para
cada par, escolher o melhor dos dois programas **fixo em todas as situações**,
usando todas as respostas reservadas somente nesse cálculo oracular. O valor
é o mínimo das somas de arrependimento das duas escolhas, dividido pelo
número de comparações. Não fornecer esse oráculo ao agente, nem mudar portão.

Em Pong/conhecidos, qualquer regra determinística que recebe apenas as
palavras tem arrependimento mínimo0,156250 nessa amostra. Primário0,125000,
linear0,140625 e dois quadros0,046875 ficam abaixo disso. É evidência finita
de uso útil de contexto; não é garantia sobre novas amostras ou políticas
estocásticas que acertem por acaso, nem compreensão de objetos/leis humanas.
Os testes enumeram independentemente as quatro escolhas fixas possíveis
e reproduzem os limites das quatro partições.

Breakout/conhecidos tem mínimo0,015625, já atingido pelo braço só palavras;
novos tem mínimozero. Pong/novos tem mínimo0,031250, abaixo de todos os braços.
Nas imagens16quadros e dois quadros, não foram encontradas colisões que
forçassem arrependimento nesse banco; os limites oraculares por observação
sãozero. Ausência de colisão não prova que a representação é suficiente
ou que pode ser aprendida com eficiência. Uma tabela de memorização das
respostas137não constitui um aprendiz nem evidência de inteligência.

Próxima pergunta: os programas têm semântica de execução aprendida, ou a
função só interpola códigos de palavras? `word_features`/RBFnão impõem
composição de efeitos. `PROXIMA_LEITURA_SEMANTICA.md` propõe conferir a
identidade de diferença entre sufixos após um prefixo comum, usando história
intermediária real apenas como oráculo diagnóstico e controles de erro/
decisão além da consistência. Esse diagnóstico pode usar bancos já guardados,
sem outra coleta longa. O objetivo de mente humana e descobertas de fronteira
permanece não demonstrado; não há evidência para declarar uma peça única
faltante ou proximidade quantificada.

## Quadragésimo quinto diagnóstico: composição por prefixo comum

`SEMANTICA_PREFIXO_COMUM.md` e o manifesto foram congelados antes das
predições. Não houve ajuste ou chamadasALE. Nos bancos mistos97/103 de
Breakout/Pong, todos os28pares elegíveis satisfazem exatamente o cancelamento
físico da recompensa de prefixo. São sete raízes por banco, em seis episódios,
com apenas um bloco de prefixo comum. Cem raízes sem prefixo foram excluídas
somente pela sintaxe. Não extrapolar para prefixos longos.

Os sete corpos congelados131 foram consultados com palavras completas na
história inicial, sufixos nessa mesma história e sufixos na história após
o prefixo real. Esta última consulta é oracular: não demonstra capacidade
de prever essa história. Assinaturas de sondagens não foram lidas.

Arrependimento final do primário, com desempate uniforme para previsõeszero:

| Banco | Completo/início | Sufixo/início | Sufixo/meio real | Controlezero |
|---|---:|---:|---:|---:|
| Breakout97 |0,571429|1,000000|0,857143|0,500000|
| Breakout103 |0,571429|0,428571|0,571429|0,357143|
| Pong97 |0|0|0|0|
| Pong103 |0,714286|0,428571|0,571429|0,428571|

Pong97 temzero contrastes reais no horizonte final; seuszeros não são acertos
de decisão. Os outros bancos têm5/4/4contrastes reais, respectivamente.
Nenhum prefixo terminou em97; uma raizBreakout103 já tinha terminado no
meio. Todas foram mantidas, sem escolher casos pela recompensa.

O oráculo intermediário não resgata consistentemente o primário: piora
frente ao completo emBreakout97, empata emBreakout103 e melhora emPong103,
mas nesse último piora frente ao sufixo na história inicial e ao controlezero.
Os demais braços também têm resultados mistos; o braço só palavras tem
previsões idênticas com as duas histórias de sufixo, como deve. Não há evidência
para afirmar que basta melhorar o transporte perceptual. Leitura de sufixos
e semântica de execução continuam sem validação; não são uma causa única
deAGI identificada com perfeição.

O controlezero tem consistência perfeita e capacidade de decisão nula.
Logo, reduzir apenas a discrepância da igualdade não constitui sucesso.
Nenhum braço foi promovido e nenhum portão anterior foi reescrito. São
dados de desenvolvimento já vistos, não confirmação independente.

Seis testes de contrato passaram. Auditor separado recontou28casos físicos,
100exclusões,3.488eventos e reexecutou10.080previsões escalares por aritmética
dual/RBF independente: discrepância máxima3,98e-11. Métricas, curvas e
episódios reconciliados. Cálculo principal5,62segundos de parede, excluindo
implementação/congelamento/auditoria. Resultados e comandos completos em
`SEMANTICA_PREFIXO_COMUM.md`; originalCpermanece intacto.

## Quadragésimo sexto mecanismo: transporte executável de funções

`transporte_funcoes.py` implementa ações como operadores sobre33funções
sensoriais: constante eRBFde32suportes retirados das observações de treino.
Alvo local: recompensa econtinuação*funções do sucessor. Corpo de programa:
`v_vazio=0;v_aQ=r_a+M_a*v_Q`. Isso impõe concatenação por execução, evitando
que o kernel de palavras tenha de inventar essa lei. A acurácia física não
é garantida pela álgebra, nem a família finita constitui invenção irrestrita.

Código, controles, suporte32, regularização0,05 e portão foram congelados
em`TRANSPORTE_FUNCOES.md` antes do ajuste. Ambos os corpos97 ficaram congelados
antes de ler alvos103 neste piloto; dados de desenvolvimento já vistos.
861/1.024transições locais emBreakout/Pong, quatro ajustes contando controles
embaralhados, nenhum passoALE. Parâmetros separados por jogo, mesma regra:
não é transferência de corpo entre domínios nem mundo novo.

EmPong103/horizonte16, o primário passou o filtro local: arrependimento0,15625
contra0,5625(alvos embaralhados),0,25(sem contexto),0,59375(sem transporte),
0,25(ações permutadas),0,40625(zero). MSE1,846340 também abaixo de todos.
EmBreakout, arrependimento0,1875 contra0,15625embaralhado e0,1875sem contexto/
sem transporte; o controle sem transporte também temMSEmenor. O portão
conjunto exige ambos os jogos: falhou. Nenhum modelo foi promovido, nenhum
critério alterado. Ainda não há evidência para afirmar proximidade humana.

Cinco testes de execução/exportação/término/controles passaram. Auditor
separado reexecutou1.152eventos pela matriz aumentada, conferiu métricas e
veredito; discrepância máxima2,75e-14. Cálculo principal6,78segundos de parede.
Corpos e tabelas completas em`RESULTADO_TRANSPORTE_FUNCOES.md`; reprodução
em`TRANSPORTE_FUNCOES.md`. Limites: só família numérica escolhida por nós,
mundos conhecidos, nenhuma investigação autônoma de problemas de fronteira.

## Quadragésimo sétimo diagnóstico: de onde vem o erro de retorno

`DECOMPOSICAO_TRANSPORTE.md` implementou a identidade telescópica que separa
recompensa, transporte ponderado pela função de sufixo e raiz já terminada.
Corpos congelados, cinco braços, todas as32raízes de103 por jogo/horizontes
4/8/16; zero ajustes ou chamadasALE. Diagnóstico de desenvolvimento.

Ler recompensa aprendida em todas as histórias futuras reais piorou o
arrependimento do primário:Breakout0,1875→0,40625;Pong0,15625→0,46875. Retirar
idealmente sua parcela de erro de recompensa produziu arrependimentozero
nos dois, mas usa respostas verdadeiras e não é capacidade do agente.
Parcelas quadráticas não são culpa independente: o termo cruzado negativo
recompensa/transporte cancela parte do erro. Nenhuma causa única deAGI inferida.

Cinco testes passaram; auditor independente recontou960comparações e15.575
transições com discrepância máxima2,12e-14. Principal1,32segundos. Resultado,
equações, escopo e comandos em`RESULTADO_DECOMPOSICAO_E_CREDITO.md`.

## Quadragésimo oitavo mecanismo: resolver pesos pela consequência inteira

`credito_sequencias_transportadas.py` constrói um vetor de crédito para cada
programa por composição dos operadores, tornando os132pesos de retorno a
metade linear resolvível com funções/transporte fixos. Não é receita geral
de abdução emARC nem medida de promessa. Treino:512pares prefixais por jogo,
espelhados, derivados dos mesmos registros97. Zero chamadasALE. Cinco ajustes
por jogo; controles e regularização congelados antes do cálculo. Corpos novos
congelados antes de ler103 nesta rodada; dados103já vistos, desenvolvimento.

O primário piorou o arrependimento:Breakout0,1875→0,21875 ePong0,15625→0,375.
MSEde contraste0,882421/2,098752; ambos os portões falharam. A leitura local de
recompensa também piorou. Pesos efetivos de retorno não são recompensas causais
validadas. A solução numérica sem regularização no treino deixaMSE0,075752/
0,024362 para a família fixa; não chamar isso prova de impossibilidade geral.

Três testes passaram; auditor independente validou crédito, equações normais,
controles, métricas e1.354eventos. Principal2,48segundos, dez ajustes, nenhum
modelo promovido. Fontes/sistemas/corpos e comandos em
`CREDITO_SEQUENCIAS_TRANSPORTADAS.md`. Evidências e próxima hipótese em
`RESULTADO_DECOMPOSICAO_E_CREDITO.md`. Representações gerais a partir de dados
crus, transferência de corpos entre mundos e investigação humana permanecem
não demonstradas; o objetivo de mente humana continua ativo e não atingido.

## Quadragésimo nono mecanismo: funções por pares de contraexemplos

Geração real de comparações afins com tanh e produtos, com coeficientes e
origens derivados dos episódios de construção 0–3; família limitada escolhida
por nós. Base também usa somente esses episódios. Validação 4–5, corpos
congelados antes de avaliação 6–7 e 103. Ambos os bancos já eram conhecidos:
desenvolvimento, não confirmação independente. Detalhes de linguagem e
comparação em `GERADOR_FUNCOES_CONTRAEXEMPLOS.md`.

Nenhuma função selecionada. Em Breakout, 9 dos 32 candidatos primários
melhoram a decisão e 5 melhoram também o erro quadrático, mas em apenas um
caso; o protocolo exige dois. Direções aleatórias também têm 5 melhoras
conjuntas locais. Em Pong, os oito pares de validação têm retorno igual:
ausência de margem registrada como inconclusiva. Portão conjunto falhou;
previsões finais dos geradores são as da base, não ganho de função aprendida.

Diagnóstico separado dos 256 candidatos individuais dos quatro geradores
nos dois jogos mostrou que, em Breakout, nenhum corrige a linha 18; parte
corrige a linha 21. Cobertura insuficiente do pool, sem prova sobre combinações
arbitrárias. Leitura local melhora em algumas funções, inclusive em controles;
oráculos com histórias futuras/respostas verdadeiras não contam como agentes.
Erro de recompensa e transporte decompostos com termos cruzados, sem
atribuição causal independente. Não identificar uma única causa de AGI.

Cinco testes do gerador e quatro de cobertura passaram; auditorias recontaram
128 candidatos/1.920 comparações do piloto e 256 corpos/2.048 previsões do
diagnóstico. Discrepâncias máximas 1,20e-13 e 8,03e-13. Auditor original teve
falha de reprodução ao importar NumPy antes da configuração de BLAS. Fonte
preservada; bootstrap anterior à importação executou o mesmo auditor e passou,
sem mudar critérios. Principal 6,20 segundos no piloto e 3,16 no diagnóstico;
zero chamadas ALE. Relatório em `RESULTADO_GERADOR_FUNCOES.md`.

Próxima hipótese: gerar distinções pela resposta à mesma sequência de ações,
incluindo sua evolução temporal, com cobertura cruzada verificável antes
da avaliação. Ainda não implementada nem evidência de mente humana. Não
ampliar o pool ou relaxar o critério para salvar este resultado.

## Quinquagésimo mecanismo: distinções prospectivas e banco novo cruzado

`funcoes_prospectivas.py` implementou propostas por erro de previsão cruzada
de continuação vezes funções sensoriais seguintes, para mesma ação entre
episódios. Mantém linguagem limitada de comparações comtanh/produtos, com
corpos e origem. Testemunhas limitadas a512, índices sem consultar resultados;
ajuste de operadores em todas as transições de construção. SemLLM no aprendiz.

Fontes/programas/portão congelados antes da coleta149/151/157 nos dois mundos
conhecidos. Quatro programas comuns na construção/validação e quatro novas
composições na avaliação. Corpos congelados antes de coletar157. Bancos com
histórias intermediárias e todas as ramificações custeadas. Desta vez validação
e quatro células finais satisfazem o requisito de contraste contextual.

Prospectivo não escolhe função emBreakout; emPong escolhe7e13. Validação melhora,
mas os dados novos não confirmam vantagem contra todos os controles:
Pong/conhecidos arrependimento0,234375 versus0,296875sem funções e0,1875aleatório;
Pong/novos0,25 versus0,265625sem funções e0,226563zero. EmBreakout o primário é
a base. Todas as quatro células falham o portão; nenhum modelo promovido.

Diagnóstico separado decompôs os corpos congelados sem ajuste. Entregar todas
as histórias futuras reais pioraPong/novos0,25→0,390625; a leitura de recompensa
ainda não generaliza. EmBreakout/novos ajuda0,296875→0,203125, sem zerar erro.
Remover erro de recompensa por oráculo ajuda, mas não constitui capacidade
executável ou prova de que reajuste de pesos bastará. Termos cruzados preservados.

Nove testes novos passaram. Auditoria470ajustes/6.144comparações reservadas,
48replays físicos e160.438chamadasALE reconciliadas. Coleta~105s, ajuste15,05s,
diagnóstico4,07s. Experiência física equivale a641.752quadros somando ramificações,
não a105s de experiência humana. Fontes/bancos/corpos congelados, fontesoriginais
intactas. Relatório e precedentes em`RESULTADO_PROSPECTIVA.md`; comandos em
`PROSPECTIVA.md`. Objetivo humano permanece ativo, não atingido.

## 51. Leitura de eventos e limite algébrico do transporte linear

Correção: a etapa 50 fornecia histórias reais, mas previa a recompensa antes
de cada bloco; não testava reconhecer antes/depois. `leitor_eventos.py` agora
separa isso, mantendo bases congeladas. Ajuste só em149, congelamento antes
do diagnóstico151/157 e da coleta163. O campo escalar compartilhado falhou
nas quatro células finais com cobertura suficiente; nenhum corpo promovido.

Diagnóstico posterior, sem ajuste, em163: transição antes/depois reduz MSE
local frente a antes em24,12% noPong (8/8 episódios) e2,43% emBreakout (6/8).
Usa observações futuras reais, tem mais coeficientes e não é capacidade de
decisão antecipatória. Sem alegação de igualdade computacional ou efeito geral.

Derivação e verificação em `fechamento_leitor_eventos.py`: com mesmos dados,
características e regularização ridge, `beta+K@gamma=r_antigo`. Transportar o
novo leitor com esse projetor reproduz o preditor antigo: 1.536comparações,
erro máximo3,13e-12. Essa classe precisa acrescentar informação preditiva à
representação ou mudar a evolução aprendida para obter algo novo nessa operação.

Seis testes passaram; auditoria independente de36leitores/9.216comparações,
16replays físicos e74.804chamadasALE adicionais. Reconhecimento posterior
129.168previsões auditadas. Experiência299.216quadros, não47s de experiência
humana. `RESULTADO_EVENTOS.md` traz limites, números e próximo contraste.
Objetivo humano permanece ativo, não atingido. Original intacto.

## 52. Biblioteca visual da experiência, rodada incompleta por custo

Preservadas trajetórias espaciais completas em cinza, reexecutando149/151.
Os quatro bancos reduzidos coincidem integralmente com os anteriores.
Aprendidas cinco funções locais por mundo de8192patches de inovação, sem
consultar recompensa/ação. Origem por covariância e execução independente
auditadas; os mesmos pesos executam nos dois jogos, benefício não testado.

Oito testes de sensor/filtros e quatro de dicionário passaram. Dados:
6.959transições,5.822grupos de história+ação, zero colisões observadas que
reúnam histórias completas/recompensas distintas. Não prova suficiência ou
capacidade de aprender; não autoriza culpar diretamente a compressão.

O piloto com todos os controles atingiu120s em checkpoint antes de completar
Pong, preservando12corpos parciais. Manifesto completo ausente,179 não
coletado, portão não avaliado. Nenhuma promoção ou escolha de melhor sensor.
Perfil de um corpo confirmou equivalência de suportes/coeficientes/previsões
da implementação alternativa, não ganho de velocidade. A interpretação
machine-readable corrige o rótulo `faster` sem alterar o resultado congelado.

91.398chamadasALE/365.592quadros adicionais,32replays, coleta67,99s. Funções
e dados reais entregues, comparação funcional ainda inconclusiva.
`RESULTADO_RESOLUCAO.md` preserva custos, limites e próximo perfil por fase.
Objetivo humano permanece ativo, não atingido; original preservado.

## Etapa 53 — representação espacial, teste completo e contexto

A tentativa interrompida foi preservada; uma rodada própria usa otimização
numérica com equivalência verificada e aceleração pareada de 2,71–3,09 vezes.
Dez modelos completos e placebos congelados em 92,57 s antes da experiência 179.
74.686 chamadas físicas novas nos dois jogos; 16 replays certificados.

Quatro células com cobertura, nenhuma passa o portão. MSE aprendido no H16:
Breakout conhecido 0,68289 / novo 0,74528; Pong conhecido 0,84270 / novo 1,25914.
Arrependimento: 0,140625 / 0,25 / 0,140625 / 0,296875, respectivamente.
Ganhos locais não vencem todos os controles; em Pong novo, zero é melhor.
Auditoria completa de 6.912 comparações; reexecução com erro máximo 1,42e-14.

Diagnóstico sem ajuste mostra contexto útil contra sua ablação somente em
Pong conhecido e prejudicial nas composições novas dos dois jogos. O primário
não supera o melhor sinal constante por par calculado retrospectivamente em
nenhuma célula; esse oráculo não é agente aprendido. V2 corrige o nome da
referência: outros sensores são comparados à ablação do primário, não à própria.
Valores e portão preservados. Onze testes entre compactação e diagnóstico v2
passaram. `RESULTADO_RESOLUCAO_COMPACTA.md` contém custos, dados, hashes e limites.
Não demonstra mente humana, causa única, mundo desconhecido ou novidade mundial.
