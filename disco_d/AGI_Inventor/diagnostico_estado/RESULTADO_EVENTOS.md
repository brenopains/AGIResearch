# Etapa 51 — reconhecer consequências não é antecipá-las

Data: 2026-09-29. Objetivo de mente de nível humano não atingido. Não há medida
de proximidade humana, inovação mundial demonstrada ou causa única de AGI.
O projeto original não foi editado. Código, corpos e experiência deste trabalho
estão isolados em `D:\AGI_Inventor\diagnostico_estado`.

## O resultado que muda o próximo passo

O leitor de transição melhora o reconhecimento local de eventos em experiência
nova, mas utiliza observações posteriores à ação. Transportá-lo pela mesma base
e pelos mesmos operadores lineares **produz o preditor antigo**. Essa identidade
explica por que uma leitura melhor do futuro já ocorrido, sozinha, não cria uma
previsão melhor para escolher a ação. Não é uma impossibilidade geral de AGI.

A hipótese adicional de um campo escalar compartilhado também não passou:
`F(H)=phi(H)@theta`, com evento `F(H_next)-F(H_current)` e composição pelos
transportes existentes. Todas as quatro células primárias tinham cobertura e
falharam o portão congelado. Nenhum corpo promovido como capacidade geral.

## Correção do diagnóstico anterior

Em `RESULTADO_PROSPECTIVA.md` foi escrito que fornecer histórias futuras reais
não resgatava a leitura de consequências. O cálculo era `phi[t]@reward[action]`,
antes de cada bloco de oito chamadas ALE; ele ainda tentava prever o evento.
Portanto, o resultado não provava falha de reconhecer o evento a partir das
observações antes e depois. Essa inferência era ampla demais. Resultados antigos
e fontes congeladas foram preservados; a correção é acrescentada aos relatórios.

## Implementação e desenho

`leitor_eventos.py` implementa nove braços exportados: antes, depois, diferença,
transição, potencial compartilhado, sucessor embaralhado, recompensa embaralhada,
média por ação e zero. Cada corpo possui definição, referência de características
e coeficientes inspecionáveis. O executor numérico não usa Python gerado ou LLM.

`piloto_leitor_eventos.py` congela fontes/protocolo, ajusta somente em 149,
congela corpos, pontua desenvolvimento 151/157 e avalia composição em experiência
nova 163. `coletar_eventos.py` coleta ramos físicos sem modificar o coletor
anterior. As bases da etapa anterior permanecem congeladas; há 37 coordenadas,
das quais posições reservadas podem ser zero. A base primária é sem funções
adicionais, a prospectiva é secundária. Cada jogo aprende parâmetros próprios;
isso não demonstra transferência de um corpo entre mundos.

As entradas são histórias de imagens cinza reduzidas por médias de blocos
10×10, resultando em 21×16 células por quadro, 16 quadros por história. Não são
pixels completos nem uma representação inventada irrestritamente. Não avaliar
um humano vendo imagens completas como se recebesse essas mesmas observações.

Na decisão, o campo recebe somente a história inicial. A previsão de estado
terminal mantém a última observação em um bloco absorvente; essa é uma convenção
do protocolo de padding, não uma regra inventada pelo sistema. Clones ALE são
usados apenas para o avaliador físico, nunca como entrada cognitiva.

## Reconhecimento local

No desenvolvimento 157, MSE antes→transição:
Breakout 0,059075→0,056524 (redução 4,3%);
Pong 0,086194→0,067333 (redução 21,9%). Isso motivou a separação entre leituras.

Um diagnóstico **posterior ao resultado de planejamento**, explicitamente
identificado como tal, pontuou todos os leitores já congelados na experiência
163. Não reajustou nem selecionou corpos, não alterou o portão confirmatório:

| Jogo | Antes | Transição antes/depois | Sucessor embaralhado | Recompensa embaralhada | Zero |
|---|---:|---:|---:|---:|---:|
| Breakout | 0,051631 | 0,050377 | 0,053579 | 0,062705 | 0,058879 |
| Pong | 0,096364 | 0,073121 | 0,102597 | 0,186249 | 0,203984 |

Transição reduz MSE frente a antes em 2,43% no Breakout (6/8 episódios) e
24,12% no Pong (8/8 episódios). Frente ao controle com sucessor embaralhado,
melhora em 7/8 e 8/8 episódios. Não foi estimado intervalo de confiança nem
demonstrada generalização a um jogo desconhecido. O leitor de transição tem
mais coeficientes que o de antes: não alegar igualdade de trabalho computacional.

Não confundir a fração de recompensas arredondadas corretamente com uma prova
exata ou desempenho humano. Breakout tem somente 189 recompensas não nulas em
3.210 transições, Pong 809 em 3.966: zero já arredonda corretamente em 94,1% e
79,6%, respectivamente. O MSE nos eventos não nulos da transição é 0,730515 e
0,230783; reconhecimento ainda apresenta erro relevante.

## Planejamento com a diferença de potencial

Critério fixo: no horizonte 16, MSE e arrependimento com desempate justo menores
que todos os cinco controles nas quatro células primárias. Horizontes 4/8 são
reportados sem substituir o principal. Todas as células têm preferências reais
em ambas as direções por par e contraste em pelo menos três episódios.

| Célula | Arrependimento campo | Antigo | Zero | MSE campo | Antigo | Zero |
|---|---:|---:|---:|---:|---:|---:|
| Breakout / programas conhecidos | 0,140625 | 0,078125 | 0,343750 | 0,637478 | 0,518681 | 0,750000 |
| Breakout / composições novas | 0,375000 | 0,281250 | 0,320313 | 0,820313 | 0,817312 | 0,796875 |
| Pong / programas conhecidos | 0,265625 | 0,265625 | 0,304688 | 1,182541 | 1,201603 | 1,171875 |
| Pong / composições novas | 0,187500 | 0,140625 | 0,171875 | 0,570373 | 0,601674 | 0,531250 |

Veredito `event_field_gate_failed`. Falha não decorre apenas de igualdade com
uma ablação: o campo perde do preditor antigo no arrependimento em três células
e empata na quarta. Ganho em uma métrica isolada não salva o portão.

O grafo de histórias de construção é compatível com diferenças inteiras de
potencial: Breakout 1.132 nós, 12 componentes, 1.667 restrições; Pong 1.607 nós,
23 componentes, 1.963 restrições. Há 547 e 379 restrições além de uma floresta,
mas repetições podem ser triviais; compatibilidade finita não certifica uma lei
universal, suficiência de observação ou aprendizagem eficiente. A auditoria
conferiu compatibilidade por caminhos, independentemente da união ponderada.

Mesmo sem regularização, na família fixa de diferenças de características,
mínimos quadrados com corte numérico de posto 1e-10 deixam MSE de construção
0,052090 (Breakout) e 0,107423 (Pong). Acrescentar as duas funções prospectivas
previamente admitidas em Pong reduz para 0,106242. É um limite empírico dessa
família numérica, não prova de impossibilidade para toda representação.

## Uma razão algébrica para não insistir apenas no leitor

Sejam X e Z as características antes e depois de uma ação, y o evento observado
e alpha a regularização idêntica. O leitor de transição ajusta:

```text
min ||X beta + Z gamma - y||² + alpha (||beta||² + ||gamma||²)
```

A primeira equação normal implica:

```text
(XᵀX + alpha I) beta + XᵀZ gamma = Xᵀy
K = (XᵀX + alpha I)^(-1) XᵀZ
r_antigo = (XᵀX + alpha I)^(-1) Xᵀy
beta + K gamma = r_antigo
```

O transporte incondicional K é a soma do transporte vivo com o terminal,
ajustados sobre os mesmos dados com a mesma regularização. Ler antes/depois e
transportar a parte de depois fornece o mesmo vetor de recompensa antigo.
Compondo com o mesmo transporte vivo, obtém-se o mesmo valor de programa.

`fechamento_leitor_eventos.py` congelou essa verificação antes de executá-la.
Foi definida depois do resultado principal e não muda seu portão. Conferiu
1.536 comparações nas duas bases, nos dois jogos, em três horizontes:
diferença máxima de coeficientes 8,17e-12, de previsões 3,13e-12, resíduo de
equações normais 2,44e-12. A derivação é exata sob as condições declaradas;
as instâncias computadas são aproximações numéricas, não provas numéricas exatas.

Não é uma novidade mundial reivindicada. Não usa medida de promessa nem
compressão do buraco. Mostra uma restrição específica do mecanismo atual:
o ganho retrospectivo não se converte em capacidade antecipatória apenas por
reajustar esse leitor com o mesmo projetor linear.

## O que construir a seguir

Não repetir ajustes do leitor nessa mesma base esperando previsão nova. O alvo
seguinte é uma **representação que acrescente informação sobre consequências
antes da ação**, ou um modelo de evolução que capture dependências ausentes da
projeção linear atual. Nenhuma das duas alternativas está demonstrada aqui.

O próximo contraste deve separar perda sensorial, perda na representação e erro
do modelo: preservar imagens com maior resolução para uma amostra custeada;
verificar colisões entre histórias comprimidas e consequências diferentes;
gerar distinções executáveis a partir desses contraexemplos; testar retenção dos
eventos no reconhecimento e, separadamente, antecipação com informação disponível
no início. Colisões observadas são contraexemplos concretos; sua ausência não
prova que o estado seja suficiente. Validar fora das situações que propuseram
as distinções, com controles e orçamento compartilhado, antes de aumentar escala.

Essa direção não é uma receita suficiente para mente humana. Continuam sem
demonstração: representação em mundos desconhecidos, eficiência comparável à
humana, reutilização entre modalidades/domínios e investigação autônoma de
problemas de fronteira. A melhora local de reconhecimento não substitui esses
requisitos e não permite dizer que faltaria uma única peça.

## Contabilidade e reprodução

- Seis testes do executor passaram. Sintaxe dos novos drivers verificada.
- Corpos ajustados em 2,44 s e congelados antes de diagnóstico e coleta 163.
- Coleta nova: 34.003 chamadas ALE em Breakout e 40.801 em Pong, 74.804 no total.
  Com salto de quatro quadros, são 299.216 quadros (~83,12 minutos somados
  a 60 quadros/s), incluindo caminhada, todos os ramos e replays.
- Coletas cerca de 23,60 s e 23,56 s; maior RSS amostrado 183.816.192 bytes,
  aproximadamente 175,30 MiB. Cada banco ficou abaixo de 120 s/50.000 chamadas.
  RSS amostrado não é contenção rígida nem mede processos externos.
- Essas chamadas são adicionais: a bancada prospectiva anterior custou
  160.438 chamadas, incluindo construção, desenvolvimento e avaliação. Seus
  corpos são reutilizados; não apresentar a nova coleta como custo total de aprender.
- Auditoria independente conferiu 36 leitores, 9.216 comparações de planejamento,
  16 replays integrais e a contabilidade física. Erro máximo de coeficientes
  1,06e-11 e da matriz aumentada 8,93e-14.
- Diagnóstico posterior de reconhecimento conferiu 129.168 previsões por
  definição independente; discrepância máxima 3,25e-15. Nenhuma coleta/ajuste extra.
- A checagem de fechamento não coletou nem ajustou corpos.

Protocolo/comandos principais em `EVENTOS.md`. Complementos posteriores:

```powershell
py -3.11 -B diagnosticar_eventos_frescos.py prepare
py -3.11 -B diagnosticar_eventos_frescos.py run
py -3.11 -B fechamento_leitor_eventos.py prepare
py -3.11 -B fechamento_leitor_eventos.py run
```

Saídas recusam sobrescrita. Os resultados desta rodada já existem; esses
comandos documentam a reprodução da sequência, não devem sobrescrever a evidência.
Corpos e resultados completos em `resultados/eventos*.json`, `.npz` e `.jsonl`.
As verificações preservam hashes de fontes e dependências. Arquitetura original
foi apenas lida: SHA256 `da47cc53b795305f416e22dc6f120607919000d1e224f6d4e4cf88f2ac5ba8ad`.
