# Próxima pergunta técnica, após o piloto negativo

Não sabemos se o sistema está perto do nível humano. Não existe nos resultados
atuais uma escala que permita dizer quanto falta. Nem os jogos nem acertar
variáveis numéricas demonstram pesquisa autônoma ou generalidade humana.

Resultado novo: os desfechos condicionais contêm informação útil de recompensa
local, mas a composição prevê retornos falsamente favoráveis. Substituir só
o estado inicial pela medida real não elimina a falha. Isso rejeita a solução
"basta melhorar o leitor inicial" para esse mecanismo. Não rejeita percepção
como problema em outros domínios, nem estabelece um gargalo único da AGI.

A pergunta agora é: **o estado preserva a informação que distingue futuros
sob sequências de ações ou o modelo perde essa informação ao atualizá-lo?**

O diagnóstico seguinte precisa comparar, no mesmo treino e nas mesmas
sequências reservadas, três níveis explicitamente separados:

1. Estado anterior real e modelo de um passo: medir recompensa, término e
   próximas consequências. Isso testa se a informação atual é suficiente
   para o preditor testado, sem erro acumulado de trajetória.
2. Próximo estado previsto e leitura aprendida de consequências: medir o que
   se perde na composição. Um bom erro médio de vetor não basta; exigir
   preservação das diferenças entre ações e dos eventos relevantes futuros.
3. História crua adicional, com mesmo orçamento e validação independente:
   testar se recupera diferenças que o estado atual não preserva. Se recuperar,
   há evidência de perda de memória/informação; se não, isso não prova que
   o estado seja suficiente, pois o novo extrator pode também falhar.

Controles necessários: histórias embaralhadas preservando marginais, mesma
dimensão/custo, mesmo orçamento de observação; futuros reservados apenas no
avaliador. Treino deve conter sequências com troca de ação, pois composição
de modelos treinados só em programas constantes muda a distribuição consultada.
Não chamar ausência de melhora de impossibilidade ou raiz perfeitamente
identificada. Registrar o alcance e a hipótese que cada contraste elimina.

Antes de avaliar, congelar o código e escrever um portão conjunto: melhorar
previsão sob composição **e decisão real** nos dois ambientes, contra todos
os controles relevantes. Só então reservar semente nova para confirmação.
Não usar futuros reais para escolher a política final, não trocar o primário
depois dos resultados e não tentar salvar resultado com mais horas de busca.

Essa sequência ainda é pesquisa de um mecanismo, não receita garantida de AGI.
Depois de um portão positivo, é necessário testar a mesma regra de aprendizagem
em mundos diferentes, sem inserir variáveis ou regras próprias de cada mundo,
e em tarefas de investigação verificáveis. O objetivo permanece em aberto.

## Resultado da intervenção de história

O teste acima avançou em `MEMORIA_RESIDUAL.md`: foram coletadas transições
mistas e geradas32variáveis da história a partir dos resíduos fora do ajuste.
Os dois portões falharam. Houve menores erros locais e menor erro médio de
retorno em Pong, mas a escolha entre sequências piorou nos dois jogos.

Isso ainda não identifica perfeitamente informação ausente versus modelo
errado. Acrescenta uma restrição concreta à próxima tentativa: conservar a
**diferença de consequência entre sequências**, e demonstrar benefício de
decisão, precisa ser medido separadamente de acertar o vetor/retorno médio.
Variáveis geradas da história por correlação linear não garantiram isso.
Os estados iniciais eram medidas reais; melhorar só o leitor inicial não
resolve esse resultado. Próximo: gerador orientado por pares de consequências
confundidas, com orçamento/controles iguais e sem usar futuros reservados.

## Resultado da comparação direta de programas

`CONTRASTES_PROGRAMAS.md` testou uma função executável de diferença de retornos,
evitando acumulação de estados médios. Nos mesmos mundos,64palavras completas
novas por jogo não coincidem com o treino. Ambos os portões falharam. Houve
melhora média frente à regressão de retornos absolutos, mas os intervalos
incluem zero; Breakout foi melhor ignorando o estado, e Pong empatou com a
escolha constante. Os controles impedem interpretar isso como compreensão
da situação ou transferência geral.

Próxima pergunta: construir a leitura sensorial **condicionada ao programa
que está sendo comparado**, em vez de uma variável temporal fixa para toda
pergunta, ajuda a conservar diferenças reais? O candidato tem de ser gerado
somente de treino, ter corpo completo e vencer leitura condicionada aleatória/
embaralhada, informação marginal das palavras e leitura fixa. É hipótese
nova de integração, não mecanismo inédito demonstrado ou garantia de AGI.

## Resultado da leitura por programa e limite de identificação

`LEITURAS_POR_PROGRAMA.md` foi executado com sete braços novos e os12controles
anteriores. Corpos completos recebem só pixels passados; nenhum portão passou.
No horizonte128, acrescentar projeções quadráticas não alterou uma única
escolha frente à leitura linear. Pong tem sinal médio a favor de ler pixels,
mas IC do contraste contra ignorá-los inclui zero; Breakout empata. Não há
evidência para atribuir a esse gerador o salto em representações pedido.

O diagnóstico posterior encontrou ausência de cruzamento dos programas
longos entre situações: por comprimento4/8/16, cada par aparece uma vez.
O grafo finito admite explicação por valores de palavras sem contexto;
isso não equivale a generalização para palavras novas. Prefixos curtos
de Pong contêm incompatibilidades contextuais, mas pouco contraste real.
Essa restrição do desenho precisa ser tratada antes de responsabilizar
exclusivamente percepção, composição, dados ou seletor.

Próximo teste especificado em `EXPERIENCIA_CRUZADA.md`: mesmas perguntas
executáveis em situações diferentes, partições distintas para leitura de
contexto e novas composições, corpos/critério congelados antes dos resultados.
Se não houver inversões de preferência/contraste contextual suficiente,
inconclusivo; não selecionar casos reservados para fabricar inversões.
Não confundir essa correção de bancada com nova arquitetura ou AGI.

## Resultado da experiência cruzada

A bancada foi executada em131/137, com palavras comuns entre32situações
e partições distintas para contexto e composições novas. Pong/conhecidos
mostrou uso útil de pixels: o primário supera só palavras e história
embaralhada, com limites inferiores deIC95%positivos. Seu arrependimento0,125
também fica abaixo de0,15625, o mínimo empírico de uma regra determinística
que só recebe as palavras nesse banco. É uma evidência limitada de contexto.

O portão completo falhou: dois quadros foram melhores que16, e o ganho
contra a versão linear não está confirmado. As demais três partições
são inconclusivas por contraste contextual insuficiente pré-especificado.
Não atribuir isso a uma única causa nem elevar um controle a primário.
Reparos de instrumento, fontes efetivas, auditoria e comandos estão em
`COMANDOS_EXPERIENCIA_CRUZADA.md`; nenhum modelo promovido.

Agora separar contexto de **semântica de execução**: os kernels de palavra
atuais impõem relações entre alternativas, mas não composição dos seus
efeitos. Conferir nos registros `Δ(H,P+Q,P+R)=Δ(H_após_P,Q,R)` para prefixos
realmente comuns; alimentarH_após_Papenas como oráculo diagnóstico. Medir
acerto/decisão junto de consistência, pois um modelozero passa a igualdade.
Detalhes em `PROXIMA_LEITURA_SEMANTICA.md`. Não tratar essa leitura futura
como capacidade de simular, nem como "promessa" de prefixoARC.

## Resultado do prefixo comum

`SEMANTICA_PREFIXO_COMUM.md` foi executado e auditado sem coleta/ajuste.
Cancelamento físico exato nos28casos, todos com prefixo de um bloco. Entregar
a história intermediária real não resgatou consistentemente o primário:
contra as palavras completas, piorouBreakout97, empatouBreakout103 e melhorou
Pong103, mas ali perdeu do sufixo na história inicial e do controlezero.
Pong97 não tem contraste de retorno nesses sete pares. É diagnóstico pequeno
de desenvolvimento, não teste independente ou identificação de uma causa única.

O próximo mecanismo precisa ligar a concatenação de ações à composição
de efeitos aprendidos. Não basta fazer sua igualdade algébrica passar:
zero já passa. Exigir previsão e decisão contra controles que removam
o transporte aprendido. Nem essa hipótese é garantia de mente humana.

## Transporte de funções implementado

`TRANSPORTE_FUNCOES.md` implementou essa ligação em uma família finitaRBF,
sem sondas, código aprendido arbitrário ou regras específicas de cada jogo.
EmPong103, o transporte melhorou decisão eMSE frente a todos os controles
de desenvolvimento; emBreakout103, não. O portão conjunto falhou; corpos
preservados, nenhuma promoção. `RESULTADO_TRANSPORTE_FUNCOES.md` contém
valores, comandos e alcance. Passar a identidade algébrica não foi tratado
como prova de acerto físico. O gerador ainda é uma família que nós escolhemos;
invenção de representação e pesquisa humana permanecem não demonstradas.

## Atribuição e intervenção no crédito

`RESULTADO_DECOMPOSICAO_E_CREDITO.md` contém a intervenção seguinte: decompôs
exatamente o erro em recompensa/transporte/término, e tentou resolver apenas
os pesos de retorno por consequências inteiras. Histórias reais futuras não
recuperaram a leitura atual; remover erro de recompensa por oráculo ajudou,
mas o ajuste executável de crédito piorou ambos os jogos. Portões preservados.
Isso orienta gerar funções a partir de consequências confundidas, sem afirmar
que representação seja a causa única. O banco precisa também cruzar as mesmas
perguntas entre situações; histórias embaralhadas ajustam bem parte do treino.

## Funções novas: efeito local não basta

`RESULTADO_GERADOR_FUNCOES.md` registra o gerador implementado e auditado.
Ele cria comparações não lineares de histórias e produtos, com corpos e
origens reais. Nenhuma promoção. Em Breakout algumas funções melhoram um
caso, mas nenhum candidato individual dos quatro pools melhora o segundo;
em Pong os oito pares de validação não distinguem escolhas por retorno.
Esses são limites diferentes: cobertura do pool e ausência de contraste no
instrumento. Não concluir que toda invenção de representação falhou.

O próximo gerador deve investigar consequências prospectivas: histórias
com mesma representação atual, mas respostas diferentes à mesma sequência
de ações. Medir separadamente se uma distinção explica consequência e se
seu transporte permite usar essa explicação antes de observar o futuro.
Construção/validação precisam repetir essas perguntas em situações diversas.
Isso permanece hipótese de mecanismo limitado; proximidade humana, causa
única e transferência geral ainda não foram demonstradas.

## Distinções prospectivas testadas com cobertura suficiente

`RESULTADO_PROSPECTIVA.md` concluiu essa intervenção com dados novos e corpos
congelados antes da coleta de avaliação. Pares pela consequência de uma ação
geram variáveis executáveis, mas não perguntas arbitrárias de múltiplos passos.
Semântica composta foi testada em programas conhecidos e composições novas.
Cobertura suficiente nas quatro células; portão falhou em todas.

EmPong existem funções com efeito, mas nos programas conhecidos o controle
aleatório é melhor e nas composições novaszero é melhor. Entregar estados
futuros reais não resgata a leitura aprendida de recompensa; emBreakout ajuda
parcialmente. Assim o próximo alvo deve preservar eventos relevantes na
descrição sensorial, junto à evolução temporal dessas descrições. Prever uma
baseRBF futura não demonstrou bastar. Não inferir desse resultado uma causa
única deAGI nem abandonar os requisitos de transferência e autonomia de fronteira.

## Correção da leitura e fechamento linear — etapa 51

Entregar histórias reais no diagnóstico anterior ainda significava prever o
evento antes do bloco; não reconhecer o evento a partir de antes/depois.
Reconhecimento foi medido separadamente em `RESULTADO_EVENTOS.md`: melhora
local, mas depende do sucessor observado. O campo escalar compartilhado falha
na experiência nova com cobertura suficiente.

Há um limite derivado, além do nulo empírico: com X=phi(antes), Z=phi(depois)
e a mesma regularização, ridge conjunta `X beta+Z gamma` implica
`beta+K gamma=r_antigo`, onde K é ridge de Z sobre X. A leitura retrospectiva
melhor fecha no mesmo preditor quando transportada nessa base. Verificação
em 1.536comparações, erro máximo3,13e-12, sem coleta/ajuste adicional.

Esse resultado refuta a esperança de resolver essa operação apenas mudando
o leitor, mantendo tudo o resto. Não demonstra uma causa única deAGI.
Próximo contraste: perda sensorial versus perda na representação versus
evolução inadequada. Preservar resolução, procurar contraexemplos observados
e testar distinções executáveis fora dos exemplos que as propuseram.

## Observações espaciais preservadas — etapa52 incompleta

`RESULTADO_RESOLUCAO.md` introduz funções locais de inovação temporal com
corpo e origem auditados, além de quadros completos em cinza para perguntas
posteriores. Nenhuma colisão observada entre história reduzida+ação e
histórias completas/recompensas distintas em6.959transições. Portanto, não
apareceu evidência direta de que a compressão determine esses erros; ausência
de colisão finita tampouco prova que a observação seja suficiente.

Comparação de informação/geometria sensorial ficou incompleta ao atingir o
orçamento de ajuste. Não inferir ganho/falha de capacidade dos12corpos
parciais. Próximo: medir fases de extração/dicionário/projeção e viabilizar
o mesmo teste completo, preservando critérios e contabilizando o esforço
anterior. Nenhuma aproximação dessa camada certifica mente humana.

## Comparação espacial concluída — etapa 53

`RESULTADO_RESOLUCAO_COMPACTA.md` concluiu os dois jogos e todos os controles.
Compactação preservou instâncias numéricas e acelerou ajustes em 2,71–3,09
vezes. Ajuste completo em 92,57 s; corpos antes da coleta 179, portão intacto.
Quatro células com cobertura, nenhuma aprovada. Esta família PCA/RBF com
transporte linear não demonstrou o salto de representação.

Diagnóstico posterior distingue mudar escolhas de usar contexto com benefício:
o primário varia seus sinais, mas a ablação é melhor nas composições novas dos
dois jogos. Benefício aparece em Pong conhecido, onde filtros aleatórios ainda
vencem. O primário não supera a melhor constante retrospectiva por par em
nenhuma célula. Essa constante usa respostas reservadas e não é agente aprendido.

O próximo alvo é estado orientado por consequências de ações compostas;
preservar mais variação visual não bastou. Isso não identifica causa única
de AGI. Separar percepção, leitor de eventos e propagação continua necessário.
Pares prospectivos já tentados não podem ser anunciados novamente como inéditos.
74.686 chamadas novas, 16 replays; 6.912 comparações auditadas. Objetivo aberto.
