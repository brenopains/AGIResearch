# Próxima medição: separar pergunta de situação

Este documento especifica uma correção de desenho; não registra uma execução,
uma confirmação independente nem uma arquitetura nova. A família de leituras
por programa falhou nos dois portões. No teste128passos, produtos quadráticos
produziram as mesmas32escolhas que a versão linear em cada jogo. Não ampliar
grau/dataset apenas para procurar um resultado positivo nesse banco conhecido.

O diagnóstico posterior encontrou uma limitação adicional: cada par de
programas longos aparece em uma situação apenas. Nas comparações de4/8/16blocos,
as amostras de treino97 e desenvolvimento103 admitem valores de palavras que
interpolam todas as diferenças sem usar o contexto. Há repetição de programas
curtos, mas poucos retornos não nulos; em Pong existem incompatibilidades
contextuais nesses prefixos. Não confundir esse resultado com prova de que
um modelo só de palavras generaliza, ou com causa única do fracasso.

Antes de outra intervenção de representação, a bancada deve distinguir estas
hipóteses: a situação precisa ser lida; a família proposta consegue lê-la;
a regra de ajuste consegue aprender seu efeito com as experiências disponíveis.

## Contrato do próximo banco

Gerar antecipadamente um conjunto pequeno de programas anônimos e executar
**os mesmos programas em todas as situações**. Guardar proposta e hash antes
de medir qualquer retorno. Capturar história passada antes das ramificações;
certificar repetição de uma ramificação por episódio. Sem inserir objetos,
coordenadas, nomes do jogo ou assinatura de sondas como entrada do aprendiz.

Separar episódios completos de treino/teste. Conservar uma partição que testa
programas conhecidos em situações novas, para isolar a leitura de contexto;
outra testa novos programas repetidos nessas situações, para avaliar composição.
Essas duas capacidades terão relatórios distintos. Misturá-las impede localizar
a falha. Ajuste e leitura das respostas da primeira partição não poderão alterar
corpos antes de avaliar a segunda. Nenhum uso das antigas sementes será tratado
como confirmação; escolher novas sementes/propostas antes da coleta.

Parâmetros da implementação seguinte, antes de coletar:

- Dois jogos de desenvolvimento já usados; não chamar transferência de domínio.
- Mundo de treino: semente131. Mundo reservado:137. Nenhum banco com essas
  sementes estava presente nesta pasta na especificação.
- Oito episódios por semente, oito checkpoints separados por128chamadasALE
  aleatórias; capturar os checkpoints0/2/4/6, total32raízes. Caminhada e ramos
  têm RNG/estados separados. Clonar/restaurar antes de cada programa, absorver
  término sem reiniciar a ramificação e registrar todo custo físico.
- Quatro programas comuns de16blocos/4ações, RNG9831; quatro novos, RNG9833.
  Rejeitar apenas duplicação sintática. As mesmas quatro palavras de treino
  aparecem em todas as raízes131; na avaliação137 executar as oito palavras
  em todas as raízes. Uma repetição completa da primeira palavra por episódio
  certifica o replay; nenhum retorno escolhe as palavras ou raízes.
- Treino: pares(0,1)/(2,3) nos comprimentos1/2/4/6/8/10/12/16.512pares derivados
  das128trajetórias,1.024valores usados pelo ajustador, sistema espelhado1.024².
  Registrar também os demais retornos físicos coletados, sem contá-los como
  dados disponíveis ao ajuste de qualquer braço. Regularização0,05; mesma
  família e sete braços do piloto de leituras, nenhuma alteração pelo teste.
- Na semente137, avaliar separadamente pares conhecidos(0,1)/(2,3) e novos
  (4,5)/(6,7), em32/64/128passos. Cada raiz é a unidade de decisão; calcular
  média dos dois pares por raiz e agruparbootstrap por oito episódios.
- Teto50.000chamadasALE por jogo/semente. No máximo dois processos,30minutos
  de parede do piloto,6GiB deRSS agregado monitorado. Interrupções geram
  checkpoint e inconclusivo; não aumentar teto nem reciclar respostas parciais
  para declarar o critério satisfeito. Confirmar monitor antes de iniciar.

O primário quadrático permanece pré-especificado; uma versão linear positiva
não substituirá seu veredito. Os novos modelos não serão ajustados em137.
Corpos e hashes precederão a leitura de retornos137. Assim a mudança investigada
é cruzar a experiência, não procurar outra família após ver os resultados.

## Diagnóstico em três níveis

1. **Identificação do banco.** Na partição avaliada128passos, exigir os dois
   pares repetidos com preferências estritamente opostas entre situações e
   diferenças não nulas em pelo menos três episódios. Sem isso, diagnóstico
   inconclusivo de contexto nessa partição. Medir em todas as raízes; não
   procurar/filtrar situações pelo retorno reservado. Essa exigência pode
   falhar; nesse caso o banco não testa adequadamente a pergunta proposta.
2. **Suficiência da observação.** Contextos com história observável idêntica,
   programas idênticos e consequências incompatíveis são testemunhas de
   insuficiência dessa observação. Ausência dessas colisões não prova suficiência.
   Versões de maior resolução ou memória são comparações diagnósticas separadas,
   com seus custos; estados ocultos continuam apenas referência oracular.
3. **Aprendizagem e composição.** Comparar o corpo aprendido com marginal de
   programa, situação embaralhada, alvos embaralhados e família sensorial mais
   simples. Se contexto conhecido generaliza mas novas composições não,
   concentrar intervenção na composição; se ambos falham, não atribuir toda
   a falha à composição nem aumentar escala cegamente.

Portão de capacidade por partição128: ganho médio de arrependimento positivo,
IC95% inferior estritamente positivo e pelo menos dois episódios positivos
contra cada um dos seis controles, além da identificação contextual acima.
Bootstrap10.000/RNG9837, agrupado por episódio; erros de diferenças e32/64passos
ficam secundários. Portões das duas partições são separados; êxito da hipótese
conjunta exige ambos. Não promover qualquer controle depois que o primário
falhar. O critério aqui é de um **experimento novo**, não muda os dois vereditos
negativos anteriores. Mesmo positivo exigirá transferência de mecanismos
aprendidos para mundos diferentes e investigação autônoma verificável antes
de sustentar a meta de mente de nível humano.
