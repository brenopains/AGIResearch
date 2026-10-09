# Próxima hipótese: inventar perguntas que distinguem mundos

O resultado confirmado em `PRE_REGISTRO_PLANOS_109.md` permite compor um
estado definido por consequências de ações. Ainda escolhemos manualmente
quais perguntas definem esse estado: quatro ações, oito passos, quadro
21×16. O mecanismo seguinte deve aprender **quais intervenções e quais
resumos sensoriais perguntar**, não receber uma lista de objetos, leis ou
variáveis do domínio.

## Gerador de perguntas por contraexemplo

Uma representação `z(história)` é inadequada se encontra duas histórias
`h1,h2` que considera equivalentes, mas uma mesma continuação `u` produz
consequências diferentes. Esse par e a continuação são uma **testemunha
executável** do que falta. O procedimento proposto é:

1. Formar pares próximos no estado atual, usando apenas episódios de treino
   independentes; verificar, com clones ou novas repetições, que a diferença
   futura ultrapassa a variação causada por ruído. Não tratar proximidade
   numérica como identidade matemática.
2. Sintetizar por busca incremental uma continuação curta `u` de ações
   primitivas que separa o par. A busca aumenta comprimento apenas quando
   programas mais curtos não bastam; seu custo inclui ações e observações.
3. Extrair do resíduo entre os dois resultados uma leitura sensorial
   executável `g`: região, combinação de canais ou projeção aprendida apenas
   dos pixels/sons/números observados. A nova pergunta é
   `q(h)=g(observar(executar(u,h)))`. Guardar `u`, `g`, custo, exemplos e
   contraexemplos; um nome ou vetor sozinho não basta.
4. Treinar um extrator `E(história)` para prever a resposta de `q` **antes**
   de executar `u`. Usar respostas reais somente no treino; na avaliação,
   o estado vem da experiência bruta disponível. Aprender como cada ação
   atualiza o novo estado e testar sequências que não estavam no treino.
5. Promover `q` apenas se reduzir erros futuros e melhorar escolha de ação
   fora do episódio e da semente de descoberta, com custo de obtenção e
   definição contabilizado. Uma pergunta útil para um par, mas sem ganho
   para decisões novas, permanece candidata.

Os programas de ação são construídos por composição sobre um alfabeto mínimo
de ações do ambiente; nenhum conceito do jogo entra como primitiva. A regra
de refinamento por contraexemplo é geral, mas **não garante** que a busca
encontre perguntas curtas ou que o extrator consiga prevê-las. Esse é o
teste de fato. O mecanismo não depende de pontuação vaga de "promessa": cada
pergunta proposta precisa vir de uma divergência causal reproduzível.

## Piloto de até 30 minutos de CPU

Usar primeiro os estados clonados já guardados em Breakout e Pong. Selecionar
pares só em 97, sintetizar perguntas sem olhar 103/109, e reservar outra
semente intacta para confirmação. Comparar com perguntas de mesmo custo e
mesma dimensão escolhidas aleatoriamente, representação fixa de quatro
efeitos, pixels diretos e ação permutada. Relatar separadamente:

- pares verdadeiramente indistinguíveis na representação atual mas com
  futuros diferentes;
- fração para a qual a busca acha uma pergunta que os separa;
- precisão do extrator da pergunta a partir de dados crus;
- estabilidade da atualização sob planos inéditos;
- ganho em recompensa/arrependimento da decisão, não só previsão visual.

Se o primeiro número for zero, os estados testados não expõem o gargalo;
ampliar mundos ou perturbações, sem declarar suficiência. Se houver
testemunhas mas a busca falhar, o gerador ou orçamento é o gargalo. Se a
pergunta for encontrada e o extrator falhar, o gargalo é percepção. Se
predição funcionar e a composição falhar, é dinâmica do estado. Se tudo
isso funcionar e a decisão não melhorar, a informação descoberta não é
suficiente para o objetivo. Essa decomposição é o caminho para avançar sem
transformar cada falha em uma aposta vaga em escala ou datasets.

Uma confirmação nesses dois jogos ainda não demonstraria inteligência geral
nem descoberta científica. Antes dessa alegação, o **mesmo gerador**, sem
ajuste de conceitos, teria de funcionar em mundos de física, relações,
vídeo e tarefas com objetivos novos; e medir resolução de problemas, não
apenas compressão de observações.
