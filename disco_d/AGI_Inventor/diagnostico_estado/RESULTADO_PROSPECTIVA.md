# Gerador prospectivo: implementação, avaliação nova e diagnóstico

O objetivo de mente artificial de nível humano continua não atingido.
O mecanismo foi implementado e testado; o portão de capacidade falhou nas
quatro células. Nenhum modelo foi promovido como capacidade geral.
O projeto original permaneceu intacto, com trabalho isolado em D:.

## Implementação efetiva

`funcoes_prospectivas.py` cria funções que distinguem histórias observadas
quando a mesma ação produz consequências que a representação anterior
confunde. Usa predição cruzada por episódio e alvos de construção:
continuação multiplicada pelas funções da história seguinte. Não usa nomes
de objetos, regiões, velocidades ou regras de cada jogo.

As propostas são comparações afins com tanh e produtos de duas comparações,
com pares, coeficientes e origem guardados. O alfabeto é uma família limitada
escolhida por nós. São consequências de **um bloco**; o modelo tenta compor
esses efeitos em sequências. Não inventa ainda um repertório arbitrário de
perguntas de múltiplos passos. Dar alvos futuros no treino não é disponibilizar
histórias futuras como entrada ao decidir na avaliação.

O corpo de cada programa usa operadores aprendidos sobre essas funções:
`v_empty=0; v_aQ=r_a+M_a@v_Q; value=phi(history)@v_word`.
Corpos completos, parâmetros, árvores e eventos são exportados em NPZ/JSON
e JSONL. Até 512 testemunhas para proposta, selecionadas por episódio/ação
sem consultar resultados; todos os dados válidos de construção para ajuste
dos operadores. Limite de 24 comparações, 8 produtos e 4 funções admitidas.

A direção geral tem precedentes: representar estado por previsões de
observações condicionadas a ações em
[Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf),
e aprender dicionários de observáveis para aproximar dinâmica em
[EDMD com aprendizado de dicionário](https://arxiv.org/abs/1707.00225).
Este protótipo não é implementação integral desses métodos nem demonstra
uma técnica inédita mundialmente. A contribuição investigada aqui é a
proposta por contrastes prospectivos, conectada ao executor deste projeto.

## Separação e suficiência do instrumento

Fontes, programas, orçamento e critérios foram congelados antes da coleta:
construção149, validação151, avaliação157. Oito episódios e 32 situações
por banco. Quatro programas comuns entre situações em construção/validação;
na avaliação, os mesmos quatro e quatro composições reservadas. Guardam-se
todos os estados sensoriais intermediários, recompensa e término.

Os corpos foram congelados **antes de coletar157**. Não houve ajuste após
essa avaliação. Sementes novas, mas dois jogos conhecidos: não demonstrar
aprendizagem de um mundo nunca visto pelo desenvolvimento.

A falha anterior de cobertura foi removida neste instrumento. Ambos os pares
de programas preferem lados opostos em situações diferentes. A validação tem
contraste nos oito episódios, com34comparações não empatadas emBreakout e27
emPong. A avaliação também satisfaz cobertura nas quatro células:35/33
comparações não empatadas emBreakout/conhecidos/novos;26/18 emPong. Portanto,
esta rodada falha com cobertura suficiente; não é inconclusiva por empate.

## Escolha em validação e resultado reservado

EmBreakout, o gerador prospectivo não teve função escolhida. O gerador por
recompensa escolheu a função11. EmPong, o prospectivo escolheu7e13, o controle
com alvos futuros embaralhados escolheu9e21 e o aleatório escolheu10e2.
São escolhas de desenvolvimento, não promoção de biblioteca geral.

Na validação dePong, o arrependimento cai0,390625→0,140625 com o prospectivo;
controle embaralhado chega0,203125 e aleatório0,15625. Essa melhora precisa
sobreviver fora da validação. Os dados reservados mostram:

| Avaliação157,16blocos,64comparações | Prospectivo | Sem funções novas | Alvos futuros embaralhados | Direções aleatórias | Zero, empate justo |
|---|---:|---:|---:|---:|---:|
| Breakout, programas conhecidos |0,125|0,125|0,125|0,125|0,304688|
| Breakout, composições novas |0,296875|0,296875|0,296875|0,296875|0,304688|
| Pong, programas conhecidos |0,234375|0,296875|0,234375|0,1875|0,320313|
| Pong, composições novas |0,25|0,265625|0,265625|0,28125|0,226563|

Valores são arrependimento médio: menor é melhor. EmBreakout, os empates com
a base ocorrem porque não há funções prospectivas no corpo. Não atribuir
seu desempenho à geração. EmPong/conhecidos, as novas funções melhoram a
base e a sua remoção piora o resultado, mas o aleatório é melhor. EmPong/
novos, a diferença frente à base é pequena e o controlezero é melhor.

O erro quadrático também não vence todos os controles. EmPong/conhecidos:
prospectivo1,233336, embaralhado1,215972 e aleatório1,190111. EmPong/novos:
prospectivo1,065056 contra0,982946 sem contexto e0,921875 zero.
Logo, nenhuma célula passa o portão completo. Não relaxar o critério,
ignorar controles ou elevar um resultado local ao rótulo de capacidade geral.

## Diagnóstico posterior: leitura e transporte

Depois da avaliação, congelou-se um diagnóstico separado de leitura:
`diagnosticar_prospectiva_composicao.py`. Zero ajustes, zero chamadas ALE,
todos os contextos das duas partições, três braços e horizontes4/8/16.
Decomposição telescópica com recompensa, transporte, término e termos cruzados.
Previsões reexecutadas independentemente para a frente.

| Primário,16blocos | Decisão composta | Todas as histórias futuras reais, recompensa aprendida | Remoção ideal do erro de recompensa |
|---|---:|---:|---:|
| Breakout, conhecidos |0,125|0,078125|0|
| Breakout, novos |0,296875|0,203125|0,015625|
| Pong, conhecidos |0,234375|0,328125|0|
| Pong, novos |0,25|0,390625|0|

As duas últimas colunas usam informação futura ou respostas verdadeiras:
**oráculos de diagnóstico, não capacidades do agente**. EmPong, entregar
as histórias futuras reais piora a decisão. A leitura aprendida de recompensa
não generaliza adequadamente, mesmo sem exigir que o modelo gere essas
histórias. EmBreakout, a informação futura ajuda, mas deixa erro.

Retirar idealmente erro de recompensa reduz quase todo o arrependimento;
isso não prova que simplesmente reajustar seus pesos resolveria. A intervenção
usa respostas verdadeiras. Os termos cruzados mostram cancelamento entre
erros; não atribuir parcelas percentuais de culpa independente. A conclusão
é localizada nesses leitores, corpos e dados, não uma causa única deAGI.

## Contabilidade e verificação

- Nove testes novos passaram: cinco do gerador e quatro da bancada.
- Seis bancos coletados, oito replays integrais exatos por banco:48replays.
- Auditoria recontou470ajustes de candidatos e6.144comparações reservadas.
- Contabilidade física reconciliada:160.438chamadas ALE. Com salto de quatro
  quadros, são641.752quadros simulados, somando todas as ramificações. Não
  confundir o tempo de cálculo do PC com poucos minutos de experiência humana.
- Construção usa47.004chamadas; validação44.394; avaliação69.040. Experiência
  é compartilhada entre os braços, mas trabalho computacional de proposta difere.
- Coleta durou cerca de105segundos no total; ajuste principal15,05segundos;
  diagnóstico posterior4,07segundos. Cada coleta ficou abaixo de120segundos.
- Maior RSS amostrado do coletor:183.955.456bytes, cerca de175,43MiB. Esse
  monitor é amostrado e não é contenção rígida ou limite de processos externos.
- Diferença máxima de funções independentes7,63e-14 e de previsão1,51e-14.
- Diagnóstico produziu2.304registros; discrepância telescópica9,33e-15 e
  de reexecução9,77e-15. O veredito permanece `capability_gate_failed`.

Protocolo e comandos em`PROSPECTIVA.md`; resultado completo
`resultados/prospectiva.json`, auditoria`resultados/prospectiva_auditoria.json`.
Decomposição em`resultados/prospectiva_composicao.json` e seus eventos.
Fontes, parâmetros e bancos são identificados por hashes. Os resultados são
exclusivos: reexecutar comandos que salvam novamente não sobrescreve registros.

## Implicação para o próximo trabalho

O gerador prospectivo funciona como executor de propostas e encontra funções
com efeito em parte dos dados. Não demonstrou vantagem geral sobre os controles
nem resolveu o leitor de consequências. A análise anterior não podia separar
isso da cobertura insuficiente; o banco novo permite essa distinção.

A próxima hipótese deve atacar como o dado sensorial vira uma descrição
executável que preserva eventos relevantes em situações e composições novas.
O alvo deve incluir a generalização da leitura de consequências, além do
fechamento dinâmico das funções. Apenas prever funções RBF da imagem seguinte
não mostrou essa capacidade. Perguntas de múltiplos passos, invariância,
continuidade e reutilização entre mundos continuam questões abertas para o
projeto, não soluções já disponíveis.

Ainda faltam demonstrações de invenção de representação em mundos desconhecidos,
eficiência comparável à humana, transferência de funções entre domínios e
investigação autônoma de fronteira. Esta rodada não estima proximidade humana
e não reduz esses requisitos a uma única peça suficiente.

## Correção posterior sobre o tempo da leitura — etapa 51

O diagnóstico acima que usa histórias futuras reais ainda calculava
`phi[t] @ reward[action]` antes de cada bloco. Não era reconhecimento a partir
de observações antes/depois. A inferência de que o sistema não conseguia ler
consequências já ocorridas era ampla demais. Fontes e números permanecem
preservados. `RESULTADO_EVENTOS.md` separa as operações: reconhecimento melhora
localmente, mas o campo escalar testado falha no planejamento; transportar o
leitor de transição pelo mesmo projetor linear reproduz o preditor antigo.
