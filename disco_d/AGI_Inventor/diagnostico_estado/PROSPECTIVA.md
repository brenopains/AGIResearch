# Distinções geradas por consequências futuras

O objetivo continua sendo representação geral, eficiência de aprendizagem,
transferência e investigação autônoma. Esta bancada examina uma hipótese
necessária para o próximo mecanismo, sem redefinir inteligência humana como
sucesso em dois jogos. Não garante AGI nem novidade científica mundial.

## Mecanismo

O gerador recebe histórias sensoriais, ações anônimas e consequências
observadas. Suportes RBF vêm de histórias de construção. As perguntas são
os valores dessas funções na história seguinte, multiplicados pela
continuação real; a coordenada constante mede continuação. Não recebe nomes
de objetos, velocidades, regiões, regras ou conceitos do mundo.

Predição cruzada deixa um episódio inteiro fora por vez. Para a mesma ação,
pares de episódios distintos são ordenados pela diferença dos resíduos
prospectivos, descontada a distância da representação atual. Cada par
produz uma comparação afim com tanh sobre a história bruta. Também se
propõem produtos antes de aprovar suas partes. Até 24 unidades, 8 produtos
e 4 coordenadas novas. É uma linguagem finita escolhida por nós: não é
invenção de programas arbitrários. As respostas futuras são alvos observados
de construção; não se tornam entradas disponíveis antes de agir no teste.

Propostas usam no máximo 512 testemunhas, amostradas por episódio/ação em
índices uniformemente espaçados, sem consultar recompensa ou consequência.
Os operadores finais usam todas as transições válidas de construção.
Registrar também aliases brutos, origem de cada par e custo de ajuste.

## Separação antes da coleta

Fontes, programas e critérios congelados antes de coletar novos dados.
Construção em 149, validação em 151 e avaliação em 157, nos dois mundos
conhecidos. Cada banco tem oito episódios e quatro situações por episódio,
nos pontos 0, 2, 4 e 6 de uma caminhada fixa de 128 chamadas por ponto.

Quatro programas anônimos comuns às situações em construção/validação;
esses quatro e quatro composições novas na avaliação, todos com 16 blocos
de oito chamadas ALE. Guardar histórias intermediárias de 16 quadros
comprimidos, recompensas e continuação. Todas as ramificações contam como
experiência: chamadas com salto de quatro quadros não são quatro vezes
menos tempo físico. Clonar restaura dinâmica; a história inicial vem da
caminhada real, não do cache visual após outro programa. Reexecutar o
primeiro programa em um ponto por episódio e conferir trajetória inteira.

Bases/dicionários congelados antes das chamadas de seleção. A validação
pode já ter sido coletada, mas não participa da geração ou ajuste de seus
parâmetros. Corpos selecionados congelados **antes da coleta de 157**.
Nenhum resultado de 157 realimenta geração, seleção ou configurações.
As sementes são novas nesta bancada; os jogos, objetivos e decisões de
desenho já são conhecidos. Não chamar isso avaliação de mundos desconhecidos.

## Controles e seleção

Gerador prospectivo; futuras respostas vetoriais embaralhadas dentro da
ação; gerador anterior por recompensa; direções aleatórias. Todos usam
a mesma linguagem/tamanho e recebem as mesmas observações de construção.
Operadores sempre ajustados sobre consequências verdadeiras; embaralhamento
intervém apenas na proposta. Custo computacional real dos geradores difere
e deve ser reportado, sem alegar orçamentos de trabalho idênticos.

Referências: nenhuma função nova, remoção integral das coordenadas novas
sem modificar o bloco antigo, remoção apenas do contexto inicial e zero
com desempate justo. Dimensão dos modelos permanece 37, com posições
inutilizadas preenchidas com zero.

Verificar cobertura em validação antes da seleção: cada um dos dois pares
de programas tem de preferir ambos os lados em situações distintas, e deve
haver contraste em pelo menos três episódios. Sem isso, manter a base e
declarar instrumento inconclusivo, sem escolher casos por respostas.

Reutilizar o seletor congelado anterior: expansão numérica no treino,
melhora de arrependimento e erro quadrático na validação, com pelo menos
dois casos melhorados, até quatro escolhas gulosas. Caso significa comparação;
duas comparações de uma raiz não equivalem a episódios independentes. Não
apresentar escolha de desenvolvimento como promoção de biblioteca geral.

## Portão antes da avaliação

Quatro células: os dois jogos, com programas conhecidos e novos. Todas
precisam ter cobertura contextual suficiente, uma função primária escolhida,
menor arrependimento e menor erro quadrático que **todos** os controles em
16 blocos. Avaliar também 4/8 blocos sem trocar o horizonte principal.
Falta de cobertura é inconclusiva; comparação suficiente sem ganho falha.
O portão completo só passa se as quatro células passarem. Nem passar
demonstra transferência de um corpo entre domínios ou mente de nível humano.

## Recursos, evidências e comandos

Um worker por vez, CPU/BLAS com duas threads. Teto 50.000 chamadas ALE e
120 segundos por banco, 120 segundos de cálculo de ajuste. Coletor verifica
seu RSS com teto 6 GiB e registra máximo amostrado; não é contenção rígida
por sistema operacional nem limite global de outros processos do usuário.
Chunks por raiz e checkpoints de contagem não permitem retomar exatamente
o estado ALE: interrupção permanece inconclusiva, com custo anterior salvo.
Não ampliar parâmetros ou orçamento para salvar a rodada.

Eventos reais de coleta, geração, rejeição, escolha e avaliação em JSONL,
corpos completos em NPZ/JSON e hashes de todas as fontes/dados/parâmetros.
Auditor verifica contabilidade/replays registrados, regenera propostas,
reajusta por Cholesky, reexecuta previsões para a frente e confere portões.

```powershell
cd D:\AGI_Inventor\diagnostico_estado
py -3.11 -B testes_funcoes_prospectivas.py
py -3.11 -B testes_bancada_prospectiva.py
py -3.11 -B piloto_prospectivo.py prepare
py -3.11 -B coletar_prospectiva.py breakout 149
py -3.11 -B coletar_prospectiva.py pong 149
py -3.11 -B coletar_prospectiva.py breakout 151
py -3.11 -B coletar_prospectiva.py pong 151
py -3.11 -B piloto_prospectivo.py fit
py -3.11 -B coletar_prospectiva.py breakout 157
py -3.11 -B coletar_prospectiva.py pong 157
py -3.11 -B piloto_prospectivo.py evaluate
py -3.11 -B auditar_prospectiva.py
```

Resultados não são sobrescritos. Projeto original em C: permanece intacto.
Este protocolo testa uma peça de percepção/composição, não substitui
inventar representações de mundos desconhecidos, aprender entre modalidades
e conduzir pesquisa em problemas de fronteira autonomamente.
