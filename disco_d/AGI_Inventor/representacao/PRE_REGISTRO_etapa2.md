# Pré-registro — Etapa 2: Breakout observacional (Atari-HEAD)

Escrito antes de rodar. Segunda etapa da aposta "representações que a própria
experiência corrige". Aqui não há grandeza conservada óbvia: o que a proposta pede
é **explicar mudanças de movimento** — quando a previsão por inércia de uma
trajetória falha, construir estados RELATIVOS entre as entidades e aprender
condições sobre eles. Nenhum ramo do aprendiz sabe o nome do jogo.

## Dados

Atari-HEAD, `breakout.zip` (17 sessões em ordem de nome). Já tocadas antes: 0, 1,
2, 3. Três trios NOVOS (ajuste / seleção / avaliação):
A = 4 / 5 / 6 · B = 7 / 8 / 9 · C = 10 / 11 / 12.
Orçamento: ajuste = primeiros 2.400 quadros da sessão; seleção = 1.200; avaliação
= 3.000 quadros, sem atualizar nada.

**Leitor corrigido** (os dois defeitos apontados na proposta): TODOS os quadros
são mantidos (inclusive os de ação nula); o tempo é o índice do quadro (tempo de
jogo — a duração em ms no arquivo é o tempo de decisão humana, não física). O
estado usado para PREVER é **causal**: só o passado disponível. O olhar humano não
é lido.

## Percepção

O código da Etapa 1, **sem mudar limiares**: fragmentos por cromaticidade,
rastreio pelo vizinho mais próximo (≤ 6 px), e o revisável com compor / cortar /
religar (e religar usando q, se a descoberta achar algo). Única adição: cada
fragmento e trilha carrega sua caixa envolvente (necessária para medir contato).
CONGELADO × REVISÁVEL, a volta do revisável escolhida na SELEÇÃO por λ_sel, como
na Etapa 1.

## Relações (`relacoes.py`)

Para cada entidade em movimento com estado causal no quadro t e posição observada
em t+1: resíduo da inércia r = pos(t+1) − (pos(t) + v(t)). Duas árvores de
regressão (profundidade 8, folha ≥ 10) prevêem r:
- **INDEPENDENTE**: só o estado da própria entidade (u, v, u̇, v̇)
- **RELACIONAL**: o mesmo + as 2 entidades mais próximas em t (pela caixa): vetor
  até o ponto mais próximo da caixa delas e a velocidade delas

"Intervalo de mudança de movimento" = amostras de avaliação com |r| > 1,5 px.

## Critérios (com tamanho mínimo de efeito em todos)

| código | pergunta | exigência |
|---|---|---|
| E2-rel | a relação é útil? | no revisável, erro nos intervalos de mudança: RELACIONAL ≤ 0,80 × INDEPENDENTE, e IC95 (bootstrap por blocos de 50 amostras) da diferença acima de zero, em ≥ 2 de 3 trios |
| E2-rev | a revisão perceptiva importa? | alvos comuns = REGIÕES conexas não-fundo (qualquer cor) em t+1 cuja região mais próxima em t está a 0,5–8 px; erro = distância à previsão mais próxima do sistema (teto 8 px). REVISÁVEL ≤ 0,90 × CONGELADO e IC95 acima de zero em ≥ 2 de 3 |
| E2-ctl | controle | relações embaralhadas entre amostras (no ajuste e na avaliação): ganho do RELACIONAL sobre o INDEPENDENTE ≤ 5% em ≥ 2 de 3 |

Leitura (não é critério): importância das variáveis na árvore relacional, e qual
entidade (cor, posição média) é mais vezes a parceira nos intervalos de mudança.

## Previsão honesta

E2-rel 0,40 (a posição absoluta já ensina as paredes à árvore INDEPENDENTE; só a
raquete e os tijolos que somem exigem relação); E2-rev 0,45; E2-ctl 0,85.

## Orçamento

Fumaça nas sessões 0 / 1 / 2 (já tocadas) com orçamento pequeno; piloto ≤ 60 min;
≤ 2 processos; ≤ 3 GB.

## Ajuste antes de rodar qualquer coisa

O alvo comum do E2-rev era "fragmentos"; trocado por "regiões conexas não-fundo"
antes de qualquer execução, porque o alvo por fragmento penaliza o `compor` por
construção (a raquete composta tem um centróide; cada borda borrada sozinha fica
a px dele). Se nenhuma grandeza for achada em nenhuma volta, o revisável usa a
volta 1 (compor/cortar/religar por continuidade).

## Fumaça (sessões 0/1/2, já tocadas) e dois consertos de INSTRUMENTO antes do piloto

Fumaça: E2-rel razão 0,99; E2-rev razão 0,99; importância das relações ≈ 0. Motivo
achado: (1) a "distância até a caixa" mente para formas côncavas — a parede do
Breakout é um U, a bola fica sempre DENTRO da caixa dela, e a parceira mais próxima
era sempre a parede a distância 0. Conserto: vetor até o PIXEL de borda mais
próximo do outro (até 1024 pixels de borda por fragmento). (2) o alvo comum do
E2-rev contava regiões estáticas cujo centróide anda quando um tijolo some.
Conserto: alvo só se a região se TRANSLADOU inteira (os cantos da caixa andam com o
centróide, tolerância 1,5 px). Critérios, limiares e previsões: inalterados.

## Segunda fumaça, e a AÇÃO como observação (antes do piloto)

Com o contato pela borda, a fumaça continuou nula (razão 1,02). A decomposição das
amostras de mudança mostrou por quê: ~59 das ~106 mudanças são a RAQUETE parando e
partindo — causadas pela ação do jogador, que nenhum dos dois modelos via. A
proposta já previa: "nos vídeos, as ações humanas servirão como observações".
Mudança: os DOIS modelos (independente e relacional) recebem a ação observada no
quadro t (nula = −1, mantida). O controle embaralha só as colunas relacionais
(a ação fica). Critérios, limiares e previsões: inalterados.

## Terceira fumaça, e a quebra que a revisão acha passa a valer para o estado

Com a ação: razão 0,97, ainda nulo; importância de `dv` = 0. Causa: o `cortar`
DETECTA o quique (passo foge da extrapolação) e o `religar` junta as metades de
volta — mas a junção era esquecida, e o estimador de velocidade atravessava o
quique (≈3 quadros de velocidade média antes/depois, artefatos que nenhuma relação
explica). Mudança: o `religar` registra os pontos de junção e o ESTADO CAUSAL
(usado só na etapa 2) não os atravessa. A etapa 1 não usa o estado causal: seus
resultados não mudam. Critérios, limiares e previsões: inalterados.

## Última mudança antes do piloto: o modelo de EVENTO que a proposta descreve

Quarta fumaça ainda nula (0,97); `dv` sem importância. Olhando quadro a quadro: a
bola anda (±2, ±1) px/quadro, um quique é resíduo de 2 px, e há ~20 quiques em
1.200 quadros no meio de ~2.000 amostras sem mudança — uma árvore de regressão
sobre TODAS as amostras aprende "nada muda". Eu tinha simplificado demais a
proposta, que diz: "reunir os estados anteriores desses eventos e compará-los com
transições sem falha, construir estados relativos e ajustar condições". Agora: um
classificador falha × não-falha (eventos pesados por igual, profundidade 6, folha
≥ 5) e um regressor do efeito treinado SÓ nos eventos (profundidade 4, folha ≥ 3);
previsão = efeito se P(falha) > 0,5. Vale para o independente E o relacional.
Esta é a última mudança: o piloto roda a seguir qualquer que seja a fumaça.
Critérios, limiares e previsões: inalterados.

## RESULTADO DO PILOTO (trios A, B, C — sessões nunca tocadas)

| trio | E2-rel razão (mudança) | E2-rev razão | volta escolhida na seleção | parceira dominante nos eventos |
|---|---|---|---|---|
| A (4/5/6) | 1,00 | 1,01 | volta1 | a parede em U |
| B (7/8/9) | 1,13 | 1,00 | **volta0** | tijolos / parede |
| C (10/11/12) | 0,99 | 1,00 | **volta0** | a RAQUETE (159 de 280 eventos); classificador usa dx1, dy1 (0,42 + 0,50) |

**VEREDITO: E2-rel FALHA (0/3) · E2-rev FALHA (0/3) · E2-ctl passa (3/3, trivial).**

Leitura:
1. A revisão foi escolhida pelo critério da etapa 1 — λ de uma grandeza
   CONSERVADA na seleção. No Breakout não há conservação que importe para a tarefa,
   e em 2 de 3 trios a seleção escolheu NÃO revisar. O julgamento da revisão tem de
   ser o benefício no próprio mundo (erro de previsão), como a proposta dizia
   ("julgada em outra parte"); eu transportei o critério errado.
2. Eventos raros: 36–280 mudanças em ~2.600 transições. As variáveis relacionais
   mais sobreajustam do que ajudam.
3. No trio C o classificador de eventos se apoiou no vetor até a parceira mais
   próxima, e a parceira dominante foi a raquete — pista de relação, mas sem compor
   (volta0), parte disso são as bordas borradas da própria raquete. Não se afirma.
4. Sem verdade de identidade no Atari-HEAD, as substituições diagnósticas não
   rodam aqui; no ALE (etapa 3) a RAM do jogo dá posição de bola e raquete.
