# Pré-registro — a máquina inventa seus instrumentos de medida

Escrito antes de qualquer execução do mecanismo. Proposta trazida pelo usuário
(dois textos de 25/09): "quando duas situações que considero iguais respondem de
maneiras diferentes à mesma experiência, minha representação está incompleta; a
experiência que revela essa diferença fornece o material para construir a variável
que falta" — z_e(h) = E[φ_e(observação futura) | h, executar e].

## O que já existe e o que seria novo (dito antes)

Estado como previsão de experiências: Predictive State Representations. Achar a
distinção que falta por contraexemplo: aprendizado por consultas (Angluin, 1987).
Previsões auxiliares ajudando RL em Atari: SPR, EfficientZero, BBF. **Novo seria:**
aprender JUNTOS o sensor, a experiência e a variável, a partir dos próprios
contraexemplos, em pixels crus, com orçamento humano, num PC, e isso servir para
DECIDIR e em jogos NUNCA vistos com o código congelado.

## Mundos

ALE 0.12.1, frame skip 4. Ações ANÔNIMAS (índices do conjunto mínimo). Na coleta,
ações determinísticas (sticky 0) — necessário para ramificar da mesma situação; na
avaliação de pontuação, sticky 0,25 (padrão contra decorar sequências).
Desenvolvimento: Breakout, Pong. **Lacrados** (`LACRE_jogos.json`, sorteio com
semente, sha256 D5F6673B…): Alien, Frostbite, Private Eye, Road Runner — só rodam
no fim, com o código congelado (hash dos arquivos registrado antes).

Orçamento por jogo e por braço: **100.000 passos de agente** (Atari 100k), contando
TODOS os passos das ramificações. Régua externa: pontuação normalizada pelo humano
(HNS), tabela do BBF (Schwarzer et al., 2023, Tab. A.1).

## O que o aprendiz recebe

Imagem (luminância 42×32 por média de blocos), recompensa, fim de episódio, ações
anônimas, e a capacidade de SALVAR/RESTAURAR a situação durante a coleta (como
repetir um experimento de laboratório). Não recebe: nome do jogo, vidas (RAM),
objetos, coordenadas. Pressupostos declarados: médias por bloco, diferença entre
instantes, decomposição de baixo posto, árvores e regressão linear.

Representação base r0(h): quadro atual 21×16 ⊕ diferença para o anterior 21×16
(672 números).

## Experiências

Programa = segmentos (ação, repetições), horizonte 8 passos. Biblioteca inicial L0:
repetir cada ação 8 vezes; e pares (a por 4, b por 4) — até 16 pares sorteados.
Uma SONDA: a cada 16 passos de jogo, salva a situação, executa duas experiências a
partir dela (restaurando entre elas), registra o desfecho de cada uma e continua.

## Os três braços (mesmo orçamento, mesma coleta, mesmo decisor)

- **U**: experiências sorteadas de L0; representação r0.
- **C**: experiências escolhidas onde o modelo atual está CONFUSO (maior erro
  esperado de previsão do desfecho dado r); biblioteca L0 fixa; representação r0
  fixa; sensores fixos.
- **I**: como C, e além disso:
  - SENSORES aprendidos: decomposição não-negativa (NMF) das diferenças |desfecho
    de e_i − desfecho de e_j| da mesma situação; φ_k = mudança dentro do padrão k
  - EXPERIÊNCIAS construídas a partir das que contradisseram o modelo (encurtar,
    estender com espera, combinar segmentos de duas contraditórias)
  - VARIÁVEIS: z_{e,k}(r0(t)) = antecipação (árvores) de φ_k após executar e,
    admitida se R² ≥ 0,1 em sondas retidas; entra na representação (r0 ⊕ z)
- **I-placebo**: como I, mas a identidade da experiência é embaralhada entre os
  registros antes de aprender sensores e variáveis (ação ↔ consequência destruída).

Decisor igual para todos: fitted Q-iteration com árvores extremas sobre a
representação do braço, com todas as transições reais (inclusive as das
ramificações), recompensa cortada em ±1, γ = 0,97; ε decai de 1 a 0,1 na coleta.

## Medidas

- **Antecipação** em episódios NOVOS (política aleatória, sem aprender): 200
  situações; em cada uma, duas experiências de L0; desfecho numa medida NEUTRA
  comum (diferença do quadro 21×16 após 8 passos + recompensa). Cada braço prevê com
  sua representação (regressão treinada nas próprias sondas). Erro quadrático médio.
- **Decisão**: 5 episódios de avaliação (sticky 0,25, ε = 0,05, teto 10.000 passos
  cada — abaixo dos 27.000 do padrão, por tempo; declarado), pontuação → HNS.
- **Aparência trocada**: avaliação repetida com os canais de cor permutados.

## Critérios

| código | pergunta | exigência |
|---|---|---|
| I0 | ela inventa algo? | I admite ≥ 2 variáveis nos dois jogos de desenvolvimento; I-placebo admite ≤ 1 |
| I1 | antecipa a distinção em episódios novos? | erro de antecipação de I ≤ 0,85 × C nos dois jogos; I-placebo NÃO fica ≤ 0,95 × C |
| I2 | melhora decisões? | HNS de I ≥ HNS de C + 0,05 e ≥ HNS de U + 0,05 nos dois jogos |
| I3 | vale em jogos nunca vistos? | nos 4 lacrados, código congelado: I1 (≤ 0,85 × C) em ≥ 3 de 4 E HNS de I ≥ HNS de C em ≥ 3 de 4 |

Régua externa (sem portão, por honestidade): HNS de cada braço ao lado de DER, SPR,
EfficientZero e BBF.

## Previsão honesta

I0 0,70 · I1 0,40 · I2 0,20 · I3 0,15. Em pontuação absoluta, espero todos os
braços com HNS perto de 0 na maioria dos jogos — muito abaixo dos sistemas
publicados (DER tem IQM 0,18; BBF 1,05). O decisor por árvores é o elo mais fraco:
se I2 falhar com I1 passando, a leitura é "antecipa, mas o decisor não aproveita",
não "a aposta falhou".

Uma semente por jogo e braço (tempo); se algo passar, 3 sementes antes de afirmar.
Limites: ≤ 2 processos, ≤ 3 GB.

## Fumaça (Breakout, 20.000 passos) e um conserto antes do piloto de desenvolvimento

Dois defeitos de execução corrigidos (a árvore de vizinhos e o decisor ficavam
com a dimensão antiga quando nasciam variáveis; o decisor agora guarda quantas
variáveis conhece — elas só se acumulam). Observado, sem mudar critérios:
- o braço I criou 12 sensores e 8 variáveis; os sensores caíram na faixa de BAIXO
  da tela, em posições horizontais diferentes (onde agir faz diferença) — sem nome;
- antecipação e pontuação iguais entre os braços com 1/5 do orçamento;
- **construção de experiências: ZERO** — com 4 ações e cortes 4+4, toda variante já
  existia em L0. Conserto antes do piloto: variantes com cortes 2/6 e 6/2 (prefixo,
  espera antes, costura), que L0 não contém.
Critérios, limiares e previsões: inalterados.

## RESULTADO DO PILOTO DE DESENVOLVIMENTO (hash do código 27DA4543…)

| jogo | braço | antecipação (EQM) | HNS | variáveis | experiências |
|---|---|---|---|---|---|
| Breakout | U | 0,0002689 | +0,024 | 0 | 16 |
| Breakout | C | 0,0002689 | +0,017 | 0 | 16 |
| Breakout | I | 0,0002644 | −0,024 | 16 | 39 |
| Breakout | P | 0,0002725 | −0,024 | 16 | 40 |
| Pong | U | 0,0003815 | +0,008 | 0 | 22 |
| Pong | C | 0,0004105 | −0,008 | 0 | 22 |
| Pong | I | 0,0004036 | +0,025 | 16 | 34 |
| Pong | P | 0,0004059 | −0,008 | 16 | 34 |

**VEREDITO: I0 FALHA (o placebo também admite 16 variáveis) · I1 FALHA (I/C =
0,983 nos dois) · I2 FALHA (HNS I−C = −0,042 e +0,034) · I3 NÃO RODADO** — os
lacrados ficam guardados para uma versão que passe no desenvolvimento (um lacrado
só serve uma vez).

Leitura:
1. Os sensores inventados caíram onde a ação age DIRETO: faixa de baixo no
   Breakout, coluna da direita em alturas diferentes no Pong — a própria raquete.
   Achou o "eu", não a relação (a bola, "posicionar aqui faz a bola voltar").
2. A admissão aceitava prever o desfecho de UMA experiência — majoritariamente a
   dinâmica que ocorre de qualquer jeito; por isso o placebo inventa tanto quanto.
   A hipótese é sobre o CONTRASTE entre experiências da mesma situação; o desenho
   não a cobrava. Defeito de desenho, não do princípio.
3. Experiências de 8 passos (~0,5 s) não alcançam consequências atrasadas — risco
   que a proposta já apontava.
4. Decisor fraco (tudo ≈ aleatório), mas não é o primeiro gargalo: I1 falhou antes.
Previsão pré-registrada era I0 0,70 · I1 0,40 · I2 0,20: tudo falhou.
