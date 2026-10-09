# Pré-registro — "tornar-se X": o ponto de vista como operação da mente

Escrito em 08/10/2026, antes de rodar os mundos de teste. Ideias do usuario de 08/10
(corpo virtual, mundo interno, autoprojeção: "o cientista que se sente a célula";
"talvez inteligência geral seja mudar a posição a partir da qual o mundo é modelado").

## O que já existe (dito antes)

Heider e Simmel (1944): pessoas atribuem intenção a pontos que se movem. Teoria da
mente como planejamento inverso (Baker, Saxe e Tenenbaum, 2009); Machine Theory of
Mind (Rabinowitz et al., 2018). Regras de cardumes inferidas no referencial do
peixe focal (Katz et al., 2011) — ali o CIENTISTA escolhia o referencial. SciNet
(Iten et al., 2020) achou a representação heliocêntrica. Regressão esparsa de leis
(SINDy, 2016; nossa própria linha de MDL).

**Novo seria:** a PRÓPRIA mente escolher, por compressão, de QUEM é o ponto de vista
que torna cada ser simples — o próprio corpo dele, o corpo de outro ser, a arena —
contra quadros-placebo sem dono; e UMA operação servir tanto para física (mola,
ímã, satélite) quanto para intenção (persegue, foge, guarda posição no corpo do
líder), dando de graça "quem persegue quem" e uma taxonomia (objeto × agente ×
ligado a outro) que ninguém nomeou.

## Mundo e mente

`mundo.py`: arena 2D com 5 seres; a mente recebe só posições com ruído (σ = 2·10⁻⁵),
24 s a 20 Hz. Tipos de desenvolvimento: mola, ímã, persegue, foge, satélite,
formação. Reserva (nunca vistos no desenvolvimento): intercepta (mira onde o alvo
vai estar), distância (aproxima de longe, afasta de perto).

`encarnar.py`: para cada ser X e cada ponto de vista (mundo, eu, ser Y, junto de Y,
2 placebos em passeio aleatório), átomos = só geometria do quadro (posição/
velocidade relativas, distância, 1/d, 1/d², direção, rapidez do dono do quadro);
lei = mínimos quadrados ortogonais com parada por bits, classe só-átomos ou
átomos+produtos (a de menos bits). Ponto de vista escolhido = menos bits.

O que é meu (declarado): a lista de pontos de vista (mundo/eu/ser/junto — mas é a
MESMA operação aplicada a todo ser), os átomos geométricos, produtos de grau 2.

## Teste (lacrado: sementes nunca usadas no desenvolvimento)

- T1: 40 mundos, tipos sorteados entre os 6 de desenvolvimento (sementes 1000–1039).
- T2: 20 mundos, cada um com intercepta e distância + 3 sorteados (sementes 2000–2019).
- Escolha do ponto de vista e da lei nos primeiros 75% do tempo; previsão da
  aceleração nos últimos 25% (nunca vistos).

| código | pergunta | exigência |
|---|---|---|
| A1 | descobre de quem é a ação? | nos seres com alvo (persegue, foge, satélite, formação), o ser mais citado na lei escolhida = o alvo verdadeiro em ≥ 80% (acaso: 25%) |
| A2 | o placebo engana? | ponto de vista placebo escolhido em ≤ 5% dos seres |
| A3 | o ponto de vista paga em previsão? | nos agentes (persegue, foge): mediana de R² no futuro (escolhido − só-mundo, mesmo motor) ≥ +0,10 e R² escolhido mediano ≥ 0,8 |
| A4 | taxonomia sem nome: | lei de mola/ímã NÃO cita outro ser em ≥ 85%; lei de persegue/foge/satélite/formação cita outro ser em ≥ 85% |
| A5 | atravessa para comportamentos nunca vistos? | em intercepta e distância (T2): alvo certo em ≥ 60% e R² no futuro escolhido − mundo ≥ +0,10 (mediana) |

Formação atrás de líder em movimento uniforme é ambígua nos dados ("ser eu" e "ser
o líder" giram juntos): A1 conta o alvo pela lei, não pelo nome do ponto de vista;
o ponto de vista será reportado como certo / equivalente (diferença < 20 bits) /
errado, descritivo.

## Previsão honesta

A1 0,80 · A2 0,85 · A3 0,70 · A4 0,70 · A5 0,50.

Leitura fixada antes: se A1–A4 passarem, uma única operação ("tornar-se X")
descobre leis físicas, intenções e alvos sem rótulo, e a taxonomia objeto/agente
emerge do ponto de vista que comprime. Isto é um degrau (mundo construído por mim,
com regras que eu escrevi), NÃO teoria da mente humana nem AGI. O passo seguinte é
o corpo da própria mente (aprender o próprio esquema corporal por contingência) e
usar ESSE modelo de si para entender os outros ("se eu fosse X").

## Correção de régua ANTES de abrir os mundos lacrados (achada no mundo de desenvolvimento 5)

R² no futuro engana quando o ser já parou: a mola em repouso tem aceleração = só
ruído, a mente erra exatamente o ruído do instrumento (0,0195 × σ = 0,0196) e o R²
dá 0,28. **A3 e A5 passam a usar erro em unidades do ruído conhecido** (1 = perfeito),
como o verificador do organismo: A3 = erro_mundo / erro_escolhido mediano ≥ 1,5× E
erro escolhido mediano ≤ 2,0 ruídos; A5 = alvo ≥ 60% E razão mediana ≥ 1,5×.
E no ponto de vista "ser Y"/"junto de Y" o ser citado inclui o PRÓPRIO Y (a relação
X–Y está no quadro, os átomos chamam-se "eu.*"); antes isso contava como "não citou".
Limiares de A1, A2, A4 inalterados. Previsões inalteradas.

## RESULTADO (60 mundos lacrados, 300 seres, 411 s) — `resultados/teste.json`

| código | resultado | exigência | |
|---|---|---|---|
| A1 alvo certo (T1) | **0,938** (n = 130; acaso 0,25) | ≥ 0,80 | PASSA |
| A2 placebo escolhido | **0,017** | ≤ 0,05 | PASSA |
| A3 agentes, erro mundo/escolhido | **1,30×** (erro escolhido 1,01 ruído; n = 72) | ≥ 1,5× e ≤ 2,0 | **FALHA** |
| A4 física sem citar outro / ligados citando | **0,929 / 1,000** | ≥ 0,85 cada | PASSA |
| A5 reserva (nunca vistos): alvo / razão | **0,70 / 1,98×** (n = 40) | ≥ 0,60 e ≥ 1,5× | PASSA |

Por tipo (erro em ruídos no futuro, escolhido × só-mundo; alvo certo):
persegue 1,00 × 6,92, alvo 1,00 · foge 1,01 × 1,09, alvo 1,00 · formação 1,45 × 5,42,
alvo 0,82 (ponto de vista certo 24, errado 15) · satélite 1,09 × 1,09, alvo 0,88 ·
mola 1,00 × 1,01 · ímã 1,01 × 1,01 · intercepta 1,69 × 9,93, alvo 0,55 ·
distância 3,44 × 23,97, alvo 0,85.

Leitura: A3 falha por causa dos FUGITIVOS — depois que viram as costas ao
perseguidor correm reto e constante, e no futuro não sobra nada que o ponto de
vista explique melhor (1,09× vs 1,01). Nos perseguidores o ganho é 6,9×; nos
comportamentos nunca vistos, 2× (intercepta 5,9×, distância 7,0×). Uma operação
só achou de quem é a ação em 94% (acaso 25%), a taxonomia objeto/agente/ligado
emergiu sem nome (93% / 100%), e o placebo quase nunca venceu (1,7%).
Degrau, em mundo construído por mim — não teoria da mente humana.
