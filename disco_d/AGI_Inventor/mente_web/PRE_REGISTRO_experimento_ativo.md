# Pré-registro — as MÃOS dela (transplante 11): experimento escolhido para separar rivais

Escrito em 09/10/2026, **antes** de rodar as sementes registradas (20261110 a 20261119).
Código congelado neste commit: `laboratorio.py` (seção 5, MÃOS), `arena.py` (`?exp=`),
`experimento_ativo.py`.

## De onde vem

A segunda revisão do GPT (09/10) propôs: além de assistir, ela deve **agir para decidir entre
teorias rivais** — fazer o experimento cujo resultado as rivais preveem de forma mais
diferente. Até o transplante 10 ela só assistia. A pergunta aqui é estreita e mensurável:

> Escolher ela mesma a cena (onde cada coisa começa e com que velocidade) para SEPARAR as
> explicações rivais faz ela prever melhor o mundo do que só assistir — e melhor do que
> mexer no mundo ao acaso?

## O mundo

A mesma arena do transplante 10 (`arena.py`): vídeos de 24 s, 5 quadros/s, 128×128, câmera
afim desconhecida por vida, oclusor, corpos da mesma cor, corpo bicolor. Uma VIDA:

| episódios de treino | mundo | lei | conceito útil |
|---|---|---|---|
| 0–9 | giro | a = +1,0·s²·rot90(v) | s² |
| 10–19 | mola_h | a = −0,09·h²·x | h |
| 20–29 | contragiro | a = −1,4·s²·rot90(v) | s² (reuso) |

Ao fim de cada mundo, **6 episódios de TESTE** novos daquele mundo, sorteados pelo mundo
(sementes [vida, 6000+...]), iguais para os três braços, previstos sem aprender com eles.

## As mãos (o que ela faz)

Pedido em coordenadas que ELA percebe (centro 0, meia-largura 1, por segundo):
`arena://k?exp=x,y,vx,vy;...` — a arena converte pela câmera que ela não conhece e roda a lei
daquela visita. O mundo limita: ≤ 4 corpos, posição no quadro, rapidez ≤ 0,6.

1. **Rivais** (`rivais`): a lei da teoria que explicou o último episódio da arena; a mesma
   teoria com cada outra opção da sala de espera (outro conceito, ou nenhum); e a mesma
   teoria re-ajustada em 4 reamostras das suas trajetórias.
2. **Candidatos** (`propoe_experimento`): 300 intervenções em situações que ela VIU (o estado
   de uma coisa num episódio da teoria atual, rapidez × 0,7–1,3, direção girada até 30° ou
   invertida com prob. 0,25, posição + ruído 0,05). Valem só os que: a teoria atual prevê no
   quadro por 6 s; ficam no quadro nos 3 s sob TODAS as rivais; os olhos dela acompanham
   (rapidez ≤ meio passo máximo de rastreio por quadro, medido pelo relógio dos vídeos dela);
   e 3 corpos que não se encostam (≥ 0,2) no caminho previsto.
3. **Escolha**: ATIVO = os 3 de maior discordância prevista entre as rivais (média das
   distâncias à atual em 1, 2 e 3 s, cada uma cortada em 64 px). ALEATÓRIO = os mesmos
   candidatos e regras, escolha ao acaso.
4. **Depois de ver** (`resolve_experimento`, nos TRÊS braços): bits pareados por trajetória de
   cada rival contra a atual; se vence uma rival de outra opção de conceito por mais de
   log2(nº de rivais), a sala de espera é refeita na hora.

## Braços (o ÚNICO que muda é quem escolhe o começo de cada episódio de treino)

- **PASSIVO** — o mundo sorteia (e só aceita corpos que ficam à vista os 24 s, pela lei
  verdadeira — uma curadoria que ela não tem).
- **ALEATORIO** — ela monta a cena ao acaso entre os mesmos candidatos (controle de "mexer no
  mundo", "3 corpos sempre", "outras rapidezes").
- **ATIVO** — ela monta a cena onde as rivais mais discordam.

O primeiro episódio de cada vida é do mundo nos três (sem teoria não há rivais).

## Medidas

- **erro de teste** (primária): para cada mundo, a teoria que explicou o ÚLTIMO episódio de
  treino prevê os 6 episódios de teste (2 s à frente, mediana entre trajetórias por
  episódio, mediana dos 6); por vida, a média geométrica dos três mundos (px em 128).
- **secundária, descritiva**: o mesmo com a teoria que guarda mais episódios de treino do
  mundo.
- **conceito certo no fim**: o conceito em uso na teoria de teste tem |corr| ≥ 0,99 com a
  grandeza certa (s², h, s²) em estados percebidos de episódios novos.
- **discriminação**: em cada episódio montado, bits entre a melhor e a pior rival
  (max − min dos bits pareados); por vida, a mediana.

## Critérios (fixados agora; 10 vidas)

| critério | passa se |
|---|---|
| **E1** escolher pela discordância vence escolher ao acaso | erro de teste ATIVO < ALEATORIO em ≥ 7/10 vidas |
| **E2** agir vence só assistir | erro de teste ATIVO < PASSIVO em ≥ 7/10 vidas |
| **E3** chega ao conceito certo mais vezes | (vida × mundo) com conceito certo no fim: ATIVO ≥ ALEATORIO + 3 (em 30) |
| **E4** (sanidade das mãos) os experimentos dela separam as rivais | discriminação ATIVO > ALEATORIO em ≥ 8/10 vidas |

## Checagens dev (sementes 101–106, separadas das registradas) — o que mudou e por quê

1. **dev101** (3 vidas): candidatos uniformes no quadro com rapidez de 0,75× o 5º a 1,25× o
   95º percentil do que ela mediu. Falha de projeto: a faixa crescia com o que ela mesma
   pedia — corrida até rapidezes que ela não rastreia (mais de 8 px por quadro); episódios
   sem nenhum objeto, e a mesma proposta repetida para sempre (o sorteio só dependia do
   estado do laboratório, que não mudava). Teste: PASSIVO 2,5 px, ALEATORIO 13,2, ATIVO 11,9.
2. **dev101b** (3 vidas): candidatos = intervenções em situações vistas; teto de rapidez pelos
   olhos dela; sorteio com contador de propostas. ALEATORIO 3,1 px ≈ PASSIVO 2,5; ATIVO 9,2:
   a maior discordância caía na borda do quadro e em corpos que se fundiam (até 16 trilhas
   num episódio de 3 corpos).
3. **dev101c** (6 vidas): + "à vista sob todas as rivais" e "não se encostam no caminho".
   Teste (mediana entre vidas): PASSIVO 2,35 px, ALEATORIO 4,17, ATIVO 4,74. ATIVO < ALEATORIO
   em 3/6; ATIVO < PASSIVO em 1/6; conceito certo no fim ATIVO 7, ALEATORIO 9, PASSIVO 11 (em
   18); discriminação ATIVO > ALEATORIO em 5/6.

Parei de ajustar aqui: três rodadas de dev bastam para tirar defeitos de projeto (corrida,
borda, fusão); continuar ajustando a regra até o ATIVO ganhar nas sementes dev seria
escolher o resultado.

## Previsões (honestas, a partir do dev)

- **E4 provavelmente PASSA** (~65%): as mãos fazem o que dizem — criam situações em que as
  rivais discordam de verdade.
- **E1 provavelmente FALHA** (~25% de passar): no dev, escolher pela discordância empatou com
  o acaso. A discordância entre reamostras é maior onde ela tem menos dados (as bordas do que
  já viu); o teste é a distribuição natural do mundo. É o viés conhecido da "amostragem por
  incerteza" em aprendizado ativo.
- **E2 quase certamente FALHA** (~5%): a curadoria do mundo (corpos à vista 24 s pela lei
  verdadeira; na mola_h, órbitas quase tangenciais) dá dados melhores do que qualquer cena
  que ela consegue montar sem conhecer a lei.
- **E3 provavelmente FALHA** (~10%).

Se E1 falhar, a consequência para a mente viva é direta: a ação "experimentar" NÃO entra
como regra fixa; entra como uma ação a mais na economia dela, paga pelo que reduz da própria
surpresa no laboratório — se não pagar, ela deixa de escolher.

## RESULTADO

(preenchido depois da rodada registrada, sem mudar nada acima)
