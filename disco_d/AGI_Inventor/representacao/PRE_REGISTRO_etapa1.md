# Pré-registro — Etapa 1: percepção revisável nas órbitas vistas por pixels

Escrito antes de qualquer execução. Proposta de origem: "aprender representações
que a própria experiência consegue corrigir" (trazida pelo usuário). Esta etapa
testa a hipótese central só no mundo observacional das órbitas; Breakout vem
nas etapas 2 (observacional, Atari-HEAD) e 3 (ativo, ALE).

## A hipótese

A descoberta de regularidades, quando pode **pedir revisão da percepção que a
alimenta** (reidentificar, dividir, compor, ligar lacunas), chega a uma grandeza
equivalente ao momento angular `h` a partir de pixels, e chega **melhor** do que
o mesmo sistema com a percepção congelada.

## O mundo (gerador do avaliador; o aprendiz não vê nada disto)

- Newton com GM = 1 (a mesma aceleração de `vendor/novaphys__orbita.py`), RK4 com
  passo 0,005 como em `inventa.py`. Condições iniciais como `inventa.trajetorias`:
  r ∈ [0,8; 2,5], velocidade 0,8–1,2 da circular, ângulo ±0,4 rad, 30% retrógradas.
- Cada episódio: **3 corpos** em torno de um centro **invisível**, 10 s de vídeo,
  30 quadros/s com **tempos reais com jitter de ±10%** (os tempos são entregues).
- Câmera **afim fixa por mundo** e desconhecida: escala 16–22 px/unidade, rotação
  aleatória, anisotropia 0,85–1,15, cisalhamento ±0,15, reflexão em 50%, centro na
  imagem ±10 px. Imagem 128×128 RGB, fundo preto, discos anti-aliased.
- **Armadilhas de percepção**, cada uma em 50% dos episódios, sorteadas:
  - T1 cruzamento: dois dos corpos com a MESMA cor (fundem-se quando se cruzam)
  - T2 corpo bicolor: um corpo pintado em duas metades de cores diferentes
  - T3 oclusão: um retângulo cinza estático de 24×24 px por cima
- Divisão por mundo: **ajuste 12 episódios (120 s)**, **seleção 6 (60 s)**,
  **avaliação 12** com **troca de aparência** (outra paleta, raio 3,5 px em vez de
  2,5, metades do bicolor em cima/baixo em vez de esquerda/direita).
- **3 mundos** (sementes 1, 2, 3), cada um com sua câmera.

## O que o aprendiz recebe

Quadros RGB, tempos, fronteiras de episódio. **Não** recebe: número de corpos,
centro, coordenadas, identidades, máscaras, `h`, a lei, nomes.

Primitivas declaradas (nossas): fragmento = componente conexo de mesma
cromaticidade quantizada; continuidade = vizinho mais próximo de mesma
cromaticidade com checagem ida-e-volta; agregação (centróide, área); diferença
entre instantes (ajuste polinomial local de 7 quadros com os tempos reais);
monômios de grau ≤ 2 do estado z = (u, v, u̇, v̇): q(z) = zᵀQz + bᵀz, 14 termos,
resolvidos **conjuntamente** (sem enumerar subconjuntos) por autovetor
generalizado (variância dentro das trajetórias / variância total). Entidades sem
movimento (mediana < 1 px/s) não entram na descoberta.

## Os dois sistemas (mesma percepção inicial)

- **CONGELADO**: fragmentos → trilhas pelo vizinho mais próximo; entidade =
  trilha. Nada é revisto.
- **REVISÁVEL**: parte do mesmo, e cada revisão nasce de uma evidência medida:
  - *compor*: duas trilhas coexistentes com deslocamento relativo constante
    (desvio ≤ 0,35 px em ≥ 10 quadros) e adjacentes viram uma entidade
  - *cortar*: um quadro onde a área salta ≥ 1,6× a mediana da trilha (fusão) ou o
    passo foge ≥ 3 px da extrapolação local é retirado e a trilha é cortada ali
  - *religar*: fins e começos de segmentos com lacuna ≤ 1,5 s são religados por
    atribuição ótima; custo = erro de extrapolação (px) e, **depois que q existe**,
    + 2·|Δq|/σ_q. Hipóteses alternativas (manter × trocar) competem pelo custo.
  - laço: percepção → descoberta de q → revisão usando q → nova descoberta (2
    voltas). Uma volta de revisão só é mantida se **melhorar a seleção** (λ ou
    erro de previsão nos 6 episódios de seleção); a avaliação nunca decide nada.

## Medidas (só o avaliador vê a verdade)

- **corr**: |correlação| entre q médio de cada entidade de avaliação (≥ 60
  quadros, em movimento) e o `h` verdadeiro do corpo majoritário dela.
- **λ_aval**: fração da variância de q que fica dentro das trajetórias.
- **previsão**: campo de aceleração aprendido por vizinhos (k = 8) sobre as
  posições do ajuste; rolagem de 2 s a partir de estados de avaliação; erro médio
  (px) contra a posição verdadeira projetada. Com e sem projeção que mantém q.
- **identidade**: pureza das entidades (fração de quadros cujo corpo verdadeiro é
  o majoritário da entidade).

## Critérios (fixados agora)

| código | pergunta | exigência |
|---|---|---|
| E1-h | pixels chegaram a `h`? | REVISÁVEL: corr ≥ 0,95 **e** λ_aval ≤ 0,10 em ≥ 2 de 3 mundos, com a aparência trocada |
| E1-uso | a descoberta ajudou? | erro de rolagem com projeção ≤ 0,80 × sem, em ≥ 2 de 3 |
| E1-rev | a revisão importa? | REVISÁVEL > CONGELADO em pureza **e** (λ_aval menor **ou** rolagem menor), em ≥ 2 de 3. Se o CONGELADO errar identidade em < 2% dos quadros, a armadilha não pegou: **inconclusivo**, não positivo |
| E1-ctl | controle | quadros embaralhados no tempo dentro de cada episódio: nenhuma grandeza aceita (λ_seleção > 0,5) em 3 de 3 |

Diagnóstico (não é critério): repetir com identidade verdadeira, com posição
verdadeira e com velocidade verdadeira, para localizar o gargalo se algo falhar.

## Previsão honesta

E1-h 0,50; E1-uso 0,45; E1-rev 0,40; E1-ctl 0,90. A parte mais incerta é se o
autovetor com os 14 termos juntos fica em `h` ou vira uma mistura com r² e v²,
que também variam pouco dentro de órbitas quase circulares.

## Orçamento

Checagem ≤ 15 min; piloto ≤ 90 min agregados; ≤ 2 processos; ≤ 3 GB de memória
(a máquina tem ~2 GB livres de verdade). No limite: encerrar como inconclusivo,
com os eventos de proposta, revisão e rejeição gravados.

## Mudanças depois da CHECAGEM (mundo 101, separado), ANTES do piloto

A checagem (1 mundo à parte, conjuntos 4/2/4) mostrou:
- **defeito meu**: `rastreia` marcava os fragmentos da entrada; a segunda chamada
  (sistema revisável) herdava as marcas e montava trilhas quebradas — o revisável
  deu "nenhuma grandeza". Corrigido: o rastreador não toca mais a entrada.
- **a armadilha T1 não pegava**: o congelado errou identidade em 0,1% dos quadros
  (< 2%), porque dois corpos da mesma cor em órbitas quaisquer quase nunca se
  cruzam em 10 s. O teste decisivo seria inconclusivo por construção. Mudança:
  em T1 o corpo gêmeo tem raio ±5% do primeiro, velocidade ±3% e **sentido oposto**,
  começando 0,6–1,6 rad adiante — cruzam-se de verdade (o problema clássico de
  dois objetos iguais que se atravessam).
- dado que não muda nada nos critérios: o congelado já chegou a corr 0,967 com `h`
  e λ_aval 0,079; com identidade verdadeira, corr 1,000 e λ 0,012.

Critérios, limiares e previsões: inalterados.

## Segunda checagem (mundo 101), e UMA mudança de medida antes do piloto

Com o rastreador corrigido: revisável corr 1,000, λ_aval 0,001, pureza 1,000
(escolheu na seleção a volta "religar usando q"; coeficientes ≈ v·u̇ − u·v̇ com
termos lineares u̇, v̇ — o centro invisível saiu dos termos lineares); congelado
corr 0,917, λ_aval 0,211, pureza 0,997. Rolagem: revisável 3,89 → 3,52 px com q
(−9,5%, abaixo do piso de 20%); congelado 5,40 → 10,65 px (a grandeza errada
PIORA a previsão).

**Mudança (e o motivo, antes do piloto):** "a armadilha pegou" era medido só pela
pureza < 0,98. A checagem mostrou que o dano da armadilha no congelado não é troca
de identidade — é centróide corrompido nos quadros de fusão e trilha quebrada, que
a pureza não vê (0,997 com λ 0,211 contra 0,011 com identidade verdadeira). O
objetivo do critério é o mesmo (só vale se a percepção congelada de fato sofrer);
a medida passa a ser: pureza do congelado < 0,98 **OU** λ_aval do congelado ≥ 2×
o λ_aval com identidade verdadeira. Tudo o mais — E1-h, E1-uso, E1-rev (pureza
maior E λ ou rolagem melhor), E1-ctl, limiares e previsões — inalterado.

## RESULTADO DO PILOTO (mundos 1, 2, 3 — nunca tocados pelas checagens)

| mundo | sistema | corr com h | λ_aval | pureza | rolagem sem → com q (px) |
|---|---|---|---|---|---|
| 1 | congelado | 0,993 | 0,041 | 0,986 | 5,73 → 5,11 |
| 1 | revisável | 0,986 | 0,029 | 0,986 | 2,81 → 2,56 |
| 2 | congelado | 0,962 | 0,096 | 0,962 | 6,46 → 5,26 |
| 2 | revisável | 1,000 | 0,005 | 0,998 | 3,36 → 2,66 |
| 3 | congelado | 0,923 | 0,129 | 0,965 | 7,74 → 5,58 |
| 3 | revisável | 0,994 | 0,027 | 0,988 | 2,11 → 1,70 |

Identidade verdadeira (diagnóstico): λ_aval 0,021 / 0,018 / 0,020, corr ≥ 0,999.
Controle com o tempo embaralhado: nenhuma grandeza aceita nos 3.
Escolha do revisável na seleção: volta1, volta2 (religar usando q), volta2.
Coeficientes do revisável nos 3 mundos ≈ ±(u·v̇ − v·u̇) + termos lineares em u̇, v̇
(o centro invisível), com câmeras afins diferentes.

**VEREDITO: E1-h PASSA (3/3) · E1-rev PASSA (2/3; armadilha pegou em 2/3) ·
E1-ctl PASSA (3/3) · E1-uso FALHA (1/3: −9%, −21%, −19,4%).**

Ressalvas: órbitas e armadilhas foram desenhadas conhecendo o alvo (primeira
validação, não mundo desconhecido); a checagem (mundo 101) serviu para corrigir um
defeito, tornar a armadilha T1 real e corrigir a MEDIDA de "armadilha pegou" — tudo
registrado acima, antes do piloto. Os limiares das operações de revisão são meus.

## POS-HOC (não pré-registrado): separar as voltas na avaliação — `separa_voltas.py`

| mundo | volta0 sem revisão λ/pureza | volta1 continuidade | volta2 usando q | volta2 PLACEBO |
|---|---|---|---|---|
| 1 | 0,041 / 0,986 | 0,029 / 0,986 | **0,006 / 0,999** | 0,029 / 0,986 |
| 2 | 0,096 / 0,962 | 0,008 / 0,997 | 0,005 / 0,998 | 0,007 / 0,997 |
| 3 | 0,129 / 0,965 | 0,028 / 0,988 | 0,027 / 0,988 | 0,028 / 0,988 |

O script imprimiu "SIM, 3 de 3", mas a regra que escrevi não tinha tamanho mínimo
de efeito. Leitura honesta: a revisão pela DESCOBERTA decide no mundo 1 (e o
placebo não faz nada); nos mundos 2 e 3 empata (diferenças de 0,001–0,003). O
grosso do ganho da revisão vem da CONTINUIDADE (cortar + religar por
extrapolação). A descoberta corrigindo a percepção existe, mas foi decisiva em 1
de 3 mundos.
