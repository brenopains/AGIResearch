# Pré-registro — o laboratório (transplante 10): descobrir, reutilizar e revisar um conceito, sem ninguém dizer qual

Escrito em 09/10/2026, **antes** de rodar as sementes registradas (20261010 a 20261019).
Código congelado: `laboratorio.py`, `arena.py`, `experimento_laboratorio.py` neste commit.

## De onde vem

A revisão externa de 09/10 (GPT, lendo o ARQUITETURA.md inteiro) e a auditoria do código
concordam: os mecanismos que sobreviveram aos controles no AGI 2.0 nunca foram ligados à
mente viva — objetos pela continuidade (Etapa 1), invenção de grandeza (`inventa`, 10/10),
reanálise com olhos frescos (`caderno`, 37% → 88%), crise contra o ruído + reanálise por
consenso (`autonomo`, 83% × 68%), conceito julgado pelo que paga + sala de espera
(`conceito_util`, 10/10 e 10/10). O marco proposto: **ela descobrir um conceito útil que
ninguém forneceu, reutilizá-lo num problema diferente e revisá-lo quando a realidade o
contradiz** — de pixels, sem o nome do mundo.

## O mundo (lado do avaliador; a mente não vê nada disto)

Arena `arena.py`: vídeos de 24 s, 5 quadros/s, 128×128 RGB, 2–4 corpos, câmera afim
desconhecida (uma por vida), oclusor cinza em 50% dos episódios, dois corpos da mesma cor em
30%, corpo bicolor em 30%. Uma VIDA = 72 episódios, as leis trocam sem aviso:

| episódios | mundo | lei | conceito útil |
|---|---|---|---|
| 0–15 | giro | a = +1,0·s²·rot90(v) | s² (rapidez ao quadrado) |
| 16–31 | mola_h | a = −0,09·h²·x (centro invisível) | h (momento angular) |
| 32–47 | contragiro | a = −1,4·s²·rot90(v) | s² de novo, lei diferente |
| 48–63 | mola | a = −0,25·x | nenhum (h, energia e Fradkin se conservam, mas a lei não precisa) |
| 64–71 | giro de novo | igual ao primeiro | s² |

## O que a mente recebe

Quadros e tempos. Não recebe: quantos corpos, quem é quem, o centro, a câmera, a lei, o
nome do mundo, o que se conserva. Primitivas declaradas (minhas): fundo = mediana no tempo;
objeto = região conexa do que difere do fundo; continuidade por atribuição ótima; religar
sumiços de até 3 s; estado por ajuste quadrático local; alfabeto = 14 monômios de grau ≤ 2
de (x, y, vx, vy), combinações esparsas de até 4; lei = aceleração como combinação de termos
(BIC); conceito na lei = q, q² e cada um vezes x, y, vx, vy.

## Braços (mesma percepção, mesmos episódios)

- **COMPLETO** — o órgão que vai para a mente viva.
- **CONCLUSOES** — a lei de nascimento fica; conceito só entra como correção empilhada;
  nada guardado muda de teoria (o "junta dados, leis velhas" do caderno).
- **SEM_CONCEITO** — sem invenção.
- **PLACEBO** — a invenção roda sobre pontos SORTEADOS entre trajetórias (mesmo alfabeto,
  mesmo procedimento, sem estrutura) e passa pela mesma sala de espera.
- **SEM_TRANSFERENCIA** — cada teoria só usa conceitos que ela mesma inventou.

## Medidas

- **erro da 1ª tentativa**: antes de aprender com o episódio, a melhor teoria prevê onde
  cada coisa estará 2 s depois; mediana entre trajetórias, px num quadro de 128. Agregado
  por vida = mediana dos episódios 4–15 de cada uma das fases giro, mola_h e contragiro.
- **identidade do conceito**: |correlação| entre o conceito em uso no último episódio da
  fase e s² ou h verdadeiros, em estados percebidos de episódios NOVOS dos quatro mundos.
- **permanência**: oclusões (visível → escondido ≥ 2 quadros → visível); sucesso se a mesma
  trilha cobre o corpo (≤ 3 px) antes e depois. Comparada à percepção sem religar.

## Critérios (fixados agora; 10 vidas)

| código | pergunta | exigência |
|---|---|---|
| L1 | inventa s² sem ninguém dizer? | COMPLETO, fim do giro: conceito em uso com \|corr(s²)\| ≥ 0,99 em ≥ 8/10 |
| L2 | revisa: troca para h quando o mundo muda? | COMPLETO, fim da mola_h: \|corr(h)\| ≥ 0,99 em ≥ 8/10 |
| L3 | reutiliza num problema diferente? | COMPLETO, fim do contragiro: o MESMO conceito do giro e \|corr(s²)\| ≥ 0,99 em ≥ 8/10 |
| L4 | conservado não é útil? | COMPLETO, fim da mola: nenhum conceito em uso em ≥ 8/10 |
| L5 | o conceito paga? | erro 1ª tentativa COMPLETO ≤ 0,85 × SEM_CONCEITO (mediana entre vidas) E COMPLETO melhor em ≥ 8/10 |
| L6 | placebo não paga? | PLACEBO ≥ 0,90 × SEM_CONCEITO (mediana) E COMPLETO melhor que PLACEBO em ≥ 8/10 |
| L7a | percebe a troca de mundo? | COMPLETO declara crise nos 5 primeiros episódios após CADA troca (16, 32, 48) em ≥ 7/10 |
| L7b | reconhece o mundo de volta? | no episódio 64, sem crise e prevendo com a teoria que explicou o fim do giro, em ≥ 7/10 |
| L8 | reanalisar vence acumular conclusões? | COMPLETO < CONCLUSOES no erro da 1ª tentativa em ≥ 7/10 |
| L9 | reutilizar paga? | erro 1ª tentativa nos episódios 32–39: COMPLETO < SEM_TRANSFERENCIA em ≥ 7/10 |
| P1 | permanência do objeto | religar ≥ 0,70 dos eventos E percepção congelada ≤ 0,35 (somando as vidas) |
| C1 | controle | o inventor com a régua aceita alguma grandeza nas trajetórias reais do giro em ≥ 9/10 E nenhuma com os pontos permutados em ≥ 9/10 |

## Previsões (honestas, das vidas de desenvolvimento)

L1 0,35 · L2 0,70 · L3 0,30 · L4 0,50 · L5 0,50 · L6 0,60 · L7a 0,50 · L7b 0,55 ·
L8 0,30 · L9 0,40 · P1 0,65 · C1 0,85.

Leitura fixada antes: L1+L3+L2 juntos são o marco ("descobrir, reutilizar, revisar"). Se L5
passa e L1 falha, o conceito ajuda mas não é identificável com a precisão exigida. Se L8
falha, o ganho da reanálise do caderno não se repete neste substrato (registrar como nulo,
não esconder). Se L6 falha, o ganho de L5 é "mais variáveis", não "o conceito certo".

## Desenvolvimento (sementes 101, 102, 103 — nunca as registradas), na ordem

Cada mudança veio de um defeito visto nas vidas de desenvolvimento, todos meus:
1. Inventor conjunto de 14 termos (o da Etapa 1, a 30 quadros/s) dava s² com |corr| 0,68:
   a 5 quadros/s o ruído da velocidade espalha peso. → combinações ESPARSAS de até 4 termos
   (o princípio do `inventa.py`, 10/10).
2. Com 3–6 trajetórias qualquer forma quadrática "se conservava" (43 conceitos em 44
   episódios). → mínimo de 16 trajetórias (8 acham, 8 validam).
3. A mesma ideia reaparecia mais precisa e era descartada como duplicata. → REVISÃO do
   conceito (mesmo nome, forma nova, leis re-derivadas) quando se conserva melhor.
4. A régua por soma deixava uma trajetória ruim decidir; a mediana premiava o que se
   conserva só na maioria (centro do círculo). → média sem as 20% piores; bits contados por
   trajetória.
5. Lei com conceito explodia ao extrapolar q² fora da faixa vista; a média dos quadrados
   ficava dominada por 1 trajetória. → conceito limitado à faixa vista; utilidade PAREADA
   por trajetória (bits = Σ log2 razão de erros − custo de escolher).
6. Conceito adotado sem nenhum termo dele na lei. → opção que a lei não usa não é opção.
7. Alarme por pontos (correlacionados) → crises falsas. Tentei: unidade = trajetória
   (melhor); relativo à inércia (pior em vídeos lentos); duas escalas como critério único
   (a teoria da mola_h passou a "explicar" a mola). Ficou: crise quando o episódio SOZINHO
   prevê melhor que a teoria (bits pareados) OU o erro é anômalo nas DUAS escalas (px e
   relativo) ao ruído da própria teoria.
8. Hipótese de um episódio com ruído largo virava sumidouro (explicava mal o mundo seguinte
   e escondia a crise). → hipótese que nada confirma em 3 episódios é abandonada.
9. Oclusões duravam mais que a janela (1,5 s → 3,0 s) e o portão fixo de 8 px recusava
   religar curvas longas. → portão cresce com metade do caminho percorrido escondido.
   Permanência 0,48 → 0,74, pureza de identidade 0,99.
10. Controle de "quadros embaralhados" era vazio (0 trajetórias sobreviviam). → trocado pelo
    inventor real sobre pontos permutados entre as trajetórias percebidas.
11. Sala de espera com 3 vagas era tomada por quase-invariantes baratos. → 5 vagas.

Estado nas 3 vidas de desenvolvimento ao congelar: L1 2/3, L2 3/3, L3 2/3, L4 3/3, L5 3/3
(razão 0,86), L7a 3/3, L7b 3/3, P1 0,74 × 0,24, C1 3/3. Numa das vidas (103) s² é mal
conservado NA PERCEPÇÃO (λ 0,036 contra 0,003–0,006 nas outras): corpos de rapidez
parecida e ruído de 5 quadros/s — propriedade daquela vida, não defeito; esperado de novo.

## Orçamento

≤ 30 min de máquina (4 processos), sem re-rodar sementes registradas. Se o código quebrar
no meio: conserto declarado e re-execução INTEIRA, registrada como tal.

## RESULTADO (sementes 20261010–20261019, 612 s, código do commit df5d358, nada re-rodado)

| código | resultado | veredito | previsão |
|---|---|---|---|
| L1 inventa s² no giro | 5/10 com \|corr\| ≥ 0,99 (mais 2 com 0,97 e 0,94; 2 com 0,87 e 0,83; 1 sem conceito) | **FALHA** | 0,35 |
| L2 revisa para h na mola_h | **9/10** | PASSA | 0,70 |
| L3 o MESMO conceito no contragiro, \|corr(s²)\| ≥ 0,99 | 5/10 (s² ≥ 0,99 no fim do contragiro em 7/10, mas em 2 era outro conceito, reinventado mais preciso) | **FALHA** | 0,30 |
| L4 mola sem conceito | **9/10** | PASSA | 0,50 |
| L5 o conceito paga | erro 1ª tentativa 2,57 px × 4,82 sem conceito (**0,53×**); melhor em **10/10** | PASSA | 0,50 |
| L6 placebo não paga | placebo 4,24 px = **0,88×** do sem-conceito (exigia ≥ 0,90); COMPLETO melhor que placebo em **10/10** | **FALHA** | 0,60 |
| L7a crise em ≤ 5 episódios após cada troca | **10/10** | PASSA | 0,50 |
| L7b reconhece o giro de volta | **9/10** | PASSA | 0,55 |
| L8 reanalisar vence acumular conclusões | 6/10 (razão 0,81) | **FALHA** | 0,30 |
| L9 reutilizar paga (início do contragiro) | **10/10** (mediana 3,6 × 4,9 px) | PASSA | 0,40 |
| P1 permanência | religar **0,714** × congelada 0,182 | PASSA | 0,65 |
| C1 controle | reais com alguma grandeza 10/10; permutadas sem nenhuma 10/10 | PASSA | 0,85 |

Inércia (nada acelera) erra 13,3 px na mesma medida.

**Leitura, pela regra fixada antes.** O marco "descobrir → reutilizar → revisar" passou
na parte de REVISAR (L2 9/10, crises 10/10, mola recusada 9/10) e na de o conceito PAGAR
(L5, L9 10/10); falhou na IDENTIDADE exata do primeiro conceito (L1 5/10, L3 5/10): o s²
inventado no giro muitas vezes é uma forma aproximada (0,83–0,97), que paga na previsão
mas não passa na régua de 0,99. A causa já estava medida no desenvolvimento: a 5 quadros/s
o ruído da velocidade deixa s² mal conservado em parte das vidas. O conceito aproximado
foi reutilizado e pagou (L9 10/10) — o que transferiu foi uma ideia ÚTIL, nem sempre a
EXATA.

**L6 falhou por pouco e de um jeito que importa:** o placebo (mesmo alfabeto, pontos
sorteados) ganhou 12% sobre não ter conceito — parte do ganho é "mais variáveis na lei",
como o `inventa.py` já tinha visto (p = r·vr ajudou a reescrever o termo fraco). Mas o
conceito de verdade vale 0,53× e o placebo 0,88×: a maior parte do ganho é estrutura.

**L8 (6/10, razão 0,81):** reanalisar com olhos frescos foi melhor que acumular
conclusões na mediana, mas não em 7 de 10 vidas. O efeito grande do caderno (+36 pontos)
não se repetiu neste substrato; aqui a lei de nascimento + correções empilhadas fica
perto. Registrado como resultado fraco, não como confirmação.

**Instrumento sem poder:** "falhas antigas recuperadas" deu 4/4 em todas as fases, nos
dois braços — não discrimina nada e não deve ser citado como evidência.
