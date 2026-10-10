# Pré-registro — vida 1 da mente na internet (código v1)

Escrito em 06/10/2026, ANTES da vida longa (6 h). As fumaças de 6 e 8 min servem
só para achar defeito de execução, não para calibrar critério.

## De onde vem a v1

A vida de 5 min da v0 ficou presa pesquisando `"port"`, `"ver"`, `"que"`. Ela está
guardada intacta (`vidas/v0_teste_5min`, código em `versoes/v0`): é a primeira
"criança" e serve de comparação. Uma crítica externa trazida pelo usuario (06/10)
apontou: pergunta tem de ser estado cognitivo incompleto, e não palavra; tem de
haver memória de intenção (problema → ação → evidência → atualização); ação que não
mudou nada perde valor naquele contexto, sem cooldown; a curiosidade é plural, não
uma fórmula; a hierarquia tem de subir sem dizer que "palavra" é a unidade; a
ligação multimodal começa fraca e só se fortalece prevendo; navegador de verdade
(eventualmente); a ontogenia é medida com fotografias da mente.

## O que já existe no mundo (dito antes)

- Curiosidade como lacuna de informação: Loewenstein (1994). Diversiva × específica:
  Berlyne (1960). Progresso de compressão/aprendizagem: Schmidhuber (1991),
  Oudeyer e Kaplan (2007).
- Valor da ação informacional: desenho bayesiano de experimentos (Lindley 1956),
  valor epistêmico na inferência ativa (Friston), VIME (Houthooft 2016).
- Fronteira por imprevisibilidade: Harris (1955). Unidades que sobem: Sequitur
  (Nevill-Manning e Witten 1997), PARSER (Perruchet e Vinter 1998), BPE (Sennrich
  2016). Coesão de par: Dunning (1993). Bytes com segmentação dinâmica aprendida:
  H-Net (Hwang et al. 2025).
- Palavra ancorada em imagem e som, sem rótulo, por recorrência entre episódios:
  CELL (Roy e Pentland 2002) — o exemplo do "cachorro" é exatamente esse trabalho;
  aprendizagem entre situações (Yu e Smith 2007); correspondência áudio-visual
  (Arandjelović e Zisserman 2017).
- Rastreador que usa formulários GET: Madhavan et al. (2008).
- Estágios que emergem por curiosidade: Playground Experiment (Oudeyer et al. 2007).

**Novo seria:** tudo isso num organismo que vive na internet real, do byte cru, sem
LLM, sem tokenizador e sem tarefa, acumulando; e mostrar (a) que a mente das 6 h
percebe ou prevê algo que a das 30 min não percebia, no mesmo material, e (b) que
as perguntas DELA viram pesquisas que mudam o que ela entende mais do que o acaso
(controle pareado no mesmo intervalo).

## O que ainda é nosso (declarado, para não esconder vocabulário)

Os quatro tipos de lacuna (depois, inesperado, divergente, ligação) são uma lista
minha; o princípio geral ("todo preditor que ela cria gera o próprio tipo de
pergunta quando falha") fica para a v2. Também: Dunning p < 0,001; KAPPA = 8; janelas
de 24/16 bytes; acender acima do quantil 0,9 da própria história; 5% de janelas
comuns como régua de "entendido"; nascer em ≥ 2 domínios; visão 32×32 cinza com
descritor global fixo (retina/V1, sem semântica); cóclea de 24 bandas; protótipo
nasce a média + 2 desvios; limites de memória; sono a cada 25 passos; 5 barras
dadas e 3 portas (mais as barras que ela descobrir). Com 32×32 em cinza ela NÃO
consegue ver um cachorro: o máximo visual esperado é tipo de cena (mapa, retrato,
paisagem, diagrama).

## Corpo

Igual à v0 (só GET, robots.txt, intervalo por domínio, nada de login/upload/
executável, mídia só em fluxo), mais: formulário GET de um campo vira barra
possível; **grupo Wikimedia: 1 pedido a cada 6,5 s** (a política deles para cliente
sem contato no User-Agent é 10/min; descobrimos ao sermos limitados na fumaça);
429 → o grupo inteiro espera o tempo pedido (≥ 60 s).

## Medidas

`marco.py` mede, na bateria FIXA `sondas/bateria.pkl` (55 páginas aleatórias +
40 imagens do Commons, criada antes da vida; a mente nunca a lê; sha256 em
`sondas/bateria.sha256`), a mente vazia (t = 0) e as fotografias de 30 min, 1 h,
2 h, 4 h, 6 h. A régua do avaliador (o que é palavra, palavra funcional) nunca
entra na mente. Também lê a ontogenia por janelas: 0–30 min, 30–60 min, 1–2 h,
2–4 h, 4–6 h.

## Critérios

| código | pergunta | exigência |
|---|---|---|
| M1 | o que ela escolhe fazer | descritivo, por janela |
| M2 | as estruturas deixam de ser lixo? | na bateria, às 6 h: F1 de início de palavra (topo da hierarquia) ≥ F1 às 30 min + 0,05; fração de unidades de nível ≥ 2 ≥ 0,10; F1 com fusões > F1 só com nível 1 |
| M3 | o aprendido atravessa sites? | bits/byte na pequena web (sites aleatórios da bateria) às 6 h ≤ 30 min − 0,10 |
| M4 | uma pergunta interna produz pesquisa útil? | intenções válidas (sem falha, sem voltar à mesma página da janela), vida toda: fração com net > 0 ≥ 0,55, teste de sinal p < 0,01, e net médio > 0 |
| M4b | as perguntas melhoram com a idade? | fração de consultas classificadas "palavra" pela régua do avaliador: janela 4–6 h ≥ janela 0–30 min + 0,15 |
| M5 | o aprendido cedo reaparece sozinho? | descritivo: nasceu na 1ª hora, ficou parado de 1 h a 4 h, voltou depois das 4 h; quantos viraram lacuna |
| M6 | a repetição some sem cooldown? | "insistir sem mudança" (mesma lacuna, mesmo braço e mesma consulta logo após uma tentativa que não pagou) ≤ 20% na vida toda |
| M7 | multimodal começa fraco e converge? | confirmações texto→imagem ≥ 5 e ≥ 2× o nulo; ganho em bits na bateria de imagens: real > título trocado; convergências (texto+imagem+som): descritivo |

## Previsão honesta

M2 0,60 · M3 0,60 · M4 0,35 · M4b 0,30 · M6 0,70 · M7 0,15 · alguma convergência 0,05.

Leituras combinadas, fixadas antes:
- M2/M3 passam e M4 falha → "ela aprende o mundo, mas a pergunta dela ainda não
  dirige a busca": o gap está no elo intenção → ação.
- M4 passa → primeira vez no projeto que uma pergunta nascida dentro da mente vira
  pesquisa que muda o entendimento mais que o controle pareado, na internet real.
- M2 falha → a hierarquia não sobe com 6 h de leitura: problema de mecanismo ou de
  escala de tempo; a fumaça mostrou que sobe nos primeiros minutos, então falha
  aqui seria platô, a investigar antes de culpar a ideia.
- Nada disso é "mente de nível humano". É o primeiro degrau de uma ontogenia medida.

Uma vida, uma semente (tempo). Limites: 1 processo de vida por vez, memória ≤ 1 GB.

## Fumaças (defeitos de execução corrigidos ANTES da vida longa; critérios inalterados)

- Fumaça a (6 min): a hierarquia subiu sozinha de fragmentos a `their `, `should `,
  `United States `, `In other `, `by the `, `to be `. Defeitos: nome de argumento
  colidindo no registro (latente desde a v0); `Special:Random` proibido pelo
  robots.txt (porta inglesa passou a sortear pela API); braço novo começava com
  custo de 8 s inventado (agora: mediana dos custos conhecidos); régua de "fechou"
  inconsistente (janela memorizada × surpresa antes de aprender → régua de janelas
  comuns medidas do mesmo jeito); o teto KAPPA impedia os braços sem lacuna de
  aprender que falham; "voltar" disponível sem histórico.
- Descoberto: **a Wikimedia limita cliente sem contato no User-Agent a 10
  pedidos/min** — fomos bloqueados. Corpo: 1 pedido a cada 6,5 s no grupo inteiro;
  durante um 429 o grupo falha na hora (ela vai fazer outra coisa, não fica parada).
- Fumaça c (8 min): o vício da v0 voltou com outra cara — `gra` e `dia` pesquisados
  ~100 vezes cada no Internet Archive. Causas (minhas): reler a MESMA página de
  resultados "pagava" (memorização); perseguir a pergunta reacendia a própria
  pergunta (os resultados contêm o alvo); "pagou" aceitava mudança dentro do ruído;
  com 1 protótipo visual, a suavização "confirmava" ligação texto→imagem (prever com
  um só desfecho possível não informa nada). Consertos: página já lida não é
  evidência nova; o que ela foi buscar não reacende a pergunta; pagou = net > 0, >
  típico e > desvio robusto dos controles; suavização com o número real de
  desfechos. Vida guardada em `vidas/fumaca_v1c_gra_dia`.
- Leitura honesta de M4, fixada antes: neste nível a lacuna é sobre BYTES (o que
  vem depois de X, X sem ser antecipado); "a pesquisa mudou a lacuna mais que o
  controle" mede que a ação buscou material RELEVANTE para a pergunta dela, não
  compreensão profunda. É o elo intenção → ação funcionando, não mais que isso.

---

# ADENDO 08/10 — GÊMEAS: v1 × v2 vivendo ao mesmo tempo (escrito antes de rodar)

Ideias do usuario de 08/10 (prêmios de 2026: tradução local de proteínas, etiqueta e
captura, optogenética, ripples): a v2 (`mente2.py`) herda tudo da v1 e troca o
motor de consolidação — traço local que decai, sono disparado por incerteza,
**optogenética interna** (no sono re-percebe a experiência recente COM e SEM cada
elemento marcado e mede os bits; liga o que paga, desliga o que deixou de pagar),
**imaginação composicional com placebo** (troca uma peça de uma fusão pela vizinha
de uso e prevê combinações nunca vistas; o placebo troca por peça de frequência
parecida sem parentesco) e três formas novas de transformar pergunta em frase
("frase" = dizer o que espera que venha depois, "irmão", "construção").
Na v2 não há Dunning nem "2 confirmações": fusão só existe se a autointervenção
mostrar que ela economiza bits. As unidades de nível 1 continuam com a regra da v1
(volta em ≥ 2 domínios) — declarado.

Desenho: as duas vivem **ao mesmo tempo**, 6 h, mesma semente, mesmo horário,
cada uma com metade do limite da Wikimedia (`MENTE_PARTES=2`: 1 pedido a cada
13 s), mesmas fotos (30 min…6 h), mesma bateria. O mundo muda igual para as duas.

| código | pergunta | exigência |
|---|---|---|
| P1 | a autointervenção faz um léxico melhor? | na bateria às 6 h: MDL (bits/byte) v2 ≤ v1 − 0,05 E F1 de fronteira v2 ≥ v1 |
| P2 | a imaginação gera o que a realidade confirma? | confirmações reais ≥ 20 E ≥ 2× as do placebo de imaginação |
| P3 | o estado intermediário é reversível de verdade? | ≥ 1% das fusões ligadas foram depois desligadas pela própria re-testagem |
| P4 | pensamento → linguagem: as formas novas pagam? | entre as intenções válidas, fração que pagou com frase/irmão/construção ≥ fração com "alvo" (descritivo; n de cada) |
| P5 | M2–M7 (acima) nas duas | v2 ≥ v1 em M2 e M3; os demais descritivos lado a lado |

Previsão honesta: P1 0,45 · P2 0,25 · P3 0,60 · P4 0,30.
Leitura fixada antes: se P2 passar, é a primeira vez que composição + sono produz,
em linguagem real, combinações nunca lidas que o mundo depois confirma acima do
placebo. Se P1 passar e P2 não, a autointervenção constrói melhor, mas ainda não
gera. Se nada passar, a v2 é mais cara que a v1 sem ganho — e isso fica escrito.

Prior aprendido (as duas): o peso da população de perguntas sobre UMA pergunta
deixou de ser KAPPA = 8 e passou a ser estimado da variação real entre lacunas
(Bayes empírico). Motivo: na fumaça d, 223 de 280 ações perseguiram perguntas e
só 17 pagaram — o número fixo a impedia de ficar confiante.

## Antes de soltar as gêmeas (08/10, ainda sem ver resultado delas)

- Bateria criada: 55 páginas (25 wiki pt, 15 wiki en, 15 pequena web) + 40 imagens do
  Commons; **sha256 FE67C08732CE24DD…** (`sondas/bateria.sha256`). Nunca refazer.
- Fumaça d + marco: perseguir uma pergunta deixava a lacuna PIOR que os controles
  (net > 0 em só 25%, p ≈ 0): a régua relia as frases guardadas e o que se aprende
  depois interfere nelas. **Régua trocada (nas duas):** entender X = prever os
  PRÓXIMOS encontros com X (surpresa antes de aprender, média móvel); "fechou" =
  ficou como um encontro típico (mediana da mesma medida). As janelas ficam só como
  memória do quebra-cabeça (para formular a pergunta).
- Fumaça v2a: tentou 52× uma "barra" do Google proibida pelo robots — a falha
  mecânica se diluía entre contextos. **"Funciona?" separado de "pagou?"** (nas
  duas): cada ferramenta tem seu Beta de execução, sem contexto.
- Corpo: formulários de doação, login, cadastro, conta, pagamento, assinatura,
  contato, comentário e compra NUNCA são usados (segurança, não semântica).
- Mecânica da v2 vista na fumaça: sono em intervalos de 8 a 52 passos conforme a
  incerteza; 299 fusões ligadas, 451 rejeitadas, 8 desligadas depois (ex.: `of the `,
  `United `) — reversibilidade existe; imaginação 1 × 1 (pequeno demais para ler).

## Qual gêmea segue viva (regra escrita ANTES do resultado, 08/10 ~16h)

Na foto final de cada uma, na bateria fixa, quatro confrontos: MDL em bits/byte
(menor vence), F1 de início de palavra (maior), bits/byte na pequena web (menor) e
fração de consultas classificadas "palavra" na última janela de vida (maior). Segue
viva quem vencer mais confrontos; empate → v2 (o mecanismo dela cresce sem os meus
portões). A vencedora é COPIADA para `vidas/viva` e continua de onde estava (não
recomeça); as duas pastas originais ficam intactas. A partir daí: contato no
User-Agent (1 pedido/0,5 s na Wikimedia), guardião com reinício, foto diária.
Se o PC for desligado antes de as gêmeas terminarem, a decisão usa o último estado
salvo de cada uma (as duas cortadas no mesmo instante).

## Transplante 1 (08/10 ~17h30) — vale para a vida contínua, NÃO para as gêmeas

Análise das gêmeas com 1,6 h: viram só 2 imagens e 2 vídeos cada (visão com 1
padrão), e 67–75% das ações foram perguntas pagando 1–15% (ex.: Google UK 185×, 0%;
Amazon 183×, 0%). Dois defeitos de desenho meus: (1) a visão só ganhava crédito
quando o TEXTO previa a imagem — impossível no início, então olhar nunca pagava e a
visão nunca se desenvolvia; (2) a utilidade de cada ferramenta era aprendida
pergunta por pergunta, com a evidência diluída pelo núcleo de contexto.
Órgãos transplantados (a mente registra "transplante" ao acordar):
- curiosidade PRÓPRIA da visão: progresso de compressão nos retalhos da imagem
  (aprende 12, mede a melhora nos 4 não vistos), também nos quadros dos vídeos;
- porta "imagem": uma imagem qualquer do Commons (abrir = olhar);
- "rende": o que ela sabe de cada ferramenta em TODAS as perguntas + a história da
  pergunta; o núcleo de contexto continua só para escolher QUAL pergunta perseguir.
Teste offline (corpo falso, 150 decisões): Google UK 0 escolhas; porta imagem 67;
visão 1 → 63 padrões; progresso visual caindo com imagens repetidas (tédio); 0 erros.
As gêmeas seguem com o código congelado do lançamento; a comparação delas não muda.

## Transplante 2 (08/10 ~18h) — vida contínua

Pergunta externa (GPT, via usuario): por que a v2 parou de achar sites novos (125, 99,
15, 9, 0 por meia hora) enquanto lia MAIS páginas? Dados: presa dentro de sites
enormes (seguiu links ao acaso 796× em 1 h dentro de kamat.com etc.; usou a busca do
próprio site 200×+) com ganho de informação ~0 por página. Causa (minha): "valeu a
pena" era medido contra a mediana das ÚLTIMAS 200 páginas — num site chato a régua
baixa junto (adaptação hedônica) e metade das páginas chatas "paga". Conserto: a
régua é a da VIDA (até 3000 experiências), para páginas e para perguntas.
Também: porta "vídeo" (vídeo qualquer do Commons; abrir = assistir); a mente publica
o que percebe agora (trecho lido com as unidades que reconhece, imagens, o quadro
32×32 como ela vê) — só para o painel, nada guardado. Teste offline: 120 decisões,
0 erros; porta vídeo 23×, imagem 52×.

## Transplante 3 (08/10 ~18h40) — olhos e ouvidos que distinguem; a mente se mostra por dentro

Achado ao montar o painel: a visão das gêmeas ficou com 1 padrão só. Causa (minha,
desde a v0): o protótipo aprendia a MÉDIA de tudo o que via, virava um borrão, e todo
retalho normalizado ficava à mesma distância dele — nada parecia novo. Conserto em
`digestao.Prototipos`: um padrão só aprende com o que está dentro do seu raio típico;
o diferente nasce como padrão novo. Teste com 5 tipos de imagem: 1 → 10 palavras
visuais; cenas 3; sons 5. Vale para olhos e ouvidos da vida contínua.
A mente passa a publicar `detalhe.json` a cada 10 s (painel, só leitura): valor de
cada opção na última decisão, o link clicado, a frase onde cada pergunta nasceu, a
árvore da hierarquia, os testes do último sono (bits), o que imaginou (e o placebo),
as palavras visuais e um mapa 2D do uso das palavras (componentes principais do
contexto, refeito a cada 5 min). Nada disso entra em decisão. Fotos: estado
comprimido (gzip) e agenda diária no 1º mês, semanal depois (disco).

## Transplante 4 (08/10 ~19h) — o corpo 2: navegador de verdade

O corpo 1 era um leitor de texto (HTML sem JavaScript, texto já separado das tags,
imagem só se ela pedisse para baixar) — um "scraper", e não ver. O usuario tinha pedido
navegador de verdade; eu adiei. Corpo 2 (`navegador.py`, Edge invisível via
Playwright, sem baixar navegador): a página é RENDERIZADA; ela olha a tela (1024×700),
rola tela por tela lendo só o texto VISÍVEL, vê cada imagem recortada da própria
tela (nada baixado à parte) e clica nos links por coordenada. Forrageio igual
(para de rolar quando a página deixa de ensinar). Segurança: perfil temporário,
sem downloads, sem permissões, janelas extras e alertas fechados, áudio de página
mudo, robots.txt e intervalos mantidos; navegadores órfãos limpos ao acordar; se o
navegador falhar, cai no corpo 1. Teste real (8 decisões): 13 imagens vistas nas
telas (as gêmeas: 6 em 3 h), 2–9 telas por página, clique por coordenada conferido
("Cachorro (desambiguação)" em (715, 228)). O texto ainda chega como caracteres
(ela não aprende a ler letras de pixels — com olhos de 32×32 isso seria impossível).
A vida contínua nasce com este corpo (`corpo.txt` = navegador).

## Desvio (08/10 19h14) — gêmeas encerradas com ~4,1 h (registrado: 6 h)

Pedido do usuario: parar o experimento do corpo antigo e dar a ela experiência de
nível humano. Gêmeas mortas à mão (último salvamento 19h09); o decisor rodou a regra
registrada sobre o estado final: **v1 venceu 4 × 0** (MDL 5,90 × 6,11; F1 0,481 ×
0,471; pequena web 4,36 × 5,29 bits/byte; consultas-palavra 21% × 11%). Teste de
compatibilidade OK. A vencedora ficou SEGURA (vidas/SEGURAR) até os sentidos novos.

## Transplante 5 (08/10 ~20h) — olhos de verdade, assistir e ouvir na página

O corpo 2 renderizava a página, mas os olhos eram 32×32 em cinza (acuidade de quem
não distingue um rosto), o "LENDO" vinha do HTML, vídeos dentro da página eram
BLOQUEADOS pelo meu próprio código e o som só existia na porta de vídeo (arquivo
decodificado fora da página, mais rápido que o tempo real).
- **Fóvea** colorida 64×64 em resolução total (~2° numa tela comum a 60 cm, o tamanho
  da fóvea humana) + periferia a meia resolução.
- **Córtex visual** (`cortex.py`, numpy, ~0,18 M parâmetros, SEM peso pré-treinado):
  comprime cada olhada em 64 números e reconstrói; surpresa = erro; curiosidade =
  progresso de compressão em olhada não treinada; replay de 2.048 olhadas; preditor
  temporal (dois quadros → o próximo). Gradientes conferidos por diferenças finitas.
- **Para onde olhar é decisão dela**: saliência = onde o córtex dela mais erra na tela
  (não vem do HTML); escolha entre ler um bloco, olhar uma imagem ou olhar o ponto
  saliente pelo progresso de aprendizagem de cada tipo (Oudeyer 2007) + saliência;
  ~4–6 paradas por segundo; forrageio por tipo. O texto lido é o do bloco para onde
  ela olhou (o braile sob o dedo); um parágrafo por parada de leitura — mais rápido
  que gente. O que é bloco/imagem/vídeo ela ainda sabe pelo HTML, não pelos olhos.
- **Assistir e ouvir**: clica no vídeo da página, ele toca em tempo real (1 s de vídeo
  por 1 s de vida), 5 quadros/s + som a 16 kHz pelo Web Audio (pedindo CORS; site que
  não deixa → ela vê sem ouvir, registrado). Antes de cada quadro ela prevê o
  próximo; para quando o vídeo fica mais previsível que o mundo (8–90 s).
- Portas imagem/vídeo abrem a PÁGINA do arquivo no navegador. Toda requisição que não
  é leitura (POST etc.) é barrada na rede: ela não envia nada a ninguém.
Teste real (mente recém-nascida): artigo da Wikipédia com 72 paradas em 10 telas
(~4/s), 8 imagens; vídeo do Commons: apertou o play sozinha, 25 s, 125 quadros, som
em 124/125 trechos, surpresa ao prever o próximo quadro 0,31 → 0,08. RAM da mente
nova 374 MB (córtex ~70 MB). Protótipos antigos (32×32) ficam no estado, sem uso.

## Transplante 6 (08/10 ~20h45) — córtex que enxerga letras; ligação texto-imagem contra o acaso

Análise dos primeiros 20 min da viva com os sentidos novos (pedida pelo usuario):
- o córtex v1 PAROU de aprender em ~4.000 olhadas (erro 0,175 → 0,030 e plano até
  9.000): teto de capacidade (a olhada inteira espremida em 64 números) + ladrilhos
  8×8 sem sobreposição (reconstrução quadriculada);
- palavras visuais COLAPSADAS: uma só levava 69% do uso; as 172 ligações texto-imagem
  eram quase todas frequência ("About", "Us", "is", "do" ligados à mesma palavra
  visual) — o controle de mentira chegou a 101;
- a pausa de 0,18 s por parada do olhar era humanização cosmética (minha);
- o 1º vídeo não tinha régua de tédio: assistiu até o teto de 90 s.
Também visto: buscas com pedaços de texto ("by the", "their", "com/") eram 87% das
ações e rendiam 3% (4/136); o órgão 'rende' (nascido no transplante) as derrubou
sozinho para 18% em ~13 min. Sem conserto meu: é ela aprendendo.

Consertos:
- **Córtex v2** (`cortex.py`; v1 guardado em `cortex_v1.py`/`cortex_v1.npz`): RETINA
  (tira o brilho médio e normaliza o contraste de cada olhada — sem isso a v2 aprendia
  só a desenhar fundo branco) + convoluções que se sobrepõem (filtro 4, passo 2) +
  MAPA RETINOTÓPICO 8×8×32 (2.048 números; cada célula ≈ 22 px ≈ uma letra) +
  preditor temporal LOCAL (vizinhança 3×3 dos dois quadros anteriores). Gradientes
  conferidos por diferenças finitas. Mesma tela de texto, 600 passos: erro em pixels
  0,0212 (v1) → 0,0056 (v2); a reconstrução da v2 é LEGÍVEL ("Cópia arq", "e outubro",
  "www.fifa.c", com o azul dos links); a v1 dava barras cinzas. Custo 46 ms/passo.
- **Palavras visuais por região**: cada célula do mapa com contraste vira palavra
  (fundo liso não vira). Para ligar imagem a texto usam-se só as 2 palavras mais
  RARAS presentes (vistas ≥ 3 vezes).
- **Ligação só conta acima do acaso**: por palavra visual, confirmações reais × do
  controle de mentira; válida se binomial unilateral < 5% (e ≥ 3 reais). Convergência
  texto-imagem-som só com ligação válida. Painel mostra confirmadas / mentira / acima.
- Sem pausa artificial no olhar (~8 paradas/s); vídeo para por HABITUAÇÃO (surpresa
  de prever o próximo quadro abaixo da metade da do início, medida após 2 s; mín. 15 s).
- `viver.py` passa a obedecer PARAR/REINICIAR (salva antes de sair).
Migração: ligações e palavras visuais presas ao código v1 apagadas (o resto da mente
fica); córtex renasce. Teste na mente de teste: migração OK, 0 erros; teste de
compatibilidade da viva OK (passo 7394).

## Transplante 6b e 7 (08/10 ~21h20–21h35) — estatística honesta, moeda única, vocabulário estável

Achados de 33 min com o córtex v2: 82% das ações = "artigo aleatório da Wikipédia-pt"
(viagem de 4 s, paga 20%: caça-níquel); 1 vídeo na vida inteira (o da Lua: 101 s
contados como CUSTO — erro meu: o tempo que ela escolhe ficar era cobrado como espera);
vocabulário visual bateu o teto de 4.096 em 30 min e reciclava palavras (ligações
texto-imagem no acaso: 258 reais × 269 de mentira); as 7 "acima do acaso" eram
falsos positivos de muitas comparações.
- 6b: correção de Benjamini-Hochberg (5%) sobre todas as palavras visuais testadas;
  escrita de arquivos do painel com nova tentativa (conflito de leitura no Windows).
- 7: MOEDA ÚNICA — ganho de uma experiência = fração do próprio erro de previsão que
  ela removeu (texto: bits na reserva da página; visão: olhada não treinada; vídeo: cada
  quadro previsto antes de ser visto), por segundo; "pagou" = acima da mediana da vida
  com TODAS as modalidades juntas. CUSTO = só a viagem até a experiência (o tempo que
  ela escolhe ficar entra no ganho por segundo; valor marginal, Charnov 1976); custos
  antigos descartados. VOCABULÁRIO VISUAL ESTÁVEL (`digestao.Vocabulario`, k-médias
  online, 1.024 palavras, nenhuma reciclada); 345 ligações presas a palavras recicladas
  apagadas. Teste: viagem até vídeo 7,5 s (antes 101 s); compatibilidade OK; viva
  acordou no passo 8127 e a 1ª escolha foi a porta de imagens.

## Transplante 8 (08/10 ~22h) — VERBOS VISUAIS e IMAGINAÇÃO ligados ao organismo (PRÉ-REGISTRO)

Pergunta do usuario: ao ver a Lua "girar", ela extraiu a rotação como algo transportável e
componível? Não (o preditor só antecipava quadros; nada separava transformação de cena;
e o aprendizado estava no córtex v1). E o achado que mudou o rumo: os órgãos fortes do
projeto (AGI 2.0/novaphys, núcleo ARC, representacao, encarnar) NUNCA foram ligados à
mente viva, e a viva (v1) não imagina: a imaginação da v2 acertou 29 × 13 do placebo
e a regra do duelo (só texto, minha) não via isso.

Órgãos ligados (mesma vida, mesma economia, com controle):
1. **Verbos visuais** (`verbos.py`): E tira de (mapa antes, depois) um código τ de 8
   números com média sobre a imagem (o "como mudou", sem o "o quê"); F aplica τ a um
   mapa (com coordenadas); G compõe dois τ sem aplicar (segunda ordem). Aprende SÓ da
   vida dela: quadros seguidos de vídeo e micro-sacadas do próprio olho (8 px, comando
   conhecido). Verbos = k-médias estável de τ (até 64), sem rótulo. Gradientes conferidos.
2. **Imaginação visual verificada**: no vídeo, a cada 1 s ela aplica o verbo de agora 5
   vezes e imagina o quadro de daqui a 1 s; quando ele chega, compara com placebo (verbo
   de outro momento da vida dela, aplicado igual) e com "nada muda".
   Em imagens paradas, imagina a imagem sob um verbo dela (rotação mental) — SEM
   verificação, só mostrado no painel e rotulado assim.
3. **Imaginação de linguagem** (órgão da v2 transplantado): no sono e ENQUANTO lê (página
   que ensinou acima da mediana vira molde), troca uma peça pela vizinha de uso e espera
   ver no mundo; placebo de mesma frequência sem parentesco. Orçamento AUTORREGULADO:
   12 × (confirmadas+1)/(placebo+1), entre 6 e 60 por sono.
4. **Sonda** (a cada 10 sonos; só mede, nunca treina): 24 olhadas dela com transformações
   conhecidas (girar 12°, aproximar 1,2×, deslizar 8 px, cor/brilho).

Linhas de base medidas ANTES (cortex de teste, mesmas 24 olhadas):
  órgão sem treino:            separa 0,615 · transporte vence 0,021 · composição vence 0,000
  treinado só com deslizar:    separa 0,896 · transporte vence 0,500 (girar 0,92, aproximar
                               0,63, deslizar 0,46, cor 0,00) · composição vence 0,333
  → "separa" é medida FRACA (acaso real ≈ 0,6, não 0,25); valem transporte e composição.

Critérios (na viva, primeira sonda após ≥ 2 h com o órgão):
  V1 TRANSPORTE: vence > 0,25 E a família "girar" > 0,5 → ela tem um verbo de girar que
     atravessa cenas. (Cuidado já visto: só deslizar já deu girar 0,92 no teste — então
     V1 só conta como ROTAÇÃO se girar > deslizar-treinado-só (0,92) OU se a razão
     mediana de erro do girar < 0,9.)
  V2 COMPOSIÇÃO: vence > 0,5 (girar→aproximar com verbos de OUTRA olhada bate "só um" e
     "verbos errados").
  V3 IMAGINAÇÃO VISUAL: em ≥ 50 conferências, imaginado < placebo E < parado em > 50%.
  V4 IMAGINAÇÃO DE LINGUAGEM: confirmadas > placebo, binomial unilateral p < 0,05.
Minhas previsões: V1 0,35 (rotação depende de ela VER rotação em vídeo); V2 0,25;
V3 0,55; V4 0,7 (a v2 já passou).

## Transplante 9 (08/10 ~22h40) — atenção que volta e acelera, divagar, inferir, corpo (PRÉ-REGISTRO)

1. **Voltar e acelerar** (lendo): bloco com surpresa acima do q90 da vida = "não entendi";
   depois de seguir adiante, o olho VOLTA e relê (1 vez por página, mesma tela). Metade das
   vezes PLACEBO: relê um bloco que ela tinha entendido. Bloco cujo começo está abaixo do
   q25 = previsível: lê só 30% (passa o olho).
   R1: com ≥ 60 páginas de cada, progresso da página (reserva) com releitura verdadeira >
   com placebo. Previsão 0,4 (o efeito, se houver, é pequeno e a medida é ruidosa).
2. **Divagar**: braço novo da MESMA economia ("divagar"): revive olhadas antigas (replay) ou
   sonha acordada (olhadas antigas transformadas pelos verbos dela, e aprende com o que
   imaginou). Ganho medido no PRESENTE (16 olhadas mais recentes, fora do replay).
   R2: ganho médio no presente > 0 E ela continua indo ao mundo (divagar < 60% das ações).
   Previsão 0,5. Referência humana: ~47% do tempo acordado divagando (Killingsworth e
   Gilbert 2010) — não é meta, é régua.
3. **Inferir** (`inferencia.py`, princípio do núcleo do ARC): a regra mais curta (constante,
   inércia, ciclo, markov; MDL) da sequência de verbos do vídeo prevê o próximo verbo ANTES.
   R3: com ≥ 200 previsões, regra > max(o mais frequente, repete o último) + 5 pontos.
   Previsão 0,25: com 3 verbos borrados, a regra tende a ser "inércia" = a aposta burra.
   Limite declarado: DSL do ARC e inventor de grandezas pedem OBJETOS rastreados; não ligados.
4. **Corpo** (o próprio olho): cada micro-sacada tem comando conhecido; modelo de corpo
   (comando → mudança, mínimos quadrados) testado nos 60 pares mais recentes, fora do ajuste.
   R4: acerta o comando > 0,5 (acaso 0,25; placebo ≈ 0,25). Mede também quantas mudanças
   do MUNDO (vídeo) ela tomaria por movimento próprio (sem critério; descritivo).
   Previsão 0,6.
DESVIO (22h36): divagar como BRAÇO da economia dominou (22 de 25 ações; custo ~0,5 s contra
5–10 s das outras ações) com ganho no presente quase sempre NEGATIVO (−0,27 a +0,08): reviver
o passado atrapalhava ver o agora, e o "pagou" binário não punia o prejuízo. Trocado por
divagar no TÉDIO (3 últimas experiências abaixo da mediana da vida), com chance = fração das
divagações recentes que ajudaram o presente (5%–90%) e modo (reviver/sonhar) escolhido pelo
que cada um rendeu. R2 passa a ser: ganho médio no presente > 0 depois da autorregulação.

## Transplante 10 (09/10 ~22h30) — o LABORATÓRIO e a porta ARENA (PRÉ-REGISTRO)

Origem: revisão externa de 09/10 (GPT, lendo o ARQUITETURA.md inteiro) e a auditoria que
a confirmou: os mecanismos que sobreviveram aos controles no AGI 2.0 (objetos pela
continuidade, invenção de grandeza, reanálise com olhos frescos, crise contra o ruído,
conceito julgado pelo que paga) nunca tinham sido ligados à vida. Confirmado também: a vida
contínua é a v1 com transplantes (a v1 venceu as gêmeas 4 × 0 pela regra registrada antes);
a plasticidade local da v2 ficou fora por essa regra, não por esquecimento.

Órgãos (`laboratorio.py`, `arena.py`):
1. **Laboratório**: TODO vídeo que ela assiste (internet ou arena) passa também por ele —
   objetos = o que se move contra o fundo, identidade pela continuidade, permanência (o que
   some até 3 s e reaparece onde iria estar é a mesma coisa); o episódio CRU fica guardado
   (até 3.000); teorias de movimento com a 1ª tentativa medida ANTES de aprender; crise
   quando o próprio vídeo, sozinho, prevê melhor que a teoria (ou o erro é anômalo nas duas
   escalas); reanálise por consenso que reinterpreta episódios guardados; grandezas
   conservadas inventadas e julgadas pelo que pagam na previsão; vocabulário de conceitos
   compartilhado entre teorias. Fica em `laboratorio.pkl` (atômico, com a anterior).
2. **Porta "arena"**: um mundo físico (giro, mola_h, contragiro, mola; troca a cada 12
   visitas, sem aviso) que ela assiste em tempo real, 5 quadros/s. O estado do MUNDO fica em
   `arena.json`; a mente nunca o lê.
NÃO muda: a economia. O laboratório aprende e é medido; o que ela escolhe continua decidido
pela moeda única (a arena paga como qualquer vídeo: pelo que o córtex aprende a prever).

Validação offline ANTES (PRE_REGISTRO_laboratorio.md, 10 vidas registradas): 8 de 12
critérios — revisão para h 9/10, conceito paga 0,53× em 10/10, reuso paga 10/10, crise na
troca 10/10, mola sem conceito 9/10, permanência 0,71 × 0,18; FALHARAM a identidade exata
de s² (5/10), o mesmo conceito no contragiro (5/10), o placebo (0,88×, exigia ≥ 0,90) e
reanálise × conclusões (6/10).
Teste da mente (`teste_laboratorio.py`): mente nova, porta aparece, 7 visitas, salva,
ACORDA com o laboratório igual, painel recebe o resumo, 0 erros de órgão no diário.
Antes de ligar na viva: `python teste_laboratorio.py vidas/viva` (trabalha numa CÓPIA).

Critérios na vida (medidos com `python marco_laboratorio.py vidas/viva`, que só lê):
  A1 ESTÁVEL: nas primeiras 6 h com o órgão, 0 eventos `erro_orgao` do laboratório e
     nenhuma queda nova do guardião. Previsão 0,8.
  A2 ECONOMIA (descritivo, com alarme): fração das ações em `porta:arena` nas primeiras
     24 h. Previsão 2–15%. Acima de 50% = caça-níquel: conserto registrado.
  A3 DESCOBERTA NA VIDA: com ≥ 96 visitas à arena (2 ciclos dos 4 universos), existe
     conceito com |corr(s²)| ≥ 0,99 E conceito com |corr(h)| ≥ 0,99. Previsão 0,45.
  A4 INTERNET (descritivo): fração dos vídeos da internet com objetos rastreáveis;
     quantos uma teoria já existente explica; se nasce algum conceito de teoria só de
     internet. Previsão: poucos objetos (câmera que mexe = "fundo_instavel"), muitas
     crises, nenhum conceito útil — o que der fica escrito.
Limites declarados: o alfabeto (14 monômios) e os mundos da arena são meus; ela só
ASSISTE (não age); 5 quadros/s limitam a precisão das velocidades (a causa medida da
falha de L1); o laboratório não fala com a linguagem nem com os verbos visuais ainda.
Painel (só leitura, nada entra em decisão): abas LABORATÓRIO, DESCOBERTAS e HOLOGRAMA,
com os arquivos `laboratorio_vivo.json`, `lab_quadro.jpg`, `holograma.json` e
`verbos_anim.jpg` que a mente publica.

## Transplante 11 (10/10 ~00h) — MÃOS, VÍDEO MENTAL e o laboratório NA ECONOMIA (PRÉ-REGISTRO)

Origem: o pedido do usuario (09/10: "ela deveria ir sozinha", imaginação universal, "um
vídeo mental") e a segunda revisão do GPT (experimentos que separam teorias rivais).

Órgãos (todos em `laboratorio.py`, seções 5 e 6; `arena.py` aceita `?exp=`):
1. **Laboratório na moeda única**: cada episódio paga o PROGRESSO de aprendizagem daquela
   região do mundo (a 1ª tentativa, sempre fora da amostra, contra a mediana das 10
   anteriores da mesma fonte e do mesmo tipo — assistido ou montado), em [−1, 1].
2. **Ação "experimentar"** (aparece quando há teoria da arena): ela monta a cena onde as
   rivais dela mais discordam; depois de ver, o laboratório julga qual rival previu melhor
   e, se outra opção de conceito venceu, refaz a sala de espera na hora.
3. **Vídeo mental**: o laboratório guarda a aparência das últimas 40 cenas (fundo + recorte
   de cada coisa) e imagina filmes novos com as leis dela; o painel mostra o filme que ela
   imagina agora (HOLOGRAMA).
4. **Modo de divagar "sonhar_lab"**: filmes imaginados treinam o preditor de tempo do
   córtex; o ganho é medido na surpresa do último vídeo REAL; compete com "reviver" e
   "sonhar" pela média dos ganhos (exploração de 20%, como antes).
NÃO muda: o algoritmo registrado do laboratório (vida registrada re-rodada: 72/72 idêntica).

Validação offline ANTES:
- Mãos (PRE_REGISTRO_experimento_ativo.md, 10 vidas): E4 PASSA (8/10, os experimentos
  separam as rivais); E1, E2, E3 FALHAM — escolher onde as rivais mais discordam a fez prever
  PIOR o mundo típico (5,56 px × 3,60 ao acaso × 3,27 só assistindo). Por isso
  "experimentar" é só mais uma ação, paga pelo que rende.
- Vídeo mental (PRE_REGISTRO_video_mental.md, 10 vidas): o filme imaginado antes bate com o
  real melhor que "tudo reto" (10/10) e que a lei do outro mundo (10/10); sonhar com a lei
  dela ensina a visão mais que sonhar reto (9/10), não mais que rever vídeos reais (2/10).
Teste da mente (`teste_laboratorio.py`): + 3 experimentos forçados, progresso na moeda,
cenas guardadas, `video_mental.jpg` publicado; `teste_compat.py` numa cópia de mente antiga: OK.

Critérios na vida (medidos com `python marco_laboratorio.py vidas/viva`, que só lê, e com o
diário):
  B1 ESTÁVEL: nas primeiras 6 h com o transplante 11, 0 eventos `erro_orgao` (laboratório,
     mãos) e nenhuma queda nova do guardião. Previsão 0,8.
  B2 MÃOS (descritivo, com alarme): fração de "experimentar" entre as ações de arena
     (porta:arena + experimentar) nas primeiras 24 h. Previsão: menor que a de porta:arena.
     Alarme: "experimentar" acima de 50% de TODAS as ações = caça-níquel, conserto registrado.
  B3 SONHAR COM O LAB (descritivo): fração das divagações em "sonhar_lab" e o ganho médio.
     Previsão: vira o modo mais escolhido quando disponível (o preditor de tempo é o que mais
     tem a aprender); se a média dos últimos 20 ganhos ficar negativa, ela mesma para.
  B4 IMAGINAÇÃO CONFERIDA NA VIDA: com ≥ 24 visitas à arena desde o transplante, o filme
     imaginado pela teoria atual vence o "tudo reto" em ≥ 8 de 10 filmes (marco, universo
     atual). Previsão 0,7 (cai se a teoria atual for uma hipótese recém-nascida de uma troca).
Limites declarados: imaginação plana + tempo (sem 3D); só coisas que se movem contra um fundo
parado; a aparência é colada, não gerada (ela não inventa coisas que nunca viu, só lugares e
movimentos novos para elas); o filme não tem som; e o conceito do laboratório ainda não fala
com a linguagem nem com os verbos visuais (identidade conceitual entre órgãos: próximo passo).
