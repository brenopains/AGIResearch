# Organismo cognitivo — desenho, precedentes e mapa das ideias

Data: 2026-10-05. Código em `D:\AGI_Inventor\organismo`. O projeto original
(`C:\Users\usuario\Documents\AGI 2.0`) foi só lido.

## O que já existe no mundo (dito antes de construir)

| Ideia da pasta | Precedente mais próximo | O que ele NÃO fez |
|---|---|---|
| Mesoestado: ativação que decai e reacende por contexto | ACT-R (ativação de base + espalhamento); ECAN do OpenCog (importância de curto/longo prazo); LIDA (coalizões no espaço de trabalho global); memória prospectiva | Não acumula o que a mente *pensou* sobre um problema entre encontros como objeto executável |
| Sem números mágicos; recursos disputados | NARS (Pei Wang): AIKR, "bag" com orçamento prioridade/durabilidade/qualidade | Nenhum resultado de nível humano |
| Crença ≠ potencial gerativo | NARS separa valor-verdade de orçamento | Não usa ficção como fonte de ideias com crença baixa |
| Rigidez variável conforme a falha | Temperatura do Copycat (Hofstadter & Mitchell 1993); recozimento simulado | Domínio de analogias de letras |
| Módulo = hábito consolidado | chunking do Soar; compilação de produções do ACT-R; DreamCoder; biblioteca de habilidades do Voyager | Voyager depende de LLM |
| Orçamento interno aprendido | Metarraciocínio racional (Russell & Wefald 1991; Lieder & Griffiths 2017); seleção de algoritmo | — |
| Internet como mundo | WebGPT, agentes de navegador, Voyager | Todos centrados em LLM pré-treinado |

**O que este passo testa e ninguém testou:** essas dinâmicas *juntas*, por
cima de um material que inventa conceitos a partir do dado (o que no projeto
passou portão com placebo), num teste em que o código fica congelado, os
problemas exigem conceitos ausentes, e o reuso é medido em perguntas lacradas
com ablação e restauração. Não é "a mente de nível humano": é o primeiro teste
de que essa organização paga.

## Três camadas

1. **Substrato** (invariável, "física do cérebro"): material do domínio
   (`material_fisica.py`: lei por MDL ortogonal, grandeza por autovetor
   generalizado, direção derivada, preditor de pergunta), HDC de contexto
   (`hdc.py`), contabilidade de custo, verificação interna, percepção do
   contexto. Constantes declaradas em cada arquivo; nenhuma é teto de conteúdo.
2. **Organismo plástico** (`organismo.py`): tudo é *estrutura* — operações do
   material, conceitos, direções, leis, problemas abertos, coalizões. Nenhuma
   tem existe/não existe: tem registros (contexto, pagou?) → crença de eficácia
   por contexto (Beta com núcleo de semelhança HDC), força, estabilidade,
   ativação recente, potencial gerativo, fonte. Escolha = Thompson do valor por
   custo. Orçamento externo por problema; distribuição interna aprendida.
3. **Metaplasticidade**: a constante de esquecimento por tipo de estrutura se
   ajusta pelo que a reativação de algo latente rende; o custo de cada operação
   é aprendido pelo uso (por unidade de trabalho).

## Mapa: cada ideia → mecanismo → como é testada

Arquivos da pasta `ideias`: (A) "A mente tem, de forma dinâmica…", (B) "ta, mas
notei…", (C) "2. Ela aprenderia isso…", (D) "Cara, sendo bem honesto…",
(E) "Sim, e acho que depois de validarmos…", (F) "Sim, e nem precisar dar
mecanismos…", (G) "Sim, e sem dar links…", (H) "Sim, mas ser flexível…".

| # | Ideia | Mecanismo | Chave / braço | Portão |
|---|---|---|---|---|
| A1 | Cognições persistentes (problema aberto, hipótese, resposta preparada) | estrutura `aberto` guarda dados de todos os encontros, candidatos tentados, melhor lei parcial, leis derivadas | `meso` / SEM_MESO | G2 |
| A2 | Ativação decai e reacende por contexto ("0,04 → 0,92") | ativação ACT-R + crença por contexto; evento `reativacao` com intervalo | `meso`; PLACEBO_MESO (contexto sorteado) | G2, ontogenia |
| A3 | Memória do que PENSOU, não só do que aconteceu | SEM_MESO guarda episódio bruto e recupera por contexto (RAG); ORGANISMO guarda o desenvolvimento | SEM_MESO vs ORGANISMO | G2 |
| A4 | Invenção = recombinação de estruturas latentes | conceito vira átomo de conceito novo (profundidade); sonho recombina abertos × estruturas novas | `sonho` | G8, G10 |
| A5 | HDC como campo de ativação, não depósito | HDC só codifica contexto; leis/dados/programas fora do vetor | substrato | — |
| A6 | Sonho/replay em ócio | sobra do orçamento → ajuste do aberto com estruturas nascidas depois dele; lei com fonte `sonhado` | `sonho` / SEM_SONHO | G10 |
| A7 | Pessoa A (ensaiou) × B (não ensaiou) | é exatamente ORGANISMO × SEM_MESO no mesmo fluxo | — | G2 |
| B1 | Orçamento externo, distribuição interna aprendida | metacontrole Thompson por contexto; custo aprendido | `meta` / SEM_META | G9 |
| B2 | Mesoestado como propriedade geral (potencial→…→dissolvendo) | rótulos derivados de (registros, ativação, estabilidade, força) no log | sempre | ontogenia |
| B3 | Módulo = hábito consolidado | `coalizao`: sequência que resolve em ≥2 problemas vira estrutura executável | `coal` / SEM_COAL | G4 |
| B4 | Vocabulário sem nome que vira conceito | conceito nasce como `c17` (sem nome), ganha força pelo uso | sempre | ontogenia |
| B5 | Metacontrole "estado X → estratégia Y rende" | crença por contexto em cada operação | `meta` | G9 |
| B6 | Comportamento novo pela regra geral de preservação | mesma regra de crédito para tudo | sempre | — |
| B7 | Três escalas (rápido, meso, consolidado) | espaço de trabalho do problema / abertos e ativação / estabilidade alta | sempre | — |
| C1/C2 | Aprender a escolher representações com a cultura; não imitar; confiabilidade da fonte aprendida | crença da fonte aprendida por corroboração de AFIRMAÇÕES (leis) em problema percebido | `proc2` | G6 (brinquedo); internet: fase 3 |
| C3/C5 | Metaestabilidade: falhar apesar do investimento → afrouxar | temperatura: cresce com gasto sem progresso e com falha recente; alarga o contexto, aplaina crença, deixa entrar o improvável | `regime` / SEM_REGIME | G5 |
| C4 | Ecologia: sem fitness único; estrutura que habilita outras | crédito atrasado aos pais e à operação que a gerou; potencial gerativo | `eco` / SEM_ECO | G11 |
| C6 | Espaço de arquiteturas; só energia/memória/tempo finitos | orçamento por problema + memória externa (MEM_PONTOS) com compactação | `tetos` / TETOS | G3 |
| C7 | Gradientes de existência | registros + força + estabilidade + ativação + potencial + fonte | `tetos` | G3 |
| D1 | Teste decisivo: código congelado, conceitos ausentes, reuso não antecipado | perguntas LACRADAS sorteadas antes; ablação contrafactual na mesma mente | — | G7 |
| D2 | "Falho de um jeito que não expresso" → criar representação | confiança calibrada por contexto (`op:confiar`) + refutação lembrada → não para na resposta que o mundo já reprovou | sempre | ontogenia |
| E1–E4 | Internet como ambiente; digestão com perda; régua "o que mudou dentro" | NÃO implementado nesta etapa (a própria ideia E diz "depois de validarmos"); a régua E4 é o relatório de ontogenia | — | fase 3 |
| F1 | Capacidades existentes como material | o material reimplementa lei, grandeza, direção, pergunta do projeto | — | — |
| F2 | Substrato / organismo / metaplasticidade | as três camadas acima | — | — |
| F3 | Hardcode só a capacidade de desenvolvimento | lista de constantes do substrato; braço TETOS religa os MAX_* | — | G3 |
| F4/F5 | Morfogênese; criar organização, não só representação | coalizões + abertos + leis viram anatomia funcional; log mostra | — | G4, ontogenia |
| G1–G5 | Browser como corpo; logs; preferências; ontogenia em 24 h; palavra aterrada | logs de ontogenia já completos aqui; browser na fase 3 | — | fase 3 |
| H1–H6 | Verdade ≠ utilidade; suspender restrição; ficção latente; recontextualizar | prior de LEI = fonte (crença); prior de IDEIA = potencial gerativo sob temperatura; nada é apagado por ser ficção; SEM_PROC2 apaga o que vem de fonte não crível | `proc2` / SEM_PROC2 | G6 |

## O que NÃO está aqui (dito para não parecer feito)

- Internet multimodal, browser como corpo, linguagem (E, G): fase 3, depois dos portões.
- Percepção que aprende (o contexto é percebido por traços fixos do substrato).
- Um segundo domínio fora da física: necessário antes de falar em generalidade.
- O alfabeto do material (monômios, expoentes, operações de direção) ainda é meu.
