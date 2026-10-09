# Próximas fases — da vida 2 até a internet aberta

Atualizado em 2026-10-05, depois da vida 1 e das correções do usuario (LLMs não
viveram a internet como ambiente; o contraexemplo que eu dei não vale para um
organismo com currículo próprio) e da proposta da outra IA (escopo aprendido;
sensores preparados mas semanticamente ignorantes; micro-internet antes da aberta).

A tese, na forma que eu assino: **a digestão não deve ser entregue pronta; o que
precisa ser entregue é um organismo capaz de descobrir como digerir.**

## Fase 2a — alcance aprendido (EM ANDAMENTO: `PRE_REGISTRO_vida2.md`)

Cada estrutura aprende quão universal ela é. Portão duro: vencer a MELHOR
largura fixa, não só a minha antiga.

## Fase 2b — mesoestado mais rico

O mesoestado da vida 1 guardava dado, candidatos e melhor lei — e empatou com o
episódio bruto. O que ainda não guarda (proposta, a testar contra o atual):
- **o que quase encaixou**: a assinatura do RESÍDUO de cada tentativa falha
  (o que sobrou sem explicação) — reacender quando um resíduo novo se parece
  com um antigo, não quando o contexto inteiro se parece;
- **expectativas pendentes**: "se este mundo voltar, a lei L deve valer" —
  testadas de graça quando voltar, e a refutação vira informação;
- **conflitos**: duas leis válidas em subconjuntos diferentes do mesmo dado
  (sinal de variável escondida — o caminho para inventar estado).
Portão: rico − atual em recorrências, com placebo de resíduo embaralhado.

## Fase 2c — a linguagem cognitiva que cresce

O metacontrole custou 2,2× para empatar com a ordem fixa porque escolhe entre
poucas operações que eu escrevi. Coalizão já é operação composta; falta a
operação NOVA nascer de padrão de falha (oito nulos do projeto dizem que propor
de fora falha; construir do dado funciona — o caminho é derivar operação da
sequência de residuos, não sortear).

## Fase 3 — micro-internet controlada (antes da aberta)

**Mundo:** um recorte pequeno e auditável servido localmente, que eu consiga
inspecionar inteiro: artigos (Wikipédia em português, ~300 páginas de física,
biologia, história), páginas que se contradizem, ficção de domínio público
(Machado de Assis, Gutenberg), tutoriais e documentação, discussões, e
**vídeos curtos COM ÁUDIO** de licença livre (Wikimedia Commons: experimentos de
física, pêndulos, órbitas, animais). Ninguém diz aonde ir.

**Corpo:** navegador real (Playwright), só-leitura: pesquisar no índice local,
abrir link, ler trecho, rolar, voltar, assistir trecho de vídeo. Nada de login,
formulário ou envio.

**Sensores — nascem prontos (estrutura, não semântica):**
- texto: bytes/caracteres Unicode, sem dicionário, sem tokenizador;
- vídeo: quadros em baixa resolução, luminância, diferença temporal, talvez
  amostragem log-polar;
- áudio: energia em bandas de frequência ao longo do tempo (uma cóclea fixa);
- **áudio e vídeo SINCRONIZADOS no mesmo relógio** — o organismo recebe o par
  (quadro, bandas) a cada instante, para que coincidência temporal seja dado;
- memória temporal e capacidade de associar.

**Não nascem prontos:** palavra, fonema, objeto, "é o mesmo objeto", agente,
causa, significado. Aparecem como estrutura sem nome (`estrutura #839`) e só
depois, talvez, se ligam ao nome que os humanos usam.

**Mecanismos de desenvolvimento (gerais):** agrupar o que se move/varia junto,
prever o próximo instante (texto, quadro, banda), comparar episódios, testar
intervenções (no navegador: escolher o que abrir), formar invariâncias,
coincidência áudio-vídeo.

**Dois braços de olho** (proposta da outra IA, que assino): um com codificador
visual pré-treinado (para testar memória, mesoestado, metacontrole JÁ, sem
esperar a visão), outro com o sensor quase cru. A reivindicação final só vale
para o braço sem semântica pré-treinada.

**Precedentes (ditos antes):** aprender correspondência áudio-vídeo sem rótulo
já foi feito (Arandjelović & Zisserman, "Look, Listen and Learn", 2017);
segmentação semântica emergindo sem rótulo também (DINO, 2021) — com dataset e
computo enormes. Agentes de curiosidade e modelos de mundo existem. O que não
existe: isso acontecendo incrementalmente, ativamente, dentro de um organismo
com ontogenia, procedência e mesoestado, numa máquina de 4 GB de vídeo.

**Portões da fase 3 (a pré-registrar):**
1. Aprendizagem ativa: o organismo que navega sozinho prevê páginas e trechos
   novos melhor do que o MESMO organismo recebendo as mesmas páginas em ordem
   sorteada, e melhor que RAG.
2. Reuso: estrutura nascida numa página reduz a surpresa em outra página de
   tema diferente (alcance aprendido, agora entre modalidades/temas).
3. Ficção: ideia nascida em ficção ajuda num conteúdo científico SEM a crença
   da fonte de ficção subir.
4. Volta espontânea: o organismo retorna, sem instrução, a algo visto horas
   antes quando o contexto pede (evento de reativação).
5. Áudio-vídeo: estrutura que liga um padrão de banda a um padrão visual
   surge, e removê-la piora a previsão dos dois.

**Precisa da sua autorização antes:** baixar o recorte (texto ~50 MB; vídeos
~100–300 MB, só licença livre), instalar Playwright e um navegador
(~150 MB). A máquina aguenta texto folgado e vídeo em baixa resolução.

## Fase 4 — internet aberta

Só depois dos portões da fase 3. Mesmo corpo, mesmos limites de segurança.

## O que fica meu, dito para não parecer resolvido

O alfabeto do material, a lista de operações, a percepção de contexto da física
e os sensores de base. A fase 2c e o braço cru da fase 3 são os que atacam isso.
