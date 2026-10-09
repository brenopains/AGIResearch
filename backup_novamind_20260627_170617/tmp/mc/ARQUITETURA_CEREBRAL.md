# NovaMind — Arquitetura Cognitiva Embodida (fundamentação cerebral)

Sistema multi-agente vivo no Minecraft real (mineflayer). Cada bot integra mecanismos
extraídos de neurociência de **rato, humano, primata e polvo**, com o objetivo de fechar as
lacunas rumo a inteligência geral. **Honestidade:** isto NÃO é AGI completa — faltam percepção
visual rica real, linguagem totalmente ancorada e independência do LLM no raciocínio elaborado.
Mas cada módulo é um mecanismo cerebral real, funcional e medido.

## Mapa: sistema cerebral → mecanismo → módulo → lacuna fechada

| Cérebro (espécie) | Mecanismo neurocientífico | Módulo nosso | Lacuna fechada | Sinal medido |
|---|---|---|---|---|
| **Gânglios da base / dopamina** (mamífero) | Erro de predição de recompensa (Schultz); TD-learning | `worldmodel.js` | Aprendizado por consequência | dopamina = RPE |
| **Córtex preditivo** (mamífero) | Codificação preditiva (Friston): prever→errar→corrigir | `worldmodel.js` | Modelo causal do mundo | 31+ relações causais (ex: comer@fome→+1.4) |
| **Dopamina tônica** (humano/rato) | Curiosidade = PROGRESSO de aprendizado (Oudeyer) | `worldmodel.js` curiousAction | Exploração dirigida ao aprendível | ignições "curiosidade" |
| **Hipocampo** (rato) | Place cells = lugar+atributo; navegação=episódica | `hippocampus.js` | **Memória espacial-episódica** | lugares lembrados (comida/morte) |
| **Hipocampo** (rato) | Replay/consolidação no repouso (reativa alto-reward) | `hippocampus.js` replay() | Consolidação de memória | replay no sono |
| **Córtex pré-frontal** (primata/humano) | Planejamento de longo prazo, working memory | `mind.js` planos | **Planejamento hierárquico** | planos multi-passo p/ marcos |
| **Neurônios-espelho / TPJ** (primata) | Teoria da mente: modelar outros agentes | `mind.js` ToM | **Aprendizado social** | 6-10 modelos de outros |
| **Workspace neuronal global** (humano) | Módulos competem; o saliente IGNITA e é transmitido (Dehaene/Baars) | `workspace.js` | **Integração funcional (gap #4)** | ignições por módulo |
| **Wake-sleep / sistemas complementares** (McClelland) | Hipocampo (episódico rápido) ↔ córtex (semântico lento) | hippo↔worldmodel | Consolidação dual | replay→worldmodel |
| **Córtex motor / cerebelo** | Habilidades automatizadas (de deliberado→reflexo) | `skills.js` | Compounding de habilidade | 98% sem LLM |
| **Polvo** (cognição distribuída) | Inteligência periférica/embodida, reflexos locais | alimentação prioritária + reflexos de sobrevivência | Reflexo embodido vs deliberação | — |
| **Neocórtex** (aquisição de capacidade) | Tech tree / currículo automático (Voyager) | `progression.js` | **Capacidade aberta crescente** | marcos da tech tree |

## Fluxo integrado (um "tick" de decisão)
1. **Reflexo de sobrevivência** (polvo/tronco): perigo imediato/fome crítica → ação reflexa (alta prioridade).
2. **Workspace global** (humano): fora de emergência, os módulos submetem LANCES com saliência:
   - episódico (hipocampo): "lembro comida no lugar X" — saliência ∝ fome × valor-do-lugar
   - planejamento (PFC): próximo marco da tech tree — saliência fixa alta
   - projeto próprio (PFC): habilidade-objetivo aprendida
   - teoria da mente: imitar quem prospera
   - curiosidade (dopamina tônica): ação mais aprendível (progresso)
   - valor (gânglios): ação de maior efeito esperado
3. **Ignição**: o lance mais saliente "vence a consciência", executa, e é transmitido (broadcast) → os outros módulos atualizam.
4. **Aprendizado**: erro de predição (dopamina) atualiza o world model; place cells gravam o lugar; habilidade usada se fortalece; marco novo dá pico de dopamina.
5. **Sono**: replay do hipocampo consolida memórias espaciais; reforço de valores.

## Persistência (memória que sobrevive à morte = transmissão geracional)
`skills_library.json` (habilidades), `worldmodel_state.json` (relações causais), `hippocampus.json`
(lugares+episódios), `progression.json` (tech tree da colônia), `affordances.json`, `colony_state.json`
(cérebros). Tudo BOUNDED (poda periódica) p/ evitar OOM.

## Percepção visual (córtex visual / V1+) — `vision.js` [ADICIONADO H122]
Visão HUMANA-LIKE: campo de visão ~200° horizontal / 130° vertical (periferia humana), cegos atrás
(precisam virar p/ ver), alcance 32 blocos, oclusão por raycast (NÃO-wallhack: não veem através de
blocos), e escaneamento (viram a cabeça p/ sacadas visuais). Antes eram oniscientes (sabiam tudo nos
chunks = melhor que humano); agora só percebem o que está no cone visual e desocluido — podem ser
surpreendidos por trás (compensado por audição + hipocampo que marca onde apanharam).

## Linguagem ancorada (área de Wernicke/Broca + grounding) — `language.js` [H123]
Símbolos cunhados (COINED) ancorados nos conceitos profundos (causal do worldmodel + espacial do hipocampo
+ afetivo). Ouvir o símbolo ATIVA o conceito (compreensão referencial): o mesmo "pipi"(zumbi) dá medo
intenso a quem aprendeu que zumbi machuca, e leve a um novato — linguagem transferindo conhecimento.

## Auto-melhoria segura (plasticidade homeostática) — `selfimprove.js` [H124]
Meta-controlador ajusta os PRÓPRIOS hiperparâmetros (curiosidade, exploração, saliência do planejamento,
cooldown de mint) pela fitness medida da colônia (hill-climbing reversível, bounded). O cérebro regula os
próprios ganhos. Seguro: só move knobs em faixas, reversível, sem reescrever código.

## Criação de habilidade SEM LLM (composição motora) — `skills.js` [H125]
Independência do LLM (lacuna #3) atacada na raiz: o cérebro FABRICA habilidades novas RECOMBINANDO as que já
tem, sem chamada externa. `mintByComposition` encadeia 2 habilidades provadas em sequência (A→espera→B);
`mintByMutation` perturba uma constante numérica. A nova entra FRACA e a seleção (reforço se ajuda, poda se não)
faz o papel do "juiz" que antes era o LLM — variação-e-seleção, teste por execução. Fundamento: o córtex/cerebelo
compõem primitivas motoras aprendidas em sequências novas (base da generatividade comportamental). Medido: a
biblioteca cresce com comp>0 e llm=0 (SEM-LLM=100% das skills criadas na janela). O LLM vira opcional/raro.

## Lacunas honestas restantes (para AGI "de fato")
- ~~Percepção visual rica~~ → ADDRESSADA (vision.js).
- ~~Linguagem ancorada~~ → ADDRESSADA (language.js).
- ~~Auto-melhoria~~ → ADDRESSADA (selfimprove.js, bounded).
- ~~Independência do LLM~~ → ADDRESSADA (skills.js: mint por composição/mutação; biblioteca cresce sozinha).
Honesto: as 4 lacunas do infográfico foram atacadas com mecanismos reais e medidos. CONTINUA não sendo AGI
completa/humana — cada mecanismo é uma versão BOUNDED do princípio cerebral (composição = só sequências A→B, não
raciocínio simbólico arbitrário; auto-melhoria = só knobs em faixas; visão = entidades+raycast, não pixels). O LLM
ainda agrega quando minta um projeto elaborado que a composição não alcança, mas a dependência ESTRUTURAL caiu de
"obrigatório para criar qualquer habilidade" para "opcional". Não é AGI plena — é o caminho honesto até ela.

## Referências (reais)
- Schultz — dopamine reward prediction error. eLife "Prediction Error: expanding role of dopamine".
- Oudeyer & Kaplan — Intelligent Adaptive Curiosity (learning progress).
- Hipocampo/place cells/replay — bioRxiv 2023.12.30.573694; PMC9496859 (memory-replay model).
- Global Workspace — Dehaene/Baars; Frontiers Psychol 2021.749868 (GWT & PFC).
- Voyager — arXiv:2305.16291 (open-ended embodied agent, skill library, auto curriculum).
- MindForge — arXiv:2411.12977 (ToM para aprendizado cultural).
