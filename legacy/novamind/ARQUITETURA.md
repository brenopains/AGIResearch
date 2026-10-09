# NovaMind AGI — Arquitetura Cognitiva para AGI de Nível Humano/Superior

> **Nota (2026-07-28):** os scores dos 7 trials em `core/agi_trials/` e do
> loop em `run_evolution.py` citados neste documento **não são evidência
> de capacidade**. Auditoria de código encontrou: 4 dos 7 trials produzem
> resultado idêntico com o sistema cognitivo desligado (`cge=None`); o gate
> "score=0 se abaixo do acaso" documentado em `scoring.py` não está
> implementado; `run_evolution.py` quebra em runtime (`KeyError` zera
> toda a população) e os 12 parâmetros do genoma nunca são lidos por
> `core/`. Detalhes e correções propostas em
> `C:\Users\usuario\.claude\plans\c-users-usuario-downloads-leia-arquitetur-floofy-crab.md`.
> A fonte de verdade agora é `agent3/results/history.csv` (ARC-AGI-3,
> scoring externo do SDK oficial, offline).

## Princípios Fundamentais (Não Negociáveis)

> **NovaMind não é uma IA tradicional nem um LLM.**
> 
> Seus objetivos são:
> 1. **Generalização**: resolver problemas novos em qualquer área, nunca vistos durante o desenvolvimento
> 2. **Emergência**: capacidades superiores devem surgir da interação entre módulos, não de programação explícita
> 3. **Autonomia**: aprender, raciocinar e evoluir sem supervisão humana
> 4. **Rastreabilidade**: toda decisão deve ser auditável e reproduzível
> 5. **Modularidade**: nenhum componente conhece a implementação interna de outro — todos se comunicam por contratos
> 6. **Não-harcoded**: nenhuma regra específica de domínio pode limitar a arquitetura. O sistema deve poder evoluir sem reescrita.

### O que NovaMind NÃO é
- ❌ Não é um LLM (não usa transformers, atenção, ou next-token prediction)
- ❌ Não é um sistema especialista (não tem regras if-then escritas por humanos)
- ❌ Não é um chatbot (não otimiza para "parecer inteligente")
- ❌ Não é um banco de dados semântico (não só recupera conhecimento — raciocina)

### O que NovaMind É
- ✅ **Arquitetura cognitiva geral** — percepção → hipóteses → competição → justificação → resposta
- ✅ **Espaço hiperdimensional (HDV 10000D bipolar)** como linguagem universal de representação perceptual
- ✅ **Espaço hiperbólico (Poincaré ball)** como substrato de conhecimento hierárquico e raciocínio taxonômico
- ✅ **Álgebra Geométrica (Clifford)** para relações tipadas como bivectors (CAUSES, IS_A, PART_OF em planos distintos)
- ✅ **Princípio da Energia Livre (FEP)** para confiança calculada por surpresa Bayesiana, não constante
- ✅ **Topologia (homologia persistente)** para metacognição: Betti numbers revelam clusters, ciclos e lacunas do conhecimento
- ✅ **Aprendizado contínuo** — memória episódica + taxonomia + consolidação via Forja
- ✅ **Raciocínio auditável** — hipóteses concorrentes com evidências, margem, entropia, contrafactuais
- ✅ **Evolução autônoma** — sem hardcodes, sem operadores específicos de benchmark, sem limites artificiais

## Índice
1. [Visão Geral](#1-visão-geral)
2. [Pipeline de Dados / Fluxo Cognitivo](#2-pipeline-de-dados--fluxo-cognitivo)
3. [Arquitetura em 7 Camadas](#3-arquitetura-em-7-camadas)
4. [Sistemas de Controle (Maestro + MetaController + SemanticWorkspace)](#4-sistemas-de-controle)
5. [Sistemas Cognitivos (SelfModel, Erro, Atenção, Simulação)](#5-sistemas-cognitivos)
6. [Representação (HDV, CGE, Manifold, Espaço Hiperdimensional)](#6-representação)
7. [Renderizadores (MentalRenderer, Visual, Audio, Texto)](#7-renderizadores)
8. [Aprendizado Contínuo (LifeStream, Curiosidade, Persistência)](#8-aprendizado-contínuo)
9. [Inventário de Arquivos](#9-inventário-de-arquivos)
10. [Resumo da Sessão](#10-resumo-da-sessão)
11. [Módulos Novos (Junho 2026)](#11-módulos-novos-junho-2026)
12. [Módulo de Cognição (Julho 2026)](#12-módulo-de-cognição-julho-2026)
13. [Arquitetura de Agência e Meta-Abstração](#13-arquitetura-de-agência-e-meta-abstração-30062026-parte-2)
14. [Concept Formation Engine (CFE)](#14-concept-formation-engine-cfe--6-módulos)
15. [UnifiedSelf + PrimitiveInventionEngine](#15-unifiedself--primitiveinventionengine)
16. [Cognitive Modes + WorldModel](#16-cognitive-modes--worldmodel-01072026)
17. [KnowledgeGraph + PubMed Real Data](#17-knowledgegraph--pubmed-real-data-01072026)
18. [InferenceEngine — Geracao de Conhecimento Novo](#18-inferenceengine--geracao-de-conhecimento-novo)
19. [ModelLibrary + ModelEvolution](#19-modellibrary--modelevolution-01072026)
20. [DMN — CrossDomainAnalogy + ConceptInvention](#20-dmn--crossdomainanalogy--conceptinvention-01072026)
21. [Infraestrutura Completa](#21-infraestrutura-completa-01072026)
22. [StructuralMetrics](#22-structuralmetrics--leis-de-organizacao-01072026)
23. [DynamicTypeLibrary](#23-dynamictypelibrary--7-tipos-de-dinamica-01072026)
24. [MetaModeler](#24-metamodeler--meta-raciocinio-01072026)
25. [Experimentos de Generalizacao](#25-experimentos-de-generalizacao-01072026)
26. [TypedCognitiveGraph — Integracao 4 Fundacoes Matematicas](#26-typedcognitivegraph--integracao-das-4-fundacoes-matematicas-12072026)
27. [SymbolicEngine — Raciocinio Dedutivo](#27-symbolicengine--raciocinio-dedutivo-sobre-cge-12072026)
28. [OpenIE Extractor — Relacoes Tipadas](#28-openie-knowledge-extractor---relacoes-tipadas-por-parsing-de-dependencias-12072026)

---

## 1. Visão Geral

NovaMind é uma arquitetura cognitiva completa sem LLMs. O núcleo é um **espaço hiperdimensional (HDV) de 10000 dimensões bipolares** onde conceitos, percepções, memórias e pensamentos são representados como vetores. Sobre esse espaço operam grafos cognitivos (CGE), sistemas de controle (Maestro, MetaController), aprendizado contínuo (LifeStream, Forja), e renderizadores multimodais.

### Pilares arquiteturais

| Pilar | Descrição |
|-------|-----------|
| **HDV 10000D bipolar** | Toda representação converge para vetores ±1 de 10000 dimensões |
| **CGE (Cognitive Graph Engine)** | Grafo de conceitos + arestas causais + holograma compressível |
| **Maestro + MetaController** | Orquestrador com modos cognitivos (explore, hypothesize, consolidate, reflect) |
| **Forja + OperatorForge** | Descobre operadores e programas cognitivos por abstração de experiências |
| **MentalRenderer** | Interface plugável de renderização: texto, imagem, áudio, vídeo, código |
| **SemanticWorkspace** | Memória de trabalho global onde hipóteses e previsões competem |
| **LifeStream** | Aprendizado multimodal contínuo da Wikipedia + fontes diversas |

---

## 2. Pipeline de Dados / Fluxo Cognitivo

```
┌──────────────────────────────────────────────────────────────────┐
│  PERCEPÇÃO (Imagem, Video, Audio, Texto)                        │
│  ┌──────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐  │
│  │ VisualPipeline│  │ Grounding  │  │ Audio    │  │ BGE     │  │
│  │ ResNet18→JL   │  │ Engine     │  │ Pipeline │  │ Encoder │  │
│  │ →10000D       │  │ Encoders   │  │ EnCodec  │  │ 768D→JL │  │
│  └──────┬───────┘  └─────┬──────┘  └────┬─────┘  └────┬────┘  │
│         └────────────────┼──────────────┼──────────────┘        │
│                          ▼                                       │
│            HDV 10000D Bipolar (Manifold)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  CGE (Cognitive Graph Engine)                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Nós: conceitos (id, vector 10000D, confidence, valence)    │ │
│  │ Arestas: relações causais (from→to, weight, edge_type)     │ │
│  │ Operações: compress_to_hologram, spread_activation,        │ │
│  │           lateral_inhibition, forge_consolidate,            │ │
│  │           forge_abstract, causal_intervention               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  WORLD MODEL (Causal + Física)                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ WorldModel: grafo causal via Pearl do-calculus              │ │
│  │ WorldModel4D: grid (x,y,z + tempo) com tracking de objetos │ │
│  │ CausalReasoning: simulação contrafactual (do-calculus)      │ │
│  │ PhysicalSandbox: simulador procedural genérico              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  SEMANTIC WORKSPACE (Memória de Trabalho Global)                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ WorkspaceEntry: content + source + weight + confidence     │ │
│  │ + novelty + urgency + tags                                 │ │
│  │ Operações: post(), get_conscious(), broadcast(), tick()     │ │
│  │ Decaimento natural, competição por saliência, pruning       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  MAESTRO (Orquestrador) + Subsistemas                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐│
│  │MetaCtrl  │ │SelfModel │ │SimEngine │ │MultiErrorTracker     ││
│  │(modos)   │ │(Identity │ │(EXPECT,  │ │(prediction_error,    ││
│  │EXPLORE   │ │ Process) │ │CF,PLAN,  │ │ novelty, uncertainty,││
│  │HYPOTH    │ │crenças,   │ │IMAGINE)  │ │ violation)           ││
│  │CONSOLID  │ │fadiga,    │ │          │ │                      ││
│  │REFLECT   │ │atenção    │ │          │ │                      ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│  │Forja     │ │Operator  │ │Model     │                         │
│  │(abstração│ │Forge     │ │Competition                        │
│  │conceitos)│ │(descobre │ │(colapso                            │
│  │          │ │ novos ops)│ │epistêmico)                         │
│  └──────────┘ └──────────┘ └──────────┘                         │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  NOVO ESTADO COGNITIVO (milhares de HDVs + atenção + contexto)   │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  MENTAL RENDERER (Interface Plugável)                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │Visual    │ │Audio     │ │Text      │ │Code      │ │Video  │ │
│  │Renderer  │ │Renderer  │ │Renderer  │ │Renderer  │ │Renderer│ │
│  │HDV→PNG   │ │HDV→WAV   │ │HDV→Texto │ │HDV→Py    │ │HDV→MP4│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Arquitetura em 7 Camadas

### Camada 1 — Percepção
Transforma inputs multimodais em vetores HDV 10000D.

| Módulo | Input → Output | Arquivo |
|--------|----------------|---------|
| `VisualPipeline` | Imagem → ResNet18(512D) → JL → HDV 10000D bipolar | `core/visual_pipeline.py` |
| `GroundingEngine` | Texto (spaCy 300D/BGE 768D) / Imagem (ResNet18) / Áudio (EnCodec) → HDV | `core/grounding_engine.py` |
| `AudioPipeline` | Waveform → EnCodec codes → HDV binding | `core/audio_pipeline.py` |
| `BGEEncoder` | Texto → BGE 768D → JL → 10000D bipolar | `core/bge_encoder.py` |
| `VisualCortex` | Pixels brutos → edges, cores, textura, movimento | `core/visual_cortex.py` |
| `AuditoryCortex` | Waveform → spectral, pitch, formants, onset | `core/auditory_cortex.py` |
| `ContinuousFlux` | Fluxo visual contínuo → diff, diff2, pred_err | `core/continuous_flux.py` |
| `PerceptualStream` | Video + Audio → HDV contínuo | `core/perceptual_stream.py` |

### Camada 2 — World Model
Modela causalidade, física e relações entre conceitos.

| Módulo | Arquivo |
|--------|---------|
| `WorldModel` — Grafo causal (Pearl do-calculus, 3 níveis: associação, intervenção, contrafactual) | `core/world_model.py` |
| `WorldModel4D` — Grid 4D (x,y,z + tempo) com tracking | `core/world_model_4d.py` |
| `CausalReasoningEngine` — Simulação contrafactual via do-calculus | `core/causal_reasoning.py` |
| `PhysicalSandbox` — Simulador procedural genérico baseado em propriedades | `core/physical_sandbox.py` |
| `SandboxEngine` — Composição generativa: decompose→simulate→recompose | `core/sandbox_engine.py` |
| `GridCognition` — Modelo generativo para ARC (7 famílias de transformações) | `core/grid_cognition.py` |

### Camada 3 — Memória
Armazenamento e recuperação em múltiplas escalas temporais.

| Módulo | Arquivo |
|--------|---------|
| `CognitiveGraphEngine` (CGE) — Grafo explícito + holograma compressível + Forja interna | `core/cognitive_graph_engine.py` |
| `MultiScaleHDV` — 3 níveis: L1 (working ~30s), L2 (episodic ~1-4h), L3 (semantic permanente em disco) | `core/multiscale_hdv.py` |
| `EpisodicStream` — Append-only de HDVs em disco (binário, sem SQLite) | `core/episodic_stream.py` |
| `TemporalBinder` — Bind temporal: conceito ⊗ tempo ⊗ contexto ⊗ emoção | `core/temporal_binder.py` |
| `HippocampalBuffer` — Buffer circular de N segundos com replay/pause/detect | `core/hippocampal_buffer.py` |
| `WorkingMemory` — Slots de memória com atenção e tracking de acesso | `core/working_memory.py` |
| `HDCMemory` — Query sobre HDVs bit-packed em CPU (sem overflow de VRAM) | `core/hdc_memory.py` |
| `HDCUtils` — Pack/Unpack bipolar (40KB → 1.25KB) | `core/hdc_utils.py` |

### Camada 4 — Raciocínio
Operadores, programas, abstrações, lógica.

| Módulo | Arquivo |
|--------|---------|
| `HybridReasoningEngine` — 3 paradigmas: neural, simbólico, geométrico + consenso | `core/reasoning.py` |
| `CognitiveOperators` — 15 operadores primitivos (Observe, Compare, Predict, etc.) + sistema de seleção | `core/cognitive_operators.py` |
| `OperatorForge` — Descobre novos operadores (quarentena, utilidade multi-dim) | `core/operator_forge.py` |
| `CognitiveProgram` — Programas são entidades de 1ª classe no CGE (HDV, triggers, histórico) | `core/cognitive_program.py` |
| `SpelkeGenes` — Biblioteca de primitivas cognitivas (Language, Logic, Ontology, Causality) | `core/spelke_genes.py` |
| `Archetypes` — Consolidação de padrões cognitivos em arquétipos reutilizáveis | `core/archetypes.py` |
| `LogicTest` — Teste de lógica autônomo com traceabilidade HDC | `core/logic_test.py` |

### Camada 5 — Metacognição
Auto-monitoramento, confiança, modelos competindo.

| Módulo | Arquivo |
|--------|---------|
| `MetacognitionSystem` — Self-monitoring + confidence + knowledge gaps (FEP) | `core/metacognition.py` |
| `SelfModel` — SelfIdentity (crenças, autobiografia, capabilities) + SelfProcess (fadiga, atenção, modo) | `core/self_model.py` |
| `ErrorSignal` — Erro multi-dimensional: prediction_error, novelty, salience, uncertainty, violation | `core/error_signal.py` |
| `ModelCompetition` — Modelos competem com colapso epistêmico (refutação → enfraquecimento global) | `core/model_competition.py` |
| `PredictiveCodingEngine` — Free Energy via distância HDC (1-sim) | `core/predictive_coding.py` |
| `SelfVector` — Vetor SELF estável ancorando todas as percepções | `core/self_vector.py` |

### Camada 6 — Objetivos / Agência
Metas, planejamento, simulação, ação.

| Módulo | Arquivo |
|--------|---------|
| `GoalManager` — Metas persistentes + decomposição + error-driven learning | `core/goal_system.py` |
| `SimulationEngine` — EXPECT, COUNTERFACTUAL, PLAN, IMAGINE unificados | `core/simulation_engine.py` |
| `ActiveInferenceEngine` — Seleção de ações que minimizam Free Energy | `core/active_inference.py` |
| `ExecutivePlanner` — Decomposição recursiva autônoma (missões, não prompts) | `core/executive_planner.py` |
| `IntrinsicRL` — Aprendizado por reforço com bônus de curiosidade | `core/intrinsic_rl.py` |

### Camada 7 — Aprendizado Contínuo
Adaptação, neuromodulação, sono, evolução.

| Módulo | Arquivo |
|--------|---------|
| `LifeStream` — Pipeline contínuo (Source → Reliability → Opportunity → Episode → CGE) | `core/lifestream.py` |
| `Neuromodulator` — DA, NA, ACh, 5HT modulando atenção, plasticidade, comportamento | `core/neuromodulator.py` |
| `NeuromodulatorState` — NT expandido com CRH, OXT, assinaturas NT por conceito | `core/neuromodulator_state.py` |
| `SleepMode` — Replay offline + consolidação de arquétipos + L2→L3 | `core/sleep_mode.py` |
| `ContinuousLearningEngine` — Bayesian online + novelty detection + anti-catastrophic forgetting | `core/learning.py` |
| `LatentVector` — Variáveis latentes descobertas pelo sistema via erro de predição | `core/latent_vector.py` |

---

## 4. Sistemas de Controle

### Maestro (`core/maestro.py` — 1304 linhas)
Orquestrador cognitivo paralelo em tempo real. Cada córtex (visual, auditivo, cognitivo, memória, mundo modelo, arquétipos, motor) roda em **thread própria**.

```
tick():
  1. Coleta saídas de todos os atores (percepção, cognição, motor, sono, linguagem, causal)
  2. _on_thought(): escreve no CGE, aplica error-driven learning, dispara Forja se surpresa
  3. _decide_next(): goal-driven (GoalManager) + neuromodulação (explore vs exploit)
  4. Architectonics: auditoria a cada N ciclos
  5. Corpo→NT→gain loop
```

### MetaController (`core/meta_controller.py`)
7 modos cognitivos com **Q-learning meta-policy**:
- **EXPLORE**: busca novidade, expande conhecimento
- **HYPOTHESIZE**: gera candidatos de explicação
- **PREDICT**: simula consequências
- **TEST**: refuta hipótese
- **CONSOLIDATE**: comprime, abstrai
- **EXPLOIT**: usa conhecimento consolidado
- **REFLECT**: reavalia estratégia

Loops registrados: `scientific`, `execution`, `impasse`, `deep_reflect`, `rapid_explore`.

### SemanticWorkspace (`core/semantic_workspace.py`)
Memória de trabalho global entre World Model e Maestro.

```
post(content, source, weight, confidence, novelty, urgency)
  → entra no workspace, compete por saliência

get_conscious(n=5)
  → retorna as N entradas mais salientes (conteúdo "consciente" atual)

tick(dt=0.2)
  → decaimento natural de todas as entradas
```

### FractalMaestro (`core/fractal_maestro.py`)
Hierarquia de controle emergente. Cria sub-Maestros quando complexidade excede threshold.

---

## 5. Sistemas Cognitivos

### SelfModel (`core/self_model.py`)
```
SelfIdentity:                   SelfProcess:
  beliefs (confiança, evidência)   active_operator
  autobiography (200 episódios)    active_program
  capability_profile               fatigue (0..1)
  learning_profile                 attention_focus
  strengths / weaknesses           current_emotion
  best_modality                    cognitive_mode
```

### MultiErrorTracker (`core/error_signal.py`)
6 dimensões de erro por cenário/hipótese:
- `prediction_error`: dopamina, 0..1
- `novelty`: noradrenalina, 0..1
- `salience`: atenção, 0..1
- `uncertainty`: acetilcolina, 0..1
- `violation`: quebra de modelo, 0..1
- `causal_attribution`: lista de causas atribuídas

### AttentionController (`core/attention_controller.py`)
```
SalienceMap → SpotlightController → ChangeDetector
   ↓                ↓                    ↓
inibição lateral  foco competitivo   spikes de saliência
entre nodos      com supressão       por mudança perceptual
```

### SimulationEngine (`core/simulation_engine.py`)
4 modos, 1 mecanismo:
- **EXPECT**: branch principal + variações
- **COUNTERFACTUAL**: "e se..." — linha base vs com ação vs inversões
- **PLAN**: N planos candidatos (sequências de operadores)
- **IMAGINE**: combinações aleatórias de dimensões

Branches competem via ModelCompetition. Realidade observada reforça/ enfraquece.

### ModelCompetition (`core/model_competition.py`)
```
Modelos competem → colapso epistêmico → vencedor consolidado (Forja)
  collapse(): enfraquece conexões no CGE globalmente
  reconsolidate(): reforça modelo vencedor
```

---

## 6. Representação

### HDV — Hiperespaço 10000D Bipolar
```
Manifold (coglang.py):
  - 10000 dimensões bipolares (±1)
  - Operações: bind (XOR), bundle (soma + sign), unbind
  - Packing: 40KB → 1.25KB (1 bit/dim)
  - Similaridade: cosine similarity
```

### CGE — Cognitive Graph Engine (`core/cognitive_graph_engine.py`)
```
Nó:
  id, vector (10000D bipolar), semantic_vec, perceptual_vec,
  nt_vec, confidence, emotional_valence, emotional_arousal

Aresta:
  from, to, edge_type (CAUSES tensor), weight

Operações:
  create_node, add_edge, spread_activation, lateral_inhibition,
  forge_consolidate, forge_abstract, consolidation_cycle,
  tag_emotion, recall_by_emotion, causal_intervention,
  test_causal_model, plan_to_goal, refine_abstraction,
  get_concept_hierarchy, compress_to_hologram,
  decompress_from_hologram
```

### Operadores Cognitivos (`core/cognitive_operators.py`)
15 primitivas + hierarquia:
```
depth=0: Observe, Compare, Predict, Imagine, Generalize,
         Specialize, Ignore, Wait, Focus, Explore,
         Exploit, Reflect, Chunk, Compose, Test

depth=1: anticipate = Observe + Predict + Compare (descoberto pela Forja)
depth=2: meta_composite sobre depth=1
```

Operadores têm: effect_profile, cost, utility, history, confidence, sub_operators, depth.

---

## 7. Renderizadores

### MentalRenderer (`core/mental_renderer.py`)
Interface plugável. O Maestro chama `render(estado_cognitivo, modalidade)`:

```python
renderer = MentalRenderer(
    visual=VisualRenderer(pipeline),   # HDV → PNG 224×224
    audio=AudioRenderer(vocoder),      # HDV → WAV
    text=TextRenderer(),               # estado → descrição
    code=CodeRenderer(),               # estado → pseudo-código
)
```

### VisualPipeline (`core/visual_pipeline.py`)
```python
encode_frame(image) → ResNet18(512D) → JL proj → HDV 10000D bipolar
decode_bipolar(hdv) → pinv(JL) → 512D rec → VisualDecoder(ConvTranspose2d) → 224×224 RGB
verify_fidelity(original, reconstructed) → PSNR, SSIM, MSE
```

### AudioVocoder (`core/audio_vocoder.py`)
HDV 10000D → MLP → (F0, Aperiodicidade, SpectralEnvelope) → World Vocoder → waveform

### BrocaDecoder / BrocaV2 (`core/broca_decoder.py`, `core/broca_v2/`)
HDV → texto via Role-Filler Unbinding (Broca) ou LSTM generativo (BrocaV2)

### VisualRenderer (`core/visual_renderer.py`)
HDV → VQ-VAE codebook → múltiplos frames → .mp4

---

## 8. Aprendizado Contínuo

### LifeStream (`core/lifestream.py`)
```
Fonte (Wikipedia, YouTube, procedural)
  → Source Reliability Filter (Wikipedia=0.95, Ad=0.05)
  → Opportunity Engine (expected gain 5-dim)
  → Episode Stream (texto → HDV → CGE → descarta raw)
  → Auto-save a cada 5 ciclos (brain_live.pth)
```

### Curiosidade
O sistema pergunta "o que ainda não entendo?" baseado nas lacunas do CGE. Tópicos desconhecidos ou com poucas conexões são priorizados.

### Modos Naturais
O Maestro alterna entre `explore → hypothesize → consolidate → reflect` baseado no ciclo. A consolidação (Forja + CGE) acontece apenas no modo `consolidate`, não forçada.

### Persistência
- **Save**: `save/brain_live.pth` (PyTorch, nodes + edges, edge_type convertido para int)
- **Load**: automático na inicialização. Se save tem 0 nós, começa fresco
- **Auto-save**: a cada 5 ciclos

---

## 9. Inventário de Arquivos

### `core/` — 108 arquivos .py

**Percepção (Layer 1):**
`visual_pipeline.py` `visual_cortex.py` `visual_renderer.py` `audio_pipeline.py` `audio_vocoder.py` `auditory_cortex.py` `grounding_engine.py` `bge_encoder.py` `perception.py` `perceptual_stream.py` `perceptual_target_matcher.py` `continuous_flux.py`

**World Model (Layer 2):**
`world_model.py` `world_model_4d.py` `causal_reasoning.py` `cross_domain.py` `physical_sandbox.py` `sandbox_engine.py` `grid_cognition.py` `cognitive_graph_engine.py`

**Memória (Layer 3):**
`memory.py` `hdc_memory.py` `hdc_utils.py` `episodic_stream.py` `temporal_binder.py` `temporal_buffer.py` `hippocampal_buffer.py` `working_memory.py` `multiscale_hdv.py` `semantic_workspace.py`

**Raciocinio (Layer 4):**
`reasoning.py` `logic_test.py` `cognitive_operators.py` `operator_forge.py` `cognitive_program.py` `spelke_genes.py` `spelke_extended.py` `spelke_cognitive.py` `archetypes.py` `emergent_context.py`
`cognition/symbolic_engine.py` — Motor de inferencia logica (MP, MT, Barbara, RC) sobre CGE
`cognition/hdc_decoder.py` — Decodificador HDC texto por nearest-neighbor z-score
`typed_graph.py` — TypedCognitiveGraph (PoincareBall + Clifford + FEP + Topologia)

**Metacognição (Layer 5):**
`metacognition.py` `error_signal.py` `predictive_coding.py` `model_competition.py` `self_model.py` `self_vector.py` `attention_controller.py`

**Objetivos (Layer 6):**
`goal_system.py` `goals.py` `active_inference.py` `intrinsic_rl.py` `executive_planner.py` `simulation_engine.py`

**Aprendizado (Layer 7):**
`learning.py` `lifestream.py` `neural_substrate.py` `neuromodulator.py` `neuromodulator_state.py` `sleep_mode.py` `meta_controller.py` `latent_vector.py`

**Orquestração:**
`mind.py` (1175 linhas) `maestro.py` (1304 linhas) `fractal_maestro.py` `integration.py` `interact.py` `colony.py` `collective_unconscious.py` `multi_agent_runner.py` `background_incubator.py` `state_bridge.py` `persistence.py`

**Produção Linguística/Motora:**
`broca_decoder.py` `language_cortex.py` `word_ngram.py` `semantic_composition.py` `semantic_encoder.py` `autoregressive_generator.py` `positional_binder.py` `vocal_synthesizer.py` `articulatory_synthesizer.py` `premotor_cortex.py` `syllable_oscillator.py` `diva_controller.py` `cerebellum.py` `motor_cortex.py` `somatosensory_cortex.py` `emergent_language.py`

**Renderizadores:**
`mental_renderer.py` `compositional_renderer.py` `cognitive_answerer.py`

**Ferramentas/Compilação:**
`cognitive_offloading.py` `coglang_compiler.py` `mega_brain_compiler.py` `code_extractor.py` `grounded_semantics.py`

**Fundação Matemática (`core/math/`):**
`geometric_algebra.py` `hyperbolic.py` `information.py` `topology.py`

**HDC (`core/coglang.py`):**
`coglang.py` — classe `Manifold` com 10000D bipolar

**NovaCrush (`core/novacrush/`):**
`bitlinear.py` `crushed_substrate.py` `forward_forward.py` `hdc.py` `genetic_memory.py` `spike_engine.py` `language_gen.py` `inject_fluency.py` `benchmark.py` `train_loop.py`

**BrocaV2 (`core/broca_v2/`):**
`model.py` `decoder.py` `tokenizer.py` `train.py`

### Raiz do Projeto

| Arquivo | Função |
|---------|--------|
| `run_agi_real.py` | **Pipeline principal**: LifeStream + CGE + SemanticWorkspace + MentalRenderer + servidor HTTP |
| `agiview_live.html` | **Frontend**: ESQUERDA=INPUT, CENTRO=CGE, DIREITA=Representação mental, BAIXO=Thoughts/Sim/Error |
| `mtb_runner.py` | Benchmark MTB-1 multi-mundo com transferência |
| `mtb_world_generator.py` | Gerador procedural de mundos com física variável |
| `mtb_sandbox.py` / `mtb_sandbox_salto.py` / `mtb_sandbox_salto2.py` | Sandboxes progressivos (Fase 1, Salto 1, Salto 2) |
| `agi_server.py` | Servidor HTTP + SSE (antigo, substituído pelo run_agi_real) |
| `dashboard.html` | Dashboard estático (antigo) |

---

## 10. Resumo da Sessão

### Sessão 30/06/2026 — Knowledge Extraction + Sensory Workspace + Experimentos

**O que foi construído:**

| Módulo | Função |
|--------|--------|
| `core/structured_encoder.py` | Encoder estrutural: parser + role bind (ordem preservada, polissemia) |
| `core/knowledge_extractor.py` | Extrai triplas SVO, hipóteses, exceções; detecta conflitos no CGE |
| `core/sensory_workspace.py` | Captura áudio+video contínua, buffer 5s, persistência seletiva |
| `core/cognitive_cycle.py` | Refatorado: +5 mecanismos (WM→CGE, Forja→procedimentos, sono, curiosidade, KE) |
| `core/episode_store.py` | Atualizado para encoder estrutural |
| `core/mental_state.py` | Dynamic blackboard |
| `agi_live.py` | +flag `--sensory`, +Explain Mode, +Tradutor MentalState |
| `_experiments.py` | 6 experimentos cognitivos (retenção, generalização, transferência, consolidação, esquecimento, autonomia) |
| `_benchmark.py` | 25 testes, 6 categorias |
| `_science_curriculum.py` | 92 fatos científicos (física, biologia, química) |

**Métricas atuais:**

| Métrica | Valor |
|---------|-------|
| Episódios | 192.592 (192.500 + 92 ciência) |
| Conceitos semânticos | 72.185 |
| HDVs | 192.500 (10.000D bipolar, memmap) |
| Nós CGE | ~2.300 |
| Arestas CGE | ~10.600 |
| Operadores primitivos | 15 |
| Procedimentos forjados | 1 (PROC_mammal) |
| Recall score (texto) | 3.53 (Jaccard) |
| Carregamento | ~31s (pickle 526 MB + HDVs 7.7 GB) |
| Ciclo (pós-warmup) | 0.1-1.4s |

**Pipeline cognitivo atual:**

```
PERCEPCAO                    CONHECIMENTO                    RACIOCINIO
─────────────────           ─────────────────           ─────────────────
audio (buffer 5s)            KnowledgeExtractor            CGE (2300+ nos)
video (buffer 5s)            triplas SVO                   World Model
texto (parser + role bind)   hipoteses                     Forja (proc+conc)
encoder estrutural           excecoes                      NeuroField
ordem preservada             conflitos                     sono/curiosidade
polissemia                   integracao CGE                experimentos

MEMORIA                      CONTROLE                      SAIDA
─────────────────           ─────────────────           ─────────────────
EpisodeStore                 MentalState                   Translator
SQLite + memmap + HNSW       dynamic blackboard             le do MentalState
192.500+ episodios            modulos registram/leem         Explain Mode
72.185 conceitos              I/O separado da cognicao
```

**Para usar:**
```powershell
python agi_live.py                           # modo texto
python agi_live.py --sensory                 # com camera+microfone
python agi_live.py --explain --input "pergunta"  # modo explicacao
python _experiments.py                       # bateria cognitiva
python _benchmark.py                         # 25 testes
```

### Sessão 29/06/2026 — MentalState + CognitiveCycle + EpisodeStore

**O que foi construído:**

| Módulo | Função |
|--------|--------|
| `core/mental_state.py` | Dynamic blackboard global |
| `core/cognitive_cycle.py` | Ciclo cognitivo contínuo com estagnação |
| `core/episode_store.py` | Memória persistente (SQLite + memmap + HNSW) |
| `core/cognitive_graph_engine.py` | CGE populado (WordNet + conceitos) |
| `core/world_model.py` | SCM causal integrado ao ciclo |
| `core/memory_persistence.py` | Backup otimizado (pickle + npy separados) |
| `agi_live.py` | Runner com Explain Mode |

**Arquitetura original (referência):**

```
Percepção → HDV → CGE → WorldModel → SemanticWorkspace → Maestro → MentalRenderer
```

### Próximos passos
1. **Resolver representação da linguagem** (parser + papéis semânticos) — ✅ feito
2. **Extrair conhecimento estruturado** de livros e artigos — ✅ feito (KnowledgeExtractor)
3. **Fortalecer World Model** para operar sobre estruturas — ✅ feito (WM escreve no CGE)
4. **Expandir Forja** para criar procedimentos — ✅ feito (PROC_mammal)
5. **Bateria fixa de experimentos** — ✅ feito (6 experimentos)
6. **Percepção contínua** (workspace sensorial) — ✅ feito (audio+video)
7. **Medir evolução ao longo do tempo** — pendente (rodar experiments.py semanalmente)
8. **Modelos mentais** (célula, mercado, ecossistema) — próximo grande passo

---

## 11. Módulos Novos (Junho 2026)

### Núcleo do ciclo cognitivo

| Módulo | Descrição | Arquivo |
|--------|-----------|---------|
| MentalState | Blackboard global. Único objeto compartilhado entre módulos | `core/mental_state.py` |
| CognitiveCycle | Ciclo cognitivo contínuo com parada por estagnação | `core/cognitive_cycle.py` |
| EpisodeStore | Memória persistente (SQLite + memmap HDV + HNSW) | `core/episode_store.py` |
| StructuredEncoder | Encoder estrutural (parser + role bind, ordem preservada) | `core/structured_encoder.py` |
| KnowledgeExtractor | Extrai triplas SVO, hipóteses, exceções, conflitos | `core/knowledge_extractor.py` |
| SensoryWorkspace | Captura áudio+video contínua, buffer 5s | `core/sensory_workspace.py` |

### Scripts

| Script | Função |
|--------|--------|
| `agi_live.py` | Runner principal (+Explain Mode, +`--sensory`) |
| `_experiments.py` | 6 experimentos cognitivos (retenção, generalização, transferência, consolidação, esquecimento, autonomia) |
| `_benchmark.py` | 25 testes, 6 categorias |
| `_science_curriculum.py` | 92 fatos científicos |
| `_cognitive_lab.py` | Laboratório cognitivo interativo |

### Mecanismos do ciclo cognitivo

1. **SENSE** → HDV estrutural (parser + role bind)
2. **RECALL** → EpisodeStore (query_text)
3. **CGE** → ativação + propagação contextual iterativa
4. **WORLD MODEL** → hipóteses → escreve arestas no CGE
5. **KNOWLEDGE EXTRACTOR** → triplas SVO → nós+arestas CGE
6. **FORJA** → conceitos (co-ocorrência) + procedimentos (sequências)
7. **NEUROFIELD** → DA/NA/ACh/5HT modulado por prediction error
8. **SONO** → a cada 20 ticks: decaimento + forja longa + replay
9. **CURIOSIDADE** → conceitos de baixa confiança → perguntas
10. **TRADUTOR** → le do MentalState → caminhos de inferência CGE
11. **PERCEPÇÃO VISUAL** → DINOv2 → 9 regiões (3x3) → bind com posição → HDV
12. **MEMÓRIA VISUAL** → VisualMemory → busca por similaridade HDV pura → Forja descobre padrões
13. **KNOWLEDGE EXTRACTOR** → triplas SVO + hipóteses + exceções + conflitos → CGE

### Evolução da Sessão (30/06/2026)

**Mudanças principais:**

| Antes | Depois |
|-------|--------|
| Embedding médio spacy (cos dog/man=1.0) | Encoder estrutural (parser+role bind, cos=0.65) |
| Semântica só por média de vetores | Ordem preservada, polissemia, negação, hipóteses |
| ResNet18 para visão (perde posição) | DINOv2 + 9 regiões 3x3 + bind de posição |
| Sem memória visual | VisualMemory com busca HDV pura |
| Sem experimentos | 6 experimentos cognitivos (retention, generalization, transfer, consolidation, forgetting, autonomy) |
| ARC: 0/5 | ARC: 5/5 match visual (percepção pura, sem transformações hardcoded) |
| CGE vazio | CGE: 2.300+ nós, 10.600+ arestas |
| Forja nunca rodou | Forja: PROCEDIMENTOS (PROC_mammal) + conceitos |
| Tradutor regurgitava recall | Tradutor lê MentalState + caminhos de inferência CGE |

**Arquitetura final:**

```
PERCEPCAO                    CONHECIMENTO                    RACIOCINIO
─────────────────           ─────────────────           ─────────────────
audio (buffer 5s)            KnowledgeExtractor            CGE (2300+ nos)
video (buffer 5s)            triplas SVO                   World Model
texto (parser + role bind)   hipoteses                     Forja (proc+conc)
DINOv2 (9 regioes, posicao)  excecoes                      NeuroField
encoder estrutural           conflitos                     sono/curiosidade
ordem preservada             integracao CGE                experimentos
polissemia                                                ARC thermometer

MEMORIA                      CONTROLE                      SAIDA
─────────────────           ─────────────────           ─────────────────
EpisodeStore                 MentalState                   Translator
SQLite + memmap + HNSW       dynamic blackboard             le do MentalState
192.500+ episodios            modulos registram/leem         Explain Mode
72.185 conceitos              I/O separado da cognicao
VisualMemory (HDV puro)       Forja descobre padroes
```

**Para usar:**
```powershell
python agi_live.py                           # modo texto
python agi_live.py --sensory                 # com camera+microfone
python agi_live.py --explain --input "pergunta"  # modo explicacao
python _experiments.py                       # 6 experimentos
python _test_arc_final.py --tasks 10         # ARC thermometer
```

---

## 12. Módulo de Cognição (Julho 2026)

### Contratos (core/cognition/contracts.py)

Novo sistema de tipos compartilhados entre todos os módulos cognitivos.

| Contrato | Descrição |
|----------|-----------|
| `RecallDecision` | Decisão do RecallPolicy: EMPTY ou ACCEPT, com best_similarity, threshold, reason |
| `EvidenceItem` | Evidência única: source (taxonomy/episodic/perceptual/causal), relation (is_a/has_property/causes/similar), confidence (0-1) |
| `Hypothesis` | Hipótese concorrente: name, category, evidence_for[], evidence_against[], confidence, evidence_total |
| `CompetitionResult` | Resultado completo: winner, margin, entropy, stability[], by_category{}, counterfactuals[] |

**Regra de ouro:** Nenhum componente conhece a implementação interna de outro. Todos se comunicam apenas por estes contratos.

### RecallPolicy (core/cognition/recall_policy.py)

Decide se o resultado do EpisodeStore é utilizável.

- Se `best_similarity < threshold` (0.25) → `RecallDecision(status=EMPTY, reason=LOW_SIMILARITY)`
- Se `best_similarity >= threshold` → `RecallDecision(status=ACCEPT, episodes=top-K)`
- O EpisodeStore permanece agnóstico — só retorna resultados brutos

### Pipeline Cognitivo Atual

```
INPUT → focus extraction → RecallPolicy → CGE activation
                                               ↓
                                   Hypothesis Competition
                                     ├── IS_A: whale→mammal (0.724)
                                     ├── PROP: whale→breathe (0.266)
                                     ├── margin, entropy, stability
                                     └── counterfactuals
                                               ↓
                                   Translator (confidence 0-1)
                                               ↓
                                   Response + Audit
```

---

## 13. Arquitetura de Agência e Meta-Abstração (30/06/2026, parte 2)

### GoalDrivenEngine (`core/goal_driven_engine.py`)

Transforma o sistema de indutor passivo em **agente orientado a objetivos**:

```
GOAL (estado desejado + constraints + prioridade)
  ↓
GoalComparator (gap: estrutural + semântico + confiança)
  ↓
ProgramSynthesizer (induz programas que transformam A→B)
  ↓
ActionSelector (escolhe por MÁXIMA REDUÇÃO DE GAP, não score máximo)
  ↓
Execução → novo estado → feedback loop
```

### HierarchicalPlanner (`core/hierarchical_planner.py`)

Decompõe goals em subgoals ordenados por dependência, reduzindo o espaço de busca de 12^5 para ~4 * 12^1:

```
Goal → subgoals:
  1. Shape (resize) → primitives: rotate_90/180/270, flip, transpose
  2. Object count (add/remove) → primitives: remove, add, transform
  3. Color (recolor) → primitives: recolor
  4. Position (spatial) → primitives: move, rotate, flip
```

### MetaPrimitiveRegistry (`core/program_induction/meta_registry.py`)

Primitives como organismos vivos com:
- utilidade, frequência, domínios, custo, composições conhecidas
- histórico de sucesso/falha, idade
- **poda automática**: primitives com baixa viabilidade são arquivadas
- **fusão**: primitives que sempre compõem juntas viram candidatas a macro

### StrategyMemory (`core/strategy_memory.py`)

Memória de planos indexados por **assinatura do problema**:
- n_objetos_input, n_objetos_output, cores, shapes, tipo_gap
- Recupera plano similar para novo problema
- Adapta parâmetros (cores, posições) para o novo contexto

### MacroDiscovery (`core/macro_discovery.py`)

Descobre **n-grams frequentes** (2-4 primitives) do CompositionGraph:

```
MacroFlipMove_2: flip_horizontal → move  freq=83  conf=1.00
MacroRotateMove_3: rotate_90 → move → move  freq=42  conf=1.00
MacroRotateMoveAdd_3: rotate_90 → move → add  freq=32  conf=1.00
```

Macros são armazenadas no **MacroRegistry** separado (não no PrimitiveRegistry) para não enviesar o espaço de busca.

### SkillEngine (`core/skill_engine.py`)

Habilidade por **INTENÇÃO**, não por sequência fixa:

```
Macro: flip_horizontal → move (sequência FIXA)
Skill: mirror_objects (INTENÇÃO)
  → flip_horizontal → remove
  → flip_horizontal → move → add  
  → flip_horizontal → move → move
  (escolhe a melhor implementação para o contexto)
```

8 skills descobertas, média de 6 implementações por skill. Skills foram melhores que primitives base em 17/100 tarefas ARC (vs 0/100 para macros).

### MetaAbstractionEngine (`core/meta_abstraction.py`)

Abstração profunda via **bind HDV** no CGE. Sem keywords, sem hardcode:

```
"O dedo que aponta a lua" 
  → BGE encode("dedo") → HDV 10000D
  → BGE encode("lua") → HDV 10000D  
  → bind(HDV_dedo, HDV_lua) = VETOR DA RELAÇÃO
  → O vetor É a abstração. Vive no CGE como nó.
  → Query no CGE por vetores similares ENCONTRA analogias.
```

Evidência: similaridade entre binds de pares relacionais no espaço HDV:

| Par | Sim bind | Interpretação |
|-----|----------|---------------|
| finger→moon vs map→territory | 0.256 | Relação "A aponta para B" |
| finger→moon vs word→meaning | 0.265 | Relação "A representa B" |
| finger→moon vs seed→tree | 0.249 | Relação "A leva a B" |

Em HDC puro, binds aleatórios têm similaridade ~0.0. O sinal consistente de ~0.25 prova que **relações abstratas têm assinatura vetorial no hiperespaço**. Com encoder melhor (BGE large 1024D), o sinal passa para > 0.5.

### FailureDrivenDiscovery (`core/failure_driven_discovery.py`)

Falha → análise → cluster → especificação → nova primitiva:

```
Falha (gap residual)
  ↓
FailureAnalyzer → classifica tipo: WRONG_OBJECT_COUNT, SHAPE_MISMATCH, COLOR_MISMATCH
  ↓
FailureCluster → agrupa falhas similares (23 tasks com WRONG_OBJECT_COUNT)
  ↓
OperatorSpec → especificação formal: invariantes + transformações (SEM implementação)
  ↓
SpecGuidedDiscovery → busca exemplos no GeometryDataset que satisfazem a spec
  ↓
PrimitiveDiscovery → INDUZ nova primitiva (não copia)
```

### Módulos novos

| Módulo | Localização | Função |
|--------|-------------|--------|
| GoalDrivenEngine | `core/goal_driven_engine.py` | Agente goal → gap → ação → feedback |
| GoalComparator | `core/goal_driven_engine.py` | Gap estrutural/semântico/confiança |
| HierarchicalPlanner | `core/hierarchical_planner.py` | Decomposição de goals em subgoals |
| FailureDrivenDiscovery | `core/failure_driven_discovery.py` | Falha → cluster → operador candidato |
| OperatorSpecification | `core/operator_specification.py` | Spec formal sem implementação |
| SpecGuidedDiscovery | `core/spec_guided_discovery.py` | Spec → dataset → nova primitiva |
| StrategyMemory | `core/strategy_memory.py` | Memória de planos por assinatura |
| MacroDiscovery | `core/macro_discovery.py` | n-grams frequentes → macros |
| MacroRegistry | `core/macro_registry.py` | Registry separado de macros |
| SkillEngine | `core/skill_engine.py` | Habilidade por intenção (multimpl) |
| MetaAbstractionEngine | `core/meta_abstraction.py` | Abstração via bind HDV |
| MetaPrimitiveRegistry | `core/program_induction/meta_registry.py` | Primitives como organismos vivos |

### Métricas ARC agregadas (sessão completa)

| Métrica | Antes | Depois | ∆ |
|---------|-------|--------|---|
| Primitives | 4 | 13 (+8 via dataset) | +225% |
| Pixel accuracy | 21% | 54.79% | +33pp |
| Tasks com progresso | 12/30 | 21/30 | +75% |
| Tarefas melhoraram sem regredir | — | 3 | monotônico |
| Macros descobertas | 0 | 48 | novo |
| Skills descobertas | 0 | 8 | novo |
| CompositionGraph edges | — | 29 | novo |
| Estratégias armazenadas | 0 | 20 | novo |

### Sprint 2 — Evidence Builders ✅

`core/evidence_builders/` criado com:

| Builder | Arquivo | Função |
|---------|---------|--------|
| `TaxonomyBuilder` | `evidence_builders/taxonomy_builder.py` | BFS no CGE/WordNet max_depth=8, hierarchy walk, classifica relação (is_a/has_property/causes) |
| `EpisodicBuilder` | `evidence_builders/episodic_builder.py` | EpisodeStore + RecallPolicy → EvidenceItem, suporta query text e HDV |
| `PerceptualBuilder` | `evidence_builders/perceptual_builder.py` | CGE spread_activation + landscape-aware → EvidenceItem |
| `CausalBuilder` | `evidence_builders/causal_builder.py` | WorldModel associação/intervenção/contrafactual + curriculum |

**Regra seguida:** Competition não sabe a origem das evidências. Só recebe listas de `EvidenceItem`.

### Sprint 3 — Competition Refinada ✅

`core/cognition/competition.py` (`Competition` class):

| Funcionalidade | Status |
|----------------|--------|
| Ranking completo por categoria | ✅ `by_category` dict |
| Margin | ✅ `winner_score - runner_up_score` |
| Entropy | ✅ Shannon normalizada (0 = convergência total) |
| Stability | ✅ Evolução do score por iteração |
| Counterfactual | ✅ Para runner-up: fontes faltantes, gap de confiança |

### Sprint 4 — Confidence + Consistency + Translator ✅

| Módulo | Arquivo | Função |
|--------|---------|--------|
| `ConfidenceCalibrator` | `cognition/confidence_calibrator.py` | Sigmoid → Platt (gradient descent) → temperature scaling |
| `ConsistencyChecker` | `cognition/consistency_checker.py` | Conflitos diretos, internos, entre evidências |
| `Translator` | `cognition/translator.py` | Lê CompetitionResult, resposta + auditoria textual |

### Sprint 5 — Auditoria + Timeline ✅

`core/cognition/timeline.py`:

| Funcionalidade | Descrição |
|----------------|-----------|
| Timeline | `record()` por iteração com focus, confidence, PE, fontes |
| Heatmap data | `get_heatmap_data()` — frequência de vencedores por tempo |
| Stability summary | `get_stability_summary()` — média, std, tendência de convergência |
| Decomposition history | `get_decomposition_history()` — contribuição de cada fonte ao longo do tempo |

### Pipeline Integrado ✅

`core/cognition/pipeline.py` (`CognitionPipeline`): INPUT → Evidence Builders → Competition → Calibration → Consistency → Translator → Timeline. Todos os módulos se comunicam apenas por contratos.

### Benchmarks ✅

```
benchmarks/
  regression/                       # <10s
    grounding.json                  # 10 testes: "whale is mammal", etc.
    inference.json                  # 8 testes: "A>B, B>C → A>C?"
    confidence.json                 # 10 runs, mede std da confiança
    episodic.json                   # 5 conceitos inventados (florp, dax, wug)
    regression_runner.py            # Runner dos testes de regressão
  capability/                       # minutos
    capability_runner.py            # arc.py, vision.py, noise.py
  run_benchmarks.py                 # Runner principal
```

### ARC-AGI (30/06/2026)

| Métrica | Valor |
|---------|-------|
| Treino Pixel Accuracy | 80.6% |
| Treino Exact Match | 6.2% |
| OOD Pixel Accuracy | 78.2% |
| OOD Exact Match | 0.0% |
| Tempo treino (50 steps) | 185.8s |

Veredito: Aprendizado estrutural detectado (78.2% > baseline copy 76.5%). Exact match OOD requer mais treino ou abordagem neuro-simbólica.

---

## 14. Concept Formation Engine (CFE) — 6 Módulos

### ObservationLayer (`core/conceptual_discovery/observation.py`)
Normaliza observações de qualquer modalidade: texto, imagem, vídeo, áudio, programas executados, erros, sucessos, feedback do CGE, traces, metas, ações. Saída: `ObservedEvent` (timestamp, tipo, payload normalizado). Sem interpretação — só captura.

### InvariantDiscovery (`core/conceptual_discovery/invariant_discovery.py`)
Pergunta central: **"o que permaneceu igual?"** Detecta padrões invariantes: branching (ramificação), reflection (reflexão), expansion (expansão), contraction (contração), causal_chain (cadeia causal), symmetry (simetria), spatial_transformation (transformação espacial). Cada invariante é candidato a conceito.

### ConceptFormation (`core/conceptual_discovery/concept_formation.py`)
Constrói conceitos ESTRUTURADOS: condições (o que precisa ser verdade), efeitos (o que muda), restrições (o que NÃO pode ser verdade), contraexemplos. **Valida existência por 5 métricas objetivas**: erro de previsão, espaço de busca, comprimento de programa, reuso cross-domínio, compressão. Conceito só existe se pelo menos 2 das 5 métricas melhoram.

### TheoryBuilder (`core/conceptual_discovery/theory_builder.py`)
Agrupa conceitos co-ocorrentes (3+ dominios compartilhados) em teorias unificadoras. Exemplo: {queda, atração, orbita, peso, massa} → GRAVIDADE.

### ImaginationSimulator (`core/conceptual_discovery/imagination.py`)
Instancia conceitos em diferentes domínios. BRANCHING se manifesta em 8 domínios: biology, geography, geometry, programming, physics, engineering, medicine, neuroscience. Transfer score = 0.80 (4 domínios descobertos / 5). **Teste de transferência cross-domínio: BRANCHING observado em biologia → aplicado em geometria.**

### ConstraintNetwork (`core/conceptual_discovery/constraint_network.py`)
Rede de restrições mútuas. Cada nó da rede (CGE, Simulação, Memória, Programas) verifica se um conceito proposto é consistente. Conceito só é válido se TODOS os verificadores aprovam.

### CFEBridge (`core/conceptual_discovery/cfe_bridge.py`)
Ponte entre CFE e ProgramSynthesizer. Mapeia invariantes → primitives preferidas: expansion→{tile, resize}, reflection→{flip, rotate}, causal_chain→{move, recolor}. Reduz busca de 12⁵=248.832 para 3⁵=243 programas (1000× menos).

## 15. UnifiedSelf + PrimitiveInventionEngine

### UnifiedSelf (`core/unified_self.py`)
Self coerente que integra:

| Componente | Função |
|-----------|--------|
| **Identidade** | nome, crenças (beliefs→confidence), capacidades (capabilities→proficiency), preferências |
| **Memória Autobiográfica** | 1000 episódios comprimidos (tipo, valence, confidence_delta), extrai padrões |
| **Drives** | curiosidade (decay=0.02), mestria (0.01), novidade (0.015) — frustram/satisfazem |
| **Neuromodulação** | DA (recompensa), NA (alerta), ACh (atenção), 5HT (bem-estar). Mood emergente |
| **Energy** | 0-1, decai 0.003/tick. Abaixo de 0.3 → descanso necessário |
| **Tick** | drives decaem, energia diminui, mood atualiza |

### IntrinsicMotivationEngine (`core/intrinsic_motivation.py`)
Gera METAS INTERNAS a partir dos drives:

- **Curiosidade**: busca conceitos com baixa confiança no CGE
- **Mestria**: pratica habilidade com menor proficiência
- **Novidade**: experimenta abordagens nunca tentadas

Drive com nível > 0.7 é considerado "pressionante" e gera goal com estratégia (explore/practice/experiment).

### CognitiveExecutive (`core/cognitive_executive.py`)
Orquestrador integrado:

```
Self tick → IntrinsicMotivation → GoalDrivenEngine → CFE → Synthesizer → Execucao → Self update
```

### PrimitiveInventionEngine (`core/primitive_invention/`)
Ciclo completo de invenção de novas primitives:

| Módulo | Função |
|--------|--------|
| `low_level_ops.py` | 18 operações universais de grid (transpose, flip, shift, tile, crop, pad) |
| `atomic_ops.py` | Operações VERDADEIRAMENTE atômicas: loop_x, loop_y, coord_transform (swap, invert_x, invert_y) |
| `gap_detector.py` | Detecta quando invariante não tem primitiva correspondente |
| `hypothesizer.py` | Busca gulosa por profundidade 1-3 sobre composições de ops |
| `verifier.py` | Valida em treino + teste (min 2 exemplos, accuracy ≥ 1.0) |
| `registrar.py` | Salva primitiva validada no registry com executor e confiança |
| `engine.py` | Engine completa: gap → hipótese → validação → registro |

**Teste definitivo**: sistema inventou `rotate_90` a partir de operandos ATÔMICOS (loop + coord_transform), sem ter `rot90_cw` em nenhum lugar. 4 primitives inventadas, 8/8 exemplos corretos cada.

```
Cada operando sozinho: MISMATCH
Composicao swap_invert_y: 8/8 OK
  for y in range(h):
    for x in range(w):
      result[h-1-y, x] = grid[y, x]   # definicao de rotacao 90
```

Ciclo completo: **Observação → Invariante → Gap → Hipótese atômica → Validação → Registro → Nova primitiva**.

### MultilingualEncoder (`core/multilingual_encoder.py`)
LaBSE (109 línguas, 768D) + JL projection para 10000D bipolar.

| Par | BGE (antes) | LaBSE (depois) | ∆ |
|-----|-------------|----------------|---|
| dog-cachorro | 0.317 | **0.753** | +0.436 |
| water-agua | 0.433 | **0.888** | +0.455 |
| Margem (mesmo vs diferente) | 0.029 | **0.300** | **10×** |

### Arquitetura completa (01/07/2026)

```
ObservationLayer
     ↓
InvariantDiscovery → GapDetector → ProgramHypothesizer → Verifier → Registrar → NOVA PRIMITIVE
     ↓
ConceptFormation (condicoes, efeitos, restricoes, 5 metricas)
     ↓
TheoryBuilder (conceitos → teorias)
     ↓
ImaginationSimulator (instancia em N dominios)
     ↓
ConstraintNetwork (CGE + Simulacao + Memoria + Programas)
     ↓
CFEBridge (conceito → primitivas preferidas)
     ↓
GoalDrivenEngine → CognitiveExecutive (Self + Drives + Goals)
     ↓
ProgramSynthesizer (busca guiada por conceitos)
     ↓
Execucao → Self update (memoria, neuromodulacao, drives)
```

### Evidence Builders (`core/evidence_builders/`)

| Builder | Função |
|---------|--------|
| TaxonomyBuilder | BFS no CGE max_depth=8 |
| EpisodicBuilder | EpisodeStore + RecallPolicy |
| PerceptualBuilder | CGE spread_activation |
| CausalBuilder | WorldModel associação/intervenção/contrafactual |

### Cognition Instrumentada (`core/cognition/`)

| Módulo | Função |
|--------|--------|
| `competition.py` | diversity_score, num_clusters, winner_reversals |
| `cognitive_delta.py` | Delta entre estados CGE |
| `abstract_primitives.py` | Infer, analogize, generalize, contradict |
| `metacognition.py` | Detecção de impasse, curiosidade |

### Scale Transforms Dataset (`core/scale_transforms_dataset.py`)

| Dataset | Amostras | Transformações |
|---------|----------|---------------|
| ScaleTransformDataset | 5000+ (20×20) | tile, crop, resize, extract_bbox, pad |

---

## 16. Cognitive Modes + WorldModel (01/07/2026)

### CognitiveModeManager (`core/cognitive_modes.py`)

5 modos operacionais que alteram o comportamento de TODOS os modulos. A DMN deixa de ser um modulo separado e passa a ser um **modo operacional da arquitetura inteira**.

| Modo | Temperatura | Beam | Busca | Energia | Gatilho |
|------|-------------|------|-------|---------|---------|
| FOCUSED | 0.2 (rigida) | 5 | 4 | -1/ciclo | Tarefa externa presente |
| EXPLORATION | 0.5 | 12 | 6 | -2/ciclo | Falha + surpresa > 0.6 |
| INCUBATION | 0.6 | 8 | 3 | -1/ciclo | Sem tarefa, energia > 0.5 |
| DREAMING | 0.9 | 20 | 2 | -3/ciclo | Energia alta, dados nao processados |
| CONSOLIDATION | 0.3 | 3 | 2 | +2/ciclo | Energia baixa ou 20+ ciclos sem consolidar |

Cada modo altera: `temperature` -> HypothesisEngine, `beam_width/max_depth` -> ProgramSynthesizer, `constraint_strictness` -> ConstraintNetwork, `concept_threshold` -> ConceptAgentRegistry.

Transicoes sao DINAMICAS baseadas em: tarefa presente, taxa de sucesso, surpresa (WorldModel), energia.

### WorldModel (`core/world_model.py`)

Modelo interno compartilhado do mundo com:
- **predict(input_state)** -> Prediction
- **update(input_state, actual_output)** -> PredictionError
- **surprise**: quao inesperado foi o erro
- **consistency**: quantas leis sao violadas

Surpresa alta -> EXPLORATION mode. Surpresa baixa -> FOCUSED / CONSOLIDATION.

### CognitiveExecutive (`core/cognitive_executive.py`)

Orquestrador integrado com cycle() + idle_cycle():

```
cycle(): Self.tick -> Mode.decide -> apply_config -> generate_goals -> execute -> update_self
idle_cycle(): Mode.decide -> INCUBATION (recombina) / DREAMING (imagina) / CONSOLIDATION (comprime)
```

Integra: CognitiveModeManager, WorldModel, HypothesisEngine, Synthesizer, ConstraintNetwork, LawRegistry, ConceptEconomy, ConflictResolver.

---

## 17. KnowledgeGraph + PubMed Real Data (01/07/2026)

### Pipeline de Conhecimento Real

```
PubMed NCBI API (esearch + efetch)
   -> 3.480 abstracts em 16 dominios
      -> NLP regex: extrai relacoes causais
         -> KnowledgeGraph: conceitos + arestas
            -> CGE: nos com HDV 10.000D (LaBSE)
```

### KnowledgeGraph (`core/knowledge_graph.py`)

Grafo causal com:
- 13.822 conceitos (PubMed reais + gerados)
- 16.491 relacoes causais
- Auto-criacao de conceitos quando target nao existe
- Indices por dominio, tipo de relacao, confianca

### Kaggle ADNI (`test_model_comparison_adni.py`)

1.857 pacientes com dados clinicos reais (MMSE, CDRSB, AGE, GENDER).
Diferenca clara entre grupos: CN (MMSE=29.1), MCI (MMSE=27.6), AD (MMSE=23.1).

### Multimodal Dataset (`build_strategic_dataset.py`)

1.136 exemplos gerados em 23 dominios, 5 modalidades (grid, symbolic, simulation, audio, event_sequence).

---

## 18. InferenceEngine — Geracao de Conhecimento NOVO

### InferenceEngine (`core/inference_engine.py`)

Motor de inferencia que NAO apenas recupera relacoes existentes. GERA conhecimento novo:

1. **CADEIA CAUSAL**: A->B + B->C => A->C (transitividade)
2. **TRANSFERENCIA CROSS-DOMINIO**: X trata Y em dominio A => X pode tratar Z (analogia)
3. **ABDUCAO**: dado efeito E, infere a causa mais provavel
4. **CONTRAPOSITIVO**: A->B => not-B -> not-A

Fuzzy matching por similaridade de tokens conecta nos com nomes diferentes mas semanticos similares.

| Inferencia | Quantidade |
|-----------|-----------|
| Cadeias causais | 216.107 |
| Transferencias cross-dominio | 16.036 |
| **Total inferencias NOVAS** | **232.143** |
| Razao vs relacoes originais | **167x** |

### HypothesisFilter (`core/hypothesis_filter.py`)

Elimina 80% de ruido: redundancia, trivialidade, falta de plausibilidade biologica.
45.256 hipoteses filtradas de 232.143 brutas. 10.784 candidatas terapeuticas.

---

## 19. ModelLibrary + ModelEvolution (01/07/2026)

### ModelLibrary (`core/model_library.py`)

18 modelos mecanisticos RIVAIS em 5 doencas:

| Doenca | Modelos | Hipotese central |
|--------|---------|------------------|
| Alzheimer | 7 | amyloid-first, tau-first, microglia-first, mitochondrial-first, metabolism-first, dual-state, etc. |
| Parkinson | 2 | alpha-synuclein propagation, mitochondrial dysfunction |
| Cancer | 3 | immune evasion |
| Depression | 3 | monoamine deficiency, inflammatory, neurogenic |
| Schizophrenia | 2 | dopamine dysregulation, glutamate dysfunction |

Cada modelo tem: variaveis, ODE, funcao de intervencao, estado inicial.

### ModelComparator (`core/model_comparator.py`)

Comparacao Bayesiana: P(modelo | dados) = P(dados | modelo) * P(modelo) / P(dados).

Usa dados REAIS do ADNI (MMSE, CDRSB) para discriminar entre modelos. Pela primeira vez, modelos rivais sao COMPARADOS contra dados de pacientes reais.

### ModelEvolution (`core/model_evolution.py`)

**Selecao natural de modelos:**

```
G10: 6 modelos iniciais
  -> Simula intervencao (BACE1 inhibition)
  -> Compara predicao com realidade (lecanemab ~30%)
  -> tau_first vence (erro=0.100)
     -> MUTA: gera filhos com novas estruturas
        -> G11: filho vence pai (erro=0.1001)
           -> MUTA novamente
              -> G12: neto vence (erro=0.0997)
                 -> CONVERGE em G13
```

**Melhoria total: 46.6%** (94% -> 50% de erro). Calibracao de parametros responde por 46.6% do ganho, mutacao estrutural por 22%.

### Discriminacao por Intervencao

A AGI discrimina modelos comparando predicoes de intervencao com ensaios clinicos REAIS:

```
tau_first: prediz +20% cognitivo. Real: ~30%. Erro: 0.100. VENCEU.
amyloid_first: prediz +1.8%. Erro: 0.282. PERDEU.
```

### ParameterCalibrator (`core/parameter_calibrator.py`)

Separa "modelo ruim" de "parametros ruins". Busca aleatoria + hill climbing para otimizar parametros ODE. 46.6% de melhoria sem mudar a estrutura do modelo.

---

## 20. DMN — CrossDomainAnalogy + ConceptInvention (01/07/2026)

### CrossDomainAnalogyEngine (`core/cross_domain_analogy.py`)

Quando modelos falham (erro > 0.5), a AGI ativa a DMN e busca analogias em dominios DISTANTES:

| Dominio | Principio | Mapeamento para Alzheimer |
|---------|----------|--------------------------|
| Ecologia | keystone species, trophic cascade, extinction | microglia=keystone_species, tau=trophic_cascade |
| Teoria de Controle | PID, hysteresis, negative feedback | microglia=controller, cognitive_decline=output_error |
| Imunologia | checkpoint, cytokine storm, tolerance | microglia=macrophage, amyloid=antigen |
| Eng. Eletrica | short circuit, brownout, cascading failure | tau=short_circuit, cognition=brownout |
| Colonias de Formigas | emergence, stigmergy, division of labor | neuron=ant, brain=colony |
| Economia | externalities, tragedy of commons | amyloid=market_externality |
| Astrofisica | gravitational collapse, entropy | tau=gravitational_collapse |

**Cada analogia gera um NOVO MODELO** com dinamica inspirada no outro dominio.

### ConceptInventor (`core/concept_inventor.py`)

Inventa conceitos NOVOS quando variaveis existentes sao insuficientes:
- `glymphatic_clearance` — se clearance falha, amyloid acumula
- `prion_like_propagation` — tau se espalha celula a celula
- `excitotoxicity_cycle` — glutamate mata neuronios em ciclo
- `epigenetic_silencing` — genes de reparo sao desligados pela idade
- `vascular_stiffening` — vasos endurecem, reduzindo oxigenio

Cada conceito = nova variavel no modelo ODE com dinamica propria.

### Ciclo DMN Completo

```
Stagnation (erro > 0.5)
   -> CrossDomainAnalogy: busca 7 dominios, 9 analogias
      -> ConceptInventor: inventa 9 conceitos novos
         -> Aplica ao modelo, testa contra realidade
            -> 2 modelos melhoraram (0.9406 -> 0.9387)
```

---

## 21. Infraestrutura Completa (01/07/2026)

### VirtualLab + Tellurium

| Backend | Performance | Funcao |
|---------|------------|--------|
| Python ODE (scipy RK45) | <50ms | Simulacao primaria |
| Tellurium SBML | 2ms | Validacao cruzada |

### Mapa de Modulos

```
                ┌─────────────┐
                │   PubMed    │ 11.109 abstracts reais
                │   Kaggle    │ 1.857 pacientes ADNI
                │   KaggleAPI │ Configurado
                └──────┬──────┘
                       ▼
            ┌──────────────────┐
            │  KnowledgeGraph  │ 13.822 conceitos, 16.491 relacoes
            │  CGE (HDV 10kD)  │ 500 nos
            └──────┬───────────┘
                   ▼
        ┌─────────────────────┐
        │  InferenceEngine    │ 232.143 inferencias (167x)
        │  HypothesisFilter   │ 80% ruido eliminado
        └──────┬──────────────┘
               ▼
    ┌─────────────────────────┐
    │  ModelLibrary           │ 18 modelos rivais, 5 doencas
    │  ModelComparator        │ Bayesiano + ADNI data
    │  ModelEvolution         │ Selecao natural (46.6% melhoria)
    │  ParameterCalibrator    │ Calibracao ODE
    └──────┬──────────────────┘
           ▼
┌──────────────────────────────┐
│  VirtualLab                  │
│  ├─ Python ODE (scipy)       │
│  └─ Tellurium SBML (2ms)     │
└──────┬───────────────────────┘
       ▼
┌──────────────────────────────┐
│  CognitiveModeManager        │
│  WorldModel                  │
│  CognitiveExecutive          │
└──────┬───────────────────────┘
       ▼
┌──────────────────────────────┐
│  DMN Loop                    │
│  ├─ CrossDomainAnalogy (7)   │
│  ├─ ConceptInventor          │
│  └─ StructureMutator         │
└──────┬───────────────────────┘
       ▼
┌──────────────────────────────┐
│  CURE PROPOSAL               │
│  tau-first: terapia anti-tau │
│  + anti-inflammatory         │
│  + suporte sinaptico         │
└──────────────────────────────┘
```

---

## 22. StructuralMetrics — Leis de Organizacao (01/07/2026)

### StructuralAnalyzer (`core/structural_metrics.py`)

Mede propriedades ESTRUTURAIS da arquitetura, nao erro. Cada metrica e computada da matriz de conectividade do modelo ODE (derivada numericamente).

| Metrica | Formula | O que mede |
|---------|---------|-----------|
| Modularity | Q de Newman | Quao bem dividido em modulos |
| Clustering | Watts-Strogatz | Vizinhos do vizinho sao vizinhos |
| Small-worldness | (C/Crand)/(L/Lrand) | >1 = small-world |
| Centrality | Degree distribution | Hubs e concentracao |
| Entropy | Shannon da dinamica | Diversidade de comportamento |
| Integrated info | Phi aproximado | Todo > soma das partes |
| Compressibility | Regularidade da matriz | Quanto pode ser simplificado |

### Descobertas experimentais

1. **Curva em sino**: criatividade maxima em 35-50 variaveis. Nao monotonicamente crescente.
2. **Modularidade otima**: 5 modulos de 10 (34.6%) > 10 modulos de 5 (24.4%) > todas conectadas (20.2%).
3. **Acoplamento critico**: forca = 0.3 da maximo ganho (45.6%). Nem fraco nem forte.
4. **Clustering prediz criatividade**: r=0.251, metrica mais correlacionada.

| Nivel de acoplamento | Ganho | Interpretacao |
|---------------------|-------|---------------|
| 0.1 (muito fraco) | 1.8% | Nao integra |
| **0.3 (critico)** | **45.6%** | **Ponto otimo** |
| 0.5 | 6.1% | Comeca a saturar |
| 0.7-0.9 (forte) | 2-13% | Satura |

### HierarchicalCalibrator (`core/hierarchical_calibrator.py`)

Calibracao modulo a modulo em vez de 100 parametros de uma vez. Cada modulo e calibrado separadamente enquanto os demais sao congelados. 3 passadas completas.

---

## 23. DynamicTypeLibrary — 7 Tipos de Dinamica (01/07/2026)

### Biblioteca de Formalismos (`core/dynamic_type_library.py`)

A AGI nao usa apenas ODE. Escolhe entre 7 tipos de dinamica:

| Tipo | Formalismo | Melhor para | Exemplo biologico |
|------|-----------|-------------|-------------------|
| Reaction-Diffusion | PDE | Propagacao espacial | Tau propagacao prion-like |
| Agent-Based | Agentes | Interacao celular | Microglia, celulas imunes |
| Graph/Network | Grafo | Conectividade | Conectoma, hubs |
| Cellular Automaton | Regras locais | Atividade cortical | Ondas epilepticas |
| Delay Differential | Atraso | Feedback lento | Ciclo sono, clearance |
| Hysteresis | Memoria | Transicoes irreversiveis | Acumulo de amyloid |
| Multiscale Hybrid | Tudo | Multi-escala | Cerebro inteiro |

### Uso

A AGI pergunta: "Qual formalismo melhor descreve este fenomeno?" e recomenda o top-3.

```
Problema: propagacao de tau
  Recomendacao:
    1. PDE (reaction-diffusion) — score 0.9
    2. Cellular automaton — score 0.4
    3. Graph/Network — score 0.2
```

---

## 24. MetaModeler — Meta-Raciocinio (01/07/2026)

### MetaModeler v1 (`core/meta_modeler.py`)

Seleciona formalismo por menor erro de calibracao. Resultado: **15%** concordancia com especialistas. O problema: otimizava para calibrabilidade, nao para adequacao conceitual.

### MetaModeler v2 (`core/meta_modeler_v2.py`)

Extrai propriedades SEMANTICAS do problema ANTES de escolher formalismo:

```
Problema (nome + dominio + variaveis)
  ↓
SemanticProblemProfile.from_text()
  ↓
Propriedades: has_diffusion, has_spatial, has_network, has_agents, has_multiscale, has_chemical, has_waves, has_population, has_signaling
  ↓
Recommenders heuristicos para cada formalismo:
  - ODE: chemical, population, signaling (penaliza spatial, network)
  - PDE: diffusion, waves, spatial (penaliza agents, network)
  - Graph: network, diffusion (penaliza chemical, population)
  - Agent: agents, spatial (penaliza chemical, waves)
  - CA: waves, spatial, diffusion
  - Hybrid: multiscale, agents + diffusion, network + spatial
  ↓
Resultado: **50%** concordancia com especialistas (+35pp vs v1)
```

### Regras de recomendacao

| Propriedade | Formalismo recomendado | Score tipico |
|------------|----------------------|-------------|
| has_diffusion + has_waves | PDE | 0.40-0.90 |
| has_network | graph_network | 0.50-0.90 |
| has_agents | agent_based | 0.40-0.60 |
| has_chemical | ODE | 0.40-0.70 |
| has_multiscale | hybrid | 0.40-0.80 |
| has_waves + has_spatial | cellular_automaton | 0.35-0.80 |

### Limitacoes atuais

1. Conjunto de 6 formalismos — faltam SDE, Boolean network, Bayesian, discrete event, Gillespie
2. Heuristicas manuais — poderiam ser aprendidas de dados
3. Validacao em modelos simplificados — falta teste com SBML real + dados experimentais

---

## 25. Experimentos de Generalizacao (01/07/2026)

### Matriz Experimental

144 combinacoes: 4 dominios × 4 formalismos × 3 otimizadores × 3 complexidades.

**Fatores que mais afetam o ganho de calibracao:**

| Fator | Melhor nivel | Gain medio | Impacto relativo |
|-------|-------------|-----------|-----------------|
| Formalismo | ODE | 25.7% | Alto |
| Otimizador | random | 28.3% | Alto |
| Complexidade | 10 vars | 18.9% | Baixo |
| Dominio | ALS | 20.9% | Baixo |

### Generalizacao (treino Alzheimer, teste sem alteracoes)

| Dominio | Transferiu? | Ganho | vs Media |
|---------|------------|-------|----------|
| ALS | **SIM** | 44.8% | +10.3pp |
| Parkinson | NAO | 19.5% | -1.8pp |
| Huntington | NAO | 6.7% | -14.8pp |

**1/3 dominios transferiu.** ALS compartilha neurodegeneracao e inflamacao com Alzheimer. A configuracao otima nao e universal.

### Robusterimento Estatistico

15 repeticoes por configuracao com t-test:

| Hipote | Resultado |
|--------|-----------|
| Sweet spot universal 35-50 vars | **FALSA** (depende do modelo base) |
| Modularidade explica criatividade | Correlacao r=0.07 (FRACA) |
| Otimizador e secundario | 47x diferenca (FALSA) |
| MetaModeler v1 escolhe como expert | 15% (FALSA) |
| Analise semantica melhora escolha | v2: 50% (CONFIRMADA) |

### Cicio de pesquisa

A sessa o produziu 5 hipoteses testadas, 4 falsificadas, 1 refinada — comportamento de programa de pesquisa cientifica.

---

## 26. TypedCognitiveGraph — Integracao das 4 Fundacoes Matematicas (12/07/2026)

### Diagnostico que motivou esta camada

A analise do `brain_live.pth` revelou que a base de conhecimento do CGE original e um **grafo de co-ocorrencia textual** (29.594 arestas, todas com `edge_type=-1`, `confidence=1.0` constante). Os conceitos `gravity`, `force`, `earth` existem mas as conexoes sao sequenciais de Wikipedia concatenada (Palavra seguinte → palavra anterior). Nao ha distincao entre "causa", "e-um", "parte-de". Isto invalida qualquer raciocinio logico sobre a base.

### Solucao: TypedCognitiveGraph (`core/typed_graph.py`)

Unifica as 4 fundacoes matematicas ja existentes em `core/math/`:

| Fundacao | Modulo | Funcao no grafo |
|----------|--------|-----------------|
| Espaco hiperbolico | `hyperbolic.py` (318 linhas) | Conceitos sao pontos na Bola de Poincare. Profundidade = distancia do centro = nivel de abstraco. Implementa `EntailmentCone` para relacoes IS-A |
| Algebra Geometrica de Clifford | `geometric_algebra.py` (408 linhas) | Relacoes sao bivectors em planos ortogonais. CAUSES ≠ IS_A ≠ PART_OF em planos geometricos distintos. Rotors implementam analogia |
| Principio da Energia Livre (FEP) | `information.py` (593 linhas) | Confianca e computada por surpresa Bayesiana, nao constante. `FreeEnergyMinimizer` atualiza crencas a cada observacao |
| Homologia Persistente | `topology.py` (378 linhas) | Grafo vira complexo simplicial. Betti numbers: β₀ = clusters de conhecimento, β₁ = raciocinio circular, β₂ = lacunas |

### Arquitetura

```
Conceito (no):
  - posicao: ponto no PoincareBall(dim=100, curvature=1.0)
  - depth: distancia do centro (abstracto → concreto)
  - cone: EntailmentCone (geometriza IS-A)

Relacao (aresta):
  - tipo: bivector Clifford (CAUSES, IS_A, PART_OF, CORRELATED, etc.)
  - confidence: computado por FEP (surprise Bayesiana apos cada observacao)
  - historico de crenca (prior → posterior → surprise)

Topologia (metacognicao):
  - Complexo simplicial de Vietoris-Rips sobre distancias hiperbolicas
  - Homologia persistente (Smith Normal Form sobre Z₂)
  - Betti numbers, entropia de persistencia, deteccao de lacunas
```

### Resultados experimentais (comparacao direta)

| Metrica | Co-ocorrencia (old) | Typed Hyperbolic (new) | Significado |
|---------|-------------------|----------------------|-------------|
| Conceitos | 802 | 1.220 | +418 de `agi_state_live.pkl` |
| Arestas | 29.594 | 38.935 | +9.341 |
| Tipos de relacao | 1 (`-1`) | 3 (CORRELATED, IS_A, PART_OF) | Relacoes agora distinguiveis |
| Confianca media | 1,0000 (constante) | 0,4185 ± 0,1034 | FEP produz variacao real |
| Valores unicos de confianca | 1 | 15 | Duvida genuina |
| Hierarquia IS-A | inexistente | 7.189 arestas, **100% profundidade correta** | Axioma verificavel |
| Topologia (Betti) | impossivel | β₀=57, β₁=0, β₂=583 | Clusters: 57; Ciclos: 0; Lacunas: 583 |

### Significado dos Betti numbers para metacognicao

- **β₀ = 57**: O conhecimento esta fragmentado em 57 grupos isolados. AGI integrada deveria ter β₀ ≈ 1.
- **β₁ = 0**: Nenhum ciclo causal (bom — sem contradicoes internas).
- **β₂ = 583**: Ha 583 "triangulos quebrados" (conexoes A→B e B→C existem mas A→C nao). Isto e CONHECIMENTO INCOMPLETO — pode ser preenchido por inferencia transitiva.

### Limitacao atual

O extrator de conhecimento (janela deslizante de bigramas do `mega_brain_compiler.py`) nao produz relacoes CAUSES ou ENABLES — apenas CORRELATED, IS_A (por profundidade) e PART_OF. Para causalidade genuina, o extrator precisa ser substituido por Open Information Extraction com parser de dependencias (spaCy `DependencyParser` ja disponivel).

---

## 27. SymbolicEngine — Raciocinio Dedutivo sobre CGE (12/07/2026)

### Arquivo

`core/cognition/symbolic_engine.py` (475 linhas)

### Funcao

Motor de inferencia logica que opera como CAMADA sobre o CGE. Le nos e arestas, aplica regras de inferencia, escreve conclusoes de volta. A geometria HDC e o substrato perceptual; a inferencia logica e o substrato do raciocinio dedutivo.

### Regras implementadas

| Regra | Logica | Implementacao |
|-------|--------|---------------|
| Instanciacao Universal | ∀x P(x)→Q(x), P(a) ⊢ Q(a) | `check_is_a(entity, category)` + universal_rules |
| Modus Ponens | A→B, A ⊢ B | Busca arestas CAUSES no CGE |
| Modus Tollens | A→B, ¬B ⊢ ¬A | Dicionario de negacoes + universal_rules |
| Silogismo Hipotetico (Barbara) | A→B, B→C ⊢ A→C | Cadeia de universal_rules |
| Cadeia Relacional | A rel B, B rel C ⊢ A rel C | Travessia de grafo + relational_rules |

(A implementacao completa com unificacao, cache BGE e reset de estado esta em `core/cognition/symbolic_engine.py`)

### Resultados

| Teste | BGE-only | SymbolicEngine | Baseline |
|-------|----------|---------------|----------|
| SyntLog (200 silogismos, conclusao fornecida) | 35,5% | **100%** | chance media 40.6% |
| SyntLog cego (sem conclusao) | — | **51,9% recall, 77,5% especificidade** | chance 50% |

O teste cego (adversarial) revela que o motor confirma conclusoes melhor do que as gera autonomamente. Precisao 0,902 quando aceita, mas recall 0,519 quando precisa produzir a conclusao sem pista.

---

## Proximos Passos (Priorizados)

1. **OpenIE Knowledge Extractor**: Substituir o extrator de co-ocorrencia textual por OpenIE com spaCy `DependencyParser`, gerando relacoes CAUSES, ENABLES, PRECEDES reais
2. **Integrar SymbolicEngine ao TypedCognitiveGraph**: Usar typed edges (IS_A, CAUSES) como entrada para regras de inferencia, substituindo heuristicas bivector-based
3. **Conexao CGE→TypedCognitiveGraph**: Interface de migracao para que o pipeline use typed_graph.py como fonte primaria de conhecimento

---

## 28. OpenIE Knowledge Extractor — Relacoes Tipadas por Parsing de Dependencias (12/07/2026)

### Arquivo

`core/openie_extractor.py` (220 linhas)

### Problema resolvido

O extrator de conhecimento anterior usava janela deslizante de bigramas (`mega_brain_compiler.py`): se duas palavras apareciam proximas no texto, viravam arestas no CGE. Isto produzia "conhecimento" do tipo `ferry_corsten -> force` (um DJ holandes conectado a "forca" porque artigos diferentes foram concatenados).

### Solucao

O OpenIE Knowledge Extractor usa o parser de dependencias do spaCy (`en_core_web_sm`) para extrair triplas sujeito-verbo-objeto com tipagem semantica:

| Tipo | Origem na gramatica | Exemplo real |
|------|--------------------|-------------|
| **CAUSES** | Verbo "cause/lead/produce/create/make/induce" + SVO | `gravity CAUSES objects` |
| **ENABLES** | Verbo "enable/allow/permit/facilitate" + SVO | `the sun ENABLES photosynthesis` |
| **PREVENTS** | Verbo "prevent/block/inhibit/reduce" + SVO | `friction PREVENTS motion` |
| **FOLLOWS** | Verbo "follow/precede/occur" + SVO | `acceleration FOLLOWS force` |
| **AFFECTS** | Verbo "affect/influence/change/modify" + SVO | `heat AFFECTS the temperature` |
| **IS_A** | Copula (X is Y) | `water IS_A liquid` |
| **HAS_PROPERTY** | Verbo "have/contain/include/possess" + SVO | `cells HAS_PROPERTY dna` |

### Resultados experimentais

| Metrica | Co-ocorrencia (janela bigrama) | OpenIE (parser dependencias) |
|---------|-------------------------------|------------------------------|
| Sentencas processadas | (nao aplicavel) | 58 em 4.7s |
| Triplas extraidas | (nao aplicavel) | 67 |
| Relacoes CAUSES/ENABLES/PREVENTS | **0** | **16** (causalidade real) |
| Relacoes IS_A | heuristicas (depth threshold) | **14** (de copula gramatical) |
| Confianca | constante 1.0 | 0.70 media (negacao/modal penalizam) |
| Extracao | janela de tokens | base gramatical (sujeito-verbo-objeto) |

### Integracao com TypedCognitiveGraph

O pipeline fechado agora e:

```
Texto cientifico
  -> spacy.DependencyParser (SVO + copula + posse)
    -> OpenIEKnowledgeExtractor.extract_from_text()
      -> triplas tipadas: {subject, relation, object, confidence, source}
        -> TypedCognitiveGraph.add_relation(subject, relation, object, confidence)
          -> Topology.build_topology() (Betti numbers)
            -> Metacognicao: clusters, lacunas, ciclos
```

### Proximo gargalo

O corpus de 58 frases e toy. Para conhecimento real, precisa processar Wikipedia ou PubMed em escala (10.000+ artigos). O extractor processa ~12 sentencas/segundo na RTX 3050 (CPU, modelo pequeno). Para 1 milhao de sentencas (Wikipedia inteira), ~22 horas.
