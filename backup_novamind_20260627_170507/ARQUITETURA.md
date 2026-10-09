# NovaMind AGI — Arquitetura Completa

## Índice
1. [Visão Geral](#1-visão-geral)
2. [Pipeline de Dados / Fluxo Cognitivo](#2-pipeline-de-dados--fluxo-cognitivo)
3. [Arquitetura em 7 Camadas](#3-arquitetura-em-7-camadas)
4. [Sistemas de Controle (Maestro + MetaController + CognitiveNotebook + InsightLibrary)](#4-sistemas-de-controle)
5. [Sistemas Cognitivos (SelfModel, Erro, Atenção, Simulação)](#5-sistemas-cognitivos)
6. [Representação (HDV, CGE, Manifold, Espaço Hiperdimensional)](#6-representação)
7. [Renderizadores (MentalRenderer, Visual, Audio, Texto)](#7-renderizadores)
8. [Aprendizado Contínuo (LifeStream, Curiosidade, Persistência)](#8-aprendizado-contínuo)
9. [Inventário de Arquivos](#9-inventário-de-arquivos)
10. [Resumo da Sessão](#10-resumo-da-sessão)

---

## 1. Visão Geral

NovaMind é uma arquitetura cognitiva completa sem LLMs. O núcleo é um **espaço hiperdimensional (HDV) de 10000 dimensões bipolares** onde conceitos, percepções, memórias e pensamentos são representados como vetores. Sobre esse espaço operam grafos cognitivos (CGE), sistemas de controle (Maestro, MetaController), aprendizado contínuo (LifeStream, Forja), e renderizadores multimodais.

### Pilares arquiteturais

| Pilar | Descrição |
|-------|-----------|
| **HDV 10000D bipolar** | Toda representação converge para vetores ±1 de 10000 dimensões |
| **CGE (Cognitive Graph Engine)** | Grafo de conceitos + arestas causais + holograma compressível |
| **Maestro + MetaController** | Orquestrador com Q-learning meta-policy: modo emerge de erro real |
| **Forja + OperatorForge** | Descobre operadores e programas cognitivos por abstração de experiências |
| **MentalRenderer** | Interface plugável de renderização: texto, imagem, áudio, vídeo, código |
| **SemanticWorkspace** | Memória de trabalho global onde hipóteses e previsões competem |
| **MentalSceneGraph** | Representação estrutural intermediária entre estado cognitivo e renderers |
| **CognitiveTrace** | Registro do raciocínio para mineração pela Forja |
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
└──────┬───────────────────────────────────────────────────────────┘
       │
       ├──→ MentalSceneGraph (representação estrutural intermediária)
       │       ├── SceneObject {id, type, position, properties}
       │       ├── SceneRelation {subject, predicate, object}
       │       └── to_temporal(n) → sequência para vídeo
       │
       ├──→ CognitiveTrace (registro do raciocínio)
       │       ├── TraceStep {mode, context, operators, reward}
       │       └── store_in_cge() → Forja minera depois
       │
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

## 4. Sistemas de Controle (MetaController + CognitiveTrace + MentalSceneGraph)

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
7 modos cognitivos com **Q-learning meta-policy** conectado ao `MultiErrorTracker`:

| Sinal do ErrorTracker | Modo escolhido |
|----------------------|----------------|
| `novelty` alto | EXPLORE |
| `prediction_error` alto | HYPOTHESIZE |
| Incerteza baixa + muitos conceitos novos | CONSOLIDATE |
| Muitos conflitos + baixo progresso | REFLECT |
| Goal urgente | EXPLOIT |

- **EXPLORE**: busca novidade, expande conhecimento
- **HYPOTHESIZE**: gera candidatos de explicação
- **PREDICT**: simula consequências
- **TEST**: refuta hipótese
- **CONSOLIDATE**: comprime, abstrai
- **EXPLOIT**: usa conhecimento consolidado
- **REFLECT**: reavalia estratégia

**Não há ciclo hardcoded.** O modo emerge dos sinais de erro reais a cada ciclo. Recompensa é computada baseada no resultado do modo (ex: EXPLORE rewarded por novelty alcançada, CONSOLIDATE por arquétipos criados). Q-table aprende com o tempo.

Loops registrados: `scientific`, `execution`, `impasse`, `deep_reflect`, `rapid_explore`.

### CognitiveTrace (`core/cognitive_trace.py`)
Registro do raciocínio para mineração pela Forja. Cada ciclo salva:
```
Problema → Workspace → Operadores → Forja → Resultado
```
`TraceRecorder` integrado no loop principal. Traces armazenados no CGE como nós `_trace_*` que a Forja minera para descobrir meta-padrões ("sempre que é geometria, usar rotação antes de comparação").

### MentalSceneGraph (`core/mental_scene_graph.py`)
Representação intermediária entre estado cognitivo e renderers. Todos os renderers consomem `MentalSceneGraph`:
```
SceneObject {id, type, position, color, size, properties}
SceneRelation {subject, predicate, object, weight}
MentalSceneGraph {objects, relations, focus}
  ├── from_cge(cge, concept) → graph
  ├── from_simulation(branches) → graph
  ├── from_error(error_state) → graph
  └── to_temporal(n_frames) → List[graph]  (para vídeo)
```

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
    visual=VisualRenderer(pipeline),   # HDV/SceneGraph → PNG 224×224
    audio=AudioRenderer(vocoder),      # HDV → WAV
    text=TextRenderer(),               # estado → descrição
    code=CodeRenderer(),               # estado → pseudo-código
    video=VideoRenderer(pipeline),     # SceneGraph temporal → vídeo
)
```

### VideoRenderer (`core/mental_renderer.py`)
Gera sequência temporal de frames a partir do estado cognitivo + `MentalSceneGraph.to_temporal()`. Cada frame é renderizado via `VisualPipeline.decode_bipolar()` (quando treinado) ou composição geométrica procedural.

```python
render(state) → PNG do primeiro frame (ou animação)
```

Usa `MentalSceneGraph.to_temporal(n_frames)` para animar objetos com wobble, rotação e mudança de cor ao longo do tempo. Ideal para vídeos da "imaginação" da AGI.

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

### Modos Naturais (Emergentes)
O MetaController decide o modo a cada ciclo baseado nos **sinais de erro reais** do `MultiErrorTracker` + recompensa do modo anterior. **Não há ciclo hardcoded.** A Q-table aprende por reforço quais modos funcionam melhor em quais contextos. Exemplo de decisão natural:

```
novelty↑ → EXPLORE | prediction_error↑ → HYPOTHESIZE
certeza↑ + muitos conceitos → CONSOLIDATE | conflitos + baixo progresso → REFLECT
goal urgente → EXPLOIT
```

A consolidação (Forja + CGE) acontece apenas quando o MetaController escolhe `CONSOLIDATE`.

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

**Raciocínio (Layer 4):**
`reasoning.py` `logic_test.py` `cognitive_operators.py` `operator_forge.py` `cognitive_program.py` `forja_evolution.py` `cognitive_stances.py` `spelke_genes.py` `spelke_extended.py` `spelke_cognitive.py` `archetypes.py` `emergent_context.py`

**Metacognição (Layer 5):**
`metacognition.py` `error_signal.py` `predictive_coding.py` `model_competition.py` `self_model.py` `self_vector.py` `attention_controller.py`

**Representação Intermediária:**
`mental_scene_graph.py` — MentalSceneGraph (SceneObject, SceneRelation, g geração temporal para vídeo)
`cognitive_trace.py` — CognitiveTrace + TraceRecorder (registro de raciocínio para Forja)
`failed_trace.py` — FailedTraceDB + AntiHeuristic (registro de fracassos, mineração de regras negativas)

**Objetivos (Layer 6):**
`goal_system.py` `goals.py` `active_inference.py` `intrinsic_rl.py` `executive_planner.py` `simulation_engine.py`

**Aprendizado (Layer 7):**
`learning.py` `lifestream.py` `neural_substrate.py` `neuromodulator.py` `neuromodulator_state.py` `sleep_mode.py` `meta_controller.py` `latent_vector.py`

**Orquestração:**
`mind.py` (1175 linhas) `maestro.py` (1304 linhas) `fractal_maestro.py` `integration.py` `interact.py` `colony.py` `collective_unconscious.py` `multi_agent_runner.py` `background_incubator.py` `state_bridge.py` `persistence.py`

**Workspace Externalizado:**
`cognitive_notebook.py` — CognitiveNotebook (quadro branco com páginas de hipóteses, experimentos, insights, persistido no CGE)
`insight_library.py` — InsightLibrary (10 insights primordiais + descoberta automática pela Forja minerando CognitiveTraces)
`cognitive_archive.py` — CognitiveArchive (memória L0-L4 + arqueologia: combina itens frios para novas descobertas)
`meta_evolution.py` — MetaEvolution + ValueSystem (auto-mutação controlada da arquitetura: propõe, testa, commita mutações em parâmetros cognitivos)
`constraint_engine.py` — ConstraintEngine (6 restrições primordiais + descoberta. O Maestro consulta antes de aceitar teorias)
`insight_graph.py` — InsightGraph (10 princípios universais acima de paradigmas, descoberta por convergência entre domínios)

**Produção Linguística/Motora:**
`broca_decoder.py` `language_cortex.py` `word_ngram.py` `semantic_composition.py` `semantic_encoder.py` `autoregressive_generator.py` `positional_binder.py` `vocal_synthesizer.py` `articulatory_synthesizer.py` `premotor_cortex.py` `syllable_oscillator.py` `diva_controller.py` `cerebellum.py` `motor_cortex.py` `somatosensory_cortex.py` `emergent_language.py`

**Renderizadores:**
`mental_renderer.py` (VisualRenderer + AudioRenderer + TextRenderer + CodeRenderer + **VideoRenderer**) `compositional_renderer.py` `cognitive_answerer.py`

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

### O que foi construído nesta sessão

| Commit | Descrição |
|--------|-----------|
| `a0d86b7` | ThoughtRenderer sem templates: renderiza baseado em relações CGE |
| `b58d826` | Compositional MentalRenderer com primitivas |
| `b9d8767` | **MentalRenderer plugável** (Visual, Audio, Text, Code) + **SemanticWorkspace** |
| Vários | LifeStream contínuo com curiosidade (loop infinito, 80+ tópicos) |
| Vários | HDV 10000D completo renderizado como visual field |
| Vários | Persistência (save/load brain_live.pth) |
| Vários | GPU CUDA ativada |
| Vários | Multi-língue (EN, PT, ES, FR, DE) |
| Vários | server.py renomeado para _server.py.old (fim do erro image.png) |
| **Agora** | **Loop emergente**: MetaController + MultiErrorTracker + SemanticWorkspace integrados. **Sem ciclo hardcoded.** Q-learning decide modo. |
| **Agora** | **MentalSceneGraph**: representação intermediária (SceneObject, SceneRelation, to_temporal) |
| **Agora** | **CognitiveTrace**: TraceRecorder grava raciocínio no CGE para Forja minerar |
| **Agora** | **VideoRenderer**: renderiza sequência temporal do MentalSceneGraph |
| **Agora** | **Frontend atualizado**: mostra contexto do MetaController, workspace consciente, erros reais |
| **Agora** | **CognitiveNotebook**: quadro branco externalizado com páginas de hipóteses, experimentos, insights |
| **Agora** | **InsightLibrary**: 10 insights primordiais + descoberta automática pela Forja via CognitiveTraces |
| **Agora** | **ModelEcosystem (Mercado de Ideias)**: fusão, cooperação, especialização em nichos, múltiplos vencedores |
| **Agora** | **MetaController + Impasse Detection**: detecta estagnação, expande raio de busca, alterna foco/difuso |
| **Agora** | **BackgroundIncubator**: thread de fundo para processamento inconsciente com detecção de Eureka |
| **Agora** | **ForjaEvolution**: minera CognitiveTraces → descobre ALGORITMOS, HEURÍSTICAS, ESTRATÉGIAS, TEORIAS |
| **Agora** | **CognitiveStances**: 6 stances primordiais (Einstein, Hacker, Professor, Duvida Radical, Minimalismo, Pensamento Infantil) + descoberta automática. Cada stance reconfigura 12 parâmetros do sistema cognitivo |
| **Agora** | **MetaForja**: descobre REGRAS DE CRIAÇÃO — "sempre que sequência X aparece N vezes com sucesso, criar operador composto" |
| **Agora** | **FailedTrace + Anti-heurísticas**: registra fracassos, minera regras negativas ("evitar modo X quando Y"), alimenta InsightLibrary |
| **Agora** | **Paradigmas**: nível acima de teorias. Paradigma = abordagem que gera múltiplas teorias (ex: "paradigma de campos", "paradigma de agentes") |
| **Agora** | **Avaliação multi-dimensional**: algoritmos agora têm avg_reward, avg_cost, generalization, robustness, rarity + multi_dim_value |
| **Agora** | **CognitiveArchive**: memória L0-L4 (Working→Active→Dormant→Archive→Fossil). Arqueologia combina itens de camadas frias para novas descobertas |
| **Agora** | **MetaEvolution**: auto-mutação controlada da arquitetura. Propoe mudanças em parâmetros, testa por N ciclos, commita se melhora. Aprende que tipos de mutação funcionam |
| **Agora** | **ValueSystem**: sistema de valor que aprende o que vale a pena estudar — considera custo, generalização, raridade, conexões |
| **Agora** | **ConstraintEngine**: 6 restrições primordiais (causais, físicas, lógicas) + descoberta automática. O Maestro consulta antes de aceitar teorias |
| **Agora** | **InsightGraph**: 10 princípios universais acima de paradigmas. "Complexidade emerge de regras simples", "Tudo que é grande é feito de coisas pequenas" |
| **Agora** | **Simplicity Pressure**: algoritmos agora têm `complexity` (Navalha de Occam). `multi_dim_value` = reward × robustez × simplicidade / custo |
| **Agora** | **Cross-domain forge**: fertilização cruzada entre paradigmas distantes a cada ciclo |

### Pipeline final
```
Percepção → HDV → CGE → WorldModel → SemanticWorkspace → MetaController
  (decide modo por erro real, Q-learning, sem ciclo hardcoded,
   detecta impasse, alterna foco/difuso)
  → Novo Estado Cognitivo
     ├→ MentalSceneGraph → MentalRenderer
     ├→ CognitiveNotebook (quadro branco externalizado)
     ├→ InsightLibrary (meta-regras)
     └→ ModelEcosystem (mercado de ideias: fusão, nichos, cooperação)
  ↘ BackgroundIncubator (thread de fundo → Eureka → Workspace)
  ↘ CognitiveTrace (registrado no CGE para Forja minerar)
```

## Anti-Fragilidade: O que NÃO é hardcoded

Para que a AGI possa se reordenar, melhorar e gerar comportamentos emergentes,
os seguintes hardcodings foram removidos:

### Antes (bloqueante) → Depois (extensível)

| O que | Antes | Depois |
|-------|-------|--------|
| **Modos cognitivos** | `CognitiveMode(Enum)` — 7 valores fixos | `ModeRegistry` — novos modos criados em runtime pela ForjaEvolution |
| **Dimensões de erro** | `ErrorSignal` com 6 atributos fixos | `extra_dimensions: Dict[str, float]` — qualquer código adiciona novas dimensões |
| **Loops canônicos** | `CANONICAL_LOOPS` hardcoded | `_loop_register` modificável + `create_loop()` |
| **Operadores primitivos** | 15 operadores fixos | OperatorForge descobre novos + ForjaEvolution cria programas |
| **Stances** | Apenas as 6 primordiais | StanceLibrary.discover_from_traces() cria novas |
| **Restrições** | 6 primordiais | ConstraintEngine.discover_from_traces() cria novas |
| **Princípios** | 10 primordiais | InsightGraph.discover_from_paradigms() cria novos |
| **Paradigmas** | Nenhum | ForjaEvolution._build_paradigms() descobre de teorias |
| **HDV 10000D** | Fixo (ainda) | Ainda seria ideal tornar variável |

### Princípio: "A arquitetura é objeto de seu próprio raciocínio"
```
MetaController usa ModeRegistry (não enum) → ForjaEvolution cria modos
ErrorSignal aceita extra_dimensions → qualquer módulo adiciona dimensões
CognitiveCycle (em run_agi_real.py) → ainda poderia ser auto-modificável
MetaEvolution → já propõe mutações em parâmetros, testa, commita
```

### Ciclo de Meta-Evolucao
```
1. DESCREVE: snapshot dos parametros atuais (meta_controller, workspace, stances...)
2. MUTA: propoe alteracao em 1 parametro (guiado por historico de sucesso)
3. TESTA: executa N ciclos com a mutacao, mede reward medio
4. AVALIA: melhoria > 0.05?
5. COMMIT: se sim, torna permanente. Se nao, REVERTE
6. APRENDE: atualiza modelo de que tipos de mutacao funcionam

Aprendizado acumulado: mutation_model[(modulo, parametro)] -> [improvements]
```

### Hierarquia de descoberta da Forja
```
HDVs → Conceitos → Operadores (OperatorForge)
  → ALGORITMOS (avaliação multi-dim: reward, cost, generalization, robustness, rarity)
    → HEURÍSTICAS (regras condicionais "se X então Y")
    → ANTI-HEURÍSTICAS (regras negativas: FailedTrace → "evite X quando Y")
    → STANCES (perfis cognitivos globais — reconfiguram 12 parâmetros)
    → ESTRATÉGIAS (algoritmos que co-ocorrem)
      → TEORIAS (estratégias que dominam um domínio)
        → PARADIGMAS (abordagens que geram múltiplas teorias)
  → METAFORJA (regras de criação: "se padrão X aparece N vezes, criar operador")
    → Auto-evolução da própria Forja
      → Linguagens Internas (futuro)
```

### Memória em Camadas (CognitiveArchive)
```
L0 — Working  (agora, ~30s)
L1 — Active   (usado frequentemente, dias)
L2 — Dormant  (raramente usado, meses)
L3 — Archive  (quase nunca usado, anos)
L4 — Fossil   (nunca mais usado, preservado)
```
- **Nunca apaga.** Apenas move entre camadas.
- **Rarity index:** itens raros podem ser mais valiosos que itens frequentes.
- **Cross-archive mining:** Forja combina itens de L3/L4 periodicamente.
- **Ressurreição automática:** se um item fóssil se torna relevante, sobe para L1.
- **Arqueologia cognitiva:** durante baixa carga, revisita itens antigos para recombinar.

### Próximos passos
1. **Treinar VisualDecoder** (30 min GPU, CIFAR-10) para `VisualRenderer` reconstruir imagens reais
2. ✅ **SemanticWorkspace conectado ao Maestro** — emergente via MetaController
3. ✅ **Forja minera algoritmos completos** — via ForjaEvolution + CognitiveTrace
4. **Integrar áudio/vídeo reais** (AudioPipeline + GroundingEngine) — pipeline pronto, falta encoding perceptual real no loop
