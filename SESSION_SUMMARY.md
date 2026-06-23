# Sessão: Fala Fluida HDC + MLP Autoregressiva + Blueprint ASI
## Data: 21 Jun 2026 | Duração: ~4h

---

## ✅ CONQUISTAS CONSOLIDADAS

### PositionalBinder (core/positional_binder.py)
- **Fala fluida via binding posicional**: `POS_i * word_i`, decode por unbind
- **Chunking (Lei de Miller)**: chunk_size=4, SNR ~50:1, sem crosstalk
- **Validação**: "the dog runs" → sim=0.50 cada palavra; "dog playing guitar in the park" (7 pal, 2 chunks) → 100% correto
- **Marco**: ordem sequencial emerge da geometria HDC, sem next-token prediction

### AutoregressiveGenerator (core/autoregressive_generator.py)
- **Pipeline**: MLP (preferida) → W matrix (fallback) → target chunks (fallback final)
- **MLP**: `Linear(10000,4000) → ReLU → Linear(4000,10000)`, 80M params, saída FLOAT (sem sign)
  - Treino: 286K pares Wikipedia em 11 min (CosineEmbeddingLoss + AMP)
  - Saída float normalizada: sim=0.5-0.9 com palavras do vocabulário
  - Bias "the" devido ao domínio único de treino (Wikipedia inglesa)
- **W matrix**: descontinuada (colapso na média estatística)
- **Fallback**: target chunks garantem output correto quando MLP não prediz

### NumericEncoder (core/numeric_encoder.py)
- Level Encoding: ativa bloco de dimensões proporcional a n
- Similaridade Jaccard: sim(5,6)=0.91, sim(5,100)=0.17
- Reconhecimento textual via GroundingEngine (n-gram hash)

### Infraestrutura de Dados
- **Simple Wikipedia**: 5000 artigos processados, 775K pares de transição
- **Vocabulário**: 137.899 palavras únicas salvas em `wiki_vocab.txt`
- **MLP treinada**: `D:\novamind_data\transition_mlp.pt` (320 MB)

---

## ❌ FALHAS DOCUMENTADAS

### Matriz W (Outer Product)
- `W = sum(outer(next, current))` — álgebra fechada
- Colapso: weighted average de 775K transições vira ruído
- Resultado: "the" domina todas as posições (sim ~0.24)

### nn.Linear + Gradient Descent
- `nn.Linear(10000, 10000)` treinado com CosineEmbeddingLoss + AdamW
- Mesmo colapso que outer product: 10000 dims insuficientes para diversidade
- Cosine sim ~0.12-0.17 (pouco acima de aleatório)

### Lição Fundamental
**Memória (armazenamento) precisa ser binária/esparsa (±1). Raciocínio (processamento) precisa ser contínuo (float).** O `sign()` destrói a magnitude da predição. MLP com saída float dá sim 4-5× maior que binarizada.

---

## 🧠 BLUEPRINT ASI (4 Mecanismos Pendentes)

Diagnóstico baseado em fMRI 7T, ECoG, optogenética 2023-2026:

### 1. Neuromodulação Multi-Dimensional
- **Atual**: Maestro usa Free Energy escalar (1D)
- **Necessário**: Vetor [DA, NA, ACh, 5HT]
  - DA (Dopamina): progresso, direciona exploração
  - NA (Noradrenalina): surpresa, foca atenção no erro
  - ACh (Acetilcolina): plasticidade, abre janela de aprendizado
  - 5HT (Serotonina): inibição de impulsos, status
- **Impacto**: Forja só dispara quando NA (surpresa) + ACh (plasticidade) simultâneos

### 2. Modo Sleep (Consolidação Offline)
- **Atual**: Aprendizado em tempo real, sem replay
- **Necessário**: Quando PC ocioso, Maestro pega 1000 chunks aleatórios do HDV e recombina no Sandbox
- **Impacto**: Combinações que reduzem Free Energy são consolidadas como novos Archetypes
- **Mecanismo**: Descoberta analógica (sorte determinística)

### 3. Vetor SELF (Affordances)
- **Atual**: SandboxEngine processa cenas sem observador
- **Necessário**: Vetor fixo SELF. Toda percepção passa por bind com SELF
- **Impacto**: Maçã não é só Manifold vermelho — é vermelho + "ao alcance da mão"
- **Mecanismo**: Cria noção de "meu ambiente", permite regras de ação física

### 4. Buffer Temporal Hierárquico
- **Atual**: Chunks de 4 palavras isolados (sem contexto entre chunks)
- **Necessário**: Buffer deslizante dos últimos 10 chunks com decaimento
- **Impacto**: Contexto do chunk 1 ressoa (fracamente) no chunk 11
- **Mecanismo**: Bundle contínuo com decaimento exponencial no hiperespaço

---

## 📁 ARQUIVOS RELEVANTES

| Arquivo | Descrição |
|---------|-----------|
| `core/positional_binder.py` | PositionalBinder: fala fluida posicional |
| `core/autoregressive_generator.py` | MLP + W + fallback |
| `core/numeric_encoder.py` | Intuição numérica Jaccard |
| `core/interact.py` | REPL com /generate, /math |
| `novamind/scripts/train_mlp_mixed.py` | Dataset misto Wikipedia + CodeSearchNet |
| `D:\novamind_data\transition_mlp.pt` | MLP treinada (Wikipedia, 286K pares) |
| `D:\novamind_data\wiki_vocab.txt` | 137K palavras do vocabulário |

---

## 📐 CONSTANTES CRÍTICAS

- Dimensão HDC: 10.000
- Tamanho vetor: 40 KB (float32) / 10 KB (bipolar ±1)
- Chunk size: 4 palavras (Lei de Miller)
- MLP: 80M parâmetros (Linear 10K→4K + Linear 4K→10K)
- W matrix: 100M parâmetros (10K×10K) — DESCONTINUADA
- VRAM: RTX 3050 4GB
- RAM: 16GB DDR4
- CPU: i5-11400H
