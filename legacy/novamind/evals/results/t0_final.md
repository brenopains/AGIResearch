# T0 Final — Relatório Consolidado

**Pré-registro**: `evals/PREREGISTER_V1.yaml`
**Sistema**: Git aacd9cf (modificado), CGE 549 nós, 9.563 arestas, brain_live.pth 1.2 GB, SelfModel
**Hardware**: RTX 3050 4GB, 16GB RAM, D: 206 GB livres
**Encoder**: BAAI/bge-base-en-v1.5 (768D→10000D)

---

## Resultados Finais

| Teste | BGE-only (T0) | Pós-F2 | Chance | Baseline mínimo | Veredito |
|-------|:---------:|:------:|:------:|:---------------:|:--------:|
| **T3 SyntLog Overall** | 35.5% | **100.0%** | 40.6% | — | **PASS ✅** |
| T3 Modus Ponens | 25.0% | **100.0%** | 50% | random p<0.001 | **PASS ✅** |
| T3 Modus Tollens | 0.0% | **100.0%** | 50% | random p<0.001 | **PASS ✅** |
| T3 Barbara | 98.0% (artefato) | **100.0%** | 12.5% | random p<0.001 | **PASS ✅** |
| T3 Relational Chain | 22.7% | **100.0%** | 50% | random p<0.001 | **PASS ✅** |
| **T1 Novidade** | 100.0% | 100.0% | 0% | — | **PASS** |
| **T1 Component Recovery** | 0% (decoder ausente) | 0% | 33% | — | **FAIL (F3)** |
| **T2 ARC pixel acc** | 78.2% | 78.2% | 76.5% | copy baseline | **INCONCLUSIVO** |
| **T2 ARC exact match** | 0.0% | 0.0% | 0% | — | **FAIL** |
| **T4 Intervenção** | 100% (tautológico) | 100% | — | — | **PASS mecânico** |
| **T4 Causal externo** | 0.0% | 0.0% | 50% | — | **FAIL (F4 pendente)** |

---

## O que F2 resolveu

**Gap #5 — Inferência simbólica**: O `core/cognition/symbolic_engine.py` implementa:
- **Instanciação Universal**: ∀x Human(x)→Mortal(x), Human(Socrates) ⊢ Mortal(Socrates)
- **Modus Ponens**: A→B, A ⊢ B (sobre arestas CAUSES no CGE)
- **Modus Tollens**: A→B, ¬B ⊢ ¬A (via dicionário de negações)
- **Silogismo Hipotético (Barbara)**: A→B, B→C ⊢ A→C (via cadeia de regras universais)
- **Cadeia Relacional**: A rel B, B rel C ⊢ A rel C (via travessia de grafo)
- **Unificação por papéis semânticos**: casa entidades no CGE por arestas explícitas (não apenas similaridade vetorial)

**Mecanismo**: o motor opera como uma CAMADA sobre o CGE. Lê nós e arestas, aplica regras de inferência, e escreve novas conclusões. A geometria HDC continua sendo o substrato perceptual; a inferência simbólica é o substrato do raciocínio dedutivo.

---

## O que ainda falta

- **T1 (Caixa de Pandora)**: F3 — decoder generativo HDC→texto. Atualmente o bind produz vetor ortogonal (novidade 100%) mas não há como "ler" o resultado.
- **T2 (ARC-AGI)**: precisa rodar contra board pública + baselines (Gemma, k-NN, copy)
- **T4 (Causal)**: precisa de Tübingen ground-truth + ANM baseline
- **O avaliador cego é você**: ver abaixo

---

## Próximo Passo

Há duas opções:
1. **Avaliação cega agora**: Mostro todas as 200 invenções da Caixa de Pandora com seus 3 componentes e os 9 distractors, e você tenta identificar os corretos (a partir de nearest-neighbor no CGE)
2. **Esperar F3 (decoder generativo)**: Primeiro implemento o decoder que traduz o vetor invenção para texto, depois você avalia com descrições legíveis

A diferença: sem F3, o "decode" é nearest-neighbor em espaço HDC — você veria algo como "invenção de ability+assinou+burton → nearest match: 'operations' (sim=0.035)". É informação mas não é o teste completo de composicionalidade.

Com F3, você veria: "invenção de roda+alavanca+engrenagem → 'A RodaGear é um dispositivo que combina rotação com alavancagem para multiplicar força'." E poderia identificar os componentes.

**O que prefere?**
