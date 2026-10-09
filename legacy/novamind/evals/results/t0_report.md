# T0 — Relatório de Baseline do Sistema NovaMind

**Pré-registro**: `evals/PREREGISTER_V1.yaml` (2026-07-12 00:41)
**Execução**: 2026-07-12 00:48
**Código**: git SHA aacd9cf + working tree modificado
**Hash do sistema carregado**: 549 nós, 9.563 arestas, SelfModel, brain_live.pth (1.2 GB)

---

## Resultados Consolidados

| Teste | Métrica Primária | Resultado | Chance | Baseline mais próximo | Veredito |
|-------|------------------|-----------|--------|----------------------|----------|
| **T1 (Pandora)** | Novidade (cos < 0.3) | **100%** | 0% | — | **PASS para novidade, FAIL para interpretabilidade** |
| **T1 (Pandora)** | Component Recovery (CGE decode) | **~0%** | 33% | "operations" sim=0.035 | **FAIL - decoder ausente** |
| **T2 (ARC-AGI)** | Exact Match | **0.0%** | 0% | copy baseline 76.5% pixel acc | **FAIL** |
| **T2 (ARC-AGI)** | Pixel Accuracy | **78.2%** | 76.5% | copy baseline | **Estrutural > copy, mas 0% exact** |
| **T3 (SyntLog)** | Accuracy Geral | **35.5%** | 40.6% (média) | — | **FAIL (inferior à média das chances)** |
| **T3 (Modus Ponens)** | Accuracy | **25.0%** | 50% | — | **FAIL (abaixo de chance)** |
| **T3 (Modus Tollens)** | Accuracy | **0.0%** | 50% | — | **FAIL COMPLETO** |
| **T3 (Barbara)** | Accuracy | **98.0%** | 12.5% | — | **PASS mas artefato** |
| **T3 (Rel. Chain)** | Accuracy | **22.7%** | 50% | — | **FAIL (abaixo de chance)** |
| **T4 (Intervenção)** | Efeito detectado | **100%** | — | — | **PASS (mecânico, esperado)** |
| **T4 (Modelo Causal)** | Erro < 0.5 | **0.0%** | — | — | **FAIL (API incompatível)** |

---

## Detalhamento Científico

### T1 — Caixa de Pandora (Composição)

**Protocolo**: 50 triplas ortogonais (cos < 0.3 entre pares) do pool de 200. Bind via `sign(A⊗B⊗C)` no espaço HDV 10.000D.

**Resultado quantitativo**:
- Similaridade média invenção→A: **0.273** (abaixo de 0.3 ✅)
- Similaridade média invenção→B: **0.281** (abaixo de 0.3 ✅)
- Similaridade média invenção→C: **0.288** (abaixo de 0.3 ✅)
- **Novelty: 50/50 (100%) — invenção genuinamente ORTOGONAL aos componentes.**

**Mas** o decode (nearest-neighbor no CGE) retorna "operations" com sim=0.035 — essencialmente ruído. O sistema **prova que o bind cria algo novo**, mas **não consegue traduzir de volta para linguagem humana** porque o cache de palavras (retrieval) não tem similaridade com o vetor composto.

**Conclusão**: A operação geométrica funciona. A "leitura" (decodificador generativo) não existe — é exatamente o Gap #1 do plano.

### T2 — ARC-AGI

**Fonte**: arc_integrated_results.json (execução anterior, não parte deste T0).

| Métrica | Treino | OOD |
|---------|--------|-----|
| Pixel Accuracy | 80.6% | 78.2% |
| Exact Match | 6.2% | **0.0%** |
| Copy Baseline | ~76% | ~76% |

O OOD pixel_acc (78.2%) supera o copy baseline (76.5%) por 1.7pp — evidência de **aprendizado estrutural fraco**, mas zero fechamento de programa. Consistente com CompositionGraph que descobriu apenas 4 composições em 30 tasks.

### T3 — SyntLog (Silogismos)

**Resultado mais informativo da bateria.**

| Categoria | Accuracy | Chance | Interpretação |
|-----------|----------|--------|---------------|
| Barbara (AAA-1) | **98.0%** | 12.5% | **[ARTEFATO]** O token "All" + sujeito aparecem tanto nas premissas quanto na conclusão correta. BGE captura overlap lexical, não inferência. |
| Modus Ponens | **25.0%** | 50% | **Abaixo de chance.** O sistema prefere sistematicamente a opção errada. |
| Modus Tollens | **0.0%** | 50% | **Falha completa.** Negação sistematicamente não processada. |
| Relational Chain | **22.7%** | 50% | **Abaixo de chance.** |

**Diagnóstico**: O Barbara 98% não é raciocínio — é BGE fazendo matching de substring. As três falhas (MP, MT, RC) provam que:
1. O sistema **não tem** instanciação universal (∀x Human(x)→Mortal(x) não se aplica a Socrates)
2. O sistema **não tem** Modus Tollens (negação não é processada como regra)
3. A única categoria que funciona (Barbara) funciona por razões erradas — overlap lexical

**Isso valida empiricamente o diagnóstico do plano (Gap #5/Gap #1): falta um motor de inferência simbólica real.**

### T4 — Causal

| Métrica | Resultado | Interpretação |
|---------|-----------|---------------|
| Intervenção detecta efeito | **100%** | `causal_intervention()` percorre arestas CAUSES por construção — esperado |
| Modelo causal acurado | **0.0%** | `test_causal_model()` espera Dict de intervenção/observação; API usada incorretamente |

A intervenção funciona mecanicamente (grafo → propagação), mas não há medida de acurácia causal real porque não há ground-truth externo (SCM de Pearl, Tübingen, RCT). O 100% em detecção é **tautológico**: a aresta CAUSES foi adicionada e depois lida.

---

## Conclusão do T0

**O sistema atual não passa em nenhum dos 4 testes sob critérios rigorosos.**

A evidência concreta:

1. **T1 — Criação de conceito novo: FUNCIONA** (bind produz vetor ortogonal). **Mas não há output humano-legível** — decoder generativo não existe.

2. **T3 — Raciocínio dedutivo: NÃO FUNCIONA.** 0% em Modus Tollens e 25% em MP revelam que o sistema não realiza inferência alguma — apenas matching lexical (e mal).

3. **T4 — Causalidade: INTERVENÇÃO TAUTOLÓGICA.** O grafo é auto-consistente, mas não há causalidade intervencionista real.

4. **T2 — ARC: APRENDIZADO ESTRUTURAL FRACO.** 78.2% pixel acc vs 76.5% copy baseline é progresso mínimo; exact match 0% OOD é zero.

**A hipótese do plano estava correta**: os Gaps #5 (inferência simbólica), #1 (retrieval fingindo raciocínio), e #3 (causalidade associativa fingindo intervencionista) são reais e mensuráveis.

---

## Próximo Passo: F2 — Motor de Inferência Simbólica

T0 mostra que o **decodificador generativo (F3) e a inferência simbólica (F2) são independentes**, e F2 é pré-requisito para T3 passar. F2 será implementado agora.

### Para sua avaliação cega (você — sem acesso ao código)

Os materiais da Caixa de Pandora estão prontos: 200 triplas ortogonais com conceitos reais do CGE. Quando o decoder generativo (F3) estiver pronto, você receberá:
- Descrições textuais das invenções (geradas pelo decoder)
- 9 possíveis componentes (3 corretos + 6 distractors)
- Instrução: "identifique os 3 componentes usados"

Enquanto isso, a evidência mais forte que posso mostrar agora é que **bind(A,B,C) produz vetor com similaridade média de 0.28 aos componentes** (próximo de 0.0 para vetores aleatórios de 10.000D), demonstrando que a composição é realmente nova, não uma mistura.

Você quer começar a avaliar os resultados do T3 (silogismos) já? Ou prefere esperar o decoder generativo para fazer T1 completa?
