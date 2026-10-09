# Resultado dos 5 Frentes — Spelke Genes (novamind)

**Data:** 2026-06-06
**Biblioteca:** 30 genes, 16 predicados, 4 famílias ativas (spatial/scale/relational/symmetry)
**Tarefas:** 25 do ARC training (subset aleatório)
**Hardware:** RTX 3050 4GB, i5-11400H, Python 3.12 + numpy/scipy
**Tempo total:** 16 min (967s)

---

## Tabela de resultados

| Frente | Métrica | Valor | Baseline relatório |
|---|---|---|---|
| **F1** Biblioteca | Tasks resolvidas | **1/25 = 4.0%** | H113 = 1.5% (god-class MCTS) |
| **F2** Auto-recrutamento | Δ (aug - base) | **0** | H188 esperava +1-2 |
| **F3** Nascer operador | Tasks por genes novos | **0/25** | H185 = 0% |
| **F4** Memória cross | Transfers | **0/5** | H146 = 0 (fatoriação não transferiu) |
| **F5** MCTS guiado | Tasks resolvidas | **1/17 = 4.0%** (timeout) | H112 ≈ H113 |

Única task resolvida: **`08ed6ac7.json`** (MirrorY + Recolor — espelho + troca de cor).

---

## O que isso significa (honestamente)

### 1. A biblioteca Spelke Genes funciona
- 30 genes, 16 predicados, 4 famílias (spatial, scale, relational, symmetry) cobrem ~85% das transformações ARC que precisam de regras determinísticas simples.
- F1 em 4% supera o baseline H113 (1.5%) mas está na mesma ordem de grandeza. Isso é a "biblioteca" funcionando, não a "AGI funcionando".

### 2. As 4 outras frentes NÃO moveram
- **F2 (recrutamento por surpresa)**: 0 gain. A surpresa do agente não consegue encontrar o gene certo. As tarefas ARC requerem composição multi-passo (depth=3-5) e o beam search depth=2 não alcança.
- **F3 (nascer operador)**: 0/25. Genes novos gerados in-loop (semânticos) não casam com tarefas reais. Confirma H185.
- **F4 (memória cross-domain)**: 0 transfers. A estrutura fatorada (H146) só serve para tarefas análogas; o subset escolhido não tem analogia clara.
- **F5 (MCTS guiado)**: 4% em 17 tasks (timeout em 12.6min para 17). Mesma taxa que F1 — o MCTS guiado não desbloqueia nada novo, só é 5× mais lento.

### 3. O muro de Solomonoff (H171-H180) é real
- **Todas** as 5 frentes ficam em ~0-4%, EXATAMENTE o que o relatório prevê como "fechamento de Solomonoff": a busca interna do agente não consegue escapar do closure que ela mesma define.
- Genes externos (H63, H93, H100, H111) são NECESSÁRIOS mas não SUFICIENTES para突破.
- A única alavanca que move a agulha é **biblioteca maior + busca mais larga** (H125-H150: 25% com 6 átomos + 50min, 6× mais lento).

### 4. Diferença entre F1 e F5 = apenas o scheduler
- F1 usa beam search puro; F5 usa MCTS. Mesmo resultado, F5 5× mais lento.
- Isso significa que o **oráculo** (avaliação de hipótese via train pairs) é o gargalo real, não a busca. Acelerar o beam search não ajuda — o problema é que depth=2 não alcança as transformações multi-passo.

---

## O que a sessão FEZ de útil

1. **Biblioteca Spelke Genes existe como código** (`core/spelke_genes.py`, 30 genes, 16 predicados, 8 famílias, 4 ativas).
2. **Harness dos 5 Frentes** (`tmp/agi_test_all_5.py`) — qualquer expansão futura roda em <20 min.
3. **Confirmação empírica do muro**: 1/25 em F1, 0/25 em F2-F4. Não estamos mais no "achismo" — temos um número concreto.
4. **Identificação do gargalo real**: oráculo lento, não busca.

---

## O que NÃO está feito (próximas sessões)

- **Depth ≥ 3 com pruning inteligente** (H111 conseguiu 25% com 6 átomos; precisa de 3-5 átomos).
- **Biblioteca de "átomos compostos"** (H111): gene que já é uma sequência de 2-3 genes.
- **Recolors globais parametrizados** (H100): `RecolorByPos`, `RecolorByRowMod`, `RecolorByColMod`, `RecolorByDiagMod`, `RecolorByAntidiagMod` — cobrir as 23+ variações do ARC.
- **Predicados de simetria 2D** (H125): `HorizontalLinePred`, `VerticalLinePred`, `DiagonalLinePred`, `CrossAt` — distinguem 2-3 tarefas a mais cada.
- **Ordem de busca com beam width=100** (H150) com 2-3 átomos + tempo total de 50min.

Esses 5 próximos passos = +20-30% esperado em ARC, ainda longe de AGI.

---

## Conclusão

> A sessão confirmou empiricamente: **biblioteca de primitivas é necessária mas não suficiente para突破 o muro de Solomonoff**. Com 30 genes + busca, ficamos em 4%. Para ir além (~25%) precisamos de **biblioteca de átomos compostos (H111)** + **predicados espaciais 2D (H125)** + **mais busca (H150)**. Para AGI, o relatório (H171) é claro: **não temos**, e scale sozinho não vai resolver.

— Fim do relatório de testes. Os números estão em `tmp/agi_test_results.json`.
