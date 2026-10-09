# NovaMind — Estado Consolidado (honesto)

> Resumo do que está **provado**, do que está **em aberto**, e do que é **fronteira de pesquisa**.
> Princípio do projeto: ambição alta, **honestidade brutal**, nunca fingir sucesso.

## ✅ Provado (com número, reproduzível)

### 1. Memória cross-domínio FATORADA (`core/cross_domain.py`, `sci/cross/run_transfer.py`)
O risco real (memória cross-domínio ingênua vira mingau) e a solução (fatorar: estrutura compartilhada +
conteúdo do domínio + inferência de contexto). Em domínios controlados que compartilham estrutura:

| Métrica | Ingênuo (blob) | Fatorado | Isolado |
|---|---|---|---|
| Domínio de treino | 15% (interferência) | **85.6%** | — |
| Domínio **novo** (few-shot) | — | **85.3%** (transfere) | 29.5% (sem transferência) |
| Discriminação de domínio | — | **97.9%** (sabe em que mundo está) | — |

Veredito: a memória fatorada **transfere sem interferência e distingue os mundos**. É o "unificador".
Lição metodológica: meta-treinar com **domínio novo por episódio** (senão decora; a v1 decorou 8 domínios fixos).

### 2. Laboratório in-silico (`sci/lab/`)
Pipeline completo de triagem terapêutica, ancorado em bases reais:
- `/analyze` (doença → hipóteses → gauntlet PubMed+ClinicalTrials → corpo 3D)
- `/refine` (loop auto-refinante: gera→verifica→re-gera→converge) — ELA convergiu em TDP-43/SOD1/C9orf72
- `/crispr` (modo genético: harmonia = causa-raiz + sem colateral) — ELA → C9orf72 base-editing
- `/cross` (nós-mecanismo pan-doença) — neuro → SOD1/NLRP3/SQSTM1 (4/4 doenças)
- Histórico persistente + **dossiê científico HTML+CSV** (imprimível → PDF, com narrativa da arquitetura)
- Braços agênticos ReAct (`agent.py`)
Honesto: triagem **computacional** de hipóteses, não cura; wet-lab obrigatório.

## ⚠️ Em aberto (localizado, não resolvido)

### 3. Indução-de-regra visual estilo ARC (`core/grid_cognition.py`, `sci/cross/diag_grid.py`)
Tese (correta): exposição multi-prior → priors transferíveis → tarefa de regras novas (como humanos no ARC).
Diagnóstico metódico (escada), grades densas + peso na loss + métrica ACERTO EXATO:
- **Capacidade: ✅ 100%** — aprende UMA transformação fixa (depois de corrigir bug de pos-embedding).
- **Indução-de-regra few-shot: ❌ 0%** — inferir QUAL das 8 transformações do suporte trava em **todas** as
  3 arquiteturas (gargalo, por-célula+CLS, in-context).

Conclusão honesta: o gargalo **é a indução-de-regra few-shot** — exatamente o núcleo do que o ARC mede e o que
o **campo inteiro ainda não resolveu**. Localizado com precisão; **não** é tweak de transformer numa 3050.
Hipótese mais promissora p/ destravar (a testar no Colab): **grades com objetos de cores distintas** (não
ruído denso) tornam a correspondência in→out inferível — ver `sci/cross/colab_arc_induction.py`.

## 🔬 Fronteira (pesquisa, precisa de Colab/escala)
- Cracar a indução-de-regra few-shot (curriculum object-based + escala) — Colab T4.
- Plugar xadrez (`sci/chess/selfplay.py`) e grades no **mesmo** store e medir transferência cross-jogo real.

## Direção estratégica (decidida)
NÃO pré-treinar foundation model (armadilha de GPU). O LLM = largura/prior; nossa arquitetura = composição +
world-model + experiência aterrada + verificação. O "unificador" (memória fatorada) é a peça-chave provada.
