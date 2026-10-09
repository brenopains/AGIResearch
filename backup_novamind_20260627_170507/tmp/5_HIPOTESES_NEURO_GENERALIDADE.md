# 5 hipóteses NOVAS — generalidade via alinhamento neuro + a nossa arquitetura

> Linha: usar a nossa arquitetura ÚNICA (world-model fatorado + maestro/bússola + motor-célula MDL + forge +
> grafo-relacional + memória/espelho) como substrato, e um algoritmo que a MODULA rumo à generalidade autêntica.
> Honestidade: nenhuma "chega na AGI" — testam PRINCÍPIOS. Cada uma vem com viabilidade REAL marcada.
> Tags: [RODA-3050-JÁ] runnable aqui agora | [PRECISA-DADO] depende de dataset neural | [PRECISA-NUVEM] compute extra.

---

## H1 — Alinhamento ALGORÍTMICO a fMRI de raciocínio relacional (não representacional)  [PRECISA-DADO]
**Novo:** todo NeuroAI alinha features sensoriais (visão). Aqui forçamos o nosso **world-model fatorado** a
reproduzir a **fatoração estrutura⟂conteúdo** que o cérebro mostra **durante raciocínio relacional** (córtex
pré-frontal/parietal), não no córtex visual — alinhamento do **algoritmo/dinâmica**, não do snapshot.
**Dado real:** HCP (Human Connectome Project) tem a *relational-processing task* em fMRI, pública.
**Experimento:** extrair a assinatura de fatoração do HCP-relational → restringir o nosso store fatorado a
reproduzi-la → medir se melhora few-shot vs sem. **Mata se:** a direção causal for invertida (capacidade→cérebro).
**Viável?** Não hoje: HCP exige registro + download grande + pipeline fMRI. É um programa de semanas.
**Prob. honesta de mover a agulha:** ~0.15 (alto risco, mas é o único genuinamente novo com dado real).

## H2 — Bootstrap auto-supervisionado da fatoração (a fatoração EMERGE só de prever?)  [RODA-3050-JÁ]
**Novo p/ nós:** testar se o prior fatorado da NOSSA arquitetura **emerge sozinho** de predição auto-supervisionada
(mascarar/prever), **sem nunca ver uma transformação rotulada** — e se isso acelera induzir regras novas.
**Experimento:** pré-treino só-predição em grades estruturadas → sonda linear (a estrutura separou do conteúdo?)
→ few-shot em transformações nunca vistas vs controle sem pré-treino. **Mata se:** fatoração não emerge nem acelera.
**Foothold:** já temos 85% transfer com store fatorado (core/cross_domain.py) — meia Fase B feita.
**Prob.:** ~0.3.

## H3 — NEUROMODULAÇÃO como busca-de-arquitetura em tempo real (a sua intuição do "algoritmo que modula")  [RODA-3050-JÁ]
**Novo de verdade:** quase todo NeuroAI é estático. Aqui um **meta-controlador tipo neuromodulador**
(dopamina/acetilcolina/noradrenalina) **reconfigura em tempo real** QUAIS módulos ficam ativos, a plasticidade, e
o ganho da BÚSSOLA — disparado por um sinal **intrínseco** (surpresa / progresso de compressão). É a nossa maestro
virando um cérebro que **se remodula** conforme a tarefa, não uma rede fixa.
**Experimento:** treinar o modulador (RL leve / bandit) p/ maximizar Δcompressão; medir se a generalização few-shot
sobe vs maestro estático. **Mata se:** modulação não bate o estático. **Prob.:** ~0.25 (e é o mais alinhado com você).

## H4 — Consolidação por compressão em SLOTS DE ESTRUTURA (sono/wake → grid-cells, não DSL plana)  [RODA-3050-JÁ]
**Novo:** DreamCoder cresce biblioteca de primitivas PLANAS (e o conselho mostrou que satura). Aqui a consolidação
("sono") comprime sub-estruturas recorrentes em **slots de ESTRUTURA reutilizável** do world-model fatorado
(estilo grid-cell), não em primitivas isoladas. **Experimento:** wake-sleep que promove sub-estruturas a slots;
medir compounding cross-tarefa. **Mata se:** repete o muro do DreamCoder (NOVAS=0). **Prob.:** ~0.2 (parte já tentada).

## H5 — Transferência de ESTRUTURA cross-domínio como assinatura de AGI  [RODA-3050-JÁ]
**Novo:** quase todo trabalho de ARC é mono-domínio. Aqui testamos a tese central: aprender uma ESTRUTURA no
domínio A e transferir **zero/few-shot** pro domínio B com conteúdo novo (ARC ↔ Minecraft-relações ↔ xadrez),
usando o store fatorado. É a "evolução exponencial" que você quer, medida. **Foothold forte:** já deu 85% transfer
controlado. **Experimento:** estrutura aprendida em A → held-out em B → mede transferência vs blob. **Mata se:** a
transferência cai a ~chance quando o conteúdo muda. **Prob.:** ~0.35 (a mais provável de dar sinal positivo).

---

## Veredito de priorização (honesto)
- **Rodar agora (3050):** H2, H3, H5 — todas usam a nossa arquitetura, sem dado externo.
- **A mais alinhada com você:** H3 (o algoritmo que MODULA a arquitetura em tempo real).
- **A mais provável de dar sinal positivo:** H5 (tem foothold de 85%).
- **A genuinamente nova com dado real do cérebro:** H1 — mas é a inviável agora (precisa HCP).
- **"Testar todas":** as 3 runnable dá pra rodar; H1 e H4 não valem a pena agora (dado/saturação).
- **Verdade final:** nenhuma "vira AGI". Elas medem QUAL princípio gera generalidade. Sinal positivo numa delas =
  evidência publicável e direção real; sinal nulo = honestidade (descartamos e estreitamos). Nada de fingir.
