# Hipótese de engenharia reversa — o algoritmo que gera generalidade autêntica

> Objetivo: não copiar o cérebro célula a célula (inviável), mas **reverter o ALGORITMO** que ele roda e que
> produz o que falta na nossa AGI: induzir uma **regra nova** a partir de **2-3 exemplos**. Tudo aqui é viável
> numa RTX 3050 e honesto sobre o que é fato vs hipótese.

---

## 1. A pergunta certa
Errado: "que arquitetura de rede copia o cérebro?"
Certo: **"que ALGORITMO o cérebro executa que faz few-shot rule induction com generalização composicional?"**
Reverter o algoritmo = reproduzir a **assinatura computacional** dele (comportamento + estrutura interna), não
o substrato biológico. (Por isso EEG/ondas não servem: medem o substrato grosseiro, não o algoritmo.)

## 2. Convergência de 5 teorias reais (a figura que emerge)
| Teoria | Quem | O que diz sobre generalidade |
|---|---|---|
| Predictive coding / Free energy | Friston | cérebro = modelo generativo hierárquico que minimiza erro de predição = **compressão (MDL)** |
| Tolman-Eichenbaum Machine / grid cells | Whittington, Behrens | **FATORA** estrutura (relações, aprendida 1×) de conteúdo (sensorial) → transfere estrutura a conteúdo novo |
| Thousand Brains / coluna cortical | Hawkins, Mountcastle | **UM** algoritmo repetido: modelos sensoriomotores em *reference frames*; generalidade = compor reference frames |
| Bayesian Program Learning / Language of Thought | Lake, Tenenbaum, Goodman | indução de **PROGRAMA composicional** com prior de simplicidade; few-shot = inferência Bayesiana sobre programas |
| Global Workspace | Dehaene, Baars | gargalo serial que **compõe** processos paralelos vencedores |

**Síntese (o algoritmo candidato):**
> Generalidade = **inferência composicional (montar um programa)** sobre um **modelo generativo FATORADO
> (estrutura ⟂ conteúdo)**, com **prior de compressão/simplicidade (MDL)**, onde **estruturas reutilizáveis são
> COMPOSTAS** para explicar conteúdo novo — e a biblioteca dessas estruturas é construída por **predição
> auto-supervisionada** (sem rótulos), ao longo de muita experiência.

## 3. Por que isso explica o nosso platô de 27% E o sucesso humano
- Nossos solvers aprendem **regras de conteúdo** → saturam, porque o ARC é **anti-reuso de regras** (cada tarefa, regra nova).
- O humano **não aprende a regra da tarefa**: ele **compõe uma ESTRUTURA nova** a partir de uma **biblioteca de
  primitivas estruturais** (aprendida numa vida), guiado por **simplicidade**. Ele já viu "gravidade", "conter",
  "espelhar", "contar" em mil contextos; few-shot é só **inferir qual composição dessas primitivas** explica os 2-3 exemplos.
- O "missing piece" não é um truque único. São três peças acopladas:
  **(a)** a biblioteca composicional de estruturas; **(b)** o motor que compõe uma estrutura NOVA sob MDL;
  **(c)** o bootstrap auto-supervisionado (predictive coding) que constrói (a) **sem rótulos**.

## 4. A HIPÓTESE reversível (falsificável)
**H:** *A generalidade autêntica emerge quando um modelo (1) APRENDE POR PREDIÇÃO AUTO-SUPERVISIONADA, sem
rótulos, um espaço **fatorado** estrutura⟂conteúdo; e (2) faz **inferência composicional guiada por compressão**
para montar uma **estrutura nova** que explique 2-3 exemplos.*

Predições testáveis (se H é verdade):
- P1: a fatoração estrutura⟂conteúdo **emerge sozinha** da predição auto-supervisionada (sem supervisão da regra).
- P2: essa representação fatorada **acelera** a indução few-shot de transformações **NUNCA vistas**.
- P3: o viés de **simplicidade (MDL)** na composição prevê **quais** tarefas o modelo acerta (as de menor descrição).
- P4: compor **primitivas estruturais** (não regras inteiras) **fura o muro anti-reuso** do ARC.

## 5. Experimento mínimo VIÁVEL (3050, dias — não meses, não nuvem)
**"Teste do Bootstrap" — reverte o algoritmo casando a assinatura, não o cérebro:**
- **Fase A (auto-supervisão):** pré-treinar um modelo pequeno (1-3M params, cabe em 4GB) **só por predição**
  (mascarar/prever partes) em grades sintéticas **estruturadas** — **sem nunca mostrar uma transformação rotulada**.
- **Fase B (sondagem):** medir na representação aprendida:
  - (B1) sondas lineares: a estrutura (relações: conter/adjacente/rank) está **separada** do conteúdo (cores)? → testa P1.
  - (B2) few-shot: dado 2-3 exemplos de uma transformação **nova**, a representação fatorada **acelera** a indução
    vs o mesmo modelo **sem** o pré-treino preditivo (controle)? → testa P2.
  - (B3) correlação descrição-mínima ↔ acerto → testa P3.
- **Foothold real do projeto:** já medimos **85% de transferência** com world-model fatorado (TEM) vs 15% blob
  (`core/cross_domain.py`). Isso é **meia Fase B já feita**. Falta a **Fase A** (o bootstrap auto-supervisionado
  que constrói a biblioteca sem rótulo) e a **composição de estrutura NOVA** (P4).
- **Critério de morte:** se a fatoração **não emerge** da predição pura **e** não acelera few-shot → predictive
  coding sozinho não basta; o prior composicional precisa ser **explícito** (volta pra DSL/program-induction).

## 6. Honestidade brutal (pra não te enganar nem aos labs)
- Isto é uma **HIPÓTESE**, não a resposta. **Ninguém** cravou "o algoritmo do cérebro" — é problema aberto.
- Mesmo se H passar, **generalidade plena de AGI exige ESCALA**: a biblioteca humana vem de **uma vida de dados**.
  O experimento testa o **PRINCÍPIO** (o algoritmo certo), **não entrega AGI** num laptop.
- "Engenharia reversa" aqui = casar a **assinatura computacional** (few-shot + fatoração + viés de simplicidade +
  transferência estrutura→conteúdo), **não** escanear cérebro. EEG/ondas são grosseiros demais (vide discussão anterior).
- Valor real e honesto: se as Fases A/B derem sinal, você tem **evidência de qual princípio gera generalidade** —
  isso **sim** é interessante pra um laboratório, com o enquadramento certo (NeuroAI / aprendizado composicional),
  não como "AGI pronta".
