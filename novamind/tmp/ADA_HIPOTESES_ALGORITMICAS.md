# O que a Ada Lovelace criaria hoje — algoritmos pra induzir ESTRUTURA (a peça que falta)

> O H5 cravou o alvo: transferir CONTEÚDO = 80%; transferir ESTRUTURA nova = 8% (acaso). A peça que falta é um
> OPERADOR DE INDUÇÃO DE ESTRUTURA. A Ada via a Máquina Analítica "tecendo padrões algébricos como o tear de
> Jacquard tece flores". Hoje ela atacaria isto: não fatorar conteúdo, mas **inferir a GRAMÁTICA de uma estrutura
> inédita a partir de 2-3 exemplos** — operando sobre RELAÇÕES, com a LEI MÍNIMA (compressão).
> Cada hipótese: o algoritmo + o princípio de cérebro de ALTA-RES que a inspira + viabilidade.

---

## A1 — O OPERADOR DE INDUÇÃO (meta-aprender a induzir)  [RODA-3050-JÁ · vou rodar contra o H5]
**Ideia:** o H5 falhou porque a arquitetura **não tem** um inferidor de estrutura — a `StructureCore` é aditiva
(viés errado) e o contexto só re-rotula conteúdo. A correção: um **transformer que faz indução IN-CONTEXT** —
dado K exemplos (s,a,s2) de uma estrutura QUALQUER + uma query (s,a), ele **atende** ao suporte e prevê s2,
**generalizando** a estrutura (K cobre só parte dos pares → tem que inferir a regra, não decorar).
**Cérebro (alta-res):** o **hipocampo** forma esquemas relacionais RÁPIDO (one-shot) e o córtex entorrinal
codifica estrutura fatorada — o sistema que infere "o mapa" de poucas amostras. **Ataca o 8% direto.**
**Teste:** bater o 8% do H5 em estrutura arbitrária held-out. **Mata se:** ficar ~chance mesmo com atenção in-context.

## A2 — O TEAR DE RELAÇÕES (indução por alinhamento analógico)  [RODA-3050]
**Ideia (a mais "Ada"):** a regra É o **morfismo** que alinha o grafo-de-relações da entrada ao da saída.
Em vez de prever pixels, **casar objetos⟶objetos** por suas relações (conter/adjacente/rank) via graph-matching /
transporte ótimo, e **extrair a transformação como o mapeamento que torna os dois grafos correspondentes**.
**Cérebro:** córtex **pré-frontal/parietal** faz raciocínio relacional e analogia (structure-mapping, Gentner).
**Viável:** sim (graph-matching é barato). **Novo:** opera sobre relações, não conteúdo — exatamente o que o H5 pede.

## A3 — A LEI MÍNIMA (busca neural por compressão / MDL diferenciável)  [RODA-3050 · pesado]
**Ideia:** o objetivo literal vira **comprimir os demos** — achar o **menor programa neural** (sequência de
operadores relacionais) que os regenera. Gradiente + busca discreta co-otimizam o comprimento-de-descrição.
**Cérebro (alta-res):** **predictive coding / energia livre** (Friston) — o córtex minimiza erro de predição =
compressão hierárquica. **Viável:** sim, mas lento. **É a "lei mínima que gera a variedade" da Ada, formalizada.**

## A4 — O REPLAY (inferência de estrutura offline, estilo sono)  [RODA-3050]
**Ideia:** dado os demos, rodar **inferência iterativa** (message-passing / EM) que ajusta um **grafo relacional
latente** até explicá-los — como o **replay hipocampal** consolida estrutura no descanso. Não é forward único;
é "pensar de novo" até o ponto-fixo (liga com o refinamento recorrente que já temos).
**Cérebro (alta-res):** **replay hipocampal** (sequências reativadas) que constroem o mapa cognitivo offline.
**Viável:** sim. **Novo:** estrutura por inferência iterativa, não por uma passada.

## A5 — AS CÉLULAS DE ESQUEMA (substrato fatorado p/ estrutura ARBITRÁRIA, não só aditiva)  [RODA-3050]
**Ideia:** trocar a `StructureCore` aditiva por um **substrato de grid-cells generalizado** que possa representar
**qualquer** grafo de transição (não só adição cíclica) — uma base de "células de esquema" que o contexto
seleciona/combina. Conserta a causa-raiz do 8% (o viés aditivo). **Cérebro:** **grid cells entorrinais**
(Hafting, Behrens/TEM) — base fatorada que generaliza estrutura. **Viável:** sim, é refatorar o `StructureCore`.

---

## Prioridade (informada pelo H5)
1. **A1 (operador de indução in-context)** — ataca o 8% direto e é o teste mais limpo de "a peça que falta é construível?". **RODANDO AGORA.**
2. **A2 (tear de relações)** e **A5 (grid-cells generalizado)** — se A1 der sinal, são os reforços.
3. **A3 (MDL)** e **A4 (replay)** — princípios fundamentais, mais caros.

**Honestidade:** "alta-res do cérebro" aqui = usar os **achados** (grid cells, replay, esquemas hipocampais,
predictive coding) como **especificação de projeto** das peças — não baixar fMRI (isso é o programa de semanas da H1).
Nenhuma "vira AGI"; A1 testa se o gargalo medido (8% → ?) é construível. Sinal = direção real; nulo = honestidade.
