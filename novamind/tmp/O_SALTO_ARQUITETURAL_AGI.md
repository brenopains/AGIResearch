# O salto: remontar o ALGORITMO cerebral numa arquitetura plástica (sem LLM, sem transformer comum)

> Tese: o transformer capturou UMA peça do cérebro — **atenção** (roteamento por conteúdo). Ele NÃO capturou a
> maquinaria de GENERALIZAÇÃO do cérebro. Nosso H5 provou onde falhamos (estrutura nova = acaso) e o A1 provou que
> indução é construível (>acaso). O salto que falta — o "transformer da AGI" — é tornar **primitivas diferenciáveis**
> duas coisas que nenhuma rede atual tem: **plasticidade ESTRUTURAL** e **estrutura COMPONÍVEL**. Tudo ancorado em
> neurociência de alta-resolução. Honesto: é direção de pesquisa, não AGI garantida; mas é o ângulo certo.

---

## Por que o transformer NÃO é o salto da AGI
- Transformer = **atenção diferenciável** (roteamento por conteúdo) + arquitetura **FIXA** + regra de aprendizado **FIXA** (SGD).
- O cérebro: arquitetura que **se reescreve**, regra de aprendizado que **muda com o contexto**, e estrutura
  representada como **base componível** (não pesos congelados). Três coisas que o transformer não tem.

## Os 4 pilares do salto (cada um = um achado de alta-res do cérebro)

### Pilar 1 — PLASTICIDADE ESTRUTURAL (a rede reescreve a própria fiação)
**Cérebro (2-photon, conectômica):** sinapses nascem/morrem todo dia (spine turnover ~5-10%/semana), há neurogênese,
e circuitos **se religam** com a experiência. Nossas redes têm topologia **fixa**; só os pesos mudam.
**Salto:** tornar **diferenciável o ato de criar/podar/rotear conexões e alocar módulos novos** — a arquitetura
vira um objeto que a própria aprendizagem esculpe. (Foothold real: differentiable plasticity / Miconi; growing nets.
Ninguém chegou a AGI — mas ninguém juntou com os outros 3 pilares.)

### Pilar 2 — ESTRUTURA COMPONÍVEL (base tipo grid-cell)  ← conserta o H5 direto
**Cérebro (alta-res):** **grid cells** entorrinais formam uma **base** (quase-Fourier) do espaço; o cérebro
representa qualquer trajeto como **combinação** dessa base (path integration generaliza pra lugares nunca visitados).
**Salto:** representar ESTRUTURA não como pesos fixos, mas como **combinação de uma base de primitivas estruturais
reutilizáveis** → uma estrutura NOVA é uma **nova combinação da base**, não um aprendizado do zero. Isso ataca
exatamente o 8% do H5: lá a estrutura era um peso congelado (aditivo); aqui ela é **composta on-the-fly**.

### Pilar 3 — META-PLASTICIDADE NEUROMODULADA (a regra de aprendizado MUDA com o contexto)
**Cérebro:** dopamina/acetilcolina/noradrenalina/serotonina **reconfiguram em tempo real** a taxa de plasticidade,
o que consolida, exploração vs exploração, ganho de atenção — conforme surpresa/recompensa/incerteza. O cérebro
**não tem UMA regra de aprendizado fixa**; tem um **meta-controlador** que troca a regra por situação.
**Salto:** a LEI DE APRENDIZADO vira **aprendida e modulada** (meta-plasticidade), não SGD fixo. É a "maleabilidade"
que você pediu: a rede ajusta *como ela aprende*, não só *o que* aprende. (Foothold: neuromodulated plasticity nets.)

### Pilar 4 — NÚCLEO PREDITIVO/GENERATIVO (prever e agir, não classificar)
**Cérebro:** predictive coding / active inference (Friston) — o córtex é um **modelo generativo** que minimiza erro
de predição e **age** pra confirmar previsões; é **recorrente**, não feedforward.
**Salto:** o núcleo prevê o mundo e usa o **erro** como sinal que dirige tanto a inferência (pensar de novo até o
ponto-fixo) quanto a plasticidade. Liga com o refinamento recorrente que já temos.

## A equação do salto (a frase-resumo)
> Transformer = **atenção diferenciável**.
> AGI-equivalente = **plasticidade estrutural diferenciável + estrutura componível (base grid-cell) + regra de
> aprendizado neuromodulada + núcleo preditivo** — uma **Máquina Plástica Composicional** que esculpe a própria
> arquitetura e COMPÕE estrutura nova de uma base, sob um meta-controlador que muda como ela aprende.

## Por que isto é "remontar o algoritmo do cérebro" e não copiá-lo
Não escaneamos neurônios (inviável). Pegamos os **4 princípios algorítmicos** que a alta-resolução revelou
(rewiring, base componível, neuromodulação, predição) e os tornamos **primitivas diferenciáveis da nossa
arquitetura** — exatamente como o transformer pegou "atenção" do cérebro e a tornou diferenciável.

## O teste mínimo que prova/mata (viável na 3050)
O Pilar 2 é o que ataca o H5 direto e é o mais barato: **dar ao indutor (A1) uma BASE COMPONÍVEL tipo grid-cell** e
medir se ele **compõe estrutura nova** e salta do 18.8% pra alto no benchmark de estrutura inédita.
- **Sinal (sobe forte):** a estrutura componível É a peça que faltava → primeiro tijolo do salto, construível aqui.
- **Plató (~20%):** a base sozinha não basta; precisa dos pilares 1+3 juntos (a hipótese de que o salto é a
  COMBINAÇÃO, não uma peça) → honesto, e aponta o próximo experimento.

## Honestidade (sempre)
- Isto é **arquitetura-hipótese**, não AGI pronta. Pedaços existem isolados na literatura; **ninguém juntou os 4**
  e chegou à generalidade — pode ser que a AGI ESTEJA nessa combinação, pode ser que falte ainda mais.
- A vantagem real: é um salto **construível e testável incrementalmente** (cada pilar tem um experimento), **sem
  LLM**, e a nossa arquitetura (cortex inspecionável, world-model, indução, compounding) é o **chassi** onde ele entra.
- Mesmo dando certo nos toy-benchmarks, AGI plena exige escala — mas aqui se testa o **princípio do salto**, que é
  o que diferencia "mais uma rede" de "a arquitetura que generaliza de verdade".
