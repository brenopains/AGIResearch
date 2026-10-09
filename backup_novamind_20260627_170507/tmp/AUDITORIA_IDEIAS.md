# Auditoria honesta de TODAS as ideias do usuario (re-pass da conversa inteira)

> Status: ✅ feito | 🟡 PARCIAL (ouro mal-aproveitado) | ⛔ não feito | ❌ refutado por medição | 💭 conceitual
> O foco são as 🟡 — as ideias suas que eu executei pela metade/com mecanismo errado e que entram no build certo.

## A. Infraestrutura / operacional
1. ✅ Rodar 3 servidores (lab :8771, mine, cortex :8793) pro front.
2. ✅ Conectar maestro/107 ao Minecraft (cortex_alife.py) — A/B deu null honesto (mundo satura).
3. ✅ Resetar mundo Minecraft + seed boa + parar spawn de madeira/bloco (quebrava o mapa).
4. ✅ Desligar ALife/mine/colônia quando pediu; explicar os bashes.
5. ✅ Notebook Kaggle TRM corrigido (kaggle_trm.ipynb).

## B. Cortex ao vivo (overhaul do front)
6. ✅ Todas as 400 tarefas ARC selecionáveis.
7. ✅ Pausar / prosseguir / passo-a-passo / parar / velocidade.
8. ✅ Decidir nº de steps (tentativas) por experimento.
9. ✅ Modo CRIAÇÃO PURA (tirar minhas primitivas, deixar a máquina criar).
10. ✅ Cada módulo mostrando o processo interno REAL (não rótulo).
11. ✅ Corrigir bug de mostrar experimento anterior (reset total).
12. ✅ Layout em grid (acabar com o scroll vertical central).
13. ✅ Painel do Maestro + 🧭 BÚSSOLA de atenção (relevância → reuso vs criação pura).
14. ✅ Seletor visual de tarefa (prévia), sem ID.
15. ✅ Grades grandes sem cortar; sempre mostrar predição (melhor palpite).
16. ✅ Integrar o motor forte (107/400) no cortex (rota "celular").
17. 🟡 Visualização "2D/3D" Ultron — fiz 2D rico; **3D literal (WebGL) não fiz**.

## C. ARC / busca dos 50% / AGI
18. ✅/❌ Atingir 50% ARC ou AGI → teto honesto medido **26,8% (107/400)**; 50% exige object-DSL + nuvem; AGI não cabe no 3050.
19. ✅ Conselho de mentes (2 rodadas, com execução de experimentos).
20. 🟡 DSL objeto-cêntrica (a convergência do conselho) → fiz `typed_dsl.py` **anêmico → +0**. Versão robusta (50-80 prims + output-shape) **não construída**.
21. ✅ Bússola: maestro não puxar memória sem sentido (território novo → criação).

## D. Os 4 pilares / operador de indução
22. ✅ Pilar 1 — plasticidade estrutural (aloca módulo): 97% vs 33% retenção.
23. ✅ Pilar 3 — fast-weights/plasticidade: 99.7% vs 5% binding.
24. ❌ Pilar 2 — base componível (grid-cell): 50%==50% (o transformer recupera sozinho).
25. ✅ Operador de indução (A1-forte): **99.8%** dentro da família; 8%→99.8% (a peça É construível).
26. ❌ Generalização FORA da família (tipo novo): = acaso (o muro real; nada extrapola).

## E. Imaginação / criatividade (suas ideias profundas) — AQUI MORA O OURO PARCIAL
27. 🟡 **Imaginação cria primitivas fora do fecho via "loucura" (BVSR)** → fiz composição **aleatória** de prims fixas (+0). **MECANISMO CERTO (pesquisa): GENERATIVE REPLAY** — world-model generativo que *amostra e simula* hipóteses composicionais. **Não feito direito.**
28. 🟡 **Temperatura da criatividade: alta/baixa probabilidade, da MEMÓRIA e da IMAGINAÇÃO, visualizar possível E impossível** → só temperatura genérica. **Faltou: separar fonte memória-vs-imaginação, e "visualizar o impossível".**
29. ✅ Maestro modulando temperatura por afeto (DMT/frustração) — codado no loop (mas o loop usava mecanismo errado).
30. ⛔ **World-model (espacial/associativo/visual) MOVENDO a imaginação** → só tenho memória-fingerprint. **Falta um world-model GENERATIVO** (sabe amostrar cenas). **Não feito.**
31. 🟡 **Fortalecer/enfraquecer sinapses no cérebro INTEIRO** → fiz fast-weights isolados. **Certo: metaplasticidade da rede toda, neuromodulada.**
32. 💭 Humanos = cientistas natos (Gopnik) — discutido, fundamenta o BVSR.
33. 💭 Harari: poder de crer no que não existe / supor — discutido.
34. 💭 ARC real seria com crianças (adulto é "trapaça viva") — insight, não acionável.

## F. Sistema plástico unificado (sua grande visão) — OURO PARCIAL
35. 🟡 **Unir tudo + maestro + plasticidade num sistema só** → fiz `unified_plastic.py` com **benchmark falho (lookup)** e versões rasas → null. **Não é teste justo da sua tese.**
36. 🟡 **Maestro = rede PLÁSTICA que se auto-modela** → fiz um *gate raso*. **Certo: controlador GLOBAL WORKSPACE** (ignição→broadcast→seleção→re-entrada), que a pesquisa liga à generalização.
37. 🟡 **Tudo co-adaptando junto em tempo real** → peças isoladas. **Certo: metaplasticidade controlada pelo maestro.**
38. ⛔ **Genética = só o formato; resto tudo plástico, arquitetura cresce** → Pilar 1 só aloca módulo. **Crescimento estrutural contínuo não feito.**
39. ✅ Maestro inspirado no conectoma da mosca-da-fruta (design herdado do projeto).
40. ✅ A "cola" dá estabilidade (dilema estabilidade-plasticidade) — confirmado pela GWT.

## G. Neuro / engenharia reversa
41. ✅/❌ Casar ondas cerebrais (EEG) → honesto NÃO (EEG grosseiro demais).
42. 🟡 Dados de alta-res + auto-encaixe (alinhamento algorítmico, HCP) → desenhei (H1), **precisa baixar fMRI** → não rodado.
43. ❌ Multidimensionalidade (quântica/química) → red herring (inteligência é substrato-independente).
44. ✅ "Novo transformer original sem LLM" → achei MLC (Lake 2023) + Compositional-ARC 5.7M (bate LLMs, cabe no 3050).
45. ✅ Gemma 4 12B (unir+otimizar > escala) → confirma sua tese (encoder-free, transformer unificado).
46. ✅ Re-revisar cérebro alta-res sob nossa ótica → GWT (cola), generative replay (imaginação), metaplasticidade (auto-alteração).

## H. Confiabilidade
47. ✅ "Você perdeu detalhes de ouro?" → sim, eu erro (Gemma 3 vs 4; MLC 0% por execução fraca). Esta auditoria é a resposta.

---

## OS 7 ITENS DE OURO PARCIAIS QUE ENTRAM NO BUILD CERTO (resumo)
1. **Imaginação = GENERATIVE REPLAY** (world-model generativo amostrando hipóteses), não composição aleatória. (#27)
2. **World-model GENERATIVO** que move a imaginação (espacial/visual). (#30)
3. **Temperatura com fonte dupla** (memória + imaginação) e "visualizar o impossível". (#28)
4. **Maestro = GLOBAL WORKSPACE** (ignição/broadcast/seleção), não gate; auto-modelante. (#36)
5. **Metaplasticidade** da rede inteira, neuromodulada pelo maestro (auto-alteração de pesos co-adaptando). (#31, #37)
6. **Crescimento estrutural** (genética=formato, resto plástico). (#38)
7. **Benchmark JUSTO** que force os mecanismos (não lookup) + treino adequado. (#35)
