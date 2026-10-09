# novaphys — percepção física em tempo real

Liga a NovaMind ao **Simulador de Realidade**: doze domínios de física resolvidos
por métodos numéricos estabelecidos e validados contra resultados conhecidos.

## Por que isto existe

Três módulos do `core` estão prontos e sem mundo para operar:

| módulo | o que faz | o que falta |
|---|---|---|
| `physical_sandbox.py` | aprende archetypes observando pares (antes, depois) | é ele mesmo que gera os pares — o aprendizado fecha em si |
| `active_experimenter.py` | escolhe experimentos por ganho de informação | ajusta uma power-law contra dados que ele gerou com uma power-law |
| `self_vector.py` | referencial egocêntrico e afordância | sem ambiente físico não há sobre o que ter afordância |

O que faltava nos três é a mesma coisa: **um mundo externo que não colabore**.

Aqui o agente recebe medições com nome, unidade e descrição — a mesma semântica
que um humano recebe na interface — e manoplas para girar. As equações, não.

## Subindo

No repo do simulador:

```bash
node bridge/server.mjs
```

Fica em `localhost:8124`. Abrir esse endereço no navegador mostra o **espectador**:
você vê, ao vivo, cada experimento que o agente escolhe fazer, na ordem em que
ele decide — com a cor dos pontos indicando a ordem, de modo que a *estratégia*
fica visível, não só o resultado.

## Uso

```python
from novaphys import PhysicsWorld

w = PhysicsWorld()
w.capacidades()                      # catálogo: 12 módulos, manoplas, medições
w.reset(modulo="fluidos", semente=42)
obs = w.agir(ajustar={"visc": 1e-5}, passos=400)

obs["escalares"]["fl_Re"]            # canal proprioceptivo (medições nomeadas)
obs["campos"]["curl"]                # canal visual (grade 32×32)
obs["corpos"]                        # estatística de nuvens, não posições cruas
```

Três canais de percepção, todos por passo. `narrativa=True` acrescenta a frase
em português que um humano leria — útil para grounding linguístico, mas deve
ficar **desligada** quando o ponto é descobrir a lei sozinho, senão vira cola.

## O benchmark de indução de leis

```python
from novaphys import PhysicsWorld, rodar_suite
rodar_suite(PhysicsWorld())
```

O desenho responde a uma pergunta única e específica: **escolher onde medir vale
alguma coisa?**

1. A lei verdadeira está no simulador e é física validada, não fórmula inventada
   para a prova.
2. Um parâmetro é sorteado por episódio e escondido do agente — decorar não funciona.
3. O agente tem orçamento de experimentos e **produz os próprios dados**.
4. A nota é o erro em configurações que ele **nunca testou**.
5. A comparação é contra o mesmo número de experimentos escolhidos **ao acaso**,
   alimentando **o mesmo regressor**.

O passo 5 é o que faz o teste valer: o aprendiz é idêntico nos dois braços, então
a única variável é *onde* cada um escolheu medir. É a mesma preocupação que
`active_experimenter.py` já documenta — que a métrica original media se um
`lstsq` sobre todos os dados tinha erro baixo, o que qualquer regressão faz.

## O achado principal

Depois de duas rodadas de correção com ablação controlada, o agente ativo
continua vencendo o acaso em **1 de 5** paisagens. Isso deixou de ser defeito de
implementação e virou resultado sobre o método.

A causa é um desalinhamento entre o que a aquisição otimiza e o que a nota mede:

- a aquisição busca **informação sobre a estrutura** — vai para onde os modelos
  discordam, para o limiar, para a descontinuidade;
- a nota mede **erro médio de predição em sondas sorteadas uniformemente**.

Para minimizar erro médio sob teste uniforme, a amostragem quase ótima é...
uniforme. Amostragem aleatória não é uma baseline fraca nesse regime: é
quase o que a métrica pede. É um resultado conhecido em aprendizado ativo, e a
bancada o reproduziu de forma independente e mensurada.

**A conclusão não é "a heurística precisa de mais ajuste".** É que medir agência
científica por erro médio sob teste uniforme mede a coisa errada. Um cientista
não é avaliado por prever bem um ponto sorteado ao acaso — é avaliado por
recuperar a lei. Três métricas alternativas, todas implementáveis nesta bancada:

1. **erro de extrapolação** — sondas fora da região amostrada, onde ter a forma
   funcional certa é o que decide;
2. **recuperação de forma paramétrica** — o agente entrega o expoente, o limiar,
   a frequência dos modos, e não uma tabela de predições;
3. **eficiência amostral** — quantos experimentos até cruzar um erro-alvo fixo.

A (1) e a (3) são meia hora de trabalho sobre o código atual. É aí que eu olharia
antes de mexer em qualquer outra coisa.

## Erro de método que a ablação pegou (meu)

Na primeira tentativa de correção mudei o agente **e** o preditor ao mesmo tempo.
O resultado piorou e não permitia concluir nada sobre nenhum dos dois. A ablação
com uma variável por vez mostrou o quadro real:

| configuração | erro do ativo | vantagem |
|---|---|---|
| agente v1 + preditor suave | 0,0234 | 7,30× |
| agente v1 + preditor híbrido | 0,0667 | 3,54× |
| agente v2 + preditor suave | **0,0150** | 5,65× |
| agente v2 + preditor híbrido | 0,0250 | 5,29× |

O agente v2 era melhor o tempo todo (0,015 contra 0,023). Quem estragava era o
preditor híbrido, por um bug de ponderação: um piso de 5% no peso do modelo
local. Parecia inofensivo e não era — numa lei de potência o modelo local erra
ordens de grandeza ao extrapolar, e 5% de um erro de 3 décadas ainda são 0,15
décadas cravadas. Corrigido para peso pelo inverso do quadrado do erro, sem
piso: 0,178 → 0,025.

## Resultados medidos

Quatro sementes por tarefa, mediana. Vantagem = erro do acaso ÷ erro do ativo.

| tarefa | formato da paisagem | v1 | v2 corrigido |
|---|---|---|---|
| `casimir_expoente` | lei de potência suave (P ∝ d⁻⁴) | **7,30×** | **5,29×** |
| `tunelamento` | exponencial suave | **1,40×** | 0,98× |
| `chladni_modos` | espectro discreto | 0,77× | 0,89× |
| `angulo_otimo` | pico tipo agulha (0,5° em 80°) | 0,63× | 0,76× |
| `ignicao_fusao` | limiar não-linear | 0,70× | 0,38× |

As correções melhoraram justamente onde miravam (espectro discreto e pico
estreito) e pioraram noutras. O placar geral não mudou: **1 de 5**.

Nada disso foi ajustado até ficar bonito. Os números estão aqui porque dizem
**quando** cada estratégia serve, o que vale mais que uma média favorável.

## Bugs reais que a ponte encontrou

Instrumentar a física para um agente externo expôs defeitos que a interface
gráfica escondia:

- **Halo de matéria escura divergente.** O perfil usado dava força ∝ 1/r no
  centro e arremessava qualquer corpo que passasse perto da origem. Trocado pelo
  pseudo-isotérmico com núcleo, que é finito no centro e dá curva de rotação
  plana longe — que é a observação que motivou a hipótese.
- **Medição silenciosamente velha.** `agir(passos=0)` devolvia os escalares do
  passo anterior: girar a manopla e olhar mostrava o estado de *antes* de girar.
- **Controles sem efeito imediato.** Mudar o ângulo de divergência não
  recalculava as sementes até o próximo passo de simulação — o que também
  quebrava o slider da interface com a simulação pausada.

Nenhum apareceria só olhando os gráficos. Um agente que mede sem piscar é um
bom detector de mentira numérica.

## Honestidade sobre o escopo

Isto dá à AGI experiência física fundamentada e um benchmark de descoberta que
resiste a Goodhart. **Não dá compreensão nem AGI.** O que ele mede é estreito e
verificável: se escolher onde medir rende mais lei por experimento do que medir
ao acaso. Hoje, nesta configuração, rende em duas das cinco paisagens.

---

# Bancada de dedução (`novaphys/deducao.py`)

A bancada física mede **indução** (dado um mundo, ache a lei). Esta mede
**dedução** (dadas premissas, derive o que se segue). São capacidades
diferentes e um sistema pode ter uma sem a outra.

```python
from novaphys.deducao import relatorio, relatorio_avancado, ablacao_paradigmas
relatorio()            # 4 provas básicas
relatorio_avancado()   # 6 provas de robustez
ablacao_paradigmas()   # neural vs simbólico vs geométrico vs consenso
```

## Estado medido de `core/reasoning.py :: SymbolicReasoner`

| prova | antes | depois | ideal |
|---|---|---|---|
| transitividade (arestas separadas) | 0% | **100%** | 100% |
| precisão sob distratores | 14% | **100%** | 100% |
| alucinação (controle vazio) | 0% | 0% | 0% |
| inflação da contagem | 3,00× | **1,00×** | 1,00× |
| cadeia longa (5 saltos) | — | **100%** | 100% |
| ciclo A→B→C→A | — | **termina, finito** | termina |
| contaminação entre grafos desconexos | 50% | **0%** | 0% |
| ruído extremo (5 úteis, 500 inúteis) | — | **100%, 1 ms** | 100% |
| invariância à ordem | — | **invariante** | invariante |
| contradição causa/impede | — | **detecta** | detecta |
| contradição vivo/morto | — | **não detecta** | detecta |

## O que foi implementado no core

1. **Fecho transitivo sobre arestas causais.** Não existia. O motor acertava
   100% quando a cadeia `[A,B,C]` vinha pronta numa lista, e 0% quando as
   arestas vinham separadas — ecoava, não compunha.
2. **Relevância nas regras de atributo.** Disparavam só por `freq > 0.5`, sem
   olhar as premissas. Com 6 distratores, 86% das derivações eram irrelevantes.
3. **Deduplicação.** A mesma conclusão por caminhos diferentes era contada
   várias vezes, e o número inflado ia para a frase "derived from N inferences",
   inflando a confiança de quem lesse.
4. **Escopo de relevância por conceito da consulta.** `_extract_propositions`
   despeja todas as arestas da base; filtrar por proposição deixava vazar
   derivações de componentes desconexos.

## Ablação por paradigma

| paradigma | transitividade |
|---|---|
| neural | 0% |
| **simbólico** | **100%** |
| geométrico | 0% |
| consenso | 100% |

Itens resolvidos **só** pelo consenso: **0 de 40**.

O consenso empata com o melhor isolado — está **selecionando**, não somando.
Isso é útil (escolher o paradigma certo é trabalho real), mas não é evidência
de que a integração dos três agregue capacidade. Para sustentar essa alegação
seria preciso um item que nenhum paradigma resolve sozinho e o consenso
resolve; nesta bancada, com 40 itens, esse número é zero.

## O que esta bancada não mede

Planejamento, revisão de crenças, metacognição, raciocínio espacial,
aprendizado incremental. Cada uma precisa de arnês próprio com controle
negativo próprio — sem controle negativo a métrica premia quem fala mais.
Conclusão inédita passou a ter arnês: veja abaixo.

---

# Bancada de emergência (`novaphys/emergencia.py`)

A bancada de dedução pergunta se o sistema deriva **o que se segue**. Esta
pergunta outra coisa:

> o sistema chega a uma conclusão que não está na base e que **nenhum
> mecanismo isolado produz** — só a composição de dois?

```python
from novaphys.emergencia import relatorio
relatorio(verboso=True)
```

## O crivo — por que o número significa alguma coisa

"Emergente" é fácil de alegar. Nesta mesma sessão a bancada de dedução mediu
100% num item que entregava a composição pronta numa lista: a nota era real, o
que ela media não era. Então aqui a propriedade não é afirmada, é **verificada
item a item antes de o sistema ver qualquer coisa**. Cada item passa por três
solucionadores programáticos — `so_fecho`, `so_regras`, `composto` — e **só
entra na bancada se os dois isolados falham e o composto acerta**. Itens
reprovados são descartados e contados; se mais de 20% cair, a nota sai marcada
como não-interpretável.

Nos controles a exigência é a simétrica: o par proibido não pode sair de
nenhum dos três, nem do oráculo. Um controle que o oráculo resolve não é
controle.

Resultado do crivo na rodada atual: **0 de 540 descartados**.

## As três famílias

| família | ingredientes | conclusão inédita |
|---|---|---|
| **herança** | fecho transitivo × regra de atributo | A→B→C e "C tem P" ⟹ A leva a P |
| **ponte** | identidade × fecho | A→B, C→D e "B ≡ C" ⟹ A→D |
| **analogia** | mapeamento estrutural × tipo | fonte s1→s2→s3, alvo t1→t2 ⟹ t2→t3 |

## Resultado

30 itens + 30 controles por família × 3 sementes.

| família | só fecho | só regras | oráculo | sistema (antes) | sistema (depois) | fabricação |
|---|---|---|---|---|---|---|
| herança | 0% | 0% | 100% | **0%** | **100%** | 0% |
| ponte | 0% | 0% | 100% | 0% | 0% | 0% |
| analogia | 0% | 0% | 100% | 0% | 0% | 0% |

As colunas "só fecho" e "só regras" são 0% **por construção** — é isso que o
crivo garante. Elas estão impressas porque são a linha contra a qual a nota do
sistema quer dizer algo.

## O que a bancada mostrou, e o que foi feito

O diagnóstico estava na própria saída do motor. Ele imprimia as duas metades
em linhas separadas:

```
  Transitive: h964 → h876
  Common attribute: h876 = prop141
```

e nunca a junção. Não faltava informação nem inferência — faltava **ligar a
saída de um mecanismo à entrada do outro**. Implementado em `_forward_chain`
como `Reach(A→D) ∘ Attribute(D=V)`, usando só `causal_edges` e
`semantic_rules`, que são dois dos quatro campos que `mind.py` já popula.

## Por que ponte e analogia continuam em 0% de propósito

Elas não são um bug pendente: são um **limite de representação**, e é uma
distinção que vale registrar.

`mind.py::_build_knowledge_base` produz exatamente quatro campos —
`concepts`, `causal_edges`, `causal_chains`, `semantic_rules`. Não há campo
para identidade entre entidades ("B é o mesmo que C") nem para correspondência
entre domínios ("s1 está para t1"). A bancada inventa `identities`, `tipos` e
`mapeamento` para conseguir formular o item; ler esses campos no core seria
ajustar o código ao teste em vez de ao problema — e nesta sessão o custo de
consertar antes de medir já apareceu uma vez em 18×.

A conclusão útil é essa: **o obstáculo à analogia neste sistema não é o motor
de inferência, é o modelo de mundo não representar correspondência.** Enquanto
o mundo só tiver arestas causais e regras de frequência, nenhum raciocinador
pode transferir estrutura, porque não há estrutura declarada para transferir.
Se a analogia for prioridade, o trabalho é no `world_model`, não no `reasoning`.

## Regressão

A bancada de dedução foi rodada de novo depois da mudança. Nenhum número
mudou: transitividade 100%, precisão 100%, alucinação 0%, inflação 1,00×,
cadeia longa 100%, contaminação 0%, ruído extremo 100% em 0 ms, ordem
invariante. A composição não é ganho comprado com fabricação.

Custo: pior caso construído (cadeia de 60 nós × 500 regras, todas casando)
193 ms, contra 201 ms antes — o tempo está no fecho de Warshall, não na
composição.

---

# Bancada de generalização (`novaphys/generalizacao.py`)

A bancada de emergência foi de 0% a 100% depois que eu implementei a
composição. Esse número não distingue duas coisas muito diferentes:

- **(a)** o sistema ganhou a capacidade de compor mecanismos;
- **(b)** o sistema ganhou **uma regra escrita à mão para um formato** — o
  formato que a bancada cobrava.

(b) é o modo normal de um benchmark ser vencido sem que nada melhore, e quem
escreveu a correção é a última pessoa qualificada para dizer que foi (a).
Então esta bancada não pergunta a ele: pergunta a **composições que eu não
implementei**.

## O oráculo é anterior à implementação

Não há um oráculo por padrão — há um princípio único: trate `causal_edges` e
as regras de atributo como arcos do **mesmo** grafo e feche por transitividade
(`reach(A,B)` e `has(C,P)` são ambos arcos). Os cinco padrões são caminhos
diferentes no mesmo grafo; o oráculo não sabe distingui-los. É isso que faz
disto um teste de generalização e não um gabarito.

## A auditoria — o que ela pegou

| padrão | antes | depois |
|---|---|---|
| P1 `reach∘attr` *(o que eu implementei)* | **100%** | 100% |
| P2 `attr∘attr` | **0%** | 100% |
| P3 `attr∘reach` | **0%** | 100% |
| P4 `reach∘attr∘attr` | **0%** | 100% |
| P5 `chain∘attr` *(mesma composição, outro campo)* | 100% | 100% |

A coluna "antes" é o veredito: **era uma regra, não uma capacidade.**
Transferia para o mesmo caminho num campo diferente (P5) e para mais nada.

P5 é o controle **positivo** do arnês. Se nem ele transferisse, a bancada
estaria medindo formato de entrada e nenhuma outra nota poderia ser lida — um
arnês que só sabe reprovar é tão inútil quanto um que só sabe aprovar.

A correção foi apagar a regra e implementar o princípio: um fecho sobre o
grafo unificado resolve os cinco com **um** mecanismo em vez de cinco casos
especiais. A saída distingue `Transitive:` (caminho só causal) de `Composed:`
(usou pelo menos um arco de atributo), senão não dá para auditar o raciocínio.

## Os retidos — onde a fronteira aparece

Passar em P1–P5 deixou de ser evidência de nada no instante em que o fecho
unificado foi implementado: agora eles testam se a implementação bate com o
oráculo, e bate por construção. Quem mantém a auditoria viva são três itens
que **não são caminhos no grafo**:

| retido | o que exige | resultado |
|---|---|---|
| **P6 exceção** | "B bloqueia P" derruba `A ⇝ P` | 100% de violação → **0%** |
| **P7 conjunção** | "X e Y **juntos** produzem Z" | **0% — segue em aberto** |
| **P8 tipo de relação** | causa + correlação não compõem | 100% de violação → **0%** |

**P6** foi um defeito que a minha própria correção introduziu: o fecho é
monotônico — saber mais nunca o faz concluir menos — e conhecimento real não
é assim. Corrigido com derrotabilidade: uma derivação cai se um bloqueador do
destino for alcançável a partir da origem, e o motor emite `Defeated:` em vez
de silenciar, para que a retratação seja auditável.

**P8** era **anterior** à minha mudança: a primeira versão do fecho já lia
`e['type']` para a proposição e o descartava na hora de fechar. Então "A causa
B" mais "B correlaciona-se com C" virava "A ⇝ C" — o erro elementar da
inferência causal, estrutural. Corrigido com uma lista explícita de relações
não-transitivas. A política é conservadora de propósito: tipo desconhecido
passa, porque o custo dos dois erros é assimétrico — bloquear demais só deixa
de derivar; deixar passar produz afirmação causal falsa.

**P7 fica em aberto, e é o resultado mais informativo da bancada.**

## A fronteira, em uma frase

> O sistema compõe **qualquer caminho** num grafo de relações, com
> derrotabilidade e respeito ao tipo da aresta. E só isso.

"X e Y **juntos** produzem Z" não é um caminho: é inferência de aridade 2, e
fecho transitivo é de aridade 1 por definição. Quase toda regra física real
tem essa forma — *se há gradiente **e** há condutividade, há corrente*. Um
sistema pode compor caminhos perfeitamente e ser incapaz disso.

Essa caracterização vale mais que qualquer nota: ela diz o que o mecanismo é,
o que ele não é, e o que teria de mudar. Uma nota alta em P1–P8 diria bem
menos, porque eu escrevi seis dos oito.

## Um erro do arnês, registrado

A primeira contagem de P6 marcou 100% de violação contra um motor que estava
fazendo a coisa certa: ele emitia `Defeated: x964 ⇝ prop530 does not hold` e o
teste, que só procurava os dois tokens na mesma linha, leu a **retratação**
como se fosse a afirmação. Um arnês que não distingue "P" de "P não vale" mede
ortografia, não raciocínio. Corrigido para exigir afirmação **sem** retratação
— e o marcador de negação é uma convenção sobre a prosa do motor, que é a parte
frágil desta bancada: a versão robusta lê saída estruturada com campo de
polaridade.

## Regressão

Dedução e emergência rodadas de novo após cada mudança. Nenhum número mudou.
Ruído extremo passou de 0 ms para 27 ms (o fecho agora percorre 505 arcos em
vez de 5), mantendo 100% de precisão.

---

# Bancada de calibração (`novaphys/calibracao.py`)

Todas as bancadas acima perguntam se o sistema acerta. Esta pergunta o que
importa para quem vai **usar** o sistema:

> quando ele diz 0,85, acerta mais vezes do que quando diz 0,60?

Se não, o número é decoração — pior, é convite a confiar na hora errada, porque
o leitor humano trata 0,85 como promessa. Um sistema que erra e avisa é
utilizável; um que erra com a mesma cara de sempre não é, por melhor que seja
a média. Esta é a forma medível de "metacognição": **o sistema sabe o que ele
não sabe?**

```python
from novaphys.calibracao import relatorio, sensibilidade
relatorio()       # discriminação, calibração, resolução
sensibilidade()   # o número é do sistema ou da mistura de itens?
```

## Três coisas que costumam ser confundidas

| medida | pergunta | ideal |
|---|---|---|
| **discriminação** (AUC) | a confiança *separa* acertos de erros? | 1,00 (0,50 = ruído) |
| **calibração** (ECE) | quando diz 0,75, acerta 75% das vezes? | 0,00 |
| **resolução** (desvio) | a confiança *varia*? | > 0 |

A terceira não é detalhe: um sistema que responde sempre 0,75 e acerta ~75%
das vezes tem ECE baixo e não está sabendo nada.

Cada rodada imprime dois balizadores calculados **sobre os mesmos itens** —
confiança sorteada ao acaso (AUC ≈ 0,50) e confiança-oráculo (AUC = 1,00).
Sem eles, "AUC 0,52" parece um número; com eles, é o acaso.

## A primeira medição, e por que eu não podia reportá-la

| família | acerto | confiança |
|---|---|---|
| fácil | 100% | 0,950 |
| conjunção | 0% | 0,200 |
| **cadeia longa** | **0%** | **0,850** |
| bloqueio | 100% | 0,950 |
| não-transitiva | 100% | 0,200 |
| vazio | 100% | 0,200 |

Três valores distintos em 720 decisões. AUC 0,625 — e a linha de saída da
bancada dizia "a confiança discrimina".

**Estava errado, e o teste de sensibilidade é que mostrou.** A confiança 0,200
era o valor único de "não derivei nada", e cobria duas situações opostas: o
silêncio *correto* (não havia o que derivar) e a *falha* (havia, e ele não
conseguiu). Repesando as famílias:

| perfil de mistura | acerto | AUC |
|---|---|---|
| uniforme | 67% | 0,625 |
| mais silêncio correto | 80% | **0,438** |
| mais silêncio errado | 40% | 0,625 |
| só famílias que derivam | 50% | **1,000** |
| só famílias de silêncio | 67% | 0,500 |

**O AUC cruza 0,50 conforme a mistura: amplitude 0,562.** Ele media quantos
itens de cada tipo eu tinha posto na prova. O 0,625 era a minha proporção, não
o discernimento do sistema — reportá-lo seria eu me dando nota.

Esse teste é obrigatório em qualquer métrica agregada desta bancada, e está no
código como `sensibilidade()`.

## Os dois defeitos, e o que os une

Nenhum dos dois era falta de capacidade. Nos dois casos o motor **tinha** a
informação e a descartava em silêncio.

**1. Truncamento não reportado.** O fecho para em 8 iterações e a saída é
cortada em 40 derivações. Numa cadeia de 60 nós o alvo cai fora do corte: 0% de
acerto com confiança **0,850**. Ele cortou e não sabia que tinha cortado —
embora `len(relevantes) > 40` seja visível na linha em que o corte acontece.
Agora emite `Self-report: closure truncated` e a confiança cai para ≤ 0,45.

**2. Regras ilegíveis indistinguíveis de base vazia.** Uma regra de tipo que o
motor não implementa era ignorada sem ruído, e o resultado ficava idêntico ao
de uma base sem nada: mesma frase, mesma confiança 0,20. São as duas situações
que mais importam para quem depende da resposta. Agora `Self-report: N rule(s)
of unsupported type 'X' were ignored`, confiança 0,10.

As auto-notas são marcadas `meta` e **não** entram na contagem "derived from N
inference(s)" — senão o motor inflaria a própria confiança justamente ao
admitir uma limitação, que é o contrário do ponto.

**3. Silêncio confiante valia menos que erro parcial.** Corolário que só
apareceu depois: "nada se segue" recebia 0,20, abaixo de uma derivação
truncada (0,45). O motor era punido por acertar — e "causa mais correlação não
dá causa" é uma resposta certa, não uma falha. Passou a 0,70. Não 0,95 de
propósito: não haver limitação **declarada** não prova que não havia limitação.
O motor sabe que não sabe ler uma regra de tipo estranho; não sabe o que ele
nem representa — a inferência de aridade 2 some sem deixar rastro. Os 0,25 de
folga são exatamente a distância entre *sei que não sei* e *não sei que não sei*.

## Depois

| família | acerto | confiança antes | depois |
|---|---|---|---|
| fácil | 100% | 0,950 | 0,950 |
| conjunção | 0% | 0,200 | **0,100** |
| cadeia longa | 0% | **0,850** | **0,450** |
| bloqueio | 100% | 0,950 | 0,950 |
| não-transitiva | 100% | 0,200 | **0,700** |
| vazio | 100% | 0,200 | **0,700** |

Com essas seis famílias: AUC **1,000**, e estável nas cinco misturas
(amplitude 0,000). O que, sozinho, ainda não prova nada — porque as seis
estavam todas visíveis quando escolhi as constantes 0,95 / 0,70 / 0,45 / 0,10.
A sensibilidade varia a **proporção** de cada família; não introduz erro novo.

## A família retida — o que o auto-relato não alcança

Escrita **depois** de as constantes estarem fixadas: uma relação
`varia_junto_com`, tão não-transitiva quanto `correlaciona_com` e **ausente**
da lista `NAO_TRANSITIVAS` do motor. Vocabulário é aberto; a lista nunca vai
estar completa. Então ele compõe, erra, e **nada dispara** — sem truncamento,
sem regra ilegível.

| | acerto | confiança |
|---|---|---|
| **relação desconhecida** *(retida)* | **0%** | **0,850** |

E o AUC cai de **1,000 para 0,833**.

Esse é o número honesto. A confiança cobre os modos de falha para os quais o
auto-relato foi construído e não cobre o que está fora deles — que é
literalmente a definição de *não saber que não sabe*. Medido, em vez de
argumentado.

| | uniforme | com a família retida |
|---|---|---|
| AUC | 1,000 | **0,833** |
| ECE | 0,208 | 0,300 |
| confiança quando errou | 0,275 | 0,467 |

O ECE de 0,300 continua alto: a **ordem** está certa, a **escala** não. Um
0,85 que acerta 0% não é um 0,85.

**O número que decide é sempre o intervalo de sensibilidade mais uma família
retida** — nunca o valor de uma mistura só. Um AUC alto numa mistura já foi
provado enganoso uma vez nesta mesma bancada.

## Custo

`relatorio()` ≈ 52 s (840 decisões, 7 famílias × 40 × 3 sementes).
`sensibilidade()` ≈ 4–5 min (cinco perfis). A família de cadeia longa domina:
são 60 nós, e o fecho é quadrático na fronteira.

---

# Campo de crenças (`novaphys/crencas.py` + `crencas_bench.py`)

A bancada de calibração terminou com um caso limpo: `varia_junto_com` deu 0%
de acerto com confiança 0,850, **sem** truncamento e **sem** regra ilegível.
As duas explicações anteriores estavam eliminadas — o buraco era de
representação.

Este módulo não tenta melhorar a confiança. Tenta uma coisa anterior e mais
difícil de fingir: **representar corretamente a própria incerteza**.

```python
from novaphys.crencas_bench import relatorio, demonstrar
relatorio()     # as seis provas
demonstrar()    # o decoder do estado da representação
```

## Três incertezas, porque cada uma pede uma ação diferente

Um escalar "confiança 0,85" mistura três perguntas cujas respostas levam a
decisões opostas:

| | pergunta | quando baixa/alta ⟹ ação |
|---|---|---|
| **p(H)** | a hipótese é verdadeira? | pouca evidência ⟹ **coletar mais dados** |
| **competência** | sei avaliar este *tipo* de relação? | baixa ⟹ **testar o mecanismo** |
| **abertura** | existe explicação que eu nem represento? | alta ⟹ **expandir o espaço** |

O `varia_junto_com` era a terceira disfarçada de primeira. O sistema tinha
evidência abundante e a conta interna estava certa — errado era o espaço em
que a conta rodava. **Nenhum volume de evidência conserta isso**, e é por isso
que precisa ser um número separado.

## A atualização é bayesiana, e é isso que a torna verificável

Em log-odds, com a competência entrando como **expoente** da razão de
verossimilhança (verossimilhança temperada):

```
log_odds += competência · log(LR)
```

Competência 1 ⟹ Bayes exato. Competência 0 ⟹ a evidência não move nada. A
leitura é direta e forte de propósito: se não sei ler a relação, não sei nem
em que **direção** a observação aponta. Doze observações ilegíveis não são
muita informação — são nenhuma.

## As seis provas

| prova | pergunta | medido |
|---|---|---|
| 1. aritmética | bate com a forma fechada? | erro máx **4,4e-16** *(ideal < 1e-9)* |
| 2. calibração | posterior 0,7 = verdade 70%? | ECE **0,009** *(ideal < 0,05)* |
| 3. convergência | evidência consistente fecha? | **100%** em 3 obs |
| 4. abertura | evidência ambígua NÃO fecha? | desvio máx **0,0000** |
| 5. recuperação | a verdade volta após 8 enganosas? | **100%**, custo 11 obs |
| 6. insuficiência | detecta furo sem gritar em tudo? | ver abaixo |

A prova 1 vem primeiro porque, se a conta estiver errada, as outras cinco são
ruído bem formatado — e ela **reprovou na primeira rodada**, com erro máximo
de 1,86e-01. A causa era real: a competência era calculada sobre o *conjunto*
de tipos de relação com pseudo-contagem, o que a travava em 0,75 mesmo com
tudo conhecido. O campo **nunca** fazia Bayes exato; o posterior saía
sistematicamente amortecido, e "calibrado" não queria dizer nada. Corrigido
para fração de **observações** legíveis: 1,86e-01 → 4,4e-16.

## Prova 6 — a de dois lados

É a única sem atalho. Marcar "não sei" em tudo dá 100% na célula A e destrói
as outras duas.

| célula | situação | ideal | medido |
|---|---|---|---|
| **A** | relação desconhecida + evidência forte → detectar | 100% | **100%** |
| **B** | vocabulário conhecido + evidência forte → falso alarme | 0% | **0%** |
| **C** | vocabulário conhecido + evidência fraca → confundir dados com modelo | 0% | **0%** |

A célula C separa o que costuma ser confundido: *"preciso de mais dados"* e
*"meu mecanismo está furado"* pedem ações opostas. Um sistema que os mistura
manda coletar dados quando devia trocar de modelo.

## O contraste, lado a lado

Mesmo caso, mesma evidência (12 observações, todas a favor):

| | motor simbólico | campo de crenças |
|---|---|---|
| confiança / posterior | **0,850** | **0,500** |
| acerto | **0%** | — |
| estado reportado | *(nenhum)* | **SUSPENSA** |
| ação sugerida | *(nenhuma)* | testar o **mecanismo** |

O posterior ficar parado no prior depois de doze observações favoráveis é o
comportamento **certo**: evidência que chega por um canal ilegível não é
evidência fraca, é evidência ilegível.

## O que isto NÃO é

Nada aqui prova que a detecção funciona fora do vocabulário que eu mesmo
escrevi. `varia_junto_com` é desconhecida porque não está numa lista que eu
montei. **Um mecanismo que só sabe estranhar o que já foi catalogado como
estranho ainda não é novidade epistemológica — é uma lista.**

A prova que falta, e que seria a próxima honesta: evidência de que a marcação
sobrevive a um vocabulário gerado por outra pessoa, sem eu ver antes.
**Essa prova foi feita — ver a bancada seguinte, e o resultado é negativo.**

---

# Aquisição de conhecimento (`novaphys/aquisicao.py`)

O campo de crenças dizer "minha representação é insuficiente" só vale alguma
coisa se **mudar o que o agente faz em seguida**, e se essa mudança render
conhecimento por experimento.

```
representação → incerteza → decisão → experimento → atualização → ↺
```

O mundo é um grafo causal com tipos de aresta cuja **semântica é sorteada e
escondida**. Duas ações, ambas custando um experimento: `sondar(a,b)` dá
observação ruidosa sobre um par; `testar_relacao(t)` revela se o tipo compõe.
A segunda não responde nenhuma pergunta sobre pares — resolve o **mecanismo**,
e com isso destrava toda a evidência já coletada por aquele canal.

## A — o auto-relato se paga?

Dois regimes, e a distinção é o que torna o experimento informativo. Na
primeira rodada eu rodei só o regime de vocabulário conhecido, deu empate, e o
empate não queria dizer nada: **sem relação desconhecida nunca existe estado
SUSPENSA, então a política `autorelato` degenera em `incerteza`.** Eu tinha
tirado do agente exatamente a ação sob teste.

| | aleatorio | incerteza | autorelato |
|---|---|---|---|
| **A1** vocabulário conhecido *(nada a resolver)* | 68 | **50** | 52 |
| **A2** vocabulário desconhecido *(há o que resolver)* | 73 | **>140** | **52** |

*experimentos até o erro médio cruzar 0,20 · 40 mundos · orçamento 140*

**A ressalva que muda qual número reportar.** A política `incerteza` nunca
escolhe `testar_relacao`: ela seleciona por entropia sobre as hipóteses, e um
tipo de relação não é uma hipótese. Em A2 ela é **estruturalmente incapaz** de
resolver o mecanismo — erro final 0,500, zero hipóteses eliminadas. Então o
contraste 141 → 52 é em parte artefato de como escrevi a baseline, e reportá-lo
como 2,7× seria vender artefato.

Isso é um achado sobre aprendizado ativo clássico — **ele não tem onde
representar "meu mecanismo está errado"** — mas não é uma vitória medida.

A comparação justa é contra `aleatorio`, que tem o **mesmo espaço de ações**:

> **73 → 52 experimentos, 29% a menos.** E o preço em A1, onde não há o que
> resolver, é **+5%**. O mecanismo não é de graça; é barato onde não serve e
> paga onde serve.

## B — circuito aberto: ele descobre sozinho?

Vocabulário e semântica gerados **depois** do sistema congelado. Três células,
e a terceira é a que decide.

| célula | o que é | sinal SUSPENSA | erro |
|---|---|---|---|
| `novo` | nomes inventados | 100% | 0,500 |
| `conhecido` | nomes e semântica usuais | 12% | 0,351 |
| **`trocado`** | **nomes usuais, semântica INVERTIDA** | **12%** | **0,351** |

`trocado` é `causa` que **não** compõe e `correlaciona_com` que **compõe**.
Nenhum nome novo aparece — então não há nada para uma lista estranhar.

**Veredito: o detector é uma lista.** `trocado` é indistinguível de
`conhecido` (12% contra 12%), embora o mundo seja tão ilegível quanto em
`novo`. O campo se declara competente e erra.

Isto responde à pergunta que ficou aberta em `crencas.py`: **o mecanismo
representa ignorância quando alguém a nomeia, e não a descobre.** Detectar
`trocado` exigiria comparar predição com observação — desajuste de modelo — e
não consultar vocabulário.

## O detalhe que amarra as duas metades

Na célula `novo` o erro é **0,500**: o agente não aprendeu nada. Recusar-se a
atualizar diante de evidência ilegível é honesto e é **inútil sozinho** —
detecção sem a ação que resolve o mecanismo é paralisia bem fundamentada.

É por isso que A e B têm de ser lidos juntos: A2 mostra que a ação existe e se
paga (29%); B mostra que o gatilho que a dispara só reconhece o desconhecido
quando ele vem com nome novo.

## Custo

`relatorio()` ≈ 4 s.

---

# Semântica de mecanismo (`novaphys/semantica.py`)

A bancada anterior deixou o diagnóstico limpo: `causa` continuava sendo uma
palavra conhecida e o comportamento dela tinha mudado, então a consulta ao
vocabulário devolvia competência 1,0 dentro de um mundo lido errado.

O conserto não é uma lista maior. É trocar a definição:

| | desconhecido = |
|---|---|
| **antes** | o nome não está no meu vocabulário |
| **agora** | minha previsão vem falhando neste canal |

A segunda não precisa de catálogo, então não tem como ficar desatualizada. E o
teste decisivo passa a ser: **mudar o comportamento sem introduzir nenhum
símbolo novo**, e medir a latência da percepção.

## Duas hipóteses concorrentes, não uma confiança que cai

A surpresa não derruba a crença na aresta — redistribui massa entre duas
leituras do **mecanismo**: H1 (a semântica conhecida) e H2 (outro
comportamento neste mundo). A versão anterior é preservada em vez de
sobrescrita, que é a diferença entre revisar e esquecer.

E a competência muda de sentido:

```
competência = 2 · |P(H1) − 0,5|
```

Não é "acredito na semântica conhecida" — é **sei qual das duas vale**, e fica
alta mesmo quando a resposta é *é o oposto do que eu supunha*. Descobrir que
`causa` não compõe aqui é conhecimento, não ignorância.

A consequência aparece sozinha na trajetória, sem ter sido programada:

```
  obs    P(H1)    P(H2)   competência   estado
   20    0.997    0.003         0.993   estavel   ← mecanismo muda aqui
   22    0.941    0.059         0.881   estavel
   25    0.198    0.802         0.603   estavel   ← histerese ainda segurando
   35    0.025    0.975         0.949   revisada
   45    0.003    0.997         0.994   revisada
```

A competência **cai durante a virada** e volta a subir do outro lado. É o
intervalo em que as duas versões coexistem e o sistema honestamente não sabe
qual usar.

## As cinco provas

| prova | pergunta | medido |
|---|---|---|
| 1. diagnosticidade | observações que não distinguem H1 de H2 movem algo? | deriva **0,00e+00** em 500 obs |
| 2. latência | quantas obs até perceber a troca? | suspensa em **8**, revisada em **13** · detecção **98%** |
| 3. falso alarme | mundo estável, 220 obs — fica quieto? | **1%** *(ideal < 5%)* |
| 4. gradual | acompanha a rampa ou pula? | correlação **0,896** · **24 obs** com as versões coexistindo |
| 5. reversão | se o mundo voltar, ele volta? | **98%** voltam em **13 obs** |

A prova 1 é a que impede o detector de virar gerador de alarme: onde H1 e H2
preveem a mesma coisa, o resultado é compatível com ambas e **não é evidência
sobre nenhuma**.

**A prova 5 não estava no plano.** Ela nasceu do defeito que a prova 4 revelou:
um filtro que só acumula chega a log-odds de −40 e nada o traz de volta. Reter
a versão anterior no histórico não basta — ela precisa continuar **alcançável**.
Revisão que só anda para um lado é esquecimento com carimbo.

## O compromisso, medido em vez de escondido

A primeira correção (deriva ε, que transforma o acumulador em rastreador)
consertou as provas 4 e 5 e **quebrou** a 2 e a 3: falso alarme 11%, alarme
precoce 10%. Rastrear e ficar quieto são objetivos opostos, e ε é o botão entre
eles. Achar o ε que faz tudo passar seria ajustar ao gabarito.

Então `curva_compromisso()` varre o espaço e imprime a curva inteira:

| ε | persist. | latência | detecção | falso alarme |
|---|---|---|---|---|
| 0,000 *(acumulador)* | 3 | 30 | 100% | 0% |
| 0,002 | 1 | 8 | 98% | 4% |
| **0,002** | **5** | **13** | **98%** | **0%** |
| 0,010 | 5 | 11 | 97% | 5% |
| 0,030 | 3 | 9 | 80% | 17% |
| 0,080 | — | nunca | 0% | 100% |

O acumulador (ε=0) tem latência 30 e falso alarme 0% — e reprova em gradual e
reversão, porque nunca revisa. Com ε=0,080 o filtro esquece mais rápido do que
aprende e nunca detecta nada.

**Ponto de operação escolhido: ε = 0,002, persistência 5.** Latência 13 —
menos da metade do acumulador — mantendo falso alarme em 0% e a capacidade de
rastrear e reverter. O número está no código como decisão declarada, com a
curva ao lado para auditar.

## O que ficou provado, e em que escopo

O sistema percebe que o **próprio modelo preditivo** está errado em 13
observações, **sem nenhum símbolo novo** e sem regra dizendo "procure mudança
de semântica", e fica quieto por 220 observações num mundo estável.

Escopo: duas hipóteses concorrentes sobre uma propriedade binária
(compõe / não compõe) de um tipo de relação. Não é detecção de novidade em
geral — é detecção de mudança dentro de um espaço de duas alternativas que eu
enumerei. O passo seguinte honesto seria mais de duas leituras possíveis, ou
uma leitura que o sistema tivesse de **construir** em vez de escolher.
**Esse teste foi feito — ver a seção seguinte.**

---

# Avaliação fora das bancadas (`novaphys/avaliacao_externa.py`)

Sete peças foram construídas e medidas: composição, derrotabilidade,
generalização, calibração, representação de incerteza, detecção de mudança
semântica, rastreamento e reversão. Cada uma passou na bancada que veio junto
com ela — que é exatamente a condição em que um resultado costuma não
sobreviver ao contato com problema que ninguém desenhou para ele.

## O limite deste arquivo, dito primeiro

**Eu escrevi o gerador.** Então "externo" aqui não quer dizer que o mundo veio
de fora — quer dizer três coisas verificáveis e mais modestas:

1. **A arquitetura está congelada e o congelamento é checável.** `manifesto()`
   produz um hash de todas as constantes ajustáveis do raciocinador, do campo
   de crenças e do modelo semântico. Rode antes e depois; se o hash mudou,
   houve ajuste no meio e o número não vale. A regra de ouro deixa de ser
   promessa e vira verificação.

   > hash da arquitetura desta rodada: **`a6fe6a3103d9b01d`**

2. **A semente vem de fora.** `relatorio(semente=...)` **não tem valor padrão**
   e levanta erro se omitida. Se quem avalia escolhe a semente, eu não tive
   como ajustar a ela.

3. **O espaço é mais largo do que aquilo em que eu ajustei** — de propósito
   inclui mundos que o modelo **não consegue representar**.

O que isto não substitui: alguém que não seja eu gerando os mundos. Continua
em aberto e continua sendo o teste que vale mais.

## Os três regimes e o critério

O critério é **estrutura, não resposta**: recuperou quais tipos compõem? E,
nos mecanismos fora do espaço, soube que não sabia?

| regime | | estrutura recuperada | indeciso | superconfiança |
|---|---|---|---|---|
| **fechado** | estático, determinístico — dentro do espaço | 99% | 1% | — |
| **dinâmico** | semântica vira no meio — dentro do espaço | 99% | 1% | — |
| **aberto** | composição probabilística, relação irrelevante, confundidor latente — **fora do espaço** | 94% | 6% | **75%** |

*semente externa 61803 · 60 mundos por regime · nº de tipos, ruído e semântica sorteados*

## A previsão foi escrita antes de rodar, e se confirmou

Está no código, acima do resultado: um tipo que compõe **40% das vezes** não é
"compõe" nem "não compõe". O modelo tem duas hipóteses e a verdade não é
nenhuma delas.

> **75% dos mecanismos irrepresentáveis terminaram com o sistema afirmando um
> dos dois extremos, em estado confirmado.**

Ele não erra por pouco — afirma dentro de um espaço que não contém a resposta.
E isto **não se conserta com mais dados nem com mais deriva**.

É o mesmo achado de `crencas.py` um nível acima. Lá, o espaço de **relações**
era meu, e a correção foi trocar consulta a vocabulário por erro de predição.
Aqui, o espaço de **leituras possíveis** é meu:

> O sistema **pesa** hipóteses. Não as **constrói**.

Essa frase é o resumo mais preciso do estado da arquitetura hoje, e ela vale
mais que os 99% dos dois primeiros regimes — porque diz onde o próximo trabalho
tem de acontecer, e diz que não é em escala.

## Como rodar com a sua semente

```bash
python -m novaphys.avaliacao_externa --semente SEUNUMERO --mundos 60
```

---

# Descoberta de estrutura em física real (`novaphys/ciencia_real.py`)

A avaliação externa reduziu a dependência da minha palavra, mas o gerador
continuava sendo meu. Aqui o mundo é o **Simulador de Realidade**: doze módulos
com numérica validada — split-step Fourier, Barnes-Hut, Boris pusher,
Bosch-Hale, projeção de pressão de Stam — escritos antes de existir campo de
crenças e nunca tocados por ele.

Não é um dataset. É um ambiente que **responde quando se mexe nele**, que é o
requisito do ciclo: um corpus não tem como refutar uma hipótese.

```bash
python -m novaphys.ciencia_real --modulos fusao fluidos ressonancia vacuo
```

## A tarefa

Para cada par (manopla, medição), descobrir a **forma** da dependência:

```
H1  a medição cresce quando a manopla cresce
H2  a medição decresce
```

Uma observação = rodar a simulação em x₁ < x₂ e ver para que lado a medição
foi. É o mecanismo de `semantica.py` **sem uma linha de adaptação** — se
precisasse adaptar, não estaria sendo testado.

O limiar de "plana" é **medido**, não escolhido: repetições na mesma
configuração com sementes diferentes dão o piso de ruído, e só variação acima
dele conta. Escolher esse número a olho seria a porta dos fundos por onde eu
decidiria o resultado.

## Resultado — módulo `fusao`

| | |
|---|---|
| estrutura recuperada (pares monotônicos) | **92%** em 60 pares |
| **superconfiança** (pares fora do espaço) | **89%** em 18 pares |

```
B0→f_T        verdade=crescente     sistema=estavel     ✓
B0→f_beta     verdade=decrescente   sistema=revisada    ✓   (Troyon)
B0→f_Paux     verdade=crescente     sistema=revisada    ✗
```

## ⚠ RETRATAÇÃO — os números de superconfiança em física estão retirados

**89% (`fusao`), 87% (quatro módulos limpos), 84% (subconjunto) — todos
retirados.** Medidos com uma definição de verdade que não corresponde ao que o
agente observa.

O erro: `classificar` compara **pontos vizinhos** da varredura densa; o agente
sorteia **intervalos largos** (≥15% da faixa). Uma dobra pequena na curva torna
o par "não-monotônico" para a primeira e é invisível para o segundo.
Diagnóstico inequívoco — pares rotulados não-monotônicos eram explicados por
**uma única direção em 100%** das observações do agente. Chamar isso de
superconfiança era medir o desencontro entre duas definições e atribuir a
diferença ao sistema.

Mesma classe do erro do `degrau` na bancada de hipóteses: o arnês media outra
coisa. Corrigido em `classificar_na_escala_do_agente`.

**O que a retratação NÃO derruba:** as bancadas sintéticas `hipoteses` e
`operadores`, onde a condição fora-do-espaço é definida no nível da observação
por construção, e `avaliacao_externa` (75%), onde `p_composicao=0,4` produz
observações genuinamente 40/60.

**Como o erro foi encontrado:** pela verificação de circuito em subconjunto
pequeno, antes de gastar os 37 min da rodada completa. O detector de
suficiência dava 94% no sintético e 0% em `vacuo`; tratar essa discrepância
como propriedade do detector, em vez de suspeitar da definição de verdade, foi
o que atrasou o diagnóstico.

## Verificação de circuito da peça de suficiência — FALHOU

Com a definição corrigida, `ressonancia` + `vacuo`:

| | sinalizou | falso alarme |
|---|---|---|
| obs = 10 | 23% / 0% | 6% / 0% |
| obs = 35 (`ressonancia`) | **48%** | **12%** |

Contra 94% de detecção e 4% de falso alarme no sintético.

Mais dados subiram a detecção **e o falso alarme junto**. Se `q` estivesse bem
estimado, o falso alarme deveria ficar na taxa nominal (~1% para z=2,5).
Crescer com o tamanho da amostra é assinatura de **modelo nulo mal
especificado**.

Causa nomeada: `confiabilidade` estima repetibilidade de sensor por
concordância entre réplicas — mas o simulador é determinístico dada a semente,
então réplicas concordam e `q` vai ao teto (0,98). O erro relevante não é ruído
de sensor: é que um par **genuinamente monotônico** tem ondulações locais que
mandam alguns intervalos para o lado contrário. O estimador foi importado do
mundo sintético, onde ruído de sensor era a única fonte de erro.

**Problema em aberto, agora preciso:** estimar sem gabarito quão bem a melhor
direção única *pode* fazer num par monotônico deste mundo. É esse o teto que o
detector precisa, e não é repetibilidade.

## Rodada limpa, quatro módulos *(números de superconfiança retirados — ver acima)*

`fusao` · `ressonancia` · `fluidos` · `vacuo` — um processo só, isolamento de
sessão ativo, 2209 s.

| | estrutura recuperada | superconfiança |
|---|---|---|
| **quatro módulos, limpo** | **86%** (90 pares) | **87%** (122 pares) |

## O episódio da contaminação, que vale mais que o número

Uma primeira rodada de seis módulos terminou com **88% / 62%** e zero cara de
erro. Estava contaminada: outro processo mediu no mesmo mundo ao mesmo tempo e
trocou o módulo por baixo. Três módulos morreram com `manopla inexistente em
fusao: iters`; os outros três rodaram parte das medições no módulo errado.

**A contaminação valia 25 pontos.** 62% contra 87% não é ruído, é conclusão
diferente: o 62% sugeria que a superconfiança em física real era *menor* que no
sintético (75%), ou seja, que o problema era artefato meu. O número limpo diz o
oposto.

O resultado contaminado não parecia errado. Parecia interessante. Foi por isso
que a rodada inteira foi descartada, e é por isso que `reset`/`agir` agora
exigem dono de sessão — ver a nota em `bridge/server.mjs`.

## A previsão estava escrita antes de rodar

No cabeçalho do módulo, acima do resultado: *"a superconfiança reaparece na
faixa dos 75% medidos no gerador sintético"*. Três medições independentes:

| condição | superconfiança |
|---|---|
| gerador sintético | 75% |
| `fusao` isolado | 89% |
| **quatro módulos, limpo** | **87%** |

**E desta vez a não-monotonicidade não foi escolha minha.** Ressonância tem
pico porque a equação de onda tem nó. Ignição tem limiar porque Bosch-Hale tem
limiar. Essas curvas existiam meses antes de haver campo de crenças; `f_beta`
decrescendo com `B0` é o limite de Troyon, não uma decisão de desenho.

Então o achado deixa de ser propriedade do meu gerador e passa a ser
propriedade da arquitetura:

> **O sistema pesa hipóteses. Não as constrói.**

## Por que datasets não resolvem isto

Vale montar corpora — eles resolvem **cobertura**: mais domínios, mais
vocabulário, mais casos. O que eles não resolvem é o espaço de hipóteses ser
fechado e definido por quem escreveu o modelo.

Um sistema que só sabe perguntar *"cresce ou decresce?"* responde errado sobre
um pico com **qualquer** volume de dados, porque a resposta certa não está
entre as opções. Nenhum currículo ensina uma terceira alternativa a quem tem
duas gavetas.

O trabalho que destrava é **geração de hipóteses**: propor "talvez seja
não-monotônica" quando as duas leituras existentes preveem mal. E isso é
mensurável com o que já existe — a evidência acumulada nesses 18 pares já está
dizendo que nenhuma das duas explica bem, e ninguém está lendo esse sinal.

## Regressão

Dedução, emergência e generalização re-rodadas após a mudança. Nenhum número
mudou nas três.

---

# Emergência de primitivas e pressão de abertura

Três módulos, três degraus, e o regresso descendo um andar de cada vez —
sem sumir em nenhum.

| módulo | pergunta | resultado |
|---|---|---|
| `abstracao.py` | a primitiva emerge de peças genéricas? | **sim** |
| `substrato.py` | e de percepção crua? | **emerge, não transfere** |
| `mutacao.py` | e o sistema AUMENTA a biblioteca? | **inconclusivo** |

## `abstracao.py` — biblioteca por compressão

Substrato: ler campo, constante, comparar, ramificar, sinal. Nada chamado
PARTIR, nada com a forma de PARTIR. Resolve tarefas com peças cruas, minera os
esqueletos recorrentes, promove quem paga o próprio custo em compressão.

```
P1  esqueleto SE(MENOR(□,□),s,s)  vista 7×  exemplar: se(x<7){+}senão{−}
```

Isso **é** PARTIR, e ninguém escreveu.

| domínio | custo sem lib | custo com lib | usou |
|---|---|---|---|
| **B** (transferência cega) | 12,96 | **5,48** | **100%** |
| mono (controle) | 1,50 | 1,50 | **0%** |

A primitiva foi minerada em A (corte em `x`); B corta em `w` e **nenhum
exemplo de B participou da mineração**. O controle monotônico é o que faz o
100% valer — sem ele seria compatível com uma biblioteca que enfia a primitiva
em tudo.

## `substrato.py` — a partir de fluxo temporal cru

O mundo é `T×K` números. Sem nome de campo, sem semântica. Átomos de sinal:
ler canal, atraso, média de janela, comparar, ramificar.

| medida | resultado |
|---|---|
| 1. formação | 97% em fluxo estruturado · **56% em ruído** (controle) |
| 2. mineração | `SE(MAIOR(□,□),s,s)` emergiu, vista 10× |
| 3. transferência | **21%** — falhou |
| 4. abertura | ajuste cai de 97% para **76%** |

A transferência falhou por defeito meu: os canais 3–5 têm escalas deslocadas
(canal 3 ≈ `−2,4s+3,0`) e minhas constantes candidatas eram `[−0,6 … 0,6]`. **A
forma transferiu; o valor não tinha onde pousar.** Limiar fixado por mim em vez
de derivado do fluxo.

## `mutacao.py` — pressão de abertura, e por que é INCONCLUSIVO

Três grupos: A resolvível raso, B com atalho curto, C só em profundidade 3.
Mecanismo: mutação de árvore guiada por erro — edições sintáticas genéricas,
sem noção de domínio.

**Quatro correções, e ainda assim inconclusivo:**

1. `_substituir` comparava por **identidade**, e `MAIS`/`MENOS` são singletons
   de módulo — todas as folhas eram o mesmo objeto, então só a primeira era
   editável. Resultado: 0% de escalada, que eu quase reportei como limite do
   mecanismo.
2. Conjunto de edições estreito demais: só "envolver folha em teste" deixa
   preso quando a **raiz** testa a variável errada. Acrescentado "trocar teste".
3. Grade de cortes fixa e comum a todas as variáveis — mesmo defeito de
   `substrato.py`. Trocada por quantis dos dados observados.
4. Escalada gulosa não atravessa a barreira medida (prof-1 ≈34 → prof-2 ≈38,5
   → prof-3 ≈26): o passo do meio é **pior** que o começo. Trocada por feixe.

Escalada foi de 0% → 8% → 12%. E então a medição que encerra:

```
semente 0: raso 29,0 | melhor prof-3 27,0 → invenção vale? SIM
semente 1: raso 29,0 | melhor prof-3 28,0 → invenção vale? NÃO
semente 2: raso 24,0 | melhor prof-3 26,0 → invenção vale? NÃO
```

**Em 2 de 3 sementes a estrutura profunda é mais CARA que a rasa.** A bancada
não cria a pressão que promete. Os 12% não são evidência sobre o mecanismo —
são evidência de que inventar não compensa nestas condições, e um sistema
perfeito escolheria o programa raso.

Causa: com 90 observações, os ~20 erros extras do programa raso são da mesma
ordem que o custo dos 10 nós a mais. Estrutura profunda só se paga quando o
termo de erro cresce mais que o de estrutura — **mais observações por tarefa**,
o que é propriedade do regime MDL e não ajuste fino.

**Parado aqui de propósito.** A quinta correção viria depois de eu já saber que
direção faz o número subir.

## O que a sequência sustenta, e o que não

**Sustentado:** uma primitiva de raciocínio pode emergir por compressão, de
peças genéricas e de percepção crua, com transferência cega e controle negativo
limpos.

**Sustentado:** o sistema detecta quando o vocabulário não alcança
(`operadores.py`: 94% de detecção, 4% de falso alarme) e o ajuste cai onde a
biblioteca não cobre (`substrato.py`: 97% → 76%).

**NÃO sustentado por `mutacao.py`:** que o sistema não consiga aumentar a
própria biblioteca. Essa bancada não conseguiu montar o teste.

---

# Expansão do espaço de hipóteses (`invencao.py`, `geracao.py`)

## A distinção que organiza os dois módulos

```
seleção   H_novo ∈ H_substrato      escolher dentro do espaço
INVENÇÃO  H_novo ∉ H_substrato      expandir o espaço
```

`mutacao.py` tropeçou porque lá uma primitiva nova só podia ser **composição**
dos átomos — e composição nunca sai do span. Sob aquela definição, invenção era
impossível por construção, não difícil.

A operação que sai do span é outra: **levantar uma constante a parâmetro**.

```
substrato:  MENOR(VAR, NUM)     cortes alinhados aos eixos
inventado:  MENOR(VAR, VAR)     comparação entre duas grandezas
```

`MENOR(VAR,VAR)` não é composição de nada do substrato. Verificado por
enumeração, não alegado.

## `invencao.py` — resultado

Pré-condição (força bruta, **antes** do mecanismo): custo 20–25 sem o átomo
contra **6,0** com ele, 8/8 sementes. A pressão existe.

| | rodada 1 | rodada 2 |
|---|---|---|
| inventou | 3/5 | **5/5** |
| fora do span (enumeração) | 3/3 | **5/5** |
| uso onde serve | 100% | 100% |
| **uso no controle alinhado** | **0%** | **0%** |
| ganho de descrição | +20,7 | +15,3 |

O 3/5 da primeira rodada era amostra insuficiente para a covariância cruzar o
limiar, não falha do mecanismo.

**Ressalva escrita antes de rodar:** a *operação* de generalizar foi escolhida
por mim. O sistema decidiu **quando** aplicá-la e a **qual** encaixe.

## `geracao.py` — a experiência seleciona a OPERAÇÃO?

Três operações de expansão, três famílias de tarefa, e **uma regra geradora só**
para todas: *olhe onde o programa erra e pergunte qual grandeza derivada separa
esses erros*. Sem ramo por tipo de operação — com um ramo por operação, a
informação mútua mediria o meu código.

```
                  operações disponíveis
              1          2          3
relacional    P          P          P
soma          —          S          S
produto       —          —          M

I             0,083      0,361      0,678  bits
acerto         50%        60%        73%
controle        4%         4%         4%
```

A diagonal apareceu **sem ser codificada**, e com 1–2 operações o gerador
**se abstém** nas famílias que não pode resolver em vez de propor a errada.

### Critério pré-especificado, e ele NÃO foi atingido

O relatório exige `I > 1,0` e `acerto > 0,80`. Mediu 0,678 e 73%. A saída diz
**"Parcial"**, e é isso que é.

### Estado registrado

> O sistema demonstrou **expansão do espaço de hipóteses** e **evidência
> parcial de seleção experiencial entre mecanismos de expansão**. A seleção
> ocorre com controle negativo forte (4%), mas **não atingiu o critério
> pré-especificado de robustez** (I > 1,0; acerto > 80%).

E não basta poder estatístico: **robustez fora destas três operações e destas
três famílias não foi testada.** Que o efeito sobreviva a outro repertório é
hipótese.

### O controle que falta, e é o decisivo

Embaralhar a associação entre famílias de tarefa e operações disponíveis,
mantendo as mesmas operações e a mesma quantidade de alternativas. Se o
desempenho depende da experiência real — e não da posição ou identidade
estrutural das alternativas — isso aparece. **Não foi feito.**

## Duas perguntas que continuam separadas

| | |
|---|---|
| o sistema escolhe a operação certa? | **parcialmente demonstrado** |
| o sistema descobre uma operação que ninguém forneceu? | **não demonstrado** |

## Dois defeitos desta fase, e o que eles ensinaram sobre o portão

**1. Busca gulosa com sinal padrão travado.** A lista de decisão fixava o sinal
majoritário e nunca o revisava; em tarefa equilibrada ela desistia no programa
constante. A pré-condição reprovou SOMAR e MULTIPLICAR e **abortou**, e eu
reportei que aquelas operações "não criavam pressão", atribuindo à geometria da
fronteira. Era o instrumento: medido à mão, a mesma tarefa cai de 21 para 10.

> **Um portão que reprova por defeito do próprio solucionador é pior que não
> ter portão: produz conclusão negativa com cara de rigor.**

Daí nasceu `sanidade_do_solucionador()`, que roda antes da pré-condição e
verifica que o solucionador acha a solução quando ela está no conjunto. Sem
ela, "não há pressão" tinha três causas indistinguíveis: não há pressão · a
referência está errada · o solucionador não acha o que sabemos existir.

**2. Gerador sem abstenção.** A regra faz `max_j Separação(X_j, erro)`, e com
muitas grandezas candidatas alguma parece ótima mesmo sendo independente do
erro. Resultado: **64% de propostas nas tarefas alinhadas ao eixo**. Corrigido
com seleção no treino, decisão no holdout, e H₀ por permutação — que absorve
sozinha o número de candidatos, sem correção analítica de comparações
múltiplas. 64% → **4%**.

## O ponto doce não apareceu

A previsão era que diversidade excessiva produzisse mais candidato inútil. Com
três operações, mais diversidade deu mais acerto **e menos** lixo
(50% → 30% → 18%), monotonicamente. Duas leituras possíveis e não sei qual
vale: três é pouco para o trade-off aparecer, ou a abstenção calibrada já
filtra o lixo que a diversidade geraria.

---

# Campo, memória e o que a memória carrega (`intervencao.py` → `gargalo.py`)

## O pivô: observar não basta, é preciso intervir

`campo_recursivo.py` bateu num limite que não é bug. Duas estruturas que
induzem a **mesma partição** do espaço são indistinguíveis por qualquer
quantidade de rótulos: o alvo `(v/v)` teve margem de custo **0,0** sobre a
melhor alternativa, e o alvo não-linear também **0,0**. `v0/v1 < c` é uma reta
pela origem, `v0−v1 < k` é outra reta, e numa caixa limitada elas coincidem o
bastante para o MDL não separar. Ampliar o domínio não muda nada.

Intervenção quebra a degenerescência, porque estruturas que compartilham
fronteira **não** compartilham resposta a escalonamento:

| | dobrar x | dobrar z |
|---|---|---|
| `y = x/z` | y dobra | y cai à metade |
| `y = x−z` | y soma x | y subtrai z |
| `y = x·z` | y dobra | y dobra |

Pontuando a resposta relativa em log — invariante a `E → k·E`, de propósito,
porque interessa a FORMA e não a constante:

| | classificação | intervenção |
|---|---|---|
| margem de identificação | **0,0** | **~3000** |
| recuperação da estrutura | 40% | **100%** |
| cascata E1→E2→E3, campo plano | — | 0% |
| cascata E1→E2→E3, campo cumulativo | — | **100%** |

É o resultado técnico mais sólido desta sessão. Três bugs foram encontrados
por números *impossíveis*, não por teste: margem negativa (erro de unidade,
`EQM + λ·tam` misturando escalas — corrigido para BIC próprio); margem
exatamente `log(n)` idêntica em três mundos físicos diferentes (o rival era
duplicata funcional do alvo — corrigido exigindo assinatura distinta);
recuperação 0% (o vencedor era o negativo funcional do alvo, comparado por
string — corrigido com equivalência funcional).

## Transferência entre físicas reais, e o controle pareado que a derrubou

`descoberta_fisica.py` roda contra o simulador. Fase A (`fusao`) consolida
`(v*v)`, presente em 5 de 8 compostas. Fase B (`fluidos`, sem nenhum dado de
A) usa a estrutura em 4 de 5 medições, ganho médio 10,0 contra 2,0 do controle
vazio. Contrariou minha própria previsão pré-registrada de que falharia.

O controle exaustivo pareado desfez a leitura. Comparando a descoberta contra
**todas** as alternativas de mesmo tamanho e mesmos slots:

| estrutura | descoberta? | ganho na transferência |
|---|---|---|
| `(v*v)` | sim, 5 de 8 vezes | 10,0 |
| `(v/v)` | **não** | **23,4** |

E nenhum sinal disponível no momento da descoberta prevê isso: frequência,
compressão, generalidade e estabilidade **todos** ordenam `(v*v)` acima de
`(v/v)`. Todos erram o mesmo par. O domínio C pré-declarado para desempate não
discriminou (espalhamento 3,0 contra 21,4 em fluidos) — inconclusivo.

## O placebo, e por que ele provou menos do que eu disse

Três braços com semente pré-registrada `20260808`: **A** conserva o que
descobriu, **B** zera o campo, **C** recebe estruturas aleatórias pareadas em
nós, slots e cronograma. Média: **A 50% · B 29% · C 41%**.

Reportei como "qualquer campo enriquecido ajuda; a aprendizagem cumulativa
específica está derrubada". **Superinterpretação, e retratada.** Dois motivos:

**1. Sem poder.** Duas sementes. O episódio 1 é idêntico por construção
(ninguém tem campo ainda) e o episódio 4 é opaco para todos. Sobram **6
células informativas por braço**. Um gap de 9pp contra um limiar de 10pp que eu
mesmo declarei nunca teve chance de ser atingido. "Não detectou" ≠ "não
existe".

**2. Pareou a variável errada junto com a certa.** `resolver()` busca
exaustivamente sobre `compor(campo)`, que é quadrático no campo. Parear nós e
slots pareia o número de instâncias, e portanto o **espaço de busca**:

| campo | termos | candidatos |
|---|---|---|
| vazio | 3 | 54 |
| 1 molde | 9 | 594 |
| 3 moldes | 21 | 3402 |

`A ≈ C > B` é a predição **exata** de "o campo não carrega informação nenhuma,
só multiplica a busca por 63". Conteúdo e orçamento estavam confundidos.

## Um viés nas métricas anteriores, contra a hipótese testada

O critério de cobertura de `placebo.py` e `agente_desenvolvimento.py` era
`e.tam() > 1 and ganho > 5`. Mas uma instância de molde é `E('C', corpo)`, e
`tam()` de um nó `'C'` é **1** — é o próprio ponto do MDL. Consequência:
**quando o molde consolidado explicava a medição sozinho e perfeitamente, o
agente não recebia crédito.** Só contava composição do molde com outra coisa.

Verificado em alvo sintético `y = v0·v1`: o campo contendo exatamente
`(v0*v1)` marca **0%** sob o critério antigo e 100% sob o corrigido. O viés é
**contra** a memória real. Nos dados de física ele custa 5pp exatamente ao
molde `(v/v)`, que é o melhor — o critério penalizava mais a estrutura que
mais explicava sozinha.

## `gargalo.py` — conteúdo ou orçamento?

Desenho pareado: as medições de intervenção são coletadas **uma vez** por
(módulo, semente) e todos os braços são avaliados sobre exatamente os mesmos
pares. Nenhum braço aprende nada; os campos são fixos. Seis sementes, cinco
módulos. A busca é embaralhada e o melhor é registrado a cada orçamento —
mínimos de prefixo, truncar sai de graça.

**Conteúdo, a tamanho e orçamento casados** — quatro campos de 9 termos e 594
candidatos, variando só o molde:

| orçamento | 54 | 594 | irrestrito |
|---|---|---|---|
| `(v*v)` | 40% | 49% | 49% |
| `(v/v)` | 37% | **54%** | **54%** |
| `(v+v)` | 31% | 43% | 43% |
| `(v-v)` | 14% | 33% | 33% |
| espalhamento | **26pp** | **21pp** | **21pp** |

**Tamanho puro** — o sorteio pré-registrado deu, por acaso, o controle que eu
não tinha planejado: o terceiro molde saiu duplicata do segundo.

| | termos | candidatos | cobertura |
|---|---|---|---|
| `aleat2` | 15 | 1710 | 55% |
| `aleat3` | 21 | 3402 | **55%** |

Dobrar o espaço de busca sem acrescentar conteúdo novo: **+0pp**. E a escada
de tamanho (`vazio` 26% → `aleat1` 33% → `aleat2` 55%) é inteiramente
explicada por qual molde entrou — `aleat1` = {(v-v)} rende o mesmo que (v-v)
sozinho, `aleat2` = {(v-v),(v/v)} rende o mesmo que (v/v) sozinho.

**A hipótese "o campo é só orçamento de busca" está refutada.** O veredito
automático impresso pelo código diz o contrário e está errado: ele compara
`aleat3` com `vazio` truncando ambos em 54, o que deixa `vazio` exaustivo
(54 de 54) e `aleat3` em 1,6% do seu espaço. Mede agulha-no-palheiro. Ficou no
arquivo sem remendo, com a análise do defeito ao lado — reescrever veredito
depois de ver o número é o que estes portões existem para não fazer.

## O achado que muda o desenho do próximo experimento

| orçamento | `(v-v)` | `vazio` | |
|---|---|---|---|
| irrestrito | 33% | 26% | guardar molde ruim é **de graça** |
| 54 | **14%** | 26% | guardar molde ruim **custa** |

Sem teto de busca, um candidato a mais nunca piora um mínimo — então "guardar
tudo" vence por construção e **nenhuma política de consolidação pode perder
para outra**. Com teto, guardar errado dilui a busca e machuca.

> Qualquer comparação entre políticas de consolidação — imediata, memória de
> hipóteses, memória com evidência transdomínio — só tem o que medir sob
> **orçamento de busca fixo**.

## Estado

| | |
|---|---|
| intervenção separa o que observação funde | **sustentado**, margem 0 → ~3000 |
| conteúdo do campo importa, a tamanho casado | **sustentado**, 21–26pp em 6 sementes pareadas |
| tamanho do campo, por si, importa | **refutado**, +0pp ao dobrar sem conteúdo novo |
| a política atual escolhe bem o que guardar | **não** — escolhe a 2ª de 4, e nenhum sinal da origem prevê a ordem |
| existe política melhor, e ela é aprendível | **não testado** — depende de orçamento fixo |

## `politicas.py` — o valor de uma representação é estimável antes de ser usada?

Antes de comparar políticas de consolidação, a pergunta que decide todas de uma
vez: **a utilidade medida em algumas físicas prevê a utilidade em outras?** Se
não prevê, nenhuma política pode funcionar — não por escolher mal o critério,
mas porque a informação não existe nem depois do fato.

Doze módulos partidos **antes** de olhar qualquer número, com as quatro
categorias dos dois lados para não confundir partição com categoria:

| | módulos |
|---|---|
| DECISÃO | quantum · materia · quimica · fluidos · fusao · portal |
| AVALIAÇÃO | bec · ressonancia · gravidade · relatividade · vacuo · caret |

Quatro campos de 9 termos e 594 candidatos — tamanho e orçamento idênticos, só
o conteúdo muda. Pareado: as medições são colhidas uma vez por (módulo,
semente) e todos os campos são avaliados sobre os mesmos pares. Quatro
sementes, 48 células.

### H1 passou, sem margem para interpretação

| conteúdo | DECISÃO | AVALIAÇÃO |
|---|---|---|
| `(v/v)` | **41%** | **63%** |
| `(v*v)` | 35% | 56% |
| `(v+v)` | 35% | 50% |
| `(v-v)` | 28% | 49% |

Ordem em DECISÃO e em AVALIAÇÃO: **idênticas**. Concordância de pares **6/6**,
topo coincide. A ordem de utilidade se transfere entre físicas que não
compartilham dado nenhum.

| política | rende na AVALIAÇÃO |
|---|---|
| piso — pior conteúdo | 49% |
| sortear | 54% |
| **frequência na origem → `(v*v)`** | **56%** |
| **evidência transdomínio → `(v/v)`** | **63%** |
| teto — melhor conteúdo | 63% |

Deferir a decisão e medir em outros domínios captura **todos** os 14pp
disponíveis. A política atual captura 7 dos 14.

### O veredito automático de H2 estava errado, e o defeito foi sinalizado antes

`max()` desempata pela ordem da lista, então módulos onde todos os braços
empatam carimbavam vencedor fictício e H2 lia "heterogêneo". Dos 4 módulos
creditados a `(v*v)`, **três eram empates puros** — `quimica` (0% para todos,
inclusive o campo vazio), `relatividade` (67% para todos), `caret` (75% para
todos). Contando só vencedor **estrito**:

| | módulos |
|---|---|
| `(v/v)` **6x** | quantum · materia · fluidos · ressonancia · gravidade · vacuo |
| `(v*v)` **1x** | **fusao** |
| `(v+v)` 1x | portal |
| `(v-v)` 0x | — |
| sem sinal | quimica · bec · relatividade · caret |

Dominância 6/8 = **75%**. Não é heterogêneo. Minha previsão pré-registrada de
que H2 apareceria estava **errada**.

### O mecanismo da falha, agora nomeado

`(v*v)` vence estritamente em **um** módulo: `fusao`. Que é exatamente o módulo
onde a política de frequência a consolidou, em `descoberta_fisica.py`.

> A consolidação por frequência na origem seleciona a estrutura que serve ao
> **módulo de origem**, não a que serve em geral. Não é critério subótimo — é
> **sobreajuste ao domínio de descoberta**.

Isso explica de uma vez os quatro resultados anteriores: por que `(v/v)`
transferiu melhor no controle pareado, por que nenhum dos cinco sinais da
origem previa a ordem, e por que o placebo empatava.

### Limites

- Quatro candidatas. Escolher a melhor de 4 é fácil; nada aqui diz que a ordem
  se transfere numa população grande de estruturas descobertas.
- P3 usa como critério de decisão a mesma métrica da avaliação, medida em
  outros módulos. É a pergunta certa — a métrica generaliza entre físicas? —
  mas é o **teto** de uma política dessas, não uma política implementada.
- As estruturas foram dadas, não descobertas. O elo completo
  `descobrir → suspender → medir → comprometer` ainda não rodou ponta a ponta.

## Estado após `politicas.py`

| | |
|---|---|
| o conteúdo do campo importa, a tamanho casado | **sustentado** |
| tamanho do campo, por si | **refutado** |
| a utilidade de uma representação se transfere entre físicas | **sustentado**, 6/6 |
| a política atual escolhe mal, e sabe-se por quê | **sustentado** — sobreajuste à origem |
| deferir e medir recupera o ganho | **sustentado como teto**, +7pp, alcança 100% do disponível |
| uma política implementada faz isso sozinha | **não testado** |
| o sistema inventa fora da gramática dada | **não demonstrado** |

## `ciclo.py` — INCONCLUSIVO POR CONSTRUÇÃO

O elo ponta a ponta: descobrir → suspender → medir fora da origem →
comprometer, com candidatas geradas pelo próprio sistema, orçamento de busca
fixo em 594, e 6 módulos de avaliação nunca vistos por nenhuma decisão.

| braço | cobertura na AVALIAÇÃO |
|---|---|
| campo vazio | 33% |
| piso — pior do pool | 49% |
| P0 sorteio | 59% |
| P1 frequência na origem | 62% |
| P3 evidência transdomínio | 64% |
| "teto" | 63% |

O código imprimiu **C3 · o sinal não sobrevive à autodescoberta**. A leitura
não se sustenta. Três defeitos, e o primeiro impede a pergunta de ser feita.

**(a) `compor()` compõe um nível só.** Devolve os termos, os pares binários
deles, e `abs` disso. Partindo do campo vazio — 3 variáveis — o maior
candidato tem tamanho 3. Verificado:

```
tamanhos alcançáveis a partir do campo vazio: {1: 3, 2: 3, 3: 24, 4: 24}
formas distintas com tam>=3: 8
['(v*v)','(v+v)','(v-v)','(v/v)','|(v*v)|','|(v+v)|','|(v-v)|','|(v/v)|']
```

Abstraídas, são **exatamente as quatro** que `politicas.py` recebeu de graça. O
pool de 3 moldes não é achado empírico sobre autodescoberta — é consequência
aritmética do gerador. O experimento redescobriu o pool dado e testou de novo
a pergunta já respondida.

**(b) Escolher 2 de um pool de 3 satura a comparação.** P1 e P3 compartilham
pelo menos uma candidata por construção; o espaço de discordância é de uma
posição. Piso de medição, mesma família do bug de `intervencao_ativa.py`.

**(c) O "teto" não é teto.** Definido como o melhor molde **sozinho**, mas as
políticas consolidam **dois** — daí P3 64% "superar" o teto 63%, que é
impossível e denuncia a definição. O pré-registro dizia que eu desconfiaria de
vazamento se isso acontecesse; procurei, e não é vazamento, é o teto calculado
sobre campo de tamanho diferente.

**O que sobrevive:** P3 escolheu o melhor molde do pool em **4/4 sementes**, e
P1 não. Mesma direção de `politicas.py`. Mas com pool 3 e escolha 2, isso é
fraco demais para sustentar C1 e não é o que estava pré-registrado.

**O conserto e o seu preço.** Para o pool ter profundidade, a descoberta
precisa iterar com a população carregada no campo — um molde de profundidade 1
permite que `compor()` alcance profundidade 2, como em `campo_recursivo.py`.
Isso corrige a impossibilidade de fazer a pergunta, não ajusta na direção de
um resultado desejado. Mas carregar a população no campo faz a suspensão
**custar busca**, e `gargalo.py` mede esse custo em 12pp por molde ruim a
orçamento fixo. P3 passa a pagar pelo que suspende, e pode piorar. É a versão
honesta do teste, e fica declarada antes de rodar.

---

# ARC — auditoria, síntese e busca guiada

## A auditoria do número anterior

`arc_integrated_results.json` registrava `pixel_accuracy 0.802`,
`copy_baseline 0.7663`, `exact_match 0.0048` e o veredito **"APRENDIZADO
DETECTADO"**. Três defeitos, achados lendo o código antes de rodar qualquer
coisa:

**1. O veredito não discriminava.** A cascata era `mex>0.05 → BREAKTHROUGH ·
mpx>mcp*1.15 → SINAL FORTE · mpx>chance*2 → APRENDIZADO DETECTADO`. Com
`N_COLORS=11`, `chance=0.1`, então o terceiro ramo dispara com pixel accuracy
**> 20%**. Copiar a entrada dá 76,6%. Um sistema que só copia recebia o mesmo
carimbo.

**2. Não havia semente.** `torch.manual_seed` e `np.random.seed` nunca eram
chamados. O número não era reproduzível e a variância nunca fora medida.

**3. Não havia controle negativo.** Nada comparava contra a mesma arquitetura
sem treino.

`auditoria_arc.py` — três braços sobre os mesmos lotes, 5 sementes:

| braço | pixel acc | desvio | exact match |
|---|---|---|---|
| copiar a entrada | 76,7% | 1,17% | 6/2000 |
| arquitetura sem treino | 15,1% | 12,82% | 0/2000 |
| treinado | 79,9% | 0,63% | 7/2000 |

**A1 passou:** diferença pareada **+3,15pp**, 95% dos lotes positivos, muito
acima de 2 erros padrão. **A2 passou a favor do sistema:** o treino acrescenta
**+64,7pp** sobre init aleatório — não há prior de cópia na arquitetura.

**A4 é o que importa:** exact match **7/2000 contra 6/2000 de copiar**. Uma
tarefa a mais em duas mil. Aprendizado de regularidade de pixel: sim, medido.
Resolução de ARC: não.

## `arc.py` — trocar o mecanismo, não ajustar o modelo

O ARC tem a mesma estrutura que o `novaphys` explora em física: existe um
**verificador que não pode ser convencido**. Em física é o `custo()` sobre
resposta a intervenção; no ARC é igualdade exata de grade.

| abordagem | exact match | custo |
|---|---|---|
| copiar a entrada | 0,30% | — |
| transformer treinado | 0,35% | treino em GPU |
| DSL 17 primitivas, prof 3 | 1,0% | 140s CPU |
| DSL 24 primitivas, prof 2 | 2,2% | 24s CPU |
| **DSL 24 primitivas, prof 3** | **3,5%** | 586s CPU |

**10× o modelo treinado, sem treinar nada.** Dentro da previsão pré-registrada
(2–6%), então não foi tratado como surpresa. 16 acharam o treino, 14 acertaram
o teste — precisão 87,5%.

O `B2` do pré-registro disparou na v1 (1,0% < 2%) e mandou **ampliar a
gramática antes de medir acumulação**, não interpretar ganho sobre piso ruim.
A família que mais rendeu foi lógica entre painéis (`pOU`, `pXOR`).

## `arc_biblioteca.py` — compressão sim, alcance não

Currículo com separação real: macros mineradas das 400 tarefas de **treino**,
tudo medido nas 400 de **avaliação**. Orçamento casado em 14.425 programas.

| braço | resolvidas | prof. média |
|---|---|---|
| G0 base | 14/400 | 1,71 |
| G1 freq (`simH.simV`) | 14/400 | **1,57** |
| G1 prof (mesma macro) | 14/400 | **1,57** |
| G1 placebo (`rot90.rot180`) | 14/400 | 1,71 |

**L3 — não estende alcance.** Mas o placebo permitiu a distinção fina:

> A compressão **é específica do conteúdo**. A macro minerada comprime
> profundidade; a aleatória pareada não comprime nada. Abstrair do próprio
> trabalho faz algo real — só que achar o conhecido mais barato não é
> alcançar o desconhecido.

**Dois defeitos do desenho, registrados:**

- **A mineração perdeu dois braços.** Só um fragmento recorria em 2+ tarefas,
  então `freq` e `prof` escolheram a mesma macro e **L4 não pôde
  discriminar**. O `L0` checava se havia soluções compostas — havia — mas não
  se havia diversidade para os critérios se separarem. Pré-condição
  incompleta.
- **Na profundidade 3 a busca era exaustiva.** Orçamento `14.425` = espaço
  `14.425`. Reordenar ou encurtar não pode mudar o conjunto examinado, só a
  ordem — e como a métrica é "achou ou não", o resultado é idêntico **por
  construção**. Isso vale para a biblioteca e para o guia.

> **Ordenação só importa quando o orçamento é menor que o espaço.**

O código agora avisa sozinho quando `orçamento >= espaço`.

## `arc_guia.py` — a rede propõe, o verificador dispõe

O gargalo medido é profundidade de busca: prof 3 = 14.425 programas (586s);
prof 4 = 346.201 (~4h); prof 5 = 8,3M.

A peça que faltava é uma **distribuição de proposta aprendida** — e é onde uma
rede pertence a esta arquitetura: ordenando o espaço, nunca decidindo. Se
ordenar mal, o pior caso é a ordem arbitrária de antes, e isso aparece no
número porque o orçamento é casado.

**O dataset se rotula sozinho:**

```
sorteia programa da DSL → aplica a grades REAIS de treino
    → par (entrada, saída) COM o programa que o gerou
```

Sem anotação humana, quantidade ilimitada, rótulo perfeito. Rede de **63.352
parâmetros**, cabe em 4 GB, treina em minutos. Grades de origem só do conjunto
de treino; a avaliação não entra em nenhuma etapa.

| | topo-1 | topo-3 |
|---|---|---|
| acaso | 4,2% | 12,5% |
| guia treinado até prof 3 | **44,6%** | **80,9%** |
| guia treinado até prof 4 | **46,9%** | **83,0%** |

**N0 aprovado com folga** — 10,6× o acaso. O portão existe porque um resultado
negativo causado por defeito do instrumento é pior que nenhum resultado.

## Resultado do guia — e por que a profundidade 3 não media nada

| profundidade | orçamento | espaço | base | guiado |
|---|---|---|---|---|
| 3 | 14.425 | 14.425 (**100%**) | 14/400 · prof 1,71 | 14/400 · prof 1,71 |
| 4 | 50.000 | 346.201 (14%) | 14/400 · prof 1,71 | **15/400** · prof 1,87 |

Na profundidade 3 o orçamento **igualava o espaço**. A busca era exaustiva, os
dois braços examinavam o mesmo conjunto, e a métrica é binária — o empate era
**dedutível antes de rodar**, não um achado.

> Ordenação só importa quando o orçamento é menor que o espaço.

Na profundidade 4, com 14% visível, o guia achou **uma** tarefa a mais.
+0,25pp, abaixo do limiar de 1pp pré-registrado: **N2 pela própria régua**.

A aritmética das médias sugere onde: `14 × 1,71 ≈ 24` e `15 × 1,87 ≈ 28`, então
a tarefa extra tem profundidade 4 — alcançada pela ordem, não pelo orçamento.
É inferência aritmética, não medição por tarefa. E **n=1 não é evidência**.

## A leitura que os dados sustentam

Ordenar 14% de um espaço de **cadeias lineares** compra ~1 tarefa, mesmo com
uma rede que prevê primitivas a 10,6× o acaso. O guia não é fraco — o espaço
tem a forma errada.

Evidência independente disso, nas próprias soluções: **8 das 14 usam `pOU` ou
`pXOR`**, que são as únicas primitivas onde **ramificação** está escondida
dentro de um símbolo escrito à mão. `combina(f(x), g(x))` disfarçado de
primitiva carrega a maioria das vitórias.

A ampliação indicada é de **forma**, não de lista:

```
P ::= prim | P . P              cadeia        (única forma existente)
    | combina(P, P)             ramificação   (hoje só à mão)
    | por_objeto(P)             por peça      (zero das 14 soluções)
    | onde(pred, P, P)          condicional
```

## Uma regressão minha, no ARC

A macro da biblioteca foi `simH.simV` — **sequência fixa, sem buracos**. O lado
da física usa `Molde` com `n_slots`, instanciado sobre atribuições. É a mesma
regressão que documentei em `campo_recursivo.py` (congelar índices em vez de
abstrair para slots), cometida de novo e não percebida na hora.

> A biblioteca do ARC aprende **constantes, não esquemas**.

## Contagem de vereditos automáticos errados nesta sessão

`G3` · `H2` · `C3` · `N2` (prof 3) · `N2` (prof 4) — **cinco**. Os números
brutos estavam corretos nos cinco casos; a camada de interpretação é que
falhou. O último é factualmente falso: diz "a busca exaustiva já cobre quase
tudo" numa rodada com 14% do espaço visível — string fixa herdada de outro
regime.

**Regra adotada:** reportar a tabela primeiro, o veredito depois, e tratar
qualquer leitura automatizada como suspeita até conferir contra os brutos.

---

# `prion.py` — a bancada aplicada a Creutzfeldt-Jakob

Modelo de polimerização nucleada (Masel, Jansen & Nowak 1999), forma reduzida
em momentos. **Não é uma cura** — é análise de dose mínima eficaz com portões
de pré-condição e teste de robustez.

## Quatro erros meus, todos pegos por controle

| # | erro | como foi pego |
|---|---|---|
| 1 | parâmetros "da literatura" inconsistentes: `e·b = 4,4e-9` vs `a² = 1,6e-3`, 6 ordens abaixo da decolagem | portão **P0** reprovou |
| 2 | teste de não-monotonicidade filtrava os `None`, apagando a própria propriedade | dados brutos contradiziam o veredito |
| 3 | **RK4 de passo fixo em sistema rígido** (`d`=4/dia vs `a`=0,04/dia): "200 dias" era erro de integração puro | varredura: dt=0,02→200 dias, dt=0,01→nunca |
| 4 | `LIMIAR=2e6` era 18× acima do máximo atingível `λ/a=110.000`; e `limiar=LIMIAR` como default era avaliado na definição da função | derivação analítica + inspeção da trajetória |

O erro 3 teria produzido uma tabela inteira de alvos terapêuticos feita de
artefato numérico, com casas decimais.

## Verificação

```
equilíbrio analítico   y* = 108.304
simulação (Radau)      y  = 108.303,8
```

Convergido em rtol 1e-7, 1e-9, 1e-11 — todos dão incubação 200,0 dias.

## Dose mínima que INTERROMPE a progressão

| alvo | dose mínima |
|---|---|
| **aumentar depuração de agregados** | **20,0%** |
| reduzir produção de PrP (ASO, ION717) | 24,5% |
| reduzir alongamento | 49,6% |
| degradar PrP-C mais rápido | 49,6% |
| reduzir fragmentação | 56,8% |

**Robustez: `a+` lidera em 17/17 configurações** (fragmentação, alongamento e
depuração variados 0,5×/1×/2×). A ordem não é escolha de parâmetro.

Interpretação: degradar um agregado remove **massa e pontas de nucleação ao
mesmo tempo**; reduzir produção ataca só o suprimento.

## Combinar DILUI, a dose total casada

```
melhor sozinho     depuração        20,0%
melhor combinação  ASO + depuração  23,5%
```

Contraria a intuição de terapia combinada. Explicação: reação em cadeia tem
**limiar** — meio empurrão em dois alvos deixa os dois abaixo da linha.

## Limite que impede leitura clínica

"20% de mudança num parâmetro" **não é** "uma dose de fármaco". Aumentar
depuração em 20% pode ser muito mais difícil que reduzir PrP em 25%, e o ASO
intratecal sabidamente consegue o segundo.

A leitura correta: **o ASO fica a 4,5 pontos do alvo teoricamente mais
eficiente** — a abordagem já em ensaio é quase ótima entre alvos únicos. Isso
valida o método; não contradiz o campo. O achado não-óbvio que sobrevive aos
controles é a **ausência de sinergia**.
