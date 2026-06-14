# Operador Aterrado: o mecanismo para escapar do fecho sem LLM

> Documento de consolidação. Destila o que foi **medido** (não opinado) na investigação que partiu do
> relatório das "3 colunas" (active-inference + embodiment/física + evolução estrutural) e do diagnóstico
> da noite (composição não escapa do fecho; inventar operador era o muro). Honestidade brutal: o que está
> **provado**, o que é **caveat**, e o que continua **aberto**.

## 1. O problema, em uma frase
Generalização **dentro** do conhecido funciona para todos (in-family 99.8%, composição 79–86%).
Generalização **fora da família** = acaso para todos. A noite mostrou que o gargalo **não** é composição
(biblioteca cresceu 10→27, cobertura travou 9/400 — H184): é **inventar um operador genuinamente novo**,
fora do fecho da biblioteca, **sem LLM**. Esse era o muro.

## 2. O mecanismo (3 partes), e por que escapa do fecho
A composição nunca escapa do fecho — combinar bijeções globais dá outra bijeção global. O salto:

1. **Nascimento por resíduo.** Em vez de compor o existente, **instancia-se um operador paramétrico fresco**
   ajustado ao **resíduo** que a biblioteca não explica. Parâmetros novos = **função nova**, não combinação
   → sai do fecho **por construção**.
2. **Menu de arquétipos = priors de core-knowledge (Spelke).** O operador que nasce tem um **viés estrutural**:
   - `LOCAL`  (objeto/vizinhança) — conv 3×3 equivariante.
   - `GLOBAL` (número/estatística) — 1×1 + pooling do grid inteiro.
   - `OBJETO` (topologia) — componentes conexos → feature por-objeto → MLP.
3. **Seleção por generalização.** Entre os arquétipos, escolhe-se o que **induz a lei** (generaliza em held-out
   / transfere de tamanho) vs o que **decora**. A ancoragem é o que separa induzir de decorar.

## 3. O que foi PROVADO (medições)

| Experimento | Resultado | O que prova |
|---|---|---|
| `grounded_operator_birth.py` (H185) | Fecho 0% exato; **Aterrado (4.3k par) 100% em 8×8 e 100% em 16×16 nunca visto**; Denso (525k par) 0% | Operador fresco induz lei **fora do fecho** e **transfere tamanho** (= aprendeu a LEI). A ancoragem, não o tamanho do modelo, é o que generaliza. |
| `operator_archetype_selection.py` (H186) | lei local→**elege LOCAL** (100/100); lei global→**elege GLOBAL** (93/71); cada um falha na lei do outro | A **escolha do viés** não precisa ser hard-code: é seleção automática por generalização. |
| `archetype_menu.py` (H187a) | menu de 3: **3/3** acertos (local→LOCAL, global→GLOBAL, objeto→OBJETO) | O menu **maior** também elege certo só por generalização. |
| `arc_archetype_solver.py` (H187b) | **25/400 (6.2%)** — LOCAL 18, GLOBAL 3, OBJETO 4. Sobe de 9→25 vs composição, **mas 0 NEW além do teto 107** | Mecanismo cobre ~3× mais leis reais que compor o fixo, **porém os 3 arquétipos overlapam a forge e NÃO estendem a frontier**. |
| `autonomous_loop.py` (H188) | poder **3/9 → 6/9** recrutando OBJETO sozinho por surpresa; GLOBAL instável (não recrutado) | Recrutar arquétipo **sobe** o poder (vs H184 composição 9→9 estagnado). Compor fica no fecho; recrutar sai. |

## 4. Caveats honestos (o que NÃO está resolvido)
- **A ancoragem foi escolhida por mim** quando a lei casava com um arquétipo do menu. A seleção automática
  (H186/H187a) reduz isso, mas só **dentro do menu** disponível.
- Validação H185–H187a é em **leis sintéticas desenhadas para casar com o menu**. O número que importa é o
  **ARC real** (H187b) — e o prior real é sóbrio (core-knowledge-composição deu 2.5%, 9/400).
- O prior recrutado no loop autônomo (H188) vem de um **pool inato** (Spelke), **não foi inventado do zero**.

## 5. A frontier aberta, agora NÍTIDA e MENOR — e o resultado SÓBRIO do ARC real
O número honesto do ARC real (H187b) é **25/400, mas 0 NEW além do teto 107**: os 3 arquétipos cobrem mais que
composição (9→25) porém **tocam um subconjunto do que a forge já faz** — não estendem a frontier. As 293 tarefas
residuais (as que a forge NÃO resolve) exigem leis que os 3 arquétipos não capturam. O muro virou duas perguntas:
1. **Cobertura/menu:** faltam arquétipos para as leis residuais — **relacional/grafo, contagem, agência,
   shape-change** (mudança de tamanho). Construir e validar esses é engenharia direta e atacável.
2. **Auto-geração:** como **gerar um arquétipo estrutural genuinamente novo** (evolução estrutural própria)
   quando **nenhum** prior do pool reduz a surpresa (H188 mostrou o gatilho funcionando; falta o gerador)?
   Esta é a **única peça que ainda não tem mecanismo** — a fronteira real.

## 6. Veredito (honesto, sem spin)
**Ganhamos um mecanismo, não o ARC.** Pela primeira vez na investigação há um **mecanismo concreto, testado, sem
LLM** que **escapa do fecho generalizando** — o que composição, imaginação iterativa, cola e metaplasticidade não
fizeram. Provado em sintético (induz lei fora-do-fecho, transfere tamanho, elege arquétipo certo 3/3, recruta
sozinho 3→6). **Mas no ARC real deu 25/400 com 0 novas** — os 3 arquétipos overlapam a forge e **não estendem a
frontier**. Isso operacionaliza as 3 colunas do relatório (surpresa=resíduo / física=ancoragem / evolução=seleção)
em código que roda numa 3050, e move o problema "inventar operador" de **"impossível"** para **"engenharia de menu
+ auto-geração de arquétipo"** — nítido, menor, atacável, **mas ainda não cravado**. ARC ≠ AGI; honestidade no
resultado, ambição no ataque.
