# AGI 2.0 — arquitetura, resultados medidos, e o que falta

Documento de estado em **2026-09-07**. Todos os números aqui vêm de execução
real com controle pareado e orçamento casado. Onde um resultado tem ressalva
que muda a leitura, a ressalva está junto do número, não numa nota de rodapé.

Benchmark: **ARC-AGI-1, conjunto de AVALIAÇÃO (400 tarefas)**. O conjunto de
treino é ~2,6× mais fácil (37/400 na mesma busca) — todo diagnóstico feito nele
engana.

---

## 1. O que o sistema é

Não é uma rede treinada. É um **sintetizador de programas** que, para cada
tarefa, resolve por dedução qual transformação leva a entrada na saída, e
verifica exigindo igualdade exata em grades que nunca viu.

### 1.1 Três motores com vieses diferentes

| motor | representação | resolve |
|---|---|---|
| **monolítico** | cadeia de primitivas `Grade → Grade` | 20/400 |
| **tipado** | 5 famílias de prior, operadores com argumentos | 15/400 |
| **inferência relaxada** | `célula(i,j) = TESTE ? LEITURA_v : LEITURA_f` | 8/400 |

A interseção entre o tipado e o relaxado é **exatamente zero**. Dois módulos
com competência completamente disjunta que nunca se falam. União verificada:
**25/400**.

### 1.2 O núcleo: dedução, não amostragem

O mecanismo central é resolver restrições em vez de sortear candidatos. Para a
meta-gramática `célula(i,j) = TESTE ? LEITURA_v : LEITURA_f`:

1. calcula-se a **máscara de bits** de cobertura de cada leitura — quais
   células ela acerta, concatenadas num único inteiro sobre todos os pares;
2. varre-se **todos** os pares (v, f) — dado `v`, o que `f` precisa cobrir está
   determinado: exatamente o que `v` deixou de fora;
3. o teste precisa ser verdadeiro onde só `v` acerta e falso onde só `f` acerta,
   o que é uma operação inteira por candidato.

Sem corte, sem heurística, sem top-K. Isso importa: a primeira versão mantinha
"os 60 melhores" e varria 3.600 de 3,8 milhões de pares — era amostragem
disfarçada, exatamente o que o arquivo existia para não fazer.

### 1.3 Aquisição de prior a partir de mundo

Primitivas **não escritas à mão**, extraídas por observação de um mundo
simulado: objetos coesos, gravidade, empurrão, agregação, simetrização, e —
o que mais rendeu — **camadas, dobra e oclusão**.

O controle é um mundo de **permutações fixas**: igualmente determinístico,
consistente e aprendível, sem estrutura física.

---

## 2. Disciplina experimental

Isto não é enfeite; é o que separa este documento de uma lista de esperanças.
Vinte e oito correções foram feitas nesta linha de trabalho, e **cada uma delas
teria virado uma conclusão falsa publicável**.

- **Pré-registro numerado** em cada arquivo, escrito antes de rodar, incluindo
  a minha previsão. Quando erro, fica registrado que errei.
- **Orçamento casado** — braços comparados no mesmo número de candidatos, e
  quando isso ficou suspeito, também no número de aplicações de operador.
- **Placebo pareado** — o braço de controle tem o mesmo tamanho, a mesma forma
  e o mesmo custo, e só não tem a estrutura.
- **Verificador exato** — igualdade célula a célula nas grades de teste.
  Ajuste extensional não passa nisso.

### Modos de falha que já enganaram o veredito automático

| erro | sintoma | o que teria sido concluído |
|---|---|---|
| população de um | veredito "CRESCE" | automodificação evolutiva funciona |
| censura de não-convergidos | média baixa | mecanismo 1,3× em vez de 2,1× |
| régua por contagem | "nunca converge" | objetivo não é identificável |
| controle sem a resposta | braço A vence sempre | prior humano ajuda |
| parâmetro escrito e não lido | 4 distâncias idênticas | efeito não depende da distância |
| arquitetura impossível | 33,3% em 4 pontos | rede é ineficiente em dado |
| amostra sem quase-erro | 99,6% e transfere | features estáticas bastam |
| geometria insatisfazível | braço seca | partição vale +2 |

**Número redondo repetido é assinatura de fronteira de expressividade, não de
curva de aprendizado.** Foi assim que a rede convolucional saturando em
exatamente 33,3% em quatro tamanhos de dado foi diagnosticada: convolução é
equivariante a translação e não *pode* expressar rotação — 100% nas 4 leis por
célula, 0% nas 8 espaciais.

---

## 3. Resultados fortes

### 3.1 Deduzir supera sortear por ~3.000×

Espaço de 178 milhões de operadores. Amostrar acha **1 útil em 3.000**. Deduzir
a partir da diferença entrada/saída acha o operador **exato, precisão 100%** —
toda lei deduzida acerta o teste.

### 3.2 Estrutura incompleta precisa de permissão para existir

| exigência de cobertura | tarefas | composições de 2 passos |
|---|---|---|
| 100% | 3 | **0** (mesmo com 2,4× o orçamento) |
| 95% | 7 | — |
| **85%** | **8** | **5** (com 41% do orçamento) |
| 70% | 7 | — |
| 50% | 7 | — |

Curva em U com ótimo em 0,85. Não é profundidade: é o **critério de aceitação
de hipótese intermediária**. Estrutura que está errada sobre 15% das células é
o que destrava o passo seguinte.

Isto é, medido, a previsão do modelo REBUS (Carhart-Harris & Friston) sobre
psicodélicos: afrouxar a precisão das crenças de alto nível libera sinal de
baixo para cima — e afrouxar demais desorganiza.

### 3.3 Estrutura vence gradiente por 2.500× em dado

Mesma lei escondida, mesmos pares, teste em grades novas:

```
ABSTRAÇÃO (elimina por equivalência)   100% com      2 exemplos
GRADIENTE (rede densa, sem viés útil)  100% com  5.000 exemplos
                                        71% com  1.000
                                         0% com    200   (72% por célula)
```

**Ressalva que é o achado:** vale 2.500× *dentro da família certa* e **zero**
fora dela. A gramática tem 110 leis porque eu escrevi 10 átomos. O prior de um
LLM foi comprado a peso de dado e não tem fronteira.

### 3.4 Objetivo precisa de quantificador

O resultado mais limpo da linha de agência. Duas medidas independentes.

**Xadrez** — aprender o que é vitória em Rei+Torre, testar em Rei+Dama (jogo
nunca visto), com amostra de **quase-erros** (mate / afogamento / xeque com uma
fuga / xeque com várias):

```
TETO de QUALQUER função booleana das features:
  L1  13 features estáticas          95,06%
      66 de 668 posições têm vetor IDÊNTICO e rótulo diferente
  L2  L1 + 1 quantificador          100,00%    zero colisões

melhor regra, enumeração exaustiva:
  L1   93,26% em KRK  →  83,88% em KQK
  L2  100,00% em KRK  → 100,00% em KQK    `atacado E sem_saida`
```

O teto é a parte que não se discute: não depende de eu ter escolhido bem as
features. É limite da **linguagem**.

**Armadilha que só a transferência pega:** com amostra fácil, a indução escolhe
a regra de um literal `sem_saida` — que **confunde ganhar com empatar**. Ajusta
100% no treino e é falsa. A evidência que força a regra certa são os
afogamentos, e eles são raros. *Treino fácil ensina errado.*

**Labirinto próprio**, 25 pares, treina num mundo e aplica noutro:

```
OBJETIVO QUANTIFICADO   saídas==1 E bolso>=2
      transf.   refutou   ações até convergir
  L1   49,6%     24/25          175
  L2   99,3%      0/25          190      custo 1,09×

OBJETIVO PROPOSICIONAL  pos==porta E bolso>=2   (as duas expressam)
  L1   49,7%      0/25          143
  L2   49,7%      0/25          172      custo 1,20×
```

L1 **se refuta sozinha em 24/25**: a eliminação chega ao conjunto vazio — a
linguagem diz "o objetivo não está em mim". Sinal limpo, não bug.

A segunda condição é o que faz os 99,3% valerem: as duas linguagens transferem
**no acaso** quando o objetivo é uma coordenada. Então a leitura certa não é
"quantificador generaliza" — é **objetivo sobre estrutura do mundo generaliza,
objetivo sobre lugar não generaliza**, e sem quantificador não dá nem para
escrever um objetivo sobre estrutura.

### 3.5 Prior extraído de mundo transfere; placebo não

Profundidade 3, orçamento casado, os dois braços pagando o mesmo preço de
diluição (o espaço vai de 13.824 para 54.872 e o orçamento corta antes do fim):

```
BASE          14/400   24 primitivas
BASE+RUÍDO    14/400   38 primitivas   →  +0    permutações
BASE+DOBRA    17/400   38 primitivas   →  +3    física + camadas
ganhas: 195ba7dc, 5d2a5c43 (painel) e d282b262 (gravidade)
PERDIDAS: nenhuma
```

Primeira vez que primitiva **não escrita por mim** move o número de referência.

**Detalhe que caiu da física e não foi desenhado:** dobrar uma grade de largura
ímpar ao meio descarta exatamente a coluna do vinco. O separador do painel some
sozinho. Ninguém escreveu "remova a coluna separadora".

### 3.6 Exploração dirigida, e sua condição de aplicabilidade

Novidade sobre o **estado completo** (posição + chaves + bolso) contra
exploração aleatória, pareado: **128 vitórias, 5 derrotas, 17 empates,
p = 0,00001**. O agente coleta chaves espontaneamente porque cada chave
destranca uma fatia inteira de estados nunca vistos.

Desenho ótimo de experimento (escolher a ação que mais separa hipóteses):

```
evidência PERTO   55 vitórias contra 57, p = 0,61   — empate
evidência LONGE   30 a 54× mais rápido             — enorme
```

**Ressalva que invalida a leitura forte:** o planejador do caso "longe" usava a
função de transição **verdadeira**, embutida no código. Com modelo **aprendido**
ele não consegue expandir a busca, cai no fallback e vira idêntico à novidade —
razão **1,0×**. O 30-54× vale sob a condição "modelo de mundo dado de graça".

### 3.7 Informatividade é monótona não-crescente

Sob eliminação pura, `min(v, n)` num estado fixo **nunca sobe**. Verificado por
força bruta em 200.000 casos, zero exceções.

Consequência dura: **um estado que hoje não informa nunca passará a informar.**
Valor informacional prospectivo não pode vir de hipóteses ficarem mais
diferentes. Sobra uma única fonte: estados ficarem **alcançáveis** — que é o
que a busca por novidade fornece.

### 3.8 Não repetir trabalho vale mais que a maioria das ideias

```
                              tarefas   candidatos/tarefa
produto cartesiano, 24 prim     14           14.424
largura com prefixo, 24 prim    16            7.009
produto cartesiano, 38 prim     17           14.424
largura com prefixo, 38 prim    20           13.766
quociente observacional         —      2,06× menos trabalho
```

`itertools.product` reenumera cada sequência do zero e gasta orçamento em
caminhos cujo prefixo já morreu. **+3 tarefas por compartilhar prefixo**, sem
nenhum vocabulário novo.

O quociente por equivalência observacional (dois programas com a mesma saída em
todas as entradas de treino são intercambiáveis) corta **2,06×** o trabalho. É
a estrutura de grupo aparecendo como desperdício: `rot90⁴` é a identidade,
`espH∘espV` e `rot90²` são o mesmo elemento de D4.

---

## 4. O que foi testado e refutado

Cada item abaixo foi levado até o fim com controle. Repetir qualquer um sem
mudar o mecanismo é gastar sessão.

| hipótese | resultado |
|---|---|
| o gargalo é vocabulário | 24 → 80 operadores: **+1 tarefa** |
| o gargalo é profundidade | feixe chegou à 10; nada além da 4 |
| o gargalo é guiar a busca | 3 mecanismos, 3 nulos, conjuntos idênticos |
| dá para fabricar o prior de LLM | rede aprendeu o amostrador, não transferiu |
| faltam os priors de Chollet | objetidade + número + agência = 3 tarefas |
| deduzir a lei de um mundo emergente | **+0**, e o nulo tem endereço (§5.2) |
| diversidade no feixe ("não viciar") | **8 → 5**, piorou |
| quantificador superlativo sobre objeto | **+1** (previ 12–20) |

Sobre os dois últimos, que são recentes e informativos:

**Diversidade piorou.** Forçar o feixe a rejeitar candidatos parecidos
(Jaccard > 0,9) perdeu 3 tarefas e não ganhou nenhuma. Neste espaço, cobertura
alta **é** informativa — os quase-clones no topo do ranking são a região certa,
e rejeitá-los joga fora a variante de que a inferência do resíduo precisa. O
que parecia vício era foco.

**Superlativo é o quantificador errado.** `argmax` sobre objetos ("o maior", "o
único") transmite uma constante para a grade inteira. O ARC pede
**∀ sobre objetos com função por objeto** — "para cada objeto, pinte-o conforme
o seu tamanho". São coisas diferentes e eu medi a errada.

### 4.1 O quantificador certo, medido depois

Corrigido o desenho, com o mapa propriedade→cor **deduzido** dos exemplos:

```
L1 por célula                3/400
+ superlativo (argmax)       4/400   +1
+ universal por objeto       6/400   +2

37d3e8b2   [T ? porObj(n_buracos) : porObj(n_buracos)]
ae58858e   [T ? porObj(tam)       : porObj(tam)]
```

"Pinte cada objeto conforme quantos buracos ele tem." "Pinte cada objeto
conforme o seu tamanho." Teste `T` — a leitura sozinha explica a grade inteira.
Quantificação universal pura, mapa resolvido de dois ou três exemplos,
verificado exato numa grade nunca vista.

Nove propriedades por objeto, todas computáveis sem saber a tarefa: tamanho,
altura, largura, **número de buracos**, ranking por tamanho, quantos objetos
compartilham esse tamanho, quantos compartilham essa cor, a própria cor, e se é
o maior.

**Tudo isto ainda é UM passo.** A mesma linguagem por célula dá 3 num passo e 8
em dois. O teto do regime de um passo é o que limita estes números, não o
vocabulário — e é exatamente por isso que §5.1 diz que o gargalo é composição.

### 4.2 A cena como vocabulário: nulo absoluto

Chavear o mapeamento pela **forma** do objeto em vez de por uma propriedade
escalar — trazer o conteúdo da cena para dentro como chave — e canonizar a
forma pela órbita do grupo D4, de modo que a mesma peça girada caia na mesma
chave (verificado: peça L e peça L girada colapsam; quadrado não).

```
sem chave de forma (perobj)      6/400
+ forma crua, caixa, forma+cor   6/400   +0
+ canonização D4                 6/400   +0
```

**Chave mais fina não é poder expressivo — é capacidade de memorizar.** Forma
só ajuda quando dois objetos compartilham todas as propriedades escalares e
diferem no desenho; e quando isso acontece, o mapa fica com mais entradas e
mais chance de encontrar forma inédita no teste. O verificador exato recusa
premiar isso, corretamente.

**A família do mapa ajustado está saturada em 6/400.** O ganho não está no
dicionário.

Isso refuta a versão *indexadora* da ideia, não a versão *recombinante* — as
peças da cena como operandos de primeira classe, giráveis e componíveis, gerando
um espaço fechado sob composição. Essa continua não testada, e o nulo a torna
mais interessante: agora se sabe que o ganho não pode vir de consulta mais fina.

### 4.3 O quociente por grupo encolhe quando o vocabulário é bom

```
24 primitivas   CRU 16/400  7009 cand   QUOCIENTE 16/400  3399 cand   2,06×
38 primitivas   CRU 20/400 13766 cand   QUOCIENTE 20/400 12738 cand   1,08×
```

O colapso cai de 2,06× para 1,08× ao acrescentar as 14 primitivas extraídas do
mundo. Elas produzem comportamentos **genuinamente distintos** — o mundo
entregou vocabulário menos redundante que o escrito à mão, que contém os
geradores de D4 e por isso se repete.

**E o orçamento não era o gargalo:** o braço quociente terminou em 12.738 de um
teto de 14.424 e deu os mesmos 20. A profundidade 3 está **exaurida em 20/400**.

Ressalva retroativa: vários nulos desta sessão foram medidos "no mesmo
orçamento" num regime em que orçamento não era restrição ativa. O limite é real
mas parcial — na enumeração por produto cartesiano o teto **era** binding, e é
exatamente por isso que a busca em largura ganha +3 (17 → 20).

---

## 5. O que falta para AGI

### 5.0 O princípio que sobreviveu a quatro tentativas de refutá-lo

> **Tudo que moveu o número foi uma forma de recusar-se a refazer trabalho já
> determinado. Tudo que não moveu foi uma forma de fazer mais.**

| moveu | o que recusa refazer |
|---|---|
| deduzir em vez de sortear (~3.000×) | o operador já está determinado — não sorteie |
| prefixo compartilhado (+3) | o prefixo já foi computado — não recompute |
| cobertura parcial (3 → 8) | a estrutura já explica 85% — não a descarte |
| harness do Twin (+53 pontos) | a explicação não fechou — não aja |

| não moveu | o que faz mais |
|---|---|
| 24 → 80 operadores | mais vocabulário |
| feixe até profundidade 10 | mais profundidade |
| três guias de busca | mais orientação |
| diversidade no feixe (8 → 5) | mais variedade |
| superlativo sobre objeto (+1) | mais leituras |
| cena como chave (+0) | mais resolução de chave |

O princípio prevê que qualquer mecanismo **aditivo** dá entre 0 e +2, e que
mecanismo que **elimina trabalho determinado** dá fator, não incremento.

Foi formulado no meio da sessão como previsão falsificável, e sobreviveu a
quatro testes — três deles contra a minha própria aposta explícita:

| mecanismo | previsto | medido |
|---|---|---|
| quantificador superlativo | +9 a +17 | **+1** |
| quantificador universal | +2 a +5 | **+2** ✓ |
| colapso por quociente | 5–15× | **1,08–2,06×** |
| cena como chave | +2 a +4 | **+0** |

Três de quatro erradas, **todas na direção otimista**, e todas em mecanismos
aditivos. O padrão do erro é mais informativo que qualquer um dos números.

### 5.1 Nem vocabulário, nem matéria-prima da composição

```
B por célula, 2 passos          8/400    5 de 2 passos
E vocabulário cheio, 2 passos  11/400    5 de 2 passos     E - B = +3
```

O vocabulário quantificado vale +3 **com** composição e valia +3 **sem** —
somam, não multiplicam. E o número que importa é a segunda coluna: **zero
composições novas**. Todo o ganho veio de soluções de um passo.

Isso derruba o argumento mecânico que motivou o experimento: *composição falha
por falta de bons intermediários; dê estruturas de cobertura alta com resíduo
pequeno e as composições aparecem*. Foi dado exatamente isso — `porObj(tam)`
explica todos os objetos e erra só o fundo — e o número ficou em 5.

**Composição não está limitada por matéria-prima.** Sobram duas hipóteses:

1. **Dois passos é pouco.** O motor faz parcial → fecha exato. Solução que
   precise de três estágios parciais está fora por construção.
2. **O critério do SEGUNDO passo.** Exige-se ≥85% no primeiro e **100%** no
   segundo. Afrouxar o primeiro foi de 3 para 8; ninguém afrouxou o segundo.
   É a mesma lição do `relaxado.py` uma camada acima, e tem precedente medido.

A segunda é a que tem melhor razão evidência/custo.

### 5.1.1 O que sobra da leitura antiga

Sete das oito refutações são de vocabulário ou de guia. Nenhuma moveu o
número. O que moveu:

- não repetir trabalho já feito (+3)
- deixar estrutura imperfeita sobreviver (3 → 8)
- deduzir em vez de sortear (~3.000×)
- extrair leis de um mundo com a forma certa (+3)

Todos são sobre **como o espaço é percorrido**, não sobre o que há nele. E o
padrão sugere a hipótese incômoda: o ARC exige compor duas ou três operações, e
quase todo mecanismo testado aumenta a **largura** em vez da profundidade útil.

### 5.2 A linguagem das leis é míope

Diagnóstico com endereço, do experimento de dedução a partir de um simulador de
partículas: das 8 leis observadas, **1** foi deduzida exatamente.

`célula(i,j) = TESTE ? LEITURA_v : LEITURA_f` é **por célula**. Queda com
empilhamento é consequência de **alcance longo** — a partícula desce até achar
apoio, o que depende de tudo abaixo dela. Não é difícil de deduzir: é
**inexpressível**.

Isso explica retroativamente por que a dobra funcionou e a física quase não:
dobrar **é** por célula, `saída(i,j) = f(a[i][j], b[i][j])`. Gravidade só entrou
porque foi escrita à mão como função Python.

**Consequência prática:** o duto de aquisição de prior só adquire leis por
célula. Só vale minerar mundos cujas leis tenham essa forma — sobreposição,
oclusão, dobra, transparência, máscara. Fluido, propagação, empilhamento:
inexpressível, por mais bonito que seja o mundo.

### 5.3 Aritmética que ninguém quer olhar

Rendimento medido de aquisição de prior por mundo: física **0,11 tarefa por
lei**; dobra **0,40**. A 0,40, chegar a 240 tarefas pede ~600 leis.

Isso não é salto, é moagem. **Nenhum mecanismo medido até aqui escala por
adição.** Os únicos com forma multiplicativa são os de método de acesso, e o
maior deles (deduzir vs sortear) já foi colhido.

### 5.4 Módulos que não conversam

O motor tipado resolve 15, a inferência relaxada resolve 8, **interseção
exatamente zero**. Duas competências disjuntas rodando lado a lado sem trocar
nada. Fazer uma rodar dentro da outra nunca foi testado, e é o mecanismo com
melhor razão evidência/custo ainda na mesa.

### 5.5 Coordenada define expressividade, e nenhuma serve para tudo

Do mapa retina→córtex (Bressloff et al., *Phil Trans R Soc B* 2001): as form
constants de Klüver são os modos de instabilidade do V1 e correspondem aos
subgrupos axiais de E(2). O mapa é aproximadamente **log-polar**, e lá
**rotação é translação**.

Verificado no nosso código: com 64 ângulos × 8 raios, `rot90` vira
deslocamento angular **exato** de 16 bins, com ida-e-volta preservando 100% da
grade. A rede convolucional cartesiana dá **0,0%** em rotação.

O cérebro não *aprende* invariância a rotação — ele muda de coordenada. E paga:
perde a translação, e compra de volta movendo o olho. Se nenhuma coordenada é
universal, o desenho certo não é escolher a melhor, é rodar **várias em
paralelo** sobre a mesma cena.

### 5.6 O que a literatura externa diz que ainda não fizemos

Paper Twin (arXiv 2608.14490): o agente escreve um modelo de mundo executável
em tempo de teste, e o harness **bloqueia toda ação** até o programa reproduzir
todas as transições observadas.

```
modelo base direto            7,8
+ harness de código          61,1     ← +53 pontos, antes de qualquer modelo
+ modelo de mundo            93,3
custo: 2,6 bilhões de tokens, 91,4 horas
```

**A disciplina de verificação como restrição mecânica vale 53 pontos.** Não é
mais inteligência; é proibir a ação até a explicação fechar. É a coisa mais
barata e mais efetiva da literatura e o nosso sistema só a aplica no
verificador final, não como restrição durante a busca.

Achado deles que casa com o nosso: **dinâmica é fácil, objetivo é o difícil**.
E o ARC-AGI-1 *entrega* o objetivo pronto — por isso agência valeu ~nada aqui.
Medimos o prior de agência no benchmark que não o exercita.

---

## 5.7 A linha de OBJETOS, medida ponta a ponta (2026-09-07/08)

Diagnóstico: a representação `célula(i,j) = TESTE ? L_v : L_f` é uma função de
**grade para grade**. Não existe objeto persistente; até `porObj` faz o objeto
evaporar depois de calcular uma célula. Por isso "copie, aumente, transforme,
faça orbitar" era impossível **por construção**, e não por orçamento.

```
estado = Grid                      ->   estado = Cena(h, w, fundo, [Objeto])
```

| etapa | resultado |
|---|---|
| `cena` — objetos persistentes, enumerador | **3**/400 |
| `cenadeduz` — deduzir em vez de enumerar | 3/400, **300× mais rápido** (1s vs 315s) |
| `cenadeduz2` — guardas levantadas (criar, deformar, cortar) | **+0** |
| `segmenta` — 8 lentes de segmentação | **6**/400 |
| `ockham` — validação cruzada + Navalha | **4**/400, **precisão 100%** |

**As três demonstrações que importavam aterrissaram**, uma cada: gravidade num
operador (`cai(todos,dir)`), corte que muda a dimensão (`recortar(unicoTam)`),
e cadeia de dois passos em que o segundo opera sobre o que o primeiro fez.

### O achado metodológico, que vale mais que o placar

`segmenta.py` mediu o custo de "tente várias e escolha a que ajusta o treino":

```
ajustou o TREINO : 127/400
passou no TESTE  :   6/400        precisão 4,7%, sobreajuste 21×
```

Oito lentes × dez propriedades = **80 famílias de hipótese** contra 2-4 pares
de treino. Quase um terço do conjunto acha *alguma* explicação que fecha, e 95%
são coincidência. **Sem o verificador exato, o número reportado seria 31,75%.**

E os dois extremos, medidos no mesmo projeto com o mesmo verificador:

| motor | liberdade | precisão |
|---|---|---|
| dedução exata | 1 resposta determinada | **100%** |
| multi-lente | 80 famílias | **4,7%** |

**A precisão é função direta de quanta liberdade a busca teve.** Ajustar o dado
dado não é evidência quando o espaço de hipóteses é grande em relação ao dado.
É a mesma doença medida no xadrez com amostra fácil, onde a indução escolheu
`sem_saida` sozinha — 100% de treino, e confundindo ganhar com empatar.

### O conserto, e a primeira parede que não é minha

`ockham.py`: validação cruzada deixe-um-de-fora **nos próprios pares de
treino** (o teste nunca é tocado), mais Navalha de Ockham com ordem de
complexidade declarada antes de rodar.

```
127 ajustam  ->  4 sobrevivem  ->  4 passam no teste
precisão 4,7%  ->  100,0%
```

Zero falso positivo. Custo: 6 → 4 tarefas, e 20 tarefas excluídas por
construção (< 3 pares de treino).

**AFIRMAÇÃO RETRATADA — e a retratação é o resultado.** Eu escrevi aqui que o
gargalo tinha passado a ser o número de exemplos, propriedade do benchmark e
não minha. `quantosex.py` mediu e é falso.

Para cada tarefa real, quantas das 80 famílias reproduzem os K primeiros pares:

```
 K   tarefas com     mediana de   famílias   PRECISÃO
     alguma família   famílias     certas    certa/viva
 1        322             2           7        0,8%
 2        204             2          14        3,5%
 3        123             1          16        7,5%
 4         35             1           8       12,9%
 5         11             1           1        5,9%
 6          1             1           0        0,0%
```

O número de tarefas com QUALQUER família viva desaba: 322 → 35 → 1. Mais
exemplos não SELECIONAM entre as famílias — **matam todas**. E a mediana já é
1-2 com UM exemplo: nunca houve multidão de candidatos esperando filtro.

Se o ARC desse oito exemplos, este motor resolveria MENOS.

A parede é a mesma das outras cinco: **a família certa não está no conjunto.**
Exemplo elimina hipótese errada; não inventa a certa. As seis paredes eram
minhas.

E 4 com precisão 100% vale mais que 6 com 4,7% para o objetivo declarado
(um sistema utilizável fora do benchmark): cobertura sem precisão é ruído com
placar.

## 5.8 Analogia entre tarefas: o gerador existe e está faminto

`analogia.py`. Esquemas abstraídos das soluções do conjunto de TREINO,
instanciados no de AVALIAÇÃO. Controle: esquemas de triplas aleatórias, mesma
quantidade e mesma distribuição de forma.

```
ESQUEMA ALEATÓRIO   0/400   9736 candidatos/tarefa
ESQUEMA DO TREINO   4/400   9548 candidatos/tarefa
```

Controle em **zero absoluto**, gastando mais orçamento. A transferência de
estrutura entre tarefas é real.

Ressalvas: o portão A0 falhou (14 soluções de treino, mínimo pré-registrado
15), então é **exploratório**; e 3 das 4 já eram resolvidas por outro motor —
o ganho de cobertura é **+1**.

**O círculo:** para ter biblioteca é preciso ter soluções; para ter soluções é
preciso ter biblioteca. 12 esquemas distintos para 14 soluções — reúso 1,17×.
Terceira medida independente do mesmo bloqueio de volume.

## 5.9 O ciclo fechado: descobrir o objetivo e agir com ele

`ciclo.py`, 12 pares de labirintos. Mundo A descobre o objetivo por eliminação;
mundo B mede ações até a primeira vitória.

```
braço              mediana   falhou
CEGO                   69     0/12
DESCOB.parcial         75    11/12
DESCOB.binário         66     1/12
VERDADEIRO             68     0/12
```

O objetivo descoberto (`bolso>=3 & saidas==1`, correto) usado como alvo binário
dá **0,97×** do objetivo verdadeiro. **O ciclo fecha.**

Duas ressalvas que impedem a leitura otimista: o ganho sobre o cego é 69 → 66,
porque o mundo é fácil e o teto está no chão; e **crédito parcial sobre
objetivo falhou em 11 de 12**. O conjunto barato (`saidas==1`, vale em qualquer
beco) captura o agente e mata de fome o caro.

**Crédito parcial é seguro para INFERÊNCIA e perigoso para CONTROLE.** Ranquear
hipóteses é passivo. Dirigir comportamento tem custo de oportunidade: toda ação
gasta no conjunto fácil é ação não gasta explorando. Isso corrige o que eu havia
afirmado antes de medir.

## 5.10 Sem limiar: remover os meus números mágicos

`residuo.py`. Melhor-primeiro sobre o conjunto de células ainda não explicadas,
sem limiar de cobertura em lugar nenhum, profundidade N.

```
por célula   5/400   passos {1: 1, 2: 3, 4: 1}
```

Contra 8/400 do motor com limiar fixo em 0,85. **Piorou** — mas os orçamentos
não são comparáveis: o antigo varria ~48.000 pares por tarefa, este teve 400
expansões, cada uma uma dedução completa.

Resolveu uma tarefa em **quatro passos**, que o motor de dois passos não podia
sequer tentar.

Leitura: remover o limiar transfere o custo do meu julgamento para a busca, e a
busca não tem orçamento para pagar. O 0,85 não era um chute — era a compressão
de uma varredura de curva que eu fiz uma vez. Tirar isso faz o sistema
**refazer trabalho já determinado**, que é exatamente o que o princípio da §5.0
diz para não fazer.

## 5.11 O padrão de CINCO níveis

Toda vez que um nível é consertado, a parede reaparece **um nível acima**,
sempre onde uma escolha foi imposta antes de o sistema começar a pensar.

| nível | conserto | parede que apareceu |
|---|---|---|
| busca | quociente, prefixo, resíduo | fator em velocidade, zero em cobertura |
| linguagem por célula | leituras quantificadas | alcance longo é inexpressível |
| objetivo | quantificador | resolvido — mas o mundo era fácil |
| objetos | dedução, guardas levantadas | o que **conta** como objeto |
| segmentação | 8 lentes | como **escolher** a lente sem ajuste |
| seleção | validação cruzada | ~~número de exemplos~~ → **retratado**: mais exemplos matam as famílias em vez de selecioná-las (`quantosex.py`) |

**Toda abstração entregue vira o teto.** O sistema opera dentro dela com
competência e para na borda desenhada. E a sexta parede, que eu havia concedido
ao benchmark, foi medida e devolvida: **as seis eram minhas.**

---

## 5.12 Experimentos de representacao

Detalhes de uma gravacao pessoal foram removidos por privacidade.

## 5.13 O sistema imunológico, testado contra três venenos

| ampliação | ajustes espúrios criados | vazaram |
|---|---|---|
| 8 modificadores de operação | +8 | **0** |
| gaveta de anomalia | +45 | **0** |
| poda (remove, não adiciona) | −12 | 0 verdadeiros perdidos |

**53 falsos positivos gerados, 53 interceptados, zero verdadeiros perdidos.**

E `ladrilho.py` produziu o princípio de calibragem:

| motor | espaço | precisão SEM portão |
|---|---|---|
| multi-lente | 80 famílias | **4,7%** |
| ladrilho | 48 hipóteses exclusivas | **100%** |

**A força da verificação deve escalar com a liberdade da geração.** Onde a
busca é livre, ajustar não é evidência. Onde o espaço é pequeno e exclusivo,
ajustar já é evidência e o portão só custa verdadeiro positivo — ele rejeitou
`00576224` por ela ter só 2 pares de treino.

## 5.14 Analogia generativa: recombinar DESTRÓI

```
restritivo   26 esquemas        7/400
generativo   1092 triplas       2/400
```

Recombinar as peças entre esquemas (teste de A + leituras de B) produz triplas
que nenhuma tarefa exibiu — e perde 5 tarefas. **O esquema vale como unidade;
as partes não são independentes.** Quinta confirmação de que ampliar custa mais
do que rende.

## 5.15 Acumulação: não dá para medir com 1% de acerto

`acumula.py`, 4 passadas (3 ordens aleatórias + 1 fácil-primeiro), biblioteca
começa vazia e cresce a cada solução:

```
Q1 = 1,3   Q4 = 1,3   biblioteca final: 3 esquemas   via-biblioteca: 1 em 4 passadas
```

**A0 falhou.** Não é evidência de que não acumula — é falta de matéria. Para
medir acumulação dentro de uma passada é preciso taxa de acerto suficiente para
construir biblioteca enquanto anda. A nossa é **1%**.

Quarta confirmação independente do bloqueio de VOLUME: 21 ocorrências de
fragmento · 14 soluções colhidas · reúso 1,42 · 1% de acerto.

**Consequência para currículo:** não é "currículo não funciona" — é que
currículo é um mecanismo de acumulação, e acumulação precisa de acertos
frequentes. Com 1%, o aluno erra 99 de 100 e não sobra nada para a aula
seguinte. Todo mecanismo que depende de aprender com os próprios acertos —
currículo, biblioteca, analogia, automodificação — esbarra na mesma escassez.

Os únicos que funcionaram hoje **não dependem de histórico**: mudar a
representação, e verificar melhor.

---

## 5.16 O teste da dica: quanto vale um professor perfeito

`dica.py`. Para cada tarefa, testam-se TODAS as 80 combinações (8 lentes × 10
propriedades) e pergunta-se se ALGUMA produz regra que reproduz o treino E
acerta o teste. **Isso usa o teste para medir — é diagnóstico, não placar, e
nunca entra na união.**

```
ORÁCULO (6,3 bits: lente + propriedade certas):   7/400
DICA de 3 bits (só a lente certa)             :   6/400
ZERO bits (o sistema sozinho)                 :   4/400

ganho 0 → 3 bits: +2      ganho 3 → 6,3 bits: +1
inalcançável mesmo com dica perfeita: 393
```

**Um professor perfeito compra três tarefas.** E das 7 alcançáveis, cinco têm
quatro ou seis combinações certas — quando a resposta está no espaço, é achável
de vários jeitos. Não é agulha no palheiro: quase nunca há agulha.

Isso fecha a linha de professor/currículo **para esta família**: o gargalo não é
seleção nem direção, é expressividade. E custou 154 segundos, matando um
sistema de cinco estágios que teria levado dias.

## 5.17 Censo estrutural: o que as 373 exigem

`inventario.py`, assinaturas válidas em TODOS os pares de treino:

| assinatura | não resolvidas | resolvidas |
|---|---|---|
| **formas_novas** | **199** | 14 |
| mais_objetos | 149 | 15 |
| formas_iguais | 40 | 7 |
| dim_ladrilho | 27 | 3 |
| dim_subgrade | 17 | 1 |

**53% das não resolvidas produzem formas que não existiam na entrada.** Onde as
formas são preservadas resolvemos 15%; onde há formas novas, 6,6%.

Todo o motor de objetos é **transformacional** — pareia objeto de entrada com
objeto de saída e lê o delta; `deltas` devolve None quando um objeto de saída
não tem origem. **Criação é a maioria e nada no projeto desenha.**

## 5.18 Emissão: desenhar em vez de transformar

`emissao.py`, quatro famílias com parâmetros lidos do diff (raio, preencher,
conectar, contorno):

```
ajustou o TREINO: 2/400    passou no teste: 2/400    precisão 100%
```

Só 2 tarefas sequer ajustam. **+2 inéditas** (070dd51e, aa18de87), união 27 → 29.

`emissao2.py` condicionou a emissão por seis predicados de fonte (cor, borda,
maior, menor, tamanho único, isolado): **zero novas**, e o ajuste ao treino
ficou em 2. As condições do ARC não são predicados de célula.

## 5.19 Relacional: o sinal mais limpo de exaustão

`relacional.py` acrescentou 8 propriedades que COMPARAM objetos entre si
(n_mesma_forma, forma_única, dentro_de, contém, toca_outro, n_vizinhos,
alinhado, cor_maioria), indo de 80 para 144 hipóteses:

```
ajustou o TREINO: 127/400   ← EXATAMENTE o mesmo que sem elas
passou no TESTE :   4/400   ← idêntico
resolvidas por propriedade relacional: 0
```

**Nem tarefa nova, nem falso positivo novo.** Modificadores criaram +8
espúrios, a gaveta +45; estas criaram **zero**. Elas induzem as mesmas
partições que as intrínsecas — são redundantes nas cenas do ARC.

Vocabulário fixo de propriedades está **exaurido**. O que seria genuinamente
relacional é dependente da tarefa ("o objeto que difere no aspecto que importa
aqui") e não cabe numa lista escrita de antemão.

## 5.20 Acumulação: não mensurável com 1% de acerto

`acumula.py`, 4 passadas, biblioteca começa vazia e cresce a cada solução:

```
Q1 = 1,3   Q4 = 1,3   biblioteca final: 3 esquemas   via-biblioteca: 1 em 4
```

A0 falhou por falta de matéria. Currículo é mecanismo de acumulação e
acumulação precisa de acertos frequentes; com 1% o aluno erra 99 de 100.

## 5.21 O placar de mecanismos, dezesseis tentativas

| categoria | tentativas | ganho |
|---|---|---|
| mudar a **representação** | 4 | +3, +3, +3, +2 |
| mudar a **busca ou seleção** | 12 | 0 a +1 |

E o placar de previsões: **23 feitas, 7 certas, 16 erradas — todas otimistas.**
As construções que entregaram foram as que vieram de MEDIDA, não de palpite:
ladrilho (da gravação do usuário) e emissão (do censo).

---

## 5.22 As saídas do ARC não formam um banco

`saida.py` mediu a premissa antes de construir sobre ela:

```
NENHUMA assinatura canônica :  291 tarefas (72,8%)
poucas_cores                :   69 (17,2%)
simétricas (h/v/d)          :  ~77 somadas
periódica / uniforme        :   14 / 6

saídas DISTINTAS: 1347 de 1363
saídas repetidas entre tarefas: 4
```

Praticamente **toda saída do ARC é única**. Não existe banco de formas
canônicas a reconhecer — o benchmark foi construído para que decorar saídas
falhe.

**Correção precisa à proposta:** gerar a figura final é a direção certa, mas
não há figura a *reconhecer* — há figura a **construir**, e cada uma é nova. É
a diferença entre reconhecer uma palavra e formar uma frase.

Segunda vez no dia que medir a premissa matou um plano antes de custar dias
(a primeira foi `dica.py` e o sistema de professor).

## 5.23 O compositor: montar a saída a partir de leituras

`compositor.py`. A entrada deixa de ser substrato a modificar e vira **fonte de
parâmetros**; a saída é uma tela nova cujo tamanho e conteúdo são lidos.

```
ajustou o TREINO : 4/400
passou no TESTE  : 4/400     precisão 100,0%
famílias: recorte 2 · tela 1 · repetir 1
INÉDITAS: 3 (3194b014, ccd554ac, cd3c21df)
```

Precisão perfeita sem precisar do portão. **União 29 → 32/400 (8,0%).**

Três famílias contribuíram uma cada: recortar o objeto selecionado por
propriedade, tela uniforme com dimensões lidas da entrada, e repetição por
contagem. Nenhuma dessas grades existia — foram construídas.

## 5.24 O placar final: dezessete tentativas

| categoria | tentativas | ganho |
|---|---|---|
| mudar a **representação** | 5 | +3, +3, +3, +2, **+3** |
| mudar a **busca ou seleção** | 12 | 0 a +1 |

A separação nunca falhou em dezessete tentativas. **Mudar o que o sistema pode
dizer rende 2-3. Mudar como ele procura rende 0-1.**

E o placar de previsões: **26 feitas, 10 certas.** As 16 erradas foram todas
otimistas — e concentradas na primeira metade, antes de eu passar a medir a
premissa antes de construir. As três últimas construções vieram de medida (o
censo apontou emissão, a gravação do usuário apontou ladrilho, o censo de
saídas apontou compositor) e as três entregaram.

---

## 6. Estado do número

```
busca monolítica, produto cartesiano        14/400   3,50%
busca monolítica, prefixo compartilhado     16/400   4,00%
  + 14 leis extraídas do mundo da dobra     20/400   5,00%
motor tipado, 5 famílias de prior           15/400   3,75%
inferência relaxada de 2 passos              8/400   2,00%

  + lente de ladrilho (3 inéditas)          3/400
  + analogia com biblioteca de 26 esquemas  7/400

  + emissão (conectar, 2 inéditas)          2/400

  + compositor (3 inéditas)                 4/400

UNIÃO verificada                            32/400   8,00%
```

Contexto externo: GPT-6 Astra faz 97,5% no ARC-AGI-1 a US$ 18–26 mil por
execução. A comparação não é de placar — é de método. O que este projeto tem
que aquele número não tem é **saber por quê**: cada ponto aqui tem controle
pareado, e cada não-ponto tem diagnóstico.

---

## 7. Se eu tivesse uma sessão a mais

Em ordem de razão evidência/custo, com o que os dados de hoje justificam:

1. **Verificação como restrição durante a busca**, não só no fim. Vale 53
   pontos na literatura e é o único item barato de tamanho grande.
2. **Fazer os módulos disjuntos conversarem** — interseção zero medida entre
   dois motores é evidência direta de ganho disponível.
3. ~~**∀ sobre objetos com função por objeto**~~ — **feito**, vale +2 num
   passo (§4.1). O que falta é rodá-lo com composição de dois passos e
   cobertura parcial, que é onde a linguagem por célula triplica (3 → 8).
4. **Alcance na linguagem das leis** — leituras que digam "a primeira célula
   ocupada abaixo", "a borda do objeto que me contém". É o mesmo degrau que no
   objetivo valeu 49,6% → 99,3%.
5. **Várias coordenadas em paralelo**, já que cada uma torna uma família de leis
   gratuita e outra impossível.

O que **não** fazer, porque já foi medido: mais primitivas, mais profundidade,
mais guia de busca, fabricar prior de LLM, e diversidade forçada no feixe.

---

## 5.25 · Três motores sobre o mesmo censo, todos zero

Esta seção registra a cadeia mais longa de fracasso da sessão, porque o padrão
dela vale mais que qualquer um dos três resultados.

### O que aconteceu

**`fluido.py` — F0, 0/400.** Escrito a partir do relato falado do usuário
resolvendo o ARC: cinco dos catorze relatos falavam de líquido ("como eu coloco
água num copo", "lava caindo tipo no Minecraft"). Quatro variantes: `retem`,
`molha`, `interior`, `cercado`. Nenhuma ajustou o treino em nenhuma tarefa.

O traço na `184a9768`, que o censo tinha marcado como `novas_enchem`:

```
par 0: mudou=96  de_fundo=45  de_ocupada=51
ocupada->: TODAS para 0
```

Não é líquido enchendo recipiente. As 51 células ocupadas que mudam **viram
fundo**: a peça some de um lugar e aparece em outro. O censo tinha casado pelo
motivo errado — buraco de quebra-cabeça também é região cercada.

**`encaixe.py` — E0, 0/400, duas vezes.** O traço abriu um censo novo, com
assinatura fechada: *o multiconjunto de formas removidas é igual ao de formas
pintadas*. Deu **14 tarefas, 13 não resolvidas**, com as transformações juntas
(id 195, r90 9, r180 3). Era o maior alvo medido da sessão inteira.

A versão 1 casava um buraco com uma peça. E0 disparou. O traço mostrou por quê:
na `184a9768` os buracos são `[1, 4, 6, 8, 262]` e as peças `[1×12, 4×3, 6, 89]`
— o buraco de 8 é fechado por **duas** peças. Buraco não é peça virada do
avesso; buraco é área a cobrir. A versão 2 ladrilha por cobertura exata. E0
disparou de novo, e aí ficou claro que o erro não era de implementação:

```
f3e62deb: peças = [8]   buracos = [1, 91]
```

O destino é fundo aberto. Não existe buraco. A assinatura "conserva forma" não
é uma família — é um saco de gatos: quebra-cabeça, escorregar até a parede,
linha que varre, espelhamento. A `dd2401ed` nem é movimento: é uma linha de `5`
que anda e converte em `1` todo `2` que ela cruza.

**`atracao.py` — A3, 0/400.** Terceira leitura, agora sobre a pergunta certa:
*para onde* a peça vai. Medido em 265 peças movidas:

| | |
|---|---|
| escorrega até travar | 80/265 = **30,2 %** |
| tarefas com 100 % | `e1d2900e` (21/21), `f3e62deb` (6/6) |
| parciais | `1c56ad9f` 32/44, `ecaa0ec1` 12/16, `9b4c17c4` 7/9 |
| direção global que sirva | **nenhuma** — em `e1d2900e` sai 5/5/5/6 nas quatro |

A direção é lida por peça, e a leitura é deduzível sem dica externa: é a única
direção em que o movimento termina **batendo num objeto** em vez de na parede.
O motor chegou a **uma peça** de reproduzir a `e1d2900e`:

```
4dir perto fixos -> [2, 0, 0]      (dois pares exatos, um errando 1 peça)
```

A peça que resiste é `(23,13)`. Ela está alinhada com o bloco, com caminho
livre, a 7,5 de distância — e não se move. A `(15,18)` tem **a geometria
idêntica**, ponto por ponto, e se move. A condição que separa as duas não está
na distância, no alinhamento, na obstrução, nem no número de blocos alinhados.
Não consegui lê-la.

### O que isto mede

Três motores, três leituras do mesmo censo, **zero tarefas**. E não foi a busca
que falhou em nenhum dos três: foi eu lendo mal o mecanismo, três vezes, sobre
dados que eu mesmo tinha medido.

Isso corrige o refino que a sessão vinha usando. Não é verdade que "o que veio
de medida entregou". O correto é mais estreito e mais duro:

> O que veio de medida **do mecanismo exato** entregou. O que veio de medida de
> uma **assinatura** — uma propriedade do par entrada/saída — deu zero, mesmo
> com a assinatura fechada, mesmo com 13 tarefas dentro dela.

Assinatura não é mecanismo. `novas_enchem` casou uma tarefa de quebra-cabeça.
"Conserva forma" casou cinco famílias diferentes. Uma assinatura diz que
*alguma coisa* é invariante; não diz qual processo produz a invariância, e é o
processo que precisa ser escrito.

E o `e1d2900e` fecha o argumento com o dado mais fino da sessão: eu tinha o
mecanismo certo, medido em 21/21 peças, implementado, e ele reproduziu dois
terços dos pares exatamente. Faltou **uma condição de uma peça** — e o custo
disso não foi ficar 95 % perto. Foi **zero**. A reprodução exata do treino não
dá crédito parcial, e é essa a razão pela qual 30,2 % de cobertura mecânica
vira 0/400.

União depois desta seção: **33/400**, inalterada.

Previsões: **35, dez certas**. As três desta seção — F2, E2, A3 — deram F3,
E3, A3. A primeira vez em trinta e cinco que eu previ o meu próprio piso, e foi
a única das três que acertou.

---

## 6 · O núcleo de descrição: o portão binário era o gargalo

Esta seção é o resultado mais importante do projeto até aqui, e ela veio de
pesquisar o estado da arte em vez de escrever o motor nº 15.

### O que a literatura diz (setembro de 2026)

| trabalho | mecanismo | resultado |
|---|---|---|
| **CompressARC** (Liao & Gu, prêmio ARC Prize 2025) | MDL puro no tempo de inferência | **20% eval**, 76K params, zero pré-treino, zero dataset |
| **Ferré, ARC-MDL / MADIL** | MDL simbólico por objeto | 5 → **29/400**, 30 s/tarefa |
| **Li & Ellis, indução × transdução** | os dois resolvem conjuntos **disjuntos** | todo topo de placar combina os dois |
| **ARC Prize 2025, relatório técnico** | tema declarado do ano | **"the refinement loop"** |

E a complementaridade com o CompressARC está na lista de falhas dos próprios
autores: eles declaram que a arquitetura **não faz** contagem, conectividade,
operações repetidas, transformações grandes nem mapa de cor — item por item o
que os nossos motores fazem.

### O diagnóstico

O sinal de feedback deste projeto era **binário**: reprodução exata ou nada.

Medido no `atracao.py`: mecanismo certo, 21/21 peças, dois dos três pares
reproduzidos exatamente, **uma peça errada**, resultado **zero**. 30,2 % de
cobertura mecânica virou 0/400.

Sem crédito parcial não há gradiente; sem gradiente não há refinamento; sem
refinamento não há composição. A "barreira de composição" medida durante semanas
não era do compositor — era **consequência do portão binário**.

### O mecanismo

Cada motor vira **codec**: uma função *parcial* que reivindica algumas células
da saída e se cala sobre as outras.

```
custo(pilha) = Σ bits_do_modelo(codec_i)              <- pago UMA vez
             + Σ_pares bits_literais(células erradas)
```

O modelo é pago **uma vez** e vale para todos os pares — é isso, e só isso, que
separa regra de memória. Uma exceção literal custa `log2(H·W) + log2(10)` bits,
então um codec só entra se economizar mais do que custa. Ockham deixa de ser
ordenação heurística e vira aritmética.

### Os resultados

**D1 · composição (a tese).**

```
070dd51e  conecta + copia
0a2355a6  cor_por_prop + copia
140c817e  cor_por_prop + cruz + contorno + copia
37d3e8b2  cor_por_prop + copia
aa18de87  conecta + copia
aa4ec2a5  preenche + cor_por_prop + contorno + copia
ae58858e  cor_por_prop + copia
```

Sete tarefas resolvidas por pilhas que **nenhum codec resolve sozinho**.
Impossível na arquitetura anterior por construção.

**C1 · a curva é convexa.** Rodando o núcleo com subconjuntos de tamanho *k*
dos codecs (61 subconjuntos × 270 tarefas):

| k | média | pilhas 2+ | ganho do k-ésimo |
|---|---|---|---|
| 1 | 0,00 | 0,00 | +0,00 |
| 2 | 0,00 | 0,00 | +0,00 |
| 3 | 0,50 | 0,50 | +0,50 |
| 5 | 2,00 | 2,00 | +1,50 |
| 8 | 2,50 | 2,50 | +0,50 |
| 11 | 3,00 | 3,00 | +0,50 |

Ganho médio da primeira metade **0,100**; da segunda metade **0,500**. Cinco
vezes maior.

Dois fatos mais fortes que a convexidade:

1. A coluna `pilhas 2+` é **igual ao total em todas as linhas**. Nenhuma tarefa
   é resolvida por um codec sozinho. **100 % do rendimento é composicional.**
2. Com k ≤ 2 o rendimento é **zero**. O mecanismo não existe até haver peças
   suficientes — que é a assinatura de composição, não de acumulação.

Ressalva: números absolutos pequenos, 6 amostras por *k*, e o vale em k=6 é
ruído amostral. O que é robusto é o zero em k≤2 e o 100 % composicional.

**Confirmação fora da amostra.** Passando de 11 para 18 codecs:

```
11 codecs ->  7 fecharam o treino,  3 no teste,  1 inédita, pilhas de até 2
18 codecs -> 14 fecharam o treino, 10 no teste,  4 inéditas, pilhas de até 4
```

Precisão 10/14 = **71 %**, contra 4,7 % do regime de múltiplas lentes.

### O que isto muda na estratégia

A regra do projeto era "mudança de representação vale +3, mudança de busca vale
0". Ela continua verdadeira e agora tem um sucessor:

> Motor grande e binário rende linear. Codec pequeno e parcial rende convexo,
> porque cada codec novo entra em pilha com **todos** os anteriores.

Parar de escrever motores e escrever muitos codecs pequenos.

União: **37/400**. Previsões: **38, treze certas** (D1, C1 e a confirmação fora
da amostra) — a primeira vez que três previsões seguidas acertam.

### O que falta, medido

O núcleo só trata as **270 tarefas de dimensões iguais**. As outras 130 precisam
de um codec de *forma* (que decide as dimensões da saída) — é o próximo degrau e
é um multiplicador, não uma soma.

---

## 6.1 · O codec de forma: rebasear em vez de escrever motor novo

O núcleo de descrição só tratava as **270 tarefas de dimensões iguais**. As 130
restantes estavam fechadas porque a pilha não sabia o tamanho da saída.

A saída burra seria escrever codecs novos para dimensões diferentes. A saída
certa é **rebasear**: um modelo de forma lê a entrada e devolve uma *grade base*
que já tem as dimensões da saída. A tarefa vira de dimensões iguais, e os 18
codecs existentes operam em cima **sem nenhuma alteração**.

```
ladrilho 3x3              base = a entrada repetida 3x3   -> copia
recorte do maior objeto   base = o recorte                -> copia, contorno
dois painéis com XOR      base = o XOR dos dois           -> mapa_cor
```

Nove famílias: `igual`, `escala(k)`, `reduz(k)`, `ladrilho(p,q,D4 por ladrilho)`,
`recorte(seleção)`, `dentro`, `painel(seleção)`, `painel_op(e/ou/xor/nem/dif)`,
`constante`.

A forma paga bits como qualquer outro modelo. `igual` custa 1 bit, então só
perde para uma forma cara quando ela economiza de verdade — e é isso, e não uma
regra especial, que protege as tarefas que já passavam.

### Resultado

```
sem forma:  10/400   formas: {igual: 10}
com forma:  20/400   formas: {igual: 10, ladrilho: 7, escala: 1,
                              painel: 1, recorte: 1}

F0 . as 10 anteriores continuam todas passando, com `igual` escolhida
F1 . 11 INÉDITAS (o portão pedia >= 8)
```

**União 33 → 44/400 (11,0 %).** O maior salto isolado da história do projeto —
maior que os +3 de todas as mudanças de representação anteriores somadas duas a
duas.

E o `ladrilho`, que era um motor inteiro de 48 hipóteses exclusivas resolvendo 3
tarefas, virou **um modelo de forma** e agora resolve 7 — porque os codecs
trabalham em cima dele em vez de ele ter que acertar sozinho.

### O que ainda não aconteceu, e é o próximo degrau

As duas mecânicas **não se cruzaram**. Os 10 casos de forma não-trivial usam
pilhas de tamanho 1; as pilhas de 2+ são todas de forma `igual`:

```
tamanho das pilhas: {1: 13, 2: 5, 4: 2}
por forma != igual: 10, todas com pilha de tamanho 1
```

Ou seja, o ganho da forma foi **aditivo**, não multiplicativo — ainda. O
multiplicador previsto (N formas × M codecs) só aparece quando uma tarefa
precisar de base rebaseada *e* de pilha profunda. Isso é o alvo da próxima
rodada, e é mensurável: basta contar tarefas com forma ≠ `igual` **e** pilha ≥ 2.

Previsões: **39, catorze certas** (F0 e F1 acertaram).

---

## 7 · O ciclo completo: gramática, linguagem, biblioteca, transdução

### O placar

**União 33 → 51/400 (12,75 %)** nesta sessão. Por degrau:

| degrau | ganho | tipo |
|---|---|---|
| núcleo de MDL | +4 | representação |
| codec de forma (rebasear) | +11 | representação |
| codec de condição aprendida | +2 | representação |
| linguagem de codecs | +1 | representação |
| gramática de testes | +3 | representação |
| pintores que propagam | +1 | representação |
| feixe, HDC, 4 ordens de conjunção, biblioteca | **0** | busca |

**Quinze mudanças de busca com resultado nulo. Seis de representação somando +22.**

### O que o sistema fez sem eu escrever o conteúdo

```
gramática de testes   gera    desl(N, ocupada)
linguagem de codecs   monta   PINTAR(...) COM ...
anti-unificação       extrai  o esqueleto comum
MDL                   julga   por bits
```

Escrevi os combinadores, os pintores e a função de custo. Tudo entre eles saiu do
sistema. E ele compôs dois codecs inventados entre si — `programa + programa +
copia` — sobre o resíduo que o primeiro deixou.

A abstração de topo que ele formou sozinho: `PINTAR(fundo & ?) COM família prop`
— escoamento para o espaço vazio. É a família que eu tentei escrever à mão duas
vezes: `fluido.py` deu F0 (0/400) e `trajetoria.py` deu T3 (zero).

### A biblioteca não transfere — resultado negativo limpo

```
biblioteca das 800 tarefas (vantagem máxima)      +0
biblioteca só do treino, medida na avaliação      +0
placebo pareado sorteado                          +0
holdout, três braços                        30, 30, 30
```

E ela **comprime**: +204 bits sobre 1191 (17,1 %) pelo critério do Stitch. Ou
seja, compressividade e transferência são coisas diferentes, e a primeira não
implica a segunda. A literatura costuma tratá-las juntas.

O motivo mais provável está na própria medida: as abstrações de topo têm
**esqueleto vazio**. O que se repete entre tarefas são os *pintores*, não os
*seletores*. E os pintores já estão disponíveis em toda tarefa por construção —
então a biblioteca redescobre algo que já estava lá. É testável: se os seletores
começarem a se repetir, a biblioteca passa a valer.

### Transdução: disjunta, mas n = 1

1/400, precisão 100 %, e a indução não resolve essa tarefa. Compatível com a
disjunção que Li & Ellis medem, e sem força estatística nenhuma.

### Uma regressão não resolvida

O núcleo caiu de **32 para 30** entre o braço dos pintores e o estado atual.
As duas perdidas são `903d1b4a` e `c7d4e6ad`, ambas com `programa` na pilha;
elas viraram `programa + copia` com resíduo 24 e 62. Reverter a semeadura da
biblioteca **não** as trouxe de volta, e eu não isolei a causa. A união não é
afetada porque as duas são cobertas por outros motores — é regressão de
cobertura do núcleo, não do projeto. Fica registrada por honestidade e como
dívida.

### Previsões

**50 no total, 18 certas.** As erradas seguem quase todas otimistas, com duas
exceções nesta sessão: a gramática (previ +0 a +1, veio +3) e o holdout (previ
H2 e acertei na versão mais forte do teste).

## 8 · A régua estava torta, e três domínios fora da grade (2026-09-10)

### A régua de bits

Uma auditoria pedida a agentes externos achou dois defeitos no `custo`, e ambos
eram meus:

- a exceção de **todos** os pares era cobrada com as dimensões de `treino[0]`.
  Em 64 % dos pares isso errava o preço por um fator entre 0,108 e 3,907;
- a identidade de cada codec na pilha era **grátis**. Agora custa
  `log2(|CODECS|)` por vaga.

E três bugs de busca, achados no caminho:

- `montar` lia `melhor[3]` antes de checar `melhor is None` — todo modelo de
  forma era descartado (o portão Z0 pegou);
- `TETO_CEL = 0`, documentado como "sem teto", virava o teto **mais restritivo**
  (40 avaliações) por causa de `max(0 // x, 40)`;
- `propor_k` custava 6.000× o tempo e rendia zero — desligado, com o motivo no
  código.

O cache de transformações foi limitado a 64 entradas (um processo chegou a 3 GB).

**Remedição, régua corrigida:**

```
núcleo sozinho      30 -> 40/400
inéditas (fora da união anterior)   5   45737921 48131b3c ce039d91 d37a1ef5 f0df5ff0
união               54 -> 59/400  (14,75 %)
pilhas de 2+ codecs 25 das 40
```

Atribuição honesta: entre as duas medições entraram a régua, o teto por célula,
as leituras (`raio_de`) e `RelacaoBinaria` juntos. O +10 do núcleo é do pacote,
não da régua isolada — não rodei a ablação. As duas regressões antigas
(`903d1b4a`, `c7d4e6ad`) continuam fora do núcleo; seguem cobertas por outros
motores.

### Agentes

Onze agentes com vieses diferentes (cético, cientista, engenheiro, LLM consciente
de si, jogador de Factorio, bitcoiner...) geraram ~35 propostas. As que viraram
teste estão no placar; as outras têm controle definido e ficaram como dívida.
**Nenhuma das 35 tocou agência** — todas melhoram a indução a partir de pares
dados. Isso é um viés da pergunta, não da resposta.

### Língua: o mesmo MDL, sem adaptação, em fala real

`lingua.py`: segmentação à de Marcken sobre a transcrição da gravação do
usuário (2.070 palavras, português falado), com o texto embaralhado como nulo.
O veredito mudou três vezes em cinco rodadas; o final:

```
treino  200 -> léxico  34   F1 held-out 0,409
treino 1000 -> léxico 113               0,489
treino 1552 -> léxico 160               0,557   (trivial 0,377)
```

Plano até ~700 palavras (memoriza), depois sobe contínuo (generaliza).
**Limitado por dado, com limiar** — o custo fixo de cada entrada do léxico só se
paga quando a palavra se repete. Segmentar não é entender; é evidência de que a
função objetivo não é truque de grade.

### Descoberta: dar a física velha e um mundo com a anomalia

A pergunta do usuário: dar ao sistema toda a física antes da relatividade e um
mundo que permita a descoberta. Versão discreta: partículas caem reto (Newton)
exceto ao lado de uma massa, onde desviam em direção a ela (a lente).

- **Newton puro**: 3/3 e 4/4. O sistema separa as duas dinâmicas por cor sozinho.
- **Mundo sujo** (`lente.py`, com colisões e bloqueio): falhou, e o motivo era o
  **mundo**. Feixe 2,5× maior e pilha 60 % mais funda deram resíduo idêntico.
- **Mundo limpo, com nulo pareado** (mesma frequência de desvio, direção sorteada):

```
lente    : fecha 4/6   acerta teste 3/6   DESCOBERTA 3/6
sorteado : fecha 0/6   acerta teste 0/6   DESCOBERTA 0/6
```

O termo achado, escrito na linguagem do sistema:
`PINTAR(desl(N, cor 8)) COM const(5)` — "a célula abaixo de uma massa vira
partícula". É a primeira **revisão de teoria** do projeto: não indução do zero,
e sim achar o termo que falta numa teoria que já funciona quase sempre, sem
inventar lei no ruído.

Ressalvas: o mundo limpo foi desenhado depois de o sujo falhar (vazamento de
desenho), n = 6, e 3/6 ficou abaixo do meu limiar pré-registrado de 5/6. As
falhas parecem **subamostragem** — com 5 pares sorteados, às vezes os desvios
caem todos para o mesmo lado. É o alvo natural de agência:
`lente_ativa_limpa.py` compara cenas sorteadas com cenas escolhidas por desacordo
de um comitê, com orçamento casado.

### Agência: um positivo que não replicou

```
rodada 1 (n=10)   passivo 3/10    ativo 5/10                  -> A1 no limiar
réplica  (n=30)   passivo 15/30   ativo 18/30   rico 15/30    -> p = 0,29
nulo sorteado     0 em todos os braços das duas rodadas
```

O A1 veio de um passivo baixo por acaso (3/10 contra 50 % na réplica):
regressão à média. O prior do projeto — escolher o experimento só paga com
modelo de mundo dado — **se mantém**. A direção (+10 pontos) se repetiu nas duas
rodadas; um efeito pequeno não está excluído, e medi-lo pede ~150 episódios
pareados. O script da réplica imprimiu "é o COMITÊ" sem R1, contra o próprio
pré-registro — bug de relatório, corrigido. A frase do veredito também é código.

### Órbitas: escrever a equação, e a história de Le Verrier

`orbita.py`, `orbita2.py`, `orbita3.py`. Mundo contínuo, teoria velha dada
(a = −r̂/r²), aceleração medida com ruído. A mesma anomalia com três causas:
massa oculta (Netuno/Vulcano), Sol achatado (−K/r⁴) e Einstein (−A·h²/r⁴). O
agente empilha modificações sobre o resíduo por MDL — termo de lei ou objeto
oculto — e **escreve a equação**.

| rodada | mudança | nulo permutado | einstein sem h (lei exata) |
|---|---|---|---|
| 1 | gulosa, ganho dentro da amostra | 1/5 limpo | 0/5 — epiciclo, extrapola −3,3 a 0,8 |
| 2 | ganho fora da amostra + seleção ortogonal | 3/5 | 3/5 |
| 3 | validação **aninhada** (10 sementes) | **10/10** | 7/10 |

Em todas as rodadas: Einstein escreve `−0,030·h²·r⁻⁴·r̂` (verdade −0,03),
o Sol achatado sai `−0,030·r⁻⁴·r̂` **sem** h, e Vulcano sai como massa oculta a
~0,1 do lugar certo, com a massa certa a 5 %. Nenhuma lei foi confundida com
objeto.

Três lições de método, todas pagas com falha:
- **R² dentro da faixa não separa lei de epiciclo; a extrapolação separa.** Na
  rodada 1, o portão de R² ≥ 0,98 aprovou uma aproximação que desaba fora da
  faixa medida;
- **validação depois da seleção não é validação.** Escolher os candidatos com
  todos os dados e validar depois deixou o nulo sujo; a validação aninhada o
  limpou. O código de Laplace, testado em braço separado, não mudou nada;
- **o nulo puro não pega nada.** Passou 5/5 desde a primeira rodada; foi o
  nulo *permutado* que pegou os dois defeitos da régua.

**O gap.** Com a grandeza certa (h) a lei é um termo forte e sai 10/10. Sem
ela, a lei vira um termo forte mais um fraco (vr²/r², ~10 % da anomalia), e o
expoente do fraco não é identificável dentro da faixa medida: 3 em 10 saíram
com r⁻¹ ou r⁻³ e desabam fora dela. Kepler resolveu inventando a grandeza (a
área varrida — h conservado). Dizer isso não é o mesmo que fazer: o sistema
não inventa grandezas, e esse é o próximo degrau.

### Inventar a grandeza: o primeiro laço invenção → vocabulário → descoberta

`inventa.py`. **Fase 1:** o agente recebe só trajetórias e procura, entre 833
combinações de até 3 monômios (grau 1 e 2 em x, y, vx, vy, mais r, 1/r, 1/r²),
a que fica constante dentro de cada órbita e varia entre órbitas. A régua é em
bits, escolhida no treino e validada em trajetórias novas. **Fase 2:** o que ele
inventou entra no caderno, e a lei é redescoberta com os mesmos dados de
laboratório da `orbita3`.

```
I0  inventou   q = −1,000·x·vy + 1,000·y·vx   (= momento angular)   10/10
    |corr(q, h)| = 1,00000;  λ ≈ 1e-5, só ruído de medida
I1  trajetórias embaralhadas: nada inventado                        10/10
I2  com q no caderno: a lei é UM termo  c·q²·r⁻⁴·r̂, extrapola 1,000  10/10
I3  placebo p = x·vx + y·vy (não conservado): extrapola               7/10
I4  sem extra, reproduz a orbita3                                     7/10
```

O placebo não ajudou porque p = r·vr: o agente só o usou para reescrever o
mesmo termo fraco. Mais variável não é mais conceito. Todas as previsões
pré-registradas acertaram, pela primeira vez no projeto.

**O que isso é e o que não é.** É a primeira vez que um conceito criado pelo
próprio agente entra no vocabulário e se paga na descoberta seguinte, contra
placebo. Não é criação do nada: o **alfabeto** é meu, e h é uma palavra de duas
letras nele. Descobrir leis de conservação a partir de trajetórias já foi feito
(AI Poincaré, Liu & Tegmark 2021). O padrão "o que eu entrego vira o teto" subiu
um nível — do conceito para o alfabeto — mas não sumiu.

**Limite achado.** No mundo com atrito, ele também "inventou" h, com λ 700×
pior, e aceitou, porque a variação entre órbitas domina o decaimento no tempo
observado. Ele não distingue conservado de *lentamente variável*. O conserto
natural é julgar λ contra o esperado só do ruído de medida, não contra zero.

**Quanto vale o conceito em dado.** Sem h, a lei de Einstein sai 0/10 com 15,
25, 40 e 80 corpos (7/10 só com 400). Com h: 0, 7, 9, 10/10. Mais de 10× em
dado.

**O laço composto, feito** (`inventa2.py`). Com q no alfabeto, a segunda
grandeza conservada — obrigada a trazer informação nova (residualizada contra
funções de q) — saiu como a energia de Einstein, com a correção relativística
escrita no conceito que o próprio agente inventou:

```
+0,50·vx² + 0,50·vy² − 1,00/r − 0,013·q²/r³        9/10 (ruído 0,002), 10/10 (0,0002)
```

Sem o conceito, o agente escreve um substituto (h² trocado pela média:
−0,005/r³) — epiciclo de novo. A vantagem de ter o conceito é quase invisível
com o ruído normal (1,1×) e cresce para ~8× com ruído 10× menor: a lei exata
desce ao piso do ruído, o substituto trava no próprio erro sistemático. O portão
pré-registrado (10×) falhou nas duas rodadas, na direção que a hipótese do
ruído prevê. Mercúrio: 43"/século só apareceu quando a medida ficou boa.

**Próximos degraus com mecanismo claro:**
- a ponte para o ARC: invariantes entre entrada e saída (contagens, cores,
  formas conservadas) inventadas e postas na gramática de testes.

### Transferência de operação: "ache o que se conserva", das órbitas ao ARC

A pergunta do usuário: que prova temos de que o sistema generaliza no substrato
certo? Em humanos, atributos não transferem (o sabor da maçã não serve à física
quântica); relações e operações transferem (Gentner), e mesmo assim a
transferência distante espontânea é rara (Gick & Holyoak; Sala & Gobet).

`invariante_arc.py` leva a operação que achou h nas órbitas para o ARC: o par de
treino entrada → saída é uma "trajetória" de dois pontos, e uma invariante é uma
combinação de grandezas da grade que a transformação conserva. Mesmo princípio,
mesma régua em bits, implementação discreta (a grade é inteira).

```
REAL   342/400 com invariante   vale no par de teste 97,4 %
NULO    56/400 (saídas de outras tarefas)          16,1 %
sem altura/largura/área:  real 97,0 %  ×  nulo 22,2 %
```

Primeira evidência de uma **operação** (não de um primitivo, não do critério)
atravessando de um substrato para outro e achando estrutura verdadeira. O que
não prova: as invariantes são quase todas "o que não muda" — estrutura barata —
e achar estrutura não é usá-la. A fase 2 usa a invariante como **crítica** das
respostas do núcleo.

`critica_arc.py` — a invariante do treino julga a resposta, sem gabarito:

```
respostas confiantes (fecham o treino): 40 certas, 16 erradas
   alarme falso nas certas   0/40
   erradas detectadas        2/16   (12,5 % — o portão pedia 25 %)
todas as respostas: 67 erradas acusadas, 67 de fato erradas (precisão 100 %)
```

Um modelo que fecha o treino já preserva o que o treino preserva, então a
crítica barata quase nunca pega o erro confiante. Ela vale como detector de erro
grátis e perfeitamente preciso para ~1/5 das respostas erradas, e pegou
`8ee62060` — resposta confiante e errada do núcleo numa tarefa que outro motor
resolve: pode **arbitrar entre motores**.

### O caderno vivo: acumular entrincheira o primeiro erro

`caderno.py`. Um universo só, 40 tarefas em sequência: 36 de laboratório (20
corpos cada, escrever a lei) e 4 de trajetória (inventar conceitos), com a lei do
universo trocando de Einstein para Sol achatado na tarefa 20. Memória de leis
com custo aprendido pelo uso (prior de "restaurante chinês": o que já pagou fica
barato de pensar) e esquecimento. Cinco braços, mesmos dados.

```
                         sem   só conceito   plástica   rígida   placebo
com conceito (6-19)       0 %      37 %          3 %       2 %      0 %
logo após a troca        82 %      84 %         80 %      78 %     84 %
adaptado (26-39)         80 %      80 %         95 %      94 %     83 %
```

O conceito vale +37 pontos. O placebo, zero. E a **memória de leis atrapalhou:
−35**. Nas tarefas 0–4, antes do conceito, ela aprendeu o epiciclo v²/r²; ele
ficou barato e passou a vencer sempre, inclusive depois de o conceito certo
existir. Esquecer por desuso não conserta, porque a ideia errada nunca fica em
desuso: ela vence. É o efeito Einstellung (Luchins, 1942) e o paradigma de Kuhn.
Onde a primeira lei aprendida estava certa, depois da troca, a memória ajudou
(+15). **A memória amplifica o que se aprendeu primeiro, certo ou errado.**

A resposta à pergunta "como vai evoluir sem acumular?" fica mais precisa:
acumular conclusões sem reavaliá-las é pior que não acumular. A rodada 2 testa o
conserto que sai da intuição do usuário — guardar **experiências**, não só
conclusões, e reanalisar os dados juntos quando um conceito novo chega.

**Rodada 2 — a reanálise conserta** (semente 20260921). Um braço novo guarda os
dados de cada tarefa; quando um conceito chega, junta tudo do mundo atual e
redescobre a lei com olhos frescos, substituindo a memória de leis:

```
                      sem   só conceito   plástica   rígida   placebo   REANÁLISE
com conceito (6-19)    0 %      30 %         21 %      19 %      0 %       90 %
adaptado (26-39)      79 %      79 %         96 %      95 %     80 %       96 %
falhas antigas (0-4) recuperadas: reanálise 50/50, só conceito 22/50, memórias 0/50
```

A memória passa de atrapalhar (−11 a −35) para ajudar (+60), e mantém o ganho
depois da troca de lei (+17). O entrincheiramento replicou na direção, mas menor
(−9/−11, o portão pedia ≤ −20). Ressalva: a reanálise mistura dois ingredientes
— mais dado (100 medidas juntas) e olhos frescos (sem as leis velhas). A revisita
por tarefa com olhos frescos recuperou 22/50; juntando os dados, 50/50.

**Rodada 3 — os dois ingredientes, separados** (semente 20260926):

```
                      só conceito   junta dados, leis velhas   junta dados, olhos frescos
com conceito (6-19)       37 %               52 %                       88 %
falhas antigas recuperadas  16/50            25/50                      50/50
```

Juntar os dados vale +15; analisar **sem** as leis velhas vale +36. Previ que só
o volume de dado importava e errei: mesmo com 5× mais dado, a lei velha barata
mais uma correção ajustam os dados dentro da faixa — epiciclo de novo — e a
validação, que também é dentro da faixa, não pega. É o princípio de Planck em
versão de máquina: quem aceita a teoria nova é a análise que não carrega a velha.

### Saber que está errado: reamostrar mede ruído, não ignorância

O gap escolhido em seguida, visto em três lugares: epiciclos que passam no R²,
a lei velha desviando a análise conjunta, a crítica que só pega 12,5 % dos erros
confiantes. A causa comum é que toda validação do projeto é fora da amostra, mas
**dentro da faixa** — e lá epiciclo e lei são iguais.

`falseia.py` — falsificação ativa com um comitê de reamostragens: simular o que
cada hipótese prevê em estados nunca medidos e ir medir onde elas discordam.

```
                     lei exata   distância média das 120 medidas escolhidas
passivo               16/20            1,83
largo (eu mando)      19/20            3,30
ativo (ele escolhe)   17/20            1,20
dúvida por reamostragem prevê lei errada: AUC 0,63
```

Falhou, e o porquê ficou visível: o agente mediu **perto do centro**. As
reamostragens concordam na forma da lei e só diferem nos coeficientes, cuja
variância é maior onde a aceleração é maior. **Reamostrar os próprios dados mede
ruído, não ignorância estrutural.** O comitê da `lente_ativa` — a agência que
não replicou — era do mesmo tipo. A rodada 2 (`falseia2.py`) troca pela dúvida
estrutural: rivais de forma diferente, pesados por bits.

**Rodada 2, com dado escasso (200 medidas):**

```
                          lei exata   onde mediu (r médio)
passivo                     4/20          1,87
largo (eu mando)            6/20          3,39
ativo, reamostragem        16/20          1,07
ativo, rivais estruturais  11/20          1,10
prevê que a lei está errada:  estrutural AUC 0,90   reamostragem 0,52
```

Três achados. **Escolher o experimento pagou +60 pontos** — a primeira vez no
projeto com modelo aprendido; a 400 medidas não aparecia porque todos estavam no
teto. **Minha premissa estava errada**: a informação não está longe — a anomalia
cai como r⁻⁴ e longe fica abaixo do ruído; os dois braços ativos foram medir
perto do centro sozinhos. E **as duas dúvidas servem a coisas diferentes**: a de
reamostragem aponta onde medir (onde o sinal é grande); a estrutural sabe quando
a lei está errada.

### O laço sem mim: a dúvida de um ajuste não é crise

`autonomo.py`. Quem decidia quando observar e inventar era eu (tarefas 5, 15,
25, 35). Com a dúvida estrutural como gatilho — observar quando a lei que acabou
de escrever é mais provavelmente errada que certa:

```
                   nunca   meu calendário   aleatório   autônomo (dúvida)
Einstein (0-19)      0 %        62 %           52 %           43 %
```

Falhou: o agente quase nunca entrou em crise cedo (2 de 10 correntes). Sem o
conceito, a lei certa não é escrevível como rival de mesmo porte — são dois
termos de ~20 bits — e a navalha dá o peso à lei simples e errada: o agente fica
**confiante**. A dúvida estrutural mede o desconhecido dentro da linguagem, não o
que falta fora dela. A rodada 2 troca pelo sinal de Kuhn: a lei que ele já tem,
aplicada aos dados **novos**, erra acima do ruído do instrumento?

**Rodada 2 — o detector funciona, a resposta não:**

```
                   nunca   meu calendário   dúvida   surpresa
Einstein (0-19)      0 %        62 %          42 %      69 %
crise na tarefa 0 (Newton não explica a anomalia):      10/10
troca de lei detectada nas tarefas 20-22:                 9/10
```

A surpresa contra o ruído é a metacognição certa para "minha teoria não dá
conta": percebe a anomalia de imediato e percebe quando o mundo muda. O ganho
sobre o calendário foi pequeno (+7) porque a resposta à crise estava errada —
gastou as observações reinventando h e nunca reanalisou os dados juntos. A
rodada 3 muda a resposta: toda crise reanalisa por consenso (começa pelos dados
que surpreenderam e incorpora as tarefas que a lei nova explica abaixo do ruído).

**Rodada 3 — o laço fecha sem mim:**

```
                   meu calendário   surpresa (resposta velha)   CONSENSO
Einstein (0-19)         68 %                66 %                  83 %
logo após a troca       88 %                88 %                  86 %
observações gastas      4,0                 3,9                   2,0
```

Os três portões passaram, e a previsão acertou até na magnitude. O agente
percebe a crise contra o ruído do próprio instrumento, inventa o conceito,
reanalisa por consenso, para de observar quando observar não rende, e percebe a
troca de lei sem ninguém dizer onde ela aconteceu — e bate o calendário que eu
desenhei. O +7 da rodada 2 era ruído; o que paga é a **resposta**. Ressalvas: um
universo só, o alfabeto da invenção é meu, o ruído do instrumento é conhecido.

### Generalidade: o mesmo laço, quando o conceito certo é outro

`generalidade.py`. Em todo experimento até aqui o conceito que faltava era h. Um
mundo novo, fisicamente consistente, onde o conceito certo é a **energia**: uma
força perpendicular à velocidade (não faz trabalho, então E se conserva
exatamente) com intensidade proporcional a E; não é central, então h não se
conserva.

Calibração: a invenção escolheu `h − 0,04·r` em 5/5 — uma invariante
**aproximada**, a que mais comprime as trajetórias — e não a energia, que é
exata. Com E dado a lei sai 10/10; com h dado, 0/10. O laço sem mudança:

```
                     nunca   calendário   consenso
magnético (0-19)       1 %       16 %        12 %
conceito inventado: momento angular 10/10, energia 0/10
```

Falhou como previsto. A crise não para — o detector sabe que a teoria não dá
conta — mas a resposta não tem remédio. **A surpresa e a reanálise generalizam;
a invenção não: ela escolhe o que mais comprime, não o que a lei precisa.** O
conserto em teste julga cada conceito candidato pelo que ele paga na lei em
crise.

**O conserto, rodada 1** (`conceito_util.py`): até 3 invariantes distintas por
observação, cada uma julgada **a jusante** — bits da lei redescoberta com ela.

```
                      calendário   consenso   ÚTIL
magnético (0-19)         18 %        19 %     100 %   (energia inventada 10/10)
Einstein (0-19)            —         84 %      32 %   (h inventado 2/10)
```

Resolveu o mundo novo e quebrou o velho. A primeira crise é na tarefa 0, e a
utilidade foi julgada com 20 medidas; no mundo Einstein uma quase-energia ganhou
por acaso, e a lei com o conceito errado ajusta dentro do ruído — a crise não
volta. É o entrincheiramento de novo, agora no nível do **conceito**. A rodada 2
aplica a mesma lição da reanálise: candidatos numa sala de espera, e a escolha
refeita em toda crise e sempre que a evidência dobra.

**Rodada 2 — o mesmo código serve aos dois mundos:**

```
                     referência          SALA DE ESPERA
magnético (0-19)     calendário 16 %         99 %
Einstein (0-19)      consenso   88 %         85 %
```

No acerto, os dois portões passaram. Na identidade do conceito falharam como
registrados — mas o portão foi mal desenhado por mim: mediu o conceito no fim da
corrente, depois da troca de lei, quando a última revisão mistura os dois mundos
e o mundo novo não precisa de conceito. Um diagnóstico reproduz as mesmas
correntes e registra o conceito em uso no fim da fase 1.

O diagnóstico (correntes 0–4 reproduzidas): na tarefa 19, o mundo magnético
usava a **energia** (correlação 1,0000) em 5/5 e o mundo Einstein usava **h**
(1,0000) em 5/5. Depois da troca de lei, o conceito foi **largado** na maioria —
no mundo novo ele não paga. Uma confirmação pré-registrada, com sementes novas e o
portão no lugar certo, transforma o diagnóstico em teste.

**Confirmação pré-registrada — passou:**

```
                     referência          sala de espera   conceito na tarefa 19
magnético (0-19)     calendário 13 %        100 %          energia  10/10
Einstein (0-19)      consenso   86 %         85 %          h        10/10
```

O mesmo código, em dois mundos cujo conceito necessário é diferente, escolhe o
conceito certo nos dois, e o larga quando o mundo muda e ele deixa de pagar.
Generalidade entre conceitos, dentro do domínio de órbitas. O que ainda não é
geral: a gramática — para o mundo magnético eu acrescentei à mão a direção n̂.
`gramatica.py` testa o agente criando essa direção sozinho.

### Gramática própria: o agente cria a direção que eu tinha posto à mão

`gramatica.py`. Em cada crise, além dos conceitos, o agente deriva direções novas
das três de base — girar 90°, somar ou subtrair e normalizar pares — sem observar
nada, e julga cada uma pela mesma utilidade dos conceitos.

```
                     gramática fixa   CRESCE
magnético, fase 1          0 %          95 %    n̂ criada em 10/10, sempre rot(v^)
Einstein, fase 1          84 %          84 %    quase nunca cria (a utilidade barra)
```

Os três portões passaram. Sem n̂ nem o conceito certo salva a lei (0 %, eu tinha
previsto ~30 %). É a terceira vez no mesmo dia que o teto desce um nível —
conceito, escolha do conceito, gramática. As operações de derivação (girar,
somar) e a profundidade 1 ainda são minhas.

### O laço num substrato discreto, com o núcleo do ARC

`laco_grade.py`, rodada registrada (recoloração: uma permutação de 9 cores que
cada tarefa só mostra em parte; a permutação troca na tarefa 15):

```
                fase 1   fase 2 (permutação trocada)
isolado           20 %        16 %
LAÇO              90 %        87 %    crises só ao aprender e na troca
pool_tudo         90 %        67 %    crises seguem até a janela se renovar
nulo (mapa sorteado): 0 %
```

Os quatro portões passaram (G3 no limite, +20). A mesma arquitetura — surpresa e
reanálise por consenso — que rodou sobre regressão contínua nas órbitas roda
sobre grades discretas com os codecs do ARC: acumular resolve a subamostragem que
a tarefa isolada não resolve, e a surpresa vê a troca de mundo em 6 de 6.

Segundo mundo de grade, agora com uma lei física — a lente gravitacional que a
tarefa isolada acertava ~50 % das vezes por falta de amostra, trocando para a
lente invertida:

```
                fase 1 (lente)   fase 2 (lente invertida)
isolado             43 %               43 %
LAÇO                85 %               88 %
pool_tudo           85 %               67 %
nulo: 0 %
```

Os quatro portões passaram (+42, 4/4, +22, 0 %), com as magnitudes previstas.
Quatro mundos, dois substratos, o mesmo laço.

### Um defeito da busca do núcleo do ARC, achado pelo laço na grade

Levando o laço para grades (`laco_grade.py`), o mundo da lente mostrou primeiro
um teto de linguagem: "cai reto a não ser que haja massa ao lado" não é
exprimível exatamente na gramática de codecs (pede negar um teste derivado). O
mundo foi trocado, antes do registro, por uma **recoloração** (permutação de 9
cores; cada tarefa vê só algumas) — e aí apareceu um defeito do próprio núcleo:

```
24 cenas da mesma permutação           custo      erros   generaliza
[mapa_cor, copia]                      69,8 bits    0      60/60
[viz_cor]  (decora cor + vizinhança)  100,4 bits    0      31/40  ← o núcleo escolhia este
[mapa_cor] sozinho                     6076 bits   708
```

A régua preferia o certo; a **busca** escolhia o decorador. `[mapa_cor]` sozinho
deixa o fundo sem dono e morre antes de receber a cópia, e `viz_cor` fecha o
treino sozinho — a busca para ali. É a cegueira do ganho imediato, quarto nível.
Conserto: todo candidato que não fecha é avaliado também **completado com
cópia**; as avaliações extras contam no teto. Na recoloração: `[mapa_cor, copia]`
com 4 e com 24 cenas, 100 % de generalização. Pré-registro da remedição das 400
(antes de rodar): núcleo 40 → 41–44, no máximo 2 perdas.

**Remedição — passou:** núcleo **40 → 44/400, zero perdas**, 14 % mais tempo.
Entraram `bf699163` (`mapa_cor + copia`, o mecanismo exato do conserto),
`4f537728` (`contorno + cruz + copia`), e duas regressões antigas recuperadas —
`903d1b4a`, a "regressão não resolvida" da seção 7, e `aa4ec2a5`. Duas são
inéditas: **união 59 → 61/400 (15,25 %)**. Um defeito achado levando o laço para
outro substrato melhorou o ARC — o retorno que o trabalho de generalidade deveria
dar.

**Correção, no mesmo dia.** O "teto de linguagem" da lente estava errado. Com a
busca já consertada, a gramática de profundidade 1 exprime a lei exatamente — o
"a não ser que haja massa ao lado" sai por contagem (a célula abaixo de uma
partícula isolada tem exatamente um vizinho ocupado). Com 16 cenas: 20/20 cenas
novas; com 8: 1/20. Faltava dado, e os testes da lente tinham rodado antes do
conserto da busca. Ligar negação de testes derivados (profundidade 2) **piorou**:
achou uma lei mais barata que fecha o treino e erra 6/20. Mais linguagem, mais
ajuste espúrio — o mesmo que a soma por objeto mostrou no ARC.

E um confundidor de orçamento: a mesma lente com 24 cenas cai para 5/20 com o teto
de avaliações do núcleo e volta a 20/20 sem ele. O teto é inversamente
proporcional ao tamanho do dado — com mais dado, menos avaliações. Isso conflita
com o laço, que junta experiências: quem acumula precisa poder pensar mais sobre
mais dado, não menos.

A resposta à pergunta do usuário — como o sistema evolui — fica medida em três
regras: **guardar experiências (dados), não só conclusões; quando a linguagem
muda, reanalisar tudo sem as conclusões antigas; e só dar crédito a memória
por um verificador quando houver um** (o trio em linha, abaixo).

### Problema em aberto com verificador rígido: nenhum trio em linha

O usuário pediu gaps medidos em problemas em aberto, com critério rígido. Um
problema do milênio só nos responde "falhou"; o meio-termo é um problema aberto
com **verificador automático e placar contínuo**. "Nenhum trio em linha"
(Dudeney, 1917): 2n pontos numa grade n × n, sem três colineares. Conhecido até
n = 68 (março de 2026, classe rot4); a conjectura diz que para n grande é
impossível.

`tres_em_linha.py` testa se o princípio do projeto — descrição curta primeiro —
atravessa da **indução** para a **construção**. Configurações simétricas têm
descrição curta (só as órbitas).

```
busca sem ordem                        chega a n = 10
descrição curta primeiro (7 simetrias) chega a n = 18   (rot4 em 6 de 7 soluções)
recorde humano + máquina               n = 68
```

O princípio atravessa: ele achou sozinho a classe dos recordes. O gap também
ficou à vista: a cada n ele gasta a primeira fatia na simetria mais curta (d4),
que quase nunca tem solução, porque não lembra que rot4 venceu no n anterior.
Liga com o caderno vivo — lá a memória entrincheirou um erro porque o crédito
vinha da própria avaliação do agente; aqui o crédito pode vir do verificador.
Hipótese em teste: **verificador rígido torna a memória segura**.

Testada (semente 20260923): com memória creditada só pelo verificador, n = 20
contra 18, e ~5× mais rápido nos mesmos n (n = 18 em 2,8 s contra 14,2 s). O
crédito foi todo para rot4, e nada entrincheirou. É o contraste limpo com o
caderno vivo: **a mesma memória ajuda quando o crédito vem de um verificador e
atrapalha quando vem da própria avaliação do agente.** A parede agora é o
construtor: n = 20 em 1,3 s, n = 22 em mais de 60 s com todo o tempo em rot4.

Analogia entre instâncias (a solução de n − 2, centrada, como preferência de
ordem): numa corrida única "chegou a 22", mas repetida em 5 sementes
(sucessos em n = 18/20/22/24: memória 5/1/0/0, analogia 5/3/0/0) nenhum método
passa de 20. O 22 era ruído. A analogia ajuda um pouco na borda e não empurra
a fronteira — a parede é o construtor. Máximo de uma corrida única não é medida.

### Dado real: 74 mil neurônios e um cérebro inteiro

Primeira vez que o projeto mede algo que não fui eu que gerei, com alfabeto que
não é meu — as grandezas foram escolhidas pelos neurocientistas.

**Allen Institute, 74.072 neurônios de camundongo acordado, 44 grandezas**
(`neuro_leis.py`). Lei linear ou de potência entre cada par, ajustada em metade
dos neurônios e aceita só se pagar bits na outra metade; nulo embaralha cada
coluna.

```
REAL 1463 leis aceitas de 1892 pares        NULO 0 de 1892
gabarito pré-registrado da literatura: 4/6
  taxa prediz taxa (+0,87) · esparsidade cai com a taxa (−0,46)
  OSI móvel × estática · taxas log-normais (assimetria 3,72 → −1,12 em log)
  FALHOU: "Fano cai com a taxa" — nestes dados sobe; a frase é sobre variação
          DENTRO de um neurônio e eu apliquei entre neurônios
  FALHOU: "área ≈ largura × altura" — área_rf é máscara em grade grossa
```

**FlyWire 783, cérebro adulto inteiro de Drosophila, 138.639 neurônios**
(`conectoma.py`): química bate com a literatura (ACh 67,1 %, glutamato 18,5 %,
GABA 14,4 %); reciprocidade **42,8×** acima do modelo que preserva os graus;
cauda pesada confirmada no teste correto (top 1 % dos neurônios = 11 % das
conexões, expoente ≈ 3,1, máximo 4,5× acima do que uma exponencial ajustada à
própria cauda permitiria). A regra de Rent deu expoente 0,716 — dentro da faixa
biológica — com ajuste frouxo (R² 0,59), e eu tinha marcado esse portão como
duvidoso antes de rodar.

**O número que importa para a arquitetura.** Das dez relações que eu importei da
literatura como gabarito nos dois conjuntos, **seis sobreviveram ao verificador**.
As quatro que caíram foram erro meu de definição operacional, não falha do
método. É a precisão medida do canal de importação quando o proponente é um LLM —
e o argumento empírico de que o verificador é inegociável.

## 9 · O alfabeto deixa de ser meu: um proponente que não sou eu (2026-09-12)

O usuário nomeou o gargalo com precisão: *"se a gente continuar desenhando tudo
ao invés de resolver o ponto que causa essa falha/gap, não vamos evoluir muito"*.
O ponto é que **o único canal de estrutura nova neste projeto sou eu escrevendo
primitivas**. Cada defeito que eu conserto sobe o teto um degrau e para. Três
frentes foram abertas contra isso ao mesmo tempo, e as três têm portão escrito
antes de rodar.

### A máquina

Eu tinha concluído que esta máquina não tinha GPU dedicada — uma consulta só
devolveu a Intel UHD, e eu escolhi um build CPU-only por causa disso. Errado: há
uma **RTX 3050 Laptop de 4 GB** (compute 8.6). Com `nvidia-smi` no lugar, o build
CUDA sobe e um modelo de 3B em q4 roda **inteiro na VRAM a 61 tok/s gerando e até
1.330 tok/s lendo**. Um 4B cabe com 3,83 GB dos 4,0 — no limite. Um 8B só com
offload parcial. A lição operacional virou memória: confirmar hardware com duas
ferramentas antes de decidir arquitetura em cima dele.

### O juiz: a desconfiança do usuário virou portão

O usuário desconfia do modelo local — *"uma mente parada no tempo com dados da
humanidade e poucos parâmetros"* — e aceita o modelo como **fonte de informação**,
não como autoridade. E a restrição é literal: o sistema lê o próprio código *"não
pra modificá-lo ... mas para identificar falhas, rodar suas próprias simulações,
criar seu próprio modelo talvez, aprender a programar, a entender contexto mesmo
escrito, entender lógica"*.

Desconfiança é hipótese testável. `juiz_modelo.py` mede a única coisa que decide
se um modelo local pode ser proponente de **arquitetura**: ele raciocina sobre um
sistema que não estava nos dados de treino dele?

* **12 questões de múltipla escolha** sobre defeitos reais deste repositório —
  cegueira do ganho imediato, o confundidor de orçamento `TETO_CEL // por_aval`,
  o portão medido depois da troca de mundo, memória entrincheirada, compressão
  que não implica utilidade, orçamento não casado. Não estão na internet: são a
  história deste projeto. Vários distratores **são os erros que eu mesmo cometi
  antes de medir** (dúvida bootstrap como gatilho, esquecer por desuso).
* **6 tarefas de programação** corrigidas por execução contra casos escondidos —
  o controle de sanidade.

Dois defeitos do instrumento foram achados e consertados antes de valer:
escrevi as 12 questões com a resposta certa quase sempre na letra B (um modelo
com viés de posição passaria de graça → alternativas embaralhadas com semente
fixa, igual para todos os braços); e um modelo que "pensa" pode gastar todo o
orçamento de tokens no raciocínio e devolver vazio, o que mediria o meu teto e
não a cabeça dele (→ segunda chance com o dobro de tokens).

```
  L0  SANIDADE: o melhor modelo acerta >= 4/6 em PROGRAMAÇÃO
  L1  SERVE DE PROPONENTE: algum modelo acerta >= 8/12 em LÓGICA
  L2  ACIMA DO ACASO: p < 0,01 no binomial contra 20%
  L3  RACIOCÍNIO > CÓDIGO: o modelo de raciocínio supera o de código por >= +3
  PREVISÃO: L0 passa, L1 FALHA (aposto 4 a 7 de 12), L2 passa, L3 passa.
```

### O canal: o modelo escreve primitivas, o MDL julga

`alfabeto_aberto.py` mostra a um modelo local os pares de **treino** de tarefas
que o núcleo não resolve e pede um **codec novo**. O contrato é minúsculo —
`nome`, `propor(treino, alvo)`, `bits(m)`, `prever(m, a)` — e é justamente isso
que torna a terceirização segura: quem escreve primeiro numa célula ganha e tudo
é cobrado em bits, então **um codec mentiroso não corrompe nada, ele paga caro e
perde**. O único dano real de uma proposta ruim é gastar orçamento de avaliação —
e é por isso que o portão casa orçamento e usa placebo pareado.

A cerca da restrição do usuário está no código, não na promessa: varredura
estática (sem `import` fora de math/itertools/collections, sem `open`, `exec`,
`eval`, sem dunder, sem estado global, exatamente uma classe com os três
métodos), prova de vida em **subprocesso com relógio**, e a saída cai em
`propostas/*.py` — isolado, entrando só como item **extra** da lista de codecs.
Nada é escrito em `descricao.py`.

`portao_alfabeto.py` separa as duas coisas que não podem ser confundidas:

* **CAPTURA** — o codec proposto a partir da tarefa X resolve o **teste** de X
  (legítimo, o teste ficou escondido; mas é o caso fácil);
* **TRANSFERE** — o codec resolve tarefas que o modelo **nunca viu**. É esta que
  decide se o alfabeto cresceu ou se o modelo escreveu uma tabela para a tarefa
  que leu.

```
  A1  CAPTURA: >= 3 das tarefas geradoras passam a ser resolvidas
  A2  TRANSFERE: >= 2 tarefas não mostradas ao modelo passam a ser resolvidas
  A3  NÃO É LOTERIA: o placebo (mesmas propostas, cores embaralhadas por uma
      bijeção fixa) não ganha nada sobre a base
  A4  NÃO QUEBRA: no máximo 1 tarefa perdida
  PREVISÃO: A1 passa (3 a 6), A2 FALHA (0 ou 1), A3 passa, A4 passa.
```

### A terceira frente: HDC como analogia, não como importação

`proponente.py` responde à pergunta do usuário — *"por que não fazemos um usando
HDC?"*. A resposta honesta é que HDC **não importa conhecimento**: ele recombina
e recupera o que já passou. Não substitui o modelo como canal, mas é exatamente a
peça da **analogia** — uma memória que se atualiza a cada experiência e diz "este
contexto se parece com aquele, lá funcionou isto". Quatro braços decidem a ORDEM
em que os candidatos já existentes são avaliados, com teto de avaliações fixo e
igual: `fixo` (ordem histórica), `sorteado` (placebo), `habito` (o que mais
funcionou antes, ignorando contexto) e `hdc` (liga contexto ⊗ candidato e pontua
por semelhança). Memória **online**: cada tarefa só vê o que veio antes dela, sem
vazamento por construção. O braço `habito` é o controle que importa — ele separa
*analogia* de *repetir o que costuma dar certo*.

```
  P0  sanidade: o braço FIXO resolve entre 30 e 44 tarefas
  P1  hdc - sorteado >= +3        P2  hdc - habito >= +2
  P3  nenhum braço fica abaixo do sorteado
  PREVISÃO: P0 e P1 passam, P2 FALHA — o contexto que eu sei descrever
  (dimensões, cores, fração não explicada, tamanho da pilha) é pobre demais.
```

---

## 10 · A loja, o índice e a composição (2026-09-13 a 2026-09-17)

Cinco dias que começaram tentando achado inédito em biologia, fecharam esse
braço com nulo, e terminaram no primeiro positivo de **composição** do projeto.
A ordem importa: quase tudo aqui veio de consertar um defeito que uma medição
anterior nomeou, e não de ideia nova.

### 10.1 Os dois nulos que fecharam o braço biológico

**Resíduo de fármaco (`residuo_farmaco.py`, `roda_farmaco.py`).** Casar
assinatura transcricional não identifica alvo: posto mediano **721/2023**, e os
4 % de top-1 são todos inibidor de proteassoma. Teto do dicionário 21 %.

**PRISM v2.1 (`prism_v2.py`), replicado em 17/09.** Pré-registro dizia "Q1
passa, 1 a 6 eventos". Deu Q1 falhou (controles **1/5** testáveis) e Q4 falhou
(0 candidatos). O único par aceito em 403 linhagens × 1.317 compostos foi
`TP53 intacto → idasanutlin` — e ele é exatamente o único genótipo **não
concentrado em tecido**, como a memória já diagnosticava. Residualizar por
tecido tira o sinal farmacogenômico junto com o confundidor.

**A regra que sai daqui, e que se repetiu o resto da semana:** um nulo só
informa se o instrumento recupera a linha de base conhecida. Com sensibilidade
1/5 em controle de livro-texto, "zero inédito" é informação sobre o
instrumento, não sobre o mundo.

### 10.2 Décimo segundo nulo: ligar não basta (`ligacao.py`)

A crítica acusa **13/13 com zero falso positivo** — detector perfeito. Realocar
**4× o orçamento exatamente onde ela acusa** resolve **zero**. Saber ONDE se
falha não ajuda quando falta O QUE procurar.

### 10.3 O limiar de dado é lei do critério (`lingua.py` + scratchpad)

O mesmo critério de bits que resolve grades aprende léxico da fala real. Testado
num **segundo domínio** (código-fonte), a forma se repete:

| N treino | léxico | F1 held-out | nulo embaralhado |
|---|---|---|---|
| 200 | 28 | 0,365 | 0,358 |
| 1000 | 30 | 0,371 | 0,358 |
| 1500 | 145 | 0,506 | 0,358 |
| 4000 | 278 | 0,560 | 0,358 |

**Abaixo do limiar o resultado é ZERO, não fração** — o léxico tem 26 entradas,
que são as 26 letras. E o gargalo **alterna** com o domínio: alargar a busca de
8 para 24 candidatos valeu +0,09 em código e só +0,02 na fala. Diagnóstico
barato: alargue a busca primeiro; se o F1 mexer, era busca (minutos), não dado
(meses).

### 10.4 Integrar exige dose E afinidade

Medido respondendo à pergunta do usuário sobre alimentar o sistema com datasets.

**Ruído não dilui, ENVENENA.** 3.000 tokens embaralhados derrubam o F1 de 0,506
para 0,379 e **colapsam o léxico para o alfabeto**. O critério não ignora o
ruído; gasta a descrição tentando comprimi-lo.

**E um confundidor que quase virou manchete:** misturar o código do `novaphys`
deu +0,095 e o veredito automático disse "juntar domínios paga". Mas o
`novaphys` tem identificadores em **português** — sobreposição de tipos com a
fala: 32,6 % contra 5,0 % da stdlib. Com stdlib no lugar: **−0,057**. Não havia
transferência entre domínios; havia mais português.

Curva de dose (fala fixa em 1000): material de alta afinidade **também machuca**
abaixo de 1:1, e material de baixa afinidade não ajuda em dose nenhuma.

### 10.5 A digestora como PORTEIRO: falhou (`digestor.py`)

Construí um porteiro — decide o que entra, descarta o resto, funde num corpus.
Filtrou **4/4 de ruído**, e mesmo assim perdeu para o sorteio, porque admitiu
material estrangeiro estruturado: as duas maiores notas de afinidade eram
stdlib. **Compressibilidade responde "consigo modelar isso?", não "isso vai me
ajudar?"** — terceira aparição dessa separação no projeto.

E o meu placebo era n=1. Em 25 sorteios o "−0,092" que anunciei vira −0,016, e o
instrumento não resolve a pergunta: composições quase idênticas dão 0,389 e
0,464.

### 10.6 A loja SOB DEMANDA: funcionou (`sob_demanda.py`)

Correção do usuário: a digestora não é porteiro, é **ativador**. Nada é
descartado; a extração acontece na hora da tarefa.

A MESMA loja de 16 blocos, duas tarefas, só a sonda muda:

| | T1 segmentar fala | T2 segmentar código |
|---|---|---|
| fundido (aceita tudo) | 0,388 (lex 28) | 0,315 (lex 27) |
| **sob demanda** | **0,572** (lex 204) | **0,474** (lex 193) |
| placebo, 12 sorteios | 0,391 ± 0,009 | 0,324 ± 0,012 |
| oráculo | 0,531 | 0,479 |

**+0,181 e +0,150 sobre o sorteio — 20 e 12 desvios.** O stdlib sai da posição
**9 em T1 para a 1 em T2**, com a loja intacta: inútil vira nutriente, medido.
O ruído fica nas 4 últimas posições nas duas tarefas, sem portão nenhum. E o
recuperador **ganhou do meu oráculo** em T1, porque trouxe código em português
junto — reconstruiu sozinho o achado de 10.4.

**A lei que isso estabelece:** guardar tudo e ordenar na hora vence filtrar na
entrada. O envenenamento de 10.4 acontece quando tudo é **fundido num modelo
só** — não é objeção à loja, é a razão de o corpus único ser o recipiente errado.

### 10.7 O índice multimodal (`hdc_indice.py`)

Espaço bipolar comum a todas as modalidades, átomos gerados por hash sob
demanda. `de_texto` e `de_fluxo` produzem vetores no **mesmo espaço**, então
`sim(sonda_de_texto, bloco_de_fluxo)` é operação definida — é o único caminho
medido para consulta multimodal.

Como ranqueador o HDC passa (topo muda com a tarefa, ~15 desvios) mas **perde da
compressão dentro do mesmo espaço** (0,530 × 0,572) e não distingue TIPOS de
longe: ruído fica intercalado com stdlib no fundo. A razão para usá-lo não é
desempenho, é **alcance**.

### 10.8 A composição: a nuvem estilhaçando (`nuvem.py`, `dois_eixos.py`)

Critério do usuário para mente forte: aplicar uma transformação a um material
que nunca foi visto sofrendo aquilo.

**Primeiro medi se o pedido tem endereço.** Com o encoder amarrando alvo a
NÍVEL, o índice agrupa por MATERIAL (−0,4201 contra nulo −0,0942) — porque o
material B codifica o latente com **sinal invertido**. Com invariância de
escala + SINAL + PAPEL o sinal **vira para +0,1185** contra nulo +0,0359: existe
eixo de transformação. Sem essa medida, um fracasso do gerador seria ambíguo
entre "gerador fraco" e "pedido sem endereço".

**Depois o gerador.** Loja com `A|regime`, `A|deriva`, `B|deriva`; **sem**
`B|regime`. Tarefa: prever `B|regime`.

| braço | acerto |
|---|---|
| teto (regra do próprio material) | 86,5 % ± 6,4 |
| **composto (regra de A aplicada a B)** | **81,2 % ± 10,6** |
| controle (transformação errada) | 50,3 % ± 5,2 |
| nulo (alvo sorteado) | 37,0 % ± 3,9 |

**A ponte é o achado real.** A loja contém a quádrupla analógica com uma célula
faltando. O par `deriva` existe nos dois materiais e é rotulado, então
`A:deriva :: B:deriva` fixa como A vira B — e aí `A:regime` diz o que `B:regime`
deve ser. Nada toca o alvo de teste. É `abducao-construir-nao-propor` de novo: a
ponte foi **construída do dado guardado**, não proposta por mim.

**Três defeitos meus no caminho, todos da mesma forma** (escrever o controle
antes de conferir que ele controla algo): fase do PC1 não fixada (teto bimodal
35,2 % ± 34,5); "conserto" por `argmax(|r|)` que não consertou, porque em A os
canais dão |r| = [0,9813 ; 0,9830] com **sinais opostos** e o argmax desempatava
na terceira decimal; e placebo sorteando de uma loja que **contém a própria
regra em teste**.

### 10.9 A matriz, e o limite que ela nomeou (`nuvem_matriz.py`)

5 materiais gerados × 4 transformações, uma célula retirada por vez.

**Rodada 1:** agregado falha por 0,1 ponto (composto 60,8 × controle 50,9,
barra +10). Separado por transformação — variável de desenho, não corte
post-hoc — as de NÍVEL compõem (+16,7 e +22,5) e as de MUDANÇA não (+4,0 e
−3,3), **com teto 62,5 e 59,6**: o recall não expressava essas regras nem com o
material próprio. O encoder amarrava nível nos atrasos 0 e 1 e nunca a diferença.

**Rodada 2**, com termos de 1ª e 2ª diferença: teto de `deriva` **62,5 → 84,8** e
de `acelera` **59,6 → 81,6**. O conserto funcionou. Mas o agregado ainda falha
(+9,3), e apareceram três coisas:

- **duas células saem ~94 % INVERTIDAS** (`mat1|acelera` 5,6 %, `mat3|acelera`
  12,2 %, com tetos 93,5 e 88,2). A informação atravessa; a fase vem trocada.
  **A fase continua sendo 1 bit não resolvido.**
- **o controle sobe junto** (regime 51,5→56,1; extremo 46,3→50,1): dar
  expressividade ao encoder ajuda também a regra errada.
- **o nulo VAZA em `deriva`** (83,3 % numa célula) porque `deriva` **é** o sinal
  da primeira diferença, que eu acabara de codificar como feature.

### 10.10 Ver o que ela imagina (`render.py`, `render_gif.py`)

Decodificar hipervetor em pixels arbitrários **não dá** — o cleanup só recupera
combinação de átomos já presentes, que é o mesmo motivo do 2/400 como gerador.
Renderizar **dá**, e sem decoder: a saída já é um fluxo T × K com estado previsto
por instante. O `render_gif` põe lado a lado o imaginado e o real.

**Limite de honestidade do GIF:** a sequência é dela; a metáfora visual
(heptágono, "estilhaçado") é minha e arbitrária. O substrato não tem vidro.

### 10.11 O que ficou medido sobre o método, e é o mais transferível

**Meu placar de previsões no dia: palpite sem mecanismo 0/4; previsão feita
depois de rastrear o mecanismo no código do gerador 3/5.** As que acertaram
foram todas do segundo tipo. A que errou tinha o mecanismo certo e a magnitude
errada — eu não tinha feito a conta (φ=0,86 dá sd(−0,14·s)≈0,12 contra 0,45 do
ruído). **Ter o mecanismo não basta; precisa da conta, e precisa perguntar qual
mecanismo é o DOMINANTE.**

**Três controles n=1 me enganaram no mesmo dia** (placebo do digestor, placebo
da nuvem, nulo da matriz). Regra: todo nulo e placebo com ≥ 10 sorteios e desvio
reportado, sem exceção.

**Dois portões mal especificados:** um comparava primeiro-contra-último N numa
curva não-monótona e disse "plano" quando o máximo interno passava a barra;
outro exigia do nulo algo que o gerador de dados não podia produzir. Antes de
escrever o portão, conferir que o contraste existe no dado.

### 10.12 O estado dos módulos novos

| arquivo | o que é | veredito |
|---|---|---|
| `digestor.py` | porteiro com afinidade + dose | **falhou**, mantido como registro |
| `sob_demanda.py` | loja sem descarte, ordenada por tarefa | **passou**, 12–20 desvios |
| `hdc_indice.py` | espaço comum multimodal, `Loja` | ranqueia; perde da compressão no mesmo espaço |
| `nuvem.py` | composição 2×2 + gerador por recall | **passou** (81,2 % × 50,3 %) |
| `dois_eixos.py` | os eixos material/transformação existem? | **passou** após encoder invariante |
| `nuvem_matriz.py` | 5×4, uma célula retirada por vez | agregado **falha** (+9,3) |
| `render.py` | HTML/canvas do episódio | **bugado**, canvas em branco |
| `render_gif.py` | GIF lado a lado, imaginado × real | funciona |

---

## 11 · Ordem superior, e o dia em que o dado real revelou uma cegueira (2026-09-18)

Um dia com um positivo forte em dado real, um achado arquitetural que invalidava
parte do que se media, e cinco defeitos de medição meus — três deles
**escondendo** resultados bons em vez de fabricar falsos.

### 11.1 Operadores de segunda ordem: compor bate aprender

O núcleo tinha combinadores de **primeira** ordem (`ENTAO`, `ATE`, `VEZES:k`,
em 42 % das pilhas vencedoras) e **zero** operadores que recebessem operador.
A única exceção era a ponte de `nuvem.py`, instância escrita à mão.

`segunda_ordem.py` transporta duas transformações para um material novo e as
compõe num preditor que **nunca esteve na loja**:

| braço | acurácia balanceada |
|---|---|
| **composto (2ª ordem)** | **82,5 %** |
| só `deriva` transportado | 79,2 % |
| só `regime` transportado | 70,4 % |
| controle (primitivas erradas) | 60,5 % |
| **teto (aprender a regra composta)** | **50,3 %** |
| nulo / chutar a classe comum | ~50 % |

**Compor bate aprender por 32 pontos.** A regra composta é difícil de
*aprender* — o recall é um voto linear e a classe positiva é rara — mas compor
duas regras simples é **exato** e passa por fora dessa limitação.

**Terceira vez que uma peça de ordem superior paga do mesmo jeito:** o ganho não
vem de mais capacidade, vem de **contornar um limite do mecanismo de baixo**.

### 11.2 A fase era 1 bit, e votar resolveu

A matriz 5×4 tinha duas células saindo ~94 % **invertidas** (`mat1|acelera`
5,6 % com teto 93,5). A informação atravessava; só o sinal vinha trocado.

Trocar a ponte única por **voto de três pontes ponderado por confiança** levou
`mat1|acelera` de **5,6 % para 92,1 %** — o teto daquela célula.

Mas o agregado não se move: +9,9 → +9,3 → +9,6 em três rodadas com consertos
substanciais. **A margem está travada porque o meu controle compartilha o
latente**: as quatro transformações são funções do mesmo `s`, então todo conserto
no transporte levanta composto e controle juntos.

### 11.3 As primitivas vêm do dado, e compõem melhor que as minhas

Correção do usuário: *"Humanos não inventam primitivas, eles extraem do mundo e
de dados."* Ele estava certo — a extração já existia medida (`inventar-a-grandeza`,
`gramatica-propria`, `lingua.segmenta`); o que faltava era **ligá-la à composição**.

`primitiva_do_dado.py` recebe 64 episódios **sem rótulo**, agrupa por
similaridade das regras, destila uma regra por grupo, transporta e compõe:

```
pureza do agrupamento   0,719   (acaso 0,250; NULO 0,438)
COMPOSTO de descobertas   85,1%
outros pares de grupos    65,0%
```

E escolheu os grupos certos sem ver nome: grupo majoritariamente `deriva` +
grupo majoritariamente `regime`, para um alvo `regime E deriva`. **85,1 % é
maior que os 82,5 % com as primitivas que eu escrevi à mão.**

O conserto que fez passar estava pré-diagnosticado: **super-agrupar** (k=8 para
4 regras) deixa subgrupos puros se formarem.

### 11.4 A cena mental cabe, e custa 50 KB

Do critério do usuário: *"um engenheiro faz isso com milhares de objetos mentais
colados numa cena"*. A pergunta virou número.

| objetos | D=2.000 | D=10.000 | D=40.000 |
|---|---|---|---|
| 32 | 0,997 | 1,000 | 1,000 |
| 100 | 0,465 | 1,000 | 1,000 |
| 300 | 0,080 | 0,837 | 1,000 |
| 800 | 0,014 | 0,170 | 0,983 |
| **capacidade** | **32** | **100** | **800** |

Dimensão 20× → capacidade 25×. **Linear.** Mil objetos numa cena custam ~50 KB;
dez mil custam ~420 KB. **Cena mental de milhares não é gargalo — é memória
barata.** Minha conta pré-registrada previu "dezenas" e errou por mais de uma
ordem, para o pessimista.

### 11.5 Canto de ave: primeira afirmação sobre questão aberta de outra área

`canto.py`, sobre dado público do Bengalese Finch song repository (39 cantos,
2.693 sílabas). A literatura disputa se o canto precisa de **chunks** ou se
Markov basta.

| modelo | bits/sílaba no retido |
|---|---|
| unigrama | 3,331 |
| bigrama | 0,398 |
| **trigrama** | **0,199** |
| 4-grama | 0,207 |
| chunk (unigrama de chunk) | 0,264 |
| **chunk + transição** | **0,201** |

**Empate preditivo** (0,201 × 0,199 contra desvios de 0,015). Mas em parâmetros:
**32 contra 250** — o chunk prevê igual sendo 8× menor. Por comprimento de
descrição, a descrição por chunk ganha.

**Três rivais, e cada um mudou a resposta.** Contra bigrama o chunk ganhava
+0,133 e eu ia anunciar "há estrutura além do par a par". Contra trigrama
perdia. Com transição, empata. **O primeiro rival que eu escolho é sempre fraco
demais e sempre a favor da hipótese em teste.**

### 11.6 Variedade paga, para a LOJA

| materiais | itens | precisão@8 | nulo |
|---|---|---|---|
| 8 | 64 | 0,456 | 0,255 |
| 16 | 128 | 0,550 | 0,249 |
| 32 | 256 | 0,631 | 0,258 |
| 64 | 512 | **0,738** | 0,246 |

Razão sobre o acaso: 1,8× → **3,0×**, com o nulo parado. Previ saturação e
errei: o mecanismo que invoquei existe mas não era o dominante — mais materiais
dão **mais exemplares por transformação**, e o topo-K passa a ter de onde
escolher.

### 11.7 Guardar o extraído compra ROBUSTEZ, não acurácia

`acervo.py` liga extrator à loja — a seta que faltava no desenho do usuário.

```
BRUTO,    recuperado  0,603     sorteado  0,481 ± 0,124
EXTRAÍDO, recuperado  0,605     sorteado  0,519 ± 0,042
```

Com recuperação boa, empatam. Com recuperação **ruim**, o bruto despenca e
oscila; o extraído degrada pouco. Concatenar um bloco de ruído e re-aprender
**destrói**; acrescentar entradas de um léxico ruim só adiciona entradas não
usadas. **O acervo deve guardar as duas formas.**

### 11.8 O ACHADO DO DIA: o fluxo nunca teve tempo

Primeiro experimento do projeto em dado que eu **não gerei**: UCI Human Activity
Recognition (61 MB, `D:\agi_dados\har`) — 30 voluntários, 9 canais inerciais,
janelas de 128 instantes, 6 atividades nomeadas.

```
encoder = nível        pureza real 0,528   nulo (tempo embaralhado) 0,537
```

**Empate.** Toda a estrutura capturada sobrevive a embaralhar o tempo.

**`de_fluxo` é um HISTOGRAMA.** Somar `bind(canal, nível)` sobre todos os
instantes descarta a ordem por construção; os atrasos não salvam, porque cada
atraso vira seu próprio histograma. **O encoder de fluxo é invariante a
permutação do tempo, e isso esteve lá desde que foi escrito**, atravessando
silenciosamente tudo construído em cima.

**Só o dado real revelou.** Nos mundos sintéticos que eu gerava, distribuição e
forma temporal andavam juntas, então o histograma parecia suficiente.

### 11.9 E o conserto: o caminho da ORDEM descobre 'andar'

Um grep disparado por uma pergunta do usuário reorganizou o projeto — há **dois
caminhos de codificação com propriedades opostas**:

```
SENSÍVEL À ORDEM (viterbi):   lingua, canto, acervo, digestor, sob_demanda
CEGO À ORDEM (soma no tempo): o caminho do fluxo / de_fluxo
```

E **todos** os resultados sólidos do projeto estão no primeiro. Então não se
construiu nada novo: aplicou-se ao fluxo real o mecanismo que já funcionava em
quatro domínios — quantização vetorial de cada instante em um símbolo, e
`lingua.segmenta` achando os motivos por MDL.

| encoder | pureza real | nulo tempo | ganho |
|---|---|---|---|
| histograma | 0,528 | 0,537 | **+0,000** |
| transição | 0,506 | 0,442 | +0,064 |
| **léxico / viterbi** | **0,778** | 0,567 | **+0,210** |

**84 motivos no real contra 15 no tempo embaralhado** — embaralhar destrói 82 %
dos motivos. Os motivos *são* estrutura temporal.

**O que ele descobriu, sem ver rótulo:**

| grupo | composição | pureza |
|---|---|---|
| 10 | WALKING:106, WALK_DOWN:6, WALK_UP:1 | **94 %** |
| 5 | WALK_UP:49, WALK_DOWN:4, WALKING:3 | 88 % |
| 0 | WALK_DOWN:69, WALKING:22 | 73 % |
| 6 | SITTING:62 | **100 %** |
| 7 | LAYING:98 | **100 %** |
| 11 | STANDING:100, SITTING:24 | 81 % |

Motivos legíveis: `pppppppppp` (símbolo repetido = postura parada);
`lllllkkkc`, `aaaarrrrr` (alternância = ciclo de passada). **Ninguém deu a noção
de "passo"; o ciclo foi achado porque se paga em bits.**

**Honestidade:** HAR é benchmark supervisionado resolvido (~96 % treinado). O
notável não é bater isso — é que um mecanismo escrito para grades do ARC, **sem
uma linha de adaptação**, descobre os conceitos sem supervisão. Quinto domínio
da mesma máquina: grade, física, fala, código, canto, sensor humano.

### 11.10 Escolher o alvo: NULO, e por defeito meu

`alvo_proprio.py` implementa seleção autotélica por **progresso de aprendizagem**
(a literatura: agentes autotélicos, MAGELLAN arXiv 2502.07709). As quatro
estratégias empataram — inclusive `difícil-primeiro`, que queima o orçamento nos
objetivos impossíveis.

**A causa invalida o teste:** só **2 de 12** objetivos tinham folga. Ganho máximo
alcançável ≈ **0,3** contra desvio de medida **0,7**. *O efeito máximo possível
era menor que o ruído* — conta de trinta segundos que eu podia ter feito antes.

Três defeitos reais achados no caminho, e valem mais que o nulo:
1. **o gerador não definia objetivo** — `lim` sorteado a cada episódio faz de
   `regime` uma regra diferente por episódio; 1 episódio = 40 episódios;
2. **o feixe CANCELA em vez de acumular** — `m1|extremo` cai de 0,788 com 3
   episódios para 0,520 com 20;
3. **a sonda de busca estava descasada** do conteúdo.

### 11.11 O padrão dos meus erros, e ele é o mais transferível

**Cinco medidas mal especificadas em um dia**, e a direção não é aleatória:

*Fabricaram sucesso inexistente (2):* placebo sorteado de uma loja que continha
a própria regra em teste; bigrama como rival do chunk.

*Esconderam sucesso real (3):* acurácia bruta num alvo 28/72 (o teto caiu de
71,8 para 50,3 ao corrigir); portão exigindo que o teto fosse informativo,
quando compor supera aprender; massa total de termos enviesada por contagem
(43 % → 86 % ao normalizar).

**O denominador comum: escrevo a estatística resumo antes de olhar a estrutura
do que estou resumindo.** Somei `|sim|` sem contar quantos termos de cada tipo
existem; usei acurácia sem olhar o balanço de classes; exigi que o teto limitasse
sem checar se aprender-direto era o melhor método.

**E o que salva nos cinco casos é o mesmo: o relatório detalhado contradiz o
veredito automático.** As linhas do topo diziam "a taxa subindo" para os grupos
de mudança enquanto o resumo dizia NÍVEL para todos. Os desvios de 34 pontos
diziam bimodal enquanto a média dizia fracasso. **Isso é argumento de projeto
para a interface de conceito: a explicação item a item é o controle contra a
própria estatística mal escrita.**

Placar de previsões do dia: **palpite sem mecanismo 0/4; com mecanismo rastreado
no código 4/9**. Os erros com mecanismo foram todos do mesmo tipo — identifiquei
um mecanismo **real** e não perguntei se era o **dominante**.

### 11.12 Estado dos módulos novos de 18/09

| arquivo | o que é | veredito |
|---|---|---|
| `segunda_ordem.py` | operador que recebe transformações | **passou** (82,5 % × 50,3 %) |
| `primitiva_do_dado.py` | descobrir primitivas sem rótulo e compor | **passou** (85,1 % × 65,0 %) |
| `explica.py` | leitura direta do vetor, verificável | **parcial** (6/7 pela leitura; resumos discordam) |
| `cena.py` | capacidade da cena mental | **passou** (800 objetos, escala linear) |
| `alvo_proprio.py` | escolher o próximo alvo | **nulo**, subdimensionado por defeito meu |
| `canto.py` | chunk × Markov em dado público | empate preditivo, chunk 8× menor |
| `variedade.py` | mais fontes ajudam o ranking? | **passou** (1,8× → 3,0×) |
| `acervo.py` | extrator → loja | extraído = bruto em acurácia, 3× menos variância |
| `har.py` | descoberta em dado real (histograma) | **revelou o defeito**: nulo empata |
| `har_lexico.py` | descoberta pelo caminho da ordem | **passou** (0,778; ANDAR 94 % puro) |

### 11.13 O que isso muda na prioridade

Eu tratava "dado real" como algo que melhoraria a **loja**. Não é isso. Dado real
revelou um defeito **arquitetural** que quatro dias de experimento sintético não
revelaram e que invalidava silenciosamente parte do que eu media. Dado que eu não
desenhei é a única coisa capaz de revelar que eu desenhei errado.

E ligar peça que funciona rendeu mais que construir peça nova, **três vezes
seguidas**: extrator → loja (`acervo`), extração → composição
(`primitiva_do_dado`), e caminho do léxico → fluxo real (`har_lexico`). O
caminho com mais retorno é continuar ligando o mecanismo do léxico — que já
provou generalidade em cinco domínios — onde ele ainda não está.

---

## 12 · Revisão externa, e o que ela derrubou (2026-09-18, noite)

Uma revisão externa independente leu `ARQUITETURA.md`, o código de `novaphys`,
as memórias e os logs. Levantou seis pontos técnicos. **Cinco procedem**, e um
deles derruba um resultado que eu tinha anunciado como positivo. Registro aqui o
que foi conferido no código, o que foi corrigido e o que eu contesto.

### 12.1 Seleção no conjunto de teste — PROCEDE, e inverte o veredito

`primitiva_do_dado.py` criava **um** conjunto `testes`, media os 28 pares de
grupos **nele**, tomava o **máximo** e reportava esse máximo. O controle era a
**média** dos outros pares no mesmo conjunto. **Máximo contra média é viés
duplo**, e a comparação não dava ao controle a mesma liberdade de escolha.

Conserto: conjunto de **seleção** e conjunto **final** disjuntos; o par é
congelado na seleção e só então avaliado no final; o controle recebe a mesma
oportunidade (melhor de um subconjunto sorteado, escolhido na seleção, avaliado
no final, média de 10 sorteios).

```
par escolhido na SELEÇÃO 85,1%  ->  no conjunto FINAL 85,1%
COMPOSTO                    85,1 %
CONTROLE (mesma seleção)    76,9 % ± 5,4      (era 65,0 % como média simples)
```

**Y1 → Y2.** A margem cai de **+20,1 para +8,2**, ~1,5 desvio, abaixo da barra
de 10 pontos pré-registrada.

**E a correção resolveu diferente do previsto, o que importa registrar:** o
número reportado **não inflou** — o par escolhido entregou o mesmo 85,1 % em
episódios frescos. **O viés estava no controle, não na medida.**

Corrigido também: o texto anunciava "64 episódios" usando `k * N_POR_TRANSF`
com k=8 grupos, quando o laço cria `len(TRANSFS) * N_POR_TRANSF` = **32**.

### 12.2 O nulo da cena usava a referência errada — PROCEDE

Em `cena.py` o evento contado é "a resposta pertence a **algum** objeto da
cena". Sob escolha uniforme a referência é **n/V**, não 1/V — que seria a chance
de um objeto *específico*. Com n=2000 e V=4096 o acaso correto é **0,49**, e eu
reportava 0,25 como se fosse alto contra 0,0002, quando estava **abaixo** do
acaso. Corrigido no código: o acaso agora é impresso por tamanho de cena.

Isso não altera a medida de capacidade (recuperação de 0,983 com 800 objetos em
D=40.000), que não depende desse nulo.

### 12.3 Os 50 KB são custo MARGINAL, e eu não disse isso — PROCEDE em parte

Os ~50 KB são o vetor da cena. O codebook de 4.096 × 40.000 em float32 ocupa
~625 MiB. Ele é **compartilhado entre todas as cenas** e não replicado, então
"50 KB por cena" está certo como custo marginal — mas sem dizer que há um
codebook fixo, a conta engana. A afirmação correta é **"50 KB por cena, mais um
codebook fixo"**.

### 12.4 HAR mede pureza transdutiva, não acurácia em participantes novos — PROCEDE

`har_lexico.py` ajusta a quantização vetorial com as janelas disponíveis,
aprende o léxico em parte delas, agrupa o conjunto e calcula pureza **nesse
mesmo conjunto**. Isso é protocolo **transdutivo**, e eu não o declarei como tal.

A divisão oficial do UCI separa **participantes** entre treino e teste
justamente para isso. O teste apropriado — e possível com a base já baixada — é:
ajustar normalização, quantização, léxico e qualquer associação grupo→atividade
**sem** os participantes finais, e só então medir.

A evidência que **sobrevive** ao apontamento: a pureza real (0,778) contra o
nulo de tempo embaralhado (0,567) é uma comparação **interna ao mesmo
protocolo**, então o ganho de +0,210 continua indicando que a estrutura
aproveitada é temporal. O que não está demonstrado é reconhecimento em pessoas
novas.

### 12.5 Vetores na mesma dimensão não são significado compartilhado — ACEITO

Eu escrevi que o espaço comum "torna `sim(sonda_de_texto, bloco_de_fluxo)` uma
operação definida". É verdade e é fraco: define a operação, não demonstra que a
palavra e o episódio físico correspondam. Nenhum teste de alinhamento entre
observação, ação, consequência e descrição foi feito.

### 12.6 A conclusão de MDL no canto precisa de convenção comparável — ACEITO

Os 32 contra 250 parâmetros foram marcados como análise **secundária** e assim
permanecem. Uma conclusão estrita por comprimento de descrição exigiria codificar
léxico, estrutura e parâmetros sob a mesma convenção, o que não foi feito.

### 12.7 O que eu contesto — o critério X1 da segunda ordem

A revisão diz que o "passou" não corresponde ao critério registrado. **Eu já
havia reportado isso** na memória `segunda-ordem-compoe`: *"X1 falha na cláusula
'+5 sobre a melhor primitiva': 82,5 × 79,2 = +3,3"*. O apontamento justo é que o
veredito **impresso pelo script** e o texto divergiram, e isso confunde.

A afirmação que eu sustento é a comparação contra o **teto** (50,3 %, habilidade
zero) e o **controle** (60,5 %), não contra a melhor primitiva — e com a ressalva
já registrada de que `deriva` sozinha é linha de base forte para `AND` por
construção.

### 12.8 O que isso muda no estado do projeto

**Resultados que caem ou encolhem:** a composição de primitivas descobertas vai
de "Y1, +20,1" para "Y2, +8,2 ± 5,4". O nulo da cena estava mal formulado.

**Resultados que sobrevivem:** a descoberta de primitivas sem rótulo (pureza
0,719 × 0,250 acaso, 0,438 nulo); a capacidade da cena e seu escalonamento
linear; o achado de que `de_fluxo` é invariante a permutação do tempo; o ganho
de +0,210 do caminho da ordem sobre o nulo temporal em HAR; e o resultado do
canto, que já tinha partição por canto e previsão em dado retido.

**A lição de método, e ela é nova:** casar **orçamento** entre braços eu já
fazia. O que faltava é casar **liberdade de escolha** — um braço que escolhe o
melhor entre N candidatos precisa ser comparado a um controle que também escolhe
entre N, não à média dos N. São coisas diferentes e eu só tinha a primeira.

**E o ponto mais desconfortável da revisão, que eu aceito:** parte importante da
adaptação continua acontecendo fora do agente. Eu escolho o problema, a
representação, o gerador, os rivais e o critério. Um desafio congelado e
independente é o que mediria quanto o próprio sistema faz — e ele não existe.

---

## 13 · O piso de ruído, e o que ele reclassifica (2026-09-18, madrugada)

Quatro medidas encadeadas. A primeira é positiva, a segunda é nula, a terceira
desmonta parte do instrumento e a quarta o conserta.

### 13.1 Confiança calibrada — a única coisa do Jev que faltava aqui

O Jev (TypeSafe AI, 15/09/2026) devolve **valores tipados com probabilidade
calibrada** dentro de um esquema fixo, sem gerar texto. Duas das três
propriedades já existiam neste projeto por convergência: síntese MDL válida por
construção e avaliação paralela de candidatos. A terceira, calibração, nunca
tinha sido testada.

No transporte HAR congelado (base 0,545):

```
MARGEM p/ o 2º colocado          similaridade BRUTA
  quartil 1   0,293                quartil 1   0,371
  quartil 2   0,504                quartil 2   0,524
  quartil 3   0,647                quartil 3   0,644
  quartil 4   0,740                quartil 4   VAZIO (satura em empates)
  Q4 − Q1 = +0,446
recusando os 25 % menos confiantes: 0,545 -> 0,629, cobertura 75 %
```

**J1 pela margem.** E o mecanismo é a lei do projeto: a margem é uma
**comparação**, a similaridade bruta é um **valor absoluto**. Importar a ideia
crua ("devolva uma probabilidade") teria dado o número saturado.

Isso destrava uma peça bloqueada: `cobertura-nao-basta` mediu que modelo total
mas errado planeja **pior** que não planejar, e que recusar-se a chutar é
proteção. Faltava o sinal para recusar.

### 13.2 Abstrair pelo padrão de repetição — NULO, e o mecanismo foi medido

Depois de `poucos_exemplos.py` falhar duas vezes em **alinhar** uma pessoa às
outras, a linha aprovada era **abstrair**: achar o que os motivos de pessoas
diferentes têm em comum. A forma canônica por ordem de primeira ocorrência
(`abcab -> 0,1,2,0,1`) é literalmente a variável ligada de
`objetivo-precisa-de-quantificador`.

```
concordância entre pessoas, MESMA atividade      0,589 -> 0,841   (+0,252)
concordância entre pessoas, atividades DIFERENTES 0,213 -> 0,516  (+0,303)
margem utilizável                               +0,376 -> +0,325
acurácia em pessoas novas                        0,537 -> 0,438
```

O mecanismo funcionou e o resultado piorou: **borrou em vez de abstrair**. A
forma canônica descarta *quais* centróides — e os centróides são justamente a
parte já compartilhada entre pessoas, porque foram congelados no treino. Abstraí
sobre o comum e guardei o pessoal, de costas.

Dois defeitos meus: o diagnóstico mediu só o numerador (sexta vez que escrevo a
estatística-resumo antes de olhar o que ela resume), e chamei fusão aleatória de
"sem estrutura" quando ela é **projeção aleatória**, que preserva quase tudo.

A margem foi testada como triagem geral em 13 braços e **reprovou** (Spearman
0,368): explica este caso, não serve de regra.

### 13.3 O piso de ruído que eu nunca tinha medido

Um braço que descarta a ordem e guarda os símbolos deu 0,585 contra 0,537 do
literal. É **bijeção** — 100 motivos, 100 rótulos —, logo não pode carregar
informação nova: só troca a string, o hash e o hipervetor do átomo. Medindo dez
bijeções do mesmo conteúdo:

```
0,544  0,515  0,479  0,613  0,591  0,593  0,567  0,576  0,588  0,598
média 0,566   desvio 0,040   faixa 0,479 a 0,613
```

**Zero bits diferentes, 13 pontos de amplitude.** `har_pessoas` publicava 0,545
como ponto, e o nulo de lá (`0,470 ± 0,003`) variava só o embaralhamento com os
átomos fixos — reportava 1/13 do ruído real.

### 13.4 O pareamento cancela, e a memória estava certa

```
real       0,566 ± 0,040     <- o nível oscila
DIFERENÇA  +0,069 ± 0,008    <- a comparação não, 10/10 positiva
```

O ruído de sorteio é **modo comum**. O `+0,075` de um sorteio só era boa
estimativa do `+0,069` verdadeiro: `motivo-e-pessoal` sobrevive, agora com barra
de erro. `har_pessoas.py` foi corrigido para rodar 10 sorteios e reportar nível e
diferença pareada separados (`clona`/`_finaliza`).

### 13.5 Uma trave que eu movi e desfiz

Ao instrumentar o pareamento, troquei o critério `P1` de "ganho ≥ 0,15" —
tamanho de efeito **pré-registrado** — por "≥ 2 desvios e positivo em todos", que
é barra de significância. O veredito virou `P1` sem que o efeito mudasse.
Desfeito: a barra pré-registrada voltou, o ganho é +0,071 e o veredito é `P2`.
O pareamento **acrescenta** estabilidade, não substitui tamanho de efeito.

### 13.6 No que as quatro convergem

13.1 mede que a **margem** calibra e o valor absoluto satura. 13.4 mede que a
**diferença** tem σ 0,008 e o nível σ 0,040. São perguntas sem relação, horas
apart, e dão a mesma forma. "Sabe comparar, não sabe avaliar em absoluto" deixou
de ser limitação do agente e virou propriedade da arquitetura inteira, medida dos
dois lados — inclusive do lado do **instrumento**.

A consequência prática vale para trás: todo número absoluto de HDC neste projeto
anunciado de um sorteio só, sem par, não foi medido. Os pareados sobrevivem.

---

## 14 · A âncora externa, e onde o pipeline perde (2026-09-19)

O usuário afirmou que já temos uma mente artificial geral. Toda comparação de HAR
até aqui era **interna** — 0,560 contra maioria 0,187 e contra o nulo de tempo
embaralhado. Nenhuma contra terceiros. A base do UCI traz os 561 atributos de
engenharia dos autores, então dá para **rodar** a linha publicada.

### 14.1 De quanto é o buraco

```
PUBLICADO  561 atributos dos autores + SVM linear      0,967   (artigo: ~0,96)
PAREADO    7 linhas de estatística por canal, cru      0,752
NOSSO      pipeline de descoberta congelado            0,560 ± 0,039
```

`E0` passou — reproduzi o artigo, então a implementação não é desculpa. A saída
de emergência que eu havia pré-registrado ("se o pareado ficar perto de 0,56, o
problema é o **dado**") fechou sozinha: o pareado deu 0,752, com
`mean/std/min/max/energia/|Δ|/sd(Δ)`. Cinco minutos de estatística batem a
máquina de descoberta por 19 pontos no mesmo dado e no mesmo split.

### 14.2 Somar em vez de competir — e a redundância

Perder sozinho e ser redundante são diagnósticos diferentes, e só o segundo
condena.

```
estatística por canal (63 col)   0,752
contagem de motivos  (120 col)   0,697      <- acha estrutura real (maioria 0,187)
AS DUAS JUNTAS                   0,740      ganho -0,012      F2
```

Depois, com a codificação que o 2×2 revelou ser melhor — **segunda olhada, e
declarada como tal**:

```
estatística + feixe HDC          0,767 ± 0,003   ganho +0,016   F2c (barra 0,02)
```

O ganho é **pequeno e real**: positivo em 5/5 sorteios, desvio 0,003, abaixo da
barra pré-registrada. A estrutura que o MDL descobre neste domínio acrescenta
~1,6 ponto sobre sete linhas de estatística. Não é zero, e não é muito.

### 14.3 O 2×2: a perda não estava onde eu achava

```
                  logística    medoide(12)
contagem            0,697        0,529
feixe HDC           0,721        0,548
```

O feixe HDC é **melhor** que a contagem crua nos dois leitores. A hipótese de que
o sinal-da-soma custava os 14 pontos estava errada. **113% da perda é o leitor de
medoides**: 12 protótipos e voto de maioria custam ~17 pontos sobre uma
logística nas mesmas colunas.

E aqui vale a distinção, porque ela não é detalhe: os medoides são o que faz os
grupos se formarem **sozinhos**, sem rótulo, para só depois receberem nome.
Trocar por uma logística compra 17 pontos e muda o que está sendo afirmado — de
*descoberta* para *classificador supervisionado sobre atributos descobertos*. A
formulação honesta é: **a representação sustenta 0,721; o agrupamento
auto-formado extrai 0,548 dela.** Esse vão é a medida precisa da distância entre
o que o sistema descobre e o que ele consegue expressar por conta própria.

### 14.4 O que isso responde

Não temos uma mente artificial geral. Temos uma máquina de descoberta que acha
estrutura real sem rótulo em cinco domínios, e que neste — o único com âncora
externa — fica 19 pontos abaixo de estatística trivial, acrescenta 1,6 ponto
sobre ela, e perde 17 pontos entre o que representa e o que seu próprio
agrupamento consegue ler.

A lição de método, e ela é cara: **comparação só interna escondeu 19 pontos de
buraco por meses.** Nenhum resultado deste projeto deve contar como capacidade
enquanto não tiver rival externo no mesmo dado.

---

## 15 · O limite de linhas mentiu (2026-09-19)

Ao montar uma validação **por pessoa** para escolher a resolução do agrupamento
sem tocar no teste, a partição encontrou 8 pessoas onde o código dizia 21.

```
n=900   ->  3 pessoas  [1,3,5]     (padrão de har.py e har_lexico.py)
n=1200  ->  4 pessoas              (N_TESTE de har_pessoas)
n=2400  ->  8 pessoas              (N_TREINO de har_pessoas)
n=7352  -> 21 pessoas              (a base inteira)
```

O "30 voluntários" da seção sobre descoberta e o "21 × 9" da seção sobre
transporte vieram da **descrição da base do UCI**, não de contar o que `max_rows`
entregou. Nenhum portão pega isso, porque não há portão sobre uma string.

### 15.1 O que replica no dado completo

| | 3–8 pessoas | 21 × 9 pessoas |
|---|---|---|
| descoberta sem rótulo (pureza × nulo temporal) | 0,778 × 0,567 | **0,725 × 0,542 ± 0,029** |
| léxico | 84 motivos | **91 motivos** |
| transporte, medoides(12) | 0,560 ± 0,039 | **0,707 ± 0,014** |
| teto da representação (logística no feixe) | 0,721 | **0,852** |
| rival trivial (7 linhas de estatística) | 0,752 | **0,901** |
| linha publicada (561 atributos + SVM) | 0,967 | 0,967 |

**A conclusão central da seção 14 sobrevive intacta.** Subimos 14,7 pontos e o
rival trivial subiu 14,9: a distância vai de **−0,192 para −0,194**.

### 15.2 O que cai

- **O −0,407 contra a linha publicada estava despareado.** Rodei o braço
  publicado no split completo e os nossos no truncado. Casado, é **−0,260**.
  Erro meu de comparação, não do dado.
- **O "leitor subdimensionado" era artefato.** Com 8 pessoas, 12 protótipos
  ficam famintos e k=24 dava +0,104; com 21 pessoas, k=24 não ganha nada (0,707)
  e k=192 recupera só 27% do vão. A resolução não era o gargalo — a escassez de
  gente era.
- **O piso de ruído é função do tamanho do teste**, não constante da
  arquitetura: ±0,040 com 4 pessoas, **±0,014** com 9.

### 15.3 E um veredito meu que disparou errado de novo

A regra de `leitor_grupos` decidia pelo *k que a validação escolheu* (384) quando
a pergunta era *em que compressão o ganho aparece* — e a curva mostrava o salto
em k=24, com 100:1 de compressão. Imprimiu `H3` onde o conteúdo era `H1`. Sétima
vez que escrevo a estatística-resumo antes de olhar a estrutura do que ela
resume; a ironia é que a lição foi registrada na véspera.

O conserto de raiz está em `har_pessoas.py`: a contagem de sujeitos agora é
**medida** do que foi carregado e a subamostra é declarada ao lado.

---

## 16 · O número oficial estava velho (2026-09-22)

### 16.1 O núcleo vai de 44 para 105/400

Remedido com saída exata no teste nas 400 tarefas, com procedência salva em
`resultados/oficial.json` — nomes, codecs, marcadores e profundidade de cada
solução. O número passa a ser **auditável**, não afirmado.

```
NÚCLEO (montar + aplicar, todas as saídas de teste exatas)
  medido 22/09/2026:  105/400   (26,2%)
  registro anterior:   44/400   (núcleo)
  registro anterior:   61/400   (UNIÃO com outros caminhos — não remedida)
```

### 16.2 Não é orçamento, e eu previ que era

```
teto  5.000  -> 31/120
teto 20.000  -> 30/120
teto 200.000 -> 30/120
só com teto grande: 0 tarefas | só com teto pequeno: 1 tarefa
```

**40× de busca acrescenta zero tarefas.** Eu previ que os 105 eram compute e
escrevi o mecanismo para sustentar — `montar` varre formas, custo é forma × teto.
O usuário previu que o trabalho em problemas em aberto havia respingado no ARC.
A previsão dele é a que sobreviveu.

A decomposição apoia essa leitura: `copia` 80 (preenchedor de fundo),
**`programa` em 27 tarefas (25,7%)**, `cor_por_prop` 10, `vizinhanca` 7,
`d4` 9, `conecta` 8. Profundidade: 42 pilhas de 1 passo, 44 de 2, 12 de 3,
3 de 4, 4 de 5.

### 16.3 Dois números meus, corrigidos por revisão externa

Eu li as **38 ocorrências** de `programa` como 38 tarefas. São 38 ocorrências em
**27 tarefas** — o codec aparece mais de uma vez na mesma pilha. 25,7%, não 36%.

E eu escrevi que "a cadeia morre na passagem de descrição para competência",
generalizando um resultado estreito. O medido é: **recombinar léxicos de
substring, deste modo, não trouxe ganho de classificação neste teste**. Isso não
estabelece que recombinar representações e operadores seja inútil. A distinção
que falta é entre a **forma** do material e o **procedimento** que ele descreve:
um léxico de substrings do Python contribuir pouco para reconhecer atividade
humana é quase esperado; um *algoritmo* expresso em Python seria outra coisa, e
não foi testado.

### 16.4 Outro número derrubado

Só **9 das 105** pilhas usam marcador (ENTAO 8, ATE 2) = **8,6%**. A seção
anterior e a memória `o-gargalo` registram *"a busca as usa em 42% das
vencedoras"*. Não se sustenta.

### 16.5 A causa raiz, e ela continua aberta

Este projeto tem disciplina rigorosa **por experimento** — pré-registro, controle
pareado, placebo, nulo do duto — e **nenhuma disciplina no agregado**. Cada peça
nova foi medida no seu próprio experimento, e ninguém rodou o banco inteiro por
sessões enquanto o código melhorava embaixo. Somado a `novaphys/` **não estar sob
controle de versão** (2.979 arquivos no git, zero de `novaphys`), a pergunta
"quando mudou?" é estruturalmente irrespondível.

O JSON de procedência resolve isso daqui para frente. Não resolve para trás.

---

## 17 · A parede: o sistema era só Sistema 2 (2026-09-22 a 24)

Três dias em que várias linhas independentes bateram no mesmo lugar, e o lugar
ganhou nome. A seção segue a ordem em que as coisas foram medidas, porque a
parede só apareceu depois de várias tentativas de contorná-la falharem.

### 17.1 Factorio: o agente acha o sistema de pesquisa sozinho

Perseguindo `automation-science-pack`, o agente parou em `recipe_not_researched`.
Consultado o jogo: **33 de 217 receitas habilitadas, 196 tecnologias, 2
pesquisadas**. 184 receitas estão trancadas atrás de pesquisa — o teto da
fábrica é um sistema que o agente não sabia existir, e que ele encontrou pelo
mesmo mecanismo da circularidade montadora↔energia: lendo o diagnóstico do
mundo depois de falhar. Três gaps abertos, não consertados: `planta_usina` ainda
materializa construção com `create_entity` (o pecado do `insert`, mudado de
lugar); motor a vapor sem encaixe no terreno do run completo; pesquisa não
dispara (0 laboratórios no mundo).

### 17.2 Modelo de mundo por domínio — `modelo_mundo.py`

Cinco mundos que já estavam em disco (ARC, CIFAR, FashionMNIST, HAR, texto)
num substrato comum 8×8, com o **operador** como moeda de travessia.

```
W0(a) domínio ainda reconhecível após normalizar   0,729  (acaso 0,200)
W0(b) precisão@8 CRUZADA (o próprio domínio sai)   0,652  (acaso 0,167) = 3,9x

veneno = domínio com a mesma forma cujo rótulo MENTE
              sem veneno -> com veneno     queda
FUNDIDO         0,652       0,411       17,4 desvios
COTA            0,646       0,555       11,3 desvios
CRÉDITO         0,654       0,654        0,0 desvios
```
W1 no veneno total. Na **dose** (ficção que mente em parte) deu **D2** pela barra
pré-registrada (queda de 2,5 desvios a 20% contra barra de 1) — a barra estava
em desvios em vez de valor absoluto, e o veredito ficou como escrito.

**Crédito por surpresa** (`calibra_surpresa`): o desfecho vem de PREVISÃO, não
de rótulo — a sonda é só (antes, depois). A 20% de mentira caiu 1,1 desvio,
onde o crédito com rótulo tinha caído 2,5. Custo de dispensar rótulo: 3,6
pontos. Penhasco entre 20% e 40% (fonte que mente 40% era zerada inteira); o
piso graduado por fragmento (`calibra_fragmento`, `forca` calibrada SÓ no atlas
limpo) moveu o penhasco para ~50%, não o removeu — a 40% voltam 7 fragmentos de
252, e o veredito automático "P1" me lisonjeava.

**Corroboração entre mundos** (`consenso`): conta em quantos mundos
independentes o operador aparece. Bate o crédito em todas as doses (0,683 x
0,600 limpo; 0,702 x 0,601 a 40%) **sem calibração nenhuma**. O ganho está nos
operadores difíceis (+0,120), não nos fáceis (+0,008) — refutando o risco de
voto de popularidade que eu tinha pré-registrado.

### 17.3 Décimo terceiro nulo: corroboração não é via de admissão

`segunda_via.py`: um estado 8×8 com valores 0–3 é um Grid ARC válido, então os
23 codecs REAIS rodaram sem adaptação em CIFAR/HAR/texto. Corroboração entre
mundos contra uso nas 105 pilhas vencedoras: ρ = +0,32 (n=22), **parcial
controlando bits +0,15**. Generalidade entre mundos e utilidade no ARC são
eixos independentes: `aritmetica` funciona nos 5 mundos e está em 0 pilhas;
`cor_por_prop` em nenhum mundo e em 10 pilhas.

### 17.4 Gradiente na admissão, e por que ele não bastou

Auditoria pedida pelo usuário ("como tu ainda tá escrevendo primitiva?"):
**23 codecs no núcleo, 23 escritos por mim, 0 vindos de geração**; nenhum
módulo de geração é carregado por `montar()`. A geração produz; a seleção nunca
adota.

`gradiente.py`: a descontinuidade de `paisagem-e-penhasco` não está no
critério, está na ATRIBUIÇÃO winner-take-all (`primeiro-a-escrever-bloqueia`).
Amolecendo a reivindicação (mistura por softmax com pesos), a derivada em w=0
sai fechada, `dL/dw_c = −Σ(1[c acerta] − p[previsto por c])`, e separa codec útil
de placebo de mesma cobertura em **4,8 desvios** — tirar `copia` MELHORA a
separação, e o maior gradiente é de um codec com cobertura 0,23.

`alcanca_novo.py`: a primeira rodada deu −1,4 desvios e eu quase registrei um
nulo. Era medida dentro da amostra — no treino, `propor` ajusta e os dois braços
acertam por construção. Fora da amostra o material inventado generaliza +0,123
sobre o arbitrário e o gradiente o alcança a **2,9 desvios**. Lição: os bits eram
a defesa contra sobreajuste; trocar bits por acerto fora da amostra é medir sem
proxy o que o proxy aproximava.

`paga.py` (R4, 4 braços, 9 núcleos, seleção deixa-um-de-fora DENTRO do treino):
```
          10  20  30  40  50  60  70  80  90 100 110 120 130 140
BASE       6   6  12  15  19  20  22  25  29  30  33  35  37  40
TUDO       6   6  12  15  19  20  22  25  29  30  33  35  37  40
GRADIENTE  6   6  12  15  19  20  22  25  29  30  33  35  37  40
PLACEBO    6   6  12  15  19  20  21  24  28  29  32  34  36  39
```
**R5**, n=140 (as 10 mais pesadas não terminaram). Nem uma divergência. Admitir
não é a parede.

### 17.5 Escala ou capacidade — e duas conclusões minhas corrigidas

`escala_ou_capacidade.py`: 24× mais treino no HAR (mesmas 21 pessoas, subamostra
aleatória), nosso método −0,007, distância para 7 linhas cresce 0,158 → 0,195.
Mais dado não fecha. Eu atribuí isso à representação.

`percepcao.py` + decomposição: **no mesmo protocolo sem rótulo, as 7 linhas dão
0,670 e nós 0,675.** Com rótulo, toda representação sobe a ~0,91. A distância
nunca foi percepção: era agrupar-sem-rótulo contra classificar-com-rótulo. O
teto da descoberta sem rótulo é ~0,70 para qualquer representação porque a
estrutura saliente é a PESSOA. Corrigidas: a âncora ("o buraco é nosso") e a
leitura da curva de escala ("o buraco é de representação").

### 17.6 Mente contínua v1 — `mente.py`

Primeira mente que vive 428.577 passos reais de Minecraft (VPT) sem reiniciar:
prevê antes de ver, mede surpresa contra o piso de cada lei, cria lei em crise
só com o que surpreendeu, leva leis entre episódios. **M0 falhou** (o regime do
inventário só vale 6% mesmo para o oráculo) e o mecanismo também (teto de 6 leis
saturado no episódio 20; prevê pior que uma lei só — as leis extras ajustavam
ruído). Pré-checagens seguintes mostraram o padrão: sessão 0%, HAR próxima
amostra 5%, melhor horizonte 11%. **Em dado real, regimes semanticamente enormes
quase não mudam o preditor. Surpresa sozinha não vê o que importa.**

Correção: eu tinha dito que a memória que se revisa "não acontece sozinha" —
o laço fechado existe em `laco-autonomo` (83% contra 68% do meu calendário).

### 17.7 Linguagem aterrada — `linguagem.py`

Sem LLM, pelo pedido explícito do usuário. Só frase + estado antes + depois: o
sistema recuperou o dicionário inteiro (6 palavras → operadores, 3 vazias,
"depois" = ordem direta, "antes" = inversa) e entende combinações nunca vistas
em mundos nunca vistos a **1,000**; sacola 0,500 onde a ordem importa; placebo 0.
Ressalvas: a regra do aprendiz foi trocada depois de ver as contagens do treino
(a primeira deu L3); compor é mecânico com operadores exatos; é o primeiro
degrau, não é ler.

### 17.8 A parede, nomeada

R4 (admitir não resolve), protocolo-não-percepção (sem desfecho tudo para em
0,70), mente v1 (surpresa não vê semântica), mais `canal-de-importacao`,
`ligar-nao-basta` e a curva de orçamento plana: **a arquitetura é um Sistema 2**.
Linguagem explícita, busca, verificação exata. Ela só conhece o que está escrito
na linguagem dela, e a linguagem só cresce quando eu escrevo. Falta o **Sistema
1**: completamento de padrão aprendido, que alcança conceito sem nome.

Literatura: Li & Ellis (ICLR 2025) — indução e transdução resolvem conjuntos
disjuntos; Schwartenbeck et al. (Cell 2023) — replay hipocampal monta hipóteses
compostas; DreamCoder e sucessor (arXiv 2306.07856) — reconhecimento amortizado
é metade do que faz a biblioteca pagar; Prospective Compression (arXiv
2605.09985) — humanos formam abstrações pensando nas tarefas futuras; H-ARC —
humanos 64,2% no eval.

### 17.9 Sistema 1 — `sistema1.py`

CNN residual com dilatações, treinada DO ZERO em cada tarefa, só nos pares de
treino, ~12 s na RTX 3050. Escolhe **a maior simetria sob a qual o treino ainda é
reproduzido exato** (d4+cor, d4, cor, nenhuma) — depois de o primeiro piloto
impor simetria e corromper o mapa de cor `0d3d703e`.

```
SISTEMA 1 (mesma dimensão)        35/262
  novas (núcleo não resolve)       13  -> 11 robustas em 3 sementes
PLACEBO CORRIGIDO (desarranjo)     0 das 134 elegíveis (Sistema 1: 25 delas)
FORMA do núcleo + SISTEMA 1        18/138 que mudam de tamanho, 4 novas (sem placebo)
```
**O primeiro placebo deu 15 e imprimiu "viés da arquitetura".** Nas 15, todos os
pares estavam no lugar certo: `shuffle` de 2 pares é a identidade em metade das
vezes, e o filtro de forma com `or pares` descartava os trocados. Corrigido para
desarranjo: 0.

### 17.10 Experiência entre tarefas — `experiencia.py`

Pré-treino na outra metade (disjunta), ajuste fino em cada tarefa:
```
DO ZERO 35 | EXPERIÊNCIA 38 | PLACEBO (pares embaralhados) 44
```
**E3.** 22–26% do placebo eram pares verdadeiros (formas de grade únicas), o que
REFORÇA a leitura: um braço ~24% relacional bateu um 100% relacional. O ganho é
estatística de grade, não experiência de como transformações funcionam — com
130 tarefas.

### 17.11 O sonho: o Sistema 2 ensina o Sistema 1 — `sonho.py`

As regras que o núcleo domina (101 reproduzidas, 74 geraram pares) aplicadas a
grades de OUTRAS tarefas: 3.606 pares com relação verdadeira. Placebo feito certo
desde o início — mesmas entradas, mesmas saídas, desarranjo, formas únicas fora
dos dois braços. Avaliação SÓ nas 188 tarefas de mesma dimensão que o núcleo não
resolve (nenhuma regra avaliada entra no sonho).
```
do zero 13 | grades 17 | SONHO 19 | SONHO-PLACEBO 12
```
**D1: +7 sobre o mesmo volume sem relação**, contra a minha previsão (D2 a 0,40).
Regra explícita virou intuição, e ajudou em tarefas que nenhuma daquelas regras
resolve. Uma semente; confirmação com a semente 1 em andamento.

### 17.12 Placar e defeitos

ARC, conjunto de TREINO, 2 tentativas: **núcleo 105 → núcleo + Sistema 1 122**;
com o sonho 124. Todas as variantes juntas 133 (mais de 2 tentativas, não
comparável). Nada disso foi medido no conjunto de AVALIAÇÃO ainda.

Defeitos de instrumento meus nestes três dias, todos achados antes de virar
conclusão, nenhum de ideia: `abs()` numa queda com sinal (melhora virou "falha");
exigir quatro codecs vivos na mesma tarefa; medir admissão dentro da amostra;
placebo por `shuffle` que devolve a identidade; placebo com grupos de tamanho 1;
conta que dobrava o erro do oráculo; `nohup` que bloqueava sem erro; estimar o
tamanho do VPT sondando UM arquivo (3,4 MB → na verdade 156 MB por episódio,
776 GB a coleção). O padrão: quando um portão dá resultado forte num sentido,
checar o controle antes de registrar.

Coleta: 355 episódios de ação do VPT (5,3 GB, 1,8 milhão de passos) + 26 com
vídeo, em `D:\agi_dados`, com manifesto de procedência gravado a cada arquivo.

### 17.13 O que vem: acúmulo

O Sistema 1 ainda esquece cada tarefa, e o sonho foi UMA volta. O que falta é o
ciclo: vigília (os dois sistemas tentam), sono (tudo que foi resolvido — regra
explícita OU função aprendida — vira gerador de sonho), e o Sistema 1 volta mais
forte. A pergunta é se a curva compõe volta após volta, medida no conjunto de
AVALIAÇÃO, que nenhuma volta toca.


## 18 · A metade geradora, a mente inteira e a virada para extração (2026-09-24)

### 18.1 Destilação não compõe — `destila.py`

A rede do Sistema 1 VIRA codec do Sistema 2 sem quebrar nada (13/13 reproduzem a
própria origem), mas nenhum conceito absorvido compôs para resolver outra tarefa:
as 2 "novas" aparecem iguais no placebo e não usam codec aprendido. **A2.**
Conceito aprendido de UMA tarefa com 3 exemplos é solução, não conceito;
primitiva reutilizável exige aprender de VÁRIAS tarefas.

### 18.2 ARC-AGI-2 e a cobrança

Núcleo sozinho no ARC-AGI-2 de avaliação: **1/120** (humanos 66%). O usuário:
"estávamos tentando garimpar no ARC-2 como o resto do mercado sem usar 99% do que
tínhamos feito e registrado no ARQUITETURA.md?" Sim — medido por importação, zero
módulos da mente usavam digestora, busca sob demanda, atlas ou linguagem.

### 18.3 A mente como quadro-negro — `mente_inteira.py`

Departamentos escrevem candidatas num quadro comum (`resultados/quadro/`): `s2`
(núcleo, prazo 60 s), `construtor`, `celobj`, `s1` (renascido do sonho). O
CRÍTICO (a invariante mais simples do treino, [[operacao-transfere-arc]]) veta; o
ÁRBITRO dá 2 tentativas, regra explícita antes de intuição. Plugar peça nova não
exige reescrever a mente. Pré-registro: Q0 alarme falso do crítico ≤ 10%; Q1 a
mente supera a melhor peça sozinha.

### 18.4 O construtor — comparar exemplos em vez de buscar — `construtor.py`

Digestora (objetos + nutrientes de cena) → HDC (alinha objeto movido; `Memoria`
guarda a regra e ordena os nutrientes da próxima cena parecida) → CONSTRÓI (tabela
nutriente → destino que vale em todo objeto de todo exemplo) → SIMULA (só responde
se reproduz todos os exemplos).

| versão | o que entrou | ARC-1 treino certas / respostas | placebo (saídas desarranjadas) |
|---|---|---|---|
| v1 | destino por objeto isolado | 8 / 10 | — |
| v2 | raios, tabela de 2 nutrientes, encadeia a FORMA do núcleo | 37 / 45 | 1 / 134 |
| v3 crua | CARIMBO lido do dado | 65 / 113 | **31 / 134 — FALHOU** |
| v3.1 | carimbo só vale se explica mudança em ≥ 2 pares | **61 / 72 (85%)** | 1 / 134 |

ARC-1 avaliação v3.1: 18/23 (78%), placebo 0/92. ARC-2 avaliação: 0 respostas.

O CARIMBO: o diagnóstico da v2 mostrou que, nas tarefas sem conflito de tabela, a
maioria falhava porque a saída CRIA conteúdo (ARC-2: 22 de 27). Cada célula que
mudou é descrita pela posição em relação ao objeto de caixa mais próxima — em cada
eixo antes/dentro/depois e a distância, exata ou "qualquer"; o alvo é o fundo em
volta OU o próprio objeto. A comparação guarda a descrição mais geral que nenhum
objeto contradiz. Leu sozinho "diagonais amarelas em volta do vermelho, cruz
laranja em volta do azul" (0ca9ddb6), "a célula vazia da caixa do L vira azul"
(3aa6fb7a), o anel (4258a5f9). Nenhum teste com nome.

**A lição que vale além do ARC:** a v3 crua passava no portão memorizando — 87
carimbos exatos são cópia da saída, e o placebo pegou (31/134). Regularidade é o
que se REPETE entre exemplos independentes; visto num par só é memória. O
conserto é post-hoc (vi o placebo antes), mas é o próprio princípio do
construtor, e os portões não mudaram. É o princípio anti-envenenamento da
extração de mundo (18.6).

### 18.5 A mente inteira no ARC-AGI-2

```
s2 1 | construtor 0 | celobj 0 | s1 0 | MENTE 1 (0,8%)
crítico: vetou 0 certas, 42 erradas (pega 46,2%)   Q0 passa, Q1 FALHOU
```
No ARC-2 as peças não se complementam. O crítico é a única que funciona como
previsto fora do ARC-1. ARC-1 de avaliação em andamento.

### 18.6 A virada: extrair do mundo humano

O usuário, no mesmo dia: "ARC pra mim é um brinquedo, quero curar doenças,
resolver problemas reais"; o que falta é "extração de nutriente cognitivo" de
mundo humano — vídeo, áudio, gameplay, xadrez, texto — organizado em HDC com
procedência, sonho que simula, "sem um deus escrevendo à mão".

O diagnóstico sincero, com os números do próprio projeto:
1. **Em todo sucesso eu escolhi as variáveis** (h = x·vy − y·vx saiu de x, y, vx,
   vy dados; os objetos do ARC são decisão minha). Em dado cru bateu na parede:
   HAR sem rótulo ~0,70 para toda representação, o fluxo era histograma, a mente
   contínua falhou M0, 24× mais dado deu −0,007. Falta o EXTRATOR, não dataset.
2. **Relevância**: humano extrai o que importa porque age e sofre consequência. A
   linguagem aterrada funcionou por isso; o HAR falhou por não ter.
3. **Primitivas inatas existem**: o gênio da vila tem objeto, número, agente e
   geometria escritos pela evolução. Alvo honesto: poucas primitivas do tipo
   inato, todo o resto extraído — hoje é o contrário.

O princípio de extração proposto no lugar do vocabulário: consumir em fluxo
prevendo o próximo instante; a surpresa corta eventos; candidatos saem do fluxo
(o que se repete, o que se move junto, o antes→depois, a transformação da
transformação); admissão só se REPETE em ≥ 2 episódios independentes E reduz a
surpresa fora da amostra (placebo: fluxo embaralhado); HDC com procedência (real
ou sonhado); o que sai do sonho entra como HIPÓTESE marcada.

Dados de mundo humano, sem cadastro, conferidos por HEAD e baixados com teto e
manifesto (`coleta.py`, fontes novas):

| mundo | fonte | teto | âncora humana cega |
|---|---|---|---|
| gente jogando xadrez | Lichess 2013 + problemas (CC0) | 0,42 GB | temas: garfo, cravada, espeto... |
| gente jogando Atari com o OLHAR | Atari-HEAD, 6 jogos (CC-BY) | 2,5 GB | ação + fixação do olho |
| vídeo egocêntrico com áudio | EPIC-KITCHENS-55 (Bristol) | 4 GB | verbo/substantivo |
| gente em casa, 1ª/3ª pessoa | Charades 480p por Range | 3 GB | 157 ações |

### 18.7 Primeiro laço de extração em dado humano: xadrez — `xadrez_digestora.py`

A máquina recebe as regras, UM contrafactual genérico ("o que esta peça atacaria
se a da frente sumisse") e o tipo das peças sem valor. Não recebe tema, valor nem
noção de lance bom. Cada lance é um EVENTO: as relações que ele cria (a peça que
andou passa a atacar X; outra minha passa a atacar Y; o inimigo passa a atacar a
que andou), abstraídas de casa e identidade. Candidato = relação criada ou par de
relações do mesmo lance. Âncora cega: temas nomeados de 40.000 problemas do
Lichess.

```
relevância = ATRAÇÃO das escolhas humanas (jan/2013, admissão em fev, Y em mar)
  334 admitidos, placebo (lance sorteado, mesmas regras) 0
  âncora X3: 0 de 6 temas | Y: +0,091 na posição do lance humano (IC +0,086 a +0,095)
relevância = DESFECHO ADIADO (peças perdidas nos 4 meios-lances DEPOIS do lance)
  39 admitidos, placebo 0 | C2: só peça pendurada (+0,31); cravada −0,24, descoberta −0,34
```
**O que se conclui:** a REPRESENTAÇÃO está certa — no reservatório, os padrões
mais alinhados a cada tema são a definição de livro, extraída sem nome: cravada =
"ataca o rei ATRAVÉS de cavalo/bispo", espeto = "através da dama até a torre",
ataque descoberto = "OUTRA peça minha passa a atacar". E essas relações preveem o
lance humano num mês nunca tocado. A SELEÇÃO está errada: nem a escolha nem o
desfecho observado de AMADORES privilegiam esses conceitos — amador pega material
e dá xeque, e não converte cravada em peça. Conceito humano tem nome pelo que
produz sob jogo BOM. A relevância tem de vir de consequência SIMULADA
("simular cenários antes de executar"), não de imitar.

Defeitos meus pegos antes de concluir: o contrafactual limpava a pilha de lances
do tabuleiro original; Y medido no mesmo mês que admitia (0,094 → 0,091 limpo); o
desfecho contava a captura do próprio lance, contra o texto pré-registrado (a
primeira rodada deu C2 pelo motivo errado).

Próximo: extrair o VALOR das peças do resultado das partidas (âncora cega
1/3/3/5/9) e medir a relevância pelo desfecho numa continuação IMAGINADA com as
regras.

### 18.8 Xadrez, continuação: valor, imaginação, e o nome aprendido com poucos exemplos

Quatro perguntas mais, cada uma pré-registrada no docstring de
`xadrez_digestora.py` antes de rodar:

```
VALOR das peças, do resultado de 21.000 partidas (sem a tabela humana)   V1
  peão 1 | cavalo 2,05 | bispo 2,51 | torre 3,13 | dama 6,29   (humano 1/3/3/5/9)
  ordem certa, bispo > cavalo como dizem os fortes, escala comprimida; placebo sem ordem
desfecho IMAGINADO (meu lance, toda resposta, capturas até aquietar)     S3 (0/6)
  o ganho IMEDIATO domina: o topo é "toma a dama indefesa"
CAUSA de ganho adiado (só lances sem captura, dados de fevereiro)       D3 (0/6)
  e o placebo dá −0,22 no ataque descoberto: a régua "rho no reservatório"
  tem estrutura própria — parte dos nulos é o instrumento
NOMEAR com 5 exemplos (conservado em ≥ 4 de 5, o mais raro)             N2 (2/6)
  espeto 13,6×, xeque duplo 17,7×; garfo e cravada caem no genérico
NOMEAR com 5 exemplos + atributo EXATO OU QUALQUER, 40.000 problemas NOVOS   N1 (4/6)
  xeque duplo 77× (m>Ik + M>Ik) | espeto 16× | descoberto 9,1× (m>I?) | garfo 2,3× (?>I?*2)
  cravada 1,91 e pendurada 1,97 raspam o 2,0 | superfície 1,1–1,7 | placebo p95 ≤ 2,0
```

**A leitura.** Descobrir SOZINHO o nome que humanos dão falhou com quatro sinais
de relevância (escolha, desfecho observado, desfecho imaginado, causa de ganho) —
e o gênio da vila também aprende "garfo" de um professor. O nome é transmissão
cultural. O que a máquina precisa ter é a REPRESENTAÇÃO certa, extraída do evento,
e COMPARAR poucos exemplos guardando o que se conserva COM VARIÁVEL. A operação
que fez passar é a mesma do carimbo do ARC (cada atributo exato ou qualquer):
atravessou de domínio — mas quem a carregou fui eu.

**Ressalvas.** Seis perguntas em sequência; a variável entrou depois de ver a
quinta falhar, mitigado por dados novos e portões idênticos. Os temas vêm do
etiquetador automático do Lichess. Magnitude de ganho, em qualquer das três
formas, não escolhe conceito — o mesmo que [[cegueira-do-ganho-imediato]] mediu
em mundo sintético, agora em dado humano.

**O laço que ficou de pé:** relações do evento (regra do mundo + um contrafactual
genérico) → comparar poucos exemplos → conservar com variável → reconhecer fora
da amostra. Próximo: o mesmo laço em mundo PERCEBIDO, onde as relações não vêm
prontas das regras (Atari-HEAD com o olhar como relevância; vídeo).

### 18.9 A mente inteira no ARC-1 de avaliação

```
s2 44 | construtor 18 | celobj 1 | s1 18 | MENTE 57/400 (14,2%)   Q1 PASSA (+13 sobre a melhor peça)
crítico: vetou 0 certas, 96 erradas
```
No ARC-1 as peças SE COMPLEMENTAM (no ARC-2, 1/120, não). Defeito meu achado no
caminho: o árbitro lia `construtor*.jsonl` e pegava os backups `_v1/_v2` — a
primeira contagem (54) usava o construtor v2. Backups agora em `antigos/`.

### 18.10 Primeiro mundo PERCEBIDO: onde o humano olha — `atari_digestora.py`

Atari-HEAD, 6 jogos, olhar humano como gabarito cego; a máquina recebe pixels,
ação, recompensa, COESÃO (região conexa não-fundo é objeto) e CONTINUIDADE. Extrai
sozinha o EU (contingência ação → movimento) e "o que informa a escolha" (o tipo
cuja posição relativa ao EU prevê a próxima ação). Treino numa sessão, teste noutra.

```
                 mais se move   informa a escolha   EU
breakout            20,4%            8,7%           1,2%
freeway             14,6%            8,6%          13,8%
montezuma            6,0%            0,0%           0,1%   (maior objeto 18,5%)
ms_pacman           13,8%           18,8%          23,9%
space_invaders      10,1%            0,2%           3,4%
seaquest            26,7%            4,7%           9,5%
VEREDITO A3 (0 de 6); escala do olhar conferida em 6/6
```
O EU saiu certo em ~4 de 6 (raquete, galinha, Pac-Man, submarino — leitura pela
cor, não portão). Mas a relevância de cima para baixo perde da saliência de
movimento: o TIPO aprendido por aparência (cor dominante + forma) quebra quando o
objeto sai borrado, e aí o alvo some. Quatro defeitos de instrumento foram pegos
na fumaça, antes da rodada que vale (acerto pela caixa, leitura do tar, objeto
por cor exata, viés da informação mútua).

Lição para a digestora de mundo: IDENTIDADE vem da continuidade (a trilha do
objeto no tempo), não da aparência; e a pergunta justa é se a relevância extraída
ACRESCENTA à saliência, como o nutriente acrescentou à superfície no xadrez.

## 19 · Inventar a própria representação do cru (2026-09-24 a 25)

O problema entregue em 24/09: falta o GERADOR de variáveis — a máquina inventar a
própria representação de um mundo novo a partir do cru. Projeto em
`D:/AGI_Inventor/representacao` e `D:/AGI_Inventor/instrumentos`.

- **Etapa 1 — órbitas em pixels até h:** de quadros crus a máquina chega a
  h = x·vy − y·vx em 3/3 mundos; a revisão da percepção importa em 2/3 (a
  revisão guiada pela descoberta só foi decisiva em 1).
- **Etapa 2 — Breakout observacional (Atari-HEAD): FALHOU** (relação 0/3, revisão
  0/3). Julguei a revisão pelo critério errado (conservação).
- **Instrumentos inventados (ALE, Atari 100k, braços U/C/I/P):** v1 FALHOU no
  desenvolvimento (I0/I1/I2). Inventa 16 variáveis, mas o placebo também; os
  sensores acharam a própria raquete, não a relação. Jogos lacrados guardados.
  Diagnóstico externo depois: a representação base descartava informação (direção
  da bola), experimentos de 8 passos curtos demais, FQI com 5 iterações.

## 20 · O organismo (2026-10-05) — `D:/AGI_Inventor/organismo`

As ideias do usuario (mesoestado, sem números mágicos, coalizões, regime, procedência
em 2 eixos, sonho) como UM organismo: tudo é estrutura com registros (contexto,
pagou?) → crença Beta por contexto (núcleo HDC), Thompson valor/custo.
- **Vida 1:** G1 NULO contra o pipeline (0,640 × 0,641), mas troca forças: ganha onde
  o conceito é local/ficção (magnético 0,52 × 0,07), perde o h universal (o contexto
  bloqueia conceito universal). Conceito sobre conceito 6/10 × 1/10.
- **Vida 2** (custo da invenção corrigido, contexto mais largo): **bate o pipeline
  +10,3 pontos em 20 sementes (17/20)**. Aprender o alcance = largura fixa boa.

## 21 · A mente na internet real (2026-10-05 a 08) — `D:/AGI_Inventor/mente_web`

Pedido firme: porta aberta, não corpus. Corpo só-leitura (GET, robots.txt, sem
login/formulário de envio), texto como BYTES (nenhum tokenizador), vídeo+áudio em
fluxo no mesmo relógio, nada guardado além do que ela aprende.

- **v0 (5 min):** presa pesquisando "port", "ver", "que". Guardada como 1ª criança.
- **v1:** pergunta = estado cognitivo incompleto (4 tipos: o que vem depois, não
  antecipado, uso divergente, ligação com imagem); memória de intenção com 8
  lacunas-controle pareadas; hierarquia (pedaços → pedaços de pedaços, Dunning);
  multimodal que só se confirma prevendo, com nulo; formulário GET de 1 campo vira
  barra que ELA descobre. Defeitos meus pegos nas fumaças (registrados): régua de
  "entendido" que sofria interferência (agora = prever os PRÓXIMOS encontros),
  falha mecânica diluída entre contextos ("funciona?" separado de "pagou?"),
  KAPPA fixo (agora Bayes empírico). Wikimedia: User-Agent sem contato = 10
  pedidos/min (contato do projeto autorizado depois).
- **v2 — plasticidade local** (ideias dos prêmios de 2026): traço local que decai,
  sono disparado pela incerteza acumulada, **optogenética interna** (no sono
  re-percebe a experiência recente COM e SEM cada junção marcada e liga/desliga
  pelos bits — sem Dunning), **imaginação composicional com placebo** (troca uma
  peça pela vizinha de uso e prevê combinações nunca lidas). Fumaça: 299 ligadas,
  451 rejeitadas, 8 desligadas depois (estado intermediário reversível).
- **Gêmeas v1 × v2** vivendo ao mesmo tempo, 6 h, pré-registradas (M2–M7, P1–P4),
  bateria fixa de 55 páginas + 40 imagens (sha256 FE67C087). Em ~3 h: 2–4 mil
  páginas, 240–480 sites, hierarquia até nível 7 ("Asia Interactive Time Zone"),
  custo de entender texto novo 4,7 → ~1,8 bits/letra, imaginação confirmada 14 ×
  6 do placebo (cedo). Problemas vistos nos dados: viram só 2 imagens e 2 vídeos;
  67–75% das ações eram perguntas pagando 1–15%; a v2 ficou presa em sites
  enormes (0 sites novos em 30 min, ganho ~0).
- **Transplantes na vida contínua** (as gêmeas ficam congeladas para a comparação):
  (1) curiosidade PRÓPRIA da visão (progresso de compressão nos retalhos), porta
  de imagem, utilidade de ferramenta aprendida em geral; (2) régua de "valeu a
  pena" da VIDA, não das últimas 200 páginas (adaptação hedônica), porta de vídeo;
  (3) olhos e ouvidos que distinguem — o protótipo virava a média de tudo (1 padrão
  visual nas gêmeas; agora 10 de 5 tipos de imagem).
- **Vida contínua:** a vencedora das gêmeas (regra registrada antes) é copiada para
  `vidas/viva`, guardião com reinício, salvamento atômico, sem-rede = ela espera e
  consolida, foto diária comprimida, início automático no Windows.
- **Painel (Jarvis):** visão geral (reator, consumo, curva), ORGANISMO (anatomia
  para leigos; clicar no órgão abre o que ele está fazendo de verdade: decisão com o
  valor de cada opção, link clicado, a frase onde cada pergunta nasceu, árvore da
  hierarquia, testes do sono em bits, imaginação × placebo, palavras visuais, mapa
  do mundo dela), O QUE ELA VÊ (o trecho lido colorido pelo que ela reconhece, a
  imagem/vídeo, o quadro 32×32 como ela vê, e as palavras acesas no mapa dela).
  Princípio do "decodificador universal": cada coisa interna é mostrada no sentido
  onde nasceu (texto → letras, visão → pixels, som → bandas) e cada composição pelas
  suas partes; operador novo se mostra pelo EFEITO (antes → depois).

## 22 · Tornar-se X: o ponto de vista como operação (2026-10-08) — `D:/AGI_Inventor/encarnar`

Ideia do usuario (autoprojeção: "o cientista que se sente a célula"). Arena com 5
seres (mola, ímã, satélite, persegue, foge, formação; reserva: intercepta,
distância); a mente recebe só posições e uma operação: re-ver o mundo do ponto de
vista de cada ser e escolher, por bits, de onde o movimento fica simples; lei por
mínimos quadrados ortogonais. Lacrado (60 mundos, 300 seres): **de quem é a ação
94% (acaso 25%)**, placebo escolhido 1,7%, taxonomia objeto/agente/ligado sem nome
93%/100%, comportamentos nunca vistos 70% e 2× melhor previsão; **A3 FALHOU**
(1,30× < 1,5×: fugitivo que já virou as costas corre reto, nada a explicar).
Defeitos pegos em dev: guloso por correlação (→ OLS), placebo de giro constante que
coincidia com o ímã, R² que engana quando o ser parou (régua trocada antes do lacre).

## 23 · Onde estamos e os próximos órgãos (2026-10-08)

O que a mente viva tem: corpo na internet real, sentidos sem semântica, hierarquia
própria, perguntas como estado, intenção medida contra controle, plasticidade local
com autointervenção, imaginação medida contra placebo. O que ela NÃO tem ainda:
significado (palavras ligadas ao que descrevem), um motivo para falar, e olhos à
altura (32×32 cinza). Próximos órgãos, a transplantar com ela viva:
1. corpo próprio e modelo de si na arena → entender os outros ("se eu fosse X");
2. escola de linguagem na arena (palavras aterradas nos pontos de vista) → frases
   novas medidas contra placebo;
3. forrageio: valor esperado de FICAR num ambiente × sair (teorema do valor
   marginal, Charnov 1976) — a régua da vida resolve a armadilha, mas não aprende
   quando um ambiente genuinamente diferente vale a pena;
4. olhos melhores (mais resolução, cor, movimento) sem semântica pronta.
