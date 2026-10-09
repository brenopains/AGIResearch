# Hipótese: desenvolvimento cognitivo com conectomas plásticos

Data: 2026-10-03. Estado: proposta técnica; importadores, simulador e comparação ainda não implementados. Nenhuma capacidade nova foi demonstrada por este documento. Projeto original preservado.

## Objetivo e hipótese falsificável

O objetivo permanece uma mente artificial de nível humano ou superior. A hipótese operacional é mais específica: regras de aprendizagem e reorganização, selecionadas em diversos mundos e acopladas ao motor existente, podem construir representações úteis em mundos novos a partir de experiência crua, com menos experiência que os controles.

Os conectomas entram como estruturas iniciais alternativas. A hipótese não pressupõe que um fragmento humano contenha inteligência humana, nem que mais complexidade produza inteligência automaticamente. O que deve evoluir é também o procedimento que aprende, forma variáveis e cria conexões.

## Fontes biológicas e limites

| Fonte | Material disponível | Papel experimental |
| --- | --- | --- |
| Humano H01 | Fragmento de córtex temporal de aproximadamente 1 mm³, com células, sinapses e reconstrução anatômica | Prior anatômico humano; registrar conexões cortadas na borda |
| Camundongo MICrONS | Volume cortical com anatomia e respostas funcionais a estímulos visuais; correspondência entre ambas para um subconjunto | Prior mamífero e possibilidade de calibrar dinâmica contra respostas observadas |
| Mosca FlyWire | Conectoma de cérebro adulto com tipos celulares e conectividade | Prior de circuitos recorrentes; versões e graus de revisão explícitos |
| Outros animais | Adicionar somente após verificar acesso, resolução e completude | Não equiparar conectividade entre regiões a um mapa de sinapses |

Fontes primárias: [H01](https://h01-release.storage.googleapis.com/landing.html), [publicação H01](https://research.google/pubs/a-connectomic-study-of-a-petascale-fragment-of-human-cerebral-cortex/), [MICrONS](https://www.microns-explorer.org/cortical-mm3), [FlyWire](https://www.nature.com/articles/s41586-024-07558-y).

Importar primeiro tabelas de conexões e metadados de subconjuntos identificados, sem baixar toda a microscopia. Um subconjunto mede somente a hipótese sobre aquele subconjunto. Guardar snapshot, hashes, identidade celular, revisão, tipos, bordas e parâmetros ausentes. Não transformar contagem anatômica de sinapses em peso funcional conhecido.

## Três escalas de mudança

1. Atividade rápida: estado recorrente, memória de trabalho, integração temporal e comunicação entre módulos.
2. Aprendizagem durante a experiência: pesos plásticos, traços de elegibilidade, modulação por erro e resultado, consolidação, criação e poda de conexões sob limites explícitos.
3. Evolução entre indivíduos: modificar os programas de plasticidade e crescimento, parâmetros dinâmicos e interfaces. Selecionar pela capacidade de aprender novas tarefas de desenvolvimento, incluindo famílias ausentes da experiência anterior daquele indivíduo.

Um genoma inicial deverá definir `atividade`, `plasticidade`, `crescimento`, `consolidacao` e `interface`. As operações admissíveis desses programas continuam sendo escolhas humanas e precisam ser documentadas. Uma regra local inicial pode combinar atividade pré/pós-sináptica, traços históricos e erro modulador; a evolução altera essa combinação. Criar ou eliminar conexões exige propostas explícitas, controle de estabilidade e contabilidade. Recompensa sozinha não implementa reorganização.

Após a seleção em desenvolvimento, congelar o genoma e o algoritmo antes dos mundos finais. Neles, permitir somente a aprendizagem durante a experiência prevista pelo protocolo. O avaliador final não participa da seleção evolutiva.

## Acoplamento bidirecional com a arquitetura existente

Fluxo: ambiente real → observações e ações → circuitos e motor existente → propostas → execução/verificação independente → resultados e contraexemplos → aprendizagem.

- O motor oferece exemplos observados, memória pertinente, operadores executáveis disponíveis, objetivos e resultados de verificações. Não oferece estado oculto do mundo, identificação da família nem solução reservada.
- Os circuitos oferecem estado perceptivo aprendido, previsões, incerteza calibrada, propostas de experiências e funções que transformam histórico bruto em variáveis utilizáveis pelo motor.
- Uma representação é um executor identificado por hash, com entradas, saídas, estado e limites definidos. Pode ser o próprio circuito congelado e sua interface; não precisa ser traduzida para uma fórmula simbólica para ser executável. Tradução simbólica é um resultado adicional a medir.
- A digestora recupera evidências, experiências e definições pertinentes. HDC ordena e referencia; programas, circuitos e evidências permanecem armazenados integralmente fora do vetor.
- O verificador confere resultados contra observações independentes. A concordância entre o simulador e o motor não basta para confirmar uma hipótese.

Contratos mínimos: `Experiencia(observacao, acao, resultado, tempo, proveniencia)`, `Representacao(executor_hash, tipos, estado, evidencias)`, `Proposta(experimento_ou_programa, dependencias, custo)`, `Feedback(resultado_observado, contraexemplo, alcance)`. Eventos de mudança de pesos, crescimento, poda e reutilização são registros reais; não descrições inventadas.

## Partes separadas e unidas

Comparar humano, camundongo e mosca separadamente; pares acoplados; trio acoplado. Priorizar comparações em etapas para evitar explodir o custo.

Unir por interfaces aprendidas, com limite de largura de banda e atrasos, preservando a identidade de cada módulo. Não conectar IDs de neurônios arbitrariamente e presumir comunicação semântica. Não atribuir antecipadamente raciocínio ao humano ou percepção à mosca. A especialização deve aparecer nos resultados e nas intervenções.

Cada comparação precisa de controles com custo equivalente: topologia embaralhada preservando graus e tipos quando disponíveis; rede artificial esparsa; módulos sem comunicação; motor sozinho; circuitos sozinhos; híbrido com plasticidade congelada; híbrido sem crescimento. Calibrar custo medido, não somente número de neurônios. A combinação precisa ganhar do conjunto desconectado com orçamento igual para sustentar vantagem do acoplamento.

## Pressão seletiva e experiência

Usar mundos de dinâmica, interação por ação, sequências sonoras e relações computacionais, intercalados. Intercalar experiências não exige processar todos os domínios simultaneamente. Separar famílias de regras, além de aparências e sementes. Adaptadores básicos de pixels, áudio e números devem ser iguais entre variantes e não entregar as variáveis procuradas.

Selecionar por aprendizagem com poucas observações, previsão sob intervenções, resolução, retenção e reutilização. Contabilizar simulação, seleção, criação de circuitos, verificações e execução de representações. Preservar diversidade de comportamento para não eliminar todos os intermediários sem ganho imediato. Nem quantidade de conexões nem surpresa isolada recebem o rótulo de inteligência.

O custo de preparação evolutiva fica separado do custo de aprender um mundo novo. Comparação com eficiência humana exige novatos humanos, mesma informação e ações, e orçamento de experiência comparável; minutos de CPU não equivalem a minutos de experiência.

## Resultado que realmente sustentaria a aposta

Com algoritmo congelado, em famílias finais nunca usadas na seleção, o agente deve criar uma representação a partir das observações, melhorar previsão e decisão e reutilizar seu executor em outra tarefa sem reescrever o corpo. Removê-la deve reduzir o ganho; restaurá-la deve recuperá-lo. Transferência entre modalidades exige teste próprio, não se deduz de duas tarefas visuais.

Placebos temporais ou causais precisam ser descritos com cuidado: destruir a estrutura relevante deve remover a vantagem que se atribuía a ela, sem exigir recompensa absoluta zero em qualquer ambiente. Embaralhar pode criar uma tarefa diferente e requer pareamento de dificuldade e informação.

Se houver ganho sem efeito de remoção da representação, a interpretação de representação causal não passa. Se somente os mundos de desenvolvimento melhorarem, a interpretação de generalização não passa. Se redes artificiais igualarem o efeito, a contribuição do prior biológico não está demonstrada, mas o mecanismo de aprendizagem ainda pode ser útil. Se somente a união maior vencer, repetir com custo equivalente antes de alegar sinergia.

Falhas devem separar: integridade dos dados, estabilidade dinâmica, comunicação, plasticidade, aquisição de representação, uso pelo motor e transferência. Essa separação reduz ambiguidades; não identifica com perfeição uma causa única quando várias interagem.

## Precedentes e próxima implementação

[Miconi 2023](https://proceedings.mlr.press/v202/miconi23a.html) já demonstra evolução, plasticidade e neuromodulação para adquirir tarefas cognitivas simples inéditas durante evolução. [Plasticidade diferenciável](https://proceedings.mlr.press/v80/miconi18a.html) também permite otimizar mecanismos plásticos. Esses resultados apoiam a viabilidade de partes do mecanismo; não demonstram mente humana nem asseguram benefício de conectomas mistos.

Próxima entrega executável: contratos e importador de um subconjunto real, simulador recorrente esparso com parâmetros explícitos, interface com uma cópia isolada do motor, e testes de procedência, estabilidade, atualização plástica e remoção/restauração. Depois, piloto com limite de 30 minutos e checkpoints para medir custo e detectar erros; ele não é a confirmação de AGI. Expandir espécies e tamanhos conforme resultados pareados, preservando o objetivo de aprendizagem geral. Nenhum download ou treinamento novo foi executado ao escrever esta proposta.
