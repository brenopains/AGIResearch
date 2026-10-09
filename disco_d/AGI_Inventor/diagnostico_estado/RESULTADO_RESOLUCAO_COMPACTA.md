# Comparação espacial concluída: capacidade geral não demonstrada

Concluí o teste que antes havia parado no orçamento. O ajuste completo terminou
em **92,57 segundos**, com corpos congelados antes da experiência nova 179.
As quatro células tiveram cobertura suficiente e **nenhuma passou o portão**.
Isso delimita esta família PCA/RBF/transporte linear; não prova impossibilidade
de aprender representações, uma causa única do erro ou proximidade humana.

O projeto original não foi editado. ARQUITETURA.md manteve SHA256
`da47cc53b795305f416e22dc6f120607919000d1e224f6d4e4cf88f2ac5ba8ad`.
Tudo desta rodada fica em `D:\AGI_Inventor\diagnostico_estado`.

## Implementação e verificação

`transporte_compacto.py` ordena/deduplica linhas por chaves binárias que mantêm
a ordem numérica, sem hashes, e reutiliza normas e a matriz duplicada durante
a seleção de suportes. Não altera funções ou alvos. Seis testes passaram.
Quatro pares cronometrados, alternando a ordem das versões nos dois conjuntos
reais, mostraram ajuste **2,71 a 3,09 vezes mais rápido**, com suportes, escala,
coeficientes e previsões preservados nas instâncias verificadas. Aceleração
computacional não equivale a ganho de inteligência.

`piloto_resolucao_compacto.py`, `coletar_resolucao_compacto.py` e o auditor
correspondente completaram ambos os jogos e todos os controles. O protocolo
`RESOLUCAO_COMPACTA.md` foi congelado antes do ajuste. Os dez modelos de sensor
foram refeitos integralmente, com dois placebos embaralhados ajustados e duas
ablações por cópia do corpo aprendido. Não houve ajuste durante avaliação.
Reusaram-se os bancos certificados 149/151; desenvolvimento 151, já conhecido,
não foi executado. A tentativa anterior, seus doze corpos parciais e custos
continuam separados. Nenhum corpo parcial foi promovido.

## Resultado reservado, horizonte 16

Menor MSE e arrependimento são melhores. O primário precisa vencer todos os
controles nas duas métricas em todas as células. Empates não passam. Não foi
escolhido outro sensor após observar resultados; horizontes 4/8 são secundários.

| Jogo / programas | MSE aprendido | MSE média | MSE aleatório | MSE zero | Arrependimento aprendido | Sem contexto | Aleatório | Portão |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Breakout / conhecidos | 0,68289 | 0,67202 | 0,70463 | 0,98438 | 0,140625 | 0,140625 | 0,140625 | Falhou |
| Breakout / novos | 0,74528 | 0,76553 | 0,78041 | 0,79688 | 0,250000 | 0,203125 | 0,265625 | Falhou |
| Pong / conhecidos | 0,84270 | 1,01747 | 0,82312 | 0,93750 | 0,140625 | 0,218750 | 0,109375 | Falhou |
| Pong / novos | 1,25914 | 1,22894 | 1,18259 | 1,12500 | 0,296875 | 0,281250 | 0,281250 | Falhou |

Em Breakout conhecido, a média tem MSE menor e vários controles empatam na
decisão. Nas composições novas há MSE local menor, mas arrependimento maior
que média, sem contexto e corpo antigo. Em Pong conhecido, o aleatório vence
o aprendido nas duas métricas. Em Pong novo, zero e o placebo embaralhado
superam o aprendido nas duas métricas.

Filtros de Pong reduzem localmente o arrependimento de Breakout novo para
0,15625; filtros de Breakout ficam em 0,3125 em Pong novo. Esse braço era
secundário. Não autoriza seleção retrospectiva, capacidade geral ou transferência
para mundo desconhecido. Pesos do modelo temporal continuam aprendidos por jogo.

Cada célula teve oito episódios com contraste e ambas as preferências por par.
Comparações não empatadas: 42/34/22/17, respectivamente. São 64 casos por
célula/horizonte; raízes e horizontes são correlacionados. As **6.912 linhas**
incluem nove braços, três horizontes e quatro células, não 6.912 experiências
independentes. Todos os controles e critérios estão no JSON do resultado.

## O contexto muda escolhas, mas não ajuda consistentemente

O diagnóstico posterior `diagnosticar_contexto_resolucao_v2.py` reconcilia as
6.912 métricas sem ajuste nem coleta. Compara o primário à ablação do mesmo
corpo aprendido. Separadamente calcula a melhor escolha constante por par,
usando respostas reservadas: um **oráculo retrospectivo**, não uma política
aprendida ou implantável. Um agente contextual poderia superá-lo; teste
unitário verifica essa possibilidade.

| Célula | Arrependimento aprendido | Constante retrospectiva | Ganho contra mesmo corpo sem contexto | Escolhas alteradas | Ajudaram / prejudicaram / neutras |
|---|---:|---:|---:|---:|---|
| Breakout / conhecidos | 0,140625 | 0,140625 | 0,000000 | 4 | 2 / 2 / 0 |
| Breakout / novos | 0,250000 | 0,203125 | -0,046875 | 22 | 2 / 6 / 14 |
| Pong / conhecidos | 0,140625 | 0,125000 | +0,078125 | 26 | 6 / 2 / 18 |
| Pong / novos | 0,296875 | 0,218750 | -0,015625 | 13 | 3 / 2 / 8 |

O primário varia as escolhas conforme a imagem; não é literalmente constante.
Há benefício contra sua ablação em uma célula, mas piora nas composições novas.
Não supera a constante retrospectiva em nenhuma célula nesta amostra. Assim,
mais coeficientes sensoriais não mostraram uso robusto do contexto. Contar
escolhas corretas difere de pesar consequências: três mudanças benéficas e
duas prejudiciais ainda podem aumentar arrependimento em Pong novo.

Corrigi um nome no diagnóstico preservando a primeira versão: o campo
`gain_over_same_body_without_context` era válido só para `learned`. Os outros
braços eram comparados à ablação do primário, não à própria. A v2 usa
`gain_over_learned_body_without_context` e declara esse escopo. Nenhum número
ou resultado principal mudou. Cinco testes da v2 passaram; os quatro testes
da primeira versão permanecem registrados.

## Auditoria e contabilidade

Auditoria passou: reconstrução dos seis bancos, origem dos filtros pela
covariância, suportes observados, escalas de treino, equações ridge por
Cholesky, placebo, ablação, composição independente, casos, métricas,
cobertura e veredito. Erro máximo de reexecução `1,42e-14`; diferença máxima
nos coeficientes `2,72e-12`. Verificações numéricas, não provas exatas do mundo.

| Trabalho | Chamadas ALE novas | Tempo medido | Observação |
|---|---:|---:|---|
| Comparação compacta pareada | 0 | 54,06 s | Oito ajustes antes desta rodada |
| Ajuste completo dos sensores | 0 | 92,57 s | Dez modelos e dois placebos ajustados |
| Breakout 179 | 33.769 | 26,27 s | Oito replays, 32 chunks |
| Pong 179 | 40.917 | 27,49 s | Oito replays, 32 chunks |
| Diagnóstico de contexto v2 | 0 | 0,057 s | Descritivo, sem aprendizagem |

Avaliação: **74.686 chamadas novas**, 298.744 quadros com salto 4, cerca de
82,98 minutos a 60 quadros/s somando ramos e replays. Os bancos reusados
custaram antes 91.398 chamadas e 67,99 segundos de coleta. Auditoria soma
166.084 chamadas históricas e 48 replays, dos quais 16 são desta avaliação.
Tempo de cálculo não é orçamento de experiência humana.

A tentativa incompleta anterior consumiu pelo menos 120 segundos de ajuste.
Perfil original e primeira otimização sem ganho confirmado ficam separados;
esse esforço não foi apagado. Tempos totais do auditor e avaliador não foram
instrumentados no resultado; não apresentar soma parcial como custo total.
Maior RSS ao exportar modelos: 622.612.480 bytes (~594 MiB), amostragem,
não pico completo ou contenção rígida. Um processo próprio por vez, duas
threads BLAS; nenhum processo do usuário interrompido. Esta rodada terminou.

## Decisão e limites

Não há evidência de uma única peça faltante ou de proximidade humana. O teste
mostra algo mais específico: **preservar variação visual por filtros aprendidos
não bastou para construir o estado que muda corretamente as consequências
de uma ação composta**. O próximo alvo é estado orientado por diferenças de
consequência sob intervenções, distinguindo percepção, leitura de eventos e
propagação/composição. Isso ainda é hipótese; não isolamos uma causa única.
Pares prospectivos anteriores também falharam: repetir essas funções com
novos nomes ou mais filtros não é um mecanismo novo.

Antes de outra aposta, exigir mesma pergunta de ação em situações de respostas
opostas; distinções construídas só no treino; vantagem sobre a própria ablação
e controles na experiência nova; composições reservadas. Diagnóstico com
estados reais futuros deve ser separado: estão indisponíveis ao agente e
medem propagação, não compreensão antecipada. Comparação humana e transferência
útil para mundos realmente desconhecidos continuam ausentes. Não promovemos
nenhum corpo como capacidade geral. O objetivo humano não foi atingido.

## Fontes e reprodução

Comandos em `RESOLUCAO_COMPACTA.md` e `CONTEXTO_RESOLUCAO_V2.md`.
Saídas existentes recusam sobrescrita. Dados/eventos: prefixo
`resultados/resolucao_compacta`, com corpos, checkpoints, bancos, JSONL e auditoria.

- Resultado SHA256: `16c7606515600b957d7e09242b389a69a653d030306e089226901c5b93291639`.
- Protocolo SHA256: `8a471dcad67106696c786e26d1f4a7ee33ed36c0a8c9b66ae90959d21771578b`.
- Corpos SHA256: `1051fa456ec39cfde15bb5a88aaaae0403b5e0dae83f8a80cfb0caf88ada1bf7`.
- Diagnóstico v2 manifesto: `d130eabb18cc98f331573abbfcdc7debfd05db66b1b7fd7f4e69c484578255ca`.
