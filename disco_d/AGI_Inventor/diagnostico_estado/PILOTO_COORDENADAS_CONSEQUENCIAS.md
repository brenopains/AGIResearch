# Crédito aprendido e coordenadas de consequências

Registro escrito antes de gerar planos desta rodada. O treino é somente 97.
103 foi usado em muitos diagnósticos e é desenvolvimento. Não apresentar
ganho aqui como confirmação ou novidade científica. Nenhum corpo ou veredito
anterior muda; todo código novo fica no D.

Intervenção: aprender funções numéricas que estimam recompensa e continuação
de oito passos para cada ação, a partir dos estados e desfechos realmente
observados. Seus parâmetros completos e definição ficam exportados, sem
Python arbitrário ou só um identificador/vetor. A geometria acrescenta essas
funções à representação atual. Normalizar as duas partes pela mediana de
distâncias em até 512 estados de treino, com peso igual, sem usar 103.

Quatro braços recebem a mesma busca de 16 blocos/largura8: crédito aprendido
com geometria atual; crédito aprendido com coordenadas reais de consequência;
crédito aprendido com coordenadas cujo treino foi embaralhado dentro da ação;
crédito aprendido com coordenadas aleatórias lineares de dimensão igual.
Nos quatro, recompensa do suporte é a previsão da mesma função aprendida e
término continua sendo a transição observada. A função de crédito e a de
geometria são distinguidas para controlar a informação usada na distância.

Comparar também com todos os braços originais do piloto anterior, que já
têm retornos medidos. Gerar e armazenar todos os planos/hashes antes de
executar qualquer plano novo. A geração recebe só dois quadros iniciais;
retornos e estados futuros ficam fora dela. Certificar 64 históricos, 256
ramos curtos e replay completo de um plano por episódio. Teto65.000 passos
ALE por jogo, até dois processos. Planos idênticos compartilham uma medida;
eventos de reuso terão custo físico zero explícito.

Primário: retorno real do braço de coordenadas reais contra cada braço novo
e cada braço anterior. Só abrir confirmação nova se houver ganho médio e
limite inferior bootstrap95% acima de zero contra todos, melhora em pelo
menos dois episódios contra cada um, sem piora por episódio contra ação fixa.
Bootstrap de oito episódios, 10.000 amostras, RNG9773. Sem margem ou contraste,
inconclusivo. Reportar custos, erros de previsão e planos não constantes.

Se o crédito melhorar erro, mas não decisão, não promover. Se a geometria
aleatória ou embaralhada vencer, rejeitar a interpretação de que a função
aprendida é responsável pelo ganho. Não reajustar escala, peso ou busca
depois de ver os resultados. Esse corpo ainda opera sobre um estado visual
aprendido por compressão: não inventa sozinho objetos ou variáveis universais.

```powershell
py -3.11 -B testes_coordenadas_consequencias.py
py -3.11 -B piloto_coordenadas_consequencias.py breakout --propor
py -3.11 -B piloto_coordenadas_consequencias.py breakout --executar
```

Repetir para `pong`. Propostas, definições, parâmetros, medidas e relatórios
em `resultados/{jogo}_coordenadas_consequencias_103*`.
