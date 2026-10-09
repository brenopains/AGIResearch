# Distribuição condicional de desfechos conjuntos

Registro anterior à geração/execução desta rodada. Não altera nenhum ensaio
anterior. Treino97, desenvolvimento103 nos dois jogos. A semente103 já foi
vista extensamente; sucesso aqui exigiria confirmação independente.

Hipótese: escolher experiências pela distância do estado de origem é uma
ligação inadequada entre estado e consequência. Extrair categorias das
consequências observadas e aprender sua probabilidade condicional pode
permitir compor melhor. Cada categoria mantém próximo estado, recompensa e
término juntos, incluindo o custo da aprendizagem e o corpo completo.

Categorias: separar somente pelos valores observados de recompensa e término;
em cada grupo, escolher até16 sucessores por cobertura de distância (primeiro
mais próximo da média, depois mais distante dos já escolhidos). Não fornecer
objetos, regras ou classes semânticas. Aprender kernel de estado anterior para
distribuição sobre essas categorias. Na consulta, cortar valores negativos e
normalizar; se a massa for zero, usar a frequência do treino. Essa escolha é
uma hipótese numérica conhecida, sem reivindicação de novidade científica.

Três braços, mesmas categorias e mesma busca16blocos/largura8:

- distribuição condicional aprendida (primário);
- associação origem/categoria embaralhada dentro de cada ação, preservando
  frequências e eventos conjuntos;
- frequência das categorias, independente do estado.

Também comparar retornos já medidos de todos os braços dos pilotos de planos
e de coordenadas. O orçamento da busca é igual entre os três braços novos.
Custos físicos novos e custos anteriores reutilizados ficam separados.
O braço de frequência executa o mesmo ajuste inicial, mas ignora o estado
ao consultar; não tem vantagem de coleta ou maior orçamento de planos.

Antes de medir qualquer plano novo, salvar planos de64raízes por jogo,
definições, parâmetros completos e hashes das fontes/treino/referências.
Verificar exportação/reexecução dos corpos e concordância entre valores da
busca e composição independente das palavras. Geração recebe só dois quadros
iniciais, sem futuros ou retornos103. Certificar64históricos e256ramoscurtos;
reexecutar o plano primário completo na primeira raiz de cada episódio.
Até dois processos, teto65.000 passos ALE por jogo. Medidas idênticas por
raiz são compartilhadas com custo físico zero explícito no reuso.

Portão fixo: retorno real primário precisa ter ganho médio e limite inferior
bootstrap95% estritamente positivos contra cada braço novo e cada referência,
melhorar em pelo menos dois episódios contra cada um e não piorar em nenhum
episódio contra ação fixa. Bootstrap de oito episódios, 10.000 amostras,
RNG9787. Sem margem real, inconclusivo. Não alterar categorias, busca ou
portão depois dos resultados. Nenhum resultado local recebe rótulo de AGI.

Em caso negativo, diagnosticar previsão de estado, recompensa e término nas
trajetórias reais mistas já coletadas. Esses futuros servem para medir erro,
nunca para escolher ações nas raízes. Não promover um braço de comparação
posteriormente por conveniência nem aumentar experiência para resgatar o teste.

```powershell
py -3.11 -B testes_desfechos_aprendidos.py
py -3.11 -B piloto_desfechos_aprendidos.py breakout --propor
py -3.11 -B piloto_desfechos_aprendidos.py breakout --executar
```

Repetir para `pong`. Saídas: `resultados/{jogo}_desfechos_aprendidos_103*`.
