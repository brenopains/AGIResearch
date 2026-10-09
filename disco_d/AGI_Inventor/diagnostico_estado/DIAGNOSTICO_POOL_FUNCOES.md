# Diagnóstico do pool de funções congelado

O piloto gerou candidatos, mas não promoveu nenhum. Sua auditoria passou
com o bootstrap que configura BLAS antes de importar NumPy. A seleção em
Pong não teve contraste de decisão; em Breakout candidatos melhoraram apenas
um caso. Esta análise identifica o alcance dessas duas limitações.

Todos os 32 candidatos de cada um dos quatro geradores, nos dois jogos,
serão ajustados individualmente sobre construção97/episódios0–3, com os
mesmos suportes, regularização0,05 e operadores de ação. Validação97/4–5
permanece fixa. Nenhum corpo novo será promovido ou salvo como solução.
Zero chamadas ALE; teto120segundos. A análise usa desenvolvimento conhecido,
não confirmação nova ou teste de inteligência humana.

Para cada candidato medir:

- Erro local de recompensa nas histórias realmente observadas na validação.
- Erro de transporte nas33coordenadas antigas e na coordenada nova ativa,
  separadamente; não comparar médias com denominadores diferentes.
- Decisão composta e sua decomposição telescópica exata em recompensa,
  transporte e fronteira terminal, incluindo os termos cruzados.
- Leitura que recebe todas as histórias futuras reais e remoção ideal de
  erro de recompensa. Ambas são oráculos de diagnóstico, não capacidades.
- Casos em que algum candidato melhora a decisão e casos que nenhum melhora.
  Escolher o melhor candidato por resposta verdadeira é outro oráculo.
- Diversidade dos episódios e âncoras de origem das funções.

O diagnóstico não muda o critério que exigia melhora em pelo menos dois
casos de validação, não seleciona um vencedor novo, não altera o portão
conjunto e não usa103para orientar a análise. Correlações entre erros não
serão interpretadas como causalidade ou culpa percentual independente.

Arquivos congelados por hashes antes do cálculo. Eventos reais com previsões
por caso permitem recontagem. Quatro testes sintéticos conferem cobertura,
empate justo, ausência de margem e rejeição de valores não finitos.

Comandos na pastaD:\AGI_Inventor\diagnostico_estado:

```powershell
py -3.11 -B testes_pool_funcoes.py
py -3.11 -B diagnosticar_pool_funcoes.py --congelar
py -3.11 -B diagnosticar_pool_funcoes.py --executar
```
