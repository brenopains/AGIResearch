# Onde a informação visual muda a decisão?

Diagnóstico posterior à avaliação 179, exclusivamente descritivo. Não é nova
confirmação, ajuste ou promoção. O portão original permanece intacto. Não usa
LLM, dados internos, novas observações físicas ou previsões ajustadas ao teste.

Reconciliar todas as 6.912 comparações com as métricas auditadas. Para cada
jogo, partição e horizonte, contar sinais previstos por par de programas:
preferência esquerda, direita ou empate. Medir quantas escolhas mudam com o
contexto e quantas melhoram/pioram o arrependimento comparadas à ablação do
mesmo corpo sem contexto. Variação de escolhas não implica utilidade.

Calcular separadamente o menor arrependimento de uma escolha constante por
par neste conjunto finito: mínimo da soma dos custos de escolher sempre
esquerda ou sempre direita. É um oráculo retrospectivo, construído com as
respostas reservadas; não implantá-lo como política aprendida. Ele delimita
o melhor desempenho das escolhas constantes nesses casos, não o de todos
os agentes. Um agente contextual pode superá-lo; o teste unitário confere isso.

Fontes, respostas e auditoria serão identificadas por hashes antes de executar
o diagnóstico. Limite de 30 segundos, stdlib, sem ajuste/coleta. Quatro testes
conferem custos, benefício real, contexto prejudicial e identidade dos casos.

```powershell
py -3.11 -B testes_contexto_resolucao.py
py -3.11 -B diagnosticar_contexto_resolucao.py prepare
py -3.11 -B diagnosticar_contexto_resolucao.py run
```

Sinais constantes podem coexistir com previsões de magnitude diferentes. Não
inferir que todos os componentes são incapazes, que só exista um gargalo,
ou que este diagnóstico revele a causa única do erro. A próxima intervenção
deve ser congelada e testada em outra experiência física, sem selecionar
retroativamente o melhor braço da avaliação atual.
