# Funções positivas: o estado previsto ainda pode ser físico?

Diagnóstico posterior aos resultados espaciais 179, sem confirmação nova,
treinamento ou chamadas ALE. O portão anterior permanece intacto. Ambos os
jogos, partições conhecidas/novas, horizontes 4/8/16 e sete corpos: cinco
sensores, placebo embaralhado e ablação do contexto. Não aplicar estes limites
ao modelo antigo, que contém outras funções possivelmente negativas.

A base atual é [1, exp(-distância²/escala)]. Logo cada função está em [0,1].
O transporte inclui continuação: a coordenada constante da previsão composta
representa massa sobrevivente m. Uma previsão física em média precisa de
0 <= m <= 1 e 0 <= função_i <= m. Com término absorvente, m não pode crescer
sob continuação. São condições necessárias, não suficientes: satisfazê-las
não prova existir distribuição física correspondente nem previsão correta.
As características de estados reais multiplicadas pelo indicador de validade
devem passar como controle positivo, inclusive após término. Tolerância 1e-8.

Não há argumento de que a simples média de funções perca inevitavelmente
ramificações: se a base fecha os operadores e a leitura é linear, expectativas
podem ser compostas corretamente. Preservar distribuições já tem precedentes
e já foi testado neste projeto; não anunciar repetição como novidade.

Reconstituir funções das trajetórias completas sem fornecer futuros ao agente.
Conciliar as 5.376 previsões correspondentes com o resultado congelado.
Aplicar a decomposição telescópica existente:
erro total = leitura local + propagação ponderada pelo sufixo + fronteira.
Reconciliar MSE com parcelas quadráticas e termos cruzados. Não tratar parcelas
isoladas como atribuição causal: podem se cancelar.

Oráculo com histórias reais antes de cada bloco ainda antecipa a recompensa
dos oito passos seguintes. Não é reconhecimento retrospectivo de eventos.
Oráculo de remoção do erro de recompensa também usa respostas reservadas;
nenhum deles é agente ou será usado para escolher uma política executável.

Fontes, corpos e dados por hashes antes da execução. Limite de 180 segundos,
um processo, duas threads BLAS, RSS amostrado inferior a 6 GiB. Incompleto é
inconclusivo. Cinco testes conferem limites, subprobabilidade e absorção.

```powershell
py -3.11 -B testes_realizabilidade.py
py -3.11 -B diagnosticar_realizabilidade.py prepare
py -3.11 -B diagnosticar_realizabilidade.py run
```

Uma violação demonstra incoerência deste estado funcional dentro das hipóteses
do transporte. Não demonstra ser a causa única do erro, nem garante que
projeção, positividade ou mais experiência resolvam composição ou AGI.
