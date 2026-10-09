# Decidir sobre uma distribuição de estados

O ensaio de identidades confirmou que média de coordenadas não é solução
robusta. Para uma função de valor não linear, `Q(E[estado])` difere em geral de
`E[Q(estado)]`. Este diagnóstico usa o mesmo filtro já congelado e o mesmo
regressor. Aprende `Q` somente nos episódios 0–5 e, no teste, calcula valor
para cada hipótese visual antes de promediar com seus pesos perceptivos.

Comparações: pixels; hipótese de menor custo; média de estados; média de
valores. Registrar em quantos casos a ação muda, se as mudanças corrigem ou
pioram arrependimento, e permutar os pesos das hipóteses. Os mundos já vistos
são exploratórios; só congelar e avaliar outros mundos se aparecer vantagem
consistente frente aos dois baselines, não só frente a pixels.

O valor neste mundo é a ocupação futura de sensores, não uma ação causal.
Mesmo um resultado positivo não prova planejamento nem generalidade.
