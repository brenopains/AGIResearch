# Correção do nome da referência no diagnóstico de contexto

O diagnóstico original permanece congelado, mas o nome do campo
gain_over_same_body_without_context era amplo demais: no_context é a ablação
do corpo learned. Para coarse, moments, random, transferred e demais braços,
a referência não é a ablação do próprio corpo. Os cálculos desses braços
continuam sendo diferenças contra a ablação do primário, e não evidência
de efeito causal do contexto nesses outros corpos.

Esta versão identifica a referência como learned_body_without_context,
nomeia a diferença gain_over_learned_body_without_context e declara que a
comparação do mesmo corpo só se aplica a learned. Não altera valores,
respostas, modelos ou portão. Um quinto teste verifica esse escopo.

Aplica o protocolo descritivo CONTEXTO_RESOLUCAO.md: 6.912 comparações,
sem ajustes/observações físicas, limite de 30 segundos. Melhor escolha
constante por par é um oráculo retrospectivo, não uma política aprendida.
Nenhuma confirmação independente ou promoção. Fontes e resultados antigos
entram no novo manifesto; a execução recusa sobrescrita.

Comandos, uma única vez por saída, no diretório isolado:

```powershell
py -3.11 -B testes_contexto_resolucao_v2.py
py -3.11 -B diagnosticar_contexto_resolucao_v2.py prepare
py -3.11 -B diagnosticar_contexto_resolucao_v2.py run
```
