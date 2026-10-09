# Diagnóstico posterior de trajetórias propostas

Os pilotos de sequências em 103 falharam; não abrir nova confirmação. Esta
coleta é diagnóstico posterior, não resgate do veredito. Medir os primeiros
quatro checkpoints de cada um dos oito episódios, sem seleção pelo erro.
Em cada raiz medir dois planos já congelados: distribuição e recompensa
embaralhada. Nenhum corpo, plano ou critério anterior pode mudar.

Reproduzir histórias; observar assinaturas de quatro consequências visuais
curtas nos estados intermediários de cada plano. Conferir a assinatura raiz
com o banco curto e retorno/terminal/passos finais com os eventos do piloto.
As assinaturas são acesso oracular a intervenções reais, indisponíveis ao
agente na raiz. Registrar todas as intervenções, teto de 65.000 passos por jogo.

Comparar a previsão original de retorno com somar crédito aprendido em 97
nos estados reais intermediários. Dois estimadores fixos: oito vizinhos do
planejador e KernelRidge já usado no diagnóstico local. Usar validade real
para absorver término no oráculo. Reportar erro, viés e retorno por plano;
uma melhora do oráculo localiza perda no caminho imaginado, não torna o
agente capaz de prever esses estados. Registrar também recompensa verdadeira
como controle positivo de soma exata.

Se o crédito continuar errado com estado real, a perda não é só composição.
Se os dois oráculos melhorarem, a próxima intervenção deve aprender ou
construir uma atualização de estado que sustente consequências relevantes,
sem confundir previsibilidade visual média com utilidade para decisão.

```powershell
py -3.11 -B diagnosticar_planos_propostos.py breakout
py -3.11 -B oraculo_planos_propostos.py breakout
```

Repetir com `pong`. Relatórios e dados em
`resultados/{jogo}_oraculo_planos_propostos_103.*`. Eventos de medidas
compartilhadas entre planos idênticos serão marcados como reuso de medida,
contando custo físico uma única vez. Não promover com dados deste diagnóstico.
