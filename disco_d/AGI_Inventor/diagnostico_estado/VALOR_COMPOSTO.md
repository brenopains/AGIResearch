# Crédito local, trajetórias e futuros distribuídos

Este é desenvolvimento com bancos já inspecionados, não confirmação nem
evidência de inteligência humana. O treino é somente 97; 103 é diagnóstico.
Todos os corpos novos ficam nesta cópia no D. Nenhum arquivo congelado muda.

`valor_composto.py` aprende, por ação, recompensa em oito passos e probabilidade
de continuar, usando estados definidos por quatro consequências visuais curtas.
O retorno é soma de incrementos, descontada pela sobrevivência prevista. Há
controles com estado fixo, modelos de ação trocados e crédito embaralhado.
`testes_valor_composto.py` certifica a soma e a absorção após término.

O ensaio falhou contra a ação constante na decisão longa. Por isso,
`diagnosticar_credito.py` substitui só a trajetória imaginada pelos estados
reais futuros, e compara compressão 32, assinatura completa e pixels. Esses
futuros jamais ficam disponíveis à política. Há oráculo positivo que deve
reconstruir o retorno exatamente, com arrependimento zero.

Nos 64 passos, ler crédito dos estados reais dá arrependimento zero em
Breakout e 0,0625 em Pong, contra 0,21875/0,15625 da ação constante. A
assinatura completa não melhora a decisão em relação à compressão nesse
ensaio. Esse resultado localiza perda no caminho de previsão, mas não prova
que a representação seja suficiente em outros mundos ou horizontes.

## Próximo diagnóstico, especificado antes de executá-lo

Hipótese: a média de futuros pode cair em um estado que não pertence a nenhum
futuro real; aplicar novamente uma função não linear à média perde informação.
Testar representação distribuída sobre experiências reais de oito passos,
com recompensa e estado seguinte associados na mesma experiência.

`futuros_distribuidos.py` monta quatro bancos de arestas observadas em 97.
Cada aresta tem estado atual, estado seguinte, incremento de recompensa e
continuação. Um kernel de distância distribui peso sobre as oito experiências
mais próximas; largura local é a distância da oitava, sem ajuste em 103.
Propagar distribuições sobre estados observados por esse kernel e somar crédito
com término absorvente. Comparar com colapsar a mesma distribuição ao estado
médio a cada bloco e com manter o estado inicial fixo.

Controles: embaralhar estados seguintes dentro da mesma ação, embaralhar
recompensas dentro da mesma ação, trocar bancos de ação. Manter tamanhos,
kernel e orçamento iguais. Oráculo de início real apenas diagnóstico. Reportar
8/32/64/128 passos, escolhas por raiz e episódios, custos e hashes. Sem
selecionar variante pelo teste e sem promover com resultado exploratório.

Se a distribuição continuar perdendo para ação constante, não alegar que esse
problema de média explica a falha. Se vencer controles no desenvolvimento,
congelar antes de abrir uma semente ainda intacta. É composição empírica de
dinâmica e crédito; não inventa um operador científico nem constitui AGI.

```powershell
py -3.11 -B testes_valor_composto.py
py -3.11 -B valor_composto.py breakout --seed 103
py -3.11 -B valor_composto.py pong --seed 103
py -3.11 -B diagnosticar_credito.py breakout
py -3.11 -B diagnosticar_credito.py pong
py -3.11 -B testes_futuros_distribuidos.py
py -3.11 -B futuros_distribuidos.py breakout
py -3.11 -B futuros_distribuidos.py pong
```
