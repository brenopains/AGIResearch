# Diagnóstico de composição, sem ajuste ou coleta

Pergunta: as funções congeladas131 respeitam a diferença de retorno após
executar um prefixo comum? Este diagnóstico é de desenvolvimento. Os mundos
e os bancos97/103 já foram vistos. Nenhum portão anterior é alterado, nenhum
modelo é promovido e nenhuma capacidade humana é inferida daqui.

Selecionar por sintaxe todas as raízes dos quatro bancos mistos97/103
(Breakout/Pong) cujo maior prefixo comum P tenha comprimento entre1 e15.
Não filtrar por recompensa ou término. Comparar as palavras completas P+Q,
P+R no horizonte16blocos; cada bloco teve8chamadasALE, frame_skip4. Curvas
secundárias em todos os horizontes de k+1 a16 são correlacionadas e não
contam como novas amostras independentes.

Antes da previsão, conferir igualdade exata das duas histórias iniciais e
das histórias/recompensas/marcadores de validade durante P. Conferir que a
diferença dos retornos completos é igual à diferença dos retornos dos
sufixos, e que posições posteriores ao término não recebem recompensa.
Uma discrepância aborta o diagnóstico; nunca é um fracasso do aprendiz.

Reexecutar os sete corpos131 inalterados nas três consultas:

1. palavras completas na história inicial;
2. sufixos na história inicial (controle sem transporte);
3. sufixos na história intermediária realmente medida (oráculo).

O item3 usa informação futura real e NÃO é um simulador aprendido nem um
planejador executável. A igualdade física examinada é:

`Delta(H0, P+Q, P+R) = Delta(H_depois_P, Q, R)`.

As entradas do aprendiz contêm apenas histórias visuais e ações anônimas.
Não ler assinaturas de sondagens ou estados internos. Sem ajuste, mudança
de hiperparâmetros, seleção de corpo ou coletaALE.

Reportar contagens, raízes, episódios, prefixos, términos, diferenças reais,
predições individuais, MSE, escolhas e arrependimento. Regra determinística:
predição >= -1e-12 escolhe esquerda. Para não beneficiar a previsãozero por
desempate, reportar também arrependimento esperado com empates uniformes:
abs(diferença_real)/2 quando abs(predição)<=1e-12; fora do empate, custo da
escolha errada. Consistência sem acurácia é insuficiente: o braçozero é
incluído nas três consultas e respeita a igualdade trivialmente.

Reportar defeito absoluto médio/máximo da igualdade prevista e desacordo de
escolha, sem transformá-los em medida de promessa ou critério de promoção.
Comparações por episódio são descritivas; não inventar significância a partir
das curvas sobrepostas. Uma melhora oracular apontaria uma oportunidade local
de transporte/composição; ausência dela não prova um único gargalo deAGI.

Congelar antes do cálculo as fontes, este protocolo, corpos, manifestos e
bancos comSHA256. Auditor separado recalcula seleção física, alvos, métricas
e predições com a forma dual (seis braços) e aritméticaRBF independente
(sétimo). EventosJSONL reais, sem dados de demonstração fabricados.

Comandos emPowerShell, dentro deD:\AGI_Inventor\diagnostico_estado:

```powershell
py -3.11 -B testes_semantica_prefixo.py
py -3.11 -B diagnosticar_semantica_prefixo.py --congelar
py -3.11 -B diagnosticar_semantica_prefixo.py --executar
py -3.11 -B auditar_semantica_prefixo.py
```

Saídas preservadas emresultados/semantica_prefixo_comum*. Sem inferência de
transferência entre domínios: cada corpo foi ajustado no próprio jogo.
