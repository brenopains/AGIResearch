# Instrumento de replay, antes de haver banco válido

As primeiras coletas97 de Breakout e Pong pararam na comparação do replay.
Não houve banco NPZ válido nem ajuste/resultado do aprendiz nessa tentativa.
O código anterior foi preservado em `coletar_transicoes_mistas_tentativa1.py`;
as propostas de sequências anteriores estão em `*_transicoes_mistas_97_planos.json`.

Falha identificada: depois da última sonda, `restoreSystemState` restaurava
a dinâmica, mas a imagem lida por `gray(ale)` ainda podia ser o quadro da
sonda contrafactual, e não o quadro real da trajetória. Um teste separado
em Pong97 consumiu160ações: o quadro após restaurar diferiu do último quadro
real guardado, confirmando esse comportamento da interface nesta configuração.

Correção do instrumento: usar como referência final `branch[-1]`, capturado
durante a ação principal antes das sondas. O replay continua sendo reexecutado
sem sondas; recompensas por bloco, comprimento, término, quadro final e todos
os16quadros comprimidos anteriores precisam concordar exatamente. Não houve
relaxamento de tolerância, alteração de plano, modelo ou portão científico.

Preservar as propostas da primeira tentativa; as sequências da versão2 são
armazenadas num arquivo novo `*_planos_v2.json`. Usam exatamente o mesmo RNG
e o mesmo corpo de geração; conferir igualdade na auditoria. O custo da nova
coleta é contado exatamente. Nas tentativas interrompidas, a telemetria
persistente estava incompleta: Breakout registrou15.423passos antes de falhar
no episódio3 e Pong falhou antes do primeiro resumo de episódio. Esses custos
parciais não serão fingidos como zero nem confundidos com o custo exato dos
bancos concluídos. Não temos seu total exato; não alegar total agregado exato
para todo o trabalho desta etapa. Ambos os processos terminaram com erro1.
