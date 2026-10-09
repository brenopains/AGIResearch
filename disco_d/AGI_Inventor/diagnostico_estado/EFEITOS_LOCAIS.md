# Medir consequências visuais locais antes de retorno longo

Pergunta: com o histórico bruto até um checkpoint, o sistema consegue prever
**o que muda** após cada um dos quatro comandos durante oito passos? Isto
separa falha de estado/dinâmica local da falha de composição até uma
recompensa distante. O alvo é o quadro final e a recompensa de oito passos;
nenhum nome de entidade do jogo é entregue ao aprendiz.

Primeiro coletar a semente de desenvolvimento 97: oito episódios, oito
checkpoints separados por 128 ações uniformemente aleatórias. Conferir os
16 quadros de cada checkpoint com o banco existente. Clonar cada estado,
executar quatro ramos pareados de oito passos e restaurar o estado antes de
continuar a trajetória. Guardar quadros finais, recompensas, terminação,
passos e eventos. Repetir um ramo como teste de replay. Orçamento máximo
20 mil passos ALE por jogo, com toda ação contabilizada.

O código de previsão será desenvolvido apenas sobre esta semente. Uma
semente nova ficará reservada para confirmar qualquer ganho. A avaliação
deve medir **mudança**, não erro de reconstrução dominado pelo fundo estático.
Comparar com persistência (nenhuma mudança), predição cega à ação e etiquetas
de ação embaralhadas; reportar por episódio e por jogo. Nenhuma pontuação
aproximada será descrita como prova de modelo de mundo geral.
