# Onde a previsão local deixa de compor

O efeito visual de uma ação por oito passos passou em dois jogos, mas não
resolveu o retorno de 128 passos. Este ensaio mede o elo intermediário em
estados da semente 97 (desenvolvimento) e 103 (já usada em confirmação local,
portanto **diagnóstico**, não nova confirmação independente).

Reconstruir os mesmos 64 checkpoints por jogo e certificar cada história
contra o banco de oito passos. De cada clone executar os mesmos quatro
comandos constantes até 128 passos, guardando os dois quadros finais e a
recompensa acumulada a cada oito passos. Parar no terminal e manter o último
quadro para horizontes posteriores. Conferir no horizonte oito, ação por
ação, os quadros, recompensas, terminais e comprimentos já salvos. Todas as
ações contam num teto de 45 mil passos ALE por jogo e semente.

Comparar, sem usar estados de teste para ajuste:

1. Regressão direta do histórico inicial ao par de quadros em horizontes
   8, 16, 32, 64 e 128, ajustada nos 64 estados de 97.
2. Composição da **mesma** transição aprendida para oito passos, aplicando o
   comando repetidamente ao par de quadros previsto. Nenhum alvo de
   horizonte maior ajusta a composição.
3. Molde médio de efeito por comando e previsão cega ao comando.
4. Atualização com estado intermediário verdadeiro, somente como oráculo
   diagnóstico, para separar erro por deslocamento de distribuição de erro
   acumulado por realimentação da própria previsão.

Usar o erro dos seis contrastes visuais entre comandos, com o mesmo peso e
normalização do teste local. Reportar por episódio e horizonte; registrar
energia do contraste e casos terminais. Composição só será considerada uma
capacidade local se superar a regressão direta e o molde em mundos novos,
com controles de rótulo/estado. Este banco 103 não pode provar isso porque
suas respostas de oito passos já foram vistas. A curva aqui orientará um
protocolo novo e congelado, sem reivindicar planejamento geral.

Após observar a curva, a investigação exploratória separará duas hipóteses:
(A) faltam estados intermediários no treino da transição; (B) a transição
se realimenta com imagens que nenhum mundo produziu. Comparar um modelo de
oito passos treinado apenas nas 64 raízes com outro treinado nas raízes mais
128 transições intermediárias por ação, amostradas da semente 97 com RNG
fixo. Em paralelo, após cada predição, projetar para o estado sucessor real
mais próximo na memória de treino, medido num subespaço de 32 dimensões
aprendido **somente** em 97. Isso forma quatro braços (sem/com treino
intermediário × sem/com projeção). Semente 103 continua apenas diagnóstico.
Reportar também avaliação em ramos sem terminação, para que fantasia após
`game_over` não explique sozinha a falha de composição.

O encaixe duro no vizinho mais próximo conteve a deriva, mas degradou
contrastes no Pong. Como diagnóstico pós-hoc, comparar também uma projeção
contínua: remover metade da componente ortogonal ao subespaço de 32
dimensões aprendido nos estados de treino, mantendo a outra metade da
previsão intacta. Sem busca de peso ou dimensão na semente 103. A finalidade
é separar "conter estados impossíveis" de "substituir identidade causal".
