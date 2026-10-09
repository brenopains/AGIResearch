# Leitura de eventos e campo de estado — protocolo antes do ajuste

O objetivo geral continua sendo uma mente de nível humano. Esta bancada não a
certifica, nem presume que uma única peça seja suficiente. Testa uma hipótese
restrita e corrige o alcance de um diagnóstico anterior.

O antigo `real_future_histories_oracle` calculava `phi[t] @ reward[action]`
**antes** de cada bloco de oito chamadas ALE. Ter a sequência real de histórias
não tornava esse leitor um reconhecedor do evento entre as duas imagens. Seu
fracasso não demonstrava incapacidade de ler consequências já ocorridas.

## Experimento e hipóteses

Manter as bases e transportes da prospectiva congelados. Ajustar leitores somente
no banco de construção 149: antes, depois, diferença com constante, transição,
diferença de um potencial compartilhado, sucessor embaralhado por ação,
recompensa embaralhada por ação, média por ação e zero. O leitor de transição tem
mais coeficientes; seus ganhos não demonstram eficiência computacional igual.
As bases contêm 37 coordenadas (32 RBF, constante e quatro posições reservadas),
com funções adicionais somente na base secundária quando previamente admitidas.
Isto não inventa uma linguagem geral nem permite funções arbitrárias.

Reconhecimento mede a recompensa entre observações reais antes/depois. Previsão
local usa somente antes. Pontuar um programa com suas observações intermediárias
reais é diagnóstico retrospectivo, nunca decisão disponível antes de executá-lo.
Os bancos conhecidos 151 e 157 são desenvolvimento, sem seleção de novos leitores
ou ajuste após sua pontuação. A família não é ampliada para salvar esse teste.

A hipótese adicional é `r=F(H_next)-F(H_current)`, com
`F(H)=phi(H)@theta`. Ajustar theta por mínimos quadrados regularizados nos dados
149, em todas as ações. Não é propriedade universal: recompensas em um ciclo
fechado com soma não nula não podem ter essa forma. Um grafo finito de histórias
observadas verifica restrições inteiras por união ponderada; compatibilidade pode
ser trivial em árvores, não prova que o estado seja suficiente ou que a função
generalize. Conflitos refutam a função desse grafo de observações, podendo também
indicar que histórias distintas foram confundidas pela compressão de pixels.

O resíduo dos mínimos quadrados não regularizados, com corte numérico 1e-10,
informa o limite empírico da família fixa de diferenças; não prova impossibilidade
matemática geral. Nenhuma igualdade aproximada é chamada de prova exata.

Planejamento usa somente a história inicial. Transporta suas características por
ação e calcula F(final previsto)-F(inicial). O transporte vivo é o já aprendido;
um transporte terminal adicional aprende `(1-continuation)*phi(next)`. A parte
terminal fica absorvente, reproduzindo o padding do protocolo físico. Essa
absorção é uma convenção fornecida, não uma descoberta do sistema.

## Congelamento e avaliação

Fontes/protocolo antes de qualquer ajuste; leitores e campo antes de pontuar
151/157 e antes de coletar semente 163. Oito episódios, quatro raízes por episódio,
os oito programas comuns anteriores, 16 blocos de oito chamadas. Semente nova em
Breakout e Pong conhecidos, não mundos inéditos. Quatro programas foram usados
na construção; quatro composições são novas. Até 50.000 chamadas, 120 segundos
por banco, 6 GiB por processo e somente um coletor de cada vez. Contabilizar
caminhada, ramos, replays e também o custo prévio de aprender as bases; frame skip
4 significa que segundos de CPU não equivalem a segundos de experiência.

Controles de planejamento: retorno composto antigo, campo com recompensas
embaralhadas, remoção do transporte terminal, remoção do contexto inicial e zero.
Primário: base sem funções adicionais. Secundário: base prospectiva aprendida;
nunca substituí-la pelo primário depois de ver os resultados. Horizontes 4/8/16;
portão no 16 exige MSE e arrependimento com desempate justo estritamente menores
que todos os cinco controles, nas quatro células jogo × programas conhecidos/novos.
Cada par deve apresentar preferências reais em ambas as direções, em pelo menos
três episódios. Sem cobertura, resultado inconclusivo. A igualdade com uma
ablação pode ocorrer se o componente removido não atua neste banco: falhar esse
portão exigente não prova inutilidade geral. Sem ajuste durante avaliação.

Auditoria reexecuta coeficientes por Cholesky, compatibilidade inteira por caminhos
em grafo, planejamento por matrizes aumentadas e métricas por casos. Verificar
corpos, dados, replays e contabilidade via hashes. Não usar RAM do jogo, posição
de objetos ou estado interno ALE como entrada cognitiva; cloneSystemState serve
apenas ao avaliador físico de ramos com a mesma raiz.

## Comandos

No diretório `D:\AGI_Inventor\diagnostico_estado`, executar sequencialmente:

```powershell
py -3.11 -B testes_leitor_eventos.py
py -3.11 -B piloto_leitor_eventos.py prepare
py -3.11 -B piloto_leitor_eventos.py fit
py -3.11 -B piloto_leitor_eventos.py diagnose
py -3.11 -B coletar_eventos.py breakout
py -3.11 -B coletar_eventos.py pong
py -3.11 -B piloto_leitor_eventos.py evaluate
py -3.11 -B auditar_leitor_eventos.py
```

As saídas recusam sobrescrita. Coleta interrompida conserva eventos/chunks e
recebe diagnóstico inconclusivo; não reiniciar ignorando as chamadas anteriores.
Um sucesso aqui seria evidência local sobre essa hipótese, não AGI, descoberta
inédita, transferência de um corpo entre jogos nem desempenho humano.
