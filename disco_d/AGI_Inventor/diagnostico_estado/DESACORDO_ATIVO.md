# Perguntas escolhidas por desacordo preditivo

Os ensaios anteriores mostram que multiplicar representações e hipóteses de
identidade raramente cria **alternativas de ação**. Aqui a pergunta é se um
agente consegue alcançar estados em que modelos plausíveis discordam da ação
preferida e, ao testar esses estados, descobrir erros decisórios novos mais
eficientemente do que a seleção aleatória.

Dois jogos de desenvolvimento, Breakout e Pong. O decisor e o comitê ficam
congelados a partir dos episódios 0–5 dos bancos `*_ciclo.npz` (semente 29).
O comitê: regressões kernel treinadas deixando um episódio de fora, em duas
visões de pixels (dois quadros reduzidos e 16 quadros reduzidos). Nenhum membro
recebe estado RAM ou retorno do piloto.

Em cada um de oito episódios novos (semente ALE 59), preparar estado comum.
Gerar 24 programas de quatro segmentos de 32 ações: o **mesmo pool** serve a
ambos os braços e sua execução custa uma vez, registrada como custo comum.
Cada programa parte do estado comum, produz uma observação visual e um snapshot.
Terminal é candidato inválido. O braço ativo escolhe quatro estados por massa
de votos fora da ação modal do comitê. Empates têm ordem aleatória previamente
fixada. O controle escolhe quatro estados uniformemente do mesmo pool.

Só depois da escolha, restaurar cada snapshot e executar os 16 programas
prefixo/sufixo (32+96) do verificador; todos os passos contam para o braço.
Contraexemplo: a ação escolhida pelo decisor congelado perde retorno para
outra ação. Métrica primária: número de episódios com ao menos um
contraexemplo. Secundárias: quantidade de sondas, arrependimento, estados com
vencedor único, custo por episódio, fração de votos em desacordo. Não ajustar
com resultados dos episódios 6–7 de qualquer banco nem usar recompensa da
trajetória para ordenar o pool.

Portão local antes de rodar: ativo encontra contraexemplo em pelo menos três
episódios líquidos a mais que aleatório em **cada** jogo, sem gastar mais
passos de verificação por episódio para quatro candidatos. Se todos os votos
forem iguais, declarar sem sinal de seleção. O piloto deve caber em 250 mil
passos ALE totais por jogo e aproximadamente dez minutos. Mesmo passando,
isso demonstraria uma habilidade local de escolher experimentos, não AGI.

Este é um teste de mecanismo no ambiente interativo real, não uma medida de
descoberta de leis científicas ou de aprendizagem contínua.
