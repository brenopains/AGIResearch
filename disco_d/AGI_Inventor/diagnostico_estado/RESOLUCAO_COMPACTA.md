# Representação espacial: rodada com execução compacta

O objetivo geral continua sendo uma mente artificial de nível humano. Este teste
examina representação sensorial e antecipação em dois jogos conhecidos. Não
certifica AGI, eficiência humana, uma única peça suficiente ou novidade mundial.
Não usa LLM, coordenadas, estado oculto nem pistas semânticas dos jogos.

## Mudança e custos preservados

A rodada anterior não concluiu o ajuste antes do checkpoint de 120 segundos.
Seus doze corpos parciais e custos permanecem registrados e não serão promovidos.
Esta é uma rodada distinta: refaz todos os ajustes usando as mesmas observações
de treino 149, sem coletá-las novamente. O desenvolvimento 151 já foi observado
e não constitui confirmação independente. A avaliação 179 ainda não foi coletada
e só será acessada depois de congelar todos os novos corpos.

CompactTransport mantém a linguagem, as funções, os 32 suportes, ridge .05 e o
mesmo algoritmo de seleção. Deduplica/ordena linhas por chaves binárias com ordem
numérica, sem hashes, e guarda normas e o dobro da matriz durante a seleção.
Seis testes verificam equivalência, inclusive composição e placebo embaralhado.
A comparação pareada nos dois conjuntos reais alternou a ordem das versões,
confirmando equivalência numérica e aceleração de 2,71 a 3,09 vezes nos ajustes.
Esse ganho é de execução; não constitui ganho de inteligência. A comparação
consumiu oito ajustes e 54,06 segundos, sem chamadas físicas adicionais.
O perfil original, a primeira tentativa de otimização sem ganho confirmado e
a tentativa interrompida também ficam disponíveis; não omitir seus custos.

## Dados e funções

Reusar trajetórias 149/151: quadros cinza 210x160 uint8 após cada chamada ALE,
frame skip 4, em chunks por raiz. A preservação começa após a conversão para
cinza; não inclui RGB nem os quadros intermediários. Os quatro bancos existentes
custaram 91.398 chamadas ALE adicionais, com 32 replays reais certificados.
Esses custos são anteriores a esta rodada; nenhuma reutilização os torna novos.
Cada banco tem oito episódios e 32 raízes. Mesmos programas exógenos congelados:
quatro para treino e oito para avaliação, dos quais quatro composições novas.

Gerar cinco filtros locais por PCA da inovação temporal de patches 10x10:
8.192 amostras exclusivamente do treino 149, subtração da média por patch,
covariância, autovetores de maiores autovalores, sinal fixado pelo coeficiente
de maior módulo, normalização L1. As funções completas têm 100 coeficientes
exportados. Essa família e seu algoritmo são fornecidos por nós, com precedentes;
o ensaio não demonstra invenção arbitrária de variáveis.

Cinco sensores: média repetida em seis canais; momentos espaciais fixos;
cinco filtros aprendidos mais média (primário); cinco filtros aleatórios mais
média; e os filtros aprendidos no outro jogo mais média (transferência secundária).
Todos usam quadros passados 0/2/4/6/8/10/12/15 e escala por canal somente das
fontes de treino, com piso .001. Modelos temporais são reajustados por jogo.
Reusar o mesmo corpo de um filtro não demonstra transferência de inteligência.

RBF com 32 suportes observados, ridge .05 para recompensa e transporte de um
passo. Controles adicionais: resultados embaralhados dentro de cada ação,
remoção do contexto inicial, corpo antigo congelado e zero. Os cinco sensores
têm iguais dimensões de coeficientes e observações; seus custos de extração
diferem e serão reportados. Nenhum reajuste após desenvolvimento ou avaliação.

## Critério congelado e execução

Mesmo portão da rodada anterior, sem mudanças após resultados: nas quatro
células jogo × programas conhecidos/novos, o primário precisa de MSE e
arrependimento de desempate justo estritamente menores que média, momentos,
aleatório, embaralhado, sem contexto, modelo antigo e zero, no horizonte 16.
Cada par deve ter preferências nas duas direções e pelo menos três episódios
com contraste. Sem cobertura: inconclusivo. Horizontes 4/8 são secundários.
Transferência é secundária e nunca substitui o primário após olhar resultados.

Um processo por vez, duas threads BLAS; checkpoint de ajuste 120 segundos,
120 segundos e 50 mil chamadas por coleta, RSS amostrado inferior a 6 GiB.
Checkpoints não constituem interrupção rígida no meio de uma operação; um corpo
que ultrapasse o limite não será elegível. Eventos e checkpoints guardam custos.
Fontes e critérios antes do ajuste; corpos e programas antes da avaliação 179.
Saídas recusam sobrescrita. Uma execução incompleta é inconclusiva, nunca sucesso.

Auditoria reconstitui histórias reduzidas de seis bancos, origens dos filtros
pelas equações de covariância, suportes observados, equações normais por
Cholesky, placebo, composição por implementação independente, casos, cobertura
e portão. Conferir dez eventos de ajuste e 6.912 comparações reservadas.
As 48 certificações de replay incluem 32 anteriores e 16 novas, não 48 novas.

No diretório D:\AGI_Inventor\diagnostico_estado, uma única vez por saída:

```powershell
py -3.11 -B piloto_resolucao_compacto.py prepare
py -3.11 -B piloto_resolucao_compacto.py fit
py -3.11 -B coletar_resolucao_compacto.py breakout 179
py -3.11 -B coletar_resolucao_compacto.py pong 179
py -3.11 -B piloto_resolucao_compacto.py evaluate
py -3.11 -B auditar_resolucao_compacto.py
```

Desenvolvimento opcional `development` usa 151, já conhecido, e não autoriza
mudar corpos/configurações. O projeto original permanece intacto. Resultado
positivo autoriza apenas investigar esta camada; negativo delimita este
mecanismo, sem provar impossibilidade matemática nem uma causa única do gap.
