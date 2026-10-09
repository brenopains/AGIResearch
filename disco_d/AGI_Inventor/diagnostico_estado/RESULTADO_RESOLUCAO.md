# Etapa 52 — funções visuais da experiência e preservação sensorial

Data: 2026-09-29. O objetivo humano não foi atingido. Esta rodada criou uma
biblioteca real de funções visuais e dados auditados, mas **a comparação de
capacidade ficou inconclusiva por atingir o teto de ajuste**. Não foi coletada
a avaliação179, não foi testado seu portão, nenhum modelo foi promovido.
Não inferir proximidade humana ou uma peça suficiente para AGI.

## O que mudou de fato

O mecanismo anterior recebia médias10x10 por quadro, seguidas de funções RBF
globais. `coletar_resolucao.py` agora preserva cada trajetória em210x160uint8,
com offsets que permitem reconstruir janelas passadas sem duplicar todos os
quadros. Essa preservação é lossless após converter para cinza; continua
descartando cor e quadros entre chamadas ALE de salto4. Não é estado completo
do mundo nem percepção multimodal geral.

`filtros_observados.py` gera cinco funções locais por mundo a partir de8192
patches de mudanças observadas, sem usar ação, recompensa ou nome de objeto.
O corpo de cada função contém100coeficientes e pode executar sobre qualquer
patch do mesmo formato. A família fornecida é PCA de inovação temporal;
aprender esses coeficientes não é invenção arbitrária de primitivas nem
descoberta mundial inédita. O tamanho de patch e a operação são escolhas humanas.

Dois bancos de construção e dois de desenvolvimento foram reexecutados. Todos
os quadros reduzidos, recompensas, terminais, programas, raízes e episódios
coincidiram com os antigos. Mantemos agora as observações espaciais que antes
não estavam disponíveis para outras perguntas. Estes bancos continuam em
jogos conhecidos; desenvolvimento151 já havia sido observado.

## Auditorias e resultados utilizáveis

1. Oito testes de sensor/filtros passaram: reprodução do codec original,
   tradução dentro de uma célula distinguida por momentos, janelas terminais,
   geração a partir da inovação e serialização de corpos. O exemplo de
   tradução fabricado testa o codec, não aprendizagem em Atari.
2. A auditoria física recontou91.398chamadas e32replays integrais. Reconstruiu
   todas as janelas reduzidas a partir das trajetórias completas e conferiu
   igualdade com os bancos antigos. Não adicionou coleta.
3. A auditoria da biblioteca conferiu autovalores/equações de covariância e
   execução independente das dez funções aprendidas. Os mesmos corpos foram
   aplicados nas imagens dos dois jogos sem reescrever pesos. Maior resíduo
   de autovetor1,76e-19, maior discrepância numérica de resposta1,51e-8.
   Isso demonstra origem/execução, **não benefício de transferência**.
4. A busca de colisões examinou6.959transições válidas em5.822grupos de
   história reduzida+ação. Não achou um grupo com histórias completas distintas
   ou recompensas conflitantes. Ausência nesse conjunto não prova suficiência
   do estado; não apareceu um contraexemplo que atribua diretamente esses
   erros à compressão10x10. Unicidade finita também permite memorização,
   portanto não certifica representação que generalize.

| Banco | Transições válidas | Grupos história+ação | Aliases de histórias completas | Conflitos de recompensa |
|---|---:|---:|---:|---:|
| Breakout149 | 1.667 | 1.337 | 0 | 0 |
| Breakout151 | 1.369 | 1.098 | 0 | 0 |
| Pong149 | 1.963 | 1.709 | 0 | 0 |
| Pong151 | 1.960 | 1.678 | 0 | 0 |

Os limites inferiores empíricos de MSE por agrupamento exato são zero nas duas
resoluções; isso significa que não há contradição observada nesses grupos,
não que um aprendiz finito possa prever perfeitamente ou aprender rapidamente.
Hashes foram usados para indexar; igualdade real dos arrays foi conferida
quando um grupo repetiu, incluindo os quadros completos.

## Por que a rodada de capacidade não terminou

`piloto_resolucao.py` compara média, momentos fixos, filtros aprendidos
(primário), filtros aleatórios e filtros vindos do outro jogo. Usa as mesmas
observações e32suportes/regularização.05. Controles adicionais: resultados
embaralhados, contexto removido, corpo antigo e zero. Não houve escolha pelo
resultado final nem mudança de sensor primário.

O ajuste atingiu a checagem de120segundos antes de completar os modelos de
Pong. Foram exportados12corpos parciais: sete deBreakout e cinco dePong.
Faltaram aleatório e transferido emPong; o manifesto completo de corpos não
existe. A exceção foi `RuntimeError: teto de ajuste; inconclusivo`. O tempo
exato não foi registrado; o limite inferior observado é120segundos. O limite
é verificado entre ajustes, não uma interrupção rígida no meio de uma operação.

`resultados/resolucao_ajuste_interrupcao.json` registra o motivo, hashes dos
artefatos parciais e ausência de avaliação. Não pontuamos apenas um jogo nem
usamos corpos incompletos para declarar ganho. Todas as quatro células do
portão original continuam sem veredito de capacidade, não reprovadas por
evidência comportamental. Os critérios não foram relaxados.

## Perfil e otimização ainda não certificada em velocidade

`dicionario_rapido.py` implementa deduplicação por hash com comparação real,
ordenação numérica lexicográfica e reutilização de normas na seleção de
suportes. Quatro testes de equivalência passaram, incluindo o modelo completo.
A versão anterior congelada não foi editada nem substituída em execução.

Um perfil posterior reproduziu o corpo aprendido deBreakout na mesma matriz
de treino: suportes e escala iguais, discrepância zero de coeficientes e
previsões. Extração5,92s, ajuste13,47s, total19,53s, uma chamada de ajuste
adicional e nenhuma coleta. RSS ao final527.622.144bytes, não pico global.

**Não foi medido um ganho de velocidade:** faltou comparação cronometrada
pareada com o executor anterior. O campo de status do perfil contém a palavra
`faster`, mas seu teste verifica apenas equivalência. Essa extrapolação de
nome foi corrigida na interpretação machine-readable
`resultados/resolucao_perfil_interpretacao.json`; o arquivo congelado original
foi preservado. Não usar aquele rótulo como evidência de aceleração.

A próxima intervenção deve medir extração, deduplicação, seleção de suportes
e projeção separadamente, preservar o problema numérico e só depois repetir
o conjunto completo dos dois mundos em uma rodada nova. Não aproveitar o teto
reiniciado como se o custo anterior não tivesse ocorrido. A comparação precisa
continuar com todos os controles e com avaliação física após congelamento.

## Contabilidade e alcance

| Coleta | Chamadas ALE | Tempo de coleta (s) | Replays |
|---|---:|---:|---:|
| Breakout149 | 22.154 | 17,43 | 8 |
| Pong149 | 24.850 | 17,32 | 8 |
| Breakout151 | 19.532 | 15,41 | 8 |
| Pong151 | 24.862 | 17,83 | 8 |

Coleta total67,99s, experiência física91.398chamadas/365.592quadros com salto4
(~101,55minutos a60quadros/s, somando caminhada, ramos e replays). Isso é custo
adicional, não substitui as coletas das etapas50/51. Não confundir tempo de
cálculo com poucos minutos de aprendizagem humana.

Maior RSS amostrado de coleta205.828.096bytes (~196,3MiB). Durante ajuste,
uma leitura do processo mostrou pico até aquele momento1.973.678.080bytes
(~1,84GiB); não foi registrado o pico completo do ajuste. A auditoria de dados
registrou até1.090.461.696bytes ao final de um banco. Não é contenção rígida
de memória. Um processo próprio foi usado por vez; todos já terminaram.

Auditoria de dados17,74s e biblioteca concluída sem ajustar coeficientes ou
coletar experiência adicional. Recalcular covariância para conferir a origem
é trabalho de auditoria e não atualização de uma biblioteca. Os patches de
inovação são correlacionados, não8192episódios independentes.

Código, protocolo e comandos principais em `RESOLUCAO.md`; dados e eventos
em `resultados/resolucao_*`. Complementos posteriores, com fontes congeladas
antes de sua execução:

```powershell
py -3.11 -B auditar_resolucao_dados.py prepare
py -3.11 -B auditar_resolucao_dados.py run
py -3.11 -B auditar_biblioteca_local.py prepare
py -3.11 -B auditar_biblioteca_local.py run
py -3.11 -B testes_dicionario_rapido.py
py -3.11 -B diagnosticar_resolucao_custo.py prepare
py -3.11 -B diagnosticar_resolucao_custo.py run
```

Saídas já existem e recusam sobrescrita. Não reiniciar a rodada interrompida
sem contabilizar a tentativa anterior. O projeto original não foi editado.

Continuam sem evidência: ganho composto dessa representação, transferência
útil para mundo desconhecido, geração irrestrita de variáveis/operadores,
planejamento de nível humano e pesquisa autônoma de fronteira. Gerar funções
sensoriais reais é trabalho concreto nessa direção; sua mera existência não
satisfaz o objetivo. O objetivo permanece ativo, não concluído.
