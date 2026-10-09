# Funções espaciais observadas — protocolo antes de coletar/ajustar

Objetivo geral: mente artificial de nível humano. Esta bancada testa uma camada
de representação/predição; não certifica AGI, uma peça suficiente, eficiência
humana ou descoberta inédita. Não usa LLM nem dados internos do jogo.

Preservar trajetórias de quadros cinza 210x160 uint8 após cada chamada ALE, em
chunks por raiz. São lossless **a partir da conversão para cinza**, não preservam
RGB, quadros intermediários do frame skip4 ou estado oculto. Guardar offsets que
permitem reconstruir qualquer história de16quadros sem duplicar todos os dados.
Reexecutar149/151 conhecidos: todos os pixels reduzidos, retornos e terminais
precisam coincidir bit a bit com a bancada antiga; essa reexecução é experiência
física adicional e entra na contabilidade.179 será nova avaliação após congelar
os corpos.32raízes/8episódios, quatro programas de treino/desenvolvimento e oito
de avaliação (quatro composições novas), mesmos programas exógenos anteriores.

Gerar cinco filtros executáveis por PCA da inovação temporal em patches10x10.
Usar8192patches apenas de149, sem consultar recompensa/ação: oito diferenças
temporais por trajetória, oito patches por diferença,128trajetórias. Subtrair
média de cada patch de diferença, decompor sua covariância, fixar sinal pelo
maior coeficiente absoluto e normalizar normaL1. Os filtros são funções locais
com100coeficientes, origem e corpo exportados. São primitivas/família fornecidas,
não invenção arbitrária. PCA e funções espaciais possuem precedentes; nenhuma
reivindicação de novidade mundial. A mesma função pode ser aplicada sem alterar
pesos no outro jogo; parâmetros do modelo temporal ainda serão aprendidos por
jogo. Isso não é transferência de inteligência geral.

Comparar sensores: média repetida em seis canais; momentos fixos espaciais até
ordem dois; cinco filtros aprendidos + média (primário); cinco filtros aleatórios
de média zero + média; corpo de filtros aprendido no outro jogo + média
(transferência secundária, nunca substituir pelo primário). Selecionar os mesmos
quadros passados0/2/4/6/8/10/12/15 para todos. Normalizar cada canal pelo desvio
padrão somente das fontes de treino, piso.001. Todos recebem as mesmas ações,
recompensas, terminais e limites de experiência.

Aprender32suportes RBF das observações resultantes, ridge.05 para recompensa e
transporte de um passo. Esse modelo temporal continua linear nas funções RBF:
o ensaio muda a informação/geometria sensorial, não resolve por hipótese todos
os problemas de evolução/composição. Dimensão de coeficientes igual nos cinco
sensores, extração de características tem custos diferentes, reportá-los.
Controles adicionais: resultados embaralhados por ação na mesma representação,
contexto inicial removido, corpo antigo congelado (16quadros reduzidos) e zero.
Nenhum reajuste após desenvolvimento151 ou avaliação179.120s/teto de ajuste,
120s/50mil chamadas por coleta,6GiB/worker; um processo de cada vez.

Portão: nas quatro células jogo × programas conhecidos/novos, cobertura com
preferências em ambas as direções por par e pelo menos3episódios contrastantes.
No horizonte16, primário deve ter MSE e arrependimento de desempate justo
estritamente menores que média, momentos, aleatório, embaralhado, sem contexto,
modelo antigo e zero. Horizontes4/8 são secundários. Sem cobertura: inconclusivo;
não relaxar o portão após resultados nem escolher o sensor com melhor avaliação.

Auditoria: reconstruir todas as histórias reduzidas, conciliar eventos/calls e
48replays reais, conferir origem dos filtros nas equações de covariância e origem
dos suportes nas observações, equações normais por Cholesky, composição pela
direção oposta, identidade dos casos e portão. Fontes/protocolo antes de coleta;
corpos e programas antes de179. Somente um resultado positivo autoriza promoção
dessa camada limitada, não do objetivo humano.

Comandos no diretório D:\AGI_Inventor\diagnostico_estado, sequencialmente:

```powershell
py -3.11 -B testes_sensor_resolucao.py
py -3.11 -B testes_filtros_observados.py
py -3.11 -B piloto_resolucao.py prepare
py -3.11 -B coletar_resolucao.py breakout 149
py -3.11 -B coletar_resolucao.py pong 149
py -3.11 -B coletar_resolucao.py breakout 151
py -3.11 -B coletar_resolucao.py pong 151
py -3.11 -B piloto_resolucao.py fit
py -3.11 -B piloto_resolucao.py development
py -3.11 -B coletar_resolucao.py breakout 179
py -3.11 -B coletar_resolucao.py pong 179
py -3.11 -B piloto_resolucao.py evaluate
py -3.11 -B auditar_resolucao.py
```

Saídas recusam sobrescrita. Coleta/ajuste interrompidos conservam custos e
artefatos; não repetir ignorando trabalho já feito. Antes de uma reexecução,
criar rodada distinta com protocolo/custos próprios. Original permanece intacto.

Contexto de pesquisa: descoberta de objetos e representação de vídeos já são
pesquisadas, não terrenos sem precedentes. [SAVi](https://slot-attention-video.github.io/)
explora dinâmica temporal e condicionamento por pistas de localização;
[DINOSAUR](https://arxiv.org/abs/2209.14860) usa características de um modelo
pré-treinado auto-supervisionado. Não estamos implementando esses métodos nem
importando seus pesos; o ponto aqui é testar funções locais baratas no mecanismo
existente. Resultados deles não demonstram o objetivo humano deste projeto.
