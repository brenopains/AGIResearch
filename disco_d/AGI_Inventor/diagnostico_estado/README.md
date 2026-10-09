# Diagnóstico de estado a partir de pixels

Código isolado em `D:\AGI_Inventor\diagnostico_estado`. Não modifica o projeto original. Não carrega os quatro jogos lacrados; o coletor aceita somente Breakout e Pong. Esta bancada investiga um bloqueio específico, não implementa uma mente de nível humano.

## Pergunta congelada antes da coleta principal

O instrumento atual inventa variáveis somente como funções do quadro atual 21×16 e da diferença para o quadro anterior. Se uma informação decisiva estiver em quadros mais antigos ou se perder na redução espacial, a criação posterior de variáveis não pode recuperá-la. O teste usa o **mesmo banco de estados e seus desfechos pareados** para comparar essas possibilidades.

Para cada estado, o ALE é clonado com RNG; executamos quatro programas de ação constantes do conjunto mínimo por 8, 32 e 128 passos. Os quatro partem do mesmo estado. Uma execução repetida por episódio verifica a restauração. O orçamento contabiliza todas as ramificações. O observador recebe pixels, ações, recompensa e tempo, sem objetos ou variáveis do jogo.

Os episódios 0–3 constroem modelos/candidatos; 4–5 selecionam; 6–7 ficam reservados. Comparações: dois quadros 21×16 (atual), dezesseis quadros 21×16, dois quadros 42×32, dezesseis quadros 42×32 e dois quadros na resolução nativa. Todos usam a mesma regressão kernel, hiperparâmetro fixo e desfechos visuais 21×16. O placebo permuta, dentro de cada estado de construção, a associação entre programa e desfecho. Escolha de ação é um diagnóstico supervisionado offline restrito a quatro programas; não é desempenho de um agente autônomo.

O gerador de memória usa o resíduo fora do próprio episódio de construção, propõe médias de regiões em atrasos de 2/4/8/15 quadros, escolhe em validação e registra a definição executável. Promover exige melhora de validação **e** de avaliação. Resultado positivo em uma única semente é indício, não confirmação. Comparar braços de dimensão diferente não isola informação de dificuldade de ajuste: melhora indica utilidade para este estimador; piora não prova ausência de informação. Ausência de diferenças de recompensa gera diagnóstico `sem_contraste_de_retorno`, nunca conclusão sobre inteligência.

O teste de fumaça de 2 raízes/episódio já mostrou que uma memória escolhida na validação piorou na avaliação; ele não entra como confirmação. Os parâmetros e códigos deste protocolo devem ser considerados congelados para a rodada principal. Não ajustar depois de ler seus resultados.

## Reprodução

Python 3.11 com NumPy e `ale-py==0.12.1` em `vendor`. Na pasta atual:

```powershell
py -3.11 -B .\coleta.py breakout --episodes 8 --roots 8 --budget 50000 --out .\resultados\breakout_bank.npz
py -3.11 -B .\avaliar.py .\resultados\breakout_bank.npz
py -3.11 -B .\coleta.py pong --episodes 8 --roots 8 --budget 50000 --out .\resultados\pong_bank.npz
py -3.11 -B .\avaliar.py .\resultados\pong_bank.npz
```

Cada coleta produz `.npz` e `.json` com SHA-256, passos realmente usados e teste de replay. A avaliação produz `_diagnostico.json`. É possível inspecionar os arrays e repetir a avaliação sem executar novamente o jogo.

O segundo ensaio exploratório tem protocolo e interpretação próprios em `PERGUNTAS.md` e `RELATORIO.md`. Reproduza-o com `py -3.11 -B .\perguntas.py breakout` e `py -3.11 -B .\perguntas.py pong`; o teto é 80.000 passos por jogo.

Os ensaios seguintes estão descritos em `CICLO.md`, `PREFIXO.md` e `DECISAO.md`. Código: `coleta_ciclo.py`, `ciclo.py`, `prefixo.py` e `decisao.py`. Testes de controle positivo: `py -3.11 -B .\testes_ciclo.py` e `py -3.11 -B .\testes_prefixo.py`. Os bancos do ciclo são reproduzidos com `py -3.11 -B .\coleta_ciclo.py breakout` e `py -3.11 -B .\coleta_ciclo.py pong`, teto de 160.000 passos por jogo. As avaliações recebem o caminho do `.npz` gerado.

O seletor exploratório de estados está em `AMOSTRADOR.md` e `amostrador.py`. O relatório reúne todos os resultados e limites de interpretação.

O compilador de contraexemplos de decisão está em `CONTRAEXEMPLOS.md` e `contraexemplos.py`. Execute `py -3.11 -B .\testes_contraexemplos.py` para o controle positivo; `py -3.11 -B .\contraexemplos.py .\resultados\breakout_ciclo.npz` (ou `pong_ciclo.npz`) reproduz o ensaio exploratório com 32 nulos.

A busca de estados está em `AQUISICAO.md` e `aquisicao.py`. `py -3.11 -B .\testes_aquisicao.py` verifica replay; `py -3.11 -B .\aquisicao.py breakout` e `py -3.11 -B .\aquisicao.py pong` repetem o piloto ativo versus aleatório. Cada braço tem teto de 160.000 passos ALE por jogo; eventos reais são salvos em JSONL.

O contrato comum está em `GERAL.md`; ampliação de memória em `CAPACIDADE.md`;
auditoria oracular em `ORACULO.md`; percepção a partir de pixels em
`PERCEPCAO_PIXELS.md`; confirmação em mundos físicos novos em
`SUITE_FISICA.md`. Reproduza o último ensaio com
`py -3.11 -B .\suite_fisica.py`. O script cria bancos por semente e
`resultados\suite_fisica.json`. O relatório contém os resultados e limites.

O filtro de identidades alternativas está em `hipoteses_estado.py`, com protocolo
em `HIPOTESES_ESTADO.md` e confirmação em `CONFIRMACAO_HIPOTESES.md`.
`py -3.11 -B .\suite_hipoteses.py` reproduz a confirmação em **uma cópia nova
de `D:\AGI_Inventor`**, pois o script recusa sobrescrever bancos existentes.
O resultado atual está em
`resultados\suite_hipoteses.json`; os bancos e relatórios por semente estão no
mesmo diretório. O portão não passou.

`decisao_crencas.py` e `DECISAO_CRENCAS.md` investigam, nos mundos já vistos,
se a incerteza perceptiva altera escolhas; o diagnóstico também está no relatório.

O teste interativo de escolha de experiências está em `DESACORDO_ATIVO.md` e
`desacordo_ativo.py`. `py -3.11 -B .\testes_desacordo.py` verifica votos e replay;
`py -3.11 -B .\desacordo_ativo.py breakout` e `pong` repetem o piloto de até
250 mil passos ALE por jogo. Os relatórios JSON e eventos JSONL correspondentes
ficam em `resultados`. O portão de ganho por episódio não passou.

A pré-condição de modelos preditivos para perguntar está em
`CALIBRACAO_MODELOS.md` e `TESTEMUNHAS.md`. Execute
`py -3.11 -B .\testes_testemunha.py` para o contrato exato;
`py -3.11 -B .\calibracao_modelos.py breakout` e `pong`, seguidos de
`py -3.11 -B .\auditar_testemunhas.py breakout` e `pong`, para o diagnóstico
nos bancos Atari. Os JSON incluem previsões por raiz. O comitê atual não
sustenta perguntas discriminantes válidas.

O banco interativo foi reconstruído e auditado com
`py -3.11 -B .\reconstruir_desacordo.py breakout` e `pong`, depois convertido
por `converter_ativos.py`. `TRANSFERENCIA_ALE.md` congela o ensaio em estados
inéditos: execute `py -3.11 -B .\transferencia_ale.py breakout` e `pong`.
O resultado está em `resultados\{breakout,pong}_transferencia_ale.json` e
`.jsonl`. Breakout alcançou arrependimento zero em 48 raízes com contraste,
mas o portão de vitória estrita por episódio falhou (3/8); Pong empatou com
ação constante e com o controle de históricos permutados. Ver `RELATORIO.md`.

`TRAJETORIA_INDEPENDENTE.md` pré-registra checkpoints temporais na semente 97.
Reproduza com `py -3.11 -B .\trajetoria_independente.py breakout` e `pong`.
Os JSON, JSONL e NPZ ficam em `resultados`. Com margem real para melhorar a
ação constante, o novo modelo falhou nos dois jogos; esse é o diagnóstico
mais forte e recente sobre a representação visual temporal atual.
`diagnosticar_vistas_transferencia.py` compara retrospectivamente cinco
visões da mesma história nesses estados. Mais quadros ou resolução, com o
mesmo regressor, não eliminaram o erro; esse resultado é exploratório.

`EFEITOS_LOCAIS.md` e `CONFIRMACAO_EFEITOS.md` descrevem a coleta e a
confirmação de consequências visuais após oito ações. Reproduza os bancos
com `py -3.11 -B .\coletar_efeitos_locais.py breakout --seed 97` e `pong`,
depois com `--seed 103`; execute
`py -3.11 -B .\confirmar_efeitos_locais.py breakout` e `pong` para comparar
o modelo visual congelado com efeito fixo por ação e dois tipos de placebo.
O portão **local** passou nos dois jogos; `RELATORIO.md` explica por que isso
ainda não prova planejamento ou uma mente geral.
`curto_para_longo.py` testa como oráculo se efeitos curtos reais bastam para
escolher pelos retornos distantes no banco de desenvolvimento; a leitura
direta não venceu a ação constante. O próximo elo é composição temporal.

`COMPOSICAO_TEMPORAL.md` documenta ramos até 128 passos; reproduza os bancos
com `coletar_horizontes.py` nas sementes 97 e 103, depois execute
`avaliar_composicao.py` e `estabilizar_composicao.py` por jogo. O resultado é
exploratório: encadear imagens previstas acumulou erro, e uma memória de
vizinhos visuais não preservou adequadamente os efeitos das ações.

`ESTADO_PREDITIVO.md` e `PRE_REGISTRO_PLANOS_109.md` documentam o estado
definido por quatro consequências de oito passos e sua confirmação em planos
mistos. `coletar_estado_preditivo.py` gera o treino 97 e a avaliação
exploratória 103; `avaliar_estado_preditivo.py` avalia ações constantes.
Para reproduzir a confirmação independente, execute por jogo as quatro
coletas/avaliações do pré-registro e depois
`py -3.11 -B .\veredito_planos_mistos.py`. O portão passou nos dois jogos,
mas o mecanismo ainda não escolhe sondas, representações ou ações por conta
própria. Os valores, controles e limites estão no `RELATORIO.md`.

`SONDAS_EXECUTAVEIS.md` documenta o gerador por contraexemplos, a seleção por
decisão e o extrator que reutiliza experiências intermediárias. Controles:
`testes_sondas.py`, `testes_sondas_decisao.py`,
`testes_extrator_experiencias.py` e `testes_conjuntos_sondas.py`.
`auditar_sondas.py` aplica o veredito aos relatórios reais: nenhuma sonda foi
promovida. O ensaio de Breakout não transferiu ganho de decisão; o de Pong
tem um portão sem margem e é inconclusivo. Os scripts e comandos completos
estão no documento.

`LEITURAS_VARIAVEIS.md` descreve fragmentos com posição variável e relações
espaciais; `CONFIRMACAO_LEITURAS_113.md` contém o portão congelado de Pong.
O piloto positivo não confirmou. Respostas reais das mesmas sondas também
não recuperaram a decisão no diagnóstico posterior. Os arquivos congelados
permanecem preservados; comandos e resultados no `RELATORIO.md`.

`VALOR_COMPOSTO.md` documenta crédito local, substituições oraculares e
distribuições de futuros. `valor_composto.py` não venceu a ação constante;
`diagnosticar_credito.py` localizou forte perda na trajetória prevista.
`futuros_distribuidos.py` preserva experiências possíveis em vez de colapsar
ao estado médio. `CONFIRMACAO_FUTUROS_127.md` congela sua avaliação independente
em Breakout: o portão passou, com arrependimento 0,234375 contra 0,468750 da
ação constante e 0,421875 da regressão direta. Pong ainda não tem ganho.
Reproduza controles e comandos pelos documentos, preservando os manifestos.
O alcance é escolha offline entre quatro programas neste jogo; nível humano
e invenção autônoma de representações continuam não demonstrados.

`PILOTO_PLANOS_PROPOSTOS.md` documenta a passagem de quatro constantes para
sequências propostas por busca. O corpo genérico é `planejador_distribuido.py`;
os controles `testes_planejador_distribuido.py` e
`testes_planejador_bancos.py` verificam troca de ação, enumeração exata num
mundo pequeno e concordância da busca com a composição das palavras.
Execute `piloto_planos_propostos.py {jogo} --propor` antes de `--executar`.
Ambos os pilotos terminaram com portão negativo; preservar as propostas e os
resultados já existentes. `diagnosticar_planos_propostos.py`,
`oraculo_planos_propostos.py` e `auditar_poda_planos.py` separam repertório,
previsão de crédito, trajetória e poda. O protocolo posterior está em
`ORACULO_PLANOS_PROPOSTOS.md`; os resultados recentes estão no `RELATORIO.md`.

`PILOTO_COORDENADAS_CONSEQUENCIAS.md` documenta funções de crédito exportáveis
e coordenadas de estado aprendidas das consequências. Controle:
`testes_coordenadas_consequencias.py`; geração/execução:
`piloto_coordenadas_consequencias.py {jogo} --propor` e `--executar`.
Os pilotos falharam contra as coordenadas aleatórias/embaralhadas; preservar
os resultados e parâmetros. `diagnosticar_transporte.py` compara previsão com
o suporte disponível; `operadores_locais.py` aplica mudanças observadas ao
contexto, com controle `testes_operadores_locais.py` e avaliação
`avaliar_operadores_locais.py`. Esses operadores perderam da galeria nos
dois jogos, sem nova coleta. Nenhuma dessas leituras foi promovida.

`PILOTO_DESFECHOS_APRENDIDOS.md` testa categorias de próximo estado, recompensa
e término mantidos juntos, com probabilidades condicionais aprendidas. Corpos
completos em `desfechos_aprendidos.py`, exportados para NPZ/JSON; controle
`testes_desfechos_aprendidos.py`. Executar `piloto_desfechos_aprendidos.py
{jogo} --propor` antes de `--executar`. Ambos os pilotos falharam; preservar
os resultados e não repetir para procurar um resultado favorável.
`diagnosticar_desfechos_aprendidos.py {jogo}` mede erros locais/de composição
nas trajetórias mistas já coletadas; `diagnosticar_raiz_desfechos.py` substitui
somente o estado inicial por sua medida real nos mesmos planos congelados.
Melhor leitura inicial não recuperou a previsão longa. Auditoria:
`py -3.11 -B auditar_desfechos_aprendidos.py`. Evidências e limites em
`RELATORIO.md`; a próxima pergunta técnica está em `GARGALO_COMPOSICAO.md`.

`MEMORIA_RESIDUAL.md` registra o teste de memória gerada por resíduos, com
transições mistas e história visual passada. `coletar_transicoes_mistas.py`
coleta os bancos97/103; `memoria_residual.py` gera projeções e dinâmica com
parâmetros completos; `avaliar_memoria_residual.py {jogo} --ajustar` congela
somente97 e `--avaliar` mede103. `testes_memoria_residual.py` verifica passado,
isolamento por episódio, variável gerada, exportação e absorção de término.
Os dois portões diagnósticos falharam: melhor erro médio não melhorou decisão.
`auditar_memoria_residual.py` confere histórias, custos, corpos e portões.
Preservar os bancos/corpos existentes. A correção de instrumento anterior
ao banco válido está em `CORRECAO_COLETA_MISTA.md`; comandos e limites em
`COMANDOS_MEMORIA_RESIDUAL.md`. Nada desta etapa foi promovido como agente.

`CONTRASTES_PROGRAMAS.md` registra a comparação direta entre programas:
`contrastes_programas.py` exporta o corpo completo de uma função de diferença
de retorno. `avaliar_contrastes_programas.py {jogo} --ajustar` congela97 antes
de `coletar_programas_reservados.py {jogo}` medir palavras novas. A entrada
de avaliação é `avaliar_contrastes_serializacao.py {jogo}`: resolve apenas
conversão de booleanos NumPy no JSON, sem editar o avaliador congelado.
Controles: `testes_contrastes_programas.py`,
`testes_serializacao_contrastes.py`, `auditar_contrastes_programas.py`.
Ambos os portões falharam: ganho frente a retornos absolutos não mostrou
uso robusto do estado além da informação marginal das palavras. Preservar
resultados e corpos. Comandos/scope em `COMANDOS_CONTRASTES_PROGRAMAS.md`.

`LEITURAS_POR_PROGRAMA.md` registra o piloto concluído de funções sensoriais
construídas para cada programa, com corpo completo e apenas pixels passados.
`leituras_por_programa.py` exporta/executa as leituras; ajuste/avaliação em
`avaliar_leituras_por_programa.py`. Ambos os portões falharam; a versão
quadrática escolheu exatamente como a linear nos32pares finais de cada jogo.
Auditoria: `auditar_leituras_por_programa.py`; contratos/reprodução e arquivos
em `COMANDOS_LEITURAS_POR_PROGRAMA.md`. Não sobrescrever fontes/corpos congelados.
`diagnosticar_identificabilidade_leituras.py` quantifica a cobertura contextual
do banco, com testes em `testes_identificabilidade_leituras.py`.

A medição de `EXPERIENCIA_CRUZADA.md` foi concluída com sementes131/137 e
programas comuns entre situações. Comandos/fontes efetivas e reparos de
instrumento: `COMANDOS_EXPERIENCIA_CRUZADA.md`. Auditoria independente:
`auditar_retomada_cruzada_2.py`. Em Pong/programas conhecidos, o primário
usa contexto e supera os nulos pertinentes, mas perde da leitura de dois
quadros: o portão completo falhou. As outras três partições são inconclusivas
por identificação contextual insuficiente. Nenhum modelo promovido.
`diagnosticar_cruzada_limites.py` e `testes_limites_cruzada.py` conferem o limite
empírico de escolhas só por programa, sem realimentar o agente. A próxima
pergunta está em `PROXIMA_LEITURA_SEMANTICA.md`. Objetivo humano não atingido.

`SEMANTICA_PREFIXO_COMUM.md` concluiu essa pergunta:28identidades físicas
certas, sem resgate consistente de decisão pelo contexto intermediário real.
Código/comandos em`diagnosticar_semantica_prefixo.py`, testes em
`testes_semantica_prefixo.py`, auditoria em`auditar_semantica_prefixo.py`.
Nenhum ajuste/coleta/promoção; dados de desenvolvimento, prefixo de um bloco.

Novo mecanismo executável em`transporte_funcoes.py`: ações transportam funções
sensoriais e os corpos de programas são produtos desses operadores. Piloto
`piloto_transporte_funcoes.py`, testes`testes_transporte_funcoes.py`, auditor
`auditar_transporte_funcoes.py`, protocolo congelado`TRANSPORTE_FUNCOES.md`.
Benefício local emPong, sem benefício conjunto nos dois jogos: portão falhou.
Detalhes em`RESULTADO_TRANSPORTE_FUNCOES.md`. Nenhuma alegação de mente humana.

`DECOMPOSICAO_TRANSPORTE.md` localizou as parcelas de erro sem alterar corpos.
Todas as histórias futuras reais não recuperam o leitor de recompensa atual.
`CREDITO_SEQUENCIAS_TRANSPORTADAS.md` tentou resolver pesos por consequências
inteiras com transporte/funções fixos; o primário piorou nos dois jogos e
ambos os portões falharam. Sem coleta/promoção. Auditorias independentes e
limites em`RESULTADO_DECOMPOSICAO_E_CREDITO.md`. Objetivo humano continua aberto.

`funcoes_contraexemplos.py` implementou geração de funções executáveis por
pares observados e composição por produto, com origem e corpos exportáveis.
O piloto não selecionou funções: em Breakout as melhoras cobrem apenas um
caso; em Pong a validação não tem contraste de decisão. Portão preservado.
`diagnosticar_pool_funcoes.py` distingue efeito local, composição e cobertura.
Nove testes e duas auditorias passaram. O auditor do piloto exige o bootstrap
`executar_auditoria_funcoes_contraexemplos.py` para configurar BLAS antes de
NumPy. O diagnóstico usa `auditar_pool_funcoes.py`. Fontes/protocolos/resultados,
controles, proteção contra sobrescrita e próximo contraste prospectivo em
`RESULTADO_GERADOR_FUNCOES.md`. Nenhuma evidência de proximidade humana.

O gerador seguinte é`funcoes_prospectivas.py`: pares são derivados de
consequências observadas após a mesma ação, mesmo sem recompensa imediata.
Bancada nova, com programas comuns e composições reservadas, em
`piloto_prospectivo.py` e`coletar_prospectiva.py`. Fontes congeladas antes da
coleta; corpos antes de157. Todas as quatro células têm cobertura suficiente
e falharam o portão. Efeito local emPong não supera todos os controles.
Auditoria:`auditar_prospectiva.py`; decomposição posterior sem ajuste:
`diagnosticar_prospectiva_composicao.py`. Nove testes novos passaram.
Detalhes, contabilidade, limites e comandos em`RESULTADO_PROSPECTIVA.md` e
`PROSPECTIVA.md`. Nenhuma função promovida como capacidade geral ou mente humana.

Etapa 51: `leitor_eventos.py` distingue reconhecimento antes/depois de previsão.
O diagnóstico anterior com histórias reais ainda antecipava cada bloco; seu
alcance foi corrigido em `RESULTADO_EVENTOS.md`. Leitores congelados melhoram
reconhecimento local, mas o campo de estado falha em todas as quatro células
de planejamento da experiência nova 163. Seis testes, 36 leitores auditados,
9.216 comparações e 16 replays físicos; 74.804 chamadas ALE adicionais.
`fechamento_leitor_eventos.py` verifica a identidade que devolve o leitor
transportado ao preditor antigo com a mesma base e regressão linear.
Protocolo `EVENTOS.md`, auditor `auditar_leitor_eventos.py`. Nenhuma AGI ou
proximidade humana demonstrada; nenhum corpo promovido como capacidade geral.

Etapa52 preserva trajetórias em resolução210x160 em cinza e aprende cinco
filtros locais de inovações observadas, sem rótulos. `sensor_resolucao.py`,
`filtros_observados.py`, `coletar_resolucao.py`, `piloto_resolucao.py`.
Quatro bancos antigos reproduzidos integralmente;91.398chamadas adicionais,
32replays. Auditoria de6.959transições não encontrou colisões exatas de
história reduzida+ação com imagens completas/recompensas distintas; isso não
prova suficiência. Corpos dos filtros têm origem e execução auditadas nos dois
jogos, sem demonstração de transferência útil. O ajuste atingiu o limite
antes de terminarPong:12corpos parciais, sem avaliação179 ou portão de ganho.
Quatro testes adicionais de dicionário e equivalência numérica passaram;
velocidade não foi comparada. `RESULTADO_RESOLUCAO.md` registra interrupção,
custos, comandos, interpretações corrigidas e próximo trabalho. Sem AGI.

Etapa 53 concluiu o mesmo teste em rodada própria: `transporte_compacto.py`
teve aceleração pareada de 2,71–3,09 vezes com instâncias numéricas preservadas.
Ajuste completo em 92,57 s; os dois bancos novos 179 foram coletados após
congelar os corpos. Quatro células com cobertura suficiente, nenhuma aprovada.
Auditoria conferiu 6.912 comparações e 48 replays históricos, 16 novos;
74.686 chamadas novas. O contexto ajudou apenas Pong conhecido contra a ablação
do mesmo corpo e piorou nas composições novas dos dois jogos. Diagnóstico v2
corrige o nome da referência para os braços diferentes do primário. Onze
testes entre compactação e diagnóstico v2 passaram. Detalhes, custos e limites
em `RESULTADO_RESOLUCAO_COMPACTA.md`. Nenhuma proximidade humana demonstrada;
original intacto, nenhum corpo promovido como capacidade geral.
