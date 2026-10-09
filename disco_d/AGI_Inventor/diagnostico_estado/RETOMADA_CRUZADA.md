# Retomar sem recoletar nem mudar algoritmo

V2 concluiu os dois bancos131 com replays certificados. O ajuste foi
interrompido pelo monitor no carregamento de `numpy.testing`, antes de salvar
qualquer corpo. O rastreio mostrou `platform.machine()` → `win32_ver()` →
`cmd /c ver`, criandocmd.exe/conhost.exe. A consulta servia a descobrir se a
máquina eraWASM; não era trabalho de aprendizagem. O registroV2 permanece
inconclusivo e inclui o pico de quatro processos detectado.

Um trabalhador separado consulta a arquitetura usando a mesma implementação
nativa `_get_machine_win32` do Python3.11 instalado, sem calcular os demais
campos de `uname`. Essa substituição existe apenas dentro do processo próprio;
nenhum arquivo de Python/NumPy instalado foi modificado. Não substituir leitura
de pixels, modelos, fontesV2, primário, controles, sementes ou portões. A
calibração verifica o limite de dois processos antes da retomada.

Retomar a partir dos bancos131 já medidos: ajustes131, coletas137 e avaliações137.
Congelar hashes do lançador, supervisor e auditoria adicional antes da retomada.
Os corpos só existem após ajuste131; os retornos137 ainda não existem no início.
Não repetir os bancos nem ajustar corpos com137. Incorporar1,797+24,828segundos
das execuções de piloto anteriores ao limite30minutos. Custos de calibração e
384passos da tentativaV1 são reportados separadamente; o coletorV2 já debita
esses passos do tetoBreakout131. O pico de processos da tentativa interrompida
é preservado, não apresentado como se toda a execução tivesse ficado em dois.

O novo supervisor aplica os mesmos limites deRSS/prazo/processos durante
as seis etapas restantes. Uma falha de capacidade ou de identificação
contextual continua falha/inconclusivo segundo o protocolo anterior. Nenhum
resultado desse reparo é considerado evidência de mente humana.
