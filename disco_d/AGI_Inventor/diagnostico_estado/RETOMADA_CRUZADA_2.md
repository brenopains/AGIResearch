# Segunda consulta do Windows identificada, corposV2 preservados

A primeira calibração do lançador cobriuNumPy.testing, mas não a consulta
`platform.system()` queALE faz em uma importação fresca. A retomada interrompeu
nessa importação, antes de salvar modelos ou coletar137. O rastreio impediu a
criação do subprocesso e localizou a chamada emvendor/ale_py/__init__.py:25.
Essa diferença de escopo da calibração é registrada; não foi falha cognitiva.

Novo lançador: emsys.platform==win32, obtermachine pela implementação nativa
do próprio Python e devolverWindows para a consulta de sistema. São os
metadados reais do host; não são variáveis do mundo ou entradas do aprendiz.
Nenhuma versão instalada deALE, NumPy ou Python é editada. Uma calibração
agora importaALE primeiro e realiza ajuste pequeno, sem salvar corpo ou ler137.

Manter os bancos131, fontesV2, limites, sementes, braços e critérios. Congelar
somente lançador/supervisor/auditoria adicionais em novos arquivos. Relatar
as três execuções interrompidas, sem apagar picos de processos. Incluir27,625s
de tempo anterior no teto1800s. O resultado permanece inconclusivo se qualquer
etapa restante falhar. Não promover controle nem trocar modelo primário.
