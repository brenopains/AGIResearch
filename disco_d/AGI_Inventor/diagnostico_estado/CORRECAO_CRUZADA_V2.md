# Correção de instrumento: framebuffer não é estado restaurado

A primeira execução foi interrompida com384chamadasALE, na primeira raiz
Breakout131. Zero raízes completas, nenhum ajuste, nenhum resultado137.
Proposta, fontes, eventos, checkpoint e registro de recursosV1 permanecem
intactos. Essa tentativa é inconclusiva e não participa dos portões.

O contrato novo exigia que `getScreenRGB()` imediatamente após restaurar
estado devolvesse a imagem da raiz. A calibração local na semente29 mostrou:
o estado completo restaurado é igual ao original; repetir o programa reproduz
retornos, término, imagens e comprimentos; porém a tela lida imediatamente
continua igual à tela final do ramo anterior. O framebuffer exibido ficou em
cache. Não é evidência de que a restauração causal falhou.

Calibração:378chamadasALE, registro `calibracao_restauracao_ale.json`. Não foi
fornecido estado interno ao aprendiz. Os programas usaram a história medida
antes da ramificação; imagens posteriores foram obtidas após ações efetivas.
O diagnóstico não recebeu respostas137. Nem programa, representação, sementes,
regularização, observações, primário, controles ou critério foram alterados.

V2 copia os seis módulos próprios da experiência em nomes novos, com manifesto
dos hashes. Substitui só a checagem de framebuffer por igualdade do estado
clonado após restauração. As funções aprendidas continuam recebendo apenas
pixels passados e palavras. A mudança de namespace preserva todos os arquivos
V1. Os384passos da tentativa anterior são debitados do teto50.000Breakout131
e reportados separadamente. A calibração29 também entra no total operacional.

131é treino parcialmente visto pela falhaV1;137segue reservado para a avaliação
com corpos congelados. Nenhuma dessas sementes representa um mundo novo: ambos
osROMs são conhecidos. Uma correção do instrumento não é ganho de capacidade,
nem resultado positivo/negativo para a hipótese. O primeiro teste do monitor
também detectou um console auxiliar `conhost.exe`; o modoDETACHED_PROCESS
removeu esse processo antes da execuçãoV1, sem criar janela visível. Os testes
de interrupção porRSS/prazo passaram antes de coletar.
