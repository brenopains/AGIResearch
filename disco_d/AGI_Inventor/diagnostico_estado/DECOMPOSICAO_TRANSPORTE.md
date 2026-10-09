# Localizar erros sem confundir oráculo com capacidade

Diagnóstico de desenvolvimento sobre os corpos97 já congelados e os bancos
mistos103 já vistos. Sem treino, coleta, promoção ou alteração dos portões.
Selecionar todas as32raízes, dois ramos e horizontes4/8/16 em ambos os jogos.
Reexecutar todos os cinco corpos, incluindo controles. Não ler assinaturas.

Para cada trajetória real, a álgebra do corpo permite esta identidade:

`previsão - retorno_real = E_recompensa + E_transporte + E_inicial_terminal`.

- `E_recompensa = soma_válida(phi(H_t)*r_a - recompensa_real_t)`.
- `E_transporte = soma_válida((phi(H_t)*M_a - c_t*phi(H_t+1))*v_sufixo)`.
- `E_inicial_terminal` é a previsão inteira caso a raiz já esteja terminada.

Aqui `c_t` é a continuação real e `v_sufixo` é o corpo executável aprendido
para as ações restantes. Usar osMdo corpo; no controle sem transporte,Mé
identidade. No controle sem contexto,phié sempre a média de treino. Tratar
término e recompensa de padding explicitamente. Verificar a identidade em
cada ramo e no contraste dos dois ramos, antes de interpretar resultados.

O erro vetorial local pode ser pequeno e ainda importar muito numa direção
de futuro. O termo escalar ponderado identifica essa direção. Não inferir
causalidade global ou suficiência perceptual a partir de uma decomposição.

Reportar MSE das diferenças de retorno, parcelas quadráticas e termos cruzados.
Estes últimos podem cancelar erros: não interpretar parcelas como percentuais
independentes de culpa. Reportar escolhas/arrependimento do corpo e das
intervenções ideais que retiram uma parcela de erro. Retirar transporte e
erro de raiz terminada equivale a ler recompensas aprendidas em cada história
real. Retirar erro de recompensa usa rótulos verdadeiros e não é uma política
executável. Incluir controle com história inicial fixa e términos reais para
separar acesso à trajetória de acesso somente ao término.

Auditoria independente reexecuta o corpo por produtos matriciais e atribui
transporte como diferença entre valores dos sufixos adjacentes, sem reutilizar
a fórmula vetorial do diagnosticador. Reconta linhas/curvas e reconcilia os
resultados anteriores. Erro permitido1e-8; nenhum resultado aproximado é
rotulado prova física. A identidade algébrica é exata nos reais e testada em
ponto flutuante. Verificação física limitada aos registros determinísticos.

Congelar fontes, protocolo, corpos e dados antes do cálculo. Cada consulta e
transição atribuída terá eventoJSONL real. Nenhum limiar novo de sucesso:
resultado para orientar pesquisa, não confirmarAGI ou uma peça única faltante.

```powershell
cd D:\AGI_Inventor\diagnostico_estado
py -3.11 -B testes_decomposicao_transporte.py
py -3.11 -B decompor_erros_transporte.py --congelar
py -3.11 -B decompor_erros_transporte.py --executar
py -3.11 -B auditar_decomposicao_transporte.py
```
