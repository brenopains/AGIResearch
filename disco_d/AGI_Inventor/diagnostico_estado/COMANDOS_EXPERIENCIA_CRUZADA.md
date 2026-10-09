# Experiência cruzada concluída, sem promoção

Pasta isolada: `D:\AGI_Inventor\diagnostico_estado`. Original intacto.
Protocolo: `EXPERIENCIA_CRUZADA.md`. Coleta, funções e avaliador efetivos são
os módulos `*_experiencia_cruzada_v2.py` e `experiencia_cruzada_v2.py`.
CopiaramV1 com hashes; a correção de framebuffer está em `CORRECAO_CRUZADA_V2.md`.
Os dois reparos do lançador estão em `RETOMADA_CRUZADA.md` e
`RETOMADA_CRUZADA_2.md`. Fontes/modelos/portõesV2 não foram editados nessas
retomadas. O supervisor efetivo é `retomar_experiencia_cruzada_2.py`.

Reexecutar contratos/inspecionar resultados sem sobrescrever os bancos:

```powershell
py -3.11 -B testes_experiencia_cruzada_v2.py
py -3.11 -B testes_limites_cruzada.py
py -3.11 -B auditar_retomada_cruzada_2.py
py -3.11 -B diagnosticar_cruzada_limites.py
```

Etapas de coleta/ajuste/avaliação já foram executadas; seus comandos recusam
sobrescrever fontes congeladas, bancos, corpos, propostas e resultados.
Não apagar esses artefatos para conseguir repetir. Uma intervenção nova
precisa de novo namespace/protocolo, mantendo as respostas137fora do ajuste.

Registros por jogo: `resultados/{jogo}_experiencia_cruzada_v2_137_avaliacao.json`.
JSONL correspondente:2.690eventos de decisão/portão por jogo. Treino131:
`resultados/{jogo}_experiencia_cruzada_v2_131_manifesto.json`, sete corpos
completosNPZ/JSON e suas contas. Proposta exógena anterior a todas as respostas:
`resultados/experiencia_cruzada_v2_propostas.json`. Retomada congelada antes
das etapas restantes: `resultados/experiencia_cruzada_retomada2_manifesto.json`.

Auditoria: `resultados/auditoria_experiencia_cruzada_retomada2.json`.
Confere os quatro bancos,1.056eventos físicos,32replays,5.380eventos de
avaliação, matrizes de ajuste e todos os quatro vereditos. Reexecutar os
corpos reproduziu as previsões sem erro observado. Nenhum controle foi promovido.

Vereditos128passos:

- Pong/programas conhecidos/contextos novos: contraste contextual suficiente,
  capacidade completa falhou. Há ganho do primário sobre ignorar/embaralhar
  contexto, porém leitura de dois quadros supera o primário.
- Breakout/contextos novos e os dois testes de programas novos: identificação
  contextual insuficiente segundo o critério congelado; inconclusivos. Não
  chamar isso de aprovação, nem de impossibilidade da capacidade.

Piloto:79,64s de parede incluindo execuções de piloto interrompidas, excluindo
implementação, calibrações e auditoria. Retomada efetiva: picoRSSamostrado
309,63MiB, máximo dois processos incluindo monitor. Tentativas anteriores
registram quatro processos auxiliares detectados e interrupção; não apagar
essa exceção. Quatro bancos completos:115.859chamadasALE. Com384da tentativa
falha e378da calibração de restauração:116.621. Passos debitados do teto por
jogo/semente; no ajuste/avaliação não há novas interações. Hardware éCPU.

`limites_empiricos_cruzada.json` é diagnóstico posterior: o melhor programa
fixo por par usa respostas reservadas para calcular um limite empírico, sem
fornecê-las ao agente. Não redefine portão e não garante desempenho futuro.
Mais detalhes, achados e limites em `RELATORIO.md`.
