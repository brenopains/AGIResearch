# Comparadores executáveis e palavras reservadas

Pasta: `D:\AGI_Inventor\diagnostico_estado`. Artefatos desta rodada já existem;
scripts recusam sobrescrita. Preservar evidências. Reproduções exigem uma cópia
separada com dependências/bancos anteriores exigidos pelos manifestos.

Inspecionar a implementação e verificar resultados existentes:

```powershell
py -3.11 -B testes_contrastes_programas.py
py -3.11 -B testes_serializacao_contrastes.py
py -3.11 -B auditar_contrastes_programas.py
```

Sequência usada antes de existirem os artefatos:

```powershell
py -3.11 -B avaliar_contrastes_programas.py breakout --ajustar
py -3.11 -B avaliar_contrastes_programas.py pong --ajustar
py -3.11 -B coletar_programas_reservados.py breakout
py -3.11 -B coletar_programas_reservados.py pong
py -3.11 -B avaliar_contrastes_serializacao.py breakout
py -3.11 -B avaliar_contrastes_serializacao.py pong
```

Usar a entrada de serialização para avaliar: `--avaliar` do script congelado
calcula corretamente, mas falha ao salvar condições NumPy no JSON. O adaptador
converte esses escalares sem editar a fonte congelada ou recalibrar critérios.

Corpos NPZ/JSON e relatórios em `resultados/{jogo}_contrastes_programas*`;
programas físicos novos e eventos em `resultados/{jogo}_programas_reservados_103*`.
Cada corpo recebe vetor de estado e dois programas de ações, devolvendo uma
diferença prevista. Verificar identidade/inversão não prova diferença correta.
Estado inicial nesta bancada é medido por sondas; não é percepção autônoma.

Resultado: dois portões negativos, sem promoção. Palavras completas novas
no ajuste de comparação não equivalem a mundos novos;103 permanece
desenvolvimento já utilizado. Objetivo humano não demonstrado.
